"""Checks for the interpolating training representation and target conversion."""
import io
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from models.catmull_rom import catmull_rom
from models.spline import SplineHead
from scripts.catmull_rom import catmull_rom as reference, catmull_rom_knots
from scripts.prepare_catmull_rom_targets import convert_target


class CatmullRomLearningTests(unittest.TestCase):
    def test_native_and_common_metrics_use_interpolation_curve(self):
        from training.evaluation.spline_metrics import evaluate_spline_graph
        from metrics.common_vessel_metrics import evaluate_common_vessels
        points = np.array([[.1,.3,.3],[.3,.8,.3],[.6,.2,.6],[.9,.7,.4]])
        metric = np.diag([32.,48.,64.])
        curve = reference(points @ metric.T,np.linspace(0,1,257)) @ np.linalg.inv(metric).T
        sampled = reference(points @ metric.T,np.linspace(0,1,64)) @ np.linalg.inv(metric).T
        prediction = dict(nodes=points[[0,-1]],edges=[[0,1]],spline_controls=[points],spline_curves=[sampled],spline_family='catmull_rom')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'target.npz'
            np.savez(path,nodes=points[[0,-1]],edges=[[0,1]],metric=metric,curves=[sampled],full_paths=curve,path_offsets=[0,len(curve)])
            native = evaluate_spline_graph(prediction,path,np.ones((64,64,64)))
            self.assertLess(native['spline_guide_distance_mm'],1e-10)
            self.assertLess(native['spline_tortuosity_error'],1e-10)
            common = evaluate_common_vessels(prediction,path)
            self.assertLess(common['common_guide_distance_mm'],1e-10)
            self.assertEqual(common['common_curve_supported_pred_samples'],common['common_curve_pred_samples'])
            # Deliberately curved points make a B-spline reinterpretation differ.
            wrong = evaluate_spline_graph({**prediction,'spline_family':'bspline'},path,np.ones((64,64,64)))
            self.assertGreater(wrong['spline_guide_distance_mm'],.1)

    def test_interpolation_reference_reversal_and_gradient(self):
        p = torch.tensor([[[0.,0.,0.],[.2,1.5,0.],[2.5,.4,.1],[2.6,2.,.5],[4.,0.,0.]]], dtype=torch.float64, requires_grad=True)
        t = torch.linspace(0, 1, 64, dtype=p.dtype)
        curve = catmull_rom(p, t)
        np.testing.assert_allclose(curve.detach()[0], reference(p.detach()[0].numpy(), t.numpy()), atol=1e-12)
        knots = torch.tensor(catmull_rom_knots(p.detach()[0].numpy()))
        torch.testing.assert_close(catmull_rom(p, knots), p)
        torch.testing.assert_close(catmull_rom(p.flip(1), t).flip(1), curve)
        curve.square().mean().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad[:,1:-1].abs().sum()), 0)
        coincident = torch.zeros(1,8,3,requires_grad=True)
        catmull_rom(coincident, t.float()).sum().backward()
        self.assertTrue(torch.isfinite(coincident.grad).all())

    def test_head_budget_endpoints_chord_and_physical_metric(self):
        bspline = SplineHead(12,16)
        head = SplineHead(12,16,family='catmull_rom')
        self.assertEqual(sum(p.numel() for p in head.parameters()), sum(p.numel() for p in bspline.parameters()))
        endpoints = torch.tensor([[[.2,.3,.4],[.8,.7,.6]]], requires_grad=True)
        metric = torch.diag(torch.tensor([-32.,-64.,48.]))
        controls, curve = head(torch.zeros(1,12), endpoints, metric)
        t = torch.linspace(0,1,64)[None,:,None]
        torch.testing.assert_close(curve, (1-t)*endpoints[:,:1]+t*endpoints[:,1:], atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(controls[:,[0,-1]],endpoints,atol=0,rtol=0)
        with torch.no_grad():
            head.network[-1].bias.copy_(torch.linspace(-2,2,18))
        controls, curve = head(torch.zeros(1,12), endpoints, metric)
        expected = reference((controls[0] @ metric.T).detach().numpy(), np.linspace(0,1,64)) @ np.linalg.inv(metric.numpy()).T
        np.testing.assert_allclose(curve[0].detach(), expected, atol=4e-7)
        curve.square().mean().backward()
        self.assertGreater(float(head.network[-1].bias.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(endpoints.grad).all())

    def test_cache_preserves_graph_and_guides(self):
        x = np.linspace(0,1,31)
        path = np.column_stack((x,.3+.1*np.sin(2*np.pi*x),x*.2)).astype('float32')
        data = dict(nodes=path[[0,-1]],edges=np.array([[0,1]]),full_paths=path,path_offsets=np.array([0,len(path)]),metric=np.diag([32.,48.,64.]).astype('float32'),boundary=np.array([True,False]),guides=np.zeros((1,64,3),dtype='float32'))
        stream = io.BytesIO(); np.savez(stream,**data); stream.seek(0)
        with np.load(stream) as cache:
            result = convert_target(cache)
        for key,value in data.items():
            np.testing.assert_array_equal(result[key],value)
        np.testing.assert_array_equal(result['controls'][0,[0,-1]],data['nodes'])
        self.assertEqual(result['controls'].shape,(1,8,3))
        physical = result['controls'][0] @ data['metric'].T
        expected = reference(physical,np.linspace(0,1,64)) @ np.linalg.inv(data['metric']).T
        np.testing.assert_allclose(result['curves'][0],expected,atol=2e-7)


if __name__ == '__main__':
    unittest.main()
