"""Small CPU checks of the real loss, freeze behavior, and paired measurements."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from configs import load_config
from models.spline import SplineHead, cubic_basis
from models.matcher import build_matcher
from training.losses.criterion import GraphCriterion
from scripts.frozen_spline_diagnostic import freeze_detector, detector_digest, frozen_step, matched_geometry


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv3d(1,4,1),torch.nn.BatchNorm3d(4))
        self.queries = torch.nn.Parameter(torch.randn(3,4))
        self.coord = torch.nn.Linear(4,6)
        self.classes = torch.nn.Linear(4,2)
        self.relation_embed = torch.nn.Linear(12,2)
        self.spline_head = SplineHead(12,16)

    def forward(self, image):
        features = self.encoder(image)
        tokens = features.mean((2,3,4))[:,None]+self.queries[None]
        return tokens, {'pred_nodes':self.coord(tokens[:,:2]).sigmoid(),
                        'pred_logits':self.classes(tokens[:,:2])}, features


class FrozenSplineTests(unittest.TestCase):
    def test_residual_loss_uses_mm_and_removes_direct_endpoint_shift(self):
        from training.engine import _move_graph_batch
        cfg,model,criterion,batch = self.fixture()
        image,targets = _move_graph_batch(batch,'cpu','image',True)
        tokens,predictions,_ = model(image)
        assignments = criterion.matcher(predictions,targets)
        geometry = targets['spline'][0]
        g = model.spline_head.greville[None,1:-1,None]
        gt = geometry['controls'];metric = geometry['metric']
        residual = (gt[:,1:-1]-((1-g)*gt[:,:1]+g*gt[:,-1:])) @ metric.T
        expected = torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=1.)
        criterion.config['loss']['spline']['target'] = 'residual'
        base = criterion.loss_spline(tokens,predictions,targets,assignments)['spline']
        shifted = predictions['pred_nodes'].detach().clone()
        shifted[:,:,:3] += torch.tensor([.03,-.02,.01])
        shifted.requires_grad_()
        moved = criterion.loss_spline(tokens,{'pred_nodes':shifted},targets,assignments)['spline']
        torch.testing.assert_close(base,expected,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(moved,base,atol=2e-6,rtol=1e-5)
        endpoint_grad,bias_grad = torch.autograd.grad(moved,(shifted,model.spline_head.network[-1].bias))
        self.assertLess(float(endpoint_grad.abs().max()),1e-5)
        self.assertGreater(float(bias_grad.abs().sum()),0)
        # With learned residual weights, endpoint FEATURES remain a gradient
        # route; the frozen diagnostic blocks that route via no_grad forward.
        with torch.no_grad(): model.spline_head.network[-1].weight.fill_(.01)
        coupled = criterion.loss_spline(tokens,{'pred_nodes':shifted},targets,assignments)['spline']
        self.assertGreater(float(torch.autograd.grad(coupled,shifted)[0].abs().sum()),0)

    def test_endpoint_only_head_ignores_tokens_at_train_and_inference(self):
        torch.manual_seed(7)
        full = SplineHead(12,16)
        endpoints_only = SplineHead(12,16,token_inputs='zero')
        endpoints_only.load_state_dict(full.state_dict())
        with torch.no_grad():
            full.network[-1].weight.normal_()
            endpoints_only.load_state_dict(full.state_dict())
        endpoints = torch.tensor([[[.1,.2,.3],[.7,.8,.9]]])
        metric = torch.diag(torch.tensor([-32.,48.,64.]))
        a,b = torch.randn(1,12),torch.randn(1,12)*5
        for training in (True,False):
            endpoints_only.train(training)
            for x,y in zip(endpoints_only(a,endpoints,metric),endpoints_only(b,endpoints,metric)):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
        self.assertGreater(float((full(a,endpoints,metric)[0]-full(b,endpoints,metric)[0]).abs().sum()),0)
        self.assertEqual(sum(p.numel() for p in full.parameters()),sum(p.numel() for p in endpoints_only.parameters()))

    def fixture(self):
        torch.manual_seed(31)
        cfg = load_config('configs/overfit_spline_v0.yaml',environment={'GNBM_OUTPUT_DIR':'/tmp/unused','SYNTHETIC_MRI_DATASET':'/tmp/unused'})
        cfg['model']['decoder'].update(object_queries=2,relation_tokens=1)
        model = TinyDetector().eval()
        image = torch.randn(1,1,4,4,4)
        with torch.no_grad():
            nodes = model(image)[1]['pred_nodes'][0,:,:3]
            metric = torch.diag(torch.tensor([32.,48.,64.]))
            controls,_ = model.spline_head(torch.zeros(1,12),nodes[None],metric)
            controls[:,1:-1,1] += .04
            curves = model.spline_head.basis @ controls
        batch = (image,torch.ones_like(image),[nodes],[torch.tensor([[0,1]])],
                 [dict(controls=controls,curves=curves,guides=curves,metric=metric)],torch.ones(1,dtype=torch.long))
        criterion = GraphCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head)
        return cfg,model,criterion,batch

    def test_head_updates_detector_and_bn_remain_exactly_fixed(self):
        cfg,model,criterion,batch = self.fixture()
        freeze_detector(model)
        digest = detector_digest(model)
        before = model(batch[0])[1]['pred_nodes'].detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        initial_head = model.spline_head.network[-1].bias.detach().clone()
        losses = [frozen_step(model,criterion,optimizer,scheduler,batch,cfg,'cpu') for _ in range(3)]
        self.assertEqual(detector_digest(model),digest)
        torch.testing.assert_close(model(batch[0])[1]['pred_nodes'],before,atol=0,rtol=0)
        self.assertFalse(model.encoder[1].training)
        self.assertFalse(torch.equal(initial_head,model.spline_head.network[-1].bias))
        self.assertTrue(all(torch.isfinite(row['total']) for row in losses))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('spline_head.')))

    def test_zero_residual_matches_identical_endpoint_chords_and_reports_pairs(self):
        cfg,model,criterion,batch = self.fixture()
        freeze_detector(model)
        with tempfile.TemporaryDirectory() as directory:
            leaf = Path(directory)/'val'; (leaf/'spline').mkdir(parents=True)
            geometry = batch[4][0]
            dense = cubic_basis(8,257)[0] @ geometry['controls'][0]
            np.savez(leaf/'spline/sample.npz',full_paths=dense.numpy(),path_offsets=[0,len(dense)])
            class Loader:
                dataset = SimpleNamespace(records=[SimpleNamespace(sample_id='sample',graph=leaf/'vtp/sample_graph.vtp')])
                def __iter__(self): return iter([batch])
            result = matched_geometry(model,criterion,Loader(),cfg,'cpu')
            self.assertEqual(result['all_pairs']['branches'],1)
            self.assertEqual(result['endpoints_within_1mm']['branches'],1)
            for prefix in ('fit','guide','endpoint_neutral_guide'):
                self.assertAlmostEqual(result['all_pairs'][prefix+'_curve_mm'],result['all_pairs'][prefix+'_chord_mm'],places=5)
            repeated = matched_geometry(model,criterion,Loader(),cfg,'cpu')
            self.assertEqual(result['assignment_sha256'],repeated['assignment_sha256'])
            stratified = matched_geometry(model,criterion,Loader(),cfg,'cpu',curvature_breakdown=True)
            self.assertEqual(sum(g['branches'] for g in stratified['curvature_groups'].values()),1)
            self.assertEqual(result['assignment_sha256'],stratified['assignment_sha256'])

    def test_greville_chord_identity_and_endpoint_translation_coupling(self):
        basis,g = cubic_basis(8,64)
        t = torch.linspace(0,1,64)
        torch.testing.assert_close(basis @ g,t)
        gt_endpoints = torch.tensor([[0.,0.,0.],[12.,4.,2.]])
        shift = torch.tensor([3.,-1.,2.],requires_grad=True)
        gt_chord = (1-g[:,None])*gt_endpoints[:1]+g[:,None]*gt_endpoints[1:]
        head = SplineHead(12,16)
        predicted_controls,_ = head(torch.zeros(1,12),(gt_endpoints+shift)[None],torch.eye(3))
        pred_chord = (1-g[:,None])*(gt_endpoints[:1]+shift)+g[:,None]*(gt_endpoints[1:]+shift)
        # Same true shape, shifted node estimates: absolute controls penalize
        # the translation while the residual diagnostic cancels it exactly.
        absolute = (predicted_controls[0,1:-1]-gt_chord[1:-1]).square().mean()
        absolute.backward()
        self.assertGreater(float(shift.grad.norm()),0)
        torch.testing.assert_close(predicted_controls[0]-pred_chord,torch.zeros_like(gt_chord),atol=2e-6,rtol=0)


if __name__ == '__main__': unittest.main()
