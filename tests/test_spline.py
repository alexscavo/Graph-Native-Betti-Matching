"""Focused checks for the fixed basis and matched-positive geometry path."""
import copy
import unittest
import numpy as np
import torch
from scipy.interpolate import BSpline
from configs import load_config
from models.spline import SplineHead, cubic_basis
from models.matcher import build_matcher
from training.losses.criterion import GraphCriterion


class SplineTests(unittest.TestCase):
    def test_two_segment_closed_branch_excluded_before_contraction(self):
        from data.spline_targets import crop_dense_paths, contract_spline_graph
        from scripts.audit_synthetic_mri_grid import SourceGraph, CenterlineEdge
        nodes = {0: np.array([1., 1., 1.])}
        edge = CenterlineEdge(0, 0, (np.array([2., 1., 1.]),))
        source = SourceGraph(nodes, [(0, 0)], [edge])
        dense = crop_dense_paths(source, np.array([[0., 3.]]*3),
                                 boundary_policy='last_inside_sample')
        self.assertEqual(dense.closed_components, 1)
        self.assertEqual(len(dense.edges), 2)
        graph, reason = contract_spline_graph(dense.positions, dense.edges,
                                             closed_components=dense.closed_components)
        self.assertIsNone(graph)
        self.assertEqual(reason, 'closed_branch')

    def test_short_parallel_branches_excluded_before_contraction(self):
        from data.spline_targets import crop_dense_paths, contract_spline_graph
        from scripts.audit_synthetic_mri_grid import SourceGraph, CenterlineEdge
        nodes = {0: np.array([1., 1., 1.]), 1: np.array([2., 1., 1.])}
        edge = CenterlineEdge(0, 1, (nodes[0], nodes[1]))
        source = SourceGraph(nodes, [(0, 1), (0, 1)], [edge, edge])
        dense = crop_dense_paths(source, np.array([[0., 3.]]*3),
                                 boundary_policy='last_inside_sample')
        self.assertEqual(len(dense.edges), 2)
        graph, reason = contract_spline_graph(dense.positions, dense.edges)
        self.assertIsNone(graph)
        self.assertEqual(reason, 'parallel_branches')
        # Neither the original records nor the cropped paths were merged.
        self.assertEqual(len(dense.edges), 2)
        valid, reason = contract_spline_graph(dense.positions, dense.edges[:1])
        self.assertIsNone(reason)
        self.assertEqual(len(valid.branches), 1)

    def test_boundary_samples_and_sparse_component_fallback(self):
        from data.spline_targets import crop_dense_paths
        from scripts.audit_synthetic_mri_grid import SourceGraph, CenterlineEdge, patch_voxel_cell_bounds
        from metrics.branch_connectivity import contract_degree_two
        nodes = {0: np.array([-2., 1., 1.]), 1: np.array([4., 1., 1.])}
        positions = tuple(np.array([x, 1., 1.]) for x in (-2., 0., 1., 2., 4.))
        source = SourceGraph(nodes, [(0, 1)], [CenterlineEdge(0, 1, positions)])
        bounds = patch_voxel_cell_bounds((0, 0, 0), (3, 3, 3))
        dense = crop_dense_paths(source, bounds, boundary_policy='last_inside_sample')
        graph = contract_degree_two(dense.positions, dense.edges)
        np.testing.assert_allclose(graph.branches[0].polyline[[0, -1]], [[0, 1, 1], [2, 1, 1]])
        self.assertEqual(dense.snapped_endpoints, 2)
        sparse = SourceGraph(nodes, [(0, 1)], [CenterlineEdge(0, 1, (nodes[0], nodes[1]))])
        dense = crop_dense_paths(sparse, bounds, boundary_policy='last_inside_sample')
        self.assertEqual(dense.sparse_sample_fallbacks, 1)
        self.assertEqual(len(dense.edges), 1)
        np.testing.assert_allclose(dense.positions[:, 0], [-.5, 2.5])

    def test_singleton_crop_contact_is_an_isolated_node(self):
        from data.spline_targets import crop_dense_paths
        from scripts.audit_synthetic_mri_grid import SourceGraph, CenterlineEdge
        from metrics.branch_connectivity import contract_degree_two
        nodes = {0: np.array([-1., -1., 0.]), 1: np.array([-1., 1., 0.])}
        edge = CenterlineEdge(0, 1, (nodes[0], np.zeros(3), nodes[1]))
        source = SourceGraph(nodes, [(0, 1)], [edge])
        bounds = np.array([[0., 2.]]*3)
        old = crop_dense_paths(source, bounds)
        kept = crop_dense_paths(source, bounds, retain_tangent_contacts=True)
        self.assertEqual(old.tangent_contacts, 1)
        self.assertEqual(len(old.positions), 0)
        self.assertEqual(len(kept.positions), 1)
        self.assertEqual(len(kept.edges), 0)
        self.assertTrue(kept.boundary[0])
        graph = contract_degree_two(kept.positions, kept.edges)
        self.assertEqual(graph.roles, {0: 'isolated'})

    def test_fixed_basis_and_zero_residual_chord(self):
        basis,_=cubic_basis(8,64)
        knots=np.r_[np.zeros(4),np.arange(1,5)/5,np.ones(4)]
        reference=BSpline(knots,np.eye(8),3)(np.linspace(0,1,64))
        np.testing.assert_allclose(basis.numpy(),reference,atol=1e-7)
        head=SplineHead(12,16)
        endpoints=torch.tensor([[[.2,.3,.4],[.8,.7,.6]]],requires_grad=True)
        metric=torch.diag(torch.tensor([-64.,-64.,64.]))
        controls,curve=head(torch.zeros(1,12),endpoints,metric)
        t=torch.linspace(0,1,64)[None,:,None]
        torch.testing.assert_close(curve,(1-t)*endpoints[:,:1]+t*endpoints[:,1:],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(controls[:,[0,-1]],endpoints,atol=0,rtol=0)
        curve.sum().backward();self.assertGreater(float(endpoints.grad.abs().sum()),0)

    def test_geometry_matching_orientation_and_both_losses(self):
        config=load_config('configs/overfit_spline_v0.yaml',environment={'GNBM_OUTPUT_DIR':'/tmp/spline-tests','SYNTHETIC_MRI_DATASET':'/tmp/unused'})
        config['model']['decoder'].update(object_queries=2,relation_tokens=1)
        head=SplineHead(12,16)
        relation=torch.nn.Linear(12,2)
        criterion=GraphCriterion(config,build_matcher(config),relation,spline_head=head)
        nodes=torch.tensor([[.2,.3,.4],[.8,.7,.6]])
        # Source queries are swapped relative to target identities.
        predicted=nodes.flip(0).reshape(1,2,3).clone().requires_grad_()
        metric=torch.diag(torch.tensor([-64.,-64.,64.]))
        with torch.no_grad():
            controls,curve=head(torch.zeros(1,12),nodes.reshape(1,2,3),metric)
        targets={'nodes':[nodes],'edges':[torch.tensor([[0,1]])],
                 'spline':[{'metric':metric,'controls':controls,'curves':curve}]}
        tokens=torch.randn(1,3,4,requires_grad=True)
        assignments=[(torch.tensor([0,1]),torch.tensor([1,0]))]
        for mode in ('controls','curve'):
            criterion.config['loss']['spline']['mode']=mode
            exact=criterion.loss_spline(tokens,{'pred_nodes':predicted},targets,assignments)
            self.assertLess(float(exact['spline']),1e-10)
        displaced=predicted+torch.tensor([.01,0,0])
        loss=criterion.loss_spline(tokens,{'pred_nodes':displaced},targets,assignments)['spline']
        loss.backward()
        self.assertGreater(float(predicted.grad.abs().sum()),0)
        self.assertGreater(float(head.network[-1].bias.grad.abs().sum()),0)
        empty=copy.deepcopy(targets);empty['edges']=[torch.empty(0,2,dtype=torch.long)]
        result=criterion.loss_spline(tokens,{'pred_nodes':predicted},empty,assignments)
        self.assertEqual(float(result['spline']),0)


if __name__=='__main__':unittest.main()
