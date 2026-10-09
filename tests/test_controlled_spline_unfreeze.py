"""Targeted checks of diagnostic gradient routing and frozen CNN/BN state."""
import unittest
import torch
from models.matcher import build_matcher
from training.engine import _move_graph_batch
from scripts.controlled_spline_unfreeze import RoutedCriterion, configure_unfreezing, frozen_state_digest, assignment_change
from tests.test_frozen_spline_diagnostic import FrozenSplineTests


class ControlledUnfreezeTests(unittest.TestCase):
    def test_routes_preserve_values_but_isolate_geometry_gradients(self):
        cfg, model, _, batch = FrozenSplineTests().fixture()
        cfg['loss']['spline']['target'] = 'residual'
        with torch.no_grad(): model.spline_head.network[-1].weight.normal_(std=.03)
        image, targets = _move_graph_batch(batch, 'cpu', 'image', True)
        raw_tokens, raw_predictions, _ = model(image)
        values = []
        for route in ['graph_only', 'tokens_only', 'joint']:
            tokens = raw_tokens.detach().requires_grad_()
            nodes = raw_predictions['pred_nodes'].detach().requires_grad_()
            predictions = dict(raw_predictions, pred_nodes=nodes)
            criterion = RoutedCriterion(cfg, build_matcher(cfg), model.relation_embed,
                                        spline_head=model.spline_head, route=route)
            assignments = criterion.matcher(predictions, targets)
            loss = criterion.loss_spline(tokens, predictions, targets, assignments)['spline']
            a,b,h = torch.autograd.grad(loss, (tokens,nodes,model.spline_head.network[-1].weight),allow_unused=True)
            values.append(loss.detach())
            self.assertGreater(float(h.abs().sum()),0)
            if route == 'graph_only': self.assertIsNone(a)
            else: self.assertGreater(float(a.abs().sum()),0)
            if route != 'joint': self.assertIsNone(b)
            else: self.assertGreater(float(b.abs().sum()),0)
        for value in values[1:]: torch.testing.assert_close(value,values[0],atol=0,rtol=0)

    def test_graph_control_matches_graph_only_detector_update(self):
        cfg, model, _, batch = FrozenSplineTests().fixture()
        cfg['loss']['spline']['target'] = 'residual'
        configure_unfreezing(model)
        before = frozen_state_digest(model)
        image, targets = _move_graph_batch(batch,'cpu','image',True)
        criterion = RoutedCriterion(cfg,build_matcher(cfg),model.relation_embed,
                                    spline_head=model.spline_head,route='graph_only')
        tokens,predictions,_ = model(image)
        losses = criterion(tokens,predictions,targets)
        detector = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('spline_head.')]
        graph = losses['total']-criterion.weights['spline']*losses['spline']
        a = torch.autograd.grad(graph,detector,retain_graph=True,allow_unused=True)
        b = torch.autograd.grad(losses['total'],detector,retain_graph=True,allow_unused=True)
        for x,y in zip(a,b):
            if x is None: self.assertIsNone(y)
            else: torch.testing.assert_close(x,y,atol=0,rtol=0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        losses['total'].backward(); optimizer.step()
        self.assertEqual(before,frozen_state_digest(model))
        self.assertFalse(model.encoder[1].training)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertEqual(assignment_change({'a':{'0':2,'1':1}},{'a':{'0':0,'1':1}}),.5)


if __name__ == '__main__': unittest.main()
