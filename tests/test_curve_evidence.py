"""Coordinate convention, undirected scoring, and real loss/inference integration."""
import unittest
import torch
from configs import load_config
from models.curve_evidence import sample_along_curves, curve_relation_logits
from models.matcher import build_matcher
from models.spline import SplineHead
from training.evaluation.inference import infer_graphs
from training.losses.criterion import GraphCriterion


class CurveEvidenceTests(unittest.TestCase):
    def test_voxel_centers_and_fractional_dhw_coordinates(self):
        axes = torch.meshgrid(torch.arange(5.), torch.arange(6.), torch.arange(7.), indexing='ij')
        features = torch.stack(axes).requires_grad_()
        voxels = torch.tensor([[[1., 2., 3.], [2.25, 3.5, 4.75]]], requires_grad=True)
        samples = sample_along_curves(features, voxels / torch.tensor([5., 6., 7.]))
        torch.testing.assert_close(samples.transpose(1, 2), voxels)
        samples.sum().backward()
        self.assertGreater(float(voxels.grad.abs().sum()), 0)
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def setup_case(self):
        torch.manual_seed(17)
        config = load_config('configs/overfit_spline_v0.yaml', environment={
            'GNBM_OUTPUT_DIR': '/tmp/evidence-test', 'SYNTHETIC_MRI_DATASET': '/tmp/unused'})
        config['model']['decoder'].update(object_queries=3, relation_tokens=1)
        head = SplineHead(12, 16, curve_evidence={'enabled': True, 'channels': 2, 'samples': 16, 'chunk_size': 2})
        relation = torch.nn.Linear(12, 2)
        tokens = torch.randn(1, 4, 4, requires_grad=True)
        nodes = torch.tensor([[.2, .2, .3], [.6, .7, .6], [.3, .6, .4]], requires_grad=True)
        metric = torch.eye(3) * 8
        image = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        return config, head, relation, tokens, nodes, metric, image

    def test_zero_correction_preserves_scores_and_orientation_is_symmetric(self):
        _, head, relation, tokens, nodes, metric, image = self.setup_case()
        objects, shared = tokens[0, :3], tokens[0, 3:]
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
        features = head.curve_evidence.encode(image)[0]
        score = lambda pairs: curve_relation_logits(relation, head, objects, shared, nodes, pairs, metric, features)
        f = torch.cat((objects[pairs[:, 0]], objects[pairs[:, 1]], shared.expand(3, -1)), -1)
        b = torch.cat((objects[pairs[:, 1]], objects[pairs[:, 0]], shared.expand(3, -1)), -1)
        torch.testing.assert_close(score(pairs), .5 * (relation(f) + relation(b)))
        with torch.no_grad():
            head.curve_evidence.scorer[-1].weight.normal_(std=.1)
        torch.testing.assert_close(score(pairs), score(pairs.flip(-1)))

    def test_candidate_scores_match_inference_and_edge_loss_reaches_geometry(self):
        config, head, relation, tokens, nodes, metric, image = self.setup_case()
        with torch.no_grad():
            head.curve_evidence.scorer[-1].weight.normal_(std=.1)
        criterion = GraphCriterion(config, build_matcher(config), relation, spline_head=head)
        predictions = {'pred_nodes': torch.cat((nodes, torch.full_like(nodes, .2)), -1)[None],
                       'pred_logits': torch.tensor([[[-2., 2.]] * 3]),
                       'curve_image_features': head.curve_evidence.encode(image)}
        targets = {'nodes': [nodes.detach()], 'edges': [torch.tensor([[0, 1]])], 'spline': [{'metric': metric}]}
        assignments = [(torch.arange(3), torch.arange(3))]
        contexts = []
        criterion._matched_edge_pool(tokens, targets['edges'], assignments, contexts)
        pair_scores = {}
        for batch, pairs in contexts:
            logits = criterion._curve_logits(tokens, predictions, targets, batch, pairs)
            pair_scores.update({tuple(sorted(p)): float(s) for p, s in zip(pairs.tolist(), logits.softmax(-1)[:, 1].detach())})
        graph = infer_graphs(tokens, predictions, relation, object_queries=3, relation_tokens=1,
                             edge_threshold=0., spline_head=head, spline_metrics=[metric])[0]
        for pair, score in zip(graph['edges'].tolist(), graph['edge_scores']):
            self.assertAlmostEqual(float(score), pair_scores[tuple(sorted(pair))], places=6)
        loss, _ = criterion.loss_edges(tokens, predictions['pred_logits'], targets['edges'], assignments, predictions, targets)
        loss.backward()
        for grad in (nodes.grad, image.grad, head.network[-1].bias.grad,
                     head.curve_evidence.encoder[0].weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
