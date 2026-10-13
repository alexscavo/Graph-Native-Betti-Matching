import torch
from configs import load_config
from models.matcher import build_matcher
from training.losses.criterion import GraphCriterion
from training.losses.relation_distance import DistanceRelationCriterion, nearby_negative_weights


def test_weighting_preserves_positive_supervision_and_negative_mass():
    labels = torch.tensor([1, 1, 0, 0, 0])
    distances = torch.tensor([1., 30., 2., 10., 20.])
    weights = nearby_negative_weights(labels, distances, 5., 5.)
    torch.testing.assert_close(weights[:2], torch.ones(2))
    torch.testing.assert_close(weights[2:].sum(), torch.tensor(3.))
    torch.testing.assert_close(weights[2] / weights[3], torch.tensor(5.))
    torch.testing.assert_close(nearby_negative_weights(labels, distances, 5., 1.), torch.ones(5))


def test_control_matches_existing_loss_and_gradients_with_permuted_assignments():
    cfg = load_config('configs/overfit_spline_v0.yaml', environment={
        'GNBM_OUTPUT_DIR': '/tmp/unused', 'SYNTHETIC_MRI_DATASET': '/tmp/unused'})
    cfg['model']['decoder'].update(object_queries=3, relation_tokens=1)
    cfg['loss']['edge']['classification'].update(name='focal', class_weights=[1., 1.])
    cfg['loss']['edge']['classification']['curriculum']['enabled'] = False
    cfg['loss']['edge']['balancing']['mode'] = 'none'
    cfg['loss']['edge']['candidates'].update(positive_cap=None, include_unmatched=False)
    cfg['loss']['degree']['enabled'] = False
    relation = torch.nn.Linear(12, 2)
    tokens = torch.randn(1, 4, 4)
    targets = {'nodes': [torch.tensor([[0., 0., 0.], [.8, 0., 0.], [.05, .05, 0.]])],
               'edges': [torch.tensor([[0, 1]])], 'spline': [{'metric': torch.eye(3) * 20}]}
    assignments = [(torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0]))]
    baseline = GraphCriterion(cfg, build_matcher(cfg), relation)
    control = DistanceRelationCriterion(cfg, build_matcher(cfg), relation, radius_mm=5., factor=1.)
    weighted = DistanceRelationCriterion(cfg, build_matcher(cfg), relation, radius_mm=5., factor=5.)
    losses, gradients = [], []
    for criterion in (baseline, control, weighted):
        torch.manual_seed(17)
        loss, _ = criterion.loss_edges(tokens, None, targets['edges'], assignments, targets=targets)
        losses.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, tuple(relation.parameters())))
    torch.testing.assert_close(losses[0], losses[1])
    for a, b in zip(gradients[0], gradients[1]): torch.testing.assert_close(a, b)
    assert weighted.last_pair_counts == {'positive': 1, 'negative': 2, 'nearby_negative': 1}
    assert any(not torch.allclose(a, b) for a, b in zip(gradients[1], gradients[2]))
