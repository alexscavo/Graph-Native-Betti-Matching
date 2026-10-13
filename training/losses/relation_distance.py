"""Pilot-only weighting of nearby GT nonedges; the main criterion is unchanged."""
import torch
from .criterion import GraphCriterion
from .focal import softmax_focal_loss


def nearby_negative_weights(labels, distances_mm, radius_mm, factor):
    """Retain positive weights and total negative mass within each graph.

    All candidates remain supervised. Nearby/far negative relative weight is
    `factor`; rescaling avoids changing overall positive/negative balance.
    """
    weights = torch.ones_like(distances_mm)
    negative = labels == 0
    if negative.any():
        weights[negative & (distances_mm < radius_mm)] = factor
        weights[negative] *= negative.sum() / weights[negative].sum()
    return weights


class DistanceRelationCriterion(GraphCriterion):
    def __init__(self, *args, radius_mm, factor, **kwargs):
        super().__init__(*args, **kwargs)
        assert radius_mm > 0 and factor >= 1
        assert self.balancing['mode'] == 'none' and not self.degree['enabled']
        assert not self.candidates['include_unmatched']
        assert self.candidates['positive_cap'] is None
        assert self.edge_config['name'] == 'focal'
        assert self.edge_config['class_weights'] == [1., 1.]
        assert self.spline_head is None or self.spline_head.curve_evidence is None
        self.radius_mm, self.factor = float(radius_mm), float(factor)
        self.last_pair_counts = {}

    def loss_spline(self, tokens, predictions, targets, assignments):
        # Geometry parameters are frozen and the pilot gives geometry zero weight.
        assert self.weights['spline'] == 0
        return {'spline': predictions['pred_nodes'].sum() * 0.}

    def loss_edges(self, tokens, node_logits, target_edges, assignments, predictions=None, targets=None):
        contexts = []
        features, labels, _ = self._matched_edge_pool(tokens, target_edges, assignments, contexts)
        zero = sum(p.sum() * 0. for p in self.relation_embed.parameters())
        if not features:
            self.last_pair_counts = dict(positive=0, negative=0, nearby_negative=0)
            return zero, zero
        weights, nearby = [], 0
        for (batch, query_pairs), label in zip(contexts, labels):
            source, target = assignments[batch]
            lookup = torch.full((self.object_queries,), -1, dtype=torch.long, device=tokens.device)
            lookup[source.to(tokens.device)] = target.to(tokens.device)
            target_pairs = lookup[query_pairs]
            assert (target_pairs >= 0).all()
            positions = targets['nodes'][batch][target_pairs]
            metric = targets['spline'][batch]['metric'].to(tokens)
            distance = torch.linalg.vector_norm((positions[:, 0] - positions[:, 1]) @ metric.T, dim=-1)
            nearby += int(((label == 0) & (distance < self.radius_mm)).sum())
            weights.append(nearby_negative_weights(label, distance, self.radius_mm, self.factor))
        logits = self.relation_embed(torch.cat(features))
        labels = torch.cat(labels)
        losses = softmax_focal_loss(logits, labels, class_weights=self.edge_config['class_weights'],
            gamma=self._classification_gamma(self.edge_config, self.progress_percent), reduction='none')
        self.last_pair_counts = dict(positive=int((labels == 1).sum()), negative=int((labels == 0).sum()),
                                     nearby_negative=nearby)
        return (losses * torch.cat(weights)).mean() + zero, zero
