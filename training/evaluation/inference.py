"""Convert RelationFormer tokens into explicit undirected graphs."""

from __future__ import annotations

import math

import torch


def _soft_degree_filter(pairs, logits, type_probabilities, penalty, threshold_logit):
    """Prefer patch-local degrees without imposing an inviolable degree cap.

    Each additional edge pays the probability-weighted cost of exceeding the
    predicted 0/1/2 degree. Junction probability has no degree penalty. The
    unmodified relation logit can still override this preference.
    """
    order = (logits[:, 1] - logits[:, 0]).argsort(descending=True).tolist()
    pair_list = pairs.tolist()
    margins = (logits[:, 1] - logits[:, 0]).tolist()
    types = type_probabilities.tolist()
    degree = [0] * len(types)
    selected = [False] * len(pairs)
    for index in order:
        left, right = pair_list[index]

        def cost(node):
            current = degree[node]
            probabilities = types[node]
            return (
                probabilities[0]
                + probabilities[1] * float(current >= 1)
                + probabilities[2] * float(current >= 2)
            )
        adjusted = margins[index] - penalty * (cost(left) + cost(right))
        if adjusted > threshold_logit:
            selected[index] = True
            degree[left] += 1
            degree[right] += 1
    return torch.as_tensor(selected, device=pairs.device, dtype=torch.bool)


@torch.no_grad()
def infer_graphs(
    tokens,
    predictions,
    relation_embed,
    *,
    object_queries,
    relation_tokens,
    node_threshold=None,
    edge_threshold=None,
    node_type_degree_caps=False,
    node_type_degree_penalty=0.0,
    spline_head=None,
    spline_metrics=None,
):
    object_features = tokens[..., :object_queries, :]
    shared_relations = tokens[
        ..., object_queries : object_queries + relation_tokens, :
    ]
    node_probabilities = predictions["pred_logits"].softmax(-1)[..., 1]
    if node_threshold is None:
        valid = predictions["pred_logits"].argmax(-1) == 1
    else:
        valid = node_probabilities > float(node_threshold)
    if float(node_type_degree_penalty) < 0:
        raise ValueError("node_type_degree_penalty must be non-negative")
    if node_type_degree_caps and node_type_degree_penalty:
        raise ValueError("hard degree caps and soft degree penalty are mutually exclusive")
    if (node_type_degree_caps or node_type_degree_penalty) and "pred_node_types" not in predictions:
        raise ValueError("degree-aware decoding requires pred_node_types")

    graphs = []
    use_evidence = spline_head is not None and spline_head.curve_evidence is not None
    if 'curve_image_features' in predictions and not use_evidence:
        raise ValueError('Curve image features require the matching spline/evidence head')
    for batch in range(tokens.shape[0]):
        query_ids = torch.nonzero(valid[batch], as_tuple=False).flatten()
        boxes = predictions["pred_nodes"][batch, query_ids]
        nodes = boxes[:, :3]
        scores = node_probabilities[batch, query_ids]
        if query_ids.numel() < 2:
            pairs = torch.empty((0, 2), dtype=torch.long, device=tokens.device)
            relation_scores = scores.new_empty((0,))
        else:
            local_pairs = torch.combinations(
                torch.arange(query_ids.numel(), device=tokens.device), r=2
            )
            query_pairs = query_ids[local_pairs]
            left = object_features[batch, query_pairs[:, 0]]
            right = object_features[batch, query_pairs[:, 1]]
            if relation_tokens:
                relation = shared_relations[batch].reshape(1, -1).expand(
                    local_pairs.shape[0], -1
                )
                forward = torch.cat((left, right, relation), dim=-1)
                reverse = torch.cat((right, left, relation), dim=-1)
            else:
                forward = torch.cat((left, right), dim=-1)
                reverse = torch.cat((right, left), dim=-1)
            relation_logits = 0.5 * (
                relation_embed(forward) + relation_embed(reverse)
            )
            if use_evidence:
                from models.curve_evidence import curve_relation_logits
                if spline_metrics is None:
                    raise ValueError('Curve evidence requires physical metric metadata')
                relation_logits = curve_relation_logits(relation_embed, spline_head,
                    object_features[batch], shared_relations[batch], predictions['pred_nodes'][batch, :, :3],
                    query_pairs, spline_metrics[batch].to(tokens), predictions['curve_image_features'][batch])
            probabilities = relation_logits.softmax(-1)[:, 1]
            if node_type_degree_penalty:
                if edge_threshold is None:
                    cutoff = 0.0
                    candidates = relation_logits.argmax(-1) == 1
                else:
                    threshold = float(edge_threshold)
                    cutoff = (
                        -math.inf if threshold == 0.0 else
                        math.inf if threshold == 1.0 else
                        math.log(threshold / (1.0 - threshold))
                    )
                    candidates = probabilities > threshold
                type_probabilities = predictions["pred_node_types"][batch, query_ids].softmax(-1)
                keep = torch.zeros_like(candidates)
                if candidates.any():
                    keep[candidates] = _soft_degree_filter(
                        local_pairs[candidates],
                        relation_logits[candidates],
                        type_probabilities,
                        float(node_type_degree_penalty),
                        cutoff,
                    )
            elif edge_threshold is None:
                keep = relation_logits.argmax(-1) == 1
            else:
                keep = probabilities > float(edge_threshold)
            pairs = local_pairs[keep]
            relation_scores = probabilities[keep]
            if node_type_degree_caps and pairs.numel():
                types = predictions["pred_node_types"][batch, query_ids].argmax(-1)
                # Junctions remain unrestricted; the four labels encode
                # within-patch degrees 0, 1, 2, and >=3 respectively.
                caps = torch.where(types == 3, len(query_ids), types)
                degrees = torch.zeros(len(query_ids), device=tokens.device, dtype=torch.long)
                selected = torch.zeros(len(pairs), device=tokens.device, dtype=torch.bool)
                for index in relation_scores.argsort(descending=True).tolist():
                    left, right = pairs[index].tolist()
                    if degrees[left] < caps[left] and degrees[right] < caps[right]:
                        selected[index] = True
                        degrees[left] += 1
                        degrees[right] += 1
                pairs = pairs[selected]
                relation_scores = relation_scores[selected]
        spline_output = {}
        if spline_head is not None:
            from models.spline import canonical_reverse
            if spline_metrics is None:
                raise ValueError('Spline inference requires normalized-to-physical metric metadata')
            metric = spline_metrics[batch].to(tokens)
            if pairs.numel():
                endpoints = nodes[pairs]
                reverse = canonical_reverse(endpoints @ metric.T)
                pairs = pairs.clone(); pairs[reverse] = pairs[reverse][:, [1,0]]
                query_pairs = query_ids[pairs]
                parts = [object_features[batch, query_pairs[:,0]], object_features[batch, query_pairs[:,1]]]
                if relation_tokens:
                    parts.append(shared_relations[batch].reshape(1,-1).expand(len(pairs),-1))
                controls, curves = spline_head(torch.cat(parts, dim=-1), nodes[pairs], metric)
            else:
                controls = nodes.new_empty((0,spline_head.controls,3))
                curves = nodes.new_empty((0,len(spline_head.basis),3))
            spline_output = {'spline_controls': controls.cpu(), 'spline_curves': curves.cpu(), 'spline_metric': metric.cpu()}
        graphs.append(
            {
                "nodes": nodes.detach().cpu(),
                "boxes": boxes.detach().cpu(),
                "node_scores": scores.detach().cpu(),
                "edges": pairs.detach().cpu(),
                "edge_scores": relation_scores.detach().cpu(),
                "query_ids": query_ids.detach().cpu(),
                **spline_output,
                **({"node_type_scores": predictions["pred_node_types"][batch, query_ids].softmax(-1).detach().cpu()}
                   if "pred_node_types" in predictions else {}),
            }
        )
    return graphs


__all__ = ["infer_graphs"]
