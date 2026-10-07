"""Decoded branch coverage and physical geometry against uncertain guides."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment

from metrics.representation_geometry import point_to_polyline_distance, sample_polyline_by_arclength, tortuosity


def evaluate_spline_graph(prediction, cache_path, segmentation, threshold_mm=1.0):
    with np.load(cache_path) as target:
        metric = target['metric'].astype(float)
        gt = target['nodes'] @ metric.T
        pred = np.asarray(prediction['nodes']) @ metric.T
        distance = np.linalg.norm(gt[:, None]-pred[None], axis=-1)
        a, b = linear_sum_assignment(distance)
        mapping = {int(j): int(i) for i, j in zip(a, b) if distance[i, j] <= threshold_mm}
        lookup = {tuple(sorted(edge)): i for i, edge in enumerate(target['edges'])}
        recovered = set(); errors = []; mask_support = []
        mask = np.asarray(segmentation).squeeze()
        if 'spline_curves' in prediction:
            controls_all = prediction['spline_controls']
            sampled_all = prediction['spline_curves']
        else:
            # Exact analytic geometry of the decoded straight-edge model.
            # Four collinear controls express a chord as a cubic Bezier curve;
            # these are derived endpoints, not predictions from a spline head.
            nodes = np.asarray(prediction['nodes'])
            edges = np.asarray(prediction['edges'], dtype=int).reshape(-1, 2)
            endpoints = nodes[edges]
            controls_all = endpoints[:, :1] + np.linspace(0, 1, 4)[None, :, None] * (endpoints[:, 1:] - endpoints[:, :1])
            sampled_all = endpoints[:, :1] + np.linspace(0, 1, 64)[None, :, None] * (endpoints[:, 1:] - endpoints[:, :1])
        for edge, controls, sampled in zip(prediction['edges'], controls_all, sampled_all):
            u, v = map(int, edge)
            if u not in mapping or v not in mapping:
                continue
            identity = tuple(sorted((mapping[u], mapping[v])))
            index = lookup.get(identity)
            if index is None or index in recovered:
                continue
            recovered.add(index)
            sampled = np.asarray(sampled)
            if (mapping[u], mapping[v]) != tuple(target['edges'][index]):
                sampled = sampled[::-1]
            curve_error = np.linalg.norm((sampled-target['curves'][index]) @ metric.T, axis=-1).mean()
            controls = np.asarray(controls)
            k = len(controls)
            knots = np.r_[np.zeros(4), np.arange(1, k-3)/(k-3), np.ones(4)]
            normalized = BSpline(knots, controls, 3)(np.linspace(0, 1, 257))
            curve = normalized @ metric.T
            offsets = target['path_offsets']
            guide = target['full_paths'][offsets[index]:offsets[index+1]] @ metric.T
            forward = point_to_polyline_distance(sample_polyline_by_arclength(curve, .25), guide)
            backward = point_to_polyline_distance(sample_polyline_by_arclength(guide, .25), curve)
            errors.append((curve_error, (forward.mean()+backward.mean())/2, abs(tortuosity(curve)-tortuosity(guide))))
            mask_support.append(float((map_coordinates(mask, (normalized*np.asarray(mask.shape)).T,
                                                       order=1, mode='constant', cval=-.5) > 0).mean()))
        tp, fp, fn = len(recovered), len(prediction['edges'])-len(recovered), len(target['edges'])-len(recovered)
        result = {'spline_branch_tp': tp, 'spline_branch_fp': fp, 'spline_branch_fn': fn,
                  'spline_matched_branches': tp, 'spline_target_branches': len(target['edges'])}
        values = np.asarray(errors).reshape(-1, 3)
        for column, key in enumerate(('spline_curve_mm', 'spline_guide_distance_mm', 'spline_tortuosity_error')):
            result[key] = float(values[:, column].mean()) if tp else float('nan')
            result[key+'_sum'] = float(values[:, column].sum())
        result['spline_mask_support_fraction'] = float(np.mean(mask_support)) if tp else float('nan')
        result['spline_mask_support_fraction_sum'] = float(sum(mask_support))
        return result


def aggregate_spline_metrics(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ('spline_branch_tp', 'spline_branch_fp', 'spline_branch_fn'))
    result = {'spline_branch_tp_total': tp, 'spline_branch_fp_total': fp, 'spline_branch_fn_total': fn,
              'spline_branch_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
              'spline_branch_coverage': tp/(tp+fn) if tp+fn else None}
    for key in ('spline_curve_mm', 'spline_guide_distance_mm', 'spline_tortuosity_error', 'spline_mask_support_fraction'):
        result[key] = sum(row[key+'_sum'] for row in rows)/tp if tp else None
    return result
