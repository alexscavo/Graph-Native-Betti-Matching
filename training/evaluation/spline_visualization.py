"""Fixed validation MRI/mask projections with decoded straight or spline edges."""
import json
from pathlib import Path

import numpy as np
import torch

from data.loaders import build_evaluation_loader
from .inference import infer_graphs


def draw_projections(image, mask, nodes, guides, prediction, path, title, prediction_label="Decoded prediction"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shape = np.asarray(image.shape)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for column, omitted in enumerate((0, 1, 2)):
        vertical, horizontal = [a for a in range(3) if a != omitted]
        for row in range(2):
            ax = axes[row, column]
            ax.imshow(image.max(axis=omitted), cmap='gray', origin='lower')
            overlay = np.ma.masked_where(~(mask > 0).any(axis=omitted), np.ones(np.delete(shape, omitted)))
            ax.imshow(overlay, cmap='Greens', alpha=.16, vmin=0, vmax=1, origin='lower')
            if row == 0:
                curves = guides
            elif 'spline_curves' in prediction:
                curves = np.asarray(prediction['spline_curves'])
            else:
                edges = np.asarray(prediction['edges'], dtype=int).reshape(-1, 2)
                curves = np.asarray(prediction['nodes'])[edges]
            anchors = nodes if row == 0 else np.asarray(prediction['nodes'])
            color = '#ffb943' if row == 0 else '#00d3c2'
            for curve in curves:
                coords = curve*shape
                ax.plot(coords[:, horizontal], coords[:, vertical], color=color, lw=1.2)
            anchors = np.asarray(anchors).reshape(-1, 3)*shape
            ax.scatter(anchors[:, horizontal], anchors[:, vertical], color=color, s=15)
            ax.set_title(('Original guide' if row == 0 else prediction_label) + ' · projection along ' + 'DHW'[omitted])
            ax.set_xlabel('DHW'[horizontal]+' (voxels)'); ax.set_ylabel('DHW'[vertical]+' (voxels)')
            ax.set_xlim(-.5, shape[horizontal]-.5); ax.set_ylim(-.5, shape[vertical]-.5)
    fig.suptitle(title+' · mask is reference; projected crossings need not be junctions')
    fig.tight_layout(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


@torch.no_grad()
def save_validation_projections(model, config, device, output, epoch, count):
    loader = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val')
    # Stable examples from different validation subjects, selected without model scores.
    by_subject = {}
    for record in loader.dataset.records:
        subject = record.sample_id.split('_')[1]
        if subject not in by_subject:
            by_subject[subject] = []
        by_subject[subject].append(record.sample_id)
    sample_ids = [ids[len(ids)//2] for _, ids in sorted(by_subject.items())][:count]
    loader = build_evaluation_loader(config, dataset_name='synthetic_mri', split='val', sample_ids=sample_ids)
    root = Path(output)/'validation-projections'
    root.mkdir(parents=True, exist_ok=True)
    (root/'samples.json').write_text(json.dumps(sample_ids, indent=2)+'\n')
    model.eval(); paths = []; index = 0
    decoder = config['model']['decoder']
    for batch in loader:
        tokens, predictions, _ = model(batch[0].to(device))
        decoded = infer_graphs(tokens, predictions, model.relation_embed,
            object_queries=decoder['object_queries'], relation_tokens=decoder['relation_tokens'],
            node_threshold=config['evaluation']['node_threshold'], edge_threshold=config['evaluation']['edge_threshold'],
            spline_head=model.spline_head, spline_metrics=([g['metric'] for g in batch[4]] if model.spline_head is not None else None))
        for j, graph in enumerate(decoded):
            sample = sample_ids[index]; index += 1
            path = root/f'epoch-{epoch:03d}'/f'{sample}.png'
            if batch[4] is not None and batch[4][j] is not None:
                guides = batch[4][j]['guides'].numpy()
                reference_nodes = batch[2][j].numpy()
            else:
                reference_root = Path(config['evaluation']['visualization_reference_root'])
                with np.load(reference_root/'val'/'spline'/f'{sample}.npz') as reference:
                    guides = reference['guides']; reference_nodes = reference['nodes']
            draw_projections(batch[0][j].cpu().squeeze().numpy(), batch[1][j].cpu().squeeze().numpy(),
                             reference_nodes, guides, graph, path,
                             f'{sample} · epoch {epoch}')
            paths.append(path)
    return paths
