#!/usr/bin/env python3
"""Create the deterministic extraction manifest for new IXI and TopBrain data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATA = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/datasets")


def paired(directory_image: Path, directory_label: Path, image_suffix: str = ""):
    images = {p.name[:-7].removesuffix(image_suffix): p for p in directory_image.glob("*.nii.gz")}
    labels = {p.name[:-7]: p for p in directory_label.glob("*.nii.gz")}
    missing_images, missing_labels = sorted(labels.keys() - images.keys()), sorted(images.keys() - labels.keys())
    if missing_images or missing_labels:
        raise ValueError(f"Unpaired files: missing images={missing_images}, missing labels={missing_labels}")
    return [(key, images[key], labels[key]) for key in sorted(images)]


def discover(root: Path) -> list[dict[str, object]]:
    rows = []
    specifications = [
        ("ixi", "mra", "train", root / "IXI/imagesTr", root / "IXI/labelsTr", ""),
        ("topbrain", "mr", "train", root / "TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_mr", root / "TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_mr", "_0000"),
        ("topbrain", "ct", "train", root / "TopBrain_Data_Release_Batches1n2_081425/imagesTr_topbrain_ct", root / "TopBrain_Data_Release_Batches1n2_081425/labelsTr_topbrain_ct", "_0000"),
        ("topbrain", "ct", "test", root / "TopBrain_Data_Release_Batches1n2_081425/imagesTe_topbrain_ct", root / "TopBrain_Data_Release_Batches1n2_081425/labelsTe_topbrain_ct", "_0000"),
    ]
    for dataset, modality, split, image_dir, label_dir, suffix in specifications:
        for subject, image_path, label_path in paired(image_dir, label_dir, suffix):
            image, label = nib.load(str(image_path)), nib.load(str(label_path))
            if image.shape != label.shape:
                raise ValueError(f"{subject}: image shape {image.shape} != label shape {label.shape}")
            if not np.allclose(image.affine, label.affine, atol=1e-4):
                raise ValueError(f"{subject}: image and label affines disagree")
            spacing = np.sqrt((label.affine[:3, :3] ** 2).sum(axis=0))
            rows.append({
                "index": len(rows), "dataset": dataset, "modality": modality,
                "split": split, "subject": subject,
                "image": str(image_path.resolve()), "label": str(label_path.resolve()),
                "shape_x": image.shape[0], "shape_y": image.shape[1], "shape_z": image.shape[2],
                "spacing_x": f"{spacing[0]:.8g}", "spacing_y": f"{spacing[1]:.8g}",
                "spacing_z": f"{spacing[2]:.8g}",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = discover(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(args.output)
    counts = {}
    for row in rows:
        key = (row["dataset"], row["modality"], row["split"])
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {len(rows)} paired volumes to {args.output}")
    for key, count in counts.items(): print("/".join(key), count)


if __name__ == "__main__":
    main()
