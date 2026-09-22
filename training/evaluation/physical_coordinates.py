"""Recover physical coordinates for normalized vessel patches from source provenance.

Generated patch NIfTI files intentionally have identity affines. Do not use
their header spacing to compute millimetre-scale evaluation metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


class PatchPhysicalCoordinates:
    def __init__(self):
        self._datasets = {}
        self._affines = {}

    def _metadata(self, patch_root: Path):
        if patch_root not in self._datasets:
            with (patch_root / "generation_config.json").open() as stream:
                generation = json.load(stream)
            with (patch_root / "source_manifest.json").open() as stream:
                manifest = json.load(stream)
            with (patch_root / "patch_index.csv").open(newline="") as stream:
                index = {row["sample_id"]: row for row in csv.DictReader(stream)}
            subjects = {str(item["patient_id"]): Path(item["image"])
                        for item in manifest["subjects"]}
            self._datasets[patch_root] = generation, index, subjects
        return self._datasets[patch_root]

    def transform(self, record, *, coordinate_space: str = "normalized") -> np.ndarray:
        """Return a 4x4 map from graph D/H/W coordinates to world millimetres."""

        if coordinate_space != "normalized":
            raise ValueError("Physical Branch F1 requires normalized patch graph coordinates")
        path = Path(record.graph)
        if path.parent.name != "vtp" or path.parent.parent.name not in {"train", "val", "test"}:
            raise ValueError(f"No generated vessel patch provenance for {path}")
        patch_root = path.parents[2]
        generation, index, subjects = self._metadata(patch_root)
        row = index[record.sample_id]
        source = subjects[row["patient_id"]]
        if source not in self._affines:
            import nibabel as nib
            self._affines[source] = np.asarray(nib.load(str(source)).affine, dtype=np.float64)
        patch_size = np.asarray(generation["patch_size"], dtype=np.float64)
        pad = np.asarray(generation["pad"], dtype=np.float64)
        start = np.asarray([int(row[f"start_{axis}"]) for axis in "dhw"], dtype=np.float64)
        if patch_size.shape != (3,) or pad.shape != (3,) or np.any(patch_size <= 0):
            raise ValueError(f"Invalid patch geometry in {patch_root}")
        patch_to_source = np.eye(4, dtype=np.float64)
        patch_to_source[:3, :3] = np.diag(patch_size)
        patch_to_source[:3, 3] = start - pad
        return self._affines[source] @ patch_to_source


def to_world_mm(nodes, transform):
    points = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    affine = np.asarray(transform, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("world transform must be a finite 4x4 matrix")
    return points @ affine[:3, :3].T + affine[:3, 3]


__all__ = ["PatchPhysicalCoordinates", "to_world_mm"]
