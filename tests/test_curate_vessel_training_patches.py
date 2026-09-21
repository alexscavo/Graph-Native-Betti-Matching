import csv
from pathlib import Path

import pytest

from scripts.curate_vessel_training_patches import curate
from scripts.generate_synthetic_mri_dataset import main as generate


def test_moves_only_empty_training_samples_and_keeps_full_grid_index(tmp_path: Path):
    root = tmp_path / "vascular_patches"
    root.mkdir()
    fields = ["sample_id", "split", "foreground_voxels", "node_count", "edge_count"]
    rows = [
        ("sample_000000_0000", "train", 0, 0, 0),
        ("sample_000000_0001", "train", 12, 0, 0),
        ("sample_000000_0002", "train", 44, 2, 1),
        ("sample_000001_0000", "test", 0, 0, 0),
    ]
    with (root / "patch_index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    for sample, split, *_ in rows:
        for part, suffix in (("raw", "data.nii.gz"), ("seg", "seg.nii.gz"), ("vtp", "graph.vtp")):
            destination = root / split / part / f"{sample}_{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("fixture")
    before = (root / "patch_index.csv").read_bytes()
    plan = curate(root, apply=False)
    assert plan["active_train_patches"] == 1
    assert plan["excluded_train_patches"] == 2
    assert plan["remaining_files_to_move"] == 6
    assert not (root / "excluded_train").exists()

    result = curate(root, apply=True)
    assert result["validation_test_unchanged"]
    assert (root / "patch_index.csv").read_bytes() == before
    assert (root / "excluded_train" / "raw" / "sample_000000_0000_data.nii.gz").is_file()
    assert (root / "test" / "raw" / "sample_000001_0000_data.nii.gz").is_file()
    assert not (root / "train" / "raw" / "sample_000000_0001_data.nii.gz").exists()
    assert (root / "train" / "raw" / "sample_000000_0002_data.nii.gz").is_file()
    assert curate(root, apply=False)["remaining_files_to_move"] == 0
    assert curate(root, apply=True) == result
    with pytest.raises(ValueError, match="Cannot regenerate into curated training view"):
        generate(["--root", str(root), "--output-dir", str(root)])
