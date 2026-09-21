from pathlib import Path
from types import SimpleNamespace

from scripts.extract_real_graph_manifest import graph_destination
from scripts.generate_synthetic_mri_dataset import validate_split_map


def test_direct_graph_output_uses_corresponding_dataset_directory():
    args = SimpleNamespace(dataset_root=Path("/dataset-parent"), output=None)
    row = {"dataset": "topbrain", "modality": "ct", "split": "test", "subject": "case_01"}
    assert graph_destination(row, args) == Path(
        "/dataset-parent/TopBrain_Data_Release_Batches1n2_081425/vascular_graphs/"
        "optimal_radius_0p75x_c095/ct/test/case_01"
    )


def test_existing_patient_splits_keep_identity_without_changing_proportions():
    source = {f"{number:06d}": (None, None, None) for number in range(10)}
    existing = {key: "train" for key in source}
    existing["000009"] = "test"
    validate_split_map(existing, source, require_exact=False)
