from collections import Counter

from scripts.prepare_real_graph_patch_sources import assign_splits


def test_assign_splits_is_exact_stratified_and_preserves_official_test():
    records = []
    index = 0
    for dataset, modality, count, source_split in (
        ("ixi", "mra", 170, "train"),
        ("topbrain", "mr", 25, "train"),
        ("topbrain", "ct", 18, "train"),
        ("topbrain", "ct", 7, "test"),
    ):
        for subject_index in range(count):
            records.append(
                {
                    "patient_id": f"{index:06d}",
                    "dataset": dataset,
                    "modality": modality,
                    "source_split": source_split,
                    "subject": f"{dataset}_{modality}_{source_split}_{subject_index}",
                }
            )
            index += 1

    first = assign_splits(records, seed=42)
    second = assign_splits(records, seed=42)

    assert first == second
    assert Counter(first.values()) == {"train": 154, "val": 33, "test": 33}
    assert all(
        first[record["patient_id"]] == "test"
        for record in records
        if record["source_split"] == "test"
    )
    for dataset, modality in {("ixi", "mra"), ("topbrain", "mr"), ("topbrain", "ct")}:
        represented = {
            first[record["patient_id"]]
            for record in records
            if (record["dataset"], record["modality"]) == (dataset, modality)
        }
        assert represented == {"train", "val", "test"}
