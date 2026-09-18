#!/usr/bin/env python3
"""Build the IXI source tree consumed by the patch generator.

Emits the layout ``audit_synthetic_mri_grid.discover_sources`` expects::

    <output>/raw/<numeric_id>.nii.gz
    <output>/seg/<numeric_id>.nii.gz
    <output>/graphs/<numeric_id>/{nodes.csv, edges.csv, graph.vvg}
    <output>/subject_map.csv        numeric id <-> IXI subject, site, spacing, provenance

Subject ids are mapped to integers because the patch generator requires numeric
patient ids (``patient_token``). The mapping is deterministic (sorted subject
order) and recorded, so provenance back to the IXI subject is never lost again.

Segmentations whose header was stripped to unit spacing have their geometry
restored from the matching brain mask, which retained it; the array shapes are
verified to agree before doing so.

The work is shardable for a Slurm array: ``--shard i --num-shards n`` processes
subjects whose index is congruent to ``i`` mod ``n``. Per-subject completion
markers make re-runs resumable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import nibabel as nib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ixi_vessel_graph import build_vessel_graph, write_source_graph  # noqa: E402


DEFAULT_ROOT = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI_dataset")

# These alternatives were selected from image-intensity evidence documented in
# docs/IXI_DATASET.md. Keep the exception explicit and preserve it in output
# provenance rather than renaming or overwriting either source annotation.
PREFERRED_SEGMENTATIONS = {
    "IXI638-HH-2786": "IXI638-HH-2786-MRA.nii(1).gz",
    "IXI661-HH-2788": "IXI661-HH-2788-MRA.nii(1).gz",
}


def complete_subjects(root: Path) -> list[str]:
    """Subjects having a volume, a segmentation and a brain mask."""

    segmentations = {
        path.name[: -len("-MRA.nii.gz")]
        for path in (root / "segmentations").glob("*-MRA.nii.gz")
    }
    volumes = {
        path.name[: -len("-MRA.nii.gz")] for path in (root / "volumes").glob("*-MRA.nii.gz")
    }
    masks = {
        path.name[: -len("-MRA_mask.nii.gz")]
        for path in (root / "Brain_masks").glob("*-MRA_mask.nii.gz")
    }
    return sorted(segmentations & volumes & masks)


def site_of(subject: str) -> str:
    parts = subject.split("-")
    return parts[1] if len(parts) > 1 else "unknown"


def segmentation_path(root: Path, subject: str) -> Path:
    """Return the documented preferred annotation path for a subject."""

    filename = PREFERRED_SEGMENTATIONS.get(subject, f"{subject}-MRA.nii.gz")
    path = root / "segmentations" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Preferred segmentation is missing: {path}")
    return path


def load_geometry(root: Path, subject: str):
    """Return image, segmentation, affine and spacing, repairing a stripped header."""

    image = nib.load(str(root / "volumes" / f"{subject}-MRA.nii.gz"))
    selected_segmentation = segmentation_path(root, subject)
    segmentation = nib.load(str(selected_segmentation))
    brain = nib.load(str(root / "Brain_masks" / f"{subject}-MRA_mask.nii.gz"))
    if image.shape != segmentation.shape:
        raise ValueError(
            f"{subject}: image {image.shape} and segmentation {segmentation.shape} disagree"
        )

    affine = segmentation.affine
    repaired = False
    zooms = tuple(round(float(z), 4) for z in segmentation.header.get_zooms())
    if zooms == (1.0, 1.0, 1.0) and brain.shape == segmentation.shape:
        affine = brain.affine
        repaired = True
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return image, segmentation, brain, affine, spacing, repaired, selected_segmentation


def process_subject(subject: str, numeric: int, args: argparse.Namespace) -> dict:
    output = args.output_dir
    marker = output / ".complete" / f"{numeric:06d}.json"
    if marker.is_file() and not args.force:
        return {**json.loads(marker.read_text()), "skipped": True}

    image, segmentation, brain, affine, spacing, repaired, selected_segmentation = (
        load_geometry(args.root, subject)
    )
    raw = np.asanyarray(image.dataobj)
    seg = np.asanyarray(segmentation.dataobj) > 0

    if args.brain_mask != "none":
        keep = np.asanyarray(brain.dataobj) > 0
        if keep.shape != seg.shape:
            raise ValueError(f"{subject}: brain mask shape {keep.shape} != {seg.shape}")
        if args.brain_mask == "zero":
            # Applied to BOTH so the target never asserts vessel where the image is
            # blank; masking only the image would leave unlearnable ground truth.
            raw = np.where(keep, raw, 0)
            seg = seg & keep

    graph = build_vessel_graph(
        seg,
        spacing=spacing,
        spur_length=args.spur_length,
        rdp_tolerance_mm=args.rdp_voxels * float(spacing.min()),
        intermediate_nodes=not args.junctions_only,
        radius_fraction=args.radius_fraction,
        max_junction_extent_mm=args.max_junction_extent_mm,
        smooth_iterations=args.smooth_iterations,
        smooth_alpha=args.smooth_alpha,
    )

    for folder in ("raw", "seg", "graphs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(raw, dtype=np.float32), affine),
        str(output / "raw" / f"{numeric}.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(np.asarray(seg, dtype=np.uint8), affine),
        str(output / "seg" / f"{numeric}.nii.gz"),
    )
    write_source_graph(output / "graphs" / str(numeric), graph, affine, seg.shape)

    beta_0, beta_1 = graph.betti()
    degrees = graph.node_degrees
    payload = {
        "subject": subject,
        "numeric_id": numeric,
        "site": site_of(subject),
        "shape": list(int(v) for v in seg.shape),
        "spacing": [float(v) for v in spacing],
        "header_repaired": bool(repaired),
        "source_segmentation": str(selected_segmentation.relative_to(args.root)),
        "segmentation_variant": (
            "preferred_duplicate"
            if subject in PREFERRED_SEGMENTATIONS
            else "canonical"
        ),
        "vessel_voxels": int(seg.sum()),
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "terminations": int((degrees == 1).sum()),
        "degree_2": int((degrees == 2).sum()),
        "junctions": int((degrees >= 3).sum()),
        "max_degree": int(degrees.max()) if graph.node_count else 0,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "skeleton_voxels": graph.skeleton_voxels,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker)
    return {**payload, "skipped": False}


def write_subject_map(output: Path, root: Path, subjects: list[str]) -> None:
    rows = [
        {
            "numeric_id": index,
            "subject": subject,
            "site": site_of(subject),
            "source_volume": f"volumes/{subject}-MRA.nii.gz",
            "source_segmentation": str(segmentation_path(root, subject).relative_to(root)),
            "segmentation_variant": (
                "preferred_duplicate"
                if subject in PREFERRED_SEGMENTATIONS
                else "canonical"
            ),
            "source_brain_mask": f"Brain_masks/{subject}-MRA_mask.nii.gz",
        }
        for index, subject in enumerate(subjects)
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "subject_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--brain-mask", choices=("none", "zero"), default="none")
    parser.add_argument("--junctions-only", action="store_true")
    parser.add_argument("--rdp-voxels", type=float, default=2.0)
    parser.add_argument("--radius-fraction", type=float, default=0.0)
    parser.add_argument("--spur-length", type=int, default=4)
    parser.add_argument("--max-junction-extent-mm", type=float, default=1.5)
    parser.add_argument("--smooth-iterations", type=int, default=5)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shards <= 0 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"Invalid shard {args.shard}/{args.num_shards}")

    subjects = complete_subjects(args.root)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        raise SystemExit(f"No complete subjects below {args.root}")
    write_subject_map(args.output_dir, args.root, subjects)

    assigned = [
        (index, subject)
        for index, subject in enumerate(subjects)
        if index % args.num_shards == args.shard
    ]
    print(
        f"{len(subjects)} complete subjects; shard {args.shard}/{args.num_shards} "
        f"handles {len(assigned)}",
        flush=True,
    )
    if args.plan_only:
        for index, subject in assigned[:10]:
            print(f"  {index:4d} {subject}")
        return 0

    failures = []

    def report(position: int, numeric: int, subject: str, result: dict) -> None:
        state = "skipped" if result["skipped"] else "done"
        print(
            f"[{position}/{len(assigned)}] {subject} (id {numeric}) {state} "
            f"nodes={result['nodes']} edges={result['edges']} "
            f"b0={result['beta_0']} b1={result['beta_1']}",
            flush=True,
        )

    if args.workers <= 1:
        for position, (numeric, subject) in enumerate(assigned, start=1):
            try:
                report(position, numeric, subject, process_subject(subject, numeric, args))
            except Exception:
                failures.append(subject)
                print(f"[{position}/{len(assigned)}] {subject} FAILED", flush=True)
                traceback.print_exc()
    else:
        # Each subject is independent and writes its own files and marker, so the
        # work parallelises cleanly across processes within a shard.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_subject, subject, numeric, args): (numeric, subject)
                for numeric, subject in assigned
            }
            for position, future in enumerate(as_completed(futures), start=1):
                numeric, subject = futures[future]
                try:
                    report(position, numeric, subject, future.result())
                except Exception:
                    failures.append(subject)
                    print(f"[{position}/{len(assigned)}] {subject} FAILED", flush=True)
                    traceback.print_exc()
    if failures:
        print(f"FAILED subjects ({len(failures)}): {', '.join(failures)}", flush=True)
        return 1
    print("shard complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
