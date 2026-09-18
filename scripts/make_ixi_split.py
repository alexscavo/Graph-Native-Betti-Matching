#!/usr/bin/env python3
"""Create a subject-level, site-stratified train/val/test split for IXI.

IXI was acquired at three sites (Guys, HH, IOP) on different scanners, so an
unstratified draw can concentrate one site in the test split and turn a domain
shift into apparent generalisation failure. Subjects are therefore split within
each site and pooled.

Split sizes match what the patch generator validates: floor(70%) train,
floor(15%) val, remainder test.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="dir holding subject_map.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    with (args.sources / "subject_map.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    total = len(rows)
    train_total = int(math.floor(total * 0.70))
    val_total = int(math.floor(total * 0.15))

    by_site: dict[str, list[str]] = {}
    for row in rows:
        by_site.setdefault(row["site"], []).append(row["numeric_id"])

    # Round-robin interleave of per-site shuffled orders, then slice by the exact
    # required sizes. Allocating quotas per site independently cannot hit the exact
    # totals (it produced train=5/val=0/test=3 where 5/1/2 was required), whereas
    # interleaving keeps sites proportionally mixed AND makes the totals exact.
    rng = np.random.default_rng(args.seed)
    shuffled = {
        site: [members[i] for i in rng.permutation(len(members))]
        for site, members in sorted(by_site.items())
    }
    order: list[str] = []
    position = 0
    while len(order) < total:
        for site in sorted(shuffled):
            if position < len(shuffled[site]):
                order.append(shuffled[site][position])
        position += 1

    assignment = {pid: "train" for pid in order[:train_total]}
    assignment.update({pid: "val" for pid in order[train_total : train_total + val_total]})
    assignment.update({pid: "test" for pid in order[train_total + val_total :]})

    counts = {split: sum(1 for v in assignment.values() if v == split) for split in ("train", "val", "test")}
    expected = {"train": train_total, "val": val_total, "test": total - train_total - val_total}
    if counts != expected:
        raise SystemExit(f"Split sizes {counts} do not match required {expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["patient_id,split"]
    lines.extend(
        f"{pid},{assignment[pid]}" for pid in sorted(assignment, key=lambda v: int(v))
    )
    args.output.write_text("\n".join(lines) + "\n")

    print(f"{total} subjects -> {counts}")
    for site, members in sorted(by_site.items()):
        per = {s: sum(1 for m in members if assignment[m] == s) for s in ("train", "val", "test")}
        print(f"  {site:8} n={len(members):4d}  {per}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
