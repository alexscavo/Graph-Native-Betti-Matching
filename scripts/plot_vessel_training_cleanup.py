#!/usr/bin/env python3
"""Plot audited vessel patch training counts from the dataset-local views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ixi", type=Path, required=True)
    parser.add_argument("--topbrain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("IXI", "TopBrain")
    records = [json.loads((root / "training_view.json").read_text())
               for root in (args.ixi, args.topbrain)]
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    positions = range(2)
    active = [row["active_train_patches"] for row in records]
    excluded = [row["excluded_train_patches"] for row in records]
    axis.bar(positions, active, color="#3b8f71", label="Usable train: nonempty graph + vessel")
    axis.bar(positions, excluded, bottom=active, color="#b9c2c7", label="Archived empty/graph-free train")
    axis.axhspan(5000, 10000, color="#edae49", alpha=.18, label="5–10k training target")
    for i, (a, e) in enumerate(zip(active, excluded)):
        axis.text(i, a / 2, f"{a:,} usable", ha="center", va="center", color="white", fontsize=11)
        axis.text(i, a + e + 300, f"{a+e:,} original", ha="center")
    axis.set(xticks=list(positions), xticklabels=names, ylabel="Training patches",
             ylim=(0, max(a + e for a, e in zip(active, excluded)) * 1.12),
             title="Full-volume-first 64³ crops, max stride 40 (≥37.5% overlap per axis)")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=3, fontsize=8)
    axis.grid(axis="y", alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
