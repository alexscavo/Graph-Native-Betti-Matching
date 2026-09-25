"""Plot measured smoke-run cost and the configured mixed-sampler composition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_ROOT = Path("/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm")
HERE = Path(__file__).resolve().parent
RUNS = {
    8: "pretrain_plants_ixi_domain_types_smoke_20260924",
    32: "pretrain_plants_ixi_domain_types_smoke_b32_20260924",
}


def measured_runs():
    results = {}
    for batch_size, run_name in RUNS.items():
        performance = RUN_ROOT / run_name / "performance.jsonl"
        if performance.exists():
            with performance.open(encoding="utf-8") as handle:
                results[batch_size] = json.loads(next(handle))
    return results


def main():
    measured = measured_runs()
    source_count, target_count = 25_900, 8_000
    epoch_draws = source_count + target_count
    summary = {
        "dataset": "Plants source + IXI vessel target",
        "unique_train_samples": {"plants": source_count, "ixi": target_count},
        "weighted_epoch_draws_expected": {
            "plants": epoch_draws / 2,
            "ixi": epoch_draws / 2,
        },
        "smoke_runs": measured,
        "note": (
            "Draw counts are sampler expectations. Smoke throughput is a short-run "
            "measurement, not a reliable full-epoch speed estimate."
        ),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    x = [0, 1]
    width = 0.32
    axes[0].bar([v - width / 2 for v in x], [source_count, target_count],
                width, label="Unique patches", color="#78909c")
    axes[0].bar([v + width / 2 for v in x], [epoch_draws / 2] * 2,
                width, label="Expected epoch draws", color="#1565c0")
    axes[0].set_xticks(x, ["Plants", "IXI"])
    axes[0].set_ylabel("Patches / draws")
    axes[0].set_title("Weighted target upsampling")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    batches = sorted(measured)
    if batches:
        labels = [f"Batch {size}" for size in batches]
        throughput = [measured[size]["samples_per_second"] for size in batches]
        memory = [measured[size]["peak_allocated_gib"] for size in batches]
        axes[1].bar(labels, throughput, color="#00897b")
        axes[1].set_ylabel("Measured samples / second / GPU")
        axes[1].set_title("One-H100 short smoke runs")
        for index, (speed, gib) in enumerate(zip(throughput, memory)):
            axes[1].text(index, speed, f"{speed:.1f}/s\n{gib:.1f} GiB",
                         ha="center", va="bottom", fontsize=8)
        axes[1].set_ylim(0, max(throughput) * 1.25)
        axes[1].grid(axis="y", alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, "Smoke data pending", ha="center", va="center")
        axes[1].set_axis_off()
    fig.suptitle("Plants + IXI domain-adaptation pretraining: input and cost checks")
    fig.savefig(HERE / "smoke_evidence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
