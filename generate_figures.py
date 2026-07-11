"""Regenerate manuscript figures from the archived confidence-interval CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MAIN_ORDER = ["rr1", "proportional_fair", "omn_pf"]
ABLATION_ORDER = [
    "proportional_fair",
    "omn_no_outage",
    "omn_no_edge",
    "omn_no_transition",
    "omn_no_nlos",
    "omn_no_fairness",
    "omn_no_gap",
    "omn_pf",
]

LABELS = {
    "rr1": "RR1",
    "max_sinr": "Max-SINR",
    "proportional_fair": "PF",
    "omn_no_outage": "No outage",
    "omn_no_edge": "No edge",
    "omn_no_transition": "No transition",
    "omn_no_nlos": "No NLOS",
    "omn_no_fairness": "No fairness",
    "omn_no_gap": "No gap",
    "omn_pf": "OMN-PF",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def by_scheduler(rows: list[dict[str, str]], experiment: str) -> dict[str, dict[str, str]]:
    return {row["scheduler"]: row for row in rows if row["experiment"] == experiment}


def values(
    rows: dict[str, dict[str, str]],
    order: list[str],
    metric: str,
) -> tuple[list[float], list[float]]:
    means = [float(rows[name][f"{metric}_mean"]) for name in order]
    ci95 = [float(rows[name][f"{metric}_ci95"]) for name in order]
    return means, ci95


def save_edge_continuity(main: dict[str, dict[str, str]], out: Path) -> None:
    c10, c10_ci = values(main, MAIN_ORDER, "edge_coverage_continuity_10db")
    c20, c20_ci = values(main, MAIN_ORDER, "edge_coverage_continuity_20db")
    x = list(range(len(MAIN_ORDER)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.bar([v - width / 2 for v in x], [100 * v for v in c10], width, yerr=[100 * v for v in c10_ci],
           capsize=4, label="10 dB", color="#3B6FB6")
    ax.bar([v + width / 2 for v in x], [100 * v for v in c20], width, yerr=[100 * v for v in c20_ci],
           capsize=4, label="20 dB", color="#E28E2C")
    ax.set_xticks(x, [LABELS[name] for name in MAIN_ORDER])
    ax.set_ylabel("Edge-user service continuity [%]")
    ax.set_title("Dynamic Edge-User Service Continuity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_rate_fairness(main: dict[str, dict[str, str]], out: Path) -> None:
    rate, rate_ci = values(main, MAIN_ORDER, "avg_rate_bpshz")
    fairness, fairness_ci = values(main, MAIN_ORDER, "fairness_over_time")
    x = list(range(len(MAIN_ORDER)))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    bars = ax1.bar(x, rate, yerr=rate_ci, capsize=4, color="#4C78A8", alpha=0.84, label="Scheduled SE")
    ax1.set_ylabel("Average scheduled SE [bps/Hz]")
    ax1.set_xticks(x, [LABELS[name] for name in MAIN_ORDER])
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    line = ax2.errorbar(x, fairness, yerr=fairness_ci, color="#D64F4F", marker="o",
                        linewidth=2, capsize=4, label="Jain fairness")
    ax2.set_ylabel("Time-averaged Jain fairness")
    ax1.set_title("Scheduled Spectral Efficiency and Fairness")
    ax1.legend([bars, line], ["Scheduled SE", "Jain fairness"], frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_ablation(
    ablation: dict[str, dict[str, str]],
    metric: str,
    threshold: str,
    out: Path,
) -> None:
    mean, ci = values(ablation, ABLATION_ORDER, metric)
    x = list(range(len(ABLATION_ORDER)))
    colors = ["#7A7A7A"] + ["#86A9D6"] * (len(ABLATION_ORDER) - 2) + ["#2E69A3"]

    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=180)
    ax.bar(x, [100 * v for v in mean], yerr=[100 * v for v in ci], capsize=3, color=colors)
    ax.set_xticks(x, [LABELS[name] for name in ABLATION_ORDER], rotation=28, ha="right")
    ax.set_ylabel(f"Edge-user continuity at {threshold} [%]")
    ax.set_title(f"OMN-PF Ablation: Edge Continuity at {threshold}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results_gap_aware/final_overnight_summary_ci.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    rows = load_rows(args.summary)
    main_rows = by_scheduler(rows, "main")
    ablation_rows = by_scheduler(rows, "ablation")

    missing_main = [name for name in MAIN_ORDER if name not in main_rows]
    missing_ablation = [name for name in ABLATION_ORDER if name not in ablation_rows]
    if missing_main or missing_ablation:
        raise ValueError(
            f"Missing summary rows. main={missing_main}, ablation={missing_ablation}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_edge_continuity(main_rows, args.outdir / "fig_dynamic_edge_continuity.png")
    save_rate_fairness(main_rows, args.outdir / "fig_dynamic_rate_fairness.png")
    save_ablation(
        ablation_rows,
        "edge_coverage_continuity_10db",
        "10 dB",
        args.outdir / "fig_ablation_edge_c10.png",
    )
    save_ablation(
        ablation_rows,
        "edge_coverage_continuity_20db",
        "20 dB",
        args.outdir / "fig_ablation_edge_c20.png",
    )

    print("Wrote figures to", args.outdir)


if __name__ == "__main__":
    main()
