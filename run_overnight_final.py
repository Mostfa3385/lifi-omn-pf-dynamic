"""Final overnight runner for the OMN-PF paper.

Default experiment:
    10 seeds x 300 trials x 75 frames
    full dynamic scenario: mobility + time-varying receiver orientation + NLOS-stress
    schedulers: rr1, max-SINR, PF, OMN-PF, OMN-PF-A
    ablation: PF + OMN-PF variants + full OMN-PF (including no-gap ablation)

Usage:
    python run_overnight_final.py --output_dir results_final --workers 4

The script is resumable: if a row for the same experiment/scheduler/seed already
exists in the raw CSV, it will be skipped on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import simulator_omn_pf_final as sim

MAIN_SCHEDULERS = ["rr1", "max_sinr", "proportional_fair", "omn_pf", "omn_pf_a"]
ABLATION_SCHEDULERS = [
    "proportional_fair",
    "omn_no_outage",
    "omn_no_edge",
    "omn_no_transition",
    "omn_no_nlos",
    "omn_no_fairness",
    "omn_no_gap",
    "omn_pf",
]

SUMMARY_METRICS = [
    "edge_coverage_continuity_10db",
    "edge_coverage_continuity_20db",
    "edge_outage_probability_10db",
    "edge_outage_probability_20db",
    "coverage_continuity_10db",
    "coverage_continuity_20db",
    "outage_probability_10db",
    "outage_probability_20db",
    "avg_outage_duration_frames",
    "p95_outage_duration_frames",
    "beam_switching_rate",
    "fairness_over_time",
    "avg_rate_bpshz",
    "service_avg_rate_bpshz_all_users",
    "mean_sinr_db",
    "median_sinr_db",
    "mean_served_fraction",
    "mean_user_throughput_bpshz",
    "p5_user_throughput_bpshz",
    "worst_user_throughput_bpshz",
    "avg_time_between_services_frames",
    "mean_service_gap_frames",
    "p95_service_gap_frames",
    "mean_longest_service_gap_frames",
    "p95_longest_service_gap_frames",
    "starvation_probability_gap10",
    "edge_served_frame_ratio",
    "center_served_frame_ratio",
    "mean_delta_used",
    "std_delta_used",
    "delta_switch_rate",
    "mean_center_pool_size",
    "mean_edge_pool_size",
    "frac_delta_0_5",
    "frac_delta_0_6667",
    "frac_delta_0_8",
    "adaptive_delta_enabled",
    "adaptive_ofdma_enabled",
    "adaptive_power_enabled",
    "adaptive_max_users_per_region",
    "adaptive_min_subcarriers_per_user",
    "multi_user_ofdma_enabled",
    "admission_min_sinr_db",
    "admission_min_sum_rate_retention",
    "max_users_per_region",
    "max_users_per_pool",
    "max_center_users_per_pool",
    "max_edge_users_per_pool",
    "min_subcarriers_per_user",
    "mean_scheduled_users_per_frame",
    "mean_scheduled_users_per_region",
    "mean_scheduled_users_per_active_region",
    "max_scheduled_users_in_region_observed",
    "fraction_active_regions_serving_1_user",
    "fraction_active_regions_serving_2_users",
    "fraction_active_regions_serving_3plus_users",
    "mean_effective_subcarriers_per_scheduled_user",
    "p5_effective_subcarriers_per_scheduled_user",
    "max_relative_region_power_budget_error",
    "time_s",
]


def parse_seed_spec(spec: str) -> list[int]:
    """Parse '1-10' or '1,2,3' into a list of seeds."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def read_done_keys(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add((row["experiment"], row["scheduler"], int(float(row["seed"]))))
            except Exception:
                continue
    return keys


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    fieldnames = list(row.keys())
    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing_reader = csv.reader(f)
            existing_header = next(existing_reader)
        # Preserve existing field order; append any new fields at the end.
        fieldnames = existing_header + [k for k in row.keys() if k not in existing_header]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    row = sim.simulate(
        trials=task["trials"],
        frames=task["frames"],
        users_per_macro=task["users_per_macro"],
        scheduler=task["scheduler"],
        include_mobility=True,
        include_orientation=True,
        include_nlos=True,
        seed=task["seed"],
        nlos_profile=task["nlos_profile"],
    )
    row = dict(row)
    row["experiment"] = task["experiment"]
    row["case"] = "combined_mobility_orientation_nlos"
    row["seed"] = task["seed"]
    row["wall_time_s"] = time.time() - start
    return row


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def summarize(raw_csv: Path, summary_csv: Path) -> list[dict[str, Any]]:
    rows = load_rows(raw_csv)
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("experiment", ""), row.get("scheduler", ""))
        groups.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for (experiment, scheduler), group_rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "experiment": experiment,
            "scheduler": scheduler,
            "n": len(group_rows),
        }
        for metric in SUMMARY_METRICS:
            values = [safe_float(r.get(metric)) for r in group_rows]
            values = [v for v in values if v is not None and math.isfinite(v)]
            if not values:
                continue
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            ci95 = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_ci95"] = ci95
        out_rows.append(summary)

    if out_rows:
        fields: list[str] = []
        for row in out_rows:
            for k in row.keys():
                if k not in fields:
                    fields.append(k)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)
    return out_rows


def write_quick_report(summary_rows: list[dict[str, Any]], report_path: Path) -> None:
    by_key = {(r["experiment"], r["scheduler"]): r for r in summary_rows}
    lines = [
        "OMN-PF Final Overnight Run Report",
        "=================================",
        "",
        "This report is auto-generated from the final overnight raw CSV.",
        "The main paper claim should compare OMN-PF primarily against conventional PF,",
        "while reporting max-SINR as a high-throughput/high-continuity reference.",
        "",
    ]
    for experiment in ["main", "ablation"]:
        lines.append(f"[{experiment.upper()}]")
        scheds = sorted([k[1] for k in by_key if k[0] == experiment])
        for scheduler in scheds:
            r = by_key[(experiment, scheduler)]
            n = r.get("n", 0)
            edge10 = r.get("edge_coverage_continuity_10db_mean", "")
            edge10ci = r.get("edge_coverage_continuity_10db_ci95", "")
            edge20 = r.get("edge_coverage_continuity_20db_mean", "")
            edge20ci = r.get("edge_coverage_continuity_20db_ci95", "")
            se = r.get("avg_rate_bpshz_mean", "")
            fair = r.get("fairness_over_time_mean", "")
            lines.append(
                f"- {scheduler:20s} n={n} "
                f"edgeC10={edge10:.4f}±{edge10ci:.4f} " if isinstance(edge10, float) else f"- {scheduler:20s} n={n} "
            )
            if isinstance(edge10, float):
                lines[-1] += f"edgeC20={edge20:.4f}±{edge20ci:.4f} SE={se:.2f} fairness={fair:.4f}"
        lines.append("")

    # Add PF vs OMN-PF deltas when available.
    pf = by_key.get(("main", "proportional_fair"))
    omn = by_key.get(("main", "omn_pf"))
    if pf and omn:
        lines.append("PF vs OMN-PF headline deltas")
        lines.append("---------------------------")
        for metric in ["edge_coverage_continuity_10db", "edge_coverage_continuity_20db", "fairness_over_time", "avg_rate_bpshz"]:
            a = pf.get(f"{metric}_mean")
            b = omn.get(f"{metric}_mean")
            if isinstance(a, float) and isinstance(b, float):
                pct = 100.0 * (b - a) / abs(a) if abs(a) > 1e-12 else float("nan")
                lines.append(f"- {metric}: PF={a:.6g}, OMN-PF={b:.6g}, delta={pct:+.2f}%")
        lines.append("")

    omn_a = by_key.get(("main", "omn_pf_a"))
    if omn and omn_a:
        lines.append("OMN-PF vs OMN-PF-A multi-user adaptive OFDMA/power deltas")
        lines.append("-------------------------------------------")
        for metric in ["edge_coverage_continuity_10db", "edge_coverage_continuity_20db", "fairness_over_time", "avg_rate_bpshz", "mean_delta_used", "delta_switch_rate", "adaptive_ofdma_enabled", "adaptive_power_enabled"]:
            a = omn.get(f"{metric}_mean")
            b = omn_a.get(f"{metric}_mean")
            if isinstance(a, float) and isinstance(b, float):
                pct = 100.0 * (b - a) / abs(a) if abs(a) > 1e-12 else float("nan")
                lines.append(f"- {metric}: OMN-PF={a:.6g}, OMN-PF-A={b:.6g}, delta={pct:+.2f}%")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_tasks(args: argparse.Namespace, done_keys: set[tuple[str, str, int]]) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    tasks: list[dict[str, Any]] = []
    experiments: list[tuple[str, list[str]]] = []
    if args.mode in ("main", "both"):
        experiments.append(("main", MAIN_SCHEDULERS))
    if args.mode in ("ablation", "both"):
        experiments.append(("ablation", ABLATION_SCHEDULERS))
    for experiment, schedulers in experiments:
        for seed in seeds:
            for scheduler in schedulers:
                key = (experiment, scheduler, seed)
                if key in done_keys:
                    continue
                tasks.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "scheduler": scheduler,
                        "trials": args.trials,
                        "frames": args.frames,
                        "users_per_macro": args.users_per_macro,
                        "nlos_profile": args.nlos_profile,
                    }
                )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("results_gap_aware"))
    parser.add_argument("--seeds", type=str, default="1-10")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--users_per_macro", type=int, default=42)
    parser.add_argument("--nlos_profile", choices=["baseline", "stress"], default="stress")
    parser.add_argument("--mode", choices=["main", "ablation", "both"], default="both")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.output_dir / "final_overnight_raw.csv"
    summary_csv = args.output_dir / "final_overnight_summary_ci.csv"
    report_txt = args.output_dir / "final_overnight_report.txt"
    metadata_json = args.output_dir / "final_overnight_metadata.json"

    done_keys = read_done_keys(raw_csv)
    tasks = build_tasks(args, done_keys)

    metadata = {
        "seeds": parse_seed_spec(args.seeds),
        "trials": args.trials,
        "frames": args.frames,
        "users_per_macro": args.users_per_macro,
        "nlos_profile": args.nlos_profile,
        "mode": args.mode,
        "workers": args.workers,
        "main_schedulers": MAIN_SCHEDULERS,
        "ablation_schedulers": ABLATION_SCHEDULERS,
        "total_pending_tasks": len(tasks),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Final overnight OMN-PF run")
    print("===========================")
    print(json.dumps(metadata, indent=2))
    print()
    print("Resumable raw output:", raw_csv)
    print("Pending tasks:", len(tasks))
    print()

    start = time.time()
    completed = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_task, task) for task in tasks]
            for fut in as_completed(futures):
                row = fut.result()
                append_row(raw_csv, row)
                completed += 1
                elapsed = time.time() - start
                avg = elapsed / completed
                remaining = avg * (len(tasks) - completed)
                print(
                    f"[{completed:03d}/{len(tasks):03d}] "
                    f"{row['experiment']:8s} {row['scheduler']:20s} seed={int(row['seed']):02d} "
                    f"edgeC10={row['edge_coverage_continuity_10db']:.4f} "
                    f"edgeC20={row['edge_coverage_continuity_20db']:.4f} "
                    f"SE={row['avg_rate_bpshz']:.2f} "
                    f"J={row['fairness_over_time']:.4f} "
                    f"task_time={row['wall_time_s']:.1f}s "
                    f"ETA={remaining/3600:.2f}h"
                )
    else:
        print("No pending tasks: raw CSV already contains all requested rows.")

    summary_rows = summarize(raw_csv, summary_csv)
    write_quick_report(summary_rows, report_txt)
    print()
    print("Done.")
    print("Raw CSV:", raw_csv)
    print("CI summary:", summary_csv)
    print("Report:", report_txt)


if __name__ == "__main__":
    main()
