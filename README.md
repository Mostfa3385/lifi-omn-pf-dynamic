# Service-Gap-Aware OMN-PF for Dynamic LiFi Attocells

This repository contains the dynamic reliability-aware extension of the
ADT-SDMA, OFDMA, and strict-FFR LiFi attocell benchmark. It adds:

- frame-based random-waypoint mobility;
- time-correlated receiver orientation;
- first-order NLOS-stress propagation;
- serving-region and beam-transition tracking;
- outage, edge, NLOS-fragility, and service-gap scheduling states;
- a service-gap-aware OMN-PF scheduler;
- multi-seed confidence-interval reporting and controlled ablations.

This repository extends the earlier static benchmark. The links below refer
only to that static baseline; the Zenodo DOI is not the archive DOI for this
dynamic OMN-PF repository:

- Static baseline GitHub: https://github.com/Mostfa3385/lifi-adt-ofdma-ffr
- Static baseline Zenodo: https://doi.org/10.5281/zenodo.19714290


## Repository Contents

```text
.
|-- simulator_omn_pf_final.py       # Dynamic channel and scheduler simulator
|-- run_overnight_final.py          # Resumable multi-seed experiment runner
|-- generate_figures.py             # Regenerates manuscript result figures
|-- run_final_overnight_windows.bat # Full Windows run
|-- run_final_overnight.sh          # Full Linux/macOS run
|-- run_timing_test.sh              # Short runtime estimate
|-- results_gap_aware/
|   |-- final_overnight_raw.csv
|   |-- final_overnight_summary_ci.csv
|   |-- final_overnight_report.txt
|   `-- final_overnight_metadata.json
|-- figures/                        # Generated PNG figures
|-- requirements.txt
|-- CITATION.cff
|-- .zenodo.json
|-- LICENSE
`-- CHANGELOG.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No GPU is required.

## Quick Smoke Test

```bash
python run_overnight_final.py --output_dir release_audit_smoke --seeds 1 --trials 1 --frames 2 --mode main --workers 1
```

The generated `release_audit_smoke/` directory is ignored by Git.

## Reproduce the Manuscript Experiment

The reported dynamic experiment uses:

- 10 independent seeds;
- 300 Monte Carlo trials per seed;
- 75 frames per trial;
- 42 users per macrocell;
- the NLOS-stress profile;
- four main schedulers and eight ablation configurations.

Run:

```bash
python run_overnight_final.py --output_dir results_gap_aware --seeds 1-10 --trials 300 --frames 75 --mode both --workers 4
```

The runner is resumable. Repeating the command skips completed
`(experiment, scheduler, seed)` rows already present in the raw CSV.

Use fewer workers if memory or thermal limits are a concern:

```bash
python run_overnight_final.py --output_dir results_gap_aware --seeds 1-10 --trials 300 --frames 75 --mode both --workers 2
```

## Regenerate Figures

```bash
python generate_figures.py --summary results_gap_aware/final_overnight_summary_ci.csv --outdir figures
```

## Main Result

The following means are computed over 10 seeds and correspond to the dynamic
scheduler comparison in manuscript Table 7.

| Scheduler | Edge C10 | Edge C20 | Scheduled SE (bps/Hz) | Jain fairness | Average outage duration (frames) |
|---|---:|---:|---:|---:|---:|
| RR1 | 2.35% | 1.38% | 88.71 | 0.0272 | 42.68 |
| PF | 6.48% | 3.05% | 176.14 | 0.0540 | 34.13 |
| OMN-PF | 8.81% | 5.00% | 183.57 | 0.0606 | 32.70 |

Relative to conventional PF, service-gap-aware OMN-PF improves:

- edge-user 10 dB continuity by approximately 36.0%;
- edge-user 20 dB continuity by approximately 63.7%;
- scheduled spectral efficiency by approximately 4.22%;
- temporal Jain fairness by approximately 12.21%;
- average outage duration from 34.13 to 32.70 frames.

Max-SINR is retained as an opportunistic high-rate reference. The paper does
not claim that OMN-PF universally dominates max-SINR.

## Scheduler Metric

OMN-PF preserves the proportional-fair rate/history ratio and multiplies it by
one reliability weight:

```text
M_u(t) = R_hat_u(t) / (R_bar_u(t) + epsilon) * W_u(t)

W_u(t) = 1
       + lambda_o O_u(t)
       + lambda_e E_u(t)
       + lambda_b B_u(t)
       + lambda_n N_u(t)
       + lambda_s S_u(t)
```

The terms represent outage risk, edge-user state, beam-transition risk,
NLOS fragility/assistance, and bounded service-gap priority. The
`omn_no_gap` ablation disables the service-gap term.

## Output Files

- `final_overnight_raw.csv`: one row per experiment, scheduler, and seed.
- `final_overnight_summary_ci.csv`: means, standard deviations, and 95% confidence intervals.
- `final_overnight_report.txt`: concise PF/OMN-PF comparison.
- `final_overnight_metadata.json`: exact run configuration.

## Scope

This is a protocol-level simulation, not a hardware testbed or full
deterministic ray tracer. First-order NLOS is used as a controlled stress
model. Higher-order diffuse reflection, measured blockage traces, noisy CSI,
handover signaling delay, LED nonlinearity, and hardware calibration are
outside the current scope.

## License

See `LICENSE`.
