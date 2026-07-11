# Service-Gap-Aware OMN-PF for Dynamic LiFi Attocells

This repository contains the dynamic reliability-aware extension of the
ADT-SDMA, OFDMA, and strict-FFR LiFi attocell benchmark. It adds:

- frame-based random-waypoint mobility;
- time-correlated receiver orientation;
- first-order NLOS-stress propagation;
- serving-region and beam-transition tracking;
- outage, edge, NLOS-fragility, and service-gap scheduling states;
- a service-gap-aware OMN-PF scheduler;
- common frame-level multi-user OFDMA admission for RR1, Max-SINR, PF, OMN-PF, and OMN-PF-A;
- adaptive OFDMA sharing and normalized regional power control in OMN-PF-A;
- multi-seed confidence-interval reporting and controlled ablations.

This repository extends the earlier static benchmark. The links below refer
only to that static baseline; the Zenodo DOI is not the archive DOI for this
dynamic OMN-PF repository:

- Static baseline GitHub: https://github.com/Mostfa3385/lifi-adt-ofdma-ffr
- Static baseline Zenodo: https://doi.org/10.5281/zenodo.19714290

The archived `v1.0.0` release of this dynamic repository is available at:

- Dynamic OMN-PF Zenodo: https://doi.org/10.5281/zenodo.21220269

The current repository state corresponds to release `v1.1.1`. A new Zenodo
version DOI should be generated from the GitHub `v1.1.1` release before this
specific updated code snapshot is cited as the version of record.

## Repository Contents

```text
.
|-- simulator_omn_pf_final.py       # Dynamic channel and scheduler simulator
|-- run_overnight_final.py          # Resumable multi-seed experiment runner
|-- generate_figures.py             # Regenerates manuscript result figures
|-- run_final_overnight_windows.bat # Full Windows run
|-- run_final_overnight.sh          # Full Linux/macOS run
|-- run_timing_test.sh              # Short runtime estimate
|-- results_variable_admission/
|   |-- final_overnight_raw.csv
|   |-- final_overnight_summary_ci.csv
|   |-- final_overnight_report.txt
|   `-- final_overnight_metadata.json
|-- results_gap_aware/               # Earlier gap-aware full run
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
- five main schedulers.

Run:

```bash
python run_overnight_final.py --output_dir results_variable_admission --seeds 1-10 --trials 300 --frames 75 --mode main --workers 4
```

The runner is resumable. Repeating the command skips completed
`(experiment, scheduler, seed)` rows already present in the raw CSV.

Use fewer workers if memory or thermal limits are a concern:

```bash
python run_overnight_final.py --output_dir results_variable_admission --seeds 1-10 --trials 300 --frames 75 --mode main --workers 2
```

## Regenerate Figures

```bash
python generate_figures.py --summary results_variable_admission/final_overnight_summary_ci.csv --outdir figures
```

## Main Result

The following means are the included `v1.1.1` variable-admission multi-user
OFDMA results in `results_variable_admission/`.

| Scheduler | Edge C10 | Edge C20 | Scheduled SE (bps/Hz) | Jain fairness | Average outage duration (frames) |
|---|---:|---:|---:|---:|---:|
| RR1 | 3.76% | 1.71% | 78.65 | 0.0435 | 38.22 |
| Max-SINR | 11.50% | 4.82% | 164.60 | 0.0936 | 41.26 |
| PF | 8.25% | 3.23% | 144.19 | 0.0855 | 31.54 |
| OMN-PF | 11.37% | 4.97% | 159.25 | 0.0859 | 32.67 |
| OMN-PF-A | 11.56% | 6.11% | 159.18 | 0.0866 | 31.07 |

Relative to conventional PF, service-gap-aware OMN-PF improves:

- edge-user 10 dB continuity by approximately 37.88%;
- edge-user 20 dB continuity by approximately 53.95%;
- scheduled spectral efficiency by approximately 10.44%;
- temporal Jain fairness by approximately 0.46%.

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

## Common Multi-User OFDMA Refactor

The dynamic simulator now uses a common multi-user OFDMA access model for all
frame-level schedulers:

```text
RR1 / Max-SINR / PF / OMN-PF
    = scheduler-specific user ranking
    + the same bounded multi-user admission
    + equal OFDMA shares
    + uniform normalized regional power

OMN-PF-A
    = the same OMN-PF ranking and admission limits
    + reliability-aware OFDMA shares
    + bounded adaptive power redistribution
```

The strict-FFR split remains fixed at `delta = 2/3`. Delta adaptation is not
used in the manuscript run and should be justified separately by sensitivity
analysis.

All main schedulers now use the same variable resource- and SINR-aware
admission rule. There is no fixed two-user cap. Users are added one at a time
in scheduler-ranked order, up to the resource ceiling implied by the current
center/edge pool size and the minimum-subcarrier requirement. Beyond the first
user, a candidate is admitted only when every selected link remains above the
post-sharing SINR threshold and the predicted regional sum rate stays above a
conservative fraction of the single-user reference rate.

With the default `delta = 2/3` and 16 minimum subcarriers per admitted user,
the theoretical ceilings are seven center-pool users or two edge-pool users,
with at most nine users in a region when both pools are active. Actual frame-
level admission is normally lower because the SINR and rate-retention checks
remain active. This makes OFDMA an actual intra-region multiple-access
mechanism rather than only a bandwidth label around a one-user time scheduler.
Users served by the same region are orthogonal in subcarrier space and therefore
do not create intra-region co-subcarrier interference.

For the fixed-resource schedulers, subcarriers are divided equally and the
regional power budget is distributed with uniform effective power density. For
`omn_pf_a`, subcarrier and power weights use bounded outage, edge, service-gap,
transition, NLOS, and channel terms. Each active region is renormalized so that
its allocated user powers sum to the original regional power budget.

New CSV diagnostics include:

- `mean_scheduled_users_per_frame`
- `mean_scheduled_users_per_region`
- `mean_scheduled_users_per_active_region`
- `max_scheduled_users_in_region_observed`
- `fraction_active_regions_serving_1_user`
- `fraction_active_regions_serving_2_users`
- `fraction_active_regions_serving_3plus_users`
- `mean_effective_subcarriers_per_scheduled_user`
- `max_relative_region_power_budget_error`

Manual smoke test:

```bash
python simulator_omn_pf_final.py --trials 2 --frames 5 --users_per_macro 42 --schedulers rr1 proportional_fair omn_pf omn_pf_a --nlos_profile stress --out dynamic_extension/results_dynamic_v2/smoke_multi_user.csv
```

Small runner test:

```bash
python run_overnight_final.py --output_dir results_multi_user_smoke --seeds 1-1 --trials 2 --frames 5 --mode main --workers 1
```

Full main experiment:

```bash
python run_overnight_final.py --output_dir results_dynamic_multi_user --seeds 1-10 --trials 300 --frames 75 --mode main --workers 4
```
