# Changelog

## 1.1.1 - 2026-07-11

- Replaced the fixed two-user admission cap with a variable resource- and SINR-aware multi-user OFDMA admission rule shared by all dynamic schedulers.
- Derived per-pool user ceilings from the current center/edge pool size and the minimum-subcarrier requirement.
- Added post-sharing SINR and regional sum-rate retention checks before admitting each additional user.
- Added occupancy diagnostics for one-user, two-user, and three-or-more-user active regions.

## 1.1.0 - 2026-07-11

- Replaced one-user-per-region frame scheduling with a common bounded multi-user OFDMA admission model for RR1, max-SINR, PF, OMN-PF, and OMN-PF-A.
- Kept the strict-FFR center/edge split fixed at delta = 2/3; delta sensitivity remains a separate experiment.
- Made OMN-PF and all conventional baselines use equal OFDMA sharing and uniform normalized regional power under the same admission limits.
- Made OMN-PF-A use the same OMN-PF ranking/admission set, followed by bounded reliability-aware OFDMA sharing and adaptive power redistribution.
- Enforced per-region power-budget preservation and added numerical budget-error diagnostics.
- Added simultaneous-user, effective-subcarrier, and multi-user OFDMA diagnostics to simulator and overnight-runner outputs.
- Marked the archived single-user dynamic results as requiring regeneration before use with this refactor.

## 1.0.0 - 2026-07-06

- Added frame-based random-waypoint mobility.
- Added time-correlated receiver tilt and azimuth.
- Added first-order NLOS-stress channel components.
- Added target-user co-channel interference evaluation.
- Added outage-, edge-, beam-transition-, NLOS-fragility-, and service-gap-aware OMN-PF scheduling.
- Added RR1, max-SINR, conventional PF, and controlled OMN-PF ablations.
- Added edge-user 10 dB and 20 dB continuity, outage-duration, service-gap, fairness, and rate metrics.
- Added resumable multi-seed experiment execution with 95% confidence intervals.
- Added the verified 10-seed, 300-trial, 75-frame result archive used by manuscript Table 7.

