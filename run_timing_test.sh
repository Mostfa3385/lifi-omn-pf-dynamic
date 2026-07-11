#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_overnight_final.py --output_dir results_timing --seeds 1 --trials 20 --frames 20 --mode main --workers 1
