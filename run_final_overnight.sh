#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_overnight_final.py --output_dir results_gap_aware --seeds 1-10 --trials 300 --frames 75 --mode both --workers 4
