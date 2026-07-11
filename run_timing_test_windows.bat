@echo off
setlocal
cd /d "%~dp0"
python run_overnight_final.py --output_dir results_timing --seeds 1 --trials 20 --frames 20 --mode main --workers 1
pause

