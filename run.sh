#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --quiet -r requirements.txt
python -m pip install --quiet -e .
mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

python scripts/pull_data.py
python scripts/universe.py
python scripts/clean.py
python scripts/candidates.py
until python scripts/cointegration.py; do echo "continuing cointegration tests"; done
python scripts/vecm.py
python scripts/pull_funding.py
python scripts/persistence.py
python scripts/high_low.py
python scripts/backtest.py
