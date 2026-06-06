#!/usr/bin/env bash
# One-shot: download -> build features -> train -> (optional) submit.
# Prereqs: ~/.kaggle/kaggle.json (or KAGGLE_KEY) + rules accepted on the website.
#   bash run_all.sh              # full + submit
#   bash run_all.sh --no-submit  # build submission.csv only
set -euo pipefail
cd "$(dirname "$0")"
COMP="optiver-realized-volatility-prediction"
SUBMIT=1
[[ "${1:-}" == "--no-submit" ]] && SUBMIT=0

echo "==> [1/4] Download"; python3 download_data.py
echo "==> [2/4] Build features"; python3 build_features.py
echo "==> [3/4] Train"; python3 train.py
echo "==> [4/4] Submit"
if [[ "$SUBMIT" -eq 1 ]]; then
  kaggle competitions submit -c "$COMP" -f output/submission.csv \
    -m "LGBM RMSPE | order-book microstructure + sub-window realized vol ($(date +%F))"
  echo "See: https://www.kaggle.com/competitions/$COMP/submissions"
else
  echo "Skipped submit. File at output/submission.csv"
fi
