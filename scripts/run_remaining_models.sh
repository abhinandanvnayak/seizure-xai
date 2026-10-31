#!/bin/bash
# Run the remaining models end to end: extract features, then evaluate each with the leakage-free protocol.
# Resumable: finished caches are skipped, so you can stop it (Ctrl-C) and run it again.
#
#   cd /Users/abhi/Code/RTDS && caffeinate -i bash scripts/run_remaining_models.sh 2>&1 | tee remaining_models.log
#
# Rough time on a MacBook Air M4: 5-7 hours in total.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
P=$(for i in $(seq -w 1 23); do echo chb$i; done)     # chb24 has no recording times and is skipped

echo "== 1/4 BIOT + LaBraM embeddings (one pass over the EDF files)"
$PY -u scripts/extract_embeddings.py $P --models biot labram

echo "== 2/4 Chronos-Bolt zero-shot scoring"
$PY -u scripts/run_zeroshot.py --forecaster chronos $P

echo "== 3/4 MOMENT embeddings"
$PY -u scripts/extract_moment.py $P

echo "== 4/4 Evaluation (original configuration and nested-validated tuning, per model)"
for m in biot labram moment; do
  $PY -u scripts/tune_frozen.py --model $m
done
$PY -u scripts/tune_zeroshot.py --model zs_chronos

echo "== done. Results: results_*.json (results_<model>_tuned.json holds both baseline and tuned)"
