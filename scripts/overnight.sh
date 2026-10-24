#!/bin/sh
# Zero-shot TimesFM on all patients, then the CBraMod stage. Logs: zeroshot_run.log, cbramod_run.log.
cd /Users/abhi/Code/RTDS
ALL=$(for i in $(seq -w 1 24); do echo chb$i; done)
.venv/bin/python -u scripts/run_zeroshot.py $ALL > zeroshot_run.log 2>&1
.venv/bin/python -u scripts/run_frozen.py --model cbramod $ALL > cbramod_run.log 2>&1
echo finished > overnight.done
