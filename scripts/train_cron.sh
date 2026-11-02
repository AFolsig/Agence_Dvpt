#!/bin/bash

echo "CRON START : $(date '+%Y-%m-%d %H:%M:%S')" >> /Users/augustinfaye/Documents/Agence_Dvpt/logs/cron_execution.log

cd /Users/augustinfaye/Documents/Agence_Dvpt

source agence/bin/activate

export $(grep -v '^#' .env | xargs)

python monitoring/generate_drift_data.py --mode reel >> logs/train_cron.log 2>&1 || exit 1

python monitoring/drift_report.py >> logs/train_cron.log 2>&1 || exit 1

python monitoring/check_drift_and_retrain.py --execute >> logs/train_cron.log 2>&1
