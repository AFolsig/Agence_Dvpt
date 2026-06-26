#!/bin/bash

cd /Users/augustinfaye/Documents/Agence_Dvpt

source agence/bin/activate

export $(grep -v '^#' .env | xargs)

python -m models.train_regression >> logs/train_cron.log 2>&1
