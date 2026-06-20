#!/bin/bash

cd /Users/augustinfaye/Documents/Agence_Dvpt

source agence/bin/activate

export $(grep -v '^#' .env | xargs)

python models/train_regression.py >> logs/train_cron.log 2>&1
