#!/bin/bash

cd ./src/

if [ $1 -eq 1 ]
then
    python3 run_nb.py $2 $3 $4
else
    python3 run_svm.py $2 $3 $4
fi