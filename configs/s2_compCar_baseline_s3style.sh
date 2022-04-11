#!/usr/bin/env bash
# s2 network + default s3 hyperparameters

DESC=s3-config-baseline
JOB_NAME=s2-compCar-baseline
DATA=s3://data/compCar256/

CONFIG=stylegan2
BATCH=32
KIMG=25000
PYTHON_ARGS="--mirror --gamma 8.2"
