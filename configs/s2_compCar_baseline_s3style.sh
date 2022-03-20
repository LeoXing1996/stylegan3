#!/usr/bin/env bash

DESC=s3-config-baseline
JOB_NAME=s2-compCar-baseline
DATA=s3://data/compCar256/

CONFIG=stylegan2
BATCH=32
KIMG=57000
PYTHON_ARGS="--mirror --gamma 8.2"
