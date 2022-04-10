#!/usr/bin/env bash

DESC=baseline
JOB_NAME=s2-compCar-baseline
DATA=s3://data/compCar256/

CONFIG=stylegan2
BATCH=32
KIMG=25000
PYTHON_ARGS="--gamma 10"
