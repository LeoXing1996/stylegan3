#!/usr/bin/env bash

DESC="baseline"
JOB_NAME=s3-compCar-baseline
DATA=s3://data/compCar256/

CONFIG=stylegan3-r
BATCH=32
KIMG=25000

PYTHON_ARGS="--mirror 1 --gamma 10"
