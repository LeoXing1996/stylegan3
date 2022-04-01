#!/usr/bin/env bash

DESC="baseline"
JOB_NAME=s3-incident-baseline
DATA=s3://data/IncidentCar1k256/

CONFIG=stylegan3-r
BATCH=32
KIMG=15000

PYTHON_ARGS="--mirror 1 --gamma 10"
