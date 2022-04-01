#!/usr/bin/env bash

DESC="s3-config-baseline"
JOB_NAME=s2-incident-baseline
DATA=s3://data/IncidentCar1k256/

CONFIG=stylegan2
BATCH=32
KIMG=57000

PYTHON_ARGS="--mirror 1 --gamma 10"
