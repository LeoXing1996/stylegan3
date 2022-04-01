#!/usr/bin/env bash

DESC="baseline"
JOB_NAME=s2-incident-baseline
DATA=s3://data/IncidentCar1k256/

CONFIG=stylegan2
BATCH=32
KIMG=15000

PYTHON_ARGS="--mirror 1 --gamma 10"
