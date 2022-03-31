#!/usr/bin/env bash

DESC="transfer-compCar-ada"
JOB_NAME=s2-incident-baseline
DATA=s3://data/IncidentCar1k256/

CONFIG=stylegan2
BATCH=32
KIMG=57000

# TODO: support transfer learning
PYTHON_ARGS="--gamma 10 --nerf_config=nerf_configs/car_from_scratch_nerf_s2.yml --resume TODO"
