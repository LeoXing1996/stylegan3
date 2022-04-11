#!/usr/bin/env bash

DESC="GIRAFFE-Freeze-Res32"
JOB_NAME=s2-compCar-baseline
DATA=s3://data/compCar256/

CONFIG=stylegan2
BATCH=32
KIMG=25000

PYTHON_ARGS="--aug noaug --gamma 10 --nerf_config=nerf_configs/car_freeze_nerf_s2.yml"
