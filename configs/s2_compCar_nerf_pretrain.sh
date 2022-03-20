#!/usr/bin/env bash

DESC="GIRAFFE-pretrain-Res32"
JOB_NAME=s2-compCar-baseline
DATA=s3://data/compCar256/

CONFIG=stylegan2
BATCH=32
KIMG=57000

PYTHON_ARGS="--aug noaug --gamma 10 --nerf_config=nerf_configs/car_pretrain_s2.yml"
