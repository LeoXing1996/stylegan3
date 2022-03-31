#!/usr/bin/env bash

DESC="GIRAFFE-pretrain-Res32-ada"
JOB_NAME=s2-incident-baseline
DATA=s3://data/incident1k/

CONFIG=stylegan2
BATCH=32
KIMG=57000

PYTHON_ARGS="--gamma 10 --nerf_config=nerf_configs/car_pretrain_s2.yml"
