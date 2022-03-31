#!/usr/bin/env bash

DESC="GIRAFFE-fromScratch-Res32-ada"
JOB_NAME=s2-incident-baseline
DATA=s3://data/IncidentCar1k256/

CONFIG=stylegan2
BATCH=32
KIMG=57000

PYTHON_ARGS="--gamma 10 --nerf_config=nerf_configs/car_pretrain_nerf_s2.yml"
