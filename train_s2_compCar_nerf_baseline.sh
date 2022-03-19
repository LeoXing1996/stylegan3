#!/usr/bin/env bash
set -x


PARTITION=mm_lol
JOB_NAME=s2-compCar-baseline
DRAIN_NODE="SH-IDC1-10-142-4-150"

CONFIG=stylegan2
WORK_DIR=./out


GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -x ${DRAIN_NODE} \
    ${SRUN_ARGS} \
    python train.py --outdir=${WORK_DIR} --cfg=${CONFIG} --gpus=${GPUS} \
                    --slurm --batch 32 --aug noaug --gamma 10 --kimg 57000 \
                    --data s3://data/compCar256/
