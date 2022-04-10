#!/usr/bin/env bash
set -x

PARTITION=mm_lol
DRAIN_NODE="SH-IDC1-10-142-4-150,SH-IDC1-10-142-4-159,SH-IDC1-10-142-4-187,SH-IDC1-10-142-4-93,SH-IDC1-10-142-4-188"

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

# load configs
VAR_FILE=$1
source ${VAR_FILE}

WORK_DIR=./out

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
                    --slurm --batch ${BATCH} --kimg ${KIMG} --data ${DATA} --desc ${DESC}\
                    ${PYTHON_ARGS}