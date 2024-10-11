#!/bin/bash

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT"

EXP_DIR="outputs"
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

TRAINING_STEPS=20000
if [ -n "${3}" ]; then
    TRAINING_STEPS=$2;
fi

NUM_EXPERTS=64
if [ -n "${4}" ]; then
    NUM_EXPERTS=$3;
fi

CAPACITY_FACTOR=3
if [ -n "${5}" ]; then
    CAPACITY_FACTOR=$4;
fi

TOP_K=1
if [ -n "${6}" ]; then
    TOP_K=$4;
fi

LOSS_WEIGHT=0.1
if [ -n "${7}" ]; then
    LOSS_WEIGHT=$5;
fi

BATCH_SIZE=32
if [ -n "${8}" ]; then
    BATCH_SIZE=$6;
fi


##
### Pre-training for MoE 46M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-capacity-factor=${CAPACITY_FACTOR} \
--moe-top-k=${TOP_K}"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 6 \
--hidden-size 512 \
--num-attention-heads 8 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.00015 \
--min-lr 0.00001 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

PILE_DATASET="\
1.0 \
/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/data/refined_web_megatron/megatron_large_refined_web_0_text_document \
1.0 \
/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/data/refined_web_megatron/megatron_large_refined_web_1_text_document"

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${PILE_DATASET} \
--vocab-file /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/models/gpt2/vocab.json \
--merge-file /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/models/gpt2/merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--bf16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn"

# COMPUTE_ARGUMENTS="\
# --bf16 \
# --DDP-impl local \
# --no-async-tensor-model-parallel-allreduce \
# --use-flash-attn"

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

export LOGLEVEL=INFO

torchrun ${DISTRIBUTED_ARGUMENTS} \
    Megatron-LM/pretrain_gpt.py \
    ${MOE_ARGUMENTS} \
    ${MODEL_ARGUMENTS} \
    ${TRAINING_ARGUMENTS} \
    ${DATA_ARGUMENTS} \
    ${COMPUTE_ARGUMENTS} \
    ${CHECKPOINT_ARGUMENTS} \
    ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train.log
