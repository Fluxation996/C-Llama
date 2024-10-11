#!/bin/bash

EXP_DIR=$1

# 512 * 1k * 400k = 200b tokens.
# 512 * 1k * 200k = 100b tokens.
# 512 * 1k * 100k = 50b tokens (default).
# 512 * 1k * 20k = 10b tokens.
TRAINING_STEPS=20000
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

NUM_EXPERTS=8
if [ -n "${3}" ]; then
    NUM_EXPERTS=$3;
fi

TOP_K=1
if [ -n "${4}" ]; then
    TOP_K=$4;
fi

LOSS_WEIGHT=0.1
if [ -n "${5}" ]; then
    LOSS_WEIGHT=$5;
fi

BATCH_SIZE=8
if [ -n "${6}" ]; then
    BATCH_SIZE=$6;
fi

##
### Pre-training for dMoE 760M parameter.
##

# MoE hyperparameters.
MOE_ARGUMENTS="\
--moe-num-experts=${NUM_EXPERTS} \
--moe-loss-weight=${LOSS_WEIGHT} \
--moe-top-k=${TOP_K}"

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 24 \
--hidden-size 1536 \
--num-attention-heads 16 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size ${BATCH_SIZE} \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.0004 \
--min-lr 0.00004 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01 \
--optimizer adafactor"

PILE_DATASET="\
1.0 \
/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/data/refined_web_megatron/megatron_large_refined_web_0_text_document \
1.0 \
/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/data/refined_web_megatron/megatron_large_refined_web_1_text_document"

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${PILE_DATASET} \
--vocab-file /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/tokenizer/gpt2-vocab.json \
--merge-file /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/tokenizer/gpt2-merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--fp16 \
--DDP-impl local \
--moe-expert-model-parallelism \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn"

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"
#python -m torch.distributed.launch
torchrun ${DISTRIBUTED_ARGUMENTS} \
       /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/Megatron-LM/pretrain_gpt.py \
       ${MOE_ARGUMENTS} \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train.log
