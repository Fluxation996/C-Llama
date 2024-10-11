#!/bin/bash
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1

MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# 与LLaMA网络结构相关的配置
custom_options="--disable-bias-linear \
                --swiglu \
                --untie-embeddings-and-output-weights \
                --swiglu-make-ffn-hidden-size-divisible-by 256 \
                --position-embedding-type rope \
                --init-method-std 0.02 \
                --disable-scaled-init-method \
                --normalization RMSNorm \
                --norm-epsilon 1e-5 \
                "

# 设置网络结构相关参数，这里是一个1.3B模型的设置
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
SEQ_LEN=2048

# micro-batchsize与global_batchsize设置，注意global_batchsize需被DP-size整除
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32

# 学习率
LR=2e-4
MIN_LR=1e-5

# 混合精度训练设置
PR=bf16

# TP/PP设置，DP=GPU总数/TP/PP
TP=1  # 不超过8
PP=1  # 要能整除transformer-block的数量


AC="sel"
DO=true
FL=true
SP=true
SAVE_INTERVAL=5000

#test_corpus_text_document
TRAIN_DATASET_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/data_processed/tmp0122/reason_5.npy"

TOKENIZER_TYPE="PretrainedFromHF"
TOKENIZER_NAME_OR_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/public/jiaran/tokenizer_v3"


TRAIN_TOKENS=70000000
WARMUP_TOKENS=102400
OUTPUT_BASEPATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo/test"


# PRETRAIN_CHECKPOINT_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo/checkpoint/dmoe_LLaMA_1.3Bx8_top2_128b/iter_0027913/mp_rank_00"
PRETRAIN_CHECKPOINT_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo/checkpoint/dmoe_LLaMA_1.3Bx8_top2_128b"

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
        --load $PRETRAIN_CHECKPOINT_PATH"
fi
if [ $AC = full ]; then
    activation_checkpoint_options=" \
        --recompute-method uniform \
        --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi
if [ $PR = fp16 ]; then
    pr_options=" \
        --fp16 \
        --initial-loss-scale 65536 \
        --use-flash-attn"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16 \
        --use-flash-attn"
fi
if [ $DO = true ]; then
    do_options=" \
        --use-distributed-optimizer"
elif [ $DO = false ]; then
    do_options=" \
                    "
fi
if [ $FL = true ]; then
    flash_options=" \
        --use-flash-attn"
elif [ $FL = false ]; then
    flash_options=" \
                    "
fi
if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
        --sequence-parallel"
elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

# NAME="${ENV}-pretrain-megatron-llama-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
NAME="sft-trail_1"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"


megatron_options=" \
        --finetune \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --train-data-path ${TRAIN_DATASET_PATH} \
        --dataloader-type cyclic \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --log-interval 1 \
        --eval-interval ${SAVE_INTERVAL} \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 8 \
        --seed 1234 \
        --position-embedding-type rope \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_NAME_OR_PATH} \
        "


#Moe的配置
MOE_ARGUMENTS=" \
    --moe-num-experts=8 \
    --moe-loss-weight=0.01 \
    --moe-top-k=2 \
    --moe-capacity-factor 0 \
    --mlp_type=glu_llama"

torchrun $DISTRIBUTED_ARGS \
        Megatron-INF/sft_gpt.py \
        ${megatron_options} \
        ${activation_checkpoint_options} \
        ${do_options} \
        ${pr_options} \
        ${sp_options} \
        ${flash_options} \
        ${load_options} \
        ${custom_options} \
        ${MOE_ARGUMENTS}
