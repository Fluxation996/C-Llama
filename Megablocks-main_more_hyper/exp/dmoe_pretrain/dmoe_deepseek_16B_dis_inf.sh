#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1  # 环境变量设置，开启Sequence parallel需打开

NNODES=${WORLD_SIZE} # 环境变量读取，无需手动设置，与DLC启动时设置的节点数量相关
NODE_RANK=${RANK}    # 环境变量读取，无需手动设置
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU} # 同上



# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node $GPUS_PER_NODE \
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
                --init-method-std 0.006 \
                --disable-scaled-init-method \
                --normalization RMSNorm \
                --norm-epsilon 1e-5 \
                "

# 设置网络结构相关参数，参考 deepseek 16B
NUM_LAYERS=28
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=10944
MOE_EXPERT_FFN_HIDDEN_SIZE=1408
MOE_SHARED_EXPERT_FFN_HIDDEN_SIZE=2816
NUM_ATTN_HEADS=16
SEQ_LEN=4096

# micro-batchsize与global_batchsize设置，注意global_batchsize需被DP-size整除
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
# deepseek 4.5K

# 学习率
LR=1.2e-4
MIN_LR=1e-6

# maxmum 4.2e-4
# 80% steps * 0.316
# 90% steps * 0.316

# 混合精度训练设置
PR=bf16

# TP/PP设置，DP=GPU总数/TP/PP
TP=1  # 不超过8
PP=4  # 要能整除transformer-block的数量

AC="none" # Activation checkpointing
DO=true   # ZERO optimizer
FL=true   # Flash-attention
SP=false  # Sequence-parallel
SAVE_INTERVAL=1000  # Checkpoint保存的step间隔

# 设置数据路径
DATA_ROOT="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/data_processed/data_0115_v1"

TOKENIZER_TYPE="PretrainedFromHF"
TOKENIZER_NAME_OR_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/public/jiaran/tokenizer_v3"

# 数据路径的写法只需要写到basename即可，不用带扩展名.bin/.idx
# 先写tokens数，再写路径
DATASET_1="webtext-redpajama_v2"
DATASET_2="webtext-refinedweb"
DATASET_3="webtext-fandom"
DATASET_4="webtext-quora"
DATASET_5="wiki-wiki-20230401_en"
DATASET_6="wiki-wikipedia_20220301_en"
DATASET_7="paper-redpajama_arxiv"
DATASET_8="book-zlib_fb2"
DATASET_9="other-hupd"
DATASET_10="journal-pubmed"
DATASET_11="financial_lesson"
DATASET_12="webtext-weixin_page"
DATASET_13="webtext-toutiao_articles"
DATASET_14="webtext-casia_chinesewebtext"
DATASET_15="qa-hdf_zh"
DATASET_16="other-out_law"
DATASET_17="webtext-zhihu_zh"
DATASET_18="webtext-zhihu_article"
DATASET_19="high_quality-xinwenlianbo_zh"
DATASET_20="wiki-baidu_baike_zh"
DATASET_21="wiki-wiki-20230401_zh"
DATASET_22="wiki-wiki_zh"
DATASET_23="high_quality-zgbk_zh"
DATASET_24="webtext-mnvbc_news_zh"
DATASET_25="paper-mnvbc_paper_zh"
DATASET_26="paper-medjournals_zh"
DATASET_27="financial_knowledge"

SUFFIX="_0115"
DATASET_PATH=" \
    73365297922  ${DATA_ROOT}/${DATASET_1}${SUFFIX} \
    14495438732  ${DATA_ROOT}/${DATASET_2}${SUFFIX} \
    4277074534  ${DATA_ROOT}/${DATASET_3}${SUFFIX} \
    1812421409  ${DATA_ROOT}/${DATASET_4}${SUFFIX} \
    2476474397  ${DATA_ROOT}/${DATASET_5}${SUFFIX} \
    535088419  ${DATA_ROOT}/${DATASET_6}${SUFFIX} \
    2341731291  ${DATA_ROOT}/${DATASET_7}${SUFFIX} \
    1287282522  ${DATA_ROOT}/${DATASET_8}${SUFFIX} \
    740963302  ${DATA_ROOT}/${DATASET_9}${SUFFIX} \
    2636962692  ${DATA_ROOT}/${DATASET_10}${SUFFIX} \
    1031286  ${DATA_ROOT}/${DATASET_11}${SUFFIX} \
    10124873401  ${DATA_ROOT}/${DATASET_12}${SUFFIX} \
    3583357411  ${DATA_ROOT}/${DATASET_13}${SUFFIX} \
    3061885702  ${DATA_ROOT}/${DATASET_14}${SUFFIX} \
    372392667  ${DATA_ROOT}/${DATASET_15}${SUFFIX} \
    124147516  ${DATA_ROOT}/${DATASET_16}${SUFFIX} \
    8115197  ${DATA_ROOT}/${DATASET_17}${SUFFIX} \
    17343147  ${DATA_ROOT}/${DATASET_18}${SUFFIX} \
    3206957  ${DATA_ROOT}/${DATASET_19}${SUFFIX} \
    4324910826  ${DATA_ROOT}/${DATASET_20}${SUFFIX} \
    570666328  ${DATA_ROOT}/${DATASET_21}${SUFFIX} \
    436417534  ${DATA_ROOT}/${DATASET_22}${SUFFIX} \
    209808756  ${DATA_ROOT}/${DATASET_23}${SUFFIX} \
    917663133  ${DATA_ROOT}/${DATASET_24}${SUFFIX} \
    30316607  ${DATA_ROOT}/${DATASET_25}${SUFFIX} \
    249232644  ${DATA_ROOT}/${DATASET_26}${SUFFIX} \
    48297869  ${DATA_ROOT}/${DATASET_27}${SUFFIX}"


TRAIN_TOKENS=128052402579    # 训练总token数
LR_DECAY_TOKENS=128052402579 # 学习率decay的范围，1.0T tokens
WARMUP_TOKENS=$(( 2000 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} )) # warmup during 2000 iters

PRETRAIN_CHECKPOINT_PATH=none
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
LR_DECAY_ITERS=$(( ${LR_DECAY_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

JOB_NAME="dmoe_deepseek_16B"
OUTPUT_BASEPATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${JOB_NAME}"
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${JOB_NAME}"
mkdir -p ${TENSORBOARD_DIR}

# Megatron的配置
SEED=42
megatron_options="  \
        --continue-on-missing-checkpoint \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --load ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 100,0,0 \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --log-interval 100 \
        --eval-interval ${SAVE_INTERVAL} \
        --eval-iters 100 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 8 \
        --seed ${SEED} \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-name-or-path ${TOKENIZER_NAME_OR_PATH} \
        "


if [ $PR = fp16 ]; then
    pr_options=" \
        --fp16 \
        --initial-loss-scale 65536"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi


#Moe的配置
# shared 2 + routed 6 outof 64
MOE_ARGUMENTS=" \
    --moe_num_layers=27 \
    --moe_expert_ffn_hidden_size=${MOE_EXPERT_FFN_HIDDEN_SIZE} \
    --moe_share_expert_ffn_hidden_size=${MOE_SHARED_EXPERT_FFN_HIDDEN_SIZE} \
    --moe-num-experts=64 \
    --moe-loss-weight=0.001 \
    --add_moe_share_expert \
    --moe-top-k=6 \
    --share_weight=1 \
    --moe_weight=1 \
    --moe-capacity-factor 0 \
    --mlp_type=glu"
# 注意 MoE config 设置成 0.125 倍的 FFNs


# export LOGLEVEL=INFO
torchrun $DISTRIBUTED_ARGUMENTS \
    Megatron-INF/pretrain_gpt.py \
    ${megatron_options} \
    ${activation_checkpoint_options} \
    ${do_options} \
    ${pr_options} \
    ${sp_options} \
    ${flash_options} \
    ${load_options} \
    ${custom_options} \
    ${MOE_ARGUMENTS}
