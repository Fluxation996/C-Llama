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
                --init-method-std 0.02 \
                --disable-scaled-init-method \
                --normalization RMSNorm \
                --norm-epsilon 1e-5 \
                "

# 设置网络结构相关参数，这里是一个1.3B模型的设置
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
SEQ_LEN=4096

# micro-batchsize与global_batchsize设置，注意global_batchsize需被DP-size整除
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

# 学习率
LR=3e-4
MIN_LR=3e-5

# 混合精度训练设置
PR=bf16

# TP/PP设置，DP=GPU总数/TP/PP
TP=1  # 不超过8
PP=1  # 要能整除transformer-block的数量

AC="none" # Activation checkpointing
DO=true   # ZERO optimizer
FL=true   # Flash-attention
SP=false  # Sequence-parallel
SAVE_INTERVAL=1000  # Checkpoint保存的step间隔

# 设置数据路径
DATA_ROOT="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/data_processed/data_13t_0116"

TOKENIZER_TYPE="PretrainedFromHF"
TOKENIZER_NAME_OR_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/public/jiaran/tokenizer_v3"

# 数据路径的写法只需要写到basename即可，不用带扩展名.bin/.idx
# 先写tokens数，再写路径
DATASET_PATH=" \
12775568 ${DATA_ROOT}/financial_CFA_0116 \
2041998 ${DATA_ROOT}/ccri_0116 \
845206 ${DATA_ROOT}/crsp_case_0116 \
184584 ${DATA_ROOT}/mayo_clinic_proceeding_0116 \
1206304 ${DATA_ROOT}/webmd_0116 \
5746398 ${DATA_ROOT}/accr_0116 \
3206957 ${DATA_ROOT}/high_quality-xinwenlianbo_zh_0116 \
7913508 ${DATA_ROOT}/usmle_keyword_in_title_0116 \
2495538 ${DATA_ROOT}/aiaiyi_zhenliaozhinan_0116 \
5834628 ${DATA_ROOT}/high_quality-leetcode_0116 \
7429744 ${DATA_ROOT}/cmcr_0116 \
8916002 ${DATA_ROOT}/dingxiangyuan_0116 \
11851066 ${DATA_ROOT}/high_quality-cpp_reference_0116 \
19979302 ${DATA_ROOT}/jmedical_0116 \
13989854 ${DATA_ROOT}/differential_diagnosis_book_0116 \
1956072 ${DATA_ROOT}/high_quality-solidot_0116 \
5760946 ${DATA_ROOT}/mayo_clinic_0116 \
37169758 ${DATA_ROOT}/icliniq_article_0116 \
13975968 ${DATA_ROOT}/high_quality-36kr_zh_0116 \
36032990 ${DATA_ROOT}/usmle_step1_keyword_0116 \
46341346 ${DATA_ROOT}/high_quality-MSD_diagnosis_treatment_zh_0116 \
36425046 ${DATA_ROOT}/cochrane_0116 \
61287826 ${DATA_ROOT}/renwei_linchuangzhushou_0116 \
58328266 ${DATA_ROOT}/other-medlive_zh_0116 \
10576198 ${DATA_ROOT}/mathqa_0116 \
18714830 ${DATA_ROOT}/ape_chatlabel_0116 \
23936496 ${DATA_ROOT}/chunyuyisheng_qa_0116 \
27694448 ${DATA_ROOT}/other-dialogstudio_0116 \
109762890 ${DATA_ROOT}/medical_book_stage1_0116 \
46313222 ${DATA_ROOT}/drug_detail_zh_0116 \
6471620 ${DATA_ROOT}/aiaiyi_peixun_0116 \
29732090 ${DATA_ROOT}/cmexam_0116 \
62588611 ${DATA_ROOT}/high_quality-leiphone_v2_zh_0116 \
50053416 ${DATA_ROOT}/high_quality-Ancient-Books_zh_0116 \
96595738 ${DATA_ROOT}/financial_knowledge_0116 \
95938236 ${DATA_ROOT}/medmcqa_0116 \
81151970 ${DATA_ROOT}/webtext-zhihu_zh_0116 \
60633214 ${DATA_ROOT}/paper-mnvbc_paper_zh_0116 \
21460 ${DATA_ROOT}/unal_case_0116 \
38728600 ${DATA_ROOT}/guidelines_0116 \
2062572 ${DATA_ROOT}/financial_lesson_0116 \
2994796 ${DATA_ROOT}/high_quality-cde_manuals_0116 \
4447632 ${DATA_ROOT}/math_challenge_0116 \
17343147 ${DATA_ROOT}/webtext-zhihu_article_0116 \
1022598 ${DATA_ROOT}/high_quality-diagnosis-and-treatment_zh_0116 \
88900316 ${DATA_ROOT}/uptodate_0116 \
85421406 ${DATA_ROOT}/qa-med_conversation_zh_0116 \
22219272 ${DATA_ROOT}/aiaiyi_case_0116 \
58149034 ${DATA_ROOT}/amps_khan_0116 \
57424862 ${DATA_ROOT}/medical_book_stage2_0116 \
1144082 ${DATA_ROOT}/hindawi_case_0116 \
24031 ${DATA_ROOT}/high_quality-solidot_v2_zh_0116 \
6867282 ${DATA_ROOT}/webtext-finance_zh_0116 \
21105680 ${DATA_ROOT}/medtiku_0116 \
3686469 ${DATA_ROOT}/qa-baidu_zhidao_qa_zh_0116 \
61655874 ${DATA_ROOT}/cdc_0116 \
496618042 ${DATA_ROOT}/webtext-mathstackexchange_0116 \
73178640 ${DATA_ROOT}/financial_exam_0116 \
109696080 ${DATA_ROOT}/healthcare_magic_0116 \
91538 ${DATA_ROOT}/ccrj_case_0116 \
227078440 ${DATA_ROOT}/qa-baike_qa_zh_0116 \
6738212 ${DATA_ROOT}/research_report_0116 \
8113578 ${DATA_ROOT}/icliniq_qa_0116 \
20508392 ${DATA_ROOT}/medical_case_handled_0116 \
44530106 ${DATA_ROOT}/other-med_wiki_zh_0116 \
756762882 ${DATA_ROOT}/financial_year_report_0116 \
872835068 ${DATA_ROOT}/wiki-wiki_zh_0116 \
2758292 ${DATA_ROOT}/jmc_case_0116 \
1141332656 ${DATA_ROOT}/wiki-wiki-20230401_zh_0116 \
1682492 ${DATA_ROOT}/webtext-indiabix_0116 \
4867894 ${DATA_ROOT}/usmle_recommend_0116 \
1096280588 ${DATA_ROOT}/paper-cnki_textbook_01_0116 \
2849016 ${DATA_ROOT}/medical_case_unhandled_0116 \
11538244 ${DATA_ROOT}/medqa_us_5choices_0116 \
9714248 ${DATA_ROOT}/nejm_0116 \
13796140 ${DATA_ROOT}/medqa_mainland_5choices_0116 \
4280707352 ${DATA_ROOT}/wiki-wikipedia_20220301_en_0116 \
946484 ${DATA_ROOT}/oxford_0116 \
7413538 ${DATA_ROOT}/ceval_0116 \
49374212 ${DATA_ROOT}/rft_0116 \
316958064 ${DATA_ROOT}/wanjuan_textbook_0116 \
108563180 ${DATA_ROOT}/lancet_0116 \
36335702 ${DATA_ROOT}/qa-reddit_qa_geography_0116 \
546759470 ${DATA_ROOT}/high_quality-parallel_corpus_0116 \
1831582926 ${DATA_ROOT}/financial_other_report_0116 \
684599476 ${DATA_ROOT}/retrieval_corpus_triviaqa_0116 \
1031942 ${DATA_ROOT}/bmj_case_0116 \
419617512 ${DATA_ROOT}/high_quality-zgbk_zh_0116 \
714409434 ${DATA_ROOT}/mimic_iv_radiology_0116 \
372392667 ${DATA_ROOT}/qa-hdf_zh_0116 \
229738336 ${DATA_ROOT}/financial_book_0116 \
43378680 ${DATA_ROOT}/book-med_textbooks_zh_0116 \
3687546 ${DATA_ROOT}/medical_general_book_0116 \
30180226 ${DATA_ROOT}/usmle_step3_keyword_0116 \
360770670 ${DATA_ROOT}/prolognotebook_0116 \
124147516 ${DATA_ROOT}/other-out_law_0116 \
498465288 ${DATA_ROOT}/paper-medjournals_zh_0116 \
90234774 ${DATA_ROOT}/usmle_step2_keyword_0116 \
304652313 ${DATA_ROOT}/retrieval_corpus_msmarco_0116 \
31856004 ${DATA_ROOT}/webtext-socratic_0116 \
2440050 ${DATA_ROOT}/wjcc_case_0116 \
2814681816 ${DATA_ROOT}/medical_book_stage3_0116 \
2131477104 ${DATA_ROOT}/paper-arxiv_0116 \
917663133 ${DATA_ROOT}/webtext-mnvbc_news_zh_0116 \
1074795062 ${DATA_ROOT}/mimic_iv_discharge_0116 \
271347151 ${DATA_ROOT}/high_quality-36kr_v2_zh_0116 \
740963302 ${DATA_ROOT}/other-hupd_0116 \
1367788232 ${DATA_ROOT}/wanjuan_exam_0116 \
4683462582 ${DATA_ROOT}/paper-redpajama_arxiv_0116 \
1840086271 ${DATA_ROOT}/sft_fp_0116 \
1152149740 ${DATA_ROOT}/webtext-reddit_en_0116 \
2574565044 ${DATA_ROOT}/book-zlib_fb2_0116 \
1122559065 ${DATA_ROOT}/other-patents_en_0116 \
4820835342 ${DATA_ROOT}/webtext-homework_study_0116 \
552434378 ${DATA_ROOT}/qa-mnvbc_qa_zh_0116 \
19811795176 ${DATA_ROOT}/wiki-wiki-20230401_en_0116 \
1458187654 ${DATA_ROOT}/amps_mathematica_0116 \
2027293520 ${DATA_ROOT}/wiki-wudao_baike_zh_0116 \
3490133456 ${DATA_ROOT}/other-patents_zh_0116 \
4363229910 ${DATA_ROOT}/k12_0116 \
8649821652 ${DATA_ROOT}/wiki-baidu_baike_zh_0116 \
4277074533 ${DATA_ROOT}/webtext-fandom_0116 \
2966154309 ${DATA_ROOT}/qa-pile_0116 \
5273925384 ${DATA_ROOT}/journal-pubmed_0116 \
14079256858 ${DATA_ROOT}/book-zlib_mobi_0116 \
1812421409 ${DATA_ROOT}/webtext-quora_0116 \
13225595430 ${DATA_ROOT}/financial_news_0116 \
6227800516 ${DATA_ROOT}/webtext-mnvbc_webtext_zh_0116 \
7099633272 ${DATA_ROOT}/webtext-wudao_zh_0116 \
30548378840 ${DATA_ROOT}/paper-nssd_0116 \
17634419698 ${DATA_ROOT}/wiki-baidubaike_full_0116 \
38467579580 ${DATA_ROOT}/journal-pile_0116 \
51418891548 ${DATA_ROOT}/book-libgen_rs_0116 \
13116516221 ${DATA_ROOT}/reddit_dialog_0116 \
58525596716 ${DATA_ROOT}/book-zlib_pdf_0116 \
17156281329 ${DATA_ROOT}/webtext-c4_zh_0116 \
11857559056 ${DATA_ROOT}/webtext-clue_zh_0116 \
68871964210 ${DATA_ROOT}/book-zlib3_0116 \
19509299157 ${DATA_ROOT}/code-starcoderdata_gt5star_0116 \
28735790165 ${DATA_ROOT}/webtext-skypile-150b_0116 \
34212349327 ${DATA_ROOT}/webtext-cc_202314_zh_0116 \
34899221067 ${DATA_ROOT}/webtext-yayi2_0116 \
171706962176 ${DATA_ROOT}/paper-scihub_pdf_0116 \
22358220032 ${DATA_ROOT}/webtext-zhihu_qa_0116 \
148004850842 ${DATA_ROOT}/book-zlib_epub_0116 \
58776054945 ${DATA_ROOT}/webtext-cc_202314_en_0116 \
57337882005 ${DATA_ROOT}/webtext-casia_chinesewebtext_0116 \
69248355763 ${DATA_ROOT}/webtext-toutiao_articles_0116 \
68260496286 ${DATA_ROOT}/webtext-cc_202249_0116 \
53199116782 ${DATA_ROOT}/webtext-c4_en_0116 \
199449722820 ${DATA_ROOT}/code-starcoderdata_0116 \
199181777028 ${DATA_ROOT}/webtext-weixin_page_0116 \
286278573775 ${DATA_ROOT}/webtext-refinedweb_0116 \
1456590888141 ${DATA_ROOT}/webtext-redpajama_v2_0116"


TRAIN_TOKENS=3353760169687    # 训练总token数
LR_DECAY_TOKENS=3353760169687 # 学习率decay的范围，1.0T tokens
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

JOB_NAME="dmoe_LLaMA_1.3Bx8_top2_3.3t"
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
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --log-interval 10 \
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
        --initial-loss-scale 65536 \
        --use-flash-attn"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16 \
        --use-flash-attn"
fi

#Moe的配置
MOE_ARGUMENTS=" \
    --moe-num-experts=8 \
    --moe-loss-weight=0.01 \
    --moe-top-k=2 \
    --moe-capacity-factor 0 \
    --mlp_type=glu"

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
