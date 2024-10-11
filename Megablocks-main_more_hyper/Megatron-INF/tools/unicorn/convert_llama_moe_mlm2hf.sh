#!/bin/bash
python main.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --megatron-path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/megatron/Megatron-Inf \
  --load-path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/megatron/Megatron-Inf/save/checkpoint/final_ver_moe/iter_0002000  \
  --save-path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/save_hf/test_final_moe \
  --model-name llama_moe \
  --template-name llama_moe \
  --print-checkpoint-structure \
  --target_params_dtype fp16
