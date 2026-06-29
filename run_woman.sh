#!/usr/bin/env bash
# Color attribute: red hat -> green hat
set -e
source "$(dirname "$0")/env.sh"

# SlerpFlow (ours)
CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
    --source_img      'examples/source/woman.png' \
    --source_prompt   'a woman wearing a red hat and a red dress' \
    --target_prompt   'a woman wearing a green hat and a red dress' \
    --output          'examples/edit-result/woman_slerp.jpg' \
    --strategy        slerp \
    --num_steps 15 --guidance 3.0 --inject 2 --slerp_t 0.5 --seed 42

# FireFlow baseline (uncomment to compare)
# CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
#     --source_img      'examples/source/woman.png' \
#     --source_prompt   'a woman wearing a red hat and a red dress' \
#     --target_prompt   'a woman wearing a green hat and a red dress' \
#     --output          'examples/edit-result/woman_fireflow.jpg' \
#     --strategy        fireflow \
#     --num_steps 15 --guidance 3.0 --inject 2 --seed 42
