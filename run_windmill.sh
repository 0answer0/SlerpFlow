#!/usr/bin/env bash
# Object replacement: windmill -> castle
set -e
source "$(dirname "$0")/env.sh"

# SlerpFlow (ours)
CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
    --source_img      'examples/source/windmill.png' \
    --source_prompt   'a white windmill in a blue sky' \
    --target_prompt   'a white castle in a blue sky' \
    --output          'examples/edit-result/windmill_slerp.jpg' \
    --strategy        slerp \
    --num_steps 15 --guidance 3.0 --inject 2 --slerp_t 0.5 --seed 42

# FireFlow baseline (uncomment to compare)
# CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
#     --source_img      'examples/source/windmill.png' \
#     --source_prompt   'a white windmill in a blue sky' \
#     --target_prompt   'a white castle in a blue sky' \
#     --output          'examples/edit-result/windmill_fireflow.jpg' \
#     --strategy        fireflow \
#     --num_steps 15 --guidance 3.0 --inject 2 --seed 42
