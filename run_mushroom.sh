#!/usr/bin/env bash
# Object replacement: mushrooms -> flowers
set -e
source "$(dirname "$0")/env.sh"

# SlerpFlow (ours)
CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
    --source_img      'examples/source/mushroom.png' \
    --source_prompt   'mushrooms on a branch in the forest' \
    --target_prompt   'flowers on a branch in the forest' \
    --output          'examples/edit-result/mushroom_slerp.jpg' \
    --strategy        slerp \
    --num_steps 15 --guidance 3.0 --inject 2 --slerp_t 0.5 --seed 42

# FireFlow baseline (uncomment to compare)
# CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
#     --source_img      'examples/source/mushroom.png' \
#     --source_prompt   'mushrooms on a branch in the forest' \
#     --target_prompt   'flowers on a branch in the forest' \
#     --output          'examples/edit-result/mushroom_fireflow.jpg' \
#     --strategy        fireflow \
#     --num_steps 15 --guidance 3.0 --inject 2 --seed 42
