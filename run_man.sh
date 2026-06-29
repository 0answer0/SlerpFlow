#!/usr/bin/env bash
# Background replacement: trees -> city
set -e
source "$(dirname "$0")/env.sh"

# SlerpFlow (ours)
CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
    --source_img      'examples/source/man.png' \
    --source_prompt   'a man sitting on a rock with trees in the background' \
    --target_prompt   'a man sitting on a rock with city in the background' \
    --output          'examples/edit-result/man_slerp.jpg' \
    --strategy        slerp \
    --num_steps 15 --guidance 3.0 --inject 2 --slerp_t 0.5 --seed 42

# FireFlow baseline (uncomment to compare)
# CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
#     --source_img      'examples/source/man.png' \
#     --source_prompt   'a man sitting on a rock with trees in the background' \
#     --target_prompt   'a man sitting on a rock with city in the background' \
#     --output          'examples/edit-result/man_fireflow.jpg' \
#     --strategy        fireflow \
#     --num_steps 15 --guidance 3.0 --inject 2 --seed 42
