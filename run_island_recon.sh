#!/usr/bin/env bash
# Inversion + reconstruction (no editing): target prompt = source prompt,
# inject 0, guidance 1. Measures how faithfully the sampler inverts an image.
set -e
source "$(dirname "$0")/env.sh"

# SlerpFlow (ours)
CUDA_VISIBLE_DEVICES=0 python run_slerp_edit.py \
    --source_img      'examples/source/island.png' \
    --source_prompt   'a small island with a tree' \
    --target_prompt   'a small island with a tree' \
    --output          'examples/recon-result/island_slerp_recon.jpg' \
    --strategy        slerp \
    --num_steps 15 --guidance 1.0 --inject 0 --slerp_t 0.5 --seed 42


