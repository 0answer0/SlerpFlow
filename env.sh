#!/usr/bin/env bash
# Model paths / cache for SlerpFlow. Source this before running the run_*.sh
# scripts:  source env.sh
#
# Optional: uncomment and edit the two paths below to use local FLUX.1-dev
# weights. If left unset, the weights are downloaded from HuggingFace
# (black-forest-labs/FLUX.1-dev) on first run instead.

# export FLUX_DEV=/path/to/flux1-dev.safetensors
# export AE=/path/to/ae.safetensors

# Optional: uncomment and edit these only when running with a local HuggingFace
# cache or in a fully offline environment.
# export HF_HOME=/path/to/.cache/huggingface
# export HF_HUB_CACHE=/path/to/.cache/huggingface/hub
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
