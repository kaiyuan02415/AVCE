#!/usr/bin/env bash
# Multi-concept object unlearning on SDXL (10-class Imagenette subset).
# Reproduces the "mc-img-10" setting used in the paper.
set -e
cd "$(dirname "$0")/.."   # -> avce/

CUDA_VISIBLE_DEVICES=0 python train_avce.py \
  --erase_concept "parachute, golf ball, garbage truck, cassette player, church, tench, english springer, french horn, chain saw, gas pump" \
  --concept_class "img-10" \
  --concept_type "object" \
  --device "cuda:0" \
  --esd_method "esd-x-strict" \
  --iterations 100 \
  --lr 2e-5 \
  --negative_guidance 0.4 \
  --lambda_attn 0.0 \
  --adv --adv_steps 5 --adv_lr 2e-4 \
  --dts \
  --concept_batch 3 \
  --mus_p 2 --mus_alpha_min 0.8 --mus_entropy_samples 20 \
  --save_path "checkpoints/"

# Single-concept variant (e.g. "golf ball"): set --concept_batch 1 and pass one concept.
