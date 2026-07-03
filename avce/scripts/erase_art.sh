#!/usr/bin/env bash
# Artistic-style unlearning on SDXL (5 styles).
set -e
cd "$(dirname "$0")/.."   # -> avce/

CUDA_VISIBLE_DEVICES=0 python train_avce.py \
  --erase_concept "Van Gogh, Monet, Pablo Picasso, Leonardo Da Vinci, Salvador Dali" \
  --concept_class "art-5" \
  --concept_type "art" \
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
