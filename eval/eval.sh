#!/usr/bin/env bash
# Example evaluation commands. Replace $CKPT with your trained AVCE checkpoint
# (a .safetensors file of edited attn2 to_k/to_v weights produced by train_avce.py).
set -e
cd "$(dirname "$0")"   # -> eval/

CKPT="/path/to/avce_checkpoint.safetensors"

# ----------------------------------------------------------------------
# 1. Object unlearning — Unlearn Acc (pretrained ResNet50 classifier)
# ----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python benchmarking/object_erase_sdxl.py \
  --target "golf ball" \
  --method "avce" \
  --removal_mode erase \
  --ckpt_name "$CKPT" \
  --gpu 0 \
  --name "avce_golf_ball"

# ----------------------------------------------------------------------
# 2. Artistic-style unlearning — CLIP_a / CLIP_c / CLIP_d
# ----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python benchmarking/artist_erasure_s_sdxl.py \
  --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
  --ckpt_name "$CKPT" \
  --dataset_path datasets/art_5.csv \
  --res_path results/art-5/avce \
  --gpu 0 \
  --is_sdxl

# ----------------------------------------------------------------------
# 3. Explicit-content unlearning — NudeNet detections on I2P
# ----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python benchmarking/nudity_eval_only_sdxl.py \
  --eval_dataset i2p \
  --ckpt_path "$CKPT" \
  --dir results/nude \
  --extra_dir avce \
  --gpu 0

# ----------------------------------------------------------------------
# 4. Generation quality — MS-COCO CLIP score / FID
# ----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python benchmarking/eval_coco_clip_sdxl.py \
  --ckpt_name "$CKPT" \
  --target "coco-avce" \
  --method "avce" \
  --prompt_file "datasets/coco_prompts.txt" \
  --max_prompts 200

# ----------------------------------------------------------------------
# 5. Latent-space auditing — Concept Retrieval Score (CRS) / Confidence (CCS)
# ----------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python audit/cce.py \
  --model_path "$CKPT" \
  --target_concept "parachute"

# ----------------------------------------------------------------------
# 6. Robustness attacks (ASR1/2/3) — see eval/attacks/README.md
# ----------------------------------------------------------------------
# Ring-A-Bell (ASR1):
#   python attacks/ring-a-bell/ring_attack_s.py --model_type sdxl --model_path "$CKPT" ...
# UnlearnDiffAtk (ASR2): see attacks/unlearndiffatk/
# P4D (ASR3):            see attacks/p4d/
