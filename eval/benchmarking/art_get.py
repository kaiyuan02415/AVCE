#!/usr/bin/env python
# generate_davinci.py — 用三个擦除模型分别生成 Mona Lisa & The Last Supper，每个模型每主题 5 张

import os
import torch
import gc
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file


# ──────────────────── 配置区 ────────────────────
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

CKPT_PATHS = [
    "/home/data1/Kaiyuan/DFM/uce_2025/uce_models/uce-art5-sdxl.safetensors",
    "/home/data1/Kaiyuan/DFM/AVCE/esd-models/sdxl/art-5_esd-x-strict_iter100_lr2e-05_attn0.1_Adv_steps5_adv_lr2e-04_DTS_wMUS.safetensors",
    "/home/data1/Kaiyuan/DFM/AVCE/esd-models/sdxl/esd/esd-art5/esd-multi_concepts_5_items-from-none-esdx.safetensors",
]

MODEL_TAGS = [
    "uce-art5",
    "esd-x-strict-mus",
    "esd-multi-concepts",
]

NUM_IMAGES_PER_PROMPT = 5
NUM_INFERENCE_STEPS = 50
OUTPUT_ROOT = "results/davinci_generation"
GPU_ID = 0

# 两个主题，各配 5 个不同种子
SUBJECTS = {
    "mona_lisa": {
        "prompt": "A painting of the Mona Lisa by Leonardo Da Vinci",
        "seeds": [42, 123, 256, 512, 1024],
    },
    "last_supper": {
        "prompt": "A painting of The Last Supper by Leonardo Da Vinci",
        "seeds": [2048, 3090, 4096, 7777, 9999],
    },
}


# ──────────────────── 工具函数 ────────────────────
def load_erased_pipeline(model_id: str, ckpt_path: str, device: torch.device, dtype=torch.float16):
    """加载 SDXL base 并将擦除模型的权重注入 UNet。"""
    print(f"\n{'='*60}")
    print(f"Loading base pipeline: {model_id}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True
    )

    print(f"Loading checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    unet_sd = pipe.unet.state_dict()
    updated, skipped = 0, []

    for k, v in state.items():
        clean_k = k
        if clean_k.startswith("unet."):
            clean_k = clean_k[5:]
        elif clean_k.startswith("module."):
            clean_k = clean_k[7:]

        if clean_k in unet_sd:
            if unet_sd[clean_k].shape == v.shape:
                unet_sd[clean_k] = v.to(device=unet_sd[clean_k].device, dtype=unet_sd[clean_k].dtype)
                updated += 1
            else:
                print(f"  [Warn] Shape mismatch: {clean_k} "
                      f"(expect {unet_sd[clean_k].shape}, got {v.shape})")
        else:
            skipped.append(clean_k)

    pipe.unet.load_state_dict(unet_sd, strict=True)
    print(f"✅ Injected {updated} tensors into UNet.")
    if skipped:
        print(f"⚠️  {len(skipped)} unexpected keys ignored (e.g. {skipped[:3]})")

    return pipe


# ──────────────────── 主流程 ────────────────────
def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for ckpt_path, tag in zip(CKPT_PATHS, MODEL_TAGS):
        # ---- 加载模型 ----
        pipe = load_erased_pipeline(BASE_MODEL_ID, ckpt_path, device)
        try:
            pipe.enable_model_cpu_offload(gpu_id=GPU_ID)
        except Exception:
            print("CPU offload failed, moving entire pipeline to device...")
            pipe.to(device)

        # ---- 对每个主题生成 5 张 ----
        for subject_key, cfg in SUBJECTS.items():
            prompt = cfg["prompt"]
            seeds = cfg["seeds"]

            save_dir = os.path.join(OUTPUT_ROOT, tag, subject_key)
            os.makedirs(save_dir, exist_ok=True)

            for i, seed in enumerate(seeds):
                generator = torch.Generator(device=device).manual_seed(seed)

                print(f"  [{tag}/{subject_key}] ({i+1}/{NUM_IMAGES_PER_PROMPT}) seed={seed}")
                image = pipe(
                    prompt,
                    generator=generator,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                ).images[0]

                filename = f"{tag}_{subject_key}_{i+1}_seed{seed}.jpg"
                image.save(os.path.join(save_dir, filename))
                print(f"    -> saved: {os.path.join(save_dir, filename)}")

                torch.cuda.empty_cache()
                gc.collect()

        # ---- 释放当前模型，腾出显存给下一个 ----
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ [{tag}] Done. Images saved to {os.path.join(OUTPUT_ROOT, tag)}/\n")

    print(f"\n🎉 All done! Results in: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()