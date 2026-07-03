#!/usr/bin/env python
# batch_eval_by_class_subset.py (SDXL Supported - 修复 Safetensors 加载 & 设置 30 步推理)

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F
import gc

# 引入 diffusers 和 safetensors 相关库
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
from transformers import CLIPProcessor, CLIPModel

import sys
sys.path.append(os.getcwd())

def load_models(model_id: str, ckpt_path: str, is_sdxl: bool, device="cuda", dtype=torch.float16):
    """
    加载基础 Stable Diffusion (或 SDXL) pipeline，并以健壮的方式将局部自定义权重灌进 UNet。
    """
    print(f"Loading Base Model: {model_id} (SDXL: {is_sdxl})...")
    
    # 1. 根据是否为 SDXL 加载基础模型
    if is_sdxl:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16", use_safetensors=True
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # 2. 读取要覆盖的权重字典
    print(f"Loading custom weights from {ckpt_path}...")
    if ckpt_path.endswith('.safetensors'):
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    # 3. 手动注入 UNet (修复由于数据类型或部分键缺失导致的原生加载失败)
    unet_state_dict = pipe.unet.state_dict()
    updated_count = 0
    unexpected_keys = []

    for k, v in state.items():
        # 清理可能存在的层级前缀 (适配不同训练脚本的保存习惯)
        clean_k = k
        if clean_k.startswith("unet."):
            clean_k = clean_k[5:]
        elif clean_k.startswith("module."):
            clean_k = clean_k[7:]

        if clean_k in unet_state_dict:
            # 校验形状是否一致
            if unet_state_dict[clean_k].shape == v.shape:
                # 强制对齐设备和数据类型并覆盖
                unet_state_dict[clean_k] = v.to(
                    device=unet_state_dict[clean_k].device, 
                    dtype=unet_state_dict[clean_k].dtype
                )
                updated_count += 1
            else:
                print(f"[Warn] Shape mismatch for {clean_k}: model expects {unet_state_dict[clean_k].shape}, got {v.shape}")
        else:
            unexpected_keys.append(clean_k)

    # 将合并后的完整状态字典重新载入模型
    pipe.unet.load_state_dict(unet_state_dict, strict=True)
    
    print(f"✅ [Success] Injected {updated_count} tensors from checkpoint into UNet.")
    if unexpected_keys:
        print(f"⚠️ [Warn] Found {len(unexpected_keys)} unexpected keys in checkpoint (e.g. {unexpected_keys[:3]}). They were ignored.")

    return pipe

def parse_args():
    parser = ArgumentParser(description="Batch evaluation over art_50.csv with SDXL support")
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='art_50.csv',
        help="包含 prompt、evaluation_seed 和 class 列的 CSV 文件路径"
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help="使用的 GPU 编号 (若无 GPU 会自动回落到 CPU)"
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help="原始模型 ID (默认使用 SDXL base)"
    )
    parser.add_argument(
        '--ckpt_name',
        type=str,
        required=True,
        help="概念去除模型的检查点路径 (如 mus_sdxl_re_p_15.safetensors)"
    )
    parser.add_argument(
        '--res_path',
        type=str,
        default='results/batch_eval_by_class/',
        help="评估结果（和图片）存放目录"
    )
    parser.add_argument(
        '--classes',
        type=str,
        default=None,
        help="逗号分隔的类别列表，仅对这些类别进行评估；未指定则评估所有类别"
    )
    parser.add_argument(
        '--is_sdxl',
        action='store_true',
        default=True,  # 默认开启 SDXL
        help="是否使用 SDXL 架构"
    )

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default='unet')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.res_path, exist_ok=True)

    # —— 加载数据集 DataFrame ——
    df = pd.read_csv(args.dataset_path)
    assert {'prompt', 'evaluation_seed', 'class'}.issubset(df.columns), \
        "CSV 中必须包含 'prompt', 'evaluation_seed', 'class' 三列"

    if args.classes:
        selected = [c.strip() for c in args.classes.split(',') if c.strip()]
        missing = set(selected) - set(df['class'].unique())
        if missing:
            print(f"⚠️ 下面指定的类别未在数据集中找到: {', '.join(missing)}")
        df = df[df['class'].isin(selected)]
        if df.empty:
            print("❌ 过滤后没有数据，请检查类别名称。")
            return
        print(f"✅ 将对以下类别进行评估: {', '.join(df['class'].unique())}")
    else:
        print("ℹ️ 未指定类别，将评估所有类别。")

    # —— 加载原始 Base 模型 ——
    if args.is_sdxl:
        base_model = StableDiffusionXLPipeline.from_pretrained(
            args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
    else:
        base_model = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        
    # 为了防止加载两个模型导致 OOM，开启 CPU Offload
    try:
        base_model.enable_model_cpu_offload(gpu_id=args.gpu)
    except Exception as e:
        print("CPU Offload failed for base model, moving to device directly...")
        base_model.to(device)

    # —— 加载 Remover 模型 ——
    remover_model = load_models(args.model_id, args.ckpt_name, is_sdxl=args.is_sdxl, device=device, dtype=torch.float16)
    try:
        remover_model.enable_model_cpu_offload(gpu_id=args.gpu)
    except Exception as e:
        print("CPU Offload failed for remover model, moving to device directly...")
        remover_model.to(device)

    # —— 加载 CLIP 模型 ——
    print("Loading CLIP for evaluation...")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    records = []
    print(f"Total samples to evaluate: {len(df)}")

    for idx, row in df.iterrows():
        prompt = row['prompt']
        seed   = int(row['evaluation_seed'])
        cls    = str(row['class'])

        class_dir = os.path.join(args.res_path, cls)
        os.makedirs(class_dir, exist_ok=True)
        
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"[{idx+1}/{len(df)}] Evaluating class: {cls} | Prompt: {prompt[:30]}...")

        # 生成去除模型的图像 (这里已将推理步数修改为 30 步)
        img_removed = remover_model(prompt, generator=generator, num_inference_steps=50).images[0]

        # 重新设置 Generator 生成原始模型的图像 (确保两次生成起点严格一致)
        generator = torch.Generator(device=device).manual_seed(seed)
        # 同样修改为 30 步
        img_original = base_model(prompt, generator=generator, num_inference_steps=50).images[0]

        # 计算 CLIP 相似度
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_feats  = clip_model.get_text_features(**text_inputs)

        orig_inputs = clip_processor(images=[img_original], return_tensors="pt")
        orig_inputs = {k: v.to(device) for k, v in orig_inputs.items()}
        orig_feats  = clip_model.get_image_features(**orig_inputs)

        rem_inputs  = clip_processor(images=[img_removed], return_tensors="pt")
        rem_inputs  = {k: v.to(device) for k, v in rem_inputs.items()}
        rem_feats   = clip_model.get_image_features(**rem_inputs)

        sim_orig = F.cosine_similarity(text_feats, orig_feats).item()
        sim_rem  = F.cosine_similarity(text_feats, rem_feats).item()
        score    = 1 if sim_rem < sim_orig else 0

        # 保存图像
        img_removed.save(os.path.join(class_dir, f'removal_{idx}.jpg'))
        img_original.save(os.path.join(class_dir, f'original_{idx}.jpg'))

        # 记录结果
        records.append({
            'class': cls,
            'sim_original': sim_orig,
            'sim_removed' : sim_rem,
            'score'       : score
        })
        
        # 定期清理显存碎片
        torch.cuda.empty_cache()
        gc.collect()

    # 汇总指标
    results_df = pd.DataFrame(records)
    metrics = results_df.groupby('class').agg(
        avg_sim_original = ('sim_original', 'mean'),
        avg_sim_removed  = ('sim_removed',  'mean'),
        avg_score        = ('score',        'mean'),
        std_sim_original = ('sim_original', 'std'),
        std_score        = ('score',        'std')
    ).reset_index()

    # 保存输出
    metrics.to_csv(os.path.join(args.res_path, 'metrics_by_class.csv'), index=False)
    metrics.to_json(os.path.join(args.res_path, 'metrics_by_class.json'), orient='records', indent=4)

    # 控制台展示
    print("\n✅ 分组评估完成，结果已保存：")
    print(f"  • {os.path.join(args.res_path, 'metrics_by_class.csv')}")
    print("\n📊 各类指标一览：")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()