#!/usr/bin/env python
# batch_eval_by_class_subset.py

import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

import sys
sys.path.append(os.getcwd())
# from utils import load_models

# —— 如果你已有 utils.load_models，请取消下面这行注释 ——
# from utils import load_models

from diffusers import StableDiffusionPipeline
import torch

def load_models(model_id: str, ckpt_path: str, device="cuda", dtype=torch.float16):
    """
    加载基础 Stable Diffusion pipeline，并把给定 .pt 权重灌进 UNet。
    model_id: diffusers hub 模型，比如 "runwayml/stable-diffusion-v1-5"
    ckpt_path: 你的擦除/编辑权重路径（.pt）
    """
    # 1. 加载基础模型
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

    # 2. 读取权重
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 3. 尝试直接载入 UNet
    try:
        missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
        print(f"[Info] Loaded ckpt into UNet. missing={len(missing)}, unexpected={len(unexpected)}")
    except Exception as e:
        print("[Warn] Direct load_state_dict failed:", e)
        # 兜底：可能需要 key 转换（CompVis → diffusers）
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint
        converted = convert_ldm_unet_checkpoint(state, pipe.unet.config, path=ckpt_path, extract_ema=True)
        missing, unexpected = pipe.unet.load_state_dict(converted, strict=False)
        print(f"[Info] Converted ckpt and loaded into UNet. missing={len(missing)}, unexpected={len(unexpected)}")

    return pipe

def parse_args():
    parser = ArgumentParser(description="Batch evaluation over art_50.csv with optional class filtering")
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
        default='runwayml/stable-diffusion-v1-5',
        help="原始 Stable Diffusion 模型 ID"
    )
    parser.add_argument(
        '--ckpt_name',
        type=str,
        required=True,
        help="概念去除模型的检查点路径"
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

    # —— 根据传入的 classes 参数进行过滤 ——
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

    # —— 加载模型 ——
    base_model = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(device)
    # remover_model = load_models(args, args.ckpt_name).to(device)
    remover_model = load_models(args.model_id, args.ckpt_name, device=device, dtype=torch.float16)

    # —— 加载 CLIP ——
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    records = []

    for idx, row in df.iterrows():
        prompt = row['prompt']
        seed   = int(row['evaluation_seed'])
        cls    = str(row['class'])

        # 为每个类别创建子目录
        class_dir = os.path.join(args.res_path, cls)
        os.makedirs(class_dir, exist_ok=True)

        # 生成去除模型的图像
        torch.manual_seed(seed)
        np.random.seed(seed)
        img_removed = remover_model(prompt).images[0]

        # 生成原始模型的图像
        torch.manual_seed(seed)
        np.random.seed(seed)
        img_original = base_model(prompt).images[0]

        # CLIP 特征提取
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

    # 汇总每类指标
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
    print("\n分组评估完成，结果已保存：")
    print(f"  • {args.res_path}/metrics_by_class.csv")
    print(f"  • {args.res_path}/metrics_by_class.json")
    print("\n各类指标一览：")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
