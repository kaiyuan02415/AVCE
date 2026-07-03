import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from math import ceil
from argparse import ArgumentParser

sys.path.append(os.getcwd())

from utils import load_models
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dbg', action='store_true', help='if True, only processes a few batches for debugging')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Path to your pruned/unlearned model checkpoint')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--benchmarking_result_path', type=str, default='results/results_seed_0/stable-diffusion/')
    parser.add_argument('--batch_size', type=int, default=16)
    # prompt 文件路径，每行一个 prompt
    parser.add_argument('--prompt_file', type=str, default='datasets/coco_prompts.txt',
                        help='Path to text file, each line is a prompt')
    # 新增参数：限制 prompt 数量
    parser.add_argument('--max_prompts', type=int, default=None,
                        help='Max number of prompts to evaluate')
    return parser.parse_args()

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    # 构造结果保存目录
    args.benchmarking_result_path = os.path.join(
        args.benchmarking_result_path, args.model_id, args.target, args.baseline, 'benchmarking', 'concept_erase'
    )
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    # 1. 从文本文件中读取所有 prompt
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(all_prompts)} prompts from {args.prompt_file}")

    # 如果设置了 max_prompts 参数，则截取前 max_prompts 条
    if args.max_prompts is not None:
        all_prompts = all_prompts[:args.max_prompts]
        print(f"Using first {args.max_prompts} prompts for evaluation.")

    # 2. 加载剪枝后的模型
    remover_model = load_models(args, args.ckpt_name)

    # 3. 加载 CLIP 模型和相关处理器
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        remover_model.to(f'cuda:{args.gpu}')
        clip_model.to(f'cuda:{args.gpu}')

    # 4. 分批处理 prompt
    num_prompts = len(all_prompts)
    num_batches = ceil(num_prompts / args.batch_size)
    average_sim = 0.0
    count_samples = 0

    for batch_idx in range(num_batches):
        if args.dbg and batch_idx > 2:
            break

        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
        prompt_batch = all_prompts[start_idx:end_idx]

        # 检查是否已存在生成的图片
        skip_this_batch = False
        out_img_path = os.path.join(args.benchmarking_result_path, f"removed_{start_idx}.jpg")
        if os.path.exists(out_img_path):
            print(f"Skipping batch {batch_idx}, found existing file {out_img_path}")
            skip_this_batch = True

        if not skip_this_batch:
            print(f"Generating images for batch {batch_idx}, prompt range [{start_idx}, {end_idx})")
            torch.manual_seed(0)
            np.random.seed(0)
            images = remover_model(prompt_batch).images
            for i, image in enumerate(images):
                global_idx = start_idx + i
                out_path = os.path.join(args.benchmarking_result_path, f"removed_{global_idx}.jpg")
                image.save(out_path)
        else:
            images = []
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                img_path = os.path.join(args.benchmarking_result_path, f"removed_{global_idx}.jpg")
                image = Image.open(img_path).convert("RGB")
                images.append(image)

        # 5. 计算 CLIP 相似度
        for i, image in enumerate(images):
            inputs = processor(images=image, return_tensors="pt").to(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
            image_features = clip_model.get_image_features(**inputs)
            text_input = tokenizer(prompt_batch[i], return_tensors="pt", padding=True).to(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
            text_features = clip_model.get_text_features(**text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T).item()
            average_sim += sim
            count_samples += 1
            print(f"Prompt: {prompt_batch[i]}")
            print(f"Similarity: {sim:.4f}")

    if count_samples == 0:
        avg_score = 0.0
    else:
        avg_score = average_sim / count_samples
    print(f"Average similarity across {count_samples} samples = {avg_score:.4f}")

    results = {"average_similarity": avg_score, "count_samples": count_samples}
    p = 'all_coco'
    out_json_path = os.path.join(args.benchmarking_result_path, f'{p}_results.json')
    print("Saving results to ", out_json_path)
    with open(out_json_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
