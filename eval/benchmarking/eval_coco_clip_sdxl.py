import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from math import ceil
from argparse import ArgumentParser
from safetensors.torch import load_file

from diffusers import StableDiffusionXLPipeline
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dbg', action='store_true')
    
    # [Modification] 新增 va 参数
    parser.add_argument('--va', action='store_true', help='Vanilla mode: Use native SDXL without loading external weights')

    # 核心路径参数
    parser.add_argument('--target', type=str, default=None, help="The concept name used for folder structure")
    parser.add_argument('--method', type=str, default='esd', help="Method name for folder structure")
    
    parser.add_argument('--baseline', type=str, default=None)
    
    # [Modification] 将 required=True 改为 False，稍后手动校验
    parser.add_argument('--ckpt_name', type=str, default=None, required=False, help='Path to your ESD .safetensors file')
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prompt_file', type=str, default='datasets/coco_prompts.txt')
    parser.add_argument('--max_prompts', type=int, default=None)
    
    parser.add_argument('--benchmarking_result_path', type=str, default='results')
    
    args = parser.parse_args()
    
    # [Modification] 校验逻辑：如果不是 va 模式，必须提供 ckpt_name
    if not args.va and args.ckpt_name is None:
        parser.error("The --ckpt_name argument is required unless --va is specified.")
        
    return args

def load_sdxl_esd_model(args, device):
    print(f"Loading SDXL Base from {args.model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float16, 
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    # [Modification] 如果是 va 模式，直接跳过加载外部权重
    if args.va:
        print("Vanilla mode enabled. Using native SDXL weights.")
    elif args.ckpt_name:
        print(f"Loading ESD weights from {args.ckpt_name}...")
        esd_state_dict = load_file(args.ckpt_name)
        pipe.unet.load_state_dict(esd_state_dict, strict=False)
        print(f"ESD weights loaded into UNet.")
    
    return pipe

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    # ================= 路径修改核心逻辑 =================
    # 统一格式: results/sdxl/[method]/[target_name]
    
    root_dir = 'results'
    model_type = 'sdxl'
    
    # [Modification] 如果开启 va 模式，method 强制为 'va'
    if args.va:
        method = 'va'
    else:
        method = args.method
    
    if args.target:
        target_name = args.target.replace(' ', '_')
    else:
        target_name = 'coco_global'

    args.benchmarking_result_path = os.path.join(root_dir, model_type, method, target_name)
    print("Benchmarking result path: ", args.benchmarking_result_path)
    
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)
    # ==================================================

    if not os.path.exists(args.prompt_file):
        print(f"Warning: {args.prompt_file} not found. Using dummy prompts.")
        all_prompts = ["a photo of a cat", "a car on the street", "a delicious burger"]
    else:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(all_prompts)} prompts.")

    if args.max_prompts is not None:
        all_prompts = all_prompts[:args.max_prompts]

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    remover_model = load_sdxl_esd_model(args, device)

    print("Loading CLIP for metric calculation...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    num_prompts = len(all_prompts)
    num_batches = ceil(num_prompts / args.batch_size)
    average_sim = 0.0
    count_samples = 0

    for batch_idx in range(num_batches):
        if args.dbg and batch_idx > 2: break

        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, num_prompts)
        prompt_batch = all_prompts[start_idx:end_idx]

        # 图片保存路径
        out_img_path_first = os.path.join(args.benchmarking_result_path, f"coco_val_{start_idx}.jpg")
        
        images = []
        if os.path.exists(out_img_path_first):
            print(f"Skipping batch {batch_idx}, loading images...")
            for i in range(end_idx - start_idx):
                img_path = os.path.join(args.benchmarking_result_path, f"coco_val_{start_idx + i}.jpg")
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                except: pass
        else:
            generator = torch.Generator(device=device).manual_seed(0)
            with torch.no_grad():
                output = remover_model(
                    prompt=prompt_batch, 
                    generator=generator,
                    num_inference_steps=50,
                    output_type="pil"
                )
            images = output.images
            for i, image in enumerate(images):
                image.save(os.path.join(args.benchmarking_result_path, f"coco_val_{start_idx + i}.jpg"))

        if len(images) > 0:
            with torch.no_grad():
                for i, image in enumerate(images):
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    image_features = clip_model.get_image_features(**inputs)
                    text_input = tokenizer(prompt_batch[i], return_tensors="pt", padding=True).to(device)
                    text_features = clip_model.get_text_features(**text_input)
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    sim = (image_features @ text_features.T).item()
                    average_sim += sim
                    count_samples += 1

    avg_score = average_sim / count_samples if count_samples > 0 else 0.0
    print(f"Average similarity: {avg_score:.4f}")

    results = {"average_similarity": avg_score, "count_samples": count_samples}
    
    # [Modification] 处理结果文件名
    if args.va:
        ckpt_tag = "vanilla"
    else:
        ckpt_tag = os.path.basename(args.ckpt_name).split('.')[0]
    
    out_json_path = os.path.join(args.benchmarking_result_path, f'coco_results_{ckpt_tag}.json')
    with open(out_json_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()