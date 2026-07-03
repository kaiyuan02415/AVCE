import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from math import ceil
from argparse import ArgumentParser

from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--dbg', action='store_true', help='If True, only processes a few batches for debugging')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Base Stable Diffusion model ID for inference')
    parser.add_argument('--benchmarking_result_path', type=str,
                        default='results/results_seed_0/stable-diffusion/',
                        help='Root directory to save benchmarking outputs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prompt_file', type=str, default='datasets/coco_prompts.txt',
                        help='Path to text file; each line is a prompt')
    parser.add_argument('--max_prompts', type=int, default=None,
                        help='Max number of prompts to evaluate')
    return parser.parse_args()


def main():
    args = input_args()
    print("Arguments:", args.__dict__)

    # Prepare result directory
    safe_model_id = args.model_id.replace('/', '_')
    result_dir = os.path.join(args.benchmarking_result_path, safe_model_id)
    print("Benchmarking result path:", result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # Load prompts
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(all_prompts)} prompts from {args.prompt_file}")
    if args.max_prompts:
        all_prompts = all_prompts[:args.max_prompts]
        print(f"Using first {len(all_prompts)} prompts for evaluation.")

    # Load base Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        pipeline.to(f'cuda:{args.gpu}')
    pipeline.enable_attention_slicing()

    # Load CLIP for similarity
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if torch.cuda.is_available():
        clip_model.to(f'cuda:{args.gpu}')

    # Batch-wise inference and evaluation
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

        # Generate or load images
        images = []
        for i, prompt in enumerate(prompt_batch):
            global_idx = start_idx + i
            out_path = os.path.join(result_dir, f"base_{global_idx}.jpg")
            if os.path.exists(out_path):
                img = Image.open(out_path).convert("RGB")
            else:
                print(f"Generating image for prompt index {global_idx}")
                generator = torch.Generator(device=f'cuda:{args.gpu}') if torch.cuda.is_available() else None
                output = pipeline(prompt, generator=generator)
                img = output.images[0]
                img.save(out_path)
            images.append(img)

        # Compute CLIP similarity
        for i, image in enumerate(images):
            prompt = prompt_batch[i]
            inputs = processor(images=image, return_tensors="pt").to(
                f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
            )
            image_features = clip_model.get_image_features(**inputs)
            text_input = tokenizer(prompt, return_tensors="pt", padding=True).to(
                f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
            )
            text_features = clip_model.get_text_features(**text_input)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T).item()

            average_sim += sim
            count_samples += 1
            print(f"Prompt: {prompt}\nSimilarity: {sim:.4f}\n")

    # Summary
    avg_score = average_sim / count_samples if count_samples else 0.0
    print(f"Average similarity across {count_samples} samples = {avg_score:.4f}")
    results = {"average_similarity": avg_score, "count_samples": count_samples}
    out_json = os.path.join(result_dir, 'base_model_results.json')
    print("Saving results to", out_json)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
