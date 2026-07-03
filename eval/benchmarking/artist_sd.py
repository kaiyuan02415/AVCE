import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
sys.path.append(os.getcwd())
from benchmarking_utils import set_benchmarking_path
from diffusers import StableDiffusionPipeline

# CLIP for similarity
from transformers import CLIPProcessor, CLIPModel


def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dbg', type=bool, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--hook_module', type=str, default='unet')
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion/')
    return parser.parse_args()


def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    # benchmarking path
    args.benchmarking_result_path = set_benchmarking_path(args)
    args.benchmarking_result_path = os.path.join(
        args.benchmarking_result_path,
        args.model_id,
        args.target,
        args.baseline,
        'benchmarking',
        'clip_scores'
    )
    print("Benchmarking result path: ", args.benchmarking_result_path)
    os.makedirs(args.benchmarking_result_path, exist_ok=True)

    # load prompts CSV
    df = pd.read_csv(f'datasets/test_{args.target}.csv')
    prompts = df['prompt'].tolist()
    seeds   = df['evaluation_seed'].tolist()

    # load base model
    model = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(args.gpu)

    # prepare CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.gpu)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    similarities = []

    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        print(f"[{i+1}/{len(prompts)}] Prompt: {prompt}, Seed: {seed}")
        # set seed
        print("seed:", seed)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        # generate image
        image = model(prompt).images[0]
        img_path = os.path.join(
            args.benchmarking_result_path, f"generated_{i}.png"
        )
        image.save(img_path)

        # compute CLIP similarity
        text_inputs = clip_processor(prompt, return_tensors="pt", padding=True).to(args.gpu)
        text_feats = clip_model.get_text_features(**text_inputs)
        image_inputs = clip_processor(images=image, return_tensors="pt").to(args.gpu)
        img_feats = clip_model.get_image_features(**image_inputs)
        sim = torch.nn.functional.cosine_similarity(
            text_feats, img_feats
        ).item()
        print("  CLIP similarity:", sim)
        similarities.append(sim)

    # aggregate
    avg_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    print(f"Average CLIP similarity: {avg_sim}, Std: {std_sim}")

    # save results
    results = {
        'avg_similarity': avg_sim,
        'std_similarity': std_sim
    }
    tag = args.model_id.replace('/', '_')
    with open(os.path.join(
        args.benchmarking_result_path,
        f'clip_results_{tag}.json'
    ), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
