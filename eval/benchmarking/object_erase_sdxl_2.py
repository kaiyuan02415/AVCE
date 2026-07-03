import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
from safetensors.torch import load_file

from diffusers import StableDiffusionXLPipeline
from torchvision.models import resnet50, ResNet50_Weights

def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1244)
    parser.add_argument('--dbg', action='store_true')
    
    parser.add_argument('--va', action='store_true', help='Vanilla mode: Use native SDXL without loading external weights')
    parser.add_argument('--name', type=str, default=None, help="Custom folder name for results")
    
    parser.add_argument('--target', type=str, default=None, required=True, help="The concept name (e.g., 'golf ball')")
    parser.add_argument('--method', type=str, default='esd', help="Method name for folder structure")
    
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--removal_mode', type=str, default='erase', choices=['erase', 'keep'])
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--ckpt_name', type=str, required=False, help="Path to the ESD .safetensors file")
    
    parser.add_argument('--res_path', type=str, default='results') 
    
    args = parser.parse_args()
    
    if not args.va and args.ckpt_name is None:
        parser.error("The --ckpt_name argument is required unless --va is specified.")
        
    return args

class CustomDatasetErasure(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.prompts = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']

        self.prompts = [(self.prompts[i], self.seeds[i], concepts_to_remove) for i in range(len(self.prompts)) if concepts_to_remove.lower() == self.labels[i].lower()]
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
    
class CustomDatasetKeep(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.dataset = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']
        self.dataset = [(self.dataset[i], self.seeds[i], self.labels[i].lower()) for i in range(len(self.dataset)) if concepts_to_remove.lower() != self.labels[i].lower()]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx][0]
        seed = self.dataset[idx][1]
        label = self.dataset[idx][2].lower()
        return prompt, seed, label

def load_sdxl_esd_model(args):
    device = f"cuda:{args.gpu}"
    print(f"Loading SDXL Base from {args.model_id}...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    
    # [Modification] 关键修复：强制 VAE 使用 FP32
    # SDXL VAE 在 FP16 下极易溢出产生 NaN，这是 "invalid value" 的主要原因
    pipe.vae.to(dtype=torch.float32)
    print("Force VAE to float32 to prevent NaNs.")

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

    root_dir = 'results'
    model_type = 'sdxl'
    
    if args.va:
        method = 'va'
    else:
        method = args.method 
    
    if args.name is not None:
        folder_name = args.name
    else:
        folder_name = args.target.replace(' ', '_') 
    
    args.benchmarking_result_path = os.path.join(root_dir, model_type, method, folder_name)
    print("Benchmarking result path: ", args.benchmarking_result_path)
    
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    csv_path = 'datasets/imagenette.csv'
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Please ensure dataset exists.")
        return

    data = pd.read_csv(csv_path)
    if args.removal_mode == 'erase':
        dataloader = torch.utils.data.DataLoader(CustomDatasetErasure(data, args.target), batch_size=args.batch_size, shuffle=False)
    else:
        dataloader = torch.utils.data.DataLoader(CustomDatasetKeep(data, args.target), batch_size=args.batch_size, shuffle=False)

    print("Number of prompts: ", len(dataloader))

    remover_model = load_sdxl_esd_model(args)

    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)
    classifier = classifier.to(f"cuda:{args.gpu}")
    classifier.eval()
    preprocess = weights.transforms()

    avg_acc = 0
    for iter, (prompt, seed, label) in enumerate(dataloader):
        if iter >= 120: break
        if args.dbg and iter > 10: break

        print(f"Iter {iter}: Prompt: {prompt[0]}")
        prompt_text = prompt[0]
        label_text = label[0]
        current_seed = seed[0].item()
        
        generator = torch.Generator(device=f"cuda:{args.gpu}").manual_seed(current_seed)
        
        with torch.no_grad():
            removal_images = remover_model(
                prompt=prompt_text,
                generator=generator,
                num_inference_steps=30,
                output_type="pil"
            ).images
            
        for i, image in enumerate(removal_images):
            save_name = f"{args.removal_mode}_{iter * args.batch_size + i}.png"
            image.save(os.path.join(args.benchmarking_result_path, save_name))

        for i, image in enumerate(removal_images):
            image_tensor = preprocess(image).unsqueeze(0).to(f"cuda:{args.gpu}")
            with torch.no_grad():
                output = classifier(image_tensor)
                
            s, indices = torch.topk(output, 1)
            indices = indices.cpu().numpy()
            pred_labels = [weights.meta["categories"][idx] for idx in indices[0]]
            pred_labels = [l.lower() for l in pred_labels]
            print(f"  Pred: {pred_labels} | True: {label_text}")

            if label_text in pred_labels:
                avg_acc += 1

    print("Object predicted in: %d/%d images" % (avg_acc, len(dataloader)))
    acc_val = avg_acc / len(dataloader) if len(dataloader) > 0 else 0.0
    print(f"Average accuracy: {acc_val}")
    
    results = {"average_accuracy": acc_val}
    
    if args.va:
        ckpt_tag = "vanilla"
    else:
        ckpt_tag = os.path.basename(args.ckpt_name).split('.')[0]
    
    result_json_name = f"results_{args.removal_mode}_{ckpt_tag}.json"
    
    with open(os.path.join(args.benchmarking_result_path, result_json_name), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()