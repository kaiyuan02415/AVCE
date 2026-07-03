# import os
# import sys
# import json
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# from argparse import ArgumentParser

# # Ensure project root is in path for utils
# sys.path.append(os.getcwd())
# from utils import load_models
# from diffusers import StableDiffusionPipeline, UNet2DConditionModel
# from torchvision.models import resnet50, ResNet50_Weights
# from benchmarking_utils import set_benchmarking_path


# def input_args():
#     parser = ArgumentParser()
#     parser.add_argument('--gpu', type=int, default=0)
#     parser.add_argument('--seed', type=int, default=1244)
#     parser.add_argument('--dbg', action='store_true')
#     parser.add_argument('--target', type=str, default=None,
#                         help='Concept/class to remove or keep')
#     parser.add_argument('--baseline', type=str, default=None,
#                         help='Baseline identifier for results path')
#     parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion',
#                         help='Base results directory')
#     parser.add_argument('--removal_mode', type=str, default=None,
#                         choices=['erase', 'keep'],
#                         help='Whether to erase or keep the target concept')
#     parser.add_argument('--hook_module', type=str, default='unet',
#                         help='Module of the model to hook for editing')
#     parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5',
#                         help='Stable Diffusion model identifier')
#     parser.add_argument('--batch_size', type=int, default=1,
#                         help='Batch size for generation')
#     parser.add_argument('--ckpt_name', type=str, default=None,
#                         help='Path to the saved checkpoint for the edited model')
#     parser.add_argument('--dataset_csv', type=str,
#                         default='datasets/imagenet-mini-50.csv',
#                         help='Path to the prompt CSV file')
#     parser.add_argument('--vanilla',
#     action='store_true',
#     help='If set, ignore --ckpt_name and load the vanilla SD model via model_id')
#     return parser.parse_args()


# # Dataset class to test concept erasure
# class CustomDatasetErasure(torch.utils.data.Dataset):
#     def __init__(self, data: pd.DataFrame, concepts_to_remove: str):
#         self.prompts = data['prompt']
#         self.seeds = data['evaluation_seed']
#         try:
#             self.labels = data['class']
#         except KeyError:
#             self.labels = data['label_str']

#         # filter to only the target class
#         self.samples = [
#             (self.prompts[i], self.seeds[i], concepts_to_remove.lower())
#             for i in range(len(self.prompts))
#             if self.labels[i].lower() == concepts_to_remove.lower()
#         ]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         prompt, seed, label = self.samples[idx]
#         return prompt, seed, label


# # Dataset class to test concept keeping
# class CustomDatasetKeep(torch.utils.data.Dataset):
#     def __init__(self, data: pd.DataFrame, concepts_to_remove: str):
#         self.seeds = data['evaluation_seed']
#         try:
#             self.labels = data['class']
#             prompts = data['prompt']
#         except KeyError:
#             self.labels = data['label_str']
#             prompts = data['prompt']

#         # exclude the target class
#         self.samples = [
#             (prompts[i], self.seeds[i], self.labels[i].lower())
#             for i in range(len(prompts))
#             if self.labels[i].lower() != concepts_to_remove.lower()
#         ]
#         print(f"Total prompts after filtering: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         prompt, seed, label = self.samples[idx]
#         return prompt, seed, label


# def main():
#     args = input_args()
#     print("Arguments:\n", args)

#     # setup benchmarking path
#     args.benchmarking_result_path = set_benchmarking_path(args)
#     args.benchmarking_result_path = os.path.join(
#         args.benchmarking_result_path,
#         args.model_id,
#         args.target,
#         args.baseline,
#         'benchmarking',
#         f'concept_{args.removal_mode}'
#     )
#     os.makedirs(args.benchmarking_result_path, exist_ok=True)

#     # Load the new prompt CSV dataset
#     data = pd.read_csv(args.dataset_csv)
#     if args.removal_mode == 'erase':
#         dataset = CustomDatasetErasure(data, args.target)
#     else:
#         dataset = CustomDatasetKeep(data, args.target)

#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=args.batch_size, shuffle=False
#     )
#     print(f"Number of batches: {len(dataloader)}")

#     # Load the concept-edited model
#     remover_model = load_models(args, args.ckpt_name)

#     # Pre-trained ResNet50 for evaluation
#     weights = ResNet50_Weights.DEFAULT
#     classifier = resnet50(weights=weights).to(args.gpu).eval()
#     preprocess = weights.transforms()

#     # Iterate and generate
#     correct = 0
#     total = 0
#     for idx, (prompt, seed, label) in enumerate(dataloader):
#         if args.dbg and idx > 10:
#             break

#         if idx >= 120:
#             break

#         prompt = prompt[0]
#         seed = seed[0].item() if isinstance(seed, torch.Tensor) else int(seed)
#         label = label[0]

#         torch.manual_seed(seed)
#         np.random.seed(seed)

#         images = remover_model(prompt).images
#         # save images
#         for i, img in enumerate(images):
#             img.save(os.path.join(
#                 args.benchmarking_result_path,
#                 f"{args.removal_mode}_{idx * args.batch_size + i}.png"
#             ))

#         # evaluation
#         for img in images:
#             x = preprocess(img).unsqueeze(0).to(args.gpu)
#             with torch.no_grad():
#                 out = classifier(x)
#             top1 = out.argmax(dim=-1).cpu().item()
#             pred = weights.meta['categories'][top1].lower()

#             # ===== 新增：逐样本打印真实类别与检测结果 =====
#             print(f"[{idx}] 真实类别: {label.lower()} | 检测结果: {pred}")

#             if pred == label.lower():
#                 correct += 1
#             total += 1

#     acc = correct / total if total > 0 else 0
#     print(f"Accuracy for {args.removal_mode}: {acc:.4f}")

#     # save results
#     result_name = args.ckpt_name or 'concept-prune'
#     result_name = os.path.basename(result_name).split('.')[0]
#     with open(os.path.join(
#         args.benchmarking_result_path,
#         f"results_{args.removal_mode}_{result_name}.json"
#     ), 'w') as fout:
#         json.dump({'accuracy': acc}, fout)


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser

# Ensure project root is in path for utils
sys.path.append(os.getcwd())
from utils import load_models
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchvision.models import resnet50, ResNet50_Weights
from benchmarking_utils import set_benchmarking_path


def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1244)
    parser.add_argument('--dbg', action='store_true')
    parser.add_argument('--target', type=str, default=None,
                        help='Concept/class to remove or keep')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Baseline identifier for results path')
    parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion',
                        help='Base results directory')
    parser.add_argument('--removal_mode', type=str, default=None,
                        choices=['erase', 'keep'],
                        help='Whether to erase or keep the target concept')
    parser.add_argument('--hook_module', type=str, default='unet',
                        help='Module of the model to hook for editing')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Stable Diffusion model identifier')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation')
    parser.add_argument('--ckpt_name', type=str, default=None,
                        help='Path to the saved checkpoint for the edited model')
    parser.add_argument('--dataset_csv', type=str,
                        default='datasets/imagenet-mini-50.csv',
                        help='Path to the prompt CSV file')
    parser.add_argument('--vanilla', action='store_true',
                        help='If set, ignore --ckpt_name and load the vanilla SD model via model_id')
    return parser.parse_args()


# Dataset class to test concept erasure
class CustomDatasetErasure(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, concepts_to_remove: str):
        self.prompts = data['prompt']
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except KeyError:
            self.labels = data['label_str']

        # filter to only the target class
        self.samples = [
            (self.prompts[i], self.seeds[i], concepts_to_remove.lower())
            for i in range(len(self.prompts))
            if self.labels[i].lower() == concepts_to_remove.lower()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, seed, label = self.samples[idx]
        return prompt, seed, label


# Dataset class to test concept keeping
class CustomDatasetKeep(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, concepts_to_remove: str):
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
            prompts = data['prompt']
        except KeyError:
            self.labels = data['label_str']
            prompts = data['prompt']

        # exclude the target class
        self.samples = [
            (prompts[i], self.seeds[i], self.labels[i].lower())
            for i in range(len(prompts))
            if self.labels[i].lower() != concepts_to_remove.lower()
        ]
        print(f"Total prompts after filtering: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, seed, label = self.samples[idx]
        return prompt, seed, label


def main():
    args = input_args()
    print("Arguments:\n", args)

    # setup benchmarking path
    args.benchmarking_result_path = set_benchmarking_path(args)
    args.benchmarking_result_path = os.path.join(
        args.benchmarking_result_path,
        args.model_id,
        args.target,
        args.baseline,
        'benchmarking',
        f'concept_{args.removal_mode}'
    )
    os.makedirs(args.benchmarking_result_path, exist_ok=True)

    # Load the prompt CSV dataset
    data = pd.read_csv(args.dataset_csv)
    if args.removal_mode == 'erase':
        dataset = CustomDatasetErasure(data, args.target)
    else:
        dataset = CustomDatasetKeep(data, args.target)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    print(f"Number of batches: {len(dataloader)}")

    # ─────────────── 仅此处加入 vanilla 分支 ───────────────
    if args.vanilla:
        print(f"Loading vanilla model: {args.model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16
        ).to(f"cuda:{args.gpu}")
        remover_model = pipe
    else:
        # 原有逻辑：加载概念编辑后的 checkpoint
        remover_model = load_models(args, args.ckpt_name)
    # ────────────────────────────────────────────────

    # Pre-trained ResNet50 for evaluation
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights).to(args.gpu).eval()
    preprocess = weights.transforms()

    # Iterate and generate
    correct = 0
    total = 0
    for idx, (prompt, seed, label) in enumerate(dataloader):
        if args.dbg and idx > 10:
            break

        if idx >= 120:
            break

        prompt = prompt[0]
        seed = seed[0].item() if isinstance(seed, torch.Tensor) else int(seed)
        label = label[0]

        torch.manual_seed(seed)
        np.random.seed(seed)

        images = remover_model(prompt).images
        # save images
        for i, img in enumerate(images):
            img.save(os.path.join(
                args.benchmarking_result_path,
                f"{args.removal_mode}_{idx * args.batch_size + i}.png"
            ))

        # evaluation
        for img in images:
            x = preprocess(img).unsqueeze(0).to(args.gpu)
            with torch.no_grad():
                out = classifier(x)
            top1 = out.argmax(dim=-1).cpu().item()
            pred = weights.meta['categories'][top1].lower()

            print(f"[{idx}] 真实类别: {label.lower()} | 检测结果: {pred}")

            if pred == label.lower():
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy for {args.removal_mode}: {acc:.4f}")

    # save results
    result_name = args.ckpt_name or 'concept-prune'
    result_name = os.path.basename(result_name).split('.')[0]
    with open(os.path.join(
        args.benchmarking_result_path,
        f"results_{args.removal_mode}_{result_name}.json"
    ), 'w') as fout:
        json.dump({'accuracy': acc}, fout)


if __name__ == '__main__':
    main()

