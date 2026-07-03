import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser

# 将当前工作目录添加到sys.path中
sys.path.append(os.getcwd())

# 修改1：移除对 load_models 的导入，因为我们直接使用 diffusers 加载模型
# from utils import load_models

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchvision.models import resnet50, ResNet50_Weights
from benchmarking_utils import set_benchmarking_path


def input_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1244)
    parser.add_argument('--dbg', action='store_true')
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--res_path', type=str, default='results/results_seed_0/stable-diffusion')
    parser.add_argument('--removal_mode', type=str, default=None, choices=['erase', 'keep'])
    parser.add_argument('--hook_module', type=str, default='unet')
    # parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--batch_size', type=int, default=1)
    # 使用 ckpt_name 指定模型的保存路径（Diffusers格式文件夹），如果为空则使用 model_id 下载模型
    parser.add_argument('--ckpt_name', type=str, default=None)
    return parser.parse_args()


# Dataset class to test concept erasure
class CustomDatasetErasure(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.prompts = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']

        # 仅选择包含目标概念的 prompt
        self.prompts = [(self.prompts[i], self.seeds[i], concepts_to_remove) 
                        for i in range(len(self.prompts)) 
                        if concepts_to_remove.lower() == self.labels[i].lower()]
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx][0]
        seed = self.prompts[idx][1]
        label = self.prompts[idx][2].lower()
        return prompt, seed, label
    

# Dataset class to test concept keeping
class CustomDatasetKeep(torch.utils.data.Dataset):
    def __init__(self, data, concepts_to_remove):
        self.dataset = data['prompt']
        self.concepts_to_remove = concepts_to_remove
        self.seeds = data['evaluation_seed']
        try:
            self.labels = data['class']
        except:
            self.labels = data['label_str']
        self.dataset = [(self.dataset[i], self.seeds[i], self.labels[i].lower()) 
                        for i in range(len(self.dataset)) 
                        if concepts_to_remove.lower() != self.labels[i].lower()]
        print(f"Number of prompts: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx][0]
        seed = self.dataset[idx][1]
        label = self.dataset[idx][2].lower()
        return prompt, seed, label
    

def main():
    args = input_args()
    print("Arguments: ", args.__dict__)

    args.benchmarking_result_path = set_benchmarking_path(args)
    args.benchmarking_result_path = os.path.join(args.benchmarking_result_path, args.model_id, args.target, args.baseline, 'benchmarking', f'concept_{args.removal_mode}')
    print("Benchmarking result path: ", args.benchmarking_result_path)
    if not os.path.exists(args.benchmarking_result_path):
        os.makedirs(args.benchmarking_result_path)

    if not os.path.exists(f'{args.benchmarking_result_path}'):
        os.makedirs(f'{args.benchmarking_result_path}')

    # Load dataset
    data = pd.read_csv(f'datasets/imagenette.csv')
    if args.removal_mode == 'erase':
        dataloader = torch.utils.data.DataLoader(CustomDatasetErasure(data, args.target), batch_size=args.batch_size, shuffle=False)
    else:
        dataloader = torch.utils.data.DataLoader(CustomDatasetKeep(data, args.target), batch_size=args.batch_size, shuffle=False)

    print("Number of prompts: ", len(dataloader))

    # 修改2：使用 diffusers 加载模型
    # 如果提供了 ckpt_name 则使用该路径，否则使用 model_id 下载模型
    if args.ckpt_name is not None:
        model_path = args.ckpt_name
    else:
        model_path = args.model_id
    print("Loading model from: ", model_path)
    remover_model = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    remover_model.to(f"cuda:{args.gpu}")
    # End of 修改2

    print("now get sp model")

    # Pre-trained ResNet50 
    weights = ResNet50_Weights.DEFAULT
    classifier = resnet50(weights=weights)
    classifier = classifier.to(args.gpu)
    classifier.eval()

    preprocess = weights.transforms()

    # test model on dataloader
    avg_acc = 0
    for iter, (prompt, seed, label) in enumerate(dataloader):
        if iter >= 100:
            break

        if args.dbg and iter > 10:
            break

        print(f"Prompt: {prompt}, Seed: {seed}, Label: {label}")
        prompt = prompt[0]
        label = label[0]
        torch.manual_seed(seed[0])
        np.random.seed(seed[0])
        # 调用 diffusers pipeline 生成图像
        removal_images = remover_model(prompt).images
        # 保存生成的图像
        for i, image in enumerate(removal_images):
            image.save(os.path.join(args.benchmarking_result_path, f"removed_{iter * args.batch_size + i}.png"))

        # 使用 ResNet50 进行评估
        for i, image in enumerate(removal_images):
            image = preprocess(image).unsqueeze(0)
            image = image.to(args.gpu)
            with torch.no_grad():
                output = classifier(image)
                
            s, indices = torch.topk(output, 1)
            indices = indices.cpu().numpy()
            pred_labels = [weights.meta["categories"][idx] for idx in indices[0]]
            print(f"Predicted labels: {pred_labels}", "True label: ", label)
            pred_labels = [l.lower() for l in pred_labels]

            if label in pred_labels:
                avg_acc += 1

    print("Object predicted in: %d/%d images" % (avg_acc, len(dataloader)))
    print(f"Average accuracy: {avg_acc / len(dataloader)}")
    results = {"average_accuracy": avg_acc / len(dataloader)}
    p = args.ckpt_name.split('/')[-1].split('.pt')[0] if args.ckpt_name is not None else 'concept-prune'
    with open(os.path.join(args.benchmarking_result_path, f"results_{args.removal_mode}_{p}.json"), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
