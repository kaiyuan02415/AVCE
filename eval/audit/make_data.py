import torch
import os
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

# 配置
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "audit_dataset"

# 审计的目标概念列表
TARGET_CONCEPTS = [
    "parachute", "golf ball", "garbage truck", "cassette player", "church",
    "tench", "english springer", "french horn", "chain saw", "gas pump"
]

# 用于构建基准的无关提示词 (Irrelevant Prompts)
# 论文建议使用模型不预期生成的、或者多样化的概念来捕捉典型的内部行为 [cite: 335, 361]
IRRELEVANT_PROMPTS = [
    "a high quality photo of a mountain landscape",
    "a futuristic city street with neon lights",
    "a cute cat sitting on a wooden table",
    "a portrait of an old man in the rain",
    "abstract colorful smoke pattern",
    "a detailed macro shot of a blooming flower",
    "a bowl of fresh fruits on a kitchen counter",
    "deep space nebula with stars"
]

def setup_directories():
    os.makedirs(os.path.join(OUTPUT_DIR, "targets"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "irrelevant"), exist_ok=True)
    for concept in TARGET_CONCEPTS:
        os.makedirs(os.path.join(OUTPUT_DIR, "targets", concept.replace(" ", "_")), exist_ok=True)

def generate_data():
    setup_directories()
    
    print(f"--> 加载原生 SDXL 模型: {MODEL_ID}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # 1. 生成目标概念图像 (每个概念生成 10 张作为参考 [cite: 322, 606])
    print("--> 正在生成目标概念图像 (Target Concept Set)...")
    for concept in TARGET_CONCEPTS:
        save_path = os.path.join(OUTPUT_DIR, "targets", concept.replace(" ", "_"))
        print(f"   正在处理: {concept}")
        for i in range(10):
            prompt = f"a high quality photo of a {concept}"
            image = pipe(prompt, num_inference_steps=30).images[0]
            image.save(os.path.join(save_path, f"{i}.png"))

    # 2. 生成无关概念图像 (生成约 120 张以建立基准分布 [cite: 553])
    print("--> 正在生成无关概念图像 (Irrelevant Concept Set)...")
    num_irrelevant = 120
    images_per_prompt = num_irrelevant // len(IRRELEVANT_PROMPTS)
    
    count = 0
    for prompt in tqdm(IRRELEVANT_PROMPTS):
        for i in range(images_per_prompt):
            image = pipe(prompt, num_inference_steps=30).images[0]
            image.save(os.path.join(OUTPUT_DIR, "irrelevant", f"irrelevant_{count}.png"))
            count += 1
            
    print(f"--> 数据生成完成！保存在 '{OUTPUT_DIR}' 目录下。")

if __name__ == "__main__":
    generate_data()