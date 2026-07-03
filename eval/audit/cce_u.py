import torch
import numpy as np
import os
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
import argparse
from safetensors.torch import load_file
from PIL import Image
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

def compute_cce_single_sample(pipe, base_unet, unlearned_unet, image_latents, prompt_embeds, pooled_embeds, add_time_ids, device, dtype):
    """计算单样本的去噪误差差异"""
    cce_scores = []
    timesteps = pipe.scheduler.timesteps 

    base_unet.eval()
    unlearned_unet.eval()

    with torch.no_grad():
        for t in timesteps:
            noise = torch.randn_like(image_latents).to(device, dtype=dtype)
            t_tensor = t.detach().clone().to(device).reshape(1)
            noisy_latents = pipe.scheduler.add_noise(image_latents, noise, t_tensor)
            
            added_cond = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
            
            # Base 误差
            noise_pred_base = base_unet(
                noisy_latents, t_tensor, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond
            ).sample
            
            # Unlearned 误差
            noise_pred_unlearned = unlearned_unet(
                noisy_latents, t_tensor, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond
            ).sample
            
            err_base = torch.nn.functional.mse_loss(noise_pred_base, noise)
            err_unlearned = torch.nn.functional.mse_loss(noise_pred_unlearned, noise)
            
            # 记录相对退化量
            cce_scores.append((err_unlearned - err_base).item())
            
    return np.mean(cce_scores) # 直接返回该样本在所有时间步的平均误差变化

def get_manual_add_time_ids(device, dtype):
    add_time_ids = [1024, 1024, 0, 0, 1024, 1024]
    return torch.tensor([add_time_ids], dtype=dtype, device=device)

def load_and_preprocess_image(image_path, pipe, device, dtype):
    try:
        image = Image.open(image_path).convert("RGB").resize((1024, 1024))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
        return latents
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def run_unlearning_audit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    print(f"\n--> [LOGICAL AUDIT] Initializing on {device}...")

    # A. 加载模型
    base_unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=dtype).to(device)
    unlearned_unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=dtype).to(device)
    
    if args.model_path.endswith(".safetensors"):
        state_dict = load_file(args.model_path)
        unlearned_unet.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in state_dict.items()}, strict=False)
    
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, unet=base_unet, torch_dtype=dtype, use_safetensors=True).to(device)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.text_encoder_2.to(device, dtype=dtype)
    pipe.vae.to(device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(50) 
    static_add_time_ids = get_manual_add_time_ids(device, dtype)

    # ==========================================
    # B. 建立严谨的基准：测算全局附带损伤 (Collateral Damage)
    # ==========================================
    print("\n[Step 1/2] Establishing Baseline (Global Collateral Damage)...")
    baseline_diffs = []
    irrelevant_dir = "audit_dataset/irrelevant"
    
    irr_files = [f for f in os.listdir(irrelevant_dir) if f.endswith(('.png', '.jpg'))] if os.path.exists(irrelevant_dir) else []
    # 彻底移除随机噪声 Padding！有多少图片就算多少
    irr_files = irr_files[:100]

    for item in tqdm(irr_files, desc="Processing Baseline"):
        p_embeds, _, pooled, _ = pipe.encode_prompt("a photo")
        latents = load_and_preprocess_image(os.path.join(irrelevant_dir, item), pipe, device, dtype)
        if latents is not None:
            diff = compute_cce_single_sample(pipe, base_unet, unlearned_unet, latents, p_embeds, pooled, static_add_time_ids, device, dtype)
            baseline_diffs.append(diff)
    
    # 计算基准损伤的均值和标准差
    mu_cd = np.mean(baseline_diffs) if baseline_diffs else 0.0
    sigma_cd = np.std(baseline_diffs) if baseline_diffs else 0.001
    
    # 定义遗忘判定阈值：必须大于全局损伤，且至少超出 2 个标准差，并附带最小物理绝对值要求(0.002)以防过度敏感
    erasure_threshold = mu_cd + max(2.0 * sigma_cd, 0.002)

    # ==========================================
    # C. 审计目标概念
    # ==========================================
    print(f"\n[Step 2/2] Auditing Target Concept: '{args.target_concept}'")
    target_folder = os.path.join("audit_dataset", "targets", args.target_concept.replace(" ", "_"))
    
    target_prompt = f"a photo of {args.target_concept}"
    t_p_embeds, _, t_pooled, _ = pipe.encode_prompt(target_prompt)
    
    target_samples = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg'))] if os.path.exists(target_folder) else []
    
    # 同样彻底移除 Target 的噪声 Padding！绝不能用纯噪声去测算遗忘率
    if len(target_samples) == 0:
        print("❌ ERROR: No target images found. Audit aborted.")
        return

    failed_to_forget_count = 0
    successfully_erased_count = 0
    target_diffs = []

    print(f"--> Analyzing {len(target_samples)} target samples...")
    for item in tqdm(target_samples, desc="Auditing Target"):
        latents = load_and_preprocess_image(item, pipe, device, dtype)
        if latents is None: continue
            
        target_diff = compute_cce_single_sample(pipe, base_unet, unlearned_unet, latents, t_p_embeds, t_pooled, static_add_time_ids, device, dtype)
        target_diffs.append(target_diff)
        
        # 严密的新逻辑：误差必须显著突破退化阈值，才算真正的遗忘
        if target_diff >= erasure_threshold:
            successfully_erased_count += 1
        else:
            failed_to_forget_count += 1

    # ==========================================
    # D. 最终报告
    # ==========================================
    esr_score = (successfully_erased_count / len(target_samples)) * 100.0
    avg_target_diff = np.mean(target_diffs)
    
    C_GREEN, C_RED, C_YELLOW, C_END = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

    print("\n" + "="*70)
    print(f"  🛡️  STRICT UNLEARNING AUDIT REPORT: {args.target_concept.upper()}")
    print("="*70)
    
    print(f"【 1. Erasure Success Rate (ESR) 】")
    print(f"  Score:   {C_YELLOW}{esr_score:.2f}%{C_END}")
    print(f"  Passed:  {successfully_erased_count}/{len(target_samples)} samples strongly erased.")
    
    print(f"\n【 2. Baseline Stats (Collateral Damage) 】")
    print(f"  Avg Damage (μ): {mu_cd:.6f}")
    print(f"  Variance (σ):   {sigma_cd:.6f}")
    print(f"  >> Strict Threshold for Erasure: {erasure_threshold:.6f}")
    
    print(f"\n【 3. Target Concept Stats 】")
    color_td = C_GREEN if avg_target_diff >= erasure_threshold else C_RED
    print(f"  Avg Target Error Diff: {color_td}{avg_target_diff:.6f}{C_END}")
    
    print("-" * 70)
    print("【 FINAL DIAGNOSIS 】")
    if esr_score > 80.0:
        print(f"  {C_GREEN}✅ TRUE UNLEARNING.{C_END} Concept visually and statistically erased.")
    elif esr_score <= 40.0:
        print(f"  {C_RED}❌ FAILED TO FORGET.{C_END} Error increase is too small to cause visual erasure.")
    else:
        print(f"  {C_YELLOW}⚠️  PARTIAL ERASURE.{C_END} Concept is degraded but not destroyed.")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Unlearned UNet")
    parser.add_argument("--target_concept", type=str, required=True, help="Concept meant to be erased")
    args = parser.parse_args()
    run_unlearning_audit(args)