import os 
import torch
import sys
import random
import numpy as np
import copy
import ast
import gc
import argparse
from tqdm.auto import tqdm
from functools import reduce
import operator
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

# 尝试引入 esd_utils，如果不存在则忽略（通常用于自定义 call）
sys.path.append('.')
try:
    from utils.sdxl_utils import esd_sdxl_call
    StableDiffusionXLPipeline.__call__ = esd_sdxl_call
except ImportError:
    pass

# ==============================================================================
# PART 1: MUS 核心数学工具 (严格复刻 mus_sdxl.py)
# ==============================================================================

def get_embeddings(model, texts, device):
    """
    统一获取 Embedding，适配 SDXL
    """
    with torch.no_grad():
        # do_classifier_free_guidance=False 获取单纯的 text embedding
        prompt_embeds, _, _, _ = model.encode_prompt(texts, do_classifier_free_guidance=False)
        tokens = model.tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        attention_mask = tokens.attention_mask.to(prompt_embeds.device)
    return prompt_embeds, attention_mask

def _compute_entropy_factor(W_old, concept_vecs, num_samples=20, bins=20, noise_sigma=0.01, eps=1e-8):
    out_dim, _ = W_old.shape
    dtype = W_old.dtype 
    device = W_old.device
    
    acts = []
    for c in concept_vecs:
        c = c.to(device=device, dtype=dtype)
        for _ in range(num_samples):
            noise = torch.randn_like(c, dtype=dtype, device=device) * noise_sigma
            acts.append((W_old @ (c + noise)).unsqueeze(1))
            
    acts = torch.cat(acts, dim=1) 
    acts = torch.nan_to_num(acts, nan=0.0)
    
    min_v = acts.min(dim=1, keepdim=True).values
    max_v = acts.max(dim=1, keepdim=True).values
    acts = (acts - min_v) / (max_v - min_v + eps)
    
    e = torch.zeros(out_dim, device=device, dtype=dtype)
    logK = torch.log(torch.tensor(float(bins), device=device))
    acts_f32 = acts.float()
    
    for i in range(out_dim):
        hist = torch.histc(acts_f32[i], bins=bins, min=0.0, max=1.0)
        p = hist / (hist.sum() + eps)
        H = -(p * (p + eps).log()).sum()
        e[i] = 1 - H / logK
        
    return e.to(dtype) 

def _compute_mi_softmask_emptyneg(W_old, c_vec, empty_vec, num_pos=5, T=0.7, p=2.0, noise_sigma=0.05, eps=1e-8):
    device = W_old.device
    dtype = W_old.dtype 
    out_dim, _ = W_old.shape

    c_vec = c_vec.to(device=device, dtype=dtype)
    empty_vec = empty_vec.to(device=device, dtype=dtype)

    pos = c_vec.repeat(num_pos,1) + noise_sigma * torch.randn(num_pos, c_vec.numel(), device=device, dtype=dtype)
    neg = empty_vec.repeat(num_pos,1) + noise_sigma * torch.randn(num_pos, c_vec.numel(), device=device, dtype=dtype)
    samples = torch.cat([pos, neg], dim=0)           
    labels  = torch.cat([torch.ones(num_pos, device=device), torch.zeros(num_pos, device=device)]) 

    acts = W_old @ samples.t()                       
    tau = acts.median(dim=1, keepdim=True).values
    Z   = (acts > tau).long()                       
    K  = 2 * num_pos
    mi = torch.zeros(out_dim, device=device, dtype=dtype)
    
    for i in range(out_dim):
        z = Z[i]
        n11 = ((z==1)&(labels==1)).sum().to(dtype) + eps
        n10 = ((z==1)&(labels==0)).sum().to(dtype) + eps
        n01 = ((z==0)&(labels==1)).sum().to(dtype) + eps
        n00 = ((z==0)&(labels==0)).sum().to(dtype) + eps
        p11, p10, p01, p00 = n11/K, n10/K, n01/K, n00/K
        p1_, p0_ = p11+p10, p01+p00
        p_1, p_0 = p11+p01, p10+p00
        mi[i] = (p11*torch.log(p11/(p1_*p_1)) + p10*torch.log(p10/(p1_*p_0)) +
                 p01*torch.log(p01/(p0_*p_1)) + p00*torch.log(p00/(p0_*p_0)))

    mi_std = (mi - mi.mean()) / (mi.std() + eps)
    m = torch.sigmoid(mi_std / T).pow(p)
    return m.view(-1,1)

def smooth_svd_on_mat1(mat1, C_stack, W_old, in_dim, device, T_sigma=0.5, p_sigma=2.0, eps=1e-8):
    dtype = W_old.dtype 
    C_stack_cpu = C_stack.float().cpu()
    U, S, _ = torch.linalg.svd(C_stack_cpu, full_matrices=False)   
    U = U.to(device)
    S = S.to(device)
    tau = torch.median(S)
    w   = torch.sigmoid((S - tau)/(T_sigma + eps)).pow(p_sigma) 
    U_w = U * w.unsqueeze(0)                                   
    P   = torch.eye(in_dim, device=device) - U_w @ U.T         
    C_res = P.to(dtype) @ C_stack                                        
    V_res = W_old @ C_res                                      
    return V_res, C_res

def mus_edit_stage(pipe, concepts_list, concept_type='object', technique='tensor', 
                   svd=False, p=2.0, erase_scale=1.0, alpha_min=0.1, 
                   entropy_samples=20, entropy_bins=20, noise_sigma=0.01,
                   T_sigma=1.0, p_sigma=1.0, lamb=0.5, device='cuda'):
    """
    Stage 1: MUS 全局遗忘逻辑
    包含原 mus_sdxl.py 的所有关键超参数
    """
    print(f"--> [MUS Stage] Erasing {len(concepts_list)} concepts as {concept_type}...")
    
    # 1. 扩展 Prompt 列表 (Old Texts)
    old_texts = []
    additional_prompts = []
    if concept_type == 'art':
        additional_prompts = ['painting by {c}', 'art by {c}', 'artwork by {c}', 'picture by {c}', 'style of {c}']
    elif concept_type == 'object':
        additional_prompts = ['image of {c}', 'photo of {c}', 'portrait of {c}', 'picture of {c}', 'painting of {c}']
    
    # 如果没有指定 type，默认使用 object 的一部分通用词
    if not additional_prompts:
         additional_prompts = ['image of {c}', 'photo of {c}', 'style of {c}']

    for c in concepts_list:
        old_texts.append(f'{c}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(c=c))
    
    # Destination prompts (Uncond)
    new_texts = [' ' for _ in old_texts]
    
    # 2. 定位要编辑的层 (Attn2)
    unet = pipe.unet
    ca_layers = []
    for name, module in unet.named_modules():
        if 'attn2' in name and module.__class__.__name__ == 'Attention':
            ca_layers.append(module)
    
    print(f"--> [MUS Stage] Layers to edit: {len(ca_layers)}")

    # 3. 预计算概念向量
    target_device = device
    embeds_all, attention_mask_all = get_embeddings(pipe, old_texts, target_device)
    concept_vecs = [emb[mask.sum()-2] for emb, mask in zip(embeds_all, attention_mask_all)]
    
    blank_emb, _ = get_embeddings(pipe, [""], target_device)
    empty_vec = blank_emb[0, 1, :].detach()
    
    C_full = torch.stack(concept_vecs, 1)          
    CCt    = C_full @ C_full.t()  

    # 4. 执行编辑循环 (To_v 和 To_k)
    for matrix_type in ['to_v', 'to_k']:
        print(f"--> [MUS Stage] Processing {matrix_type} matrices...")
        
        for layer_num, ca in enumerate(tqdm(ca_layers, desc=f'MUS-{matrix_type}')):
            module = getattr(ca, matrix_type)
            
            with torch.no_grad():
                W_old = module.weight.data.clone().to(target_device)
                dtype = W_old.dtype 
                out_dim, in_dim = W_old.shape
                
                # Init Matrices (使用传入的 lamb)
                mat1 = lamb * W_old.clone()
                mat2 = lamb * torch.eye(in_dim, device=target_device, dtype=dtype) + CCt.to(target_device, dtype=dtype)
                
                res_ctx = []

                # Accumulate Statistics
                for idx, (ot, nt, c_vec) in enumerate(zip(old_texts, new_texts, concept_vecs)):
                    emb_pair, mask_pair = get_embeddings(pipe, [ot, nt], target_device)
                    f_o  = mask_pair[0].sum().item() - 2
                    f_n  = mask_pair[1].sum().item() - 2
                    far  = max(f_o, f_n)
                    old_emb = emb_pair[0][f_o: len(emb_pair[0])-max(0,far-f_o)]
                    new_emb = emb_pair[1][f_n: len(emb_pair[1])-max(0,far-f_n)]
                    context = old_emb.detach()
                    
                    if technique == 'tensor':
                        u = module(context).detach(); u = u/u.norm()
                        v = module(new_emb).detach()
                        vals_curr = (v - (u*v).sum()*u).detach()
                    else:
                        vals_curr = module(new_emb).detach()
                    
                    ctx_v  = context.view(context.size(0), context.size(1), 1)
                    ctx_vT = context.view(context.size(0), 1, context.size(1))
                    val_v  = vals_curr.view(vals_curr.size(0), vals_curr.size(1), 1)
                    
                    for_mat1 = (val_v @ ctx_vT).sum(0)
                    for_mat2 = (ctx_v  @ ctx_vT).sum(0)
                    
                    # 使用传入的 noise_sigma
                    row_w = _compute_mi_softmask_emptyneg(W_old, c_vec, empty_vec, p=p, noise_sigma=noise_sigma)
                    
                    mat1 += erase_scale * (for_mat1.to(dtype) * row_w.to(dtype))
                    mat2 += erase_scale * for_mat2.to(dtype)
                    res_ctx.append(c_vec.detach().to(dtype))

                if svd:
                    # 使用传入的 T_sigma, p_sigma
                    C_stack = torch.stack(res_ctx, 1)
                    V_res, C_res = smooth_svd_on_mat1(
                        mat1, C_stack, W_old, in_dim, target_device,
                        T_sigma=T_sigma, p_sigma=p_sigma
                    )
                    mat1 += -erase_scale * (V_res @ C_res.t())

                # Solve & Safety Clip
                mat2_inv = torch.inverse(mat2.float().cpu()).to(target_device)
                W_new = mat1.float() @ mat2_inv.float()
                W_new = torch.nan_to_num(W_new, nan=0.0, posinf=60000.0, neginf=-60000.0).to(dtype)
                
                # Entropy Protection (使用传入的 bins, noise_sigma, samples)
                e_i = _compute_entropy_factor(
                    W_old, concept_vecs, 
                    num_samples=entropy_samples, 
                    bins=entropy_bins, 
                    noise_sigma=noise_sigma
                )
                alpha = alpha_min + (1 - alpha_min) * e_i
                alpha = torch.nan_to_num(alpha, nan=1.0)
                
                W_final = alpha.view(-1,1) * W_new + (1-alpha).view(-1,1) * W_old
                W_final = torch.nan_to_num(W_final, nan=0.0)
                
                # Apply
                module.weight.data.copy_(W_final)
            
            # GC
            if layer_num % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print("--> [MUS Stage] Completed.")
    return pipe

# ==============================================================================
# PART 2: M2 核心组件 (Aegis Flow: PCGrad, PAIA, HCM)
# ==============================================================================

def pcgrad_merge(losses, params, eps=1e-12, weights=None):
    """PCGrad with Dynamic Loss Weighting support"""
    task_grads = []
    for L in losses:
        g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        task_grads.append(list(g))
    K = len(task_grads); proj_grads = []
    for i in range(K):
        gi = [t.clone() if t is not None else None for t in task_grads[i]]
        order = torch.randperm(K).tolist()
        for j in order:
            if j == i: continue
            gj = task_grads[j]
            dot = 0; denom = 0
            for a, b in zip(gi, gj):
                if a is not None and b is not None:
                    dot += (a * b).sum(); denom += (b * b).sum()
            if dot is None or denom is None: continue 
            if dot < 0:
                coeff = dot / (denom + eps)
                for k in range(len(gi)):
                    if gi[k] is not None and gj[k] is not None: gi[k] -= coeff * gj[k]
        proj_grads.append(gi)
    
    if weights is None: weights = [1.0] * K
    merged = []
    for p_idx in range(len(params)):
        valid = []; valid_w = []
        for i in range(K):
            if proj_grads[i][p_idx] is not None:
                valid.append(proj_grads[i][p_idx])
                valid_w.append(weights[i])
        if not valid: merged.append(None)
        else:
            ws = sum(valid_w)
            if ws <= 0: merged.append(sum(valid)/len(valid))
            else:
                out = valid[0]*valid_w[0]
                for gg, ww in zip(valid[1:], valid_w[1:]): out += gg*ww
                merged.append(out/ws)
    return merged

def weighted_sample_without_replacement(items, k, probs):
    """Hard Concept Mining weighted sampler"""
    n = len(items)
    if k <= 0: return []
    k = min(k, n)
    p = np.asarray(probs, dtype=np.float64); p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0: return random.sample(items, k=k)
    p = p / s
    idx = np.random.choice(np.arange(n), size=k, replace=False, p=p)
    return [items[i] for i in idx.tolist()]

def run_paia_attack(pipe, base_unet, student_unet, original_embeds, pooled_embeds, time_ids, num_steps=5, lr=1e-2, device='cuda:0'):
    """PAIA: Adversarial Prompt Optimization"""
    adv_embeds = original_embeds.detach().clone().requires_grad_(True)
    optimizer_attack = torch.optim.Adam([adv_embeds], lr=lr)
    student_unet.eval(); base_unet.eval()
    
    for _ in range(num_steps):
        optimizer_attack.zero_grad()
        t_idx = torch.randint(400, 800, (1,)).item()
        t_idx = min(t_idx, len(pipe.scheduler.timesteps) - 1)
        t = pipe.scheduler.timesteps[t_idx].unsqueeze(0).to(device)
        latents = torch.randn((1, 4, 128, 128), device=device, dtype=original_embeds.dtype)
        
        with torch.no_grad():
            noise_base = base_unet(latents, t, encoder_hidden_states=adv_embeds, added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids}).sample
        noise_student = student_unet(latents, t, encoder_hidden_states=adv_embeds, added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids}).sample
        
        loss = torch.nn.functional.mse_loss(noise_student, noise_base)
        loss.backward()
        optimizer_attack.step()
        
    student_unet.train()
    return adv_embeds.detach()

def register_attention_hooks(unet, capture_layer_name='attn2'):
    feature_store = {}; hooks = []
    def get_hook(name):
        def hook(module, input, output): feature_store[name] = output
        return hook
    for name, module in unet.named_modules():
        if capture_layer_name in name and module.__class__.__name__ == 'Attention':
            hooks.append(module.register_forward_hook(get_hook(name)))
    return feature_store, hooks

def remove_hooks(hooks):
    for h in hooks: h.remove()

def get_esd_trainable_parameters(esd_unet, train_method='esd-u'):
    esd_params = []; esd_param_names = []
    print(f"--> Selecting parameters for M2 tuning: {train_method}")
    for name, module in esd_unet.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if train_method == 'esd-x' and 'attn2' in name:
                for n, p in module.named_parameters(): esd_param_names.append(name+'.'+n); esd_params.append(p)
            elif train_method == 'esd-x-strict' and ('attn2.to_k' in name or 'attn2.to_v' in name):
                for n, p in module.named_parameters(): esd_param_names.append(name+'.'+n); esd_params.append(p)
            elif train_method == 'esd-u' and ('attn2' not in name and 'emb' not in name and 'block' in name):
                for n, p in module.named_parameters(): esd_param_names.append(name+'.'+n); esd_params.append(p)
            elif train_method == 'esd-all' and 'emb' not in name:
                 for n, p in module.named_parameters(): esd_param_names.append(name+'.'+n); esd_params.append(p)
    return esd_param_names, esd_params

# ==============================================================================
# PART 3: 主程序
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid-Aegis: MUS Global Erasure + M2 Robust Fine-tuning')
    
    # --- General ---
    parser.add_argument('--erase_concept', type=str, required=True, help='Concepts to erase (comma separated)')
    parser.add_argument('--concept_class', type=str, required=True, help='Class name for filename')
    parser.add_argument('--erase_from', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='models/hybrid/')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)

    # --- Stage 1: MUS Parameters (Full retention of original params) ---
    parser.add_argument('--skip_mus', action='store_true', help='Skip MUS stage')
    parser.add_argument('--mus_method', type=str, default='tensor', choices=['tensor', 'replace'])
    parser.add_argument('--mus_svd', action='store_true', help='Enable SVD in MUS')
    parser.add_argument('--mus_erase_scale', type=float, default=1.0)
    parser.add_argument('--mus_p', type=float, default=2.0)
    parser.add_argument('--mus_alpha_min', type=float, default=0.1)
    parser.add_argument('--concept_type', type=str, default='object', choices=['object', 'art'], help='Prompt expansion type for MUS')
    # Extra MUS params
    parser.add_argument('--mus_lamb', type=float, default=0.5, help='Lambda regularization for MUS')
    parser.add_argument('--mus_entropy_samples', type=int, default=20, help='Entropy samples')
    parser.add_argument('--mus_entropy_bins', type=int, default=20, help='Entropy bins')
    parser.add_argument('--mus_noise_sigma', type=float, default=0.01, help='Noise sigma for stats')
    parser.add_argument('--mus_T_sigma', type=float, default=1.0, help='SVD T_sigma')
    parser.add_argument('--mus_p_sigma', type=float, default=1.0, help='SVD p_sigma')

    # --- Stage 2: M2 (ESD) Parameters ---
    parser.add_argument('--esd_method', type=str, default='esd-u', help='Train method for stage 2')
    parser.add_argument('--iterations', type=int, default=50, help='Total M2 iterations')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate for Stage 2')
    parser.add_argument('--negative_guidance', type=float, default=1.0)
    parser.add_argument('--lambda_attn', type=float, default=1.0, help='S2 protection weight')
    parser.add_argument('--adv', action='store_true', help='Enable PAIA adversarial hardening')
    parser.add_argument('--adv_steps', type=int, default=3)
    parser.add_argument('--adv_lr', type=float, default=1e-2)
    parser.add_argument('--dts', action='store_true', help='Enable Dynamic Timestep Scheduling')
    parser.add_argument('--concept_batch', type=int, default=3)

    args = parser.parse_args()
    
    # 1. 解析概念
    if ',' in args.erase_concept:
        concepts = [c.strip() for c in args.erase_concept.split(',') if c.strip()]
    else:
        concepts = [args.erase_concept.strip()]
    
    is_multi = len(concepts) > 1
    print(f"--> Target Concepts ({len(concepts)}): {concepts}")
    
    # 2. 加载模型
    device = args.device
    dtype = torch.bfloat16
    print(f"--> Loading SDXL Base...")
    
    # Teacher (Base) - 永远保持原始状态
    base_unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet").to(device, dtype)
    base_unet.requires_grad_(False)
    
    # Student (ToBeEdited) - 初始与 Base 相同
    student_unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet").to(device, dtype)
    
    # Load Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", unet=base_unet, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.text_encoder.to(device, dtype=dtype)
    pipe.text_encoder_2.to(device, dtype=dtype)
    pipe.vae.to(device, dtype=dtype)

    # ==============================================================================
    # Stage 1: MUS Global Erasure (Global Surgery)
    # ==============================================================================
    if not args.skip_mus:
        print("\n" + "="*40)
        print("   STAGE 1: MUS (Global Linear Erasure)   ")
        print("="*40)
        
        # 将 pipe.unet 临时指向 student_unet 进行编辑
        pipe.unet = student_unet
        
        # 执行 MUS
        pipe = mus_edit_stage(
            pipe, concepts, 
            concept_type=args.concept_type,
            technique=args.mus_method,
            svd=args.mus_svd,
            p=args.mus_p,
            erase_scale=args.mus_erase_scale,
            alpha_min=args.mus_alpha_min,
            entropy_samples=args.mus_entropy_samples,
            entropy_bins=args.mus_entropy_bins,
            noise_sigma=args.mus_noise_sigma,
            T_sigma=args.mus_T_sigma,
            p_sigma=args.mus_p_sigma,
            lamb=args.mus_lamb,
            device=device
        )
        
        print("--> MUS Stage completed. Student UNet weights modified in-place.")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("--> Skipping MUS Stage (Starting from Base SDXL).")

    # ==============================================================================
    # Stage 2: M2 Robust Fine-tuning (PCGrad + PAIA + HCM)
    # ==============================================================================
    print("\n" + "="*40)
    print(f"   STAGE 2: M2 (Robust Fine-tuning) - {args.iterations} Steps   ")
    print("="*40)
    
    # 恢复 pipe 指向 base (用于计算 target)，student 独立求导
    pipe.unet = base_unet 
    student_unet.requires_grad_(True)
    
    # 准备优化器
    esd_param_names, esd_params = get_esd_trainable_parameters(student_unet, train_method=args.esd_method)
    optimizer = torch.optim.Adam(esd_params, lr=args.lr)
    
    # 预计算 Concept Bank
    concept_bank = {}
    with torch.no_grad():
        _, null_embeds, _, null_pooled = pipe.encode_prompt("", device=device, do_classifier_free_guidance=True)
        
        # 共享 Erase From
        if args.erase_from and args.erase_from != args.erase_concept:
             shared_from_embeds, _, shared_from_pooled, _ = pipe.encode_prompt(args.erase_from, device=device, do_classifier_free_guidance=False)
        else:
             shared_from_embeds, shared_from_pooled = None, None

        for c in concepts:
            e_emb, _, e_pool, _ = pipe.encode_prompt(c, device=device, do_classifier_free_guidance=True)
            if shared_from_embeds is not None:
                ef_emb, ef_pool = shared_from_embeds, shared_from_pooled
            else:
                ef_emb, ef_pool = e_emb, e_pool # default to self
            
            concept_bank[c] = {
                "erase_embeds": e_emb, "erase_pooled": e_pool,
                "erase_from_embeds": ef_emb, "erase_from_pooled": ef_pool
            }
            
        add_time_ids = pipe._get_add_time_ids((1024,1024), (0,0), (1024,1024), dtype=dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(device).repeat(args.batchsize, 1)
            
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim:
            guidance_scale_tensor = torch.tensor(args.guidance_scale - 1).repeat(args.batchsize)
            timestep_cond = pipe.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim).to(device, dtype=dtype)

    # HCM Init
    HCM_EMA_BETA = 0.95
    HCM_TAU = 0.7
    HCM_UNIFORM_MIX = 0.2
    
    # DLW Init
    DLW_TAU = 0.7
    DLW_EPS = 1e-3
    
    ema_loss = {c: 1.0 for c in concepts}
    criteria = torch.nn.MSELoss()
    
    # M2 Training Loop
    pbar = tqdm(range(args.iterations), desc='M2 Fine-tuning')
    for iteration in pbar:
        optimizer.zero_grad()
        
        # DTS
        if args.dts:
            run_index = int(np.random.beta(2, 2) * (args.num_inference_steps - 1))
        else:
            run_index = random.randint(0, args.num_inference_steps-1)
        run_index = max(0, min(run_index, args.num_inference_steps - 1))
        current_t = pipe.scheduler.timesteps[run_index]
        seed = random.randint(0, 2**15)
        
        # Latents Generation
        with torch.no_grad():
            xt = pipe(random.choice(concepts) if is_multi else concepts[0], 
                      num_inference_steps=args.num_inference_steps, 
                      run_till_timestep=run_index, 
                      guidance_scale=args.guidance_scale, output_type='latent', height=1024, width=1024).images

        # -----------------------------------------------------------
        # Branch 1: Adversarial (PAIA)
        # -----------------------------------------------------------
        if args.adv and (iteration % 2 == 0):
            adv_concept = random.choice(concepts)
            adv_data = concept_bank[adv_concept]
            
            # Attack
            adv_embeds = run_paia_attack(
                pipe, base_unet, student_unet, 
                adv_data["erase_embeds"], adv_data["erase_pooled"], add_time_ids,
                num_steps=args.adv_steps, lr=args.adv_lr, device=device
            )
            
            # Defense Loss
            with torch.no_grad():
                added_cond = {"text_embeds": null_pooled, "time_ids": add_time_ids}
                target_noise = base_unet(xt, current_t, encoder_hidden_states=null_embeds, added_cond_kwargs=added_cond, timestep_cond=timestep_cond).sample
            
            added_cond = {"text_embeds": adv_data["erase_pooled"], "time_ids": add_time_ids}
            student_noise = student_unet(xt, current_t, encoder_hidden_states=adv_embeds, added_cond_kwargs=added_cond, timestep_cond=timestep_cond).sample
            
            total_loss = criteria(student_noise, target_noise)
            total_loss.backward()
            optimizer.step()
            
            desc = f"Adv ({adv_concept})"
            loss_val = total_loss.item()
            loss_attn_val = 0.0

        # -----------------------------------------------------------
        # Branch 2: Clean (S2 + PCGrad + HCM)
        # -----------------------------------------------------------
        else:
            # S2: Capture Teacher Attention (Null)
            target_attn_maps = {}
            if args.lambda_attn > 0:
                ts, th = register_attention_hooks(base_unet, 'attn2')
                with torch.no_grad():
                    pipe.unet = base_unet
                    added_cond = {"text_embeds": null_pooled, "time_ids": add_time_ids}
                    base_unet(xt, current_t, encoder_hidden_states=null_embeds, added_cond_kwargs=added_cond, timestep_cond=timestep_cond)
                    target_attn_maps = {k: v.detach().clone() for k, v in ts.items()}
                remove_hooks(th)

            # HCM Sampling
            if is_multi:
                hardness = np.array([(ema_loss[c] + 1e-3)**HCM_TAU for c in concepts], dtype=np.float64)
                hard_p = hardness / (hardness.sum() + 1e-12)
                uni_p = np.ones_like(hard_p) / len(hard_p)
                mix_p = (1.0 - HCM_UNIFORM_MIX) * hard_p + HCM_UNIFORM_MIX * uni_p
                batch_c = weighted_sample_without_replacement(concepts, min(args.concept_batch, len(concepts)), mix_p)
            else:
                batch_c = concepts

            losses = []
            loss_attn_vals = []
            loss_esd_vals = []
            
            # Shared Teacher Null
            with torch.no_grad():
                added_cond = {"text_embeds": null_pooled, "time_ids": add_time_ids}
                null_pred = base_unet(xt, current_t, encoder_hidden_states=null_embeds, added_cond_kwargs=added_cond, timestep_cond=timestep_cond).sample

            for c in batch_c:
                c_data = concept_bank[c]
                
                # Teacher Targets
                with torch.no_grad():
                    # Positive
                    added_cond = {"text_embeds": c_data["erase_pooled"], "time_ids": add_time_ids}
                    pos_pred = base_unet(xt, current_t, encoder_hidden_states=c_data["erase_embeds"], added_cond_kwargs=added_cond, timestep_cond=timestep_cond).sample
                    # Erase From
                    if c_data["erase_from_embeds"] is not None:
                        added_cond = {"text_embeds": c_data["erase_from_pooled"], "time_ids": add_time_ids}
                        erase_from_pred = base_unet(xt, current_t, encoder_hidden_states=c_data["erase_from_embeds"], added_cond_kwargs=added_cond, timestep_cond=timestep_cond).sample
                    else:
                        erase_from_pred = pos_pred

                # Student Prediction
                ss, sh = register_attention_hooks(student_unet, 'attn2')
                added_cond = {"text_embeds": c_data["erase_pooled"], "time_ids": add_time_ids}
                student_pred = student_unet(xt, current_t, encoder_hidden_states=c_data["erase_embeds"], added_cond_kwargs=added_cond, timestep_cond=timestep_cond).sample
                
                # Target Calculation (CFG-guided erasure)
                target = erase_from_pred - (args.negative_guidance * (pos_pred - null_pred))
                l_esd = criteria(student_pred, target)
                
                # S2 Loss
                l_attn = torch.tensor(0.0, device=device)
                if args.lambda_attn > 0 and target_attn_maps:
                    count = 0
                    for k, v in ss.items():
                        if k in target_attn_maps:
                            l_attn += torch.nn.functional.mse_loss(v, target_attn_maps[k])
                            count += 1
                    if count > 0: l_attn /= count
                
                remove_hooks(sh)
                
                total_l = l_esd + args.lambda_attn * l_attn
                losses.append(total_l)
                loss_esd_vals.append(l_esd.item())
                loss_attn_vals.append(l_attn.item())
                
                ema_loss[c] = HCM_EMA_BETA * ema_loss[c] + (1.0-HCM_EMA_BETA) * l_esd.item()
            
            # PCGrad Merge with DLW
            if len(losses) > 0:
                if len(losses) == 1:
                    losses[0].backward()
                    optimizer.step()
                else:
                    # DLW Weights
                    ws_vec = np.array([(ema_loss[c] + DLW_EPS)**DLW_TAU for c in batch_c], dtype=np.float64)
                    mean_v = float(ws_vec.mean()) if ws_vec.size > 0 else 1.0
                    ws_norm = ws_vec / (mean_v + 1e-12)
                    ws_norm = np.clip(ws_norm, 0.5, 2.0).tolist()
                    
                    merged = pcgrad_merge(losses, esd_params, weights=ws_norm)
                    for p, g in zip(esd_params, merged):
                        p.grad = g
                    optimizer.step()
                
                loss_val = float(np.mean(loss_esd_vals))
                loss_attn_val = float(np.mean(loss_attn_vals))
            else:
                loss_val, loss_attn_val = 0.0, 0.0
            
            desc = f"Clean (Batch={len(batch_c)})"

        pbar.set_postfix(mode=desc, loss=loss_val, attn=loss_attn_val)

    # Save
    os.makedirs(args.save_path, exist_ok=True)
    filename = f"{args.concept_class}_Hybrid_mus{args.mus_erase_scale}_iter{args.iterations}.safetensors"
    save_full_path = os.path.join(args.save_path, filename)
    
    esd_param_dict = {name: param for name, param in zip(esd_param_names, esd_params)}
    save_file(esd_param_dict, save_full_path)
    print(f"--> Saved hybrid model to {save_full_path}")