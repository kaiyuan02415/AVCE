# Robustness Attacks (ASR1 / ASR2 / ASR3)

AVCE is evaluated against three adversarial attacks that try to *recover* an
erased concept from an unlearned model. The **Attack Success Rate (ASR)** is the
fraction of attacked prompts that revive the target concept.

| Tag  | Attack          | Folder            | Upstream |
|------|-----------------|-------------------|----------|
| ASR1 | Ring-A-Bell     | `ring-a-bell/`    | https://github.com/chiayi-hsu/Ring-A-Bell |
| ASR2 | UnlearnDiffAtk  | `unlearndiffatk/` | https://github.com/OPTML-Group/Diffusion-MU-Attack |
| ASR3 | P4D             | `p4d/`            | https://github.com/joycenerd/P4D |

> These are lightly adapted copies of the upstream repositories. Large model
> checkpoints, generated `attack_results/`, and bundled image datasets have been
> removed to keep this release small — only the attack code, configs, and the
> small prompt/concept-vector files needed to run the attacks are kept. Install
> each attack's own extra dependencies as documented upstream, and download any
> required checkpoints from the original projects before running.

### Ring-A-Bell (ASR1)

```bash
cd attacks/ring-a-bell
python ring_attack_s.py \
  --exp_name "avce" \
  --output_dir "./attack_results/img-10" \
  --concepts "parachute" "golf ball" "garbage truck" "cassette player" "church" \
             "tench" "english springer" "french horn" "chain saw" "gas pump" \
  --model_type sdxl \
  --model_path "/path/to/avce_checkpoint.safetensors" \
  --population 200 --generation 3000 --eval_samples 100 --save_images
```

### UnlearnDiffAtk (ASR2) and P4D (ASR3)

See `unlearndiffatk/README.md` and `p4d/GUIDES.md` for the per-attack entry
points and config files. Point the model/checkpoint argument at your trained
AVCE `.safetensors` file.
