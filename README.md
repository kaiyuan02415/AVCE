# AVCE: Auditing-Aware Unlearning for Verifiable Concept Erasure in Diffusion Models

> *No Concept Escapes the Audit: Auditing-Aware Unlearning for Verifiable Concept
> Erasure in Diffusion Models* — official implementation.

Existing concept-erasure methods mostly intervene at the **output level** (prompt
filtering, local weight edits). At the **representation level** the model still
retains identifiable features of the target concept, leaving it vulnerable to
relearning and adversarial attacks, and detectable by latent-space auditing.

**AVCE** grounds erasure in the model's internal representations and audits the
internal vulnerability landscape to achieve erasure that withstands both
adversarial attacks *and* latent-space auditing, while scaling to multi-concept
removal. On SD 1.5 / SDXL / FLUX 1.0 it reduces attack success rates by **>5×**
and improves auditing scores by **3.8×** over the strongest baselines, while
preserving generation quality.

---

## Method overview

AVCE runs as a single self-contained pipeline ([`avce/train_avce.py`](avce/train_avce.py))
with two stages that together realize the three components from the paper:

| Paper component | Where in the code |
|---|---|
| **§4.2 Analytical Editing** — closed-form cross-attention erasure with a *Semantic Modulator* (information-theoretic, entropy + mutual-information soft-masking) that protects unrelated channels | **Stage 1 — MUS Global Erasure** (`mus_edit_stage`): one-shot analytical rewrite of every `attn2` `to_k`/`to_v` matrix |
| **§4.1 Verifiable Vulnerability Anchor** — locate the weakest geometric point in embedding space rather than editing at the original concept embedding | **PAIA adversarial probing** (`run_paia_attack`): per-iteration soft-prompt optimization that finds an embedding which revives the concept |
| **§4.3 Audit-Guided Refinement** — fine-tune with pathway-level auditing signals; scale to many concepts via difficulty-aware sampling + orthogonal gradient projection | **Stage 2 — M2 Robust Fine-tuning**: adversarial defense loss + attention consolidation + **PCGrad** (orthogonal gradient projection) + **Hard Concept Mining** / dynamic loss weighting + **DTS** (Beta-distribution dynamic timestep scheduling) |

Stage 1 produces a structurally safe initialization in closed form; Stage 2
closes residual non-linear leakage and resolves inter-concept interference for
multi-concept erasure.

---

## Repository layout

```
Official/
├── avce/                       # the AVCE method (self-contained)
│   ├── train_avce.py           #   main entry point (Stage 1 + Stage 2)
│   ├── utils/sdxl_utils.py     #   patched SDXL pipeline call (run-till-timestep)
│   └── scripts/                #   ready-to-run training examples
│       ├── erase_object.sh     #     multi-concept object unlearning (Imagenette-10)
│       ├── erase_art.sh        #     artistic-style unlearning (5 styles)
│       └── erase_nudity.sh     #     explicit-content unlearning
└── eval/                       # evaluation harness + eval datasets
    ├── benchmarking/           #   Unlearn-Acc, artist CLIP, NudeNet, COCO CLIP/FID
    ├── audit/                  #   CRS / CCS latent-space auditing (cce.py)
    ├── attacks/                #   ASR1/2/3 robustness attacks (Ring-A-Bell, UnlearnDiffAtk, P4D)
    ├── datasets/               #   evaluation prompt sets (CSV/TXT)
    ├── utils/                  #   model loading / prompt helpers
    └── eval.sh                 #   example evaluation commands
```

> This is a clean release: training data, model weights, and generated results
> are **not** included — only the method, the evaluation code, and the evaluation
> prompt sets.

---

## Installation

```bash
pip install -r requirements.txt
```

The base model is downloaded automatically from the HuggingFace Hub
(`stabilityai/stable-diffusion-xl-base-1.0`). Training uses `bfloat16` and fits
on a single 48 GB GPU (NVIDIA RTX A6000 used in the paper).

---

## Training

Run one of the example scripts, or call the entry point directly:

```bash
# Multi-concept object unlearning (10-class Imagenette)
bash avce/scripts/erase_object.sh

# Or directly — single concept:
cd avce
python train_avce.py \
  --erase_concept "golf ball" \
  --concept_class "golf_ball" \
  --concept_type "object" \
  --esd_method "esd-x-strict" \
  --iterations 100 --lr 2e-5 --negative_guidance 0.4 \
  --adv --adv_steps 5 --adv_lr 2e-4 --dts \
  --concept_batch 1 \
  --mus_p 2 --mus_alpha_min 0.8 --mus_entropy_samples 20 \
  --save_path checkpoints/
```

Pass a comma-separated `--erase_concept` to erase multiple concepts at once;
`--concept_batch` then controls the per-step concept micro-batch used by Hard
Concept Mining + PCGrad. Key flags:

| Flag | Meaning |
|---|---|
| `--erase_concept` | concept(s) to erase, comma-separated for multi-concept |
| `--concept_type` | `object` or `art` (controls prompt templates for Stage 1) |
| `--skip_mus` | skip Stage 1 and refine from the base model |
| `--mus_p`, `--mus_alpha_min`, `--mus_entropy_samples` | Semantic Modulator (Stage 1) |
| `--adv`, `--adv_steps`, `--adv_lr` | PAIA adversarial probing / defense (Stage 2) |
| `--dts` | Beta-distribution dynamic timestep scheduling |
| `--concept_batch` | per-step concept micro-batch for multi-concept consolidation |
| `--lambda_attn` | weight of the attention-consolidation term |

The output is a `.safetensors` file containing the edited `attn2` `to_k`/`to_v`
weights, which can be loaded back into SDXL for inference and evaluation.

---

## Evaluation

Edit the `CKPT` path in [`eval/eval.sh`](eval/eval.sh) and run the relevant
section. The harness covers:

- **Object unlearning** — `benchmarking/object_erase_sdxl.py` (Unlearn Acc via a pretrained ResNet50).
- **Artistic style** — `benchmarking/artist_erasure_s_sdxl.py` (CLIP_a / CLIP_c / CLIP_d).
- **Explicit content** — `benchmarking/nudity_eval_only_sdxl.py` (NudeNet detections on I2P).
- **Generation quality** — `benchmarking/eval_coco_clip_sdxl.py` (MS-COCO CLIP score / FID).
- **Latent-space auditing** — `audit/cce.py` (Concept Retrieval Score / Concept Confidence Score).
- **Robustness** — `attacks/` (Ring-A-Bell, UnlearnDiffAtk, P4D → ASR1/2/3); see [`eval/attacks/README.md`](eval/attacks/README.md).

### Evaluation datasets (`eval/datasets/`)

| File(s) | Use |
|---|---|
| `imagenette.csv`, `imagenet-*.csv` | object-unlearning prompts |
| `art_5.csv`, `test_<artist>.csv` | artistic-style prompts |
| `Nudity_ring-a-bell.csv` | explicit-content / attack prompts |
| `coco_30k.csv`, `coco_prompts.txt` | MS-COCO prompts for CLIP / FID |
| `humans.txt`, `things.txt`, `common_scenes.txt`, `mv_prompts_*.txt` | auxiliary preservation prompts |

---

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). The base diffusion
models and the third-party attack frameworks under `eval/attacks/` carry their
own licenses — please consult them before use.
