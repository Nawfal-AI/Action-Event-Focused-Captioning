# Temporal Video Action Captioning with Microsoft GIT

This repository contains a notebook for **frame-based video action understanding** using a fine-tuned **Microsoft GIT (Generative Image-to-text Transformer)** model.

The project focuses on generating **short action-focused captions** from images and video frames. Instead of producing long scene descriptions, the system predicts concise action captions such as:

```text
a man walking
a woman standing
a cat sitting
a man holding a sword
```

The final output for a video is a timestamped action timeline.

---

## Project Goal

The goal of this project is to build a system that analyzes input videos and identifies the actions happening over time.

Because the available training data is image-based, the system follows a frame-based approach:

```text
Input video
   ↓
Sample video frames
   ↓
Generate action caption for each frame
   ↓
Remove repeated actions
   ↓
Produce action timeline
```

This project is part of a USI Computer Vision course group assignment. The teammate fine-tuned BLIP on the same dataset; this repository provides the GIT counterpart for direct comparison. All hyperparameters, data splits, and evaluation protocols are aligned so the only variable is the model architecture.

---

## Main Notebook

```text
git-coco-action-caption-finetuning.ipynb
```

The notebook includes:

- loading COCO Train2014 images
- loading action-focused captions
- matching image names and image IDs
- fine-tuning Microsoft GIT on action captions
- evaluating original vs fine-tuned model
- comparing GIT with BLIP results
- showing qualitative examples with input images
- testing on a demo video
- generating a timestamped action timeline

---

## Datasets

The project uses the following datasets.

### 1. COCO Train2014 Images

Kaggle dataset:

```text
https://www.kaggle.com/datasets/torrentbrave/coco-train2014-image
```

Only the ~8,120 images referenced by the action-caption CSV are required (not the full 82K). A download helper is provided:

```bash
python _setup/download_images_subset.py \
    --csv data/coco_action/coco_50k_relaxed_action_captions_training_ready.csv \
    --out_dir data/coco/train2014 --workers 40
```

---

### 2. COCO Action Captions Training Ready

Kaggle dataset:

```text
https://www.kaggle.com/datasets/uwellcome/coco-action-captions-training-ready
```

This dataset provides the CSV file containing image references and action-focused captions.

The main target column used for fine-tuning is:

```text
action_caption
```

The CSV contains 23,117 rows covering 8,120 unique images, with two conversion types:

```text
clear_action     9,373 rows   (e.g. "two people walking")
weak_static     13,744 rows   (e.g. "metal balls sitting")
```

All rows are used for training to align with the teammate's BLIP baseline.

---

### 3. Demo Video

A short demo video for the video action timeline:

```text
data/videos/demo.mp4
```

---

## Requirements

The notebook was developed and tested on a local machine with an **RTX 4070 Laptop GPU (8 GB VRAM)**.

Main Python packages:

```text
torch>=2.5
transformers==4.44.2
accelerate==0.34.2
numpy<2
pandas
Pillow
opencv-python<4.11
tqdm
nltk
rouge-score
```

Install with:

```bash
pip install "transformers==4.44.2" "accelerate==0.34.2" "numpy<2" \
    "opencv-python<4.11" pillow tqdm pandas requests nltk rouge-score
```

> **Note:** `transformers` must be pinned to 4.44.x. Later versions deprecate the `tokenizer=` kwarg and break GIT's `or_mask_function` path on torch < 2.6. `numpy` must be < 2 to avoid ABI incompatibility with scipy 1.11 (used transitively by nltk).

---

## Model

The model used in this project is:

```text
microsoft/git-base
```

GIT (Generative Image-to-text Transformer) is a `AutoModelForCausalLM` that uses:

- **CLIP ViT image encoder** for image feature extraction (~86M params)
- **Visual projection bridge** (Linear + LayerNorm, ~590K params)
- **Transformer text decoder** with joint self-attention over `[vision_tokens, text_tokens]` (~90M params)

Unlike BLIP which uses separate cross-attention layers for vision-text fusion, GIT shares QKV weights across modalities in a single self-attention stack.

Total parameters: **176.6M** (90.2M trainable after freezing vision components).

---

## Fine-Tuning Strategy

The model is fine-tuned using:

```text
image → action_caption
```

### Training Configuration (aligned with BLIP baseline)

| Parameter | Value |
|---|---|
| Model | `microsoft/git-base` |
| Training samples | 20,789 |
| Validation samples | 2,328 |
| Batch size | 4 |
| Learning rate | 5e-5 |
| Epochs | 1 |
| Max length | 32 |
| Warmup ratio | 0.1 |
| Optimizer | AdamW (eps=1e-6) |
| Scheduler | linear warmup |
| Precision | fp32 |
| num_workers | 0 |

### GIT-Specific Stability Fixes

GIT requires three additional measures that BLIP does not need, due to its joint self-attention architecture:

1. **Freeze BOTH `image_encoder` AND `visual_projection`**

   BLIP's `model.vision_model.parameters()` covers the complete vision pathway including the cross-modal projection. For GIT, these are two separate sub-modules. Leaving `visual_projection` trainable causes gradient concentration in this small (~590K param) bridge layer, leading to NaN at step 100–300.

2. **AdamW `eps=1e-6`** (vs default 1e-8)

   Short action captions are mostly `-100` padding in labels. AdamW's second moment stays near zero on cross-modal attention params with sparse gradient signal; default eps=1e-8 allows the update step to explode.

3. **Gradient clipping + NaN tripwire**

   `clip_grad_norm=1.0` prevents large gradient spikes. Before each `optimizer.step()`, the code checks that both loss and grad_norm are finite — if either is NaN/Inf, the update is skipped entirely to prevent permanently corrupting weights.

These fixes are documented in detail in `PROJECT-STATUS.md` §4.

---

## Evaluation

The evaluation compares model predictions against:

```text
action_caption
```

Metrics used (identical to BLIP notebook):

- **BLEU-1** — unigram precision
- **BLEU-2** — bigram precision
- **METEOR** — alignment-based
- **ROUGE-L** — longest common subsequence

Evaluation protocol: 200 samples from the validation set, `random_state=42`, `num_beams=5`, `max_new_tokens=24`.

---

## Results

### Fine-tuned Model Comparison

| Metric | BLIP zero-shot | BLIP fine-tuned | GIT zero-shot | **GIT fine-tuned** |
|---|---|---|---|---|
| BLEU-1 | 0.2071 | 0.4831 | 0.2162 | **0.4852** |
| BLEU-2 | 0.1254 | 0.3609 | 0.1221 | **0.3541** |
| METEOR | 0.3130 | 0.4437 | 0.2293 | **0.4522** |
| ROUGE-L | 0.2791 | 0.5169 | 0.2609 | **0.5268** |

### Training Summary

```text
Train loss:  1.5752
Val loss:    1.4095
Skipped (non-finite) steps: 0
Training time: ~17 minutes on RTX 4070 Laptop (8 GB)
Peak VRAM usage: ~6 GB
```

---

## Video Action Timeline

For video inference, the notebook performs the following steps:

1. Load an input video.
2. Sample frames at a fixed interval (default: every 1 second).
3. Run the fine-tuned GIT model on each frame.
4. Generate action captions.
5. Remove repeated consecutive actions.
6. Output a clean timestamped action timeline.

Example output (from `data/videos/demo.mp4`):

```text
1. [0.0s] a person sitting
2. [1.0s] a woman standing
3. [2.0s] a person standing
4. [3.0s] a man holding a sword
5. [4.0s] a man standing
6. [8.0s] a woman sitting
7. [9.0s] a person sitting
```

---

## Project Structure

```text
git-coco-finetune/
│
├── README.md
├── git-coco-action-caption-finetuning.ipynb   ← main notebook
│
├── train_manual.py                            ← standalone training script
├── evaluate.py                                ← standalone evaluation script
├── setup_data_action.py                       ← data manifest builder
│
├── _setup/
│   ├── requirements.txt
│   ├── download_coco.sh
│   ├── download_images_subset.py              ← download only needed images
│   ├── build_notebook.py                      ← notebook generator
│   ├── make_demo_video.py
│   └── ...
│
├── data/
│   ├── coco_action/                           ← action-caption CSV
│   ├── coco/train2014/                        ← COCO images (~8,120)
│   ├── manifests_action/                      ← train/val JSON manifests
│   └── videos/                                ← demo video
│
├── checkpoints/
│   └── git-base-action-ft/                    ← saved fine-tuned model
│
├── results/                                   ← evaluation JSON outputs
├── logs/                                      ← training logs
│
├── vitgpt2/                                   ← ViT-GPT2 backup pipeline
│   ├── train_vitgpt2.py
│   ├── evaluate_vitgpt2.py
│   └── *.slurm
│
├── _archive/                                  ← deprecated HF Trainer scripts
│
├── train_manual.slurm                         ← Slurm job scripts (cluster)
└── eval.slurm
```

---

## Limitations

This project is a **frame-based video action-captioning pipeline**, not a full temporal video transformer.

The model is trained on image-caption pairs and then applied to sampled video frames. There is no temporal modeling — consecutive frames are captioned independently, and temporal coherence is approximated only by removing consecutive duplicate captions.

Additionally, GIT's joint self-attention architecture makes it more sensitive to noisy training data than BLIP's separate cross-attention design. The `weak_static` subset of the training data (e.g. "metal balls sitting", "the clean looking") produces high per-sample loss that can destabilize training without the stability fixes described above.

---

## Future Work

Possible improvements include:

- training on real video action datasets (e.g. ActivityNet, Kinetics)
- using temporal video transformers (TimeSformer, VideoMAE, Video Swin)
- improving frame selection with motion estimation instead of fixed-time sampling
- filtering `weak_static` training samples for cleaner gradient signal
- multi-epoch training with learning rate tuning
- evaluating with semantic similarity metrics (BERTScore, CLIPScore)

---

## Summary

This project fine-tunes Microsoft GIT on COCO action captions and applies the model to video frames to generate action-focused timelines. GIT achieves metrics comparable to the teammate's BLIP baseline (BLEU-1 0.4852 vs 0.4831, ROUGE-L 0.5268 vs 0.5169), validating that the architectural differences between joint self-attention (GIT) and cross-attention (BLIP) do not significantly affect final captioning quality — provided the GIT-specific training instabilities are properly addressed.
