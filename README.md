# Action/Event-Focused Captioning: A Three-Model Comparison

This repository contains a Computer Vision project for **action/event-focused image and video captioning**.  
The goal is to convert long descriptive captions into **short action-focused captions** that can be used to build simple video activity timelines.

Instead of producing captions such as:

```text
a man in a red shirt walking down the street
```

the models are fine-tuned to generate compact action labels such as:

```text
a man walking
```

The project compares three image-captioning architectures before and after fine-tuning:

- **BLIP**
- **ViT-GPT2**
- **Microsoft GIT**

---

## Project Motivation

General image-captioning models often generate object-rich descriptions.  
For video activity timelines, we need shorter labels that focus on the visible action.

Example:

```text
Original descriptive caption:
a man in a red shirt walking down the street

Target action caption:
a man walking
```

This project investigates whether different captioning architectures still behave differently after being adapted to the same action-captioning task.

---

## Repository Structure

```text
Action-Event-Focused-Captioning/
│
├── BLIP/
│   └── blip-coco-action-caption-finetuning.ipynb
│
├── Vit-GPT2/
│   └── vit-gpt2.ipynb
│
├── README.md
└── .gitattributes
```

Depending on the local version of the repository, the notebook filenames may differ slightly.

---

## Project Goal

The main goal is to build a system that:

1. Takes images or sampled video frames as input.
2. Generates short action/event-focused captions.
3. Compares original pretrained models with fine-tuned versions.
4. Applies the fine-tuned models to video frames.
5. Produces a simple activity timeline.

The video pipeline is frame-based:

```text
Input video
   ↓
Sample frames
   ↓
Generate action caption per frame
   ↓
Remove repeated captions
   ↓
Create activity timeline
```

---

## Dataset

The project uses a converted COCO action-caption dataset.

### COCO Train2014 Images

Kaggle dataset:

```text
https://www.kaggle.com/datasets/torrentbrave/coco-train2014-image
```

Expected Kaggle path:

```text
/kaggle/input/datasets/torrentbrave/coco-train2014-image
```

This dataset provides the actual COCO Train2014 image files.

---

### COCO Action Captions Training Ready

Kaggle dataset:

```text
https://www.kaggle.com/datasets/uwellcome/coco-action-captions-training-ready
```

Expected Kaggle path:

```text
/kaggle/input/datasets/uwellcome/coco-action-captions-training-ready
```

The important column is:

```text
action_caption
```

This column is used as the training target for all models.

---

### Demo Videos

Some notebooks also use demo videos for generating activity timelines.

Example paths:

```text
/kaggle/input/datasets/uwellcome/mydemo
/kaggle/input/datasets/nawfal123/demo-1/coffee_maker.mp4
```

---

## Data Preparation

The original COCO captions were converted into short action-focused captions.

Example conversion:

| Original caption | Action caption |
|---|---|
| A man in a red shirt walking down the street | a man walking |
| A person riding a motorcycle | person riding motorcycle |
| A woman sitting on a bench | woman sitting |

The released action-caption CSV contains **23,117 action-caption rows** used across the project.

Different notebooks may use slightly different train/validation splits:

| Model | Train / Validation Split |
|---|---|
| BLIP | 90/10 split by `image_id` |
| GIT | 90/10 split by `image_id` |
| ViT-GPT2 | fixed positional split, up to 20,000 train and 3,000 validation samples |

For BLIP and GIT, splitting by `image_id` helps reduce image leakage between training and validation.

---

## Models Compared

### 1. BLIP

Base model:

```text
Salesforce/blip-image-captioning-base
```

BLIP is a vision-language model designed for image captioning and vision-language understanding.

Training setup used in the project:

```text
Model: Salesforce/blip-image-captioning-base
Task: image → action_caption
Vision backbone: frozen
Batch size: 4 to 6
Learning rate: 5e-5 or smaller depending on notebook version
Hardware: Kaggle T4 GPU
```

---

### 2. ViT-GPT2

Base model:

```text
nlpconnect/vit-gpt2-image-captioning
```

ViT-GPT2 is implemented as a `VisionEncoderDecoderModel`:

```text
ViT encoder → visual features
GPT-2 decoder → generated caption
```

Training setup used in the project:

```text
Task: image → action_caption
ViT encoder: frozen
GPT-2 decoder: fine-tuned
Batch size: 4
Gradient accumulation: 2
Learning rate: 1e-5
Epochs: up to 3
Hardware: Kaggle T4 GPU
```

---

### 3. Microsoft GIT

Base model:

```text
microsoft/git-base
```

GIT is a generative image-to-text model that uses image patches as visual input for a causal language model.

Training setup used in the project:

```text
Task: image → action_caption
Visual backbone: frozen
Batch size: 4
Learning rate: 5e-5
Hardware: local RTX 4070 laptop / GPU environment
```

---

## General Pipeline

All three models follow the same core idea:

```text
COCO images + action-caption CSV
        ↓
Match image files with action captions
        ↓
Load pretrained captioning model
        ↓
Freeze visual backbone
        ↓
Fine-tune text generation on action_caption
        ↓
Evaluate original vs fine-tuned model
        ↓
Apply fine-tuned model to video frames
        ↓
Generate activity timeline
```

---

## Evaluation

Each model is evaluated before and after fine-tuning.

The reference caption is always:

```text
action_caption
```

The original long COCO caption is not used as the target for evaluation.

Metrics used:

- **BLEU-1**
- **BLEU-2**
- **METEOR**
- **ROUGE-L**

These metrics measure word overlap between the predicted caption and the action-caption reference.

---

## Combined Results

| Model | BLEU-1 | BLEU-2 | METEOR | ROUGE-L |
|---|---:|---:|---:|---:|
| BLIP zero-shot | 0.207 | 0.125 | 0.313 | 0.279 |
| BLIP fine-tuned | 0.483 | 0.361 | 0.444 | 0.517 |
| ViT-GPT2 zero-shot | 0.245 | 0.133 | 0.477* | 0.344 |
| ViT-GPT2 fine-tuned | 0.309 | 0.186 | 0.517* | 0.403 |
| GIT zero-shot | 0.216 | 0.122 | 0.229 | 0.261 |
| GIT fine-tuned | 0.485 | 0.354 | 0.452 | 0.527 |

\* ViT-GPT2 was evaluated with a different METEOR backend / protocol, so its METEOR value should be interpreted mainly as a within-model improvement rather than a direct comparison with BLIP and GIT.

---

## Interpretation of Results

Fine-tuning improves all three models.

Key observations:

- **BLIP** improves strongly after fine-tuning and remains well grounded on video frames.
- **GIT** also improves strongly and achieves the best ROUGE-L score in the reported comparison.
- **ViT-GPT2** improves after fine-tuning but remains weaker than BLIP and GIT on most word-overlap metrics.
- BLIP and GIT converge to a similar operating point after fine-tuning.
- Architecture still affects grounding quality on real video frames.

---

## Video Activity Timeline Demo

The fine-tuned models are also tested on a short kitchen / coffee-maker video.

The video is sampled into frames, and each frame is captioned by the fine-tuned models.

Example aligned timeline:

| Time | BLIP | ViT-GPT2 | GIT |
|---:|---|---|---|
| 0s | a coffee machine sitting | a fire hydrant sitting on top of a counter top | coffee maker sitting |
| 2s | a close up of a machine | a metal object sitting on top of a stove top | a spoon sitting |
| 4s | a person holding a coffee | a person holding a coffee pot up to the camera | a person holding a spoon |
| 6s | a coffee machine sitting | a clock mounted to the side of a wall | a faucet sitting |
| 8s | a close up of a coffee | a clock hanging off the side of a wall | a faucet sitting |
| 10s | coffee being poured | a cup sitting on top of a counter top | drinking water coffee is sitting |

This demo shows that video frames can be out-of-distribution for still-image captioning models.  
BLIP stays the most on-theme for the coffee-maker video, while ViT-GPT2 sometimes drifts to visually similar COCO objects.

---

## Action Timeline Post-Processing

For the ViT-GPT2 coffee-maker demo, an additional action-focused post-processing step was used.

This step maps raw object captions into cleaner action labels such as:

```text
coffee maker waiting
person using coffee maker
coffee preparation in progress
person operating coffee maker
person pouring coffee
cup placed near coffee maker
```

This helps produce a cleaner activity timeline, especially when raw captions are object-heavy or out-of-domain.

---

## How to Run the Notebooks

### Recommended Environment

Use Kaggle Notebooks with:

```text
Accelerator: GPU T4
Internet: ON
```

Avoid reinstalling core libraries such as PyTorch or Transformers unless necessary.

---

### Add Required Kaggle Inputs

Add the following datasets to the Kaggle notebook:

```text
COCO Train2014 image
coco-action-captions-training-ready
demo videos if running video inference
```

---

### Run Steps

1. Open the model notebook.
2. Attach the required Kaggle datasets.
3. Enable GPU.
4. Run the notebook cells in order.
5. After fine-tuning, save:
   - fine-tuned model
   - prediction CSVs
   - metrics CSVs
   - video timeline CSVs
6. Download or save heavy outputs as Kaggle notebook outputs.

---

## Requirements

Main Python packages:

```text
torch
transformers
pandas
numpy
Pillow
opencv-python
matplotlib
tqdm
nltk
rouge-score
```

If running locally:

```bash
pip install torch transformers pandas numpy pillow opencv-python matplotlib tqdm nltk rouge-score
```

---

## Files Not Included

Large files should not be committed to GitHub.

Do not upload:

```text
COCO image folders
video files
fine-tuned model folders
model zip files
Kaggle working directories
cached prediction CSV files
backup zip files
```

These should be stored in Kaggle datasets, notebook outputs, or external storage.

---

## Limitations

- The system is frame-based and does not perform full temporal video reasoning.
- Each video frame is captioned independently.
- Some actions require motion context and may be hard to infer from a single frame.
- COCO action captions are derived from image captions, not from a true video-action dataset.
- Word-overlap metrics such as BLEU and ROUGE may not fully capture semantic correctness.
- Domain-specific videos, such as coffee-maker clips, may require post-processing or domain adaptation.

---

## Future Work

Possible extensions:

- Use real video action datasets such as Kinetics, UCF101, or ActivityNet.
- Add temporal models such as VideoMAE, TimeSformer, DINOv2 + temporal transformer, or V-JEPA.
- Use meaningful frame selection instead of fixed-interval sampling.
- Remove blurry or low-quality frames before captioning.
- Add temporal smoothing across consecutive frame predictions.
- Use multiple references for more reliable evaluation.
- Replace rule-based post-processing with a learned action classifier.
- Combine two models side-by-side for stronger video timeline predictions.

---

## Project Summary

This project fine-tunes three pretrained image-captioning architectures — BLIP, ViT-GPT2, and Microsoft GIT — on the same action-caption dataset.  
The fine-tuned models are evaluated before and after adaptation and then applied to sampled video frames to generate simple action timelines.

The results show that fine-tuning consistently improves action-caption performance. BLIP and GIT achieve the strongest results, while ViT-GPT2 provides a useful lightweight baseline and demonstrates the challenges of applying still-image captioning models to real video frames.
