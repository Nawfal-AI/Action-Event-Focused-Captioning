# Temporal Video Action Captioning with ViT-GPT2

This repository contains a Kaggle notebook for **frame-based video action understanding** using a fine-tuned **ViT-GPT2 image captioning model**.

The project focuses on generating **short action-focused captions** from images and video frames. Instead of producing long scene descriptions, the system predicts concise action captions such as:

```text
person walking
person holding a cup
person using coffee maker
coffee preparation in progress
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

This makes the project a practical baseline for video action understanding using image-captioning models.

---

## Main Notebook

```text
vit-gpt2.ipynb
```

The notebook includes:

- loading COCO Train2014 images
- loading action-focused captions
- matching image names and image IDs
- fine-tuning ViT-GPT2 on action captions
- evaluating original vs fine-tuned model
- comparing ViT-GPT2 with BLIP and Microsoft GIT results
- showing qualitative examples with input images
- testing on demo videos
- generating a timestamped action timeline
- caching heavy outputs to avoid repeated computation

---

## Datasets

The project uses the following Kaggle datasets.

### 1. COCO Train2014 Images

Kaggle dataset:

```text
https://www.kaggle.com/datasets/torrentbrave/coco-train2014-image
```

Expected Kaggle input path:

```text
/kaggle/input/datasets/torrentbrave/coco-train2014-image
```

This dataset provides the COCO Train2014 image files used for training and validation.

---

### 2. COCO Action Captions Training Ready

Kaggle dataset:

```text
https://www.kaggle.com/datasets/uwellcome/coco-action-captions-training-ready
```

Expected Kaggle input path:

```text
/kaggle/input/datasets/uwellcome/coco-action-captions-training-ready
```

This dataset provides the CSV file containing image references and action-focused captions.

The main target column used for fine-tuning is:

```text
action_caption
```

The original COCO captions are used only for reference and comparison.

---

### 3. Demo Video Dataset: demo-1

Expected Kaggle input path:

```text
/kaggle/input/datasets/nawfal123/demo-1
```

Example video used in the notebook:

```text
/kaggle/input/datasets/nawfal123/demo-1/coffee_maker.mp4
```

This dataset is used for the coffee-maker action timeline demo.

---

### 4. Demo Video Dataset: mydemo

Kaggle dataset:

```text
https://www.kaggle.com/datasets/uwellcome/mydemo
```

Expected Kaggle input path:

```text
/kaggle/input/datasets/uwellcome/mydemo
```

This dataset contains additional demo videos used for testing the video action timeline.

---

## Requirements

The notebook was developed and tested in **Kaggle Notebooks** using a **T4 GPU**.

Recommended Kaggle settings:

```text
Accelerator: GPU T4
Internet: ON
```

The notebook is designed to use the default Kaggle environment. Avoid reinstalling PyTorch or Transformers unless necessary, because changing core packages can break the Kaggle runtime.

Main Python packages used:

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
```

If running outside Kaggle, install the dependencies with:

```bash
pip install torch transformers pandas numpy pillow opencv-python matplotlib tqdm nltk
```

---

## Model

The main model used in this project is:

```text
nlpconnect/vit-gpt2-image-captioning
```

This is a `VisionEncoderDecoderModel` that combines:

- **ViT encoder** for image feature extraction
- **GPT-2 decoder** for caption generation

During fine-tuning, the ViT encoder is frozen and the GPT-2 decoder is trained to generate short action-focused captions.

---

## Fine-Tuning Strategy

The model is fine-tuned using:

```text
image → action_caption
```

The original model produces general image captions, while the fine-tuned model is trained to produce shorter action descriptions.

Training settings used in the notebook include:

```text
Training samples: up to 20,000
Validation samples: up to 3,000
Batch size: 4
Gradient accumulation steps: 2
Learning rate: 1e-5
Epochs: configurable
Optimizer: AdamW
Scheduler: linear warmup schedule
```

The best model is selected using validation loss.

---

## Evaluation

The evaluation compares the model predictions against:

```text
action_caption
```

The original long COCO captions are not used as the reference because the goal is action-focused captioning.

Metrics used:

- **BLEU-1**
- **BLEU-2**
- **METEOR-style score**
- **ROUGE-L**

The notebook evaluates:

```text
Original ViT-GPT2 vs action_caption
Fine-tuned ViT-GPT2 vs action_caption
```

It also includes comparison results for teammates' BLIP and Microsoft GIT models.

---

## Model Comparison

The notebook compares:

```text
ViT-GPT2 Original
ViT-GPT2 Fine-tuned
BLIP Original
BLIP Fine-tuned
Microsoft GIT Original
Microsoft GIT Fine-tuned
```

This comparison shows how fine-tuning changes each model from general captioning to action-focused captioning.

---

## Video Action Timeline

For video inference, the notebook performs the following steps:

1. Load an input video.
2. Sample frames every few seconds.
3. Run the fine-tuned ViT-GPT2 model on each frame.
4. Generate raw captions.
5. Convert raw captions into action-focused captions.
6. Remove repeated consecutive actions.
7. Save a clean timestamped action timeline.

Example output:

```text
0.0s    coffee maker waiting
2.0s    person using coffee maker
4.0s    coffee preparation in progress
6.0s    person operating coffee maker
```

---

## Coffee Maker Demo

The notebook includes a specific demo using:

```text
/kaggle/input/datasets/nawfal123/demo-1/coffee_maker.mp4
```

Because the model is trained on COCO images, some raw predictions may describe objects instead of actions. For example, the model may confuse a coffee maker with COCO objects such as a clock, stove, or fire hydrant.

To improve the demo output, the notebook uses action-focused post-processing that maps raw captions into action labels such as:

```text
coffee maker waiting
person using coffee maker
coffee preparation in progress
person operating coffee maker
person pouring coffee
cup placed near coffee maker
```

This produces a cleaner and more useful action timeline.

---

## Heavy Outputs and Caching

The notebook includes caching to avoid rerunning expensive steps every time a Kaggle session restarts.

Cached outputs may include:

```text
original model predictions
fine-tuned model predictions
evaluation metrics
comparison CSV files
timeline CSV files
fine-tuned model folder
backup zip file
```

The notebook can create:

```text
group_v_heavy_outputs_backup.zip
```

This file should be downloaded or saved from Kaggle outputs to avoid rerunning fine-tuning and prediction generation.

---

## Files Not Included in This Repository

Large files are not included in this GitHub repository.

Do not upload:

```text
COCO image folders
video files
fine-tuned model folders
model zip files
Kaggle working output folders
cached prediction CSV files
backup zip files
```

These files should be stored in Kaggle datasets, Kaggle notebook outputs, or another external storage location.

---

## Suggested Repository Structure

```text
temporal-video-action-captioning/
│
├── README.md
├── vit-gpt2.ipynb
├── requirements.txt
└── .gitignore
```

---

## Limitations

This project is a **frame-based video action-captioning pipeline**, not a full temporal video transformer.

The model is trained on image-caption pairs and then applied to sampled video frames. Temporal understanding is approximated using frame sampling, motion estimation, and action timeline smoothing.

Future work can improve the system by using real temporal video models such as:

- V-JEPA
- DINOv2 with temporal modeling
- VideoMAE
- TimeSformer
- Video Swin Transformer

---

## Future Work

Possible improvements include:

- training on real video action datasets
- using temporal transformers for motion understanding
- improving frame selection instead of fixed-time sampling
- removing blurry or uninformative frames
- replacing rule-based post-processing with a learned action classifier
- using CLIP or another vision-language model for action mapping
- evaluating with semantic similarity metrics in addition to BLEU and ROUGE

---

## Summary

This project fine-tunes ViT-GPT2 on COCO action captions and applies the model to video frames to generate action-focused timelines. It provides a practical baseline for action-only video understanding and compares ViT-GPT2 with BLIP and Microsoft GIT models.
