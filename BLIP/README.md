# BLIP COCO Action-Caption Fine-Tuning

This repository contains a Kaggle notebook for fine-tuning a pre-trained BLIP image-captioning model to generate short action/event-focused captions.

Instead of generating a full descriptive caption such as:

```text
a man in a red shirt walking down the street
```

the fine-tuned model is trained to generate a shorter action-focused caption such as:

```text
a man walking
```

The notebook includes model fine-tuning, validation evaluation, comparison with the original BLIP model, single-image prediction, single-video prediction, and batch video timeline prediction saved to CSV.

---

## Notebook

Main notebook:

```text
blip-coco-action-caption-finetuning.ipynb
```

---

## Project Goal

The goal of this project is to adapt a general image-captioning model into an action/event-focused captioning model.

The original BLIP task is:

```text
image -> descriptive caption
```

Our adapted task is:

```text
image -> short action caption
```

This is useful for simple video event understanding, where sampled video frames can be converted into an action timeline.

Example:

```text
[0.0s] a man walking
[2.0s] a person standing
[4.0s] a person riding bicycle
```

---

## Base Model

The base model used in this notebook is:

```text
Salesforce/blip-image-captioning-base
```

It is loaded from Hugging Face using the Transformers library:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

We use this model because it is already pre-trained for image captioning. The goal is not to train a captioning model from scratch, but to gently adapt the existing model so that it produces shorter action/event captions.

---

## Fine-Tuned Model Weights

The final fine-tuned model weights are available as a Kaggle dataset:

```text
https://www.kaggle.com/datasets/uwellcome/final-weight
```

To use the fine-tuned model directly in Kaggle without retraining, add this dataset to the notebook:

```text
Add Input -> search for: uwellcome/final-weight
```

After adding it, Kaggle usually mounts it at:

```text
/kaggle/input/final-weight
```

The model can then be loaded directly using:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

SAVED_MODEL_PATH = "/kaggle/input/final-weight"

processor = BlipProcessor.from_pretrained(SAVED_MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(SAVED_MODEL_PATH)
```

This allows users to run image and video prediction directly without retraining the model.

The final-weight folder should contain files such as:

```text
config.json
generation_config.json
model.safetensors
processor_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.txt
```

---

## Required Kaggle Inputs

The notebook expects the following Kaggle inputs.

### 1. COCO Train2014 Images

The image files should follow the COCO naming format:

```text
COCO_train2014_000000000127.jpg
COCO_train2014_000000000562.jpg
...
```

The notebook searches inside `/kaggle/input` for files starting with:

```text
COCO_train2014_
```

### 2. Action-Caption CSV

The notebook uses a converted action-caption CSV file.

Expected columns:

```text
image_id
file_name
original_caption
action_caption
```

Optional columns may also exist:

```text
conversion_type
keep_for_training
```

The `action_caption` column is the target caption used for fine-tuning.

Example:

| file_name | original_caption | action_caption |
|---|---|---|
| COCO_train2014_000000000127.jpg | A man walking down the street. | a man walking |
| COCO_train2014_000000000562.jpg | Two people riding motorcycles. | two people riding motorcycles |

### 3. Fine-Tuned Model Weights

Kaggle dataset:

```text
https://www.kaggle.com/datasets/uwellcome/final-weight
```

Expected Kaggle input path after adding it:

```text
/kaggle/input/final-weight
```

This input is required if you want to use the already fine-tuned model directly without retraining.

### 4. Optional Demo Videos

For video prediction and timeline generation, the notebook can use videos from a Kaggle input folder such as:

```text
/kaggle/input/datasets/uwellcome/demo-videos
```

Supported video formats:

```text
.mp4
.avi
.mov
.mkv
.webm
```

---

## Main Pipeline

The notebook follows this pipeline:

```text
COCO images + action-caption CSV
        ↓
Match CSV file_name with actual image paths
        ↓
Clean invalid or missing rows
        ↓
Train/validation split by image_id
        ↓
Load Salesforce/blip-image-captioning-base
        ↓
Fine-tune BLIP on action_caption targets
        ↓
Save fine-tuned model and processor
        ↓
Evaluate using BLEU-1, BLEU-2, METEOR, ROUGE-L
        ↓
Compare original BLIP vs fine-tuned BLIP
        ↓
Predict action captions for custom images
        ↓
Predict action timelines for videos
        ↓
Save batch video timelines to CSV
```

---

## Training Configuration

Main training settings used in the notebook:

```python
MODEL_NAME = "Salesforce/blip-image-captioning-base"

EPOCHS = 2
BATCH_SIZE = 6
LEARNING_RATE = 5e-7
MAX_LENGTH = 7
TRAIN_RATIO = 0.90

FREEZE_VISION_ENCODER = True
USE_AMP = False
```

The vision encoder is frozen by default to reduce memory usage and preserve the visual representation learned during BLIP pre-training.

A small learning rate is used because the task is close to the original BLIP task. The model already maps images to text; the fine-tuning only changes the output style from general descriptive captions to shorter action/event-focused captions.

---

## Dataset Preparation

The notebook loads the action-caption CSV and matches each row to the correct image using the `file_name` column.

Rows are removed if:

```text
action_caption is empty
image_path is missing
image file cannot be found
```

The processed training data is saved to:

```text
/kaggle/working/final_coco_action_caption_training_data.csv
```

The train/validation split is done by `image_id`, not by individual caption rows. This reduces leakage when multiple captions belong to the same image.

---

## Important Dataset Note

The original 50,000 rows are caption rows, not necessarily 50,000 unique images.

COCO usually has multiple captions for the same image. Therefore, the same image may appear more than once with different captions.

This is normal for image-captioning training.

To check the number of unique images:

```python
print("Total rows:", len(df))
print("Unique images:", df["file_name"].nunique())
print("Repeated image rows:", len(df) - df["file_name"].nunique())
```

---

## Model Fine-Tuning

The notebook defines a PyTorch dataset for image-action-caption pairs.

Each sample contains:

```text
image
action_caption
image_id
original_caption
```

The `BlipProcessor` prepares both the image and target caption.

The model is optimized using:

```python
torch.optim.AdamW
```

with a linear learning-rate scheduler and warmup:

```python
get_linear_schedule_with_warmup
```

---

## Saving the Fine-Tuned Model

After training, the notebook saves the fine-tuned model and processor to:

```text
/kaggle/working/blip_coco_action_caption_finetuned
```

It also creates a zipped version:

```text
/kaggle/working/blip_coco_action_caption_finetuned.zip
```

This saved folder can be uploaded as a Kaggle dataset and reused later as an input dataset.

In this project, the final model weights are available here:

```text
https://www.kaggle.com/datasets/uwellcome/final-weight
```

---

## Loading the Fine-Tuned Model for Inference

To use the already fine-tuned model directly:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVED_MODEL_PATH = "/kaggle/input/final-weight"

processor = BlipProcessor.from_pretrained(SAVED_MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(SAVED_MODEL_PATH)

model.to(DEVICE)
model.eval()
```

Use this option if you do not want to retrain the model.

---

## Image Prediction

The notebook includes a function for predicting an action caption from a single image:

```python
generate_action_caption_from_image(image, model, processor)
```

Example usage:

```python
from PIL import Image

IMAGE_PATH = "/kaggle/input/example-image/example.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")

pred = generate_action_caption_from_image(
    image=image,
    model=model,
    processor=processor
)

print("Predicted action caption:", pred)
```

---

## Video Prediction

The notebook includes a function for predicting an action timeline from a video:

```python
predict_video_action_timeline(
    video_path,
    model,
    processor,
    seconds_per_frame=2,
    max_frames=30
)
```

It works by:

1. Opening the video with OpenCV.
2. Sampling frames every few seconds.
3. Sending each sampled frame to the fine-tuned BLIP model.
4. Generating one action caption per sampled frame.
5. Returning a timeline.

Example output:

```text
[0.0s] man riding bicycle
[2.0s] person standing
[4.0s] person walking
```

---

## Removing Consecutive Duplicate Captions

Because nearby video frames may look similar, the model may generate repeated captions.

The notebook includes a function to remove consecutive duplicate captions:

```python
remove_consecutive_duplicate_captions(results)
```

Example:

```text
woman sitting
woman sitting
woman sitting
person walking
```

becomes:

```text
woman sitting
person walking
```

---

## Batch Video Prediction to CSV

The notebook can process all videos inside a folder and save the results to:

```text
/kaggle/working/grouped_video_action_timeline.csv
```

The CSV is grouped by video title. The video name appears once, followed by its timeline rows.

Example:

```text
video_title,time_label,time_seconds,action_caption
Bicycle Parking,,,
,0.0s,0.0,man riding bicycle
,2.0s,2.0,person standing
,4.0s,4.0,person parking bicycle

Breakfast Cereal Phone,,,
,0.0s,0.0,person holding phone
,2.0s,2.0,person eating
```

This format is useful for presentations and qualitative demo results.

---

## Evaluation Metrics

The notebook evaluates generated captions using:

```text
BLEU-1
BLEU-2
METEOR
ROUGE-L
```

The evaluation compares:

```text
reference action captions
predicted action captions
```

Example evaluation code:

```python
MAX_EVAL_SAMPLES = 200

eval_df = df_val.sample(
    n=min(MAX_EVAL_SAMPLES, len(df_val)),
    random_state=42
).reset_index(drop=True)
```

Using a fixed `random_state` makes the sampled validation examples reproducible.

---

## Original BLIP vs Fine-Tuned BLIP

The notebook compares two models:

```text
Original BLIP
Fine-tuned BLIP
```

The original BLIP model is:

```text
Salesforce/blip-image-captioning-base
```

The fine-tuned model is the model saved after training on the action-caption dataset.

Both models are evaluated against the same action-caption validation references.

---

## Reported Example Results

Example results obtained during the experiment:

| Model | BLEU-1 | BLEU-2 | METEOR | ROUGE-L |
|---|---:|---:|---:|---:|
| Original BLIP | 0.2071 | 0.1254 | 0.3130 | 0.2791 |
| Fine-tuned BLIP | 0.4831 | 0.3609 | 0.4437 | 0.5169 |

These results show that the fine-tuned model became more aligned with the action-caption target style.

---

## How to Run on Kaggle

1. Open the notebook in Kaggle.
2. Add the required Kaggle inputs:
   - COCO Train2014 images.
   - The action-caption CSV.
   - The fine-tuned model weights dataset if using inference directly:
     ```text
     https://www.kaggle.com/datasets/uwellcome/final-weight
     ```
   - Optional demo videos.
3. Enable GPU:
   ```text
   Settings -> Accelerator -> GPU
   ```
4. Run the notebook cells from top to bottom.

If you want to use the already fine-tuned model without retraining, make sure the final weights dataset is added as input and use:

```text
/kaggle/input/final-weight
```

---

## Repository Structure

Suggested GitHub structure:

```text
blip-coco-action-caption-finetuning/
│
├── README.md
├── blip-coco-action-caption-finetuning.ipynb
├── requirements.txt
│
├── outputs/
│   └── grouped_video_action_timeline.csv
│
└── poster/
    └── BLIP_A0_Landscape_Final_Poster.pdf
```

Large model files should not be uploaded directly to GitHub. Instead, use the Kaggle dataset link:

```text
https://www.kaggle.com/datasets/uwellcome/final-weight
```

---

## Requirements

Main libraries used:

```text
torch
transformers
pillow
opencv-python
pandas
numpy
tqdm
nltk
rouge-score
scikit-learn
```

Example `requirements.txt`:

```text
torch
transformers
pillow
opencv-python
pandas
numpy
tqdm
nltk
rouge-score
scikit-learn
```

---

## Important Notes

- The notebook does not train BLIP from scratch.
- It fine-tunes a pre-trained BLIP image-captioning model.
- The output is still captioning, but focused on actions/events.
- The video pipeline is frame-based.
- It does not perform full temporal reasoning across the whole video.
- Consecutive duplicate captions are removed to make the timeline cleaner.
- The quality of the model depends strongly on the quality of the converted `action_caption` labels.

---

## Limitations

- The model predicts from sampled frames independently.
- Some actions require motion understanding and may not be clear from a single frame.
- Repeated or visually similar frames may produce repeated captions.
- The action-caption dataset is derived from COCO image captions, not from a true video action dataset.
- Some generated captions may still be generic or imperfect.

---

## Future Work

Possible improvements:

- Use a real action/video dataset such as UCF101 or Kinetics.
- Use a video-based model instead of frame-only captioning.
- Improve action-caption conversion using stronger semantic rewriting.
- Add temporal smoothing across frames.
- Evaluate on more validation samples.
- Compare with other vision-language or video-captioning models.

---

## Credits

Base model:

```text
Salesforce/blip-image-captioning-base
```

Fine-tuned model weights:

```text
https://www.kaggle.com/datasets/uwellcome/final-weight
```

Libraries used:

```text
PyTorch
Hugging Face Transformers
Pillow
OpenCV
pandas
NLTK
rouge-score
tqdm
scikit-learn
```

Dataset source:

```text
COCO 2014 images and captions
```

Action-caption labels were created by converting original COCO captions into shorter action-focused captions.

---

## Suggested Report Description

You can describe the method as:

```text
We used the pre-trained Salesforce/blip-image-captioning-base model from Hugging Face and fine-tuned it on a converted COCO action-caption dataset. The goal was to adapt the model from general image captioning to short action/event-focused caption generation. The fine-tuned model was then used to generate action captions for images and simple action timelines for videos.
```

---

## Authors

```text
Mohammed, Nawfal, Ruiqi
```
