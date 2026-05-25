# BLIP COCO Action-Caption Fine-Tuning

This repository/notebook contains an experiment for adapting a pre-trained BLIP image-captioning model to produce short action/event-focused captions.

Instead of generating a full descriptive caption such as:

```text
a man in a red shirt walking down the street
```

the fine-tuned model is trained to generate a shorter action caption such as:

```text
a man walking
```

The notebook also includes validation evaluation, comparison with the original BLIP model, single-image prediction, single-video prediction, and batch video timeline prediction saved to CSV.

---

## Notebook

Main notebook:

```text
blip-coco-action-caption-finetuning.ipynb
```

---

## Project Goal

The goal is to convert general image captioning into action/event-focused captioning.

The base task remains:

```text
image -> text
```

but the output style is changed from a long image description to a compact action caption:

```text
image -> action_caption
```

This is useful for video event understanding, where sampled frames can be converted into a simple action timeline.

---

## Base Model

The base model used in this notebook is:

```text
Salesforce/blip-image-captioning-base
```

It is loaded through the Hugging Face Transformers library:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

The model is already pre-trained for image captioning, so the notebook fine-tunes it gently instead of training a captioning model from scratch.

---

## Required Kaggle Inputs

The notebook expects these inputs to be added in Kaggle:

### 1. COCO Train2014 images

The image files should follow the COCO naming format:

```text
COCO_train2014_000000000127.jpg
COCO_train2014_000000000562.jpg
...
```

The notebook automatically searches inside `/kaggle/input` for files starting with:

```text
COCO_train2014_
```

### 2. Action-caption CSV

The notebook uses a converted CSV file:

```text
coco_50k_relaxed_action_captions_training_ready.csv
```

Expected columns:

```text
image_id
file_name
original_caption
action_caption
```

Optional columns may also exist, such as:

```text
conversion_type
keep_for_training
```

### 3. Optional demo videos

For video prediction and timeline generation, the notebook can use videos from:

```text
/kaggle/input/datasets/uwellcome/demo-videos
```

Supported video extensions:

```text
.mp4, .avi, .mov, .mkv, .webm
```

### 4. Optional saved fine-tuned model

For later inference without retraining, the notebook can load a saved model from a Kaggle input dataset, for example:

```text
/kaggle/input/datasets/uwellcome/final-weight
```

The notebook searches for a folder containing model files such as:

```text
config.json
generation_config.json
model.safetensors
processor_config.json
tokenizer.json
tokenizer_config.json
```

---

## Main Pipeline

The notebook follows this pipeline:

```text
COCO images + action-caption CSV
        ↓
Match CSV file_name with actual image paths
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
Predict action captions for images and videos
        ↓
Save batch video timelines to CSV
```

---

## Training Configuration

The main training configuration in the notebook is:

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

A small learning rate is used because the task is close to the original BLIP task: the model still maps images to text, but the output is changed to a shorter action-focused style.

---

## Dataset Preparation

The notebook loads the action-caption CSV and matches each row to its image path using the `file_name` column.

Rows with empty `action_caption` or missing image files are removed.

The final processed dataset is saved to:

```text
/kaggle/working/final_coco_action_caption_training_data.csv
```

The train/validation split is done by `image_id`, not by individual caption rows. This helps reduce leakage between train and validation when multiple captions belong to the same image.

---

## Model Fine-Tuning

The notebook defines a PyTorch `Dataset` and `DataLoader` for COCO action captions.

Each training sample contains:

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

After training, the notebook saves:

```text
model
processor
final training CSV
zipped model folder
```

Output paths:

```text
/kaggle/working/blip_coco_action_caption_finetuned
/kaggle/working/blip_coco_action_caption_finetuned.zip
```

The saved folder can later be uploaded as a Kaggle dataset and used for inference without retraining.

## Using the Fine-Tuned Model Directly

To use the fine-tuned model directly without retraining, the final saved weights must be added to the Kaggle notebook as an input dataset.

In our setup, this corresponds to the saved fine-tuned model folder/dataset, for example:

```text
final-weight
```

or a Kaggle input path such as:

```text
/kaggle/input/datasets/uwellcome/final-weight
```

This folder should contain the saved model and processor files, such as:

```text
config.json
generation_config.json
model.safetensors
processor_config.json
tokenizer.json
tokenizer_config.json
```

Then the fine-tuned model can be loaded directly using:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

SAVED_MODEL_PATH = "/kaggle/input/datasets/uwellcome/final-weight"

processor = BlipProcessor.from_pretrained(SAVED_MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(SAVED_MODEL_PATH)
```

This step is important because, without adding the final weights as a Kaggle input, the notebook can only load the original BLIP model or retrain the model again. Adding the final weights allows users to run image/video prediction immediately using the already fine-tuned model.


---

## Image Prediction

The notebook includes a function:

```python
generate_action_caption_from_image(image, model, processor)
```

Example usage:

```python
image = Image.open(IMAGE_PATH).convert("RGB")

pred = generate_action_caption_from_image(
    image=image,
    model=model,
    processor=processor
)

print("Predicted action caption:", pred)
```

This generates one short action caption for a custom image.

---

## Video Prediction

The notebook includes a function:

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
5. Returning a timeline such as:

```text
[0.0s] a man walking
[2.0s] a person standing
[4.0s] a person riding bicycle
```

A duplicate-removal function is also used to remove repeated consecutive captions.

---

## Batch Video Prediction to CSV

The notebook includes a batch video prediction cell that processes all videos in a folder and saves results to:

```text
/kaggle/working/grouped_video_action_timeline.csv
```

The CSV is grouped by video. The video title appears once, followed by its timeline rows.

Example format:

```text
video_title,time_label,time_seconds,action_caption
Bicycle Parking,,,
,0.0s,0.0,man riding bicycle
,2.0s,2.0,person standing
,4.0s,4.0,person parking bicycle
```

This format is useful for presentation and demo output.

---

## Evaluation Metrics

The notebook evaluates generated captions using:

```text
BLEU-1
BLEU-2
METEOR
ROUGE-L
```

It evaluates the fine-tuned model on a random subset of the validation set, for example:

```python
MAX_EVAL_SAMPLES = 200
```

The notebook also evaluates the original BLIP model against the same action-caption references, allowing a comparison between:

```text
Original BLIP
Fine-tuned BLIP
```

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
   - Optional demo videos.
   - Optional saved model weights.
3. Enable GPU:
   - `Settings -> Accelerator -> GPU`
4. Run the notebook from top to bottom.
5. After training, download or save:
   - the fine-tuned model folder,
   - the zipped model,
   - the processed CSV,
   - the video timeline CSV.

---

## Important Notes

- The notebook does not train BLIP from scratch.
- It fine-tunes a pre-trained BLIP model.
- The output is still captioning, but focused on actions/events.
- The video pipeline is frame-based; it does not perform full temporal reasoning across the whole video.
- Consecutive duplicate captions are removed to make the timeline cleaner.
- The quality of results depends strongly on the quality of the converted `action_caption` labels.

---

## Limitations

- The model predicts from sampled frames independently.
- It may miss actions that require temporal motion understanding.
- It may produce repeated captions for visually similar frames.
- Some actions may be ambiguous from a single image.
- The action-caption dataset is derived from COCO captions, so it is not a true video action dataset.

---

## Future Work

Possible improvements:

- Use a real action/video dataset such as UCF101 or Kinetics.
- Use a video-based model instead of frame-only captioning.
- Improve action-caption conversion with stronger semantic rewriting.
- Add temporal smoothing across frames.
- Evaluate on more validation samples.
- Compare with other vision-language or video-captioning models.

---

## Credits

Base model:

```text
Salesforce/blip-image-captioning-base
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
```

Dataset source:

```text
COCO 2014 images and captions
```

Action-caption labels were created by converting original COCO captions into shorter action-focused captions.

---

## Suggested Citation / Acknowledgement

If describing the model in a report or presentation:

```text
We used the pre-trained Salesforce/blip-image-captioning-base model from Hugging Face and fine-tuned it on a converted COCO action-caption dataset. The goal was to adapt the model from general image captioning to short action/event-focused caption generation.
```
