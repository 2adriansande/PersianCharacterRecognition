# Persian Character Recognition System

A deep-learning system that recognizes **one isolated Persian letter** in an image and tells you which of the 32 letters of the Persian alphabet it is. It has two separate models, one for **printed / digital** letters and one for **handwritten** letters. Both are MobileNetV3 Small networks built with PyTorch, and a Streamlit web app lets you pick the mode, upload an image and see the result.

## Table of Contents

1. [Current Project Status](#current-project-status)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Datasets](#datasets)
6. [Model Architecture](#model-architecture)
7. [Installation Requirements](#installation-requirements)
8. [Installation Steps](#installation-steps)
9. [Model Checkpoint Setup](#model-checkpoint-setup)
10. [How to Run the Application](#how-to-run-the-application)
11. [How to Make Predictions](#how-to-make-predictions)
12. [Example Usage](#example-usage)
13. [Training and Evaluation Commands](#training-and-evaluation-commands)
14. [Evaluation Results](#evaluation-results)
15. [Limitations](#limitations)
16. [Troubleshooting](#troubleshooting)
17. [Preparing the Repository for GitHub](#preparing-the-repository-for-github)
18. [Future Improvements](#future-improvements)
19. [Contributors](#contributors)
20. [License](#license)

---

## Current Project Status

| Area | Status |
|---|---|
| Printed / digital Persian letters | **Supported** (printed model) |
| Handwritten Persian letters | **Supported** (separate handwritten model), as isolated single letters. Evaluated only on held-out forms of the training dataset, not on other people's handwriting or photos. |
| Words, sentences, full lines | Not supported (one isolated letter per image) |
| Automatic detection of printed vs handwritten | Not implemented. You choose the mode in the app. |
| Model training and evaluation | Done for both models (see [Evaluation Results](#evaluation-results)) |
| Streamlit web app with a mode selector | Implemented in `app.py` |

Each model always answers with one of its 32 known letters, even for images it was not made for (for example handwriting sent to the printed model). See [Limitations](#limitations).

## Features

Everything listed here is implemented in the current code.

- A **recognition mode** selector in the app sidebar: Printed Persian Characters or Handwritten Persian Characters. Each mode loads its own trained model.
- Upload an image (PNG, JPG, JPEG or BMP) and preview it
- Prediction shown as the Persian letter (right-to-left), its Latin name (for example `alef`) and its Unicode code point (for example `U+0627`)
- Confidence score as a percentage, plus the top 3 predictions
- A view of the cropped, normalized image that is actually given to the model
- Automatic preprocessing (shared by both modes): converts to grayscale, detects dark-on-light or light-on-dark, crops to the character, pads to a square without stretching, resizes to 224x224
- Clear error messages for unreadable files, too-small or too-large images, and blank images, and a clear message if the handwritten model file is missing (printed mode keeps working)
- Warnings for low confidence (below 70%, a simple untuned threshold) and for characters touching the image border
- A "Reset" button to clear the result and upload another image
- Command-line prediction for both models (`src/predict.py`, `src/predict_hw.py`)

## Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python (see [Installation Requirements](#installation-requirements)) |
| Deep learning | PyTorch, torchvision (MobileNetV3 Small) |
| Image handling | Pillow, OpenCV (`opencv-python`), NumPy |
| Web app | Streamlit |
| Data and reports | pandas, scikit-learn, matplotlib, seaborn, tqdm |

## Project Structure

```
app.py                         Streamlit web application (run this)
.streamlit/config.toml         Optional Streamlit theme setting
src/
  labels.py                    The 32 letters: index, Persian character, Latin name (shared by both models)
  preprocess.py                Validation, binarization, cropping, resizing (shared by both models)
  model.py                     MobileNetV3 Small with a 32-class output layer (shared)
  augment.py                   Training-time image augmentation (shared)
  predict.py                   Loads a model and predicts one image (printed model by default)
  --- printed model ---
  prepare_data.py              Cleans the printed dataset, removes duplicates, makes the splits
  train.py                     Printed model training
  evaluate.py                  Printed model evaluation on the test split
  --- handwritten model ---
  labels_hw.py                 Maps the handwritten folders class_01 ... class_32 to the labels
  prepare_data_hw.py           Cleans the handwritten dataset and splits it by form
  train_hw.py                  Handwritten model training
  evaluate_hw.py               Handwritten model evaluation on the test forms
  predict_hw.py                Loads the handwritten model and predicts one image
  inspect_handwritten.py       Read-only dataset check (counts, sizes, contact sheet)
  inspect_forms_hw.py          Read-only check of the form numbers in the file names
  (other helper scripts)       inspect_dataset*.py, review_drops.py, flag_artifacts.py, ...
models/
  mobilenetv3_small_best.pt                  Printed model (needed for printed mode)
  handwritten_mobilenetv3_small_best.pt      Handwritten model (needed for handwritten mode)
  model_config.json                          Readable summary of the printed model (not loaded by the app)
reports/                       Evaluation outputs, confusion matrices, training curves
                               (hw_* files belong to the handwritten model)
dataset/
  manifest.csv                 Printed dataset: one row per unique image, with its split
  manifest_hw.csv              Handwritten dataset: one row per unique image, with its split and form
  processed/                   Cached training arrays and logs (large, not for GitHub)
tests/                         Unit tests for the printed data preparation
```

## Datasets

Neither image dataset is stored in this repository. Running the app does not need either dataset. Both models use the same 32 classes in the same order, defined in `src/labels.py`:

ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی

Persian kaf (U+06A9) and yeh (U+06CC) are used, not the Arabic code points U+0643 and U+064A.

### Printed dataset

- 32 classes, printed / digital style only
- Built from two merged collections (named `ds101` and `alpha` in `manifest.csv`)
- `src/prepare_data.py` removes duplicates inside a class, images whose look-alike appears under two different classes, blank images, and mislabeled wide "word" images
- After cleaning: **10,638 unique images**, split per class into **7,448 train / 1,595 validation / 1,595 test**. 49 images are flagged as glitches in the manifest; 6 of them are in the test split and are left out of evaluation.
- The split is by image, not by font (the data has no font or writer information).

### Handwritten dataset

- 32 classes, one folder per letter: `class_01` ... `class_32`. `class_NN` is the letter with index `NN-1` in `labels.py` (`class_01` = alef ... `class_32` = ye). This was checked by eye on a contact sheet with 8 samples per folder.
- **31,999 images**, PNG, grayscale, 64x64 pixels, tightly cropped, dark strokes on a light background. About 1,000 per class (999 for `class_29`), so the classes are balanced.
- File names look like `class_02_f03_s0150.png`: class number, **form number** (`f01` ... `f10`, about 3,200 images each, 100 per letter per form) and a running sample number. The form numbers show different handwriting sessions (pen thickness, size and slant change between forms). The dataset documentation was not available, so we assume one form is one writer or writing session.
- `src/prepare_data_hw.py` removes exact duplicates (2 found; 31,997 images kept). It drops nothing else on this dataset.
- **Split by form**, so the test set contains handwriting sessions the model never saw in training: train = forms 1, 3, 4, 5, 7, 8, 10 (22,398 images), validation = form 6 (3,200), test = forms 2 and 9 (6,399). The form assignment uses a fixed random seed and can be changed with `--val-forms` and `--test-forms`.

> **TODO:** add the dataset names, links, licenses and citations for the printed sources (`ds101`, `alpha`) and for the handwritten dataset (the folder was called `dataset_cropped`).

Both manifests store the **absolute image paths of the computer where they were created**. On another computer, re-run the preparation script on your own copy of the images (see [Training and Evaluation Commands](#training-and-evaluation-commands)).

## Model Architecture

Both models use the same architecture and the same preprocessing:

- **Network:** MobileNetV3 Small from torchvision, starting from ImageNet weights, with the last layer replaced by a linear layer with 32 outputs.
- **Input:** 224x224 image, three channels, normalized with ImageNet mean and standard deviation.
- **Training recipe:** stage 1 trains only the new output layer (3 epochs); stage 2 fine-tunes from feature block 8 onward (up to 25 epochs, early stopping after 7 epochs without validation improvement). AdamW, OneCycle learning rate, cross-entropy with label smoothing 0.05 and class weights, GPU augmentation (small rotation, zoom, shear, shift), seed 42. The best epoch is chosen on the validation split.
- **Handwritten model:** trained from ImageNet weights, **not** from the printed model. Its best epoch was epoch 19 (validation accuracy 98.44%); training stopped early at epoch 23.
- **Prediction:** the model gives 32 scores; softmax turns them into probabilities. The top probability is the confidence shown in the app.
- The class mapping is stored inside each checkpoint and is checked against `src/labels.py` when a model loads. If they differ, loading stops with an error.

## Installation Requirements

- **Python:** the project was developed with Python 3.12 (this is inferred from the cached files in the project). Python 3.12 is the safest choice. *TODO: confirm the exact version the author used.*
- **Operating system:** commands below are for Windows with VS Code. Running the app does not need a GPU. Training was done on an NVIDIA GPU (RTX 3050 Laptop); training on a CPU is possible but much slower.
- **Memory for handwritten training:** about 1.3 GB of RAM for the cached images.
- **Packages:** `torch`, `torchvision`, `pillow`, `numpy`, `opencv-python`, `pandas`, `streamlit`. For training, evaluation and the data scripts also: `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`.
- The project has no `requirements.txt` yet. *TODO: the author should run `pip freeze > requirements.txt` in the working environment and commit the file, so teammates get the same versions.*

## Installation Steps

Open a terminal in VS Code (**Terminal > New Terminal**) and run:

```
git clone <your-repository-url>
cd <repository-folder>
```

`git clone` downloads the project. Replace the placeholders with your repository's address and folder name.

Create and activate a virtual environment (a private set of Python packages for this project):

```
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of the terminal line. If PowerShell blocks activation, see [Troubleshooting](#troubleshooting).

Install the packages:

```
python -m pip install --upgrade pip
pip install torch torchvision pillow numpy opencv-python pandas streamlit
```

(If the project has a `requirements.txt`, use `pip install -r requirements.txt` instead.) Optional, only for training, evaluation and data scripts:

```
pip install scikit-learn matplotlib seaborn tqdm
```

In VS Code, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and pick the one inside `.venv`.

## Model Checkpoint Setup

The app needs one trained model file for each mode:

| Mode | File that must exist |
|---|---|
| Printed | `models\mobilenetv3_small_best.pt` |
| Handwritten | `models\handwritten_mobilenetv3_small_best.pt` |

- The class names and the Persian letter for each class are stored **inside** each file. When a model loads, the code checks them against `src/labels.py`, so both must be present and must match.
- `models\model_config.json` is only a readable summary of the printed model. The app does not load it.
- If a model file is not in your copy of the repository, **ask the teammate who trained it to share it** (for example by a shared drive) and place it at the path above. There is no public download link for this project.
- If only the printed file is present, the app still works in printed mode and shows a clear message in handwritten mode.
- You do not need to retrain. Training needs the image datasets (see [Training and Evaluation Commands](#training-and-evaluation-commands)).

## How to Run the Application

From the project root (the folder that contains `app.py`), with the virtual environment active:

```
streamlit run app.py
```

This starts a local web server and normally opens your browser at http://localhost:8501. Press **Ctrl+C** in the terminal to stop it.

## How to Make Predictions

1. In the **sidebar**, choose the **recognition mode** that matches your image: *Printed Persian Characters* or *Handwritten Persian Characters*. (If the sidebar is hidden, open it with the small arrow at the top left.)
2. Click **Browse files** (or drag an image in) and choose a PNG, JPG, JPEG or BMP file with **one** letter.
3. Check the preview.
4. Click **Recognize Character**.
5. Read the result:
   - **Large letter:** the predicted Persian character.
   - **Label chip:** its Latin name, for example `be` or `shin`.
   - **Code chip:** its Unicode code point.
   - **Confidence bar:** the model's probability for its top answer, as a percentage. The bar turns amber below 70%.
6. Open **Top 3 predictions and model input** to see the next best guesses and the cropped image the model used.
7. Click **Reset** to clear everything and upload another image. Switching mode also clears the previous result.

**How to read the confidence:** it is the model's own probability, not a measured accuracy. Wrong answers can still show high confidence (in the handwritten evaluation, many mistakes were between look-alike letters), and unsuitable images can also get a confident answer. Use it as a hint, not as proof.

**Best input:** one clear letter, dark on a plain light background, centered, not touching the edge, at least 16x16 pixels. For handwriting, write one letter with a dark pen on plain paper and crop the photo tightly around it.

## Example Usage

Using the web app:

1. Run `streamlit run app.py`.
2. In the sidebar choose *Printed Persian Characters* and upload a screenshot of a single printed letter, for example ب. The app shows the letter ب, the label `be`, the code `U+0628` and a confidence percentage.
3. Switch to *Handwritten Persian Characters* and upload a cropped image of one handwritten letter.

Using the command line (from the project root, with the virtual environment active):

```
python src\predict.py path\to\printed_letter.png
python src\predict_hw.py path\to\handwritten_letter.png
```

Each prints the prediction for your own image (several paths can be given). To check that the printed model loads and works on test images from the printed dataset (this needs the printed dataset images on your computer):

```
python src\predict.py --selftest
```

From your own Python code:

```python
import sys
sys.path.insert(0, "src")
from predict import load_predictor
from predict_hw import load_handwritten_predictor

printed = load_predictor()
handwritten = load_handwritten_predictor()
result = handwritten.predict("path/to/image.png")   # path, PIL image or NumPy array
print(result["char"], result["label"], result["confidence"])
```

## Training and Evaluation Commands

Only needed if you want to retrain or re-evaluate. Run them from the project root with the virtual environment active. Evaluate **once**, after training; do not tune the model on the test split.

**Handwritten model** (the folder you pass must directly contain `class_01` ... `class_32`):

```
python src\prepare_data_hw.py --root "<path to the folder containing class_01 ... class_32>"
python src\train_hw.py --smoke
python src\train_hw.py
python src\evaluate_hw.py
```

- `prepare_data_hw.py` writes `dataset\manifest_hw.csv` and two preview images in `reports\` (`hw_preview_crops.png`, `hw_preview_extremes.png`) so you can check the processed crops by eye.
- `train_hw.py --smoke` is a 1-2 minute pipeline check; it saves no model. The full run saves `models\handwritten_mobilenetv3_small_best.pt` and refuses to overwrite an existing one unless you add `--overwrite`. It cannot write to the printed model's path.
- `evaluate_hw.py` writes `reports\hw_eval_*` files.

**Printed model** (see the first lines of each script for details; the printed dataset folders are `01_alef` ... `32_ye`):

```
python src\prepare_data.py --root "<path to final_merged_dataset>"
python src\mark_glitch.py
python src\train.py
python src\evaluate.py
```

## Evaluation Results

All numbers below come from the evaluation files in `reports/` and the training logs. Both evaluations use a held-out test split that was not used for training or for choosing the best epoch.

### Handwritten model

Test split = forms 2 and 9 of the handwritten dataset (`reports/hw_eval_*`):

| Metric | Value |
|---|---|
| Test images | 6,399 |
| Accuracy | 98.59% |
| Macro F1-score | 0.9859 |
| Accuracy on form 2 (3,200 images) | 98.25% |
| Accuracy on form 9 (3,199 images) | 98.94% |
| Best validation accuracy (form 6, epoch 19) | 98.44% |

- Weakest letters by F1-score: `jim` (0.949), `kaf` (0.962), `he_jimi` (0.966), `be` (0.967), `che` (0.968).
- Most frequent mistakes (true -> predicted): `jim` -> `che` (8), `zhe` -> `ze` (8), `jim` -> `he_jimi` (7), `be` -> `pe` (7), `se` -> `te` (5), `khe` -> `he_jimi` (5).
- Looking through a random sample of 40 mistakes (`reports/hw_eval_misclassified.png`), most were between letters that differ only in the number or position of dots. Some of these samples show merged dots or look like another letter, so a part of the remaining error is probably unclear or mislabeled data. This was a visual impression from one sample, not a systematic check.

**What these numbers mean:** the test forms come from the same collection as the training forms, and the dataset does not say whether two forms were written by the same person. The numbers show how well the model handles unseen forms of this collection. They do **not** show how well it handles other people's handwriting, different pens and paper, or phone photos. *TODO: add results from your own handwriting tests, if you run them.*

### Printed model

Test split of the printed dataset (`reports/eval_summary.json`):

| Metric | Value |
|---|---|
| Test images | 1,589 |
| Accuracy | 98.80% |
| Macro F1-score | 0.988 |
| Best validation accuracy (epoch 28) | 98.74% |
| Mean confidence, correct predictions | 93.4% |
| Mean confidence, wrong predictions | 56.7% |

Median prediction time was about 14.8 ms per image on a GPU, end to end (measured on 300 images; CPU time was not measured). The printed split is by image, not by font, so accuracy on fonts the model has never seen is probably lower than shown.

Per-class scores, confusion matrices and misclassified examples for both models are in `reports/`.

## Limitations

- **Pick the right mode.** The printed model was trained on printed letters only and the handwritten model on handwritten letters only. Neither is expected to work well on the other kind of image, and there is no automatic detection.
- **One isolated letter per image.** Neither model reads words or sentences.
- **Supported files:** PNG, JPG, JPEG, BMP. Image sides must be between 16 and 8000 pixels. Blank images are rejected with an error.
- **Always an answer.** Each model always picks one of its 32 classes, including for blank-looking, noisy, rotated or multi-letter images and for symbols that are not Persian letters.
- **Confidence is not proof.** It is the softmax probability of the top class. The 70% warning threshold is a simple cut-off that was not tuned or validated.
- **Handwriting:**
  - The handwritten model was trained on tightly cropped 64x64 images from one dataset (10 forms). Very different handwriting styles, cluttered backgrounds and uncropped photos were not tested.
  - Look-alike letters that differ only by dots (for example ب پ ت ث, ج چ ح خ, ر ز ژ, د ذ) are the most common mistakes, and merged or missing dots make them harder.
  - Thresholding during preprocessing makes strokes a bit heavier and can join nearby dots.
- **Printed model:** the test split is by image, not by font, so the test score is likely optimistic for unseen fonts.
- **Rotation and blur** were not evaluated systematically for either model.
- **Letter shapes:** both models recognize the 32 isolated letters, not the connected forms letters take inside words.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `'streamlit' is not recognized` | The virtual environment is not active or Streamlit is not installed. Run `.venv\Scripts\activate`, then `pip install streamlit`. Or use `python -m streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'torch'` (or `cv2`, `PIL`, `pandas`) | A package is missing. Activate `.venv` and install it: `pip install torch torchvision`, `pip install opencv-python`, `pip install pillow`, `pip install pandas`. |
| PowerShell says running scripts is disabled when activating `.venv` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt (`cmd`) instead. |
| Printed mode shows "Model checkpoint not found" | `models\mobilenetv3_small_best.pt` is missing. See [Model Checkpoint Setup](#model-checkpoint-setup). |
| Handwritten mode says the handwritten model has not been trained | `models\handwritten_mobilenetv3_small_best.pt` is missing. Get the file from the teammate who trained it, or train it (see [Training and Evaluation Commands](#training-and-evaluation-commands)). Printed mode still works. |
| "Could not load the model: Class mapping in checkpoint differs from src/labels.py" | `src/labels.py` was changed, or the model file is from a different version. Use the model file and `labels.py` from the same version. |
| `ModuleNotFoundError: No module named 'predict_hw'` | The handwritten files are missing from `src\`. Make sure `labels_hw.py`, `prepare_data_hw.py`, `train_hw.py`, `evaluate_hw.py` and `predict_hw.py` are all in the `src` folder. |
| `streamlit run app.py` says it cannot find `app.py` | You are in the wrong folder. Use `cd` to go to the project root, the folder that contains `app.py`. |
| "Invalid image: Image appears blank" / "No character found" | The image has no visible letter, or the letter is too faint. Use a darker letter on a plain background. |
| "Image size ... outside [16, 8000] px" | The image is too small or too large. Resize it. |
| "This file could not be read as an image" | The file is damaged or is not really an image. Use a real PNG, JPG, JPEG or BMP. |
| Port 8501 already in use | Another Streamlit app is running. Stop it, or use `streamlit run app.py --server.port 8502`. |
| `prepare_data_hw.py` says the root "only contains one folder" | The class folders are one level deeper. Use the path it prints after `--root`. |
| `evaluate_hw.py` or `evaluate.py` says test images were not found | The manifests hold absolute paths from the computer where they were made. Re-run `prepare_data_hw.py` (or `prepare_data.py`) with the path to your own copy of the images. |
| `train_hw.py` says the checkpoint already exists | This protects a finished model. Add `--overwrite` to replace it, or `--out <other path>`. |
| Very slow first prediction | The model is loading for the first time. Later predictions are faster. |

## Preparing the Repository for GitHub

- **Commit:** the source code (`app.py`, `src/`, `tests/`), `README.md`, the two small manifests (`dataset/manifest.csv`, `dataset/manifest_hw.csv`), the model files in `models/`, and the small files in `reports/`. Check each model file's size first; GitHub rejects single files over 100 MB, and these models are expected to be small.
- **Do not commit the image datasets or `dataset/processed/`.** The printed cache alone is about 468 MB, and the handwritten cache adds more. Check each dataset's license and your project's privacy rules before sharing any images.
- **Check the manifests:** they contain absolute paths from the author's computer, including a Windows user folder name. Decide whether to share them as they are or replace the paths with relative ones.
- Add a `.gitignore` before the first commit. A suggested starting point:

  ```
  .venv/
  __pycache__/
  *.pyc
  dataset/processed/
  dataset/raw/
  ```

- Never commit passwords, API keys or personal files.

## Future Improvements

These are ideas, **not** completed features.

- Test the handwritten model on new handwriting (other people, pens, paper, phone photos) and report the results honestly
- Review the confidently wrong test samples for mislabeled data and clean the handwritten dataset
- Handle dots better for handwriting (for example a gentler threshold, or a larger input for the dotted look-alike letters)
- Stroke-thickness augmentation, and a split by known writer if writer information becomes available
- Automatic detection of printed vs handwritten input
- Better preprocessing for noisy backgrounds, photos and rotated letters
- Tuning the low-confidence warning, or adding a real "not a valid character" check
- Recognizing several characters or whole words
- A `requirements.txt` with pinned versions


## License

No license has been specified for this project. Until the team chooses one, all rights are reserved by the authors. Also check the licenses of the dataset sources before sharing images or trained models.