# Persian Character Recognition System

A deep-learning system that recognizes **one isolated Persian letter** in an image and tells you which of the 32 letters of the Persian alphabet it is. It has three MobileNetV3 Small models built with PyTorch: one for **printed / digital** letters, one for **handwritten** letters, and one **combined** model that handles both. A Streamlit web app lets you pick the mode, upload an image and see the result.

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
| Printed and handwritten letters with one model | **Supported** (combined model), evaluated on the same held-out test sets as the two separate models |
| Automatic detection of printed vs handwritten | Not implemented. You choose the input type in the app (or use the combined model). |
| Model training and evaluation | Done for all three models (see [Evaluation Results](#evaluation-results)) |
| Streamlit web app with an input-type selector | Implemented in `app.py` (redesigned interface, see [Features](#features)) |
| Rotated letters | Measured (see [Rotation robustness](#rotation-robustness)). The three models trained so far are reliable only up to about 10-15 degrees of tilt. An optional rotation-aware training setting for tilts up to 30 degrees is implemented, but a model trained with it has **not been evaluated yet**. Sideways or upside-down letters are not supported. |

Each model always answers with one of its 32 known letters, even for images it was not made for (for example handwriting sent to the printed model). See [Limitations](#limitations).

## Features

Everything listed here is implemented in the current code.

- A redesigned single-page interface: header, a hero section, an upload workspace with a tips panel, a results section, a recognition history and a short explanation of the processing pipeline
- An **input type** selector under the upload box: Handwritten, Printed or Combined (Printed + Handwritten). Each type loads its own trained model.
- Upload an image (PNG, JPG, JPEG or BMP) and preview it
- Prediction shown as the Persian letter (right-to-left), its Latin name (for example `alef`) and its Unicode code point (for example `U+0627`)
- Confidence score as a percentage, plus the top 3 predictions
- A view of the cropped, normalized image that is actually given to the model
- Automatic preprocessing (shared by both modes): converts to grayscale, detects dark-on-light or light-on-dark, crops to the character, pads to a square without stretching, resizes to 224x224
- Clear error messages for unreadable files, too-small or too-large images, and blank images, and a clear message if the handwritten model file is missing (the other input types keep working)
- Warnings for low confidence (below 70%, a simple untuned threshold) and for characters touching the image border
- After you click **Recognize character**, the page scrolls smoothly to the result, which fades in
- A **recognition history** of this browser session (up to 8 results). It is kept in memory only: nothing is written to disk and it clears when the page is reloaded.
- A model status line that shows whether the checkpoint file of the selected input type exists
- Upload size limited to 8 MB (set in `.streamlit/config.toml`)
- A "Reset" button to clear the result and upload another image
- Command-line prediction for all three models (`src/predict.py`, `src/predict_hw.py`, `src/predict_combined.py`)
- Configurable training-time data augmentation for the combined model (`src/augment_cfg.py`), applied on the fly with nothing saved to disk, and a preview tool to check its effect (`src/preview_augmentation.py`)
- An optional rotation-aware augmentation that imitates a tilted photo (rotate, then re-crop the way the app does), off by default
- A measurement-only rotation test (`src/evaluate_rotation.py`) that reports accuracy at any list of rotation angles for any of the three models

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
.streamlit/config.toml         Streamlit theme colors and the 8 MB upload limit
.gitignore                     Files Git should not track (virtual environment, datasets, caches, ...)
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
  --- combined model (printed + handwritten) ---
  augment_cfg.py               Configurable training augmentation (rotation, zoom, shift, optional large rotation, ...)
  preview_augmentation.py      Saves before/after images and checks that dots and strokes survive
  train_combined.py            Trains one model on both training sets
  evaluate_combined.py         Evaluates it on both test sets, separately
  predict_combined.py          Loads the combined model and predicts one image
  --- rotation test (works with all three models) ---
  evaluate_rotation.py         Measures accuracy at different rotation angles (trains nothing)
  (other helper scripts)       inspect_dataset*.py, review_drops.py, flag_artifacts.py, ...
models/
  mobilenetv3_small_best.pt                  Printed model (needed for printed mode)
  handwritten_mobilenetv3_small_best.pt      Handwritten model (needed for handwritten mode)
  combined_mobilenetv3_small_best.pt         Combined model (needed for combined mode)
  model_config.json                          Readable summary of the printed model (not loaded by the app)
reports/                       Evaluation outputs, confusion matrices, training curves
                               (hw_* files belong to the handwritten model, comb_* files to the combined
                               model, aug_preview_* are the augmentation previews, rotation_* the rotation tests)
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

### Combined model

- One MobileNetV3 Small trained on printed **and** handwritten images together: 29,807 training images (7,409 printed + 22,398 handwritten, so about 25% printed) and 4,791 validation images (1,591 printed + 3,200 handwritten).
- It uses the **existing splits** of both datasets (printed split by image, handwritten split by form) instead of a new random split. The two test sets are therefore exactly the ones the separate models were tested on, and no test image was used for training.
- Same architecture and recipe as above, trained from ImageNet weights with the augmentation described below. It ran the full 28 epochs; the best epoch was 24. Saved to `models\combined_mobilenetv3_small_best.pt`. The training script also checks, by file hash, that the printed and handwritten checkpoints were not changed.

### Data augmentation

The combined model is trained with the configurable augmentation in `src/augment_cfg.py`. It is applied **only to training images, on the fly on the GPU**; no augmented files are saved, and validation and test images are never augmented. There are no horizontal flips, because a flip would change which letter an image shows.

| Transform | Default setting |
|---|---|
| Rotation | random, -10 to +10 degrees |
| Zoom | random, 0.90 to 1.10 |
| Shift | up to 5% of the image width and height |
| Shear (slant) | small, 0.08 |
| Extra conservative zoom-in ("crop") | 15% of samples, up to 6% extra zoom |
| Thicker strokes / thinner strokes | 20% / 15% of samples |
| Mild blur | 30% of samples |
| Light pixel noise | 30% of samples |
| Large rotation with re-crop (**off by default**) | `--aug-p-rotate` = share of samples (default 0), angle uniform in plus/minus `--aug-rotate-full-deg` (default 45) |
| Sideways / upside-down (**off by default**) | `--aug-p-rot90` = share of the rotated samples turned by 90, 180 or 270 degrees (default 0) |

- A safety check puts a sample back to its original geometry if the transform would push ink into the image border (a sign that a stroke or dot would be cut off).
- **Why a separate large-rotation option exists:** the ordinary rotation setting only goes up to 10 degrees, and the border safety check above would cancel most larger rotations. Rotating the crop inside its frame is also not what happens in the app, which rotates the picture first and then crops to the character. The large-rotation option therefore rotates the crop on an enlarged canvas, finds the character's bounding box again and re-pads it with the same 12% margin as `preprocess.py`. On synthetic test letters this matched the app's real pipeline closely (average overlap about 0.82 to 0.91, against 0.28 to 0.43 for plain in-frame rotation). That was a check on synthetic images only; the effect on real accuracy has to be measured with `evaluate_rotation.py`.
- Every value can be changed from the command line, for example `--aug-rotate-deg 8` or `--aug-p-thin 0` (see `python src\train_combined.py --help`).
- **Brightness, contrast and background variation are not used.** Every image is converted to a black-and-white crop (Otsu threshold) before it reaches the model, so such changes would have almost no effect on the model input.
- `python src\preview_augmentation.py` saves before/after images (`reports\aug_preview_*.png`). On 300 printed and 300 handwritten training characters with 5 random augmentations each, the geometric step was skipped for 5.7% (printed) and 7.0% (handwritten) of samples because ink reached the border; 2.7% (printed) and 0.3% (handwritten) of augmented samples ended up with fewer separate parts than the original, which could mean a lost or merged dot.
- The printed and handwritten models were trained with the older fixed augmentation in `src/augment.py` (similar rotation, zoom 0.85 to 1.15, shift about 4%, stroke thickness, blur and noise). The new settings were **not** compared against the old ones on their own, so the results cannot say whether the augmentation itself helped.

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
| Combined | `models\combined_mobilenetv3_small_best.pt` |

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

1. Click **Browse files** (or drag an image in) and choose a PNG, JPG, JPEG or BMP file (up to 8 MB) with **one** letter. A small preview appears.
2. Under **Input type**, choose the type that matches your image: *Handwritten*, *Printed*, or *Combined* if you are not sure which kind it is. The default is Printed.
3. Click **Recognize character**. The page scrolls down to the result.
4. Read the result:
   - **Large letter:** the predicted Persian character.
   - **Label chip:** its Latin name, for example `be` or `shin`.
   - **Code chip:** its Unicode code point.
   - **Confidence bar:** the model's probability for its top answer, as a percentage. The bar turns amber below 70%.
   - **Top 3 predictions:** the next best guesses, and the cropped image the model actually used.
5. Earlier results of this session are listed under **Your recent results**. The list is not saved and clears when you reload the page.
6. Click **Reset** to clear everything and upload another image. Switching the input type also clears the previous result.

**How to read the confidence:** it is the model's own probability, not a measured accuracy. Wrong answers can still show high confidence (in the handwritten evaluation, many mistakes were between look-alike letters), and unsuitable images can also get a confident answer. Use it as a hint, not as proof.

**Best input:** one clear letter, dark on a plain light background, centered, not touching the edge, at least 16x16 pixels, and roughly upright (see [Rotation robustness](#rotation-robustness)). For handwriting, write one letter with a dark pen on plain paper and crop the photo tightly around it.

## Example Usage

Using the web app:

1. Run `streamlit run app.py`.
2. Under *Input type* keep *Printed* and upload a screenshot of a single printed letter, for example ب. The app shows the letter ب, the label `be`, the code `U+0628` and a confidence percentage.
3. Switch the input type to *Handwritten* and upload a cropped image of one handwritten letter.

Using the command line (from the project root, with the virtual environment active):

```
python src\predict.py path\to\printed_letter.png
python src\predict_hw.py path\to\handwritten_letter.png
python src\predict_combined.py path\to\any_letter.png
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

**Combined model** (needs both datasets prepared first, i.e. `dataset\manifest.csv` and `dataset\manifest_hw.csv`; it reuses the cached image arrays from the earlier training runs when they exist):

```
python src\preview_augmentation.py
python src\train_combined.py --smoke
python src\train_combined.py
python src\evaluate_combined.py
```

- `preview_augmentation.py` trains nothing; it only shows what the augmentation does.
- `train_combined.py --smoke` is a short pipeline check that saves no model. The full run saves `models\combined_mobilenetv3_small_best.pt`, refuses to overwrite an existing one unless you add `--overwrite`, and cannot write to the printed or handwritten model files. It needs about 3.5 GB of RAM while the two datasets are joined. `--printed-repeat 2` would repeat the printed images to give them more weight (the default is the natural mix, about 25% printed).
- At the end of training the script prints whether the printed and handwritten model files are unchanged. To check by hand, run `Get-FileHash models\mobilenetv3_small_best.pt -Algorithm SHA256` (and the same for the handwritten file) before and after; the values must be identical.
- `evaluate_combined.py` reports the printed and handwritten test sets separately and prints the separate models' stored results beside them. Run it once, after training.

**Rotation test and rotation-aware training** (optional; train and evaluate only when you want to, and back up `reports\comb_*` first, because training and evaluation reuse those file names):

```
python src\evaluate_rotation.py
python src\evaluate_rotation.py --model printed
python src\evaluate_rotation.py --model handwritten --angles -30 -15 0 15 30 90 180 270
python src\train_combined.py --aug-p-rotate 0.6 --aug-rotate-full-deg 30 --out models\combined_rot30_mobilenetv3_small_best.pt
python src\evaluate_rotation.py --ckpt models\combined_rot30_mobilenetv3_small_best.pt --tag rot30
python src\evaluate_combined.py --ckpt models\combined_rot30_mobilenetv3_small_best.pt
```

- `evaluate_rotation.py` trains nothing. It takes 25 test images per letter from each dataset (800 printed and 800 handwritten), rotates each original image counter-clockwise by every angle on an enlarged canvas filled with its background color, runs the normal preprocessing and the model, and writes `reports\rotation_<name>.csv` and `.png`. Use the same angles and seed before and after any change so the results can be compared.
- The rotation-aware training line writes to a **new** file name, so the existing combined model is not touched. After training, compare the new model's rotation table with the baseline below, and check with `evaluate_combined.py` that the accuracy on upright images did not drop.

**Printed model** (see the first lines of each script for details; the printed dataset folders are `01_alef` ... `32_ye`):

```
python src\prepare_data.py --root "<path to final_merged_dataset>"
python src\mark_glitch.py
python src\train.py
python src\evaluate.py
```

## Evaluation Results

All numbers below come from the evaluation files in `reports/` and the training logs. Both evaluations use a held-out test split that was not used for training or for choosing the best epoch.

### Combined model (printed + handwritten)

Evaluated once on the same two test sets as the separate models (`reports/comb_eval_*`):

| Test set | Images | Combined model: accuracy / macro F1 | Separate model: accuracy / macro F1 |
|---|---|---|---|
| Printed test split | 1,589 | 98.62% / 0.9864 | 98.80% / 0.9881 |
| Handwritten test forms 2 and 9 | 6,399 | 98.61% / 0.9861 | 98.59% / 0.9859 |
| Both together | 7,988 | 98.61% / 0.9861 | not applicable |

- Best validation accuracy (epoch 24): 98.58% overall, 98.68% on the printed validation set, 98.53% on the handwritten validation set.
- **What this means:** one combined model performed about as well as the two specialist models on these test sets. The printed difference is roughly 3 images out of 1,589 (slightly lower) and the handwritten difference is roughly 1 image out of 6,399 (slightly higher). Both are within normal sampling noise, so these results show neither an improvement nor a loss from combining the data or from the new augmentation.
- Most frequent handwritten mistakes (true -> predicted): `be` -> `pe` (8), `jim` -> `he_jimi` (8), `che` -> `jim` (7), `khe` -> `he_jimi` (5), `se` -> `te` (5), `ayn` -> `ghayn` (4), `zhe` -> `ze` (4), `jim` -> `che` (4). These are the same dot-related look-alike pairs as in the handwritten model. Printed mistakes were rare, at most 2 per pair (`se` -> `te`, `he_jimi` -> `mim`).
- The same caveats apply: the handwritten test forms come from the same collection as the training forms, the printed split is by image and not by font, and neither test set contains other people's handwriting, other fonts or phone photos.

### Rotation robustness

Measured with `src\evaluate_rotation.py` before any rotation-aware training (25 test images per letter per dataset, seed 0; angles are counter-clockwise; the pipeline rejected no images). Accuracy in percent:

**Combined model**

| Angle | -45 | -30 | -20 | -10 | 0 | 10 | 20 | 30 | 45 | 90 | 180 | 270 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Handwritten | 49.6 | 81.6 | 95.2 | 98.9 | 98.6 | 98.1 | 95.2 | 85.6 | 49.8 | 12.0 | 23.8 | 13.2 |
| Printed | 44.0 | 74.6 | 90.2 | 97.8 | 98.8 | 96.8 | 88.5 | 72.6 | 42.4 | 6.6 | 11.0 | 6.0 |

**Printed model**

| Angle | -45 | -30 | -20 | -10 | 0 | 10 | 20 | 30 | 45 | 90 | 180 | 270 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Handwritten | 38.5 | 65.1 | 76.4 | 83.2 | 86.5 | 84.0 | 77.6 | 61.3 | 28.4 | 9.1 | 14.0 | 13.1 |
| Printed | 51.6 | 80.5 | 92.5 | 99.1 | 99.2 | 98.0 | 92.1 | 81.8 | 42.8 | 7.6 | 12.1 | 9.5 |

**Handwritten model**

| Angle | -30 | -15 | 0 | 15 | 30 | 90 | 180 | 270 |
|---|---|---|---|---|---|---|---|---|
| Handwritten | 82.6 | 96.8 | 98.5 | 97.2 | 84.6 | 14.2 | 27.0 | 16.6 |
| Printed | 52.0 | 61.6 | 63.4 | 59.4 | 47.5 | 6.9 | 8.4 | 4.4 |

- **What this means:** all three models are reliable only up to about 10 to 15 degrees of tilt (the training augmentation rotated by at most 10 degrees). At 20 degrees accuracy has already dropped by several points, at 30 degrees it is roughly 73% to 86% for the combined model, at 45 degrees it is below 52%, and for sideways or upside-down images (90, 180, 270 degrees) it is close to chance (a random guess among 32 letters would be about 3%).
- The printed and handwritten rows of the specialist models at 0 degrees (86.5% for the printed model on handwriting, 63.4% for the handwritten model on printed letters) repeat the "pick the right mode" limitation. On tilted handwriting the combined model is clearly better than the printed model (for example 85.6% against 61.3% at 30 degrees). On tilted printed letters it is a few points lower than the printed model (for example 72.6% against 81.8% at 30 degrees, and 96.8% against 98.0% at 10 degrees).
- This is a measurement on rotated copies of the datasets' own test images. Real photos have other kinds of distortion (perspective, shadows), which were not measured.
- Results for a model trained with `--aug-p-rotate` are **not** reported here yet.

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
- **One isolated letter per image.** None of the models reads words or sentences.
- **Combined model:** it removes the need to choose between printed and handwritten, but its test results come only from the two datasets above. It has the same dot-related look-alike mistakes as the handwritten model, and its behavior on other people's handwriting, other fonts and photos was not tested.
- **Supported files:** PNG, JPG, JPEG, BMP. Image sides must be between 16 and 8000 pixels. Blank images are rejected with an error.
- **Always an answer.** Each model always picks one of its 32 classes, including for blank-looking, noisy, rotated or multi-letter images and for symbols that are not Persian letters.
- **Confidence is not proof.** It is the softmax probability of the top class. The 70% warning threshold is a simple cut-off that was not tuned or validated.
- **Handwriting:**
  - The handwritten model was trained on tightly cropped 64x64 images from one dataset (10 forms). Very different handwriting styles, cluttered backgrounds and uncropped photos were not tested.
  - Look-alike letters that differ only by dots (for example ب پ ت ث, ج چ ح خ, ر ز ژ, د ذ) are the most common mistakes, and merged or missing dots make them harder.
  - Thresholding during preprocessing makes strokes a bit heavier and can join nearby dots.
- **Printed model:** the test split is by image, not by font, so the test score is likely optimistic for unseen fonts.
- **Rotation:** measured, see [Rotation robustness](#rotation-robustness). Tilts beyond about 15 degrees reduce accuracy noticeably, and sideways or upside-down letters are not recognized. Blur was not evaluated systematically.
- **Letter shapes:** both models recognize the 32 isolated letters, not the connected forms letters take inside words.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `'streamlit' is not recognized` | The virtual environment is not active or Streamlit is not installed. Run `.venv\Scripts\activate`, then `pip install streamlit`. Or use `python -m streamlit run app.py`. |
| The page looks unstyled or the fonts look different | The interface loads its fonts from Google Fonts, so they need an internet connection. Offline, the app falls back to system fonts and still works. |
| The styling did not change after copying a new `app.py` | Hard-refresh the browser page (Ctrl+F5) and restart `streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'torch'` (or `cv2`, `PIL`, `pandas`) | A package is missing. Activate `.venv` and install it: `pip install torch torchvision`, `pip install opencv-python`, `pip install pillow`, `pip install pandas`. |
| PowerShell says running scripts is disabled when activating `.venv` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt (`cmd`) instead. |
| Printed mode shows "Model checkpoint not found" | `models\mobilenetv3_small_best.pt` is missing. See [Model Checkpoint Setup](#model-checkpoint-setup). |
| Handwritten mode says the handwritten model has not been trained | `models\handwritten_mobilenetv3_small_best.pt` is missing. Get the file from the teammate who trained it, or train it (see [Training and Evaluation Commands](#training-and-evaluation-commands)). Printed mode still works. |
| Combined mode says the combined model has not been trained | `models\combined_mobilenetv3_small_best.pt` is missing. Get the file from the teammate who trained it, or train it with `python src\train_combined.py`. The other modes still work. |
| `train_combined.py` says the printed manifest has no `glitch` column | Run `python src\mark_glitch.py` first (it adds the column), as for the printed model. |
| `ModuleNotFoundError: No module named 'predict_combined'` (or `augment_cfg`) | The combined-model files are missing from `src\`. Make sure `augment_cfg.py`, `preview_augmentation.py`, `train_combined.py`, `evaluate_combined.py` and `predict_combined.py` are in the `src` folder. |
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
- A `.gitignore` is included in the project root (make sure the file name keeps its leading dot when you copy it). It ignores the virtual environment, `__pycache__`, `dataset/raw/`, `dataset/processed/`, smoke-test models and reports, Streamlit secrets and `.env` files, logs, zip files, and editor and operating-system files. It deliberately does **not** ignore the manifests, the final model files in `models/` or the final reports.
- If a model file ever grows beyond 100 MB, use Git LFS or add it to `.gitignore`.

- Never commit passwords, API keys or personal files.

## Future Improvements

These are ideas, **not** completed features.

- Test the handwritten and combined models on new handwriting (other people, pens, paper, phone photos) and other printed fonts, and report the results honestly
- Compare augmentation settings one at a time to find out which ones actually help
- Review the confidently wrong test samples for mislabeled data and clean the handwritten dataset
- Handle dots better for handwriting (for example a gentler threshold, or a larger input for the dotted look-alike letters)
- Stroke-thickness augmentation, and a split by known writer if writer information becomes available
- Automatic detection of printed vs handwritten input
- Evaluate a model trained with `--aug-p-rotate` (for example 30 degrees) and check that upright accuracy does not drop
- Automatic orientation correction for sideways or upside-down letters (for example a small model that predicts 0, 90, 180 or 270 degrees)
- Better preprocessing for noisy backgrounds and photos
- Tuning the low-confidence warning, or adding a real "not a valid character" check
- Recognizing several characters or whole words
- A `requirements.txt` with pinned versions

## Contributors

| Name | Role |
|---|---|
| Adrian Paul Sande | Model Training and Evaluation |
| Marc Louie Marimon | Data Preprocessing and Analysis |
| Alec Lacap | Backend Development and Model Integration |
| Ella Marie Tiozon | Frontend Development and User Interface |
| Adrienne Lalimarmo | Documentation and Presentation |


## License

All rights are reserved by the authors.