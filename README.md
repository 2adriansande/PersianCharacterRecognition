# Persian Character Recognition System

A deep-learning system that recognizes **one isolated Persian letter** in an image and tells you which of the 32 letters of the Persian alphabet it is. It uses a MobileNetV3 Small model built with PyTorch, and a Streamlit web app for uploading images and seeing the result.

## Table of Contents

1. [Current Project Status](#current-project-status)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Installation Requirements](#installation-requirements)
8. [Installation Steps](#installation-steps)
9. [Model Checkpoint Setup](#model-checkpoint-setup)
10. [How to Run the Application](#how-to-run-the-application)
11. [How to Make Predictions](#how-to-make-predictions)
12. [Example Usage](#example-usage)
13. [Evaluation Results](#evaluation-results)
14. [Limitations](#limitations)
15. [Troubleshooting](#troubleshooting)
16. [Preparing the Repository for GitHub](#preparing-the-repository-for-github)
17. [Future Improvements](#future-improvements)
18. [Contributors](#contributors)
19. [License](#license)

---

## Current Project Status

| Area | Status |
|---|---|
| Printed / digital Persian characters | **Supported** |
| Handwritten Persian characters | **Not supported.** The model was not trained on handwriting. |
| Words, sentences, full lines | Not supported (one isolated letter per image) |
| Model training and evaluation | Done (see [Evaluation Results](#evaluation-results)) |
| Streamlit web app | Implemented in `app.py` |

The model always answers with one of its 32 known letters, even when the image is handwritten, blank-looking, rotated or contains several characters. See [Limitations](#limitations).

## Features

Everything listed here is implemented in the current code.

- Upload an image (PNG, JPG, JPEG or BMP) and preview it
- Prediction shown as the Persian letter (right-to-left), its Latin name (for example `alef`) and its Unicode code point (for example `U+0627`)
- Confidence score as a percentage, plus the top 3 predictions
- A view of the cropped, normalized image that is actually given to the model
- Automatic preprocessing: converts to grayscale, detects dark-on-light or light-on-dark, crops to the character, pads to a square without stretching, resizes to 224x224
- Clear error messages for unreadable files, too-small or too-large images, and blank images
- Warnings for low confidence (below 70%, a simple untuned threshold) and for characters touching the image border
- A "Reset" button to clear the result and upload another image
- Command-line prediction and a self-test in `src/predict.py`

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
app.py                      Streamlit web application (run this)
.streamlit/config.toml      Optional Streamlit theme setting
src/
  predict.py                Loads the model and predicts one image
  preprocess.py             Validation, binarization, cropping, resizing
  model.py                  MobileNetV3 Small with a 32-class output layer
  labels.py                 The 32 letters: index, Persian character, Latin name
  augment.py                Training-time image augmentation
  prepare_data.py           Cleans the dataset, removes duplicates, makes the splits
  train.py                  Model training
  evaluate.py               Final evaluation on the test split
  (other helper scripts)    inspect_dataset*.py, review_drops.py, flag_artifacts.py, ...
models/
  mobilenetv3_small_best.pt Trained model (needed to run the app)
  model_config.json         Readable copy of the model settings (not loaded by the app)
reports/                    Evaluation results, confusion matrix, training curves
dataset/
  manifest.csv              One row per unique image: class, source, split
  processed/                Cached training arrays and logs (large, not for GitHub)
tests/                      Unit tests for the data preparation
```

## Dataset

- **Classes:** 32, one for each Persian letter. The class list is defined in `src/labels.py`:

  ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی

- **Style:** printed / digital only (every row in `manifest.csv` has style `printed`).
- **Sources:** two merged collections, named `ds101` and `alpha` in `manifest.csv`.
- **Preparation** (`src/prepare_data.py`):
  1. Scan the class folders `01_alef` to `32_ye`.
  2. Compute a small "look-alike signature" for each image.
  3. Remove duplicates inside a class, images whose look-alike appears under two different classes, blank images, and mislabeled wide "word" images.
  4. Split the remaining unique images per class into train, validation and test.
- **Size after cleaning:** 10,638 unique images, split into 7,448 train, 1,595 validation and 1,595 test. 49 images are flagged as glitches in the manifest; 6 of them are in the test split and are left out of evaluation.
- **Persian letters:** kaf (U+06A9) and yeh (U+06CC) are used, not the Arabic versions U+0643 and U+064A.
- **The images are not stored in this repository.** `manifest.csv` records the file paths from the machine where the data was prepared. To retrain or re-evaluate you need your own copy of the images; `prepare_data.py --root` and `evaluate.py --data-root` take the folder location. **Running the app does not need the dataset.**

> **TODO:** add the dataset names, links, licenses and citations for the two sources (`ds101` and `alpha`).

## Model Architecture

- **Network:** MobileNetV3 Small from torchvision, starting from ImageNet weights.
- **Output layer:** the last layer is replaced by a linear layer with 32 outputs (`classifier[3] = Linear(1024, 32)`).
- **Input:** 224x224 image, three channels, normalized with ImageNet mean and standard deviation.
- **Training:** stage 1 trains only the new output layer (3 epochs); stage 2 fine-tunes from feature block 8 onward (up to 25 epochs, early stopping, best epoch chosen on the validation split). AdamW, OneCycle learning rate, cross-entropy with label smoothing 0.05 and class weights, small random rotation, zoom, shear and shift as augmentation. Seed 42.
- **Prediction:** the model gives 32 scores; softmax turns them into probabilities. The top probability is the confidence shown in the app.

## Installation Requirements

- **Python:** the project was developed with Python 3.12 (this is inferred from the cached files in the project). Python 3.12 is the safest choice. *TODO: confirm the exact version the author used.*
- **Operating system:** commands below are for Windows with VS Code. A GPU is not required; the app runs on CPU.
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

The app needs the trained model file:

```
models\mobilenetv3_small_best.pt
```

- The class names and the Persian letter for each class are stored **inside** this file. When the model loads, the code checks them against `src/labels.py`, so both must be present and must match. If they differ, loading stops with an error.
- `models\model_config.json` is only a readable summary. The app does not load it.
- The file is about 6 MB, which is small enough for a normal GitHub repository. If it is not in your copy of the repository, **ask the teammate who trained the model to share it** (for example by a shared drive) and place it at the path above. There is no public download link for this project.
- You do not need to retrain. Training (`src/train.py`) needs the full image dataset and takes much longer.

## How to Run the Application

From the project root (the folder that contains `app.py`), with the virtual environment active:

```
streamlit run app.py
```

This starts a local web server and normally opens your browser at http://localhost:8501. Press **Ctrl+C** in the terminal to stop it.

## How to Make Predictions

1. Click **Browse files** (or drag an image in) and choose a PNG, JPG, JPEG or BMP file with **one** printed Persian letter.
2. Check the preview.
3. Click **Recognize Character**.
4. Read the result:
   - **Large letter:** the predicted Persian character.
   - **Label chip:** its Latin name, for example `be` or `shin`.
   - **Code chip:** its Unicode code point.
   - **Confidence bar:** the model's probability for its top answer, as a percentage. The bar turns amber below 70%.
5. Open **Top 3 predictions and model input** to see the next best guesses and the cropped image the model used.
6. Click **Reset** to clear everything and upload another image.

**How to read the confidence:** it is the model's own probability, not a measured accuracy. Wrong answers can still show high confidence, and unsuitable images (handwriting, several letters, rotated letters) can also get a confident answer. Use it as a hint, not as proof.

**Best input:** a clear, dark letter on a plain light background, one letter only, reasonably centered, not touching the edge, at least 16x16 pixels.

## Example Usage

Using the web app:

1. Run `streamlit run app.py`.
2. Upload a screenshot of a single printed letter, for example ب.
3. Click **Recognize Character**.
4. The app shows the letter ب, the label `be`, the code `U+0628` and a confidence percentage.

Using the command line (from the project root, with the virtual environment active):

```
python src\predict.py path\to\image.png
```

This prints the prediction for your own image. You can pass several image paths. To check that the saved model loads and works on test images from the dataset (this needs the dataset images on your computer):

```
python src\predict.py --selftest
```

From your own Python code:

```python
import sys
sys.path.insert(0, "src")
from predict import load_predictor

predictor = load_predictor()
result = predictor.predict("path/to/image.png")   # path, PIL image or NumPy array
print(result["char"], result["label"], result["confidence"])
```

## Evaluation Results

These results come from the held-out test split, evaluated once after training (file `reports/eval_summary.json`):

| Metric | Value |
|---|---|
| Test images | 1,589 |
| Accuracy | 98.80% |
| Macro F1-score | 0.988 |
| Best validation accuracy (epoch 28) | 98.74% |
| Mean confidence, correct predictions | 93.4% |
| Mean confidence, wrong predictions | 56.7% |

Per-class scores, the confusion matrix and misclassified examples are in the `reports/` folder. Median prediction time was about 14.8 ms per image on a GPU (measured on 300 images; CPU time was not measured).

**Read these numbers carefully:** the dataset has no font or writer information, so the split is by unique image, not by font. Accuracy on fonts the model has never seen is probably lower. The robustness tests (blurry, rotated, other fonts, several letters, handwriting) are not part of these numbers.

> **TODO:** if you run the manual robustness tests, add what you observed here.

## Limitations

- **Printed / digital characters only.** Handwriting is not supported.
- **One isolated letter per image.** It does not read words or sentences.
- **Supported files:** PNG, JPG, JPEG, BMP. Image sides must be between 16 and 8000 pixels. Blank images are rejected with an error.
- **It always picks one of the 32 letters.** This includes handwriting, noise, rotated letters, several letters, and symbols that are not Persian letters.
- **Confidence is not reliable proof.** The 70% warning threshold is a simple cut-off that was not tuned or validated.
- **Optimistic test score** because the split is not by font (see above).
- **Rotation and blur** were not evaluated systematically.
- **Letter shapes:** the model recognizes the 32 isolated letters, not the connected forms letters take inside words.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `'streamlit' is not recognized` | The virtual environment is not active or Streamlit is not installed. Run `.venv\Scripts\activate`, then `pip install streamlit`. Or use `python -m streamlit run app.py`. |
| `ModuleNotFoundError: No module named 'torch'` (or `cv2`, `PIL`, `pandas`) | A package is missing. Activate `.venv` and install it: `pip install torch torchvision`, `pip install opencv-python`, `pip install pillow`, `pip install pandas`. |
| PowerShell says running scripts is disabled when activating `.venv` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt (`cmd`) instead. |
| App shows "Model checkpoint not found" | `models\mobilenetv3_small_best.pt` is missing. See [Model Checkpoint Setup](#model-checkpoint-setup). |
| "Could not load the model: Class mapping in checkpoint differs from src/labels.py" | `src/labels.py` was changed, or the model file is from a different version. Use the model file and `labels.py` from the same version. |
| `streamlit run app.py` says it cannot find `app.py` | You are in the wrong folder. Use `cd` to go to the project root, the folder that contains `app.py`. |
| "Invalid image: Image appears blank" / "No character found" | The image has no visible letter, or the letter is too faint. Use a darker letter on a plain background. |
| "Image size ... outside [16, 8000] px" | The image is too small or too large. Resize it. |
| "This file could not be read as an image" | The file is damaged or is not really an image. Use a real PNG, JPG, JPEG or BMP. |
| Port 8501 already in use | Another Streamlit app is running. Stop it, or use `streamlit run app.py --server.port 8502`. |
| Data scripts fail with "file not found" for images | `manifest.csv` holds paths from the original computer. Pass your own image folder with `--data-root` (evaluate) or `--root` (prepare_data). |
| Very slow first prediction | The model is loading for the first time. Later predictions are faster. |

## Preparing the Repository for GitHub

- **Commit:** the source code (`app.py`, `src/`, `tests/`), `README.md`, `dataset/manifest.csv` (small), the model file `models/mobilenetv3_small_best.pt` (about 6 MB) and `models/model_config.json`, and the small files in `reports/`.
- **Do not commit the image dataset or `dataset/processed/`.** The processed folder is about 468 MB, and GitHub rejects single files over 100 MB. Check the dataset's license and your project's privacy rules before sharing any images.
- **Check `manifest.csv`:** it contains absolute paths from the author's computer, including a Windows user folder name. Decide whether to share it as is or replace the paths with relative ones.
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

- Handwritten character recognition (needs a handwritten dataset and retraining)
- Better preprocessing for noisy backgrounds, photos and rotated letters
- Broader testing: more fonts, blur, rotation and other conditions, with a split by font
- Tuning the low-confidence warning, or adding a real "not a valid character" check
- Recognizing several characters or whole words
- A `requirements.txt` with pinned versions

## Contributors

| Name | Role |
|---|---|
| *TODO: name* | *TODO: role (for example model training, web app, documentation)* |
| *TODO: name* | *TODO: role* |

## License

No license has been specified for this project. Until the team chooses one, all rights are reserved by the authors. Also check the licenses of the dataset sources before sharing images or trained models.
