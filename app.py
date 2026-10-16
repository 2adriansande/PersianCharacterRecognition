"""Phase 6: Streamlit front end for the Persian Character Recognition model.

Run from the project root:
    streamlit run app.py

Uses the existing src/predict.py (load_predictor / Predictor.predict). No training or
evaluation happens here; the saved checkpoint in models/ is only loaded and queried.
"""
import hashlib
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import load_predictor          # noqa: E402
from preprocess import InvalidImageError    # noqa: E402

ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp"]

st.set_page_config(page_title="Persian Character Recognition", page_icon="🔤", layout="centered")

st.markdown(
    """
    <style>
    :root { --accent: #0d9488; --accent-2: #6366f1; }
    .block-container { max-width: 880px; padding-top: 2rem; }
    #MainMenu, footer { visibility: hidden; }

    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        color: #fff;
    }
    .hero h1 { margin: 0 0 .35rem 0; padding: 0; font-size: 2rem; color: #fff; }
    .hero p { margin: 0; opacity: .92; font-size: 1.02rem; }
    .scope-chip {
        display: inline-block; margin: .9rem 0 0; padding: .25rem .75rem; border-radius: 999px;
        background: rgba(255,255,255,.2); font-size: .82rem; font-weight: 600;
    }

    .panel-title { font-size: .8rem; font-weight: 700; letter-spacing: .08em;
                   text-transform: uppercase; opacity: .65; margin: 0 0 .5rem; }

    .result-card {
        border: 1px solid rgba(128,128,128,.3); border-radius: 18px; padding: 1.2rem 1.2rem 1rem;
        background: rgba(128,128,128,.07); text-align: center;
    }
    .rtl-char {
        direction: rtl; unicode-bidi: isolate; text-align: center;
        font-size: 7rem; line-height: 1.25; font-weight: 600;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .chip { display: inline-block; padding: .2rem .8rem; border-radius: 999px; font-weight: 600;
            border: 1px solid rgba(128,128,128,.4); margin: 0 .2rem; font-size: .95rem; }
    .conf-label { display: flex; justify-content: space-between; font-size: .85rem;
                  margin: 1rem 0 .3rem; opacity: .8; }
    .conf-track { height: 10px; border-radius: 999px; background: rgba(128,128,128,.25); overflow: hidden; }
    .conf-fill { height: 100%; border-radius: 999px;
                 background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
    .conf-fill.low { background: #f59e0b; }
    .rtl-small { direction: rtl; unicode-bidi: isolate; font-size: 1.6rem; }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)); border: 0; font-weight: 600;
    }
    div.stButton > button { border-radius: 12px; padding: .55rem 1rem; }
    [data-testid="stFileUploaderDropzone"] { border-radius: 14px; border-width: 2px; }
    [data-testid="stImage"] img { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor():
    """Loads the trained checkpoint once per server process."""
    return load_predictor()


def reset_state():
    """Clear the current result and the uploader (changing its key empties it)."""
    st.session_state.pop("result", None)
    st.session_state.pop("result_key", None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


st.session_state.setdefault("uploader_key", 0)

# ---------------------------------------------------------------- header
st.markdown(
    """
    <div class="hero">
      <h1>Persian Character Recognition</h1>
      <p>Upload an image of a single Persian letter and a MobileNetV3 Small classifier
         predicts which of the 32 letters of the Persian alphabet it shows.</p>
      <span class="scope-chip">Printed / digital characters only · handwriting not supported</span>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.expander("About the model's limits"):
    st.write(
        "The current model was trained on **printed / digital** Persian characters only. "
        "Handwritten characters are **not supported**. The model always answers with one of its 32 "
        "known letters, even for handwriting, blank-ish, rotated or multi-character images, so a "
        "high confidence score does not guarantee a correct or valid result."
    )

# ---------------------------------------------------------------- model
try:
    predictor = get_predictor()
except FileNotFoundError:
    st.error("Model checkpoint not found. Expected `models/mobilenetv3_small_best.pt` in the project folder.")
    st.stop()
except Exception as e:  # corrupted checkpoint, label mismatch, missing dependency, ...
    st.error(f"Could not load the model: {e}")
    st.stop()

# ---------------------------------------------------------------- upload
uploaded = st.file_uploader(
    "Upload a character image (PNG, JPG, JPEG or BMP)",
    type=ALLOWED_TYPES,
    key=f"uploader_{st.session_state['uploader_key']}",
)

if uploaded is None:
    st.session_state.pop("result", None)
    st.session_state.pop("result_key", None)
    st.info("Choose an image to begin.")
    st.stop()

data = uploaded.getvalue()
file_key = hashlib.sha256(data).hexdigest()
if st.session_state.get("result_key") != file_key:   # new file -> drop the old prediction
    st.session_state.pop("result", None)

try:
    image = Image.open(uploaded)
    image.load()
except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
    st.error(f"This file could not be read as an image ({e}). Please upload a valid PNG, JPG, JPEG or BMP.")
    st.stop()

col_img, col_out = st.columns(2, gap="large")
with col_img:
    st.markdown('<div class="panel-title">Uploaded image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(f"{uploaded.name} · {image.width}×{image.height}px")

# ---------------------------------------------------------------- predict
btn_col, reset_col = st.columns([3, 1])
recognize = btn_col.button("Recognize Character", type="primary", use_container_width=True)
reset_col.button("Reset", on_click=reset_state, use_container_width=True)

if recognize:
    try:
        with st.spinner("Recognizing..."):
            st.session_state["result"] = predictor.predict(image)
            st.session_state["result_key"] = file_key
    except InvalidImageError as e:
        st.session_state.pop("result", None)
        st.error(f"Invalid image: {e}")
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- result
result = st.session_state.get("result")
if result is not None and st.session_state.get("result_key") == file_key:
    with col_out:
        pct = result["confidence"] * 100
        low = " low" if result["low_confidence"] else ""
        st.markdown('<div class="panel-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-card">
              <div class="rtl-char">{result["char"]}</div>
              <span class="chip">{result["label"]}</span>
              <span class="chip">{result["codepoint"]}</span>
              <div class="conf-label"><span>Confidence</span><strong>{pct:.1f}%</strong></div>
              <div class="conf-track"><div class="conf-fill{low}" style="width:{pct:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for w in result["warnings"]:
        st.warning(w)
    if result["low_confidence"]:
        st.warning("Low confidence. The image may be unclear, rotated, handwritten or not a single printed letter.")

    with st.expander("Top 3 predictions and model input"):
        for rank, t in enumerate(result["top3"], 1):
            st.markdown(
                f'{rank}. <span class="rtl-small">{t["char"]}</span> &nbsp; `{t["label"]}` — {t["confidence"] * 100:.1f}%',
                unsafe_allow_html=True,
            )
        st.image(result["crop"], caption="Cropped and normalized image given to the model", width=160, clamp=True)