"""Phase 6: Streamlit front end for the Persian Character Recognition model.

Run from the project root:
    streamlit run app.py

Three recognition modes (sidebar): printed characters (src/predict.py, models/mobilenetv3_small_best.pt),
handwritten characters (src/predict_hw.py, models/handwritten_mobilenetv3_small_best.pt) and a combined
printed + handwritten model (src/predict_combined.py, models/combined_mobilenetv3_small_best.pt).
No training or evaluation happens here; the saved checkpoints in models/ are only loaded and queried.
"""
import hashlib
import html
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import load_predictor, DEFAULT_CKPT         # noqa: E402
from predict_hw import load_handwritten_predictor, HW_CKPT   # noqa: E402
from predict_combined import load_combined_predictor, COMBINED_CKPT   # noqa: E402
from preprocess import InvalidImageError                 # noqa: E402

ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp"]
SHORT = {"Printed Persian Characters": "Printed", "Handwritten Persian Characters": "Handwritten",
         "Combined (Printed + Handwritten)": "Combined"}
HISTORY_MAX = 8
PRINTED, HANDWRITTEN = "Printed Persian Characters", "Handwritten Persian Characters"
COMBINED = "Combined (Printed + Handwritten)"
MODE_INFO = {
    PRINTED: dict(
        chip="Printed / digital characters",
        limits=("The printed model was trained on **printed / digital** Persian characters only. "
                "Handwritten characters are **not supported** in this mode; use the Handwritten mode "
                "in the sidebar for those. The model always answers with one of its 32 known letters, even for "
                "handwriting, blank-ish, rotated or multi-character images, so a high confidence score does "
                "not guarantee a correct or valid result."),
        low="Low confidence. The image may be unclear, rotated, handwritten or not a single printed letter.",
    ),
    HANDWRITTEN: dict(
        chip="Handwritten characters · one isolated letter",
        limits=("The handwritten model recognizes **one isolated handwritten letter** per image. It was trained on "
                "tightly cropped, light-background letter images from a collection of handwriting forms, so errors "
                "are likely for very different handwriting styles, photos with cluttered backgrounds, several "
                "letters in one image, and look-alike letters (for example ب پ ت ث or ج چ ح خ). It always "
                "answers with one of its 32 known letters, so a high confidence score does not guarantee a "
                "correct or valid result."),
        low="Low confidence. The handwriting may be unclear, very different from the training data, or not a single isolated letter.",
    ),
    COMBINED: dict(
        chip="Printed or handwritten · one isolated letter",
        limits=("The combined model recognizes **one isolated letter** that is either printed / digital or handwritten, "
                "using a single network. On the held-out test images of our datasets it performed about as well as the "
                "two separate models. It has not been tested on other people's handwriting, photos with cluttered "
                "backgrounds or unusual fonts. Look-alike letters that differ only by dots (for example ب پ ت ث or "
                "ج چ ح خ) are the most common mistakes. It always answers with one of its 32 known letters, so a high "
                "confidence score does not guarantee a correct or valid result."),
        low="Low confidence. The image may be unclear, rotated, not a single isolated letter, or in a style the model has not seen.",
    ),
}

st.set_page_config(page_title="Persian Character Recognition", page_icon="🔤", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Inter:wght@400;500;600&family=Vazirmatn:wght@500;700&display=swap');
:root {
  --ink: #0b3f3c; --teal: #0e5a54; --teal-soft: #e3efe9; --gold: #c8962e; --cream: #f7f4ec;
  --card: #ffffff; --line: #e6e0d1; --muted: #5d6b69;
}
html, body, [data-testid="stApp"] { background: var(--cream); font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1120px; padding: 0 1.5rem 3rem; }
h1, h2, h3, .display, .h1, .h2, .h3, .tip-h { font-family: 'Bricolage Grotesque', 'Inter', sans-serif; letter-spacing: -.02em; }
.fa, .rtl-char, .rtl-small, .art-letter, .float, .tip-ex b { font-family: 'Vazirmatn', 'Segoe UI', Tahoma, sans-serif; }
a, a:hover, a:visited { color: inherit !important; text-decoration: none !important; }
.btn-main, .btn-main:hover { color: #fff !important; }
h1, h2, h3 { padding: 0 !important; margin-top: 0; }
.eyebrow { font-size: .7rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--teal); }

/* nav */
.nav { display: flex; align-items: center; justify-content: space-between; padding: 1rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.2rem; }
.brand { display: flex; align-items: center; gap: .7rem; }
.logo { width: 38px; height: 38px; border-radius: 10px; background: var(--teal); color: #fff; display: grid; place-items: center; font-size: 1.3rem; font-weight: 700; font-family: 'Vazirmatn', sans-serif; }
.brand b { display: block; font-size: .9rem; line-height: 1.1; } .brand small { font-size: .6rem; letter-spacing: .12em; color: var(--muted); }
.nav-links { display: flex; gap: 1.8rem; font-size: .82rem; font-weight: 500; }
.nav-links a:hover { color: var(--teal); }

/* hero */
.hero { display: grid; grid-template-columns: 1.1fr 1fr; gap: 2rem; align-items: center; padding: 2.4rem 0 3rem;
        background-image: linear-gradient(rgba(14,90,84,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(14,90,84,.05) 1px, transparent 1px);
        background-size: 44px 44px; border-radius: 18px; }
.hero-text { padding-left: .4rem; }
.hero .h1 { font-size: clamp(2.6rem, 6vw, 4.3rem); line-height: 1; margin: .9rem 0 1.2rem; font-weight: 500; color: var(--ink); }
.hero .h1 .accent { color: #0f7a6f; display: block; font-weight: 700; }
.hero p.lead { color: var(--muted) !important; font-size: 1.05rem !important; max-width: 30rem; margin: 0 0 1.6rem; line-height: 1.5; }
.btns { display: flex; gap: .9rem; align-items: center; }
.btn-main { background: var(--teal); color: #fff !important; padding: .8rem 1.3rem; border-radius: 9px; font-weight: 600; font-size: .88rem; }
.btn-main:hover { background: var(--ink); }
.btn-ghost { font-weight: 600; font-size: .88rem; padding: .8rem .6rem; }
.ticks { display: flex; gap: 1.2rem; margin-top: 1.3rem; font-size: .72rem; color: var(--muted); }
.ticks span::before { content: "✓ "; color: var(--teal); font-weight: 700; }
.hero-art { position: relative; height: 330px; margin-right: .6rem; }
.art-card { position: absolute; inset: 0 .5rem 0 1.5rem; background: var(--teal-soft); border-radius: 50px 50px 70px 50px; overflow: hidden;
            box-shadow: 0 30px 60px -30px rgba(11,63,60,.35); }
.art-cap { position: absolute; top: 1.6rem; left: 1.8rem; font-size: .6rem; letter-spacing: .12em; color: var(--muted); line-height: 1.4; }
.art-cap .fa { font-size: .78rem; letter-spacing: 0; color: var(--ink); }
.ring { position: absolute; border: 1.5px solid rgba(14,90,84,.35); border-radius: 40px 60px 60px 40px; }
.r1 { inset: 3.4rem 1.2rem 1.4rem 1.4rem; } .r2 { inset: 4.4rem 2.6rem 2.8rem 2.8rem; background: rgba(255,255,255,.35); }
.art-letter { position: absolute; inset: 3rem 0 0 0; display: grid; place-items: center; font-size: 9.5rem; font-weight: 700; color: var(--teal); line-height: 1; }
.float { position: absolute; width: 52px; height: 52px; border-radius: 14px; background: #fff; display: grid; place-items: center;
         font-size: 1.6rem; font-weight: 600; box-shadow: 0 12px 28px -12px rgba(11,63,60,.4); color: var(--ink); }
.f1 { top: 3.6rem; right: -.2rem; } .f2 { bottom: 4.2rem; left: -.2rem; } .f3 { bottom: .6rem; right: 1.6rem; }
.badge { position: absolute; top: -.6rem; right: 2.2rem; width: 62px; height: 62px; border-radius: 50%; background: #f1dfae; color: var(--ink);
         display: grid; place-items: center; text-align: center; font-size: .56rem; font-weight: 700; letter-spacing: .08em; line-height: 1.2; border: 2px solid #fff; }
.badge b { font-size: 1.15rem; display: block; letter-spacing: 0; }

/* section headings */
.sec { text-align: center; margin: 3.4rem 0 1.6rem; }
.sec.left { text-align: left; }
.num { display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 50%; border: 1.5px solid var(--gold); color: var(--gold); font-size: .65rem; font-weight: 700; margin-bottom: .6rem; }
.sec .h2 { font-size: 2rem; margin: .35rem 0 .5rem; font-weight: 600; color: var(--ink); }
.sec p { color: var(--muted); font-size: .9rem; margin: 0; }

/* workspace: two columns glued into one card (left white, right dark) */
[data-testid="stHorizontalBlock"]:has(.ws-left) { gap: 0 !important; align-items: stretch; }
[data-testid="stColumn"]:has(.ws-left) { background: var(--card); border: 1px solid var(--line); border-right: 0; border-radius: 18px 0 0 18px; padding: 1.6rem 1.7rem 1.3rem; }
[data-testid="stColumn"]:has(.ws-right) { background: var(--teal); color: #fff; border-radius: 0 18px 18px 0; padding: 1.7rem 1.7rem; }
.ws-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: .9rem; }
.ws-head b { font-size: 1.05rem; display: block; margin-top: .15rem; }
.pill { font-size: .58rem; font-weight: 700; letter-spacing: .1em; background: #efece3; color: var(--muted); padding: .25rem .6rem; border-radius: 6px; }
.field-label { font-weight: 600; font-size: .82rem; margin: .8rem 0 .1rem; } .field-help { font-size: .7rem; color: var(--muted); margin-bottom: .4rem; }
.privacy { text-align: center; font-size: .66rem; color: var(--muted); margin-top: .5rem; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed #c9c2ae; border-radius: 14px; background: #fbfaf6; padding: 2.2rem 1rem; }
[data-testid="stFileUploaderDropzone"] button { border-radius: 8px; border: 1px solid var(--line); background: #fff; color: var(--ink); font-weight: 600; }
[data-testid="stFileUploader"] small { color: var(--muted); }
[data-testid="stImage"] img { border-radius: 12px; }
div.stButton > button { border-radius: 9px; padding: .7rem 1rem; font-weight: 600; border: 1px solid var(--line); background: #fff; color: var(--ink); }
div.stButton > button[kind="primary"] { background: var(--teal); color: #fff; border: 0; }
div.stButton > button[kind="primary"]:hover { background: var(--ink); color: #fff; }
div.stButton > button:disabled { opacity: .55; }
/* segmented mode selector */
[data-testid="stRadio"] [role="radiogroup"] { gap: .3rem; background: #f1eee5; padding: .25rem; border-radius: 10px; flex-wrap: wrap; }
[data-testid="stRadio"] label { flex: 1; justify-content: center; padding: .4rem .7rem; border-radius: 8px; margin: 0; cursor: pointer; }
[data-testid="stRadio"] label > div:first-child, [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stRadio"] label p { font-size: .8rem; font-weight: 600; text-align: center; color: var(--muted); }
[data-testid="stRadio"] label:has(input:checked) { background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.12); }
[data-testid="stRadio"] label:has(input:checked) p { color: var(--ink); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #fff; margin-top: .8rem; }

/* tips panel */
.tip-tag { width: 34px; height: 34px; border-radius: 50%; border: 1px solid rgba(255,255,255,.5); display: grid; place-items: center; font-size: .55rem; font-weight: 700; letter-spacing: .08em; }
.tip-h { color: #fff; font-size: 1.35rem; font-weight: 600; margin: 1rem 0 .4rem; }
.tip-sub { color: rgba(255,255,255,.7) !important; font-size: .78rem !important; margin: 0 0 1.1rem !important; }
.tip { display: flex; gap: .8rem; padding: .85rem 0; border-top: 1px solid rgba(255,255,255,.18); }
.tip i { font-style: normal; color: var(--gold); font-size: .62rem; font-weight: 700; padding-top: .2rem; }
.tip b { font-size: .84rem; display: block; } .tip span { font-size: .72rem; color: rgba(255,255,255,.65); }
.tip-ex { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; margin-top: 1rem; }
.tip-ex div { border: 1px solid rgba(255,255,255,.2); background: rgba(255,255,255,.07); border-radius: 12px; text-align: center; padding: 1rem .4rem .6rem; }
.tip-ex b { display: block; font-size: 2rem; line-height: 1.2; font-weight: 600; } .tip-ex small { font-size: .62rem; color: rgba(255,255,255,.65); }
.tip-ex .bad b { color: rgba(255,255,255,.4); }

/* results */
.result-card { border: 1px solid var(--line); border-radius: 18px; padding: 1.4rem 1.4rem 1.2rem; background: #fff; text-align: center; }
.rtl-char { direction: rtl; unicode-bidi: isolate; font-size: 7rem; line-height: 1.25; font-weight: 700; color: var(--teal); }
.chip { display: inline-block; padding: .2rem .8rem; border-radius: 999px; font-weight: 600; background: var(--teal-soft); margin: 0 .2rem; font-size: .88rem; }
.conf-label { display: flex; justify-content: space-between; font-size: .85rem; margin: 1rem 0 .3rem; color: var(--muted); }
.conf-track { height: 10px; border-radius: 999px; background: #ece8dc; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 999px; background: var(--teal); } .conf-fill.low { background: var(--gold); }
.rtl-small { direction: rtl; unicode-bidi: isolate; font-size: 1.6rem; }
.top3 { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 1.2rem 1.4rem; }
.top3 .row { display: flex; align-items: center; justify-content: space-between; padding: .5rem 0; border-bottom: 1px solid var(--line); font-size: .9rem; }
.top3 .row:last-child { border-bottom: 0; }

/* history */
.hist { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .8rem; }
.hist-card { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: .8rem .9rem; }
.hist-card .fa { font-size: 2.2rem; line-height: 1.2; display: block; text-align: center; color: var(--teal); }
.hist-card small { display: block; color: var(--muted); font-size: .68rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.empty { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 1.2rem 1.4rem; font-size: .85rem; color: var(--muted); }
.empty b { color: var(--ink); display: block; }

/* pipeline */
.steps { display: grid; grid-template-columns: repeat(5, 1fr); background: #fff; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.step { padding: 1.1rem 1.1rem 1.3rem; border-right: 1px solid var(--line); }
.step:last-child { border-right: 0; }
.step small { color: var(--gold); font-size: .62rem; font-weight: 700; }
.step .ic { width: 34px; height: 34px; border-radius: 50%; background: var(--teal-soft); display: grid; place-items: center; margin: .6rem 0 .7rem; color: var(--teal); font-size: .8rem; }
.step b { font-size: .88rem; display: block; margin-bottom: .25rem; } .step span { font-size: .72rem; color: var(--muted); line-height: 1.4; display: block; }
.banner { background: var(--teal); color: #fff; border-radius: 14px; padding: 1.8rem 2rem; margin-top: 1rem; display: grid; grid-template-columns: 1.6fr 1fr; gap: 2rem; align-items: center; }
.banner .eyebrow { color: var(--gold); } .banner .h3 { color: #fff; font-size: 1.5rem; font-weight: 600; margin: .5rem 0; }
.banner p { color: rgba(255,255,255,.72) !important; font-size: .78rem !important; margin: 0; line-height: 1.5; }
.facts { display: grid; grid-template-columns: repeat(3, auto); gap: 1.2rem; border-left: 1px solid rgba(255,255,255,.2); padding-left: 1.6rem; }
.facts small { display: block; font-size: .56rem; letter-spacing: .1em; color: rgba(255,255,255,.55); text-transform: uppercase; }
.facts b { font-size: .85rem; } .facts .ok { color: #8fe3c0; } .facts .no { color: #f1c27a; }
.foot { display: flex; justify-content: space-between; align-items: center; margin-top: 3rem; padding-top: 1.4rem; border-top: 1px solid var(--line); font-size: .7rem; color: var(--muted); }
.foot .brand b { font-size: .75rem; }

@media (max-width: 800px) {
  .hero, .banner { grid-template-columns: 1fr; } .hero-art { height: 280px; }
  .steps { grid-template-columns: 1fr 1fr; } .nav-links { display: none; }
  [data-testid="stColumn"]:has(.ws-left) { border-radius: 18px 18px 0 0; border-right: 1px solid var(--line); }
  [data-testid="stColumn"]:has(.ws-right) { border-radius: 0 0 18px 18px; }
  .facts { border-left: 0; padding-left: 0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(mode):
    """Loads the trained checkpoint of the chosen mode once per server process."""
    if mode == HANDWRITTEN:
        return load_handwritten_predictor()
    if mode == COMBINED:
        return load_combined_predictor()
    return load_predictor()


CKPT_OF = {PRINTED: DEFAULT_CKPT, HANDWRITTEN: HW_CKPT, COMBINED: COMBINED_CKPT}


def reset_state():
    """Clear the current result and the uploader (changing its key empties it)."""
    st.session_state.pop("result", None)
    st.session_state.pop("result_key", None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("history", [])

# ---------------------------------------------------------------- nav + hero
st.markdown(
    """
    <div id="top"></div>
    <div class="nav">
      <div class="brand"><div class="logo">ب</div><div><b>Persian Character</b><small>RECOGNITION LAB</small></div></div>
      <div class="nav-links"><a href="#top">Home</a><a href="#recognition">Recognition</a><a href="#history">History</a><a href="#how">About the Model</a></div>
      <div style="width:60px"></div>
    </div>
    <div class="hero">
      <div class="hero-text">
        <div class="eyebrow">● CNN-powered image classification</div>
        <div class="h1">Persian<br>Character<span class="accent">Recognition</span></div>
        <p class="lead">Upload a handwritten or printed Persian character and let the AI identify it.</p>
        <div class="btns"><a class="btn-main" href="#recognition">Start recognition →</a><a class="btn-ghost" href="#how">How it works ›</a></div>
        <div class="ticks"><span>Handwritten input</span><span>Printed input</span><span>Instant preview</span></div>
      </div>
      <div class="hero-art">
        <div class="art-card">
          <div class="art-cap"><span class="fa">نمونه حروف فارسی</span><br>PERSIAN LETTERFORMS</div>
          <div class="ring r1"></div><div class="ring r2"></div><div class="art-letter">ش</div>
        </div>
        <div class="float f1">گ</div><div class="float f2">م</div><div class="float f3">ی</div>
        <div class="badge"><div><b>32</b>LETTERS</div></div>
      </div>
    </div>
    <div class="sec" id="recognition">
      <div class="num">01</div><div class="eyebrow">Recognition workspace</div>
      <div class="h2">Upload a character image</div>
      <p>Use a clear image with one Persian character centered in the frame.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- workspace
left, right = st.columns([1.6, 1], gap="small")
with left:
    st.markdown('<div class="ws-left"></div><div class="ws-head"><div><span class="eyebrow" style="color:var(--muted)">INPUT IMAGE</span>'
                '<b>Choose a file</b></div><span class="pill">REQUIRED</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a character image (PNG, JPG, JPEG or BMP)", type=ALLOWED_TYPES,
        key=f"uploader_{st.session_state['uploader_key']}", label_visibility="collapsed")

    image, data, file_key = None, None, None
    if uploaded is not None:
        data = uploaded.getvalue()
        try:
            image = Image.open(uploaded)
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            st.error(f"This file could not be read as an image ({e}). Please upload a valid PNG, JPG, JPEG or BMP.")
            image = None
    if image is not None:
        st.image(image, width=200)
        st.caption(f"{uploaded.name} · {image.width}×{image.height}px")

    st.markdown('<div class="field-label">Input type</div><div class="field-help">Each type uses its own trained model. '
                'Pick the one that matches your image.</div>', unsafe_allow_html=True)
    mode = st.radio("Input type", [HANDWRITTEN, PRINTED, COMBINED], index=1, key="mode",
                    format_func=SHORT.get, horizontal=True, label_visibility="collapsed")
    info = MODE_INFO[mode]

    if image is not None:
        file_key = hashlib.sha256(data).hexdigest() + "|" + mode    # a mode switch also drops the old result
    if st.session_state.get("result_key") != file_key:              # new file / new mode -> drop the old prediction
        st.session_state.pop("result", None)

    btn_col, reset_col = st.columns([3, 1])
    recognize = btn_col.button("✦  Recognize character", type="primary", use_container_width=True, disabled=image is None)
    reset_col.button("Reset", on_click=reset_state, use_container_width=True)
    st.markdown('<div class="privacy">Your image is processed in this session only and is not saved to disk.</div>',
                unsafe_allow_html=True)
    with st.expander("About the model's limits"):
        st.write(info["limits"])

with right:
    st.markdown(
        """
        <div class="ws-right"></div>
        <div class="tip-tag">TIP</div>
        <div class="tip-h">For the best result</div>
        <p class="tip-sub">Small adjustments to your image can make recognition more reliable.</p>
        <div class="tip"><i>01</i><div><b>One character only</b><span>Avoid words or connected letters.</span></div></div>
        <div class="tip"><i>02</i><div><b>High contrast</b><span>Use a dark character on a light background.</span></div></div>
        <div class="tip"><i>03</i><div><b>Keep it centered</b><span>Leave clear space around the letterform.</span></div></div>
        <div class="tip"><i>04</i><div><b>Roughly upright</b><span>Slight tilt is fine. Sideways or upside-down images are not supported.</span></div></div>
        <div class="tip-ex"><div><b>ب</b><small>✓ Ideal</small></div><div class="bad"><b>سلام</b><small>✕ Avoid</small></div></div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- predict
if recognize and image is not None:
    try:
        predictor = get_predictor(mode)
    except FileNotFoundError:
        if mode == HANDWRITTEN:
            st.error("The handwritten model has not been trained yet: `models/handwritten_mobilenetv3_small_best.pt` "
                     "was not found. Train it with `python src\\prepare_data_hw.py --root <dataset folder>` and "
                     "`python src\\train_hw.py`, or copy the trained file to that path. "
                     "Another input type may still work; choose it above.")
        elif mode == COMBINED:
            st.error("The combined model has not been trained yet: `models/combined_mobilenetv3_small_best.pt` "
                     "was not found. Train it with `python src\\train_combined.py`, or copy the trained file to that path. "
                     "The Printed and Handwritten types still work; choose one above.")
        else:
            st.error("Model checkpoint not found. Expected `models/mobilenetv3_small_best.pt` in the project folder.")
        predictor = None
    except Exception as e:  # corrupted checkpoint, label mismatch, missing dependency, ...
        st.error(f"Could not load the {mode.lower()} model: {e}")
        predictor = None
    if predictor is not None:
        try:
            with st.spinner("Recognizing..."):
                res = predictor.predict(image)
            st.session_state["result"] = res
            st.session_state["result_key"] = file_key
            hist = st.session_state["history"]
            hist.insert(0, dict(char=res["char"], label=res["label"], pct=res["confidence"] * 100,
                                mode=SHORT[mode], name=uploaded.name))
            del hist[HISTORY_MAX:]
        except InvalidImageError as e:
            st.session_state.pop("result", None)
            st.error(f"Invalid image: {e}")
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- result
result = st.session_state.get("result")
if result is not None and file_key is not None and st.session_state.get("result_key") == file_key:
    pct = result["confidence"] * 100
    low = " low" if result["low_confidence"] else ""
    st.markdown('<div class="sec left"><div class="eyebrow">Prediction</div><div class="h2">Result</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
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
    with c2:
        rows = "".join(
            f'<div class="row"><span>{rank}. <span class="rtl-small">{t["char"]}</span> &nbsp; <code>{t["label"]}</code></span>'
            f'<strong>{t["confidence"] * 100:.1f}%</strong></div>'
            for rank, t in enumerate(result["top3"], 1))
        st.markdown(f'<div class="top3"><div class="eyebrow" style="margin-bottom:.4rem">Top 3 predictions</div>{rows}</div>',
                    unsafe_allow_html=True)
        st.image(result["crop"], caption="Cropped and normalized image given to the model", width=140, clamp=True)
    for w in result["warnings"]:
        st.warning(w)
    if result["low_confidence"]:
        st.warning(info["low"])

# ---------------------------------------------------------------- history (this session only)
hist = st.session_state["history"]
if hist:
    cards = "".join(
        f'<div class="hist-card"><span class="fa">{h["char"]}</span><b style="font-size:.8rem">{html.escape(h["label"])} · {h["pct"]:.0f}%</b>'
        f'<small>{h["mode"]} · {html.escape(h["name"])}</small></div>' for h in hist)
    body = f'<div class="hist">{cards}</div>'
else:
    body = ('<div class="empty"><b>No results yet</b>Predictions you make appear here for this browser session. '
            'Nothing is saved to disk, and the list clears when you close or reload the page.</div>')
st.markdown(
    f"""
    <div class="sec left" id="history"><div class="num">02</div><div class="eyebrow">Recognition history</div>
      <div class="h2">Your recent results</div></div>{body}
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- pipeline + model banner
ckpt_ok = CKPT_OF[mode].exists()
status = '<b class="ok">Checkpoint found</b>' if ckpt_ok else '<b class="no">Not found</b>'
st.markdown(
    f"""
    <div class="sec left" id="how"><div class="num">03</div><div class="eyebrow">Under the hood</div>
      <div class="h2">From image to prediction</div>
      <p>A focused five-step pipeline prepares each image for character-level classification.</p></div>
    <div class="steps">
      <div class="step"><small>01</small><div class="ic">↑</div><b>Upload</b><span>Select a clear image containing a single character.</span></div>
      <div class="step"><small>02</small><div class="ic">◇</div><b>Preprocess</b><span>Grayscale, Otsu binarization, crop to the character and pad to a square.</span></div>
      <div class="step"><small>03</small><div class="ic">▦</div><b>Normalize</b><span>Resize to 224×224 and normalize pixel values.</span></div>
      <div class="step"><small>04</small><div class="ic">◉</div><b>CNN inference</b><span>Pass the tensor through the trained MobileNetV3 Small.</span></div>
      <div class="step"><small>05</small><div class="ic">✓</div><b>Results</b><span>Rank the 32 letters by confidence score.</span></div>
    </div>
    <div class="banner">
      <div><div class="eyebrow">✦ Research prototype</div><div class="h3">Convolutional Neural Network</div>
        <p>Three trained models (printed, handwritten and combined) share one architecture and one 32-letter label set.
           Accuracy figures for each model are reported in the project README.</p></div>
      <div class="facts"><div><small>Architecture</small><b>MobileNetV3 Small</b></div>
        <div><small>Task</small><b>Single character</b></div>
        <div><small>{html.escape(SHORT[mode])} model</small>{status}</div></div>
    </div>
    <div class="foot"><div class="brand"><div class="logo" style="width:28px;height:28px;font-size:1rem">ب</div>
      <div><b>Persian Character Recognition</b><br><small>Academic research prototype.</small></div></div>
      <div>Designed for a responsible, transparent AI demonstration.</div></div>
    """,
    unsafe_allow_html=True,
)
