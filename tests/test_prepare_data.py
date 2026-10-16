import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, pytest
from PIL import Image, ImageDraw, ImageFont
import labels, prepare_data as pdm

CANDIDATES = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\tahoma.ttf",
              r"C:\Windows\Fonts\segoeui.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
FONT = next((p for p in CANDIDATES if Path(p).exists()), None)
FOLDERS = [f"{i+1:02d}_{n}" for i, n in enumerate(
    ["alef","be","pe","te","se","jim","che","he_j","khe","dal","zal","re","ze","zhe","sin","shin","sad","zad",
     "ta","za","ain","ghain","fe","qaf","kaf","gaf","lam","mim","nun","vav","he","ye"])]


def glyph(ch, size=128, pt=70, shift=0):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    ImageDraw.Draw(img).text((20 + shift, 10), ch, font=ImageFont.truetype(FONT, pt), fill=(0, 0, 0, 255))
    return img


@pytest.fixture
def fake(tmp_path):
    root = tmp_path / "final_merged_dataset"
    for i, f in enumerate(FOLDERS):
        d = root / f; d.mkdir(parents=True)
        ch = labels.IDX_TO_CHAR[i]
        for k in range(10):
            glyph(ch, pt=40 + 6 * k).save(d / f"ds101_{i+1}_{k:05d}.png")
        glyph(ch, pt=40).save(d / "ds101_1_09990.png")             # exact copy of k=0
        glyph(ch, pt=40, shift=7).save(d / "ds101_1_09991.png")    # shifted copy
    a = root / "01_alef"
    Image.new("RGBA", (128, 128), (255, 255, 255, 255)).save(a / "ds101_1_09999.png")   # blank
    img = Image.new("RGB", (204, 197), (255, 255, 255))
    ImageDraw.Draw(img).text((5, 60), "\u0627\u0644\u0641", font=ImageFont.truetype(FONT, 60), fill=0)
    img.save(a / "alpha_1_00001.jpg")                                                  # word 'alef'
    glyph(labels.IDX_TO_CHAR[1], pt=40).save(root / "03_pe" / "ds101_9_00001.png")      # be image under pe
    junk = root / "__MACOSX" / "01_alef"; junk.mkdir(parents=True)
    (junk / "._ds101_1_00000.png").write_bytes(b"junk")
    return root


def test_class_from_folder():
    assert pdm.class_from_folder("08_he_j") == (7, "he_jimi")
    assert pdm.class_from_folder("22_ghain") == (21, "ghayn")
    with pytest.raises(ValueError):
        pdm.class_from_folder("02_alef")


@pytest.mark.skipif(FONT is None, reason="no Arabic-capable font found")
def test_pipeline(fake):
    df, dropped = pdm.scan(fake)
    assert not any("MACOSX" in p for p in df["image_path"])
    assert any(d["reason"] == "blank" for d in dropped)
    df, dropped = pdm.clean(df, dropped)
    reasons = pd.Series([d["reason"] for d in dropped]).value_counts().to_dict()
    assert reasons.get("alef_word_image") == 1
    assert reasons.get("label_conflict", 0) >= 2
    assert reasons.get("duplicate", 0) >= 32
    df = pdm.split(df)
    assert pdm.check_manifest(df)
    assert set(df["class_idx"]) == set(range(32))