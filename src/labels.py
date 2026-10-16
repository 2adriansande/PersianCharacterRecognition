"""32-class Persian alphabet label map (single source of truth).

NOTE: Persian kaf (U+06A9) and yeh (U+06CC) are used, NOT the Arabic
kaf (U+0643) / yeh (U+064A). They look similar but are different code points.
"""
import unicodedata

# (character, latin_name); list position = class index
_LETTERS = [
    ("\u0627", "alef"), ("\u0628", "be"), ("\u067E", "pe"), ("\u062A", "te"),
    ("\u062B", "se"), ("\u062C", "jim"), ("\u0686", "che"), ("\u062D", "he_jimi"),
    ("\u062E", "khe"), ("\u062F", "dal"), ("\u0630", "zal"), ("\u0631", "re"),
    ("\u0632", "ze"), ("\u0698", "zhe"), ("\u0633", "sin"), ("\u0634", "shin"),
    ("\u0635", "sad"), ("\u0636", "zad"), ("\u0637", "ta"), ("\u0638", "za"),
    ("\u0639", "ayn"), ("\u063A", "ghayn"), ("\u0641", "fe"), ("\u0642", "qaf"),
    ("\u06A9", "kaf"), ("\u06AF", "gaf"), ("\u0644", "lam"), ("\u0645", "mim"),
    ("\u0646", "nun"), ("\u0648", "vav"), ("\u0647", "he"), ("\u06CC", "ye"),
]

NUM_CLASSES = len(_LETTERS)
LABELS = [{"index": i, "char": c, "name": n, "codepoint": f"U+{ord(c):04X}"}
          for i, (c, n) in enumerate(_LETTERS)]
IDX_TO_CHAR = {d["index"]: d["char"] for d in LABELS}
IDX_TO_NAME = {d["index"]: d["name"] for d in LABELS}
CHAR_TO_IDX = {d["char"]: d["index"] for d in LABELS}
NAME_TO_IDX = {d["name"]: d["index"] for d in LABELS}


def verify_labels(verbose=False):
    """Raise AssertionError if the map is malformed. Returns the label list."""
    assert NUM_CLASSES == 32, f"expected 32 classes, got {NUM_CLASSES}"
    assert len(set(IDX_TO_CHAR.values())) == 32, "duplicate characters"
    assert len(set(IDX_TO_NAME.values())) == 32, "duplicate names"
    for d in LABELS:
        c = d["char"]
        assert len(c) == 1, f"{d['name']}: not a single code point"
        assert "ARABIC LETTER" in unicodedata.name(c), f"{d['name']}: not an Arabic-script letter"
        assert c not in ("\u0643", "\u064A"), f"{d['name']}: Arabic kaf/yeh used instead of Persian"
    if verbose:
        for d in LABELS:
            print(f"{d['index']:2d}  {d['char']}  {d['codepoint']}  {d['name']:8s} {unicodedata.name(d['char'])}")
    return LABELS


def write_label_csv(path):
    import pandas as pd
    pd.DataFrame(LABELS).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    verify_labels(verbose=True)
    print("OK: 32 labels verified")
