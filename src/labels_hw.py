"""Folder / file-name conventions of the HANDWRITTEN dataset.

The handwritten dataset has one folder per letter: class_01 ... class_32, and files named
    class_NN_fXX_sYYYY.png      (NN = class, XX = form number 01..10, YYYY = running sample number)

class_NN is the letter with index NN-1 in src/labels.py (class_01 = alef ... class_32 = ye).
This was checked by eye on a contact sheet (8 samples per folder) before it was written here.
Only the folder number is mapped; the letter list itself lives in labels.py (single source of truth),
so the printed and the handwritten model can never disagree about class order.
"""
import re

from labels import IDX_TO_CHAR, IDX_TO_NAME, NUM_CLASSES

FOLDER_PAT = re.compile(r"^class_(\d{1,2})$")
FILE_PAT = re.compile(r"^class_(\d{1,2})_f(\d+)_s(\d+)$")      # applied to Path.stem


def class_from_folder(folder_name):
    """'class_08' -> (7, 'he_jimi'). Raises ValueError for anything else."""
    m = FOLDER_PAT.match(folder_name)
    if not m:
        raise ValueError(f"Unexpected class folder name: '{folder_name}' (expected class_01 ... class_{NUM_CLASSES})")
    idx = int(m.group(1)) - 1
    if not 0 <= idx < NUM_CLASSES:
        raise ValueError(f"Folder '{folder_name}' is outside class_01 ... class_{NUM_CLASSES}")
    return idx, IDX_TO_NAME[idx]


def parse_filename(stem):
    """'class_02_f03_s0150' -> (class_idx, form, sample_no) or None if the name does not match."""
    m = FILE_PAT.match(stem)
    if not m:
        return None
    return int(m.group(1)) - 1, int(m.group(2)), int(m.group(3))
