import hashlib
import math
import re
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (float, int)):
        value = float(x)
        return value if math.isfinite(value) else None
    text = str(x).strip()
    if text == "":
        return None
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except Exception:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        value = float(match.group(0))
        return value if math.isfinite(value) else None


def normalize_id(s):
    if s is None:
        return ""
    text = str(s).strip()
    text = re.sub(r"[-_ ]pose\d+$", "", text, flags=re.IGNORECASE)
    return text.upper()


def hash_text(value):
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=128)
def _cached_smarts_pattern(smarts_text):
    text = str(smarts_text or "").strip()
    if not text:
        return None
    try:
        return Chem.MolFromSmarts(text)
    except Exception:
        return None


def validate_torsion_smarts(smarts_text):
    text = str(smarts_text or "").strip()
    if not text:
        return None
    pattern = _cached_smarts_pattern(text)
    if pattern is None:
        return "invalid SMARTS"
    if pattern.GetNumAtoms() != 4:
        return f"SMARTS must define exactly 4 atoms for a torsion, got {pattern.GetNumAtoms()}"
    return None


def calculate_torsion_from_smarts(mol, smarts_text):
    text = str(smarts_text or "").strip()
    if mol is None or not text:
        return None
    pattern = _cached_smarts_pattern(text)
    if pattern is None or pattern.GetNumAtoms() != 4:
        return None
    try:
        matches = mol.GetSubstructMatches(pattern)
    except Exception:
        return None
    if not matches:
        return None
    match = matches[0]
    if len(match) != 4:
        return None
    try:
        conf = mol.GetConformer()
    except Exception:
        return None
    try:
        angle = float(rdMolTransforms.GetDihedralDeg(conf, *[int(idx) for idx in match]))
    except Exception:
        return None
    if not math.isfinite(angle):
        return None
    return angle