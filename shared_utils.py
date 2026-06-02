import hashlib
import re


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    text = str(x).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        return float(match.group(0)) if match else None


def normalize_id(s):
    if s is None:
        return ""
    text = str(s).strip()
    text = re.sub(r"[-_ ]pose\d+$", "", text, flags=re.IGNORECASE)
    return text.upper()


def hash_text(value):
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:12]