from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Load backend preprocessing cleanly without namespace collision
_backend_prep_path = Path(__file__).resolve().parents[3] / "Text-Translation" / "sanskrit_ocr_translation" / "src" / "preprocessing.py"
_backend_dir = _backend_prep_path.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_spec = importlib.util.spec_from_file_location("backend_prep", str(_backend_prep_path))
_backend_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_mod)

# Re-export all symbols from backend preprocessing
for attr in dir(_backend_mod):
    if not attr.startswith("__"):
        globals()[attr] = getattr(_backend_mod, attr)

def clean_image(image: Image.Image, is_manuscript: bool = True) -> Image.Image:
    """Clean and enhance Sanskrit image with Sauvola binarization, deskewing, and contrast curves."""
    if image is None:
        return image
    img_np = np.array(image)
    if is_manuscript:
        cleaned_np = _backend_mod.preprocess_manuscript(img_np, auto_scale=True, deskew=True)
    else:
        cleaned_np = _backend_mod.preprocess_adaptive(img_np)
    return Image.fromarray(cleaned_np)
