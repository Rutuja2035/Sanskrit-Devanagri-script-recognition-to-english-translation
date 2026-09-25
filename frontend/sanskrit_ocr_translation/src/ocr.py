from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Load backend PaddleSanskritOCR cleanly without namespace collision
_backend_ocr_path = Path(__file__).resolve().parents[3] / "Text-Translation" / "sanskrit_ocr_translation" / "src" / "ocr.py"
_backend_dir = _backend_ocr_path.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_spec = importlib.util.spec_from_file_location("backend_ocr", str(_backend_ocr_path))
_backend_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_mod)

for attr in dir(_backend_mod):
    if not attr.startswith("__"):
        globals()[attr] = getattr(_backend_mod, attr)

PaddleSanskritOCR = _backend_mod.PaddleSanskritOCR

_OCR_INSTANCE = None

def get_ocr_engine(device: str = "cpu") -> PaddleSanskritOCR:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = PaddleSanskritOCR(device=device)
    return _OCR_INSTANCE

def perform_ocr(image: Image.Image, high_accuracy_mode: bool = True, enable_corpus_alignment: bool = True) -> str:
    """Extract Devanagari text using high-performance PaddleSanskritOCR."""
    if image is None:
        return ""
    img_np = np.array(image)
    engine = get_ocr_engine()
    result = engine.recognize(
        img_np,
        high_accuracy_mode=high_accuracy_mode,
        enable_corpus_alignment=enable_corpus_alignment
    )
    return result.get("text", "")
