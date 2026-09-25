from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load backend postprocess cleanly without namespace collision
_backend_post_path = Path(__file__).resolve().parents[3] / "Text-Translation" / "sanskrit_ocr_translation" / "src" / "postprocess.py"
_backend_dir = _backend_post_path.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_spec = importlib.util.spec_from_file_location("backend_post", str(_backend_post_path))
_backend_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_mod)

# Re-export all symbols from backend postprocess so downstream imports (like SanskritLexicalCorrector) resolve cleanly
for attr in dir(_backend_mod):
    if not attr.startswith("__"):
        globals()[attr] = getattr(_backend_mod, attr)

def clean_ocr_text(text: str) -> str:
    """Cleans extracted OCR text with Unicode NFC normalization, ligature repairs, and danda spacing."""
    if not text:
        return ""
    return _backend_mod.postprocess_text(text, is_manuscript=True)
