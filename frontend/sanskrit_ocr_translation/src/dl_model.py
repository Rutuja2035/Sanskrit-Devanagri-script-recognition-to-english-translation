from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load backend dl_model cleanly without namespace collision
_backend_dl_path = Path(__file__).resolve().parents[3] / "Text-Translation" / "sanskrit_ocr_translation" / "src" / "dl_model.py"
_backend_dir = _backend_dl_path.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_spec = importlib.util.spec_from_file_location("backend_dl", str(_backend_dl_path))
_backend_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_mod)

for attr in dir(_backend_mod):
    if not attr.startswith("__"):
        globals()[attr] = getattr(_backend_mod, attr)
