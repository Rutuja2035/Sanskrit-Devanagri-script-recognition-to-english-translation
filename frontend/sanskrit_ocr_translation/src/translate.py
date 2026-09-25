from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Load backend translator cleanly without namespace collision
_backend_trans_path = Path(__file__).resolve().parents[3] / "Text-Translation" / "sanskrit_ocr_translation" / "src" / "translate.py"
_backend_dir = _backend_trans_path.parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_spec = importlib.util.spec_from_file_location("backend_trans", str(_backend_trans_path))
_backend_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_mod)

for attr in dir(_backend_mod):
    if not attr.startswith("__"):
        globals()[attr] = getattr(_backend_mod, attr)

SanskritToEnglishTranslator = _backend_mod.SanskritToEnglishTranslator

_TRANSLATOR_INSTANCE = None

def get_translator() -> SanskritToEnglishTranslator:
    global _TRANSLATOR_INSTANCE
    if _TRANSLATOR_INSTANCE is None:
        _TRANSLATOR_INSTANCE = SanskritToEnglishTranslator()
    return _TRANSLATOR_INSTANCE

def translate_sanskrit_to_english(text: str, hf_token: str = None) -> str:
    """Translates Sanskrit Devanagari text to English via multi-tier fallback (Corpus -> Google -> MyMemory -> NLLB)."""
    if not text or not text.strip():
        return ""

    if hf_token:
        import requests
        api_url = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": text,
            "parameters": {"src_lang": "san_Deva", "tgt_lang": "eng_Latn"}
        }
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=8)
            if resp.status_code == 200:
                res = resp.json()
                if isinstance(res, list) and "translation_text" in res[0]:
                    return res[0]["translation_text"]
        except Exception:
            pass

    translator = get_translator()
    return translator.translate(text)
