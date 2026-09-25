import sys
import io
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from src.ocr import PaddleSanskritOCR
from src.preprocessing import preprocess_manuscript, preprocess_otsu, to_grayscale

img_path = Path(__file__).resolve().parent.parent / "data" / "test_evaluation" / "manuscript_1.jpg"
img = cv2.imread(str(img_path))
print("Image shape:", img.shape if img is not None else None)

ocr = PaddleSanskritOCR(device="cpu")

print("\n--- Test 1: Raw Image ---")
res1 = ocr.recognize(img)
print("Overall Confidence:", f"{res1['confidence']:.2%}")
print("Segments count:", len(res1["segments"]))
for i, s in enumerate(res1["segments"][:4]):
    print(f"  Line {i} ({s['conf']:.2%}): {s['text']}")

print("\n--- Test 2: Preprocessed with preprocess_manuscript ---")
prep_ms = preprocess_manuscript(img)
res2 = ocr.recognize(prep_ms)
print("Overall Confidence:", f"{res2['confidence']:.2%}")
print("Segments count:", len(res2["segments"]))
for i, s in enumerate(res2["segments"][:4]):
    print(f"  Line {i} ({s['conf']:.2%}): {s['text']}")

print("\n--- Test 3: Outer Border Removal (Cropping outer grid) ---")
# Crop outer margin (20 px left/right, 10 px top/bottom)
cropped = img[10:-10, 20:-20]
res3 = ocr.recognize(cropped)
print("Overall Confidence:", f"{res3['confidence']:.2%}")
print("Segments count:", len(res3["segments"]))
for i, s in enumerate(res3["segments"][:4]):
    print(f"  Line {i} ({s['conf']:.2%}): {s['text']}")

print("\n--- Test 6: UPGRADED END-TO-END PIPELINE ---")
from src.postprocess import postprocess_text

prep_upgraded = preprocess_manuscript(img, suppress_borders=True, auto_scale=False)
res6 = ocr.recognize(prep_upgraded, suppress_borders=True)
cleaned_ms = postprocess_text(res6["text"], is_manuscript=True)
print("Upgraded Confidence:", f"{res6['confidence']:.2%}")
print("Segments count:", len(res6["segments"]))
print("\n--- First 5 Cleaned Manuscript Lines ---")
for i, line in enumerate(cleaned_ms.split("\n")[:5]):
    print(f"  Line {i}: {line}")
