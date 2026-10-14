"""Ingest and preprocess new external Sanskrit datasets NOT present in the catalog PDF.

External Datasets Ingested:
1. Valmiki Ramayana Sanskrit-English Epic Parallel Corpus (`mangeshdiyewar/sanskrit_eng`)
   - 75,162 authentic Devanagari shlokas with complete English translation.
   - Preprocessing: Unicode NFC normalization, footnote removal, Sanskrit phonotactic
     validation, dual-indexing (danda-preserved & stripped), and translation dictionary integration.
2. Real Scanned Devanagari Document OCR Dataset (`vishwam-101/devanagari-ocr-dataset`)
   - Real scanned Devanagari document and manuscript pages.
   - Preprocessing: Optical auto-deskewing, adaptive Sauvola binarization, noise despeckling,
     horizontal projection line segmentation, and 48px height receptive-field normalization.
3. Classical Devanagari Sandhi & Morphological Compound Rules (`chronbmm/sanskrit-multitask-devanagari`)
   - Devanagari compound words and morphological splits.
   - Preprocessing: Unicode normalization, punctuation cleaning, and sandhi rule indexing.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Destination directories
RAMAYANA_DIR = DATA_DIR / "sanskrit_ramayana_parallel"
OCR_EXPANDED_DIR = DATA_DIR / "sanskrit_real_ocr_expanded"
OCR_EXPANDED_IMG_DIR = OCR_EXPANDED_DIR / "images"
CORPUS_FILE = DATA_DIR / "sanskrit_expanded_corpus.json"
POSTCORRECT_FILE = DATA_DIR / "sanskrit_postcorrection_rules.json"

RAMAYANA_DIR.mkdir(parents=True, exist_ok=True)
OCR_EXPANDED_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to sys.path for internal imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    auto_deskew_image,
    despeckle_image,
    normalize_line_crop,
    preprocess_sauvola,
    to_grayscale,
)
from src.postprocess import SanskritLexicalCorrector


def normalize_devanagari(text: str) -> str:
    """Canonical Unicode NFC normalization and whitespace collapsing."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    # Replace non-standard spaces, tabs, and newlines
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" +", " ", text)
    # Remove Western trailing dots after Devanagari dandas
    text = re.sub(r"([।॥|])\s*\.+", r"\1", text)
    return text.strip()


def strip_dandas_and_punct(text: str) -> str:
    """Strip dandas and punctuation for clean fuzzy dictionary lookup."""
    return re.sub(r"[।॥|\-_,.;:\n\r\t\"'!?()<>]+", " ", text).strip()


def clean_english_translation(text: str) -> str:
    """Clean English text by stripping footnotes and annotation markers."""
    if not text:
        return ""
    # Strip footnotes like *A great saint...
    text = re.sub(r"\*[^*]+(?:\*|$)", "", text)
    text = re.sub(r"### Human:\s*", "", text)
    text = re.sub(r"### Assistant:\s*", "", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def ingest_and_preprocess_ramayana_parallel(max_records: int = 75000) -> int:
    """
    Ingest, clean, validate, and index the Valmiki Ramayana Sanskrit-English Corpus.
    """
    print("=" * 70)
    print("[1/3] INGESTING VALMIKI RAMAYANA SANSKRIT-ENGLISH PARALLEL CORPUS")
    print("=" * 70)
    json_path = RAMAYANA_DIR / "ramayana_parallel.json"
    tsv_path = RAMAYANA_DIR / "ramayana_parallel.tsv"

    if json_path.exists() and json_path.stat().st_size > 1000000:
        print(f"[+] Found cached preprocessed Ramayana dataset: {json_path.name}")
        with open(json_path, "r", encoding="utf-8") as f:
            clean_pairs = json.load(f)
        print(f"[+] Loaded {len(clean_pairs)} preprocessed Ramayana verse pairs from local cache.")
        return len(clean_pairs)

    url = "https://huggingface.co/datasets/mangeshdiyewar/sanskrit_eng/resolve/main/data/train-00000-of-00001-fc32f390115f2384.parquet"
    
    print(f"[*] Downloading dataset parquet from Hugging Face...")
    t0 = time.time()
    try:
        df = pd.read_parquet(url)
        print(f"[+] Download complete: {len(df)} records in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[!] Failed to download Ramayana dataset: {e}")
        return 0

    clean_pairs: List[Dict[str, str]] = []
    lookup_dict: Dict[str, str] = {}
    valid_count = 0
    skipped_count = 0

    print("[*] Preprocessing Sanskrit shlokas and English translations...")
    for idx, row in df.iterrows():
        if idx >= max_records:
            break
        raw_text = str(row.get("text", ""))
        parts = raw_text.split("### Assistant:")
        if len(parts) != 2:
            skipped_count += 1
            continue

        raw_eng = parts[0]
        raw_skt = parts[1]

        eng = clean_english_translation(raw_eng)
        skt = normalize_devanagari(raw_skt)

        if not skt or not eng:
            skipped_count += 1
            continue

        # Linguistic validity check: must contain Devanagari and valid Sanskrit phonotactics
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", skt))
        if devanagari_chars < 5:
            skipped_count += 1
            continue

        validity = SanskritLexicalCorrector.calculate_sanskrit_validity(skt)
        if validity < 0.25:
            skipped_count += 1
            continue

        clean_pairs.append({
            "id": idx,
            "sanskrit": skt,
            "english": eng,
        })

        # Dual-indexing: with dandas and without dandas
        lookup_dict[skt] = eng
        stripped = strip_dandas_and_punct(skt)
        if stripped and stripped != skt:
            lookup_dict[stripped] = eng

        valid_count += 1

    print(f"[+] Validated and normalized {valid_count} Ramayana shloka pairs (skipped {skipped_count}).")

    # Save dedicated dataset files
    json_path = RAMAYANA_DIR / "ramayana_parallel.json"
    tsv_path = RAMAYANA_DIR / "ramayana_parallel.tsv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_pairs, f, ensure_ascii=False, indent=2)

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("id\tsanskrit\tenglish\n")
        for item in clean_pairs:
            s_clean = item["sanskrit"].replace("\t", " ")
            e_clean = item["english"].replace("\t", " ")
            f.write(f"{item['id']}\t{s_clean}\t{e_clean}\n")

    print(f"  -> Saved structured dataset to: {json_path.name} ({json_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  -> Saved TSV aligned corpus to: {tsv_path.name}")

    # Merge into system sanskrit_expanded_corpus.json
    print("[*] Merging Ramayana verses into system translation corpus...")
    existing_corpus = {}
    if CORPUS_FILE.exists():
        try:
            with open(CORPUS_FILE, "r", encoding="utf-8") as f:
                existing_corpus = json.load(f)
        except Exception:
            existing_corpus = {}

    initial_size = len(existing_corpus)
    existing_corpus.update(lookup_dict)
    new_size = len(existing_corpus)

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_corpus, f, ensure_ascii=False, indent=2)

    print(f"[+] Expanded translation corpus from {initial_size} to {new_size} entries! (+{new_size - initial_size} entries)")
    return valid_count


def ingest_and_preprocess_devanagari_ocr_scans(max_pages: int = 50, max_lines_per_page: int = 15) -> int:
    """
    Ingest, deskew, binarize, slice, and normalize real Devanagari document scans.
    """
    print("\n" + "=" * 70)
    print("[2/3] INGESTING & PREPROCESSING REAL DEVANAGARI OCR SCANNED DOCUMENTS")
    print("=" * 70)
    url = "https://huggingface.co/datasets/vishwam-101/devanagari-ocr-dataset/resolve/main/data/val-00000-of-00001.parquet"
    
    print(f"[*] Downloading document scan parquet ({url.split('/')[-1]})...")
    t0 = time.time()
    try:
        df = pd.read_parquet(url)
        print(f"[+] Download complete: {len(df)} pages in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[!] Failed to download OCR dataset: {e}")
        return 0

    processed_lines: List[Tuple[str, str]] = []
    saved_images = 0

    print(f"[*] Preprocessing scanned document pages with auto-deskew & Sauvola binarization...")
    for p_idx, row in df.iterrows():
        if p_idx >= max_pages:
            break

        img_entry = row.get("image", None)
        raw_text = str(row.get("text", ""))

        if not isinstance(img_entry, dict) or "bytes" not in img_entry:
            continue

        img_bytes = img_entry["bytes"]
        if not img_bytes:
            continue

        # Load page image
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            page_np = np.array(pil_img)
        except Exception:
            continue

        # Step 1: Optical auto-deskewing
        try:
            deskewed = auto_deskew_image(page_np, max_angle=15.0)
        except Exception:
            deskewed = page_np

        # Step 2: Grayscale and Sauvola adaptive binarization with integrated despeckle
        gray = to_grayscale(deskewed)
        cleaned = preprocess_sauvola(gray, window_size=25, k=0.15, despeckle=True)

        # Parse ground-truth lines
        gt_lines = [normalize_devanagari(l) for l in raw_text.split("\n")]
        gt_lines = [l for l in gt_lines if len(re.findall(r"[\u0900-\u097F]", l)) >= 3]

        if not gt_lines:
            continue

        # Step 3: Horizontal projection line segmentation
        # Text pixels are black (0), background is white (255)
        text_pixels = (cleaned == 0).astype(np.uint8)
        proj = np.sum(text_pixels, axis=1)

        # Identify line bands
        threshold = max(5, int(np.mean(proj) * 0.15))
        in_line = False
        start_y = 0
        line_bands: List[Tuple[int, int]] = []

        for y, val in enumerate(proj):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                if y - start_y >= 12:  # Min line height
                    line_bands.append((start_y, y))

        # Match detected line crops with ground truth lines
        matched_count = min(len(line_bands), len(gt_lines), max_lines_per_page)
        for l_idx in range(matched_count):
            y1, y2 = line_bands[l_idx]
            # Add small vertical padding
            pad = 2
            y1_pad = max(0, y1 - pad)
            y2_pad = min(deskewed.shape[0], y2 + pad)

            crop = deskewed[y1_pad:y2_pad, :]

            # Horizontal trimming: find left and right bounds
            crop_text = text_pixels[y1_pad:y2_pad, :]
            h_proj = np.sum(crop_text, axis=0)
            nonzero_x = np.where(h_proj > 2)[0]
            if len(nonzero_x) > 10:
                x1 = max(0, nonzero_x[0] - 4)
                x2 = min(deskewed.shape[1], nonzero_x[-1] + 4)
                if x2 - x1 > 20:
                    crop = crop[:, x1:x2]

            # Step 4: Line Receptive-Field Normalization (48px standard height)
            norm_crop = normalize_line_crop(crop, target_h=48)

            img_filename = f"ext_ocr_{saved_images:05d}.png"
            img_save_path = OCR_EXPANDED_IMG_DIR / img_filename
            cv2.imwrite(str(img_save_path), cv2.cvtColor(norm_crop, cv2.COLOR_RGB2BGR))

            rel_path = f"images/{img_filename}"
            label = gt_lines[l_idx]
            processed_lines.append((rel_path, label))
            saved_images += 1

    print(f"[+] Sliced, deskewed, and normalized {saved_images} Devanagari OCR text lines.")

    # Split into train (85%) and val (15%)
    split_idx = int(len(processed_lines) * 0.85)
    train_lines = processed_lines[:split_idx]
    val_lines = processed_lines[split_idx:]

    with open(OCR_EXPANDED_DIR / "train.txt", "w", encoding="utf-8") as f:
        for p, l in train_lines:
            f.write(f"{p}\t{l}\n")

    with open(OCR_EXPANDED_DIR / "val.txt", "w", encoding="utf-8") as f:
        for p, l in val_lines:
            f.write(f"{p}\t{l}\n")

    print(f"  -> Wrote {len(train_lines)} lines to {OCR_EXPANDED_DIR.name}/train.txt")
    print(f"  -> Wrote {len(val_lines)} lines to {OCR_EXPANDED_DIR.name}/val.txt")
    return saved_images


def ingest_and_preprocess_sandhi_rules(max_rules: int = 5000) -> int:
    """
    Ingest and compile Devanagari Sandhi and morphological split rules.
    """
    print("\n" + "=" * 70)
    print("[3/3] INGESTING DEVANAGARI SANDHI & MORPHOLOGICAL SPLIT RULES")
    print("=" * 70)
    url = "https://huggingface.co/datasets/chronbmm/sanskrit-multitask-devanagari/resolve/main/data/test-00000-of-00001.parquet"

    print(f"[*] Downloading sandhi multitask dataset...")
    t0 = time.time()
    try:
        df = pd.read_parquet(url)
        print(f"[+] Download complete: {len(df)} records in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"[!] Failed to download sandhi dataset: {e}")
        return 0

    new_rules: List[Dict[str, str]] = []
    seen = set()

    # Load existing rules
    existing_rules = []
    if POSTCORRECT_FILE.exists():
        try:
            with open(POSTCORRECT_FILE, "r", encoding="utf-8") as f:
                existing_rules = json.load(f)
                for r in existing_rules:
                    seen.add(r.get("surface", ""))
        except Exception:
            existing_rules = []

    print("[*] Normalizing and filtering Devanagari sandhi pairs...")
    for _, row in df.iterrows():
        if len(new_rules) >= max_rules:
            break
        sent = normalize_devanagari(row.get("sentence", ""))
        # Clean prefix markers like 'ल् '
        sent = re.sub(r"^ल्\s+", "", sent).strip()
        unsandhied = normalize_devanagari(row.get("unsandhied", ""))

        if not sent or not unsandhied or sent == unsandhied:
            continue
        if len(sent) > 60:
            continue
        if sent in seen:
            continue

        clean_unsandhied = unsandhied.replace("_", " ").strip()
        new_rules.append({
            "surface": sent,
            "split": clean_unsandhied,
        })
        seen.add(sent)

    combined_rules = existing_rules + new_rules
    with open(POSTCORRECT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_rules, f, ensure_ascii=False, indent=2)

    print(f"[+] Compiled {len(new_rules)} new Devanagari Sandhi rules.")
    print(f"  -> Total active rules in {POSTCORRECT_FILE.name}: {len(combined_rules)}")
    return len(new_rules)


def main():
    print("=" * 75)
    print("SANSKRIT VISION: NEW EXTERNAL DATASET INGESTION & PREPROCESSING")
    print("Catalog Exclusions: Muktabodha, PLMD, Cambridge, Bodleian, NMM, Epigraphia,")
    print("                   IndicOCR, CVIT, TDIL, suvooo, IIT Roorkee, DCS, GRETIL,")
    print("                   VedaWeb, INRIA, SanskritDoc, UoH, TITUS, PMIndia, CSL,")
    print("                   OPUS, Tatoeba, 108 Upanishads, Panchatantra, IndicTrans2,")
    print("                   Kalidasa works.")
    print("=" * 75)

    # 1. Ingest & Preprocess Valmiki Ramayana Parallel Corpus
    ramayana_count = ingest_and_preprocess_ramayana_parallel()

    # 2. Ingest & Preprocess Real Scanned Devanagari OCR Documents
    ocr_count = ingest_and_preprocess_devanagari_ocr_scans(max_pages=60, max_lines_per_page=12)

    # 3. Ingest & Compile Devanagari Sandhi Rules
    sandhi_count = ingest_and_preprocess_sandhi_rules(max_rules=3000)

    print("\n" + "=" * 75)
    print("INGESTION & PREPROCESSING SUMMARY")
    print("=" * 75)
    print(f"1. Valmiki Ramayana Parallel Verses Processed:     {ramayana_count:,}")
    print(f"2. Scanned Devanagari OCR Lines Normalized (48px): {ocr_count:,}")
    print(f"3. Devanagari Sandhi & Morphological Rules Added:  {sandhi_count:,}")
    print(f"4. Total Expanded Translation Corpus Records:       {len(json.load(open(CORPUS_FILE, 'r', encoding='utf-8'))):,}")
    print("=" * 75)


if __name__ == "__main__":
    main()
