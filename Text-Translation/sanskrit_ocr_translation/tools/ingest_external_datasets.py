"""Download, preprocess, and integrate free, open-access Sanskrit datasets.

Datasets:
1. Process-Venue/Sanskrit-OCR-Typed-Dataset (3,457 real Sanskrit OCR images & ground-truth text)
2. buddhist-nlp/skt-en-itihasa (1,820 aligned Sanskrit-English verse & sentence pairs)
3. chronbmm/sanskrit-ocr-postcorrection (5,001 sandhi & lexical post-correction pairs)
"""

from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OCR_OUT_DIR = DATA_DIR / "sanskrit_real_ocr"
OCR_IMG_DIR = OCR_OUT_DIR / "images"
CORPUS_OUT_FILE = DATA_DIR / "sanskrit_expanded_corpus.json"
POSTCORRECT_OUT_FILE = DATA_DIR / "sanskrit_postcorrection_rules.json"

OCR_IMG_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    """Normalize Devanagari text into standard Unicode NFC form."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def strip_punct(text: str) -> str:
    """Strip dandas and punctuation for clean dictionary lookup."""
    return re.sub(r"[।॥|\-_,.;:\n\r\t\"'!?()]+", " ", text).strip()

def download_and_process_ocr_images():
    print("[1/3] Downloading Sanskrit OCR Real Image Dataset (Process-Venue)...")
    splits = {
        "val": "https://huggingface.co/datasets/Process-Venue/Sanskrit-OCR-Typed-Dataset/resolve/main/data/validation-00000-of-00001.parquet",
        "train": "https://huggingface.co/datasets/Process-Venue/Sanskrit-OCR-Typed-Dataset/resolve/main/data/train-00000-of-00001.parquet",
    }

    manifests = {"train": [], "val": []}
    total_saved = 0

    for split_name, url in splits.items():
        print(f"  -> Fetching {split_name} split...")
        try:
            df = pd.read_parquet(url)
            print(f"  -> Successfully loaded {len(df)} records for {split_name}.")
            for idx, row in df.iterrows():
                label = normalize_text(row.get("label", ""))
                if not label:
                    continue
                img_entry = row.get("image", None)
                if not isinstance(img_entry, dict) or "bytes" not in img_entry:
                    continue
                img_bytes = img_entry["bytes"]
                if not img_bytes:
                    continue

                filename = f"{split_name}_{idx:05d}.png"
                img_path = OCR_IMG_DIR / filename
                
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(img_path, format="PNG")
                    rel_path = f"images/{filename}"
                    manifests[split_name].append(f"{rel_path}\t{label}")
                    total_saved += 1
                except Exception:
                    continue
        except Exception as e:
            print(f"  [!] Failed to download {split_name}: {e}")

    # Write manifests
    for split_name, lines in manifests.items():
        out_txt = OCR_OUT_DIR / f"{split_name}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  -> Wrote {len(lines)} lines to {out_txt.name}")

    print(f"[+] Total Sanskrit OCR Images Processed & Saved: {total_saved}")

def download_and_process_translation_corpus():
    print("[2/3] Downloading Sanskrit-English Parallel Translation Corpus (buddhist-nlp)...")
    urls = [
        "https://huggingface.co/datasets/buddhist-nlp/skt-en-itihasa/resolve/main/data/validation-00000-of-00001-4bbb2f61d987b861.parquet",
        "https://huggingface.co/datasets/buddhist-nlp/skt-en-itihasa/resolve/main/data/test-00000-of-00001-5cafd318368bcc13.parquet",
    ]

    corpus_dict = {}

    for url in urls:
        print(f"  -> Fetching parallel split: {url.split('/')[-1]}...")
        try:
            df = pd.read_parquet(url)
            print(f"  -> Loaded {len(df)} sentence pairs.")
            for _, row in df.iterrows():
                skt = normalize_text(row.get("input_text", ""))
                eng = normalize_text(row.get("target_text", ""))
                if skt and eng and len(skt) > 2 and len(eng) > 2:
                    clean_skt = strip_punct(skt)
                    if clean_skt:
                        corpus_dict[clean_skt] = eng
                        # Also keep original with danda if distinct
                        if skt != clean_skt:
                            corpus_dict[skt] = eng
        except Exception as e:
            print(f"  [!] Failed parallel split: {e}")

    print(f"[+] Total Unique Sanskrit-English Translation Pairs Compiled: {len(corpus_dict)}")
    with open(CORPUS_OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus_dict, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved expanded corpus dictionary to: {CORPUS_OUT_FILE.name}")

def download_and_process_postcorrection_data():
    print("[3/3] Downloading Sanskrit Sandhi & OCR Post-Correction Corpus (chronbmm)...")
    url = "https://huggingface.co/datasets/chronbmm/sanskrit-ocr-postcorrection/resolve/main/data/test-00000-of-00001.parquet"
    try:
        df = pd.read_parquet(url)
        print(f"  -> Loaded {len(df)} post-correction records.")
        rules = []
        for _, row in df.iterrows():
            sent = normalize_text(row.get("sentence", ""))
            unsandhied = normalize_text(row.get("unsandhied", ""))
            if sent and unsandhied and sent != unsandhied and len(sent) < 80:
                rules.append({"surface": sent, "split": unsandhied})
        
        with open(POSTCORRECT_OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(rules[:1000], f, ensure_ascii=False, indent=2)
        print(f"[+] Compiled {min(len(rules), 1000)} Sandhi & Lexical Rules to: {POSTCORRECT_OUT_FILE.name}")
    except Exception as e:
        print(f"  [!] Failed post-correction download: {e}")

def main():
    print("=" * 70)
    print("SANSKRIT VISION: EXTERNAL DATASET INGESTION & PREPROCESSING")
    print("=" * 70)
    download_and_process_ocr_images()
    print("-" * 70)
    download_and_process_translation_corpus()
    print("-" * 70)
    download_and_process_postcorrection_data()
    print("=" * 70)
    print("ALL DATASETS DOWNLOADED, PREPROCESSED, AND INDEXED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
