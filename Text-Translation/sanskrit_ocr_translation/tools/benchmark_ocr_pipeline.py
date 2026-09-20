"""Systematic Sanskrit Devanagari OCR Benchmark Framework.

Evaluates OCR performance across:
  A. Original Image
  B. Grayscale Image
  C. Baseline Preprocessing Pipeline
  D. Improved Adaptive Two-Pass OCR

Computes industry-standard metrics:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Character Accuracy (1 - CER)
  - Average & Minimum Segment Confidence
  - Percentage Improvement over Baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation import compute_cer, compute_wer
from src.ocr import PaddleSanskritOCR
from src.preprocessing import (
    preprocess_basic,
    preprocess_handwritten,
    preprocess_manuscript,
)


def run_benchmark(
    eval_dir: Path,
    rec_model_dir: str = "models/sanskrit_finetuned",
    char_dict_path: str = "data/sanskrit_dict.txt",
) -> Dict[str, Any]:
    print("=" * 80)
    print("SANSKRIT DEVANAGARI OCR SYSTEMATIC BENCHMARK")
    print("=" * 80)
    print(f"[*] Evaluation Directory: {eval_dir}")

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    test_pairs: List[Tuple[Path, str]] = []

    for img_path in sorted(eval_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                gt_text = txt_path.read_text(encoding="utf-8").strip()
                test_pairs.append((img_path, gt_text))

    if not test_pairs:
        print(f"[!] No valid test pairs found in {eval_dir}.")
        return {}

    print(f"[*] Found {len(test_pairs)} test image-reference pairs.")
    print("-" * 80)

    rec_path = PROJECT_DIR / rec_model_dir
    dict_path = PROJECT_DIR / char_dict_path
    ocr = PaddleSanskritOCR(
        rec_model_dir=str(rec_path) if rec_path.exists() else None,
        char_dict_path=str(dict_path) if dict_path.exists() else None,
    )

    strategies = [
        "A_Original",
        "B_Grayscale",
        "C_Baseline_Preprocessed",
        "D_Improved_Adaptive",
    ]

    results: Dict[str, Dict[str, List[float]]] = {
        s: {"cer": [], "wer": [], "acc": [], "conf": [], "min_conf": [], "segs": []}
        for s in strategies
    }

    sample_logs = []

    for idx, (img_path, gt_text) in enumerate(test_pairs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        sample_name = img_path.name
        print(f"[{idx}/{len(test_pairs)}] Evaluating: {sample_name}")

        # Strategy A: Original Image
        res_a = ocr.recognize(img, high_accuracy_mode=False)
        text_a = res_a.get("text", "").replace("\n", " ").strip()
        cer_a = compute_cer(gt_text, text_a)
        wer_a = compute_wer(gt_text, text_a)
        acc_a = max(0.0, 1.0 - cer_a)
        conf_a = res_a.get("confidence", 0.0)
        min_conf_a = res_a.get("min_confidence", conf_a)
        segs_a = len(res_a.get("segments", []))
        results["A_Original"]["cer"].append(cer_a)
        results["A_Original"]["wer"].append(wer_a)
        results["A_Original"]["acc"].append(acc_a)
        results["A_Original"]["conf"].append(conf_a)
        results["A_Original"]["min_conf"].append(min_conf_a)
        results["A_Original"]["segs"].append(segs_a)

        # Strategy B: Grayscale Image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        res_b = ocr.recognize(gray_img, high_accuracy_mode=False)
        text_b = res_b.get("text", "").replace("\n", " ").strip()
        cer_b = compute_cer(gt_text, text_b)
        wer_b = compute_wer(gt_text, text_b)
        acc_b = max(0.0, 1.0 - cer_b)
        conf_b = res_b.get("confidence", 0.0)
        min_conf_b = res_b.get("min_confidence", conf_b)
        segs_b = len(res_b.get("segments", []))
        results["B_Grayscale"]["cer"].append(cer_b)
        results["B_Grayscale"]["wer"].append(wer_b)
        results["B_Grayscale"]["acc"].append(acc_b)
        results["B_Grayscale"]["conf"].append(conf_b)
        results["B_Grayscale"]["min_conf"].append(min_conf_b)
        results["B_Grayscale"]["segs"].append(segs_b)

        # Strategy C: Baseline Preprocessed
        if "manuscript" in sample_name.lower():
            prep_c = preprocess_manuscript(img, auto_scale=True)
        elif "handwritten" in sample_name.lower():
            prep_c = preprocess_handwritten(img, auto_scale=True)
        else:
            prep_c = preprocess_basic(img)
        res_c = ocr.recognize(prep_c, high_accuracy_mode=False)
        text_c = res_c.get("text", "").replace("\n", " ").strip()
        cer_c = compute_cer(gt_text, text_c)
        wer_c = compute_wer(gt_text, text_c)
        acc_c = max(0.0, 1.0 - cer_c)
        conf_c = res_c.get("confidence", 0.0)
        min_conf_c = res_c.get("min_confidence", conf_c)
        segs_c = len(res_c.get("segments", []))
        results["C_Baseline_Preprocessed"]["cer"].append(cer_c)
        results["C_Baseline_Preprocessed"]["wer"].append(wer_c)
        results["C_Baseline_Preprocessed"]["acc"].append(acc_c)
        results["C_Baseline_Preprocessed"]["conf"].append(conf_c)
        results["C_Baseline_Preprocessed"]["min_conf"].append(min_conf_c)
        results["C_Baseline_Preprocessed"]["segs"].append(segs_c)

        # Strategy D: Improved Adaptive Two-Pass OCR
        res_d = ocr.recognize(img, high_accuracy_mode=True)
        text_d = res_d.get("text", "").replace("\n", " ").strip()
        cer_d = compute_cer(gt_text, text_d)
        wer_d = compute_wer(gt_text, text_d)
        acc_d = max(0.0, 1.0 - cer_d)
        conf_d = res_d.get("confidence", 0.0)
        min_conf_d = res_d.get("min_confidence", conf_d)
        segs_d = len(res_d.get("segments", []))
        results["D_Improved_Adaptive"]["cer"].append(cer_d)
        results["D_Improved_Adaptive"]["wer"].append(wer_d)
        results["D_Improved_Adaptive"]["acc"].append(acc_d)
        results["D_Improved_Adaptive"]["conf"].append(conf_d)
        results["D_Improved_Adaptive"]["min_conf"].append(min_conf_d)
        results["D_Improved_Adaptive"]["segs"].append(segs_d)

        sample_logs.append({
            "name": sample_name,
            "gt": gt_text,
            "pred_baseline": text_c,
            "acc_baseline": acc_c,
            "pred_improved": text_d,
            "acc_improved": acc_d,
            "conf_improved": conf_d,
        })

    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Strategy / Pipeline':<27} | {'Avg CER':<9} | {'Avg WER':<9} | {'Accuracy':<10} | {'Avg Conf':<9} | {'Min Conf':<9}")
    print("-" * 80)

    summary_metrics = {}
    for s in strategies:
        data = results[s]
        n = max(1, len(data["acc"]))
        avg_cer = sum(data["cer"]) / n
        avg_wer = sum(data["wer"]) / n
        avg_acc = sum(data["acc"]) / n
        avg_conf = sum(data["conf"]) / n
        avg_min_conf = sum(data["min_conf"]) / n
        summary_metrics[s] = {
            "cer": avg_cer,
            "wer": avg_wer,
            "acc": avg_acc,
            "conf": avg_conf,
            "min_conf": avg_min_conf,
        }
        print(
            f"{s:<27} | {avg_cer*100:>6.2f}%  | {avg_wer*100:>6.2f}%  | {avg_acc*100:>7.2f}%  | {avg_conf*100:>6.2f}%  | {avg_min_conf*100:>6.2f}%"
        )
    print("=" * 80)

    base_acc = summary_metrics["C_Baseline_Preprocessed"]["acc"]
    impr_acc = summary_metrics["D_Improved_Adaptive"]["acc"]
    rel_impr = ((impr_acc - base_acc) / base_acc) * 100.0 if base_acc > 0 else 0.0

    print(f"\n[+] Character Accuracy Improvement: {rel_impr:+.2f}%")
    print("    Note: Character Accuracy = 1 - CER. Model Confidence != Character Accuracy.\n")

    print("--- SAMPLE COMPARISONS ---")
    for log in sample_logs[:4]:
        print(f"\nFile: {log['name']}")
        print(f"  GT:       {log['gt']}")
        print(f"  Baseline: {log['pred_baseline']} (Acc: {log['acc_baseline']*100:.1f}%)")
        print(f"  Improved: {log['pred_improved']} (Acc: {log['acc_improved']*100:.1f}%, Conf: {log['conf_improved']*100:.1f}%)")

    return summary_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanskrit Devanagari OCR Pipeline Benchmark")
    parser.add_argument("--eval_dir", type=str, default="data/test_evaluation")
    parser.add_argument("--rec_model_dir", type=str, default="models/sanskrit_finetuned")
    parser.add_argument("--char_dict_path", type=str, default="data/sanskrit_dict.txt")
    args = parser.parse_args()

    target_dir = PROJECT_DIR / args.eval_dir
    run_benchmark(
        eval_dir=target_dir,
        rec_model_dir=args.rec_model_dir,
        char_dict_path=args.char_dict_path,
    )
