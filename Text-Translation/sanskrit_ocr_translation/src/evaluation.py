"""Systematic evaluation module for OCR and Machine Translation quality metrics.

Provides industry-standard error and similarity metrics:
- OCR Evaluation: Character Error Rate (CER), Word Error Rate (WER), Character Accuracy
- Translation Evaluation: BLEU score, chrF++ score
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import jiwer
except ImportError:  # pragma: no cover
    jiwer = None

try:
    import sacrebleu
except ImportError:  # pragma: no cover
    sacrebleu = None


__all__ = [
    "compute_cer",
    "compute_wer",
    "compute_bleu",
    "compute_chrf",
    "evaluate_ocr_sample",
    "evaluate_translation_sample",
]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) between reference and hypothesis text.

    Args:
        reference: Ground truth reference text.
        hypothesis: OCR predicted hypothesis text.

    Returns:
        float: Character error rate in 0.0 to 1.0 range.
    """
    ref = "" if reference is None else reference
    hyp = "" if hypothesis is None else hypothesis

    # Edge cases: both empty -> 0.0 error
    if not ref.strip() and not hyp.strip():
        return 0.0
    # Reference empty but hypothesis not empty -> 1.0 error
    if not ref.strip():
        return 1.0
    # Reference not empty but hypothesis empty -> 1.0 error
    if not hyp.strip():
        return 1.0

    if jiwer is None:
        if ref == hyp:
            return 0.0
        return 1.0

    try:
        return float(jiwer.cer(ref, hyp))
    except Exception:
        return 1.0 if ref != hyp else 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis text.

    Args:
        reference: Ground truth reference text.
        hypothesis: OCR predicted hypothesis text.

    Returns:
        float: Word error rate in 0.0 to 1.0 range.
    """
    ref = "" if reference is None else reference
    hyp = "" if hypothesis is None else hypothesis

    # Edge cases: both empty -> 0.0 error
    if not ref.strip() and not hyp.strip():
        return 0.0
    # Reference empty but hypothesis not empty -> 1.0 error
    if not ref.strip():
        return 1.0
    # Reference not empty but hypothesis empty -> 1.0 error
    if not hyp.strip():
        return 1.0

    if jiwer is None:
        if ref == hyp:
            return 0.0
        return 1.0

    try:
        return float(jiwer.wer(ref, hyp))
    except Exception:
        return 1.0 if ref != hyp else 0.0


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Compute Corpus BLEU score between references and hypotheses.

    Args:
        references: List of reference (ground truth) strings.
        hypotheses: List of predicted hypothesis strings.

    Returns:
        float: BLEU score in 0.0 to 100.0 scale.
    """
    if not references or not hypotheses:
        return 0.0

    min_len = min(len(references), len(hypotheses))
    if min_len == 0:
        return 0.0

    refs = [str(r) for r in references[:min_len]]
    hyps = [str(h) for h in hypotheses[:min_len]]

    # If all references and hypotheses are empty strings
    if all(not r.strip() for r in refs) and all(not h.strip() for h in hyps):
        return 0.0

    if sacrebleu is None:
        return 0.0

    try:
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        return float(bleu.score)
    except Exception:
        return 0.0


def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    """Compute Corpus chrF++ score between references and hypotheses.

    Args:
        references: List of reference (ground truth) strings.
        hypotheses: List of predicted hypothesis strings.

    Returns:
        float: chrF score in 0.0 to 100.0 scale.
    """
    if not references or not hypotheses:
        return 0.0

    min_len = min(len(references), len(hypotheses))
    if min_len == 0:
        return 0.0

    refs = [str(r) for r in references[:min_len]]
    hyps = [str(h) for h in hypotheses[:min_len]]

    # If all references and hypotheses are empty strings
    if all(not r.strip() for r in refs) and all(not h.strip() for h in hyps):
        return 0.0

    if sacrebleu is None:
        return 0.0

    try:
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        return float(chrf.score)
    except Exception:
        return 0.0


def evaluate_ocr_sample(reference: str, hypothesis: str) -> Dict[str, float]:
    """Evaluate single-sample OCR prediction against ground truth reference.

    Args:
        reference: Ground truth Devanagari/Sanskrit text.
        hypothesis: OCR predicted text.

    Returns:
        Dict[str, float]: Dictionary with 'CER', 'WER', and 'Accuracy' (0.0 to 1.0 range).
    """
    cer_val = compute_cer(reference, hypothesis)
    wer_val = compute_wer(reference, hypothesis)
    acc_val = max(0.0, 1.0 - cer_val)

    return {
        "CER": cer_val,
        "WER": wer_val,
        "Accuracy": acc_val,
    }


def evaluate_translation_sample(reference: str, hypothesis: str) -> Dict[str, float]:
    """Evaluate single-sample machine translation output against reference translation.

    Args:
        reference: Ground truth English translation.
        hypothesis: Candidate / predicted English translation.

    Returns:
        Dict[str, float]: Dictionary with 'BLEU' and 'chrF' (0.0 to 100.0 scale).
    """
    ref = "" if reference is None else reference
    hyp = "" if hypothesis is None else hypothesis

    if not ref.strip() or not hyp.strip():
        return {"BLEU": 0.0, "chrF": 0.0}

    bleu_score = compute_bleu([ref], [hyp])
    chrf_score = compute_chrf([ref], [hyp])

    return {
        "BLEU": bleu_score,
        "chrF": chrf_score,
    }
