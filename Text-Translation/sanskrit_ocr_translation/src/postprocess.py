"""Unicode text post-processing for Sanskrit Devanagari OCR output."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# Pre-compiled regular expressions for performance and readability
_ASCII_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\uFEFF\uFFFE]")
_CONSECUTIVE_SPACES_RE = re.compile(r"[^\S\n\r]+")
_DOUBLE_DANDA_VARIANTS_RE = re.compile(r"(?:॥+|\|{2,}|।{2,})")
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")


def remove_control_characters(text: str) -> str:
    """Remove non-printable ASCII control characters and BOM markers while preserving valid newlines.

    Devanagari text and standard formatting (spaces, newlines) are retained,
    along with zero-width characters (ZWJ/ZWNJ) used in Devanagari typography.

    Args:
        text: Raw input string.

    Returns:
        String with stray control characters stripped.
    """
    return _ASCII_CONTROL_RE.sub("", text)


def normalize_dandas(text: str) -> str:
    """Normalize Sanskrit dandas and ensure consistent spacing around them.

    Converts ASCII pipe variations (||, |) and multiple dandas (।।) to standard
    Devanagari double danda (॥, U+0965) and single danda (।, U+0964).
    Ensures that dandas are padded with spaces around them.

    Args:
        text: Input string with potential OCR danda distortions.

    Returns:
        String with standardized Sanskrit dandas and spacing.
    """
    # First, convert double-danda variants (||, ।।, ॥) to canonical ॥ (U+0965)
    text = _DOUBLE_DANDA_VARIANTS_RE.sub(" ॥ ", text)

    # Convert remaining single danda variants (|) and standard । to spaced । (U+0964)
    # Using negative lookaround to prevent altering already normalized double dandas (॥)
    text = re.sub(r"(?<![।॥])(?:\s*[|।]\s*)(?![।॥])", " । ", text)

    # Ensure double dandas have clean spacing around them
    text = re.sub(r"\s*॥\s*", " ॥ ", text)

    return text


# Regex for split vowel matras: consonant followed by space then dependent vowel sign/halant
_SPLIT_MATRA_RE = re.compile(r"([क-हळक्षज्ञ])\s+([ािीुूृॄॢॣेैोौ्ंःँ॒॑])")
# Regex for manuscript delimiters: middle dots, bullets, asterisks, decorative circles
_MS_DELIM_RE = re.compile(r"\s*[•·*▪]+\s*")
# Regex for orphan vertical stroke at beginning of lines (border residue)
_ORPHAN_BORDER_RE = re.compile(r"^[|।\s]+(?=[क-हअ-औ])")
# Regex for stray bracket / parentheses artifacts at line boundaries
_STRAY_BRACKET_RE = re.compile(r"^[(\[{<]\s*|\s*[)\]}>]$")
# Regex for stray ASCII colon confused with Sanskrit Visarga (: -> ः)
_ASCII_COLON_VISARGA_RE = re.compile(r"([क-हअ-औा-ौ्])\s*:\s*")


def clean_unnecessary_artifacts(text: str) -> str:
    """Strip stray non-Sanskrit bracket artifacts, scanner noise, and orphan symbols."""
    lines = []
    for line in text.split("\n"):
        cleaned = _ASCII_COLON_VISARGA_RE.sub(r"\1ः ", line)
        cleaned = _STRAY_BRACKET_RE.sub("", cleaned.strip())
        lines.append(cleaned)
    return "\n".join(lines)



def clean_manuscript_text(text: str) -> str:
    """Specialized post-processor for ancient Sanskrit manuscript OCR output.
    
    Normalizes:
    - Scribe delimiters (middle dots, bullets) into clean word boundaries.
    - Reconnects detached vowel matras (e.g., 'क े' -> 'के').
    - Cleans up phantom border residue.
    
    Args:
        text: Input Devanagari text string.
        
    Returns:
        Cleaned Sanskrit manuscript text with reconstructed words.
    """
    if not text:
        return ""

    # 1. Normalize scribe delimiters (middle dots and bullets) to spaced separators
    text = _MS_DELIM_RE.sub(" • ", text)

    # 2. Reconnect detached vowel matras and halants
    text = _SPLIT_MATRA_RE.sub(r"\1\2", text)

    # 3. Strip orphan border bars at line starts
    lines = []
    for line in text.split("\n"):
        cleaned_line = _ORPHAN_BORDER_RE.sub("", line).strip()
        lines.append(cleaned_line)

    return "\n".join(lines)


def postprocess_text(text: Optional[str], is_manuscript: bool = True) -> str:
    """Clean, normalize, and format OCR-extracted Sanskrit Devanagari text.

    Applies the following normalization pipeline:
    1. Unicode NFC normalization (combines base glyphs and dependent vowel signs/nuktas).
    2. Removal of stray ASCII control characters and BOMs while preserving Devanagari and newlines.
    3. Normalization of dandas (। and ॥) from ASCII pipe artifacts and consistent space padding.
    4. Manuscript-specific delimiter and matra reconnection if applicable.
    5. Line-by-line whitespace cleanup: strips leading/trailing spaces and collapses multiple spaces.
    6. Collapsing multiple consecutive blank lines to a single blank line.

    Args:
        text: Raw OCR recognized text string (Devanagari / Sanskrit).
        is_manuscript: Whether to apply manuscript-specific delimiter & matra cleanup.

    Returns:
        Post-processed and cleaned Sanskrit text.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # 1. Apply Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings to standard Unix \n
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove stray ASCII control characters
    cleaned = remove_control_characters(normalized)

    # 4. Normalize dandas and spacing around them
    cleaned = normalize_dandas(cleaned)

    # 5. Apply manuscript delimiter & matra reconnection
    if is_manuscript:
        cleaned = clean_manuscript_text(cleaned)

    # 6. Remove stray non-Sanskrit bracket artifacts, colons confused with visarga
    cleaned = clean_unnecessary_artifacts(cleaned)

    # 7. Process line-by-line: collapse intra-line spaces and strip leading/trailing whitespace per line
    lines = cleaned.split("\n")
    processed_lines = []
    for line in lines:
        # Collapse multiple horizontal whitespace characters to a single space
        collapsed_line = _CONSECUTIVE_SPACES_RE.sub(" ", line).strip()
        if collapsed_line:
            processed_lines.append(collapsed_line)

    # 8. Join lines and collapse multiple blank lines into a single blank line
    joined = "\n".join(processed_lines)
    result = _MULTIPLE_BLANK_LINES_RE.sub("\n\n", joined)

    # 9. Strip leading and trailing newlines/whitespace from overall text
    return result.strip()


def split_sanskrit_compounds(word: str, max_splits: int = 3) -> List[Tuple[str, str]]:
    """
    Decompose complex Sanskrit compound words (Sandhi / Samasa) into root components.
    Uses sanskrit_parser and transliteration rules.
    
    Args:
        word (str): A Devanagari Sanskrit word/compound (e.g. 'धर्मात्मा').
        max_splits (int): Maximum split alternatives to return.
        
    Returns:
        List[Tuple[str, str]]: List of (component_1, component_2) in Devanagari.
    """
    clean_word = word.strip()
    if not clean_word or len(clean_word) < 4:
        return []

    try:
        import logging
        logging.getLogger("sanskrit_parser").setLevel(logging.ERROR)
        from sanskrit_parser.base.sanskrit_base import SanskritObject
        from sanskrit_parser.parser.sandhi import Sandhi
        from indic_transliteration import sanscript

        sandhi_engine = Sandhi()
        sandhi_engine.logger.setLevel(logging.ERROR)
        obj = SanskritObject(clean_word, encoding=sanscript.DEVANAGARI)
        raw_splits = sandhi_engine.split_all(obj)

        results = []
        for left, right in list(raw_splits)[:max_splits]:
            left_dev = SanskritObject(left, encoding=sanscript.SLP1).devanagari()
            right_dev = SanskritObject(right, encoding=sanscript.SLP1).devanagari()
            if left_dev and right_dev:
                results.append((left_dev, right_dev))
        return results
    except Exception:
        # Graceful fallback: return empty if parsing fails or library encounters irregular root
        return []


__all__ = [
    "postprocess_text",
    "clean_manuscript_text",
    "normalize_dandas",
    "remove_control_characters",
    "split_sanskrit_compounds",
]

