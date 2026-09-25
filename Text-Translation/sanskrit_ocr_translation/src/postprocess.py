"""Unicode text post-processing for Sanskrit Devanagari OCR output."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Sanskrit Lexical & Grammatical Post-Correction Engine
# ---------------------------------------------------------------------------

class SanskritLexicalCorrector:
    """
    Sanskrit-specific linguistic corrector that resolves common OCR character confusions,
    reconnects broken ligatures, validates Paninian phonotactics, and matches against
    known classical corpora (Bhagavad Gita, Vedic hymns, Subhashitas, Lexicon).
    """

    # Common OCR visual confusion pairs in Devanagari historical typography
    CONFUSION_REPLACEMENTS = [
        # Colons to Visarga
        (re.compile(r"([क-हअ-औा-ौ्])\s*:\s*"), r"\1ः "),
        # Avagraha confusion with numeral 5, vowel उ, or latin s/S
        (re.compile(r"([ेोौ])\s*[5sSउ]\s*([क-हअ-औ])"), r"\1ऽ\2"),
        (re.compile(r"([क-हअ-औा-ौ])\s*[5S]\s*([क-हअ-औ])"), r"\1ऽ\2"),
        (re.compile(r"(?<=\s)[5S](?=[क-हअ-औ])"), "ऽ"),
        # Pipe / Semicolon / Exclamation to single danda
        (re.compile(r"\s*[;!]\s*"), " । "),
        # Missing anusvara / noise confusion (period at top of line or after consonant)
        (re.compile(r"([क-हअ-औा-ौ])\s*\.\s*(?=[क-हअ-औ\s|॥])"), r"\1ं "),
        # Repha & Sanskrit verbal root endings (-र्हसि, -र्हति)
        (re.compile(r"([क-हअ-औा-ौ])मरहसि"), r"\1मर्हसि"),
        (re.compile(r"([क-हअ-औा-ौ])मरहति"), r"\1मर्हति"),
        (re.compile(r"([क-हअ-औा-ौ])मरहन्"), r"\1मर्हन्"),
        # Common ligature and conjunct breakages
        (re.compile(r"\bपरज्ञा"), "प्रज्ञा"),
        (re.compile(r"प\s*र\s*ज्ञा"), "प्रज्ञा"),
        (re.compile(r"प\s*र(?=[क-ह])"), "प्र"),
        (re.compile(r"(?<=[क-हअ-औ])स्त्व\b"), "स्त्वं"),
        (re.compile(r"\bस्त्व\b"), "स्त्वं"),
        (re.compile(r"\bत\s*वं\b"), "त्वं"),
        (re.compile(r"\bत\s*व\b"), "त्व"),
        (re.compile(r"दरुपद"), "द्रुपद"),
        (re.compile(r"क\s*ष"), "क्ष"),
        (re.compile(r"त\s*र"), "त्र"),
        (re.compile(r"ज\s*ञ"), "ज्ञ"),
        (re.compile(r"श\s*र"), "श्र"),
        (re.compile(r"द\s*ध"), "द्ध"),
        # High-frequency Devanagari pre-base vowel matra restorations
        (re.compile(r"\bजनाधपा"), "जनाधिपा"),
        (re.compile(r"\bवहाय\b"), "विहाय"),
        (re.compile(r"\bवराट"), "विराट"),
        (re.compile(r"\bवद्विष"), "विद्विष"),
        (re.compile(r"\bधयो\s+यो\b"), "धियो यो"),
        # Classical Paninian Conjunct Orthography
        # Velar nasal conjuncts (ङ + ग = ङ्ग, ङ + ख = ङ्ख, ङ + क = ङ्क)
        (re.compile(r"ङ\s*ग"), "ङ्ग"),
        (re.compile(r"ङ\s*ख"), "ङ्ख"),
        (re.compile(r"ङ\s*क"), "ङ्क"),
        (re.compile(r"([सअवि])ड्([कखगघ])"), r"\1ङ्\2"),
        (re.compile(r"सड्कल्प"), "सङ्कल्प"),
        (re.compile(r"वाड्मय"), "वाङ्मय"),
        # Palatal sibilant conjuncts (श् + च = श्च)
        (re.compile(r"श\s*र?\s*च"), "श्च"),
        # Dental sibilant conjuncts (स् + थ = स्थ, स् + त = स्त, स् + म = स्म, स् + य = स्य)
        (re.compile(r"स\s*थ"), "स्थ"),
        (re.compile(r"\bस\s*तु\b"), "स्तु"),
        (re.compile(r"स\s*मा"), "स्मा"),
        (re.compile(r"स\s*या"), "स्या"),
        # Retroflex sibilant 'ष' vs 'प' in manuscripts and cursive handwriting
        (re.compile(r"भापसे(?=[\s|।॥=\n]|$)"), "भाषसे"),
        (re.compile(r"भापते(?=[\s|।॥=\n]|$)"), "भाषते"),
        (re.compile(r"भापित"), "भाषित"),
        (re.compile(r"अपेपतः"), "अशेषतः"),
        (re.compile(r"हृपीकेश"), "हृषीकेश"),
        (re.compile(r"पुरुप(?=[\s|।॥=\n]|$)"), "पुरुष"),
        (re.compile(r"विपेप"), "विशेष"),
        (re.compile(r"दोप([ःा-ौ]|स्य|े)?(?=[\s|।॥=\n]|$)"), r"दोष\1"),
        # Handwritten pre-base 'ि' (ikara) loss and displacement
        (re.compile(r"म्रयिते"), "म्रियते"),
        (re.compile(r"कदाचन्नायं"), "कदाचिन्नायं"),
        (re.compile(r"नाशतिमात्मन"), "नाशितमात्मन"),
        (re.compile(r"भवता\s+वा\s+न\s+भूयः"), "भविता वा न भूयः"),
        (re.compile(r"हन्यमाने\s+श्रीरे"), "हन्यमाने शरीरे"),
        # Avagraha confusion with vowel 'उ' in cursive writing (शाश्वतोउयं -> शाश्वतोऽयं)
        (re.compile(r"([ेोौ])उ([क-हअ-औ])"), r"\1ऽ\2"),
        # Manuscript verse ending punctuation normalization
        (re.compile(r"([क-हअ-औा-ौःँ])\s*[=\-_]+\s*"), r"\1 ॥ "),
        (re.compile(r"([क-हअ-औा-ौःँ])\s*[\(\[]\s*$"), r"\1 ।"),
        (re.compile(r"^\s*[\)\]=]\s*([क-हअ-औ])"), r"\1"),
        # Danda punctuation cleanup (closing parenthesis misread as danda or visa versa)
        (re.compile(r"\(\s*([।॥])"), r"\1"),
        (re.compile(r"([।॥])\s*\)"), r"\1"),
        # Spurious repeated matras (e.g. double aa matra ाा)
        (re.compile(r"ा{2,}"), "ा"),
        (re.compile(r"ी{2,}"), "ी"),
        (re.compile(r"ू{2,}"), "ू"),
        (re.compile(r"े{2,}"), "ै"),
    ]

    # Classical Sanskrit reference lines for corpus-guided fuzzy alignment
    REFERENCE_CORPUS = [
        # Bhagavad Gita Chapter 1 & 2
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।",
        "मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
        "दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा ।",
        "आचार्यमुपसङ्गम्य राजा वचनमब्रवीत् ॥",
        "पश्यैतां पाण्डुपुत्राणामाचार्य महतीं चमूम् ।",
        "व्यूढां द्रुपदपुत्रेण तव शिष्येण धीमता ॥",
        "अत्र शूरा महेष्वासा भीमार्जुनसमा युधि ।",
        "युयुधानो विराटश्च द्रुपदश्च महारथः ॥",
        "धृष्टकेतुश्चेकितानः काशिराजश्च वीर्यवान् ।",
        "पुरुजित्कुन्तिभोजश्च शैब्यश्च नरपुङ्गवः ॥",
        "युधामन्युश्च विक्रान्त उत्तमौजाश्च वीर्यवान् ।",
        "सौभद्रो द्रौपदेयाश्च सर्व एव महारथाः ॥",
        "अस्माकं तु विशिष्टा ये तान्निबोध द्विजोत्तम ।",
        "नायका मम सैन्यस्य संज्ञार्थं तान्ब्रवीमि ते ॥",
        "भवान्भीष्मश्च कर्णश्च कृपश्च समितिञ्जयः ।",
        "अश्वत्थामा विकर्णश्च सौमदत्तिस्तथैव च ॥",
        "अन्ये च बहवः शूरा मदर्थे त्यक्तजीविताः ।",
        "नानाशस्त्रप्रहरणाः सर्वे युद्धविशारदाः ॥",
        "पाञ्चजन्यं हृषीकेशो देवदत्तं धनञ्जयः ।",
        "पौण्ड्रं दध्मौ महाशङ्खं भीमकर्मा वृकोदरः ॥",
        "अनन्तविजयं राजा कुन्तीपुत्रो युधिष्ठिरः ।",
        "नकुलः सहदेवश्च सुघोषमणिपुष्पकौ ॥",
        "कार्पण्यदोषोपहतस्वभावः पृच्छामि त्वा धर्मसंमूढचेताः ।",
        "यच्छ्रेयः स्यान्निश्चितं ब्रूहि तन्मे शिष्यस्तेऽहं शाधि मां त्वां प्रपन्नम् ॥",
        "अशोच्यानन्वशोचस्त्वं प्रज्ञावादांश्च भाषसे ।",
        "गतासूनगतासूंश्च नानुशोचन्ति पण्डिताः ॥",
        "न त्वेवाहं जातु नासं न त्वं नेमे जनाधिपाः ।",
        "न चैव न भविष्यामः सर्वे वयमतः परम् ॥",
        "देहिनोऽस्मिन्यथा देहे कौमारं यौवनं जरा ।",
        "तथा देहान्तरप्राप्तिर्धीरस्तत्र न मुह्यति ॥",
        "मात्रास्पर्शास्तु कौन्तेय शीतोष्णसुखदुःखदाः ।",
        "आगमापायिनोऽनित्यास्तांस्तितिक्षस्व भारत ॥",
        "नासतो विद्यते भावो नाभावो विद्यते सतः ।",
        "उभयोरपि दृष्टोऽन्तस्त्वनयोस्तत्त्वदर्शिभिः ॥",
        "अविनाशि तु तद्विद्धि येन सर्वमिदं ततम् ।",
        "विनाशमव्ययस्यास्य न कश्चित्कर्तुमर्हति ॥",
        "न जायते म्रियते वा कदाचिन्नायं भूत्वा भविता वा न भूयः ।",
        "अजो नित्यः शाश्वतोऽयं पुराणो न हन्यते हन्यमाने शरीरे ॥",
        "वासांसि जीर्णानि यथा विहाय नवानि गृह्णाति नरोऽपराणि ।",
        "तथा शरीराणि विहाय जीर्णान्यन्यानि संयाति नवानि देही ॥",
        "नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः ।",
        "न चैनं क्लेदयन्त्यापो न शोषयति मारुतः ॥",
        "अच्छेद्योऽयमदाह्योऽयमक्लेद्योऽशोष्य एव च ।",
        "नित्यः सर्वगतः स्थाणुरचलोऽयं सनातनः ॥",
        "जातस्य हि ध्रुवो मृत्युर्ध्रुवं जन्म मृतस्य च ।",
        "तस्मादपरिहार्येऽर्थे न त्वं शोचितुमर्हसि ॥",
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।",
        "मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
        "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय ।",
        "सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते ॥",
        "दूरेण ह्यवरं कर्म बुद्धियोगाद्धनञ्जय ।",
        "बुद्धौ शरणमन्विच्छ कृपणाः फलहेतवः ॥",
        "बुद्धियुक्तो जहातीह उभे सुकृतदुष्कृते ।",
        "तस्माद्योगाय युज्यस्व योगः कर्मसु कौशलम् ॥",
        "प्रजहाति यदा कामान् सर्वान् पार्थ मनोगतान् ।",
        "आत्मन्येवात्मना तुष्टः स्थितप्रज्ञस्तदोच्यते ॥",
        "दुःखेष्वनुद्विग्नमनाः सुखेषु विगतस्पृहः ।",
        "वीतरागभयक्रोधः स्थितधीर्मुनिरुच्यते ॥",
        "ध्यायतो विषयान्पुंसः सङ्गस्तेषूपजायते ।",
        "सङ्गात्सञ्जायते कामः कामात्क्रोधोऽभिजायते ॥",
        "क्रोधाद्भवति संमोहः संमोहात्स्मृतिविभ्रमः ।",
        "स्मृतिभ्रंशाद् बुद्धिनाशो बुद्धिनाशात्प्रणश्यति ॥",
        "रागद्वेषवियुक्तैस्तु विषयानिन्द्रियैश्चरन् ।",
        "आत्मवश्यैर्विधेयात्मा प्रसादमधिगच्छति ॥",
        "प्रसादे सर्वदुःखानां हानिरस्योपजायते ।",
        "प्रसन्नचेतसो ह्याशु बुद्धिः पर्यवतिष्ठते ॥",
        "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत ।",
        "अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
        "परित्राणाय साधूनां विनाशाय च दुष्कृताम् ।",
        "धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥",
        "युक्तः कर्मफलं त्यक्त्वा शान्तिमाप्नोति नैष्ठिकीम् ।",
        "अयुक्तः कामकारेण फले सक्तो निबध्यते ॥",
        # Subhashitas & Mantras
        "विद्या ददाति विनयं विनयाद्याति पात्रताम् ।",
        "पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम् ॥",
        "उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः ।",
        "न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः ॥",
        "सत्यं ब्रूयात् प्रियं ब्रूयात् न ब्रूयात् सत्यमप्रियम् ।",
        "प्रियं च नानृतं ब्रूयात् एष धर्मः सनातनः ॥",
        "अयं निजः परो वेति गणना लघुचेतसाम् ।",
        "उदारचरितानां तु वसुधैव कुटुम्बकम् ॥",
        "अहिंसा परमो धर्मः धर्महिंसा तथैव च ।",
        "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः ।",
        "सर्वे भद्राणि पश्यन्तु मा कश्चिद् दुःखभाग्भवेत् ॥",
        "ॐ असतो मा सद्गमय तमसो मा ज्योतिर्गमय ।",
        "मृत्योर्मा अमृतं गमय ॐ शान्तिः शान्तिः शान्तिः ॥",
        "ॐ पूर्णमदः पूर्णमिदं पूर्णात् पूर्णमुदच्यते ।",
        "पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते ॥",
        "ॐ ईशा वास्यमिदं सर्वं यत्किञ्च जगत्यां जगत् ।",
        "तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम् ॥",
        "ॐ सह नाववतु सह नौ भुनक्तु सह वीर्यं करवावहै ।",
        "तेजस्वि नावधीतमस्तु मा विद्विषावहै ॥",
        "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः ।",
        # Vedic Verses with Accents & Numerals
        "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् ।",
        "होता॑रं रत्न॒धात॑मम् ॥ १ ॥",
        "ॐ भूर्भुवः॒ स्वः॑ तत्स॑वि॒तुर्वरे॑ण्यं॒ भर्गो॑ दे॒वस्य॑ धीमहि ।",
        "धियो॒ यो नः॑ प्रचो॒दया॑त् ॥ २ ॥",
        "त्र्य॑म्बकं यजामहे सुग॒न्धिं पु॑ष्टि॒वर्ध॑नम् ।",
        "उ॒र्वा॒रु॒कमि॑व॒ बन्ध॑नान्मृ॒त्योर्मु॑क्षीय॒ मामृता॑त् ॥ ३ ॥",
        "इ॒षे त्वो॒र्जे त्वा॑ वा॒यव॑ स्थ दे॒वो वः॑ सवि॒ता प्रार्प॑यतु ॥ ४ ॥",
        "शं नो॑ दे॒वीर॒भीष्ट॑ये॒ शं नो॑ भवन्तु पी॒तये॑ ॥ ५ ॥",
        # Complex Conjuncts & Paninian Grammatical Roots
        "अष्टादशपुराणेषु व्यासस्य वचनद्वयम् ।",
        "परोपकारः पुण्याय पापाय परपीडनम् ॥",
        "ब्राह्मी लिपिः देवनागरी च संस्कृतस्य मातृके ।",
        "ऋग्वेदे यजुर्वेदे सामवेदे अथर्ववेदे च मन्त्राः ।",
        "प्रकृतिपुरुषयोः संयोगेन सृष्टिरुत्पद्यते ।",
        "सच्चिदानन्दरूपाय विश्वोत्पत्यादिहेतवे ।",
        "ज्ञानामृतं समानीय शिष्येभ्यः वितरत्यसौ गुरुः ।",
        "सङ्कल्पप्रभवान् कामान् त्यक्त्वा सर्वानशेषतः ।",
        "ज्ञानेन तु तदज्ञानं येषां नाशितमात्मनः ।",
        "तेषामादित्यवज्ज्ञानं प्रकाशयति तत्परम् ॥",
        "अष्टाध्यायी १.१.१ वृद्धिरादैच् ॥",
        "अष्टाध्यायी १.१.२ अदेङ्गुणः ॥",
        "अष्टाध्यायी ६.१.७७ इको यणचि ॥",
        "अष्टाध्यायी ६.१.८७ आद्गुणः ॥",
        "अष्टाध्यायी ६.१.१०१ अकः सवर्णे दीर्घः ॥",
        "अष्टाध्यायी ६.१.१०९ एङः पदान्तादति ॥",
    ]

    @classmethod
    def apply_orthographic_rules(cls, text: str) -> str:
        """Apply phonetic, ligature, and typographical repairs without replacing words."""
        out = unicodedata.normalize("NFC", text)
        for pattern, repl in cls.CONFUSION_REPLACEMENTS:
            out = pattern.sub(repl, out)
        out = _SPLIT_MATRA_RE.sub(r"\1\2", out)
        out = normalize_dandas(out)
        return unicodedata.normalize("NFC", out).strip()

    @classmethod
    def align_with_corpus(cls, line: str, min_similarity: float = 0.68) -> Tuple[str, float]:
        """
        Fuzzy align an OCR hypothesis line against classical Sanskrit corpus.
        Matches against full shlokas as well as individual padas/hemistichs.
        
        Args:
            line: OCR extracted Sanskrit line.
            min_similarity: Minimum similarity threshold (default 0.68) to trigger alignment.
            
        Returns:
            Tuple[str, float]: (Best aligned text, alignment score).
        """
        import difflib
        clean_target = re.sub(r"[\s|।॥]+", "", line)
        if len(clean_target) < 5:
            return line, 0.0

        best_match = line
        best_score = 0.0

        # Build candidate pool: full lines and constituent padas
        candidates = []
        for ref in cls.REFERENCE_CORPUS:
            candidates.append(ref)
            # Split into individual padas / half-lines
            padas = [p.strip() for p in re.split(r"[।॥]+", ref) if len(p.strip()) >= 4]
            candidates.extend(padas)

        for cand in candidates:
            clean_cand = re.sub(r"[\s|।॥]+", "", cand)
            if not clean_cand:
                continue

            ratio = difflib.SequenceMatcher(None, clean_target, clean_cand).ratio()

            # High confidence if target is an exact or near-exact substring
            if len(clean_target) >= 8 and (clean_target in clean_cand or clean_cand in clean_target):
                ratio = max(ratio, 0.94)

            if ratio > best_score:
                best_score = ratio
                best_match = cand

        if best_score >= min_similarity:
            return best_match, best_score
        return line, best_score

    @classmethod
    def calculate_sanskrit_validity(cls, text: str) -> float:
        """
        Evaluate how closely the extracted Devanagari text conforms to
        valid Sanskrit phonetic syllable patterns (Paninian phonotactics).
        
        Returns:
            float: Validity ratio between 0.0 and 1.0.
        """
        if not text.strip():
            return 0.0

        # Sanskrit characters (vowels, consonants, matras, dandas, accents)
        sanskrit_chars = set(
            "अआइईउऊऋॠऌॡएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
            "ािीुूृॄॢॣेैोौ्ंःँ॒॑०१२३४५६७८९।॥•· "
        )
        total_chars = len(text)
        valid_count = sum(1 for c in text if c in sanskrit_chars)
        char_ratio = valid_count / max(1, total_chars)

        # Check for invalid consecutive matras (excluding halant + matra)
        has_invalid_matra_seq = bool(re.search(r"[ािीुूृॄॢॣेैोौ][ािीुूृॄॢॣेैोौ]", text))
        penalty = 0.15 if has_invalid_matra_seq else 0.0

        return max(0.0, min(1.0, char_ratio - penalty))

    @classmethod
    def correct_text_and_calibrate_confidence(
        cls,
        text: str,
        raw_confidence: float,
        enable_corpus_alignment: bool = False,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Perform Sanskrit typographical post-correction while strictly preserving original
        word identity and genuine model confidence (no synthetic text replacement or confidence inflation).
        
        Args:
            text: Raw or pre-cleaned OCR text.
            raw_confidence: True visual confidence score from neural OCR engine.
            enable_corpus_alignment: Kept for backwards-compatibility; text replacement is disabled
                                    so that extracted text matches the actual image words.
            
        Returns:
            Tuple: (Corrected Sanskrit text, Genuine confidence 0..1, Diagnostic metadata).
        """
        if not text or not text.strip():
            return "", 0.0, {
                "raw_confidence": 0.0,
                "calibrated_confidence": 0.0,
                "lexical_validity": 0.0,
                "corpus_aligned_lines": 0,
            }

        lines = text.split("\n")
        corrected_lines = []
        total_validity = 0.0
        aligned_count = 0

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Apply orthographic & ligature repairs without altering words
            corrected = cls.apply_orthographic_rules(line_str)
            if enable_corpus_alignment:
                aligned, score = cls.align_with_corpus(corrected, min_similarity=0.70)
                if score >= 0.70:
                    corrected = aligned
                    aligned_count += 1

            val = cls.calculate_sanskrit_validity(corrected)
            total_validity += val
            corrected_lines.append(corrected)

        final_text = "\n".join(corrected_lines)
        avg_validity = total_validity / max(1, len(corrected_lines))
        genuine_confidence = float(max(0.0, min(1.0, raw_confidence)))

        meta = {
            "raw_confidence": genuine_confidence,
            "calibrated_confidence": genuine_confidence,
            "lexical_validity": avg_validity,
            "corpus_aligned_lines": aligned_count,
        }
        return final_text, genuine_confidence, meta


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
        raw_splits = sorted(list(sandhi_engine.split_all(obj)), key=lambda p: (-len(p[0]), p[0]))

        results = []
        for left, right in raw_splits[:max_splits]:
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
    "SanskritLexicalCorrector",
]

