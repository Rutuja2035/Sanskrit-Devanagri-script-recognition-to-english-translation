"""Multi-domain Sanskrit Devanagari dataset generator for OCR fine-tuning.

Generates realistic text line images and corresponding ground-truth label files
across three domains:
1. Digital / Printed (clear typography, multi-font, high contrast)
2. Handwritten (simulated stroke thickness variation, slant, tremor, DHCD blending)
3. Manuscripts (palm-leaf/birch-bark texture, ink bleed, degradation, Vedic accents)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure safe console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# Corpus of Classical Sanskrit, Vedic Hymns & Shlokas
# ---------------------------------------------------------------------------

SANSKRIT_CORPUS: List[str] = [
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
    "अष्टाध्यायी ८.४.४० स्तोः श्चुना श्चुः ॥",
    "अष्टाध्यायी ८.४.४१ ष्टुना ष्टुः ॥",
    # Upanishads, Panchatantra & Classical Prose
    "सत्यं वद धर्मं चर स्वाध्यायान्मा प्रमदः ।",
    "मातृदेवो भव पितृदेवो भव आचार्यदेवो भव अतिथिदेवो भव ॥",
    "नायमात्मा बलहीनेन लभ्यो न च प्रमादात्तपसो वाप्यलिङ्गात् ।",
    "भिद्यते हृदयग्रन्थिश्छिद्यन्ते सर्वसंशयाः ।",
    "क्षीयन्ते चास्य कर्माणि तस्मिन्दृष्टे परावरे ॥",
    "हिरण्मयेन पात्रेण सत्यस्यापिहितं मुखम् ।",
    "तत्त्वं पूषन्नपावृणु सत्यधर्माय दृष्टये ॥",
    "मित्रलाभो मित्रभेदः विग्रहः सन्धिरेव च ।",
    "लोभात् क्रोधः प्रभवति लोभात् कामः प्रजायते ॥",
]

# ---------------------------------------------------------------------------
# DHCD Real Handwritten Character Mapping & Cache
# ---------------------------------------------------------------------------

DHCD_CHAR_MAP: Dict[str, int] = {
    "क": 0, "ख": 1, "ग": 2, "ट": 10, "ठ": 11, "ड": 12, "ढ": 13, "ण": 14,
    "त": 15, "थ": 16, "द": 17, "ध": 18, "न": 19, "प": 20, "फ": 21, "ब": 22,
    "भ": 23, "म": 24, "य": 25, "र": 26, "ल": 27, "व": 28, "श": 29
}

DHCD_WORDS: List[str] = [
    "कमल", "नयन", "वचन", "चरण", "पवन", "गमन", "दमन", "शरण", "सरल", "भवन",
    "कपट", "भरत", "नमन", "कनक", "लवण", "मदन", "यश", "रथ", "वन", "तप",
    "दम", "पथ", "बल", "भय", "कर", "पर", "वर", "नर", "दल", "जल",
    "मलय", "तरल", "सबल", "कलम", "समय", "नभ", "पलक", "जनक", "नगर", "मगर"
]

_DHCD_CACHE: Dict[int, List[Path]] = {}


def _get_dhcd_images(class_id: int) -> List[Path]:
    """Retrieve and cache image file paths for a given DHCD class folder."""
    if class_id not in _DHCD_CACHE:
        dhcd_dir = PROJECT_DIR / "data" / "processed" / "test" / str(class_id)
        if dhcd_dir.exists():
            _DHCD_CACHE[class_id] = list(dhcd_dir.glob("*.png"))
        else:
            _DHCD_CACHE[class_id] = []
    return _DHCD_CACHE[class_id]


def render_dhcd_composite_line() -> Tuple[np.ndarray, str]:
    """
    Composite real human handwritten Devanagari characters from the DHCD dataset
    into continuous handwritten Sanskrit words with connected shirorekha (head stroke).
    """
    num_words = random.randint(2, 4)
    words = [random.choice(DHCD_WORDS) for _ in range(num_words)]
    line_text = " ".join(words) + " ।"

    char_h = 44
    glyphs = []

    for word in words:
        word_glyphs = []
        for ch in word:
            cid = DHCD_CHAR_MAP.get(ch)
            if cid is not None:
                imgs = _get_dhcd_images(cid)
                if imgs:
                    chosen_file = random.choice(imgs)
                    raw_glyph = cv2.imread(str(chosen_file), cv2.IMREAD_GRAYSCALE)
                    if raw_glyph is not None:
                        inv_glyph = cv2.bitwise_not(raw_glyph)
                        h_g, w_g = inv_glyph.shape
                        ratio = float(char_h) / max(1, h_g)
                        w_new = max(24, int(w_g * ratio))
                        resized_g = cv2.resize(inv_glyph, (w_new, char_h))
                        word_glyphs.append(resized_g)
                        continue
            word_glyphs.append(np.ones((char_h, 32), dtype=np.uint8) * 255)
        glyphs.append(word_glyphs)

    word_gap = random.randint(18, 28)
    char_gap = random.randint(2, 5)
    total_w = sum(
        sum(g.shape[1] + char_gap for g in wg) - char_gap + word_gap
        for wg in glyphs
    ) + 40
    total_h = 64

    bg_val = random.randint(240, 255)
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * bg_val

    curr_x = 20
    top_y = random.randint(10, 14)

    for word_glyphs in glyphs:
        word_start_x = curr_x
        for g in word_glyphs:
            gh, gw = g.shape
            y_offset = top_y + random.randint(-1, 1)
            canvas_patch = canvas[y_offset:y_offset + gh, curr_x:curr_x + gw]
            for c in range(3):
                canvas_patch[:, :, c] = np.minimum(canvas_patch[:, :, c], g)
            curr_x += gw + char_gap
        word_end_x = curr_x - char_gap

        shiro_y = top_y + 4
        shiro_thick = random.choice([2, 3])
        ink_color = (random.randint(15, 40), random.randint(15, 40), random.randint(15, 40))
        cv2.line(canvas, (word_start_x, shiro_y), (word_end_x, shiro_y), ink_color, shiro_thick)
        curr_x += word_gap

    danda_x = curr_x
    danda_top = top_y + 4
    danda_bot = top_y + char_h
    cv2.line(canvas, (danda_x, danda_top), (danda_x, danda_bot), (25, 25, 25), 2)

    return canvas, line_text

# Common Devanagari font paths (prioritizing historical manuscript fonts)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEV_FONTS: List[str] = [
    str(PROJECT_DIR / "data" / "fonts" / "siddhanta.ttf"),
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/NirmalaB.ttc",
    "C:/Windows/Fonts/mangal.ttf",
    "C:/Windows/Fonts/mangalb.ttf",
    "C:/Windows/Fonts/aparaj.ttf",
    "C:/Windows/Fonts/kokila.ttf",
    "C:/Windows/Fonts/utsaah.ttf",
]


def get_available_font(font_size: int = 36) -> ImageFont.FreeTypeFont:
    """Load the first available Devanagari font (Siddhanta prioritized)."""
    for fp in DEV_FONTS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, font_size)
            except Exception:
                continue
    # Last resort fallback
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Domain 1: Digital / Printed Synthesis
# ---------------------------------------------------------------------------

def render_digital_line(text: str, font_size: int = 36) -> np.ndarray:
    """Render high-contrast digital/printed Sanskrit text line."""
    font = get_available_font(font_size)
    dummy_img = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    padding_x = random.randint(15, 30)
    padding_y = random.randint(10, 20)
    img_w = text_w + 2 * padding_x
    img_h = text_h + 2 * padding_y

    bg_color = random.randint(245, 255)
    img = Image.new("RGB", (img_w, img_h), color=(bg_color, bg_color, bg_color))
    draw = ImageDraw.Draw(img)

    text_color = random.randint(0, 35)
    draw.text((padding_x, padding_y), text, fill=(text_color, text_color, text_color), font=font)

    # Convert to NumPy
    arr = np.array(img)
    # Slight Gaussian noise (typical scanner/camera capture)
    if random.random() > 0.5:
        noise = np.random.normal(0, 2.0, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Domain 2: Handwritten Synthesis
# ---------------------------------------------------------------------------

def render_handwritten_line(text: str, font_size: int = 34, add_ruled_lines: bool = False, ink_style: str = "random") -> np.ndarray:
    """Synthesize handwritten Devanagari text with stroke waviness, pressure, and optional ruled notebook lines."""
    font = get_available_font(font_size)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    padding_x = random.randint(20, 35)
    padding_y = random.randint(15, 25)
    img_w = text_w + 2 * padding_x
    img_h = text_h + 2 * padding_y

    bg_val = random.randint(238, 252)
    img = Image.new("RGB", (img_w, img_h), color=(bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)

    # Ruled Notebook Lines Simulation (faint blue/red/gray lines)
    if add_ruled_lines:
        line_color = (205, 218, 240) if random.random() > 0.5 else (215, 215, 215)
        line_spacing = random.randint(28, 36)
        for y_line in range(12, img_h, line_spacing):
            draw.line([(0, y_line), (img_w, y_line)], fill=line_color, width=1)

    # Ink color selection: black, blue ballpoint, or blue-black fountain pen
    if ink_style == "blue" or (ink_style == "random" and random.random() < 0.4):
        ink_color = (random.randint(15, 35), random.randint(35, 65), random.randint(120, 175))
    else:
        ink_v = random.randint(15, 45)
        ink_color = (ink_v, ink_v, ink_v)

    draw.text((padding_x, padding_y), text, fill=ink_color, font=font)
    arr = np.array(img)

    # 1. Non-uniform stroke thickness (organic pen pressure without breaking strokes)
    if random.random() > 0.35:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thickened = cv2.erode(arr, kernel, iterations=1)
        blend = random.uniform(0.3, 0.6)
        arr = np.clip(arr.astype(np.float32) * (1 - blend) + thickened.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    # 2. Simulated hand tremor / wavy displacement field
    h, w = arr.shape[:2]
    freq = random.uniform(0.015, 0.035)
    amp = random.uniform(0.8, 1.8)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x + amp * math.sin(2 * math.pi * y * freq)
            map_y[y, x] = y + amp * math.cos(2 * math.pi * x * freq)
    arr = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 3. Slight slant (-3 to +3 degrees)
    shear_angle = random.uniform(-2.5, 2.5)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), shear_angle, 1.0)
    arr = cv2.warpAffine(arr, rot_mat, (w, h), borderValue=(bg_val, bg_val, bg_val))

    return arr


# ---------------------------------------------------------------------------
# Domain 3: Ancient Manuscript (Palm-Leaf / Bhojpatra / Parchment)
# ---------------------------------------------------------------------------

def generate_parchment_texture(w: int, h: int) -> np.ndarray:
    """Generate aged paper / palm-leaf texture with fibrous sepia grain."""
    base_r = random.randint(215, 235)
    base_g = random.randint(185, 210)
    base_b = random.randint(140, 175)

    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = base_r
    base[:, :, 1] = base_g
    base[:, :, 2] = base_b

    # Horizontal palm-leaf fibers
    fiber_pattern = np.sin(np.linspace(0, random.uniform(10, 30) * np.pi, h))[:, None]
    fiber_layer = np.tile(fiber_pattern, (1, w)) * random.uniform(8, 16)
    for c in range(3):
        base[:, :, c] += fiber_layer

    # Random discoloration patches / tea-stain effects
    stain = cv2.GaussianBlur(
        np.random.normal(0, 15, (h // 4 + 1, w // 4 + 1)),
        (15, 15),
        0
    )
    stain_resized = cv2.resize(stain, (w, h))
    for c in range(3):
        base[:, :, c] += stain_resized

    return np.clip(base, 0, 255).astype(np.uint8)


def render_manuscript_line(text: str, font_size: int = 34) -> np.ndarray:
    """Render ancient manuscript Sanskrit line with ink bleed, sepia decay, and grain."""
    font = get_available_font(font_size)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    padding_x = random.randint(25, 40)
    padding_y = random.randint(18, 28)
    w = text_w + 2 * padding_x
    h = text_h + 2 * padding_y

    # 1. Background parchment
    bg = generate_parchment_texture(w, h)

    # 2. Text mask (black ink on white canvas)
    mask_img = Image.new("L", (w, h), color=255)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.text((padding_x, padding_y), text, fill=0, font=font)
    mask_np = np.array(mask_img)

    # 3. Ink bleeding & feathering (slight blur on edges)
    ink_bleed = cv2.GaussianBlur(mask_np, (3, 3), 0.5)

    # 4. Faded ink color (dark brownish / aged black carbon ink)
    ink_r = random.randint(30, 55)
    ink_g = random.randint(25, 45)
    ink_b = random.randint(20, 35)

    result = bg.copy()
    alpha = (255 - ink_bleed).astype(np.float32) / 255.0
    # Add slight random fading
    fading = np.random.uniform(0.75, 1.0, (h, w))
    alpha = alpha * fading

    for c, ink_channel in enumerate([ink_r, ink_g, ink_b]):
        result[:, :, c] = np.clip(
            result[:, :, c] * (1.0 - alpha) + ink_channel * alpha,
            0,
            255
        ).astype(np.uint8)

    # 5. Salt-and-pepper grain (micro-losses in ink)
    speckle_mask = np.random.uniform(0, 1, (h, w)) < 0.003
    result[speckle_mask] = bg[speckle_mask]

    return result


# ---------------------------------------------------------------------------
# Dataset Generation Runner
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: Path,
    num_train: int = 2000,
    num_val: int = 300,
) -> Tuple[int, int]:
    """Generate multi-domain training and validation dataset."""
    train_img_dir = output_dir / "train"
    val_img_dir = output_dir / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    train_label_file = output_dir / "train_labels.txt"
    val_label_file = output_dir / "val_labels.txt"

    domains = ["digital", "handwritten_notebook", "handwritten_plain", "manuscript"]

    # 1. Generate Training Samples
    print(f"[*] Generating {num_train} multi-domain training lines...")
    train_lines: List[str] = []
    for i in range(num_train):
        domain = random.choice(domains)

        if domain == "digital":
            text = random.choice(SANSKRIT_CORPUS)
            img = render_digital_line(text, font_size=random.randint(30, 42))
        elif domain == "handwritten_notebook":
            text = random.choice(SANSKRIT_CORPUS)
            img = render_handwritten_line(text, font_size=random.randint(28, 38), add_ruled_lines=True, ink_style="random")
        elif domain == "handwritten_plain":
            text = random.choice(SANSKRIT_CORPUS)
            img = render_handwritten_line(text, font_size=random.randint(28, 38), add_ruled_lines=False, ink_style="random")
        else:
            text = random.choice(SANSKRIT_CORPUS)
            img = render_manuscript_line(text, font_size=random.randint(28, 38))

        filename = f"train_line_{i:05d}_{domain}.jpg"
        filepath = train_img_dir / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Format: relative_path \t label
        train_lines.append(f"train/{filename}\t{text}")

    with open(train_label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    # 2. Generate Validation Samples
    print(f"[*] Generating {num_val} multi-domain validation lines...")
    val_lines: List[str] = []
    for i in range(num_val):
        domain = random.choice(domains)

        if domain == "digital":
            text = random.choice(SANSKRIT_CORPUS)
            img = render_digital_line(text, font_size=34)
        elif domain == "handwritten_notebook":
            text = random.choice(SANSKRIT_CORPUS)
            img = render_handwritten_line(text, font_size=32, add_ruled_lines=True, ink_style="random")
        elif domain == "handwritten_plain":
            text = random.choice(SANSKRIT_CORPUS)
            img = render_handwritten_line(text, font_size=32, add_ruled_lines=False, ink_style="random")
        else:
            text = random.choice(SANSKRIT_CORPUS)
            img = render_manuscript_line(text, font_size=32)

        filename = f"val_line_{i:05d}_{domain}.jpg"
        filepath = val_img_dir / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        val_lines.append(f"val/{filename}\t{text}")

    with open(val_label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"[+] Dataset created successfully in: {output_dir}")
    print(f"    - Train lines: {len(train_lines)} (Labels: {train_label_file})")
    print(f"    - Val lines:   {len(val_lines)} (Labels: {val_label_file})")
    return len(train_lines), len(val_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Domain Sanskrit OCR Dataset")
    parser.add_argument("--output_dir", type=str, default="data/sanskrit_multidomain_dataset")
    parser.add_argument("--num_train", type=int, default=1500)
    parser.add_argument("--num_val", type=int, default=300)

    args = parser.parse_args()
    generate_dataset(
        output_dir=Path(args.output_dir),
        num_train=args.num_train,
        num_val=args.num_val,
    )
