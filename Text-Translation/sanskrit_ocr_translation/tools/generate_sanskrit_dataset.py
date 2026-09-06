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
    # Bhagavad Gita & Classical
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।",
    "मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।",
    "मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत ।",
    "अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
    "परित्राणाय साधूनां विनाशाय च दुष्कृताम् ।",
    "धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥",
    "नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः ।",
    "न चैनं क्लेदयन्त्यापो न शोषयति मारुतः ॥",
    "अजो नित्यः शाश्वतोऽयं पुराणो न हन्यते हन्यमाने शरीरे ।",
    "ध्यायतो विषयान्पुंसः सङ्गस्तेषूपजायते ।",
    "सङ्गात्सञ्जायते कामः कामात्क्रोधोऽभिजायते ॥",
    "क्रोधाद्भवति संमोहः संमोहात्स्मृतिविभ्रमः ।",
    "स्मृतिभ्रंशाद् बुद्धिनाशो बुद्धिनाशात्प्रणश्यति ॥",
    "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय ।",
    "सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते ॥",
    "दूरेण ह्यवरं कर्म बुद्धियोगाद्धनञ्जय ।",
    "बुद्धौ शरणमन्विच्छ कृपणाः फलहेतवः ॥",
    "युक्तः कर्मफलं त्यक्त्वा शान्तिमाप्नोति नैष्ठिकीम् ।",
    "अयुक्तः कामकारेण फले सक्तो निबध्यते ॥",
    # Subhashitas & Mantras
    "विद्या ददाति विनयं विनयाद्याति पात्रताम् ।",
    "पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम् ॥",
    "उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः ।",
    "न हि सुप्तस्य सिंहस्य प्रविशन्ति मुखे मृगाः ॥",
    "सत्यं ब्रूयात् प्रियं ब्रूयात् न ब्रूयात् सत्यमप्रियम् ।",
    "प्रियं च नानृतं ब्रूयात् एष धर्मः सनातनः ॥",
    "वसुधैव कुटुम्बकम् उदारचरितानां तु ।",
    "अहिंसा परमो धर्मः धर्महिंसा तथैव च ।",
    "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः ।",
    "सर्वे भद्राणि पश्यन्तु मा कश्चिद् दुःखभाग्भवेत् ॥",
    "ॐ असतो मा सद्गमय तमसो मा ज्योतिर्गमय ।",
    "मृत्योर्मा अमृतं गमय ॐ शान्तिः शान्तिः शान्तिः ॥",
    "ॐ पूर्णमदः पूर्णमिदं पूर्णात् पूर्णमुदच्यते ।",
    "पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते ॥",
    # Vedic Verses with Accents
    "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् ।",
    "होता॑रं रत्न॒धात॑मम् ॥",
    "ॐ भूर्भुवः॒ स्वः॑ तत्स॑वि॒तुर्वरे॑ण्यं॒ भर्गो॑ दे॒वस्य॑ धीमहि ।",
    "धियो॒ यो नः॑ प्रचो॒दया॑त् ॥",
    "इ॒षे त्वो॒र्जे त्वा॑ वा॒यव॑ स्थ दे॒वो वः॑ सवि॒ता प्रार्प॑यतु ।",
    "शं नो॑ दे॒वीर॒भीष्ट॑ये॒ शं नो॑ भवन्तु पी॒तये॑ ।",
    # Complex Conjuncts (Samyuktaksara) & Vocabulary
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
]

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

def render_handwritten_line(text: str, font_size: int = 34) -> np.ndarray:
    """Synthesize handwritten Devanagari text with stroke waviness and pressure."""
    font = get_available_font(font_size)
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = max(10, bbox[2] - bbox[0])
    text_h = max(10, bbox[3] - bbox[1])

    padding_x = random.randint(20, 35)
    padding_y = random.randint(15, 25)
    img_w = text_w + 2 * padding_x
    img_h = text_h + 2 * padding_y

    bg_val = random.randint(235, 250)
    img = Image.new("RGB", (img_w, img_h), color=(bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)

    ink_val = random.randint(10, 45)
    draw.text((padding_x, padding_y), text, fill=(ink_val, ink_val, ink_val), font=font)
    arr = np.array(img)

    # 1. Non-uniform stroke thickness (dilation or erosion to simulate pen pressure)
    kernel_size = random.choice([2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if random.random() > 0.5:
        arr = cv2.erode(arr, kernel, iterations=1)
    else:
        arr = cv2.dilate(arr, kernel, iterations=1)

    # 2. Simulated hand tremor / wavy displacement field
    h, w = arr.shape[:2]
    freq = random.uniform(0.02, 0.05)
    amp = random.uniform(1.0, 2.5)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x + amp * math.sin(2 * math.pi * y * freq)
            map_y[y, x] = y + amp * math.cos(2 * math.pi * x * freq)
    arr = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 3. Slight slant (-3 to +3 degrees)
    shear_angle = random.uniform(-3, 3)
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
    num_train: int = 1500,
    num_val: int = 300,
) -> Tuple[int, int]:
    """Generate multi-domain training and validation dataset."""
    train_img_dir = output_dir / "train"
    val_img_dir = output_dir / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    train_label_file = output_dir / "train_labels.txt"
    val_label_file = output_dir / "val_labels.txt"

    domains = ["digital", "handwritten", "manuscript"]

    # 1. Generate Training Samples
    print(f"[*] Generating {num_train} multi-domain training lines...")
    train_lines: List[str] = []
    for i in range(num_train):
        text = random.choice(SANSKRIT_CORPUS)
        domain = random.choice(domains)

        if domain == "digital":
            img = render_digital_line(text, font_size=random.randint(30, 42))
        elif domain == "handwritten":
            img = render_handwritten_line(text, font_size=random.randint(28, 38))
        else:
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
        text = random.choice(SANSKRIT_CORPUS)
        domain = random.choice(domains)

        if domain == "digital":
            img = render_digital_line(text, font_size=34)
        elif domain == "handwritten":
            img = render_handwritten_line(text, font_size=32)
        else:
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
