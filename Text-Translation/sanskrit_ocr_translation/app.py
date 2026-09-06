# ── Windows PyTorch DLL fix – must run before ANY torch import ──────────────
import os, sys, importlib.util as _ilu
if sys.platform == "win32":
    _spec = _ilu.find_spec("torch")
    if _spec and _spec.submodule_search_locations:
        _lib = os.path.join(list(_spec.submodule_search_locations)[0], "lib")
        if os.path.isdir(_lib):
            os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(_lib)
            except Exception:
                pass
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import numpy as np
import cv2
import time
import json
import pandas as pd
from PIL import Image
from pathlib import Path

# Setup paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

# Import internal modules
from src.preprocessing import (
    preprocess_basic,
    preprocess_adaptive,
    preprocess_otsu,
    preprocess_sauvola,
    preprocess_manuscript,
    to_grayscale,
    denoise,
    enhance_contrast,
)
from src.ocr import PaddleSanskritOCR
from src.postprocess import postprocess_text, split_sanskrit_compounds
from src.translate import SanskritToEnglishTranslator
from src.segmentation import segment_zones, detect_shirorekha, remove_shirorekha
from src.evaluation import compute_cer, compute_wer, compute_bleu, compute_chrf, evaluate_ocr_sample, evaluate_translation_sample
try:
    from src.dl_model import DHCD_MAP
except ImportError:
    # The DHCD gallery is optional and does not affect document OCR.
    DHCD_MAP = {}

st.set_page_config(
    page_title="Sanskrit Vision - OCR & Translation Pipeline",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Resource Loaders (Cached)
# ---------------------------------------------------------

@st.cache_resource
def load_paddle_ocr_model(variant: str = "Fine-Tuned Multi-Domain Sanskrit"):
    device = os.getenv("PADDLE_DEVICE", "cpu")
    if variant == "Fine-Tuned Multi-Domain Sanskrit":
        finetuned_dir = APP_DIR / "models" / "sanskrit_finetuned"
        dict_path = APP_DIR / "data" / "sanskrit_dict.txt"
        return PaddleSanskritOCR(
            device=device,
            rec_model_dir=finetuned_dir if finetuned_dir.exists() else None,
            char_dict_path=dict_path if dict_path.exists() else None,
        )
    return PaddleSanskritOCR(device=device)

@st.cache_resource
def load_translation_model():
    return SanskritToEnglishTranslator()

@st.cache_data
def load_sample_datasets():
    data_dir = PROJECT_ROOT / "data" / "processed" / "translation"
    gita_path = PROJECT_ROOT / "data" / "raw" / "translation" / "Bhagwad_Gita.csv"
    vedic_path = PROJECT_ROOT / "data" / "raw" / "translation" / "vedic.csv"
    test_csv_path = data_dir / "test.csv"
    
    datasets = {
        "gita": pd.read_csv(gita_path).head(100) if gita_path.exists() else pd.DataFrame(),
        "vedic": pd.read_csv(vedic_path).head(100) if vedic_path.exists() else pd.DataFrame(),
        "test_pairs": pd.read_csv(test_csv_path).head(50) if test_csv_path.exists() else pd.DataFrame(),
    }
    return datasets

translation_model = load_translation_model()
sample_data = load_sample_datasets()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/om.png", width=64)
st.sidebar.title("🕉️ Sanskrit Vision")
st.sidebar.caption("AI-Powered Sanskrit OCR & Machine Translation")
st.sidebar.markdown("---")

st.sidebar.subheader("OCR Engine Configuration")
selected_model_variant = st.sidebar.selectbox(
    "Active Model Variant",
    [
        "Fine-Tuned Multi-Domain Sanskrit",
        "Official PP-OCRv5 Devanagari",
    ],
    index=0,
    help="Fine-tuned multi-domain model handles digital typography, handwriting, and manuscripts."
)
paddle_ocr_model = load_paddle_ocr_model(selected_model_variant)

st.sidebar.subheader("System Status")
if paddle_ocr_model.is_available:
    st.sidebar.success(f"✓ {paddle_ocr_model.model_variant} Active")
else:
    st.sidebar.warning("⚠️ PaddleOCR is unavailable — install requirements first")

corpus_size = len(translation_model.corpus_dict)
st.sidebar.success(f"✓ Corpus Dictionary: {corpus_size:,} entries")

st.sidebar.markdown("---")
st.sidebar.markdown("**Datasets Integrated:**")
st.sidebar.markdown("• **DHCD**: 92,000+ Character Images\n• **Bhagavad Gita**: 700+ Verses\n• **Vedic Lexicon**: 1,200+ Terms\n• **Itihasa Corpus**: 94,000+ Pairs\n• **Ramayana**: 18,000+ Verses")

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------

tabs = st.tabs([
    "🏠 Home",
    "🔍 OCR Pipeline",
    "🌐 Translation",
    "📊 Evaluation (CER/BLEU)",
    "📚 Dataset Explorer",
    "ℹ️ About"
])

# =========================================================
# TAB 1: HOME
# =========================================================
with tabs[0]:
    st.title("🕉️ Sanskrit Vision")
    st.subheader("Automated Sanskrit Devanagari Recognition & English Translation Pipeline")

    st.markdown("""
    **Sanskrit Vision** digitizes and translates handwritten and printed Sanskrit manuscripts in Devanagari script using 
    image processing, PaddleOCR for Sanskrit Devanagari, and a hybrid classical-neural translation engine.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Training Pairs", "76,448", "Itihasa + Gita + Vedic")
    with col2:
        st.metric("Devanagari Classes", "46", "36 Consonants + 10 Digits")
    with col3:
        st.metric("OCR Engine", "PaddleOCR", "Devanagari document OCR")
    with col4:
        st.metric("Lookup Dictionary", f"{corpus_size:,}", "Verified Translations")

    st.markdown("---")
    st.markdown("### 🔄 End-to-End Pipeline Architecture")

    st.code("""
  [ 📷 Input Image ] 
         │
         ▼
  [ 🖼️ Image Preprocessing ] ─── (Grayscale, Denoise, Contrast CLAHE, Otsu/Adaptive Thresholding)
         │
         ▼
  [ ✂️ Line & Zone Segmentation ] ─── (Shirorekha Headline Isolation, Upper/Middle/Lower Zones)
         │
         ▼
  [ 🔍 OCR ] ─── (PaddleOCR Devanagari)
         │
         ▼
  [ 🧹 Text Post-Processing ] ─── (Devanagari Unicode Normalization & Danda Formatting)
         │
         ▼
  [ 🌐 Hybrid Translation Engine ] ─── (Classical Corpus Match -> Neural MT Fallback)
         │
         ▼
  [ 📄 English Translation & Metrics ] ─── (Attribution, CER, WER, BLEU, chrF)
    """, language="text")

    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. Navigate to **🔍 OCR Pipeline** to upload a Sanskrit manuscript or select a sample character.
    2. Inspect intermediate **preprocessing** and **zone segmentations**.
    3. Run recognition with **PaddleOCR Sanskrit**.
    4. Seamlessly push the recognized Sanskrit text to the **🌐 Translation** tab for instant English meaning.
    5. Test recognition and translation quality in the **📊 Evaluation** tab using industry standard metrics (**CER**, **WER**, **BLEU**, **chrF**).
    """)


# =========================================================
# TAB 2: OCR PIPELINE
# =========================================================
with tabs[1]:
    st.header("🔍 Sanskrit OCR Pipeline")
    st.write("Upload an image of handwritten/printed Sanskrit or select from sample test glyphs.")

    input_mode = st.radio("Select Image Source", ["Upload Custom Image", "Sample Character from DHCD Test Set"], horizontal=True)

    image_np = None

    if input_mode == "Upload Custom Image":
        uploaded_file = st.file_uploader("Upload Sanskrit Document / Character Image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
    else:
        # Sample selection from DHCD
        sample_classes = {
            "Character 'क' (ka)": "character_1_ka",
            "Character 'ख' (kha)": "character_2_kha",
            "Character 'ग' (ga)": "character_3_ga",
            "Character 'च' (cha)": "character_6_cha",
            "Character 'त' (ta)": "character_16_tabala",
            "Character 'म' (ma)": "character_25_ma",
            "Character 'श' (sha)": "character_30_motosaw",
            "Digit '०' (0)": "digit_0",
            "Digit '५' (5)": "digit_5",
            "Digit '९' (9)": "digit_9",
        }
        chosen_sample = st.selectbox("Choose a sample character class:", list(sample_classes.keys()))
        folder_name = sample_classes[chosen_sample]
        sample_dir = PROJECT_ROOT / "data" / "raw" / "ocr" / "dhcd" / "Test" / folder_name
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.png"))[:5]
            if sample_files:
                selected_sample_file = sample_files[0]
                image = Image.open(selected_sample_file)
                image_np = np.array(image)
                st.caption(f"Loaded sample: `{selected_sample_file.name}` from class `{folder_name}`")

    if image_np is not None:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Original Input Image")
            st.image(image_np, width=280)

        # Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            domain_profile = st.selectbox(
                "🎯 Document Domain Analysis Profile",
                [
                    "Auto-Adaptive (Recommended)",
                    "Ancient Manuscript & Palm-Leaf (ताड़पत्र/भूर्जपत्र)",
                    "Manual / Handwritten Script (हस्तलिखित)",
                    "Digital & Modern Printed (मुद्रित ग्रन्थ)",
                ],
                help="Automatically tunes preprocessing, contrast curves, border filters, and grammar repair for the chosen document type."
            )
        with col_ctrl2:
            st.text_input("Active OCR Model", value=paddle_ocr_model.model_variant, disabled=True)

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            suppress_borders = st.checkbox("Auto-Suppress Framing Margins", value=(domain_profile != "Digital & Modern Printed (मुद्रित ग्रन्थ)"))
        with col_opt2:
            multi_column_mode = st.checkbox("Multi-Column / Commentary Layout", value=False)
        with col_opt3:
            show_segmentation = st.checkbox("Show Zone Segmentation", value=True)

        if st.button("🚀 Run OCR Recognition", type="primary"):
            with st.spinner("Processing image and recognizing Devanagari text..."):
                start_time = time.time()

                # Preprocessing
                # Domain-Adaptive Preprocessing
                if "Ancient" in domain_profile or "Palm-Leaf" in domain_profile:
                    prep_img = preprocess_manuscript(image_np, suppress_borders=suppress_borders, auto_scale=False)
                    is_ms = True
                elif "Manual" in domain_profile or "Handwritten" in domain_profile:
                    # Handwritten documents benefit from adaptive Sauvola binarization with despeckling
                    prep_img = preprocess_sauvola(image_np, window_size=25, k=0.18, despeckle=True)
                    is_ms = False
                elif "Digital" in domain_profile:
                    # Digital documents have sharp contrast: basic gentle denoise to preserve crisp fonts
                    prep_img = preprocess_basic(image_np)
                    is_ms = False
                else:
                    # Auto-Adaptive: check image variance & color saturation
                    is_color = len(image_np.shape) == 3 and image_np.shape[2] >= 3
                    if is_color:
                        # Check sepia / yellow saturation characteristic of ancient manuscripts
                        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                        mean_sat = np.mean(hsv[:, :, 1])
                        if mean_sat > 35:  # Parchment / sepia tone
                            prep_img = preprocess_manuscript(image_np, suppress_borders=suppress_borders, auto_scale=False)
                            is_ms = True
                        else:
                            prep_img = preprocess_sauvola(image_np, despeckle=True)
                            is_ms = False
                    else:
                        prep_img = preprocess_sauvola(image_np, despeckle=True)
                        is_ms = False

                with col_img2:
                    st.subheader("Preprocessed Image")
                    st.image(prep_img, width=280)

                # Zone Segmentation
                if show_segmentation:
                    st.markdown("#### ✂️ Shirorekha & Zone Analysis")
                    try:
                        shirorekha_y = detect_shirorekha(prep_img)
                        zones = segment_zones(prep_img)
                        
                        z_col0, z_col1, z_col2, z_col3 = st.columns(4)
                        z_col0.metric("Shirorekha Y-Index", f"{shirorekha_y} px")
                        if zones["upper"].size > 0:
                            z_col1.image(zones["upper"], caption="Upper Zone (Matras / Modifiers)")
                        if zones["middle"].size > 0:
                            z_col2.image(zones["middle"], caption="Middle Zone (Main Consonants)")
                        if zones["lower"].size > 0:
                            z_col3.image(zones["lower"], caption="Lower Zone (Vowel Markers / Halant)")
                    except Exception as e:
                        st.info(f"Segmentation visualization note: {e}")

                # OCR Execution
                ocr_start = time.time()
                ocr_result = paddle_ocr_model.recognize(
                    prep_img,
                    suppress_borders=suppress_borders,
                    multi_column=multi_column_mode
                )
                ocr_duration = time.time() - ocr_start

                # Post-processing (is_ms is set dynamically by domain profile above)
                cleaned_text = postprocess_text(ocr_result.get("text", ""), is_manuscript=is_ms)
                total_duration = time.time() - start_time

                st.markdown("---")
                st.subheader("📝 Recognition Results")

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.text_area("Raw OCR Output", ocr_result.get("text", ""), height=110)
                with res_col2:
                    st.text_area("Cleaned Sanskrit (Devanagari)", cleaned_text, height=110)

                # Sanskrit Sandhi / Compound Decomposition
                with st.expander("🔍 Sanskrit Sandhi & Compound Word Decomposition"):
                    words = [w.strip() for w in cleaned_text.replace("\n", " ").split(" ") if len(w.strip()) > 3]
                    if words:
                        sandhi_records = []
                        for w in words[:15]:
                            splits = split_sanskrit_compounds(w, max_splits=2)
                            if splits:
                                sandhi_records.append({
                                    "Compound Word": w,
                                    "Primary Root Split": f"{splits[0][0]} + {splits[0][1]}",
                                    "Alternative Split": f"{splits[1][0]} + {splits[1][1]}" if len(splits) > 1 else "—"
                                })
                        if sandhi_records:
                            st.dataframe(pd.DataFrame(sandhi_records), use_container_width=True)
                        else:
                            st.caption("No complex compound words requiring Sandhi splitting detected in sample.")
                    else:
                        st.caption("Recognize text to inspect compound word breakdowns.")

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Engine Confidence", f"{ocr_result.get('confidence', 0.0):.2%}")
                m_col2.metric("OCR Duration", f"{ocr_duration:.3f}s")
                m_col3.metric("Total Pipeline Time", f"{total_duration:.3f}s")

                if ocr_result.get("segments"):
                    st.markdown("#### 🧩 Detected Sanskrit Text Lines")
                    seg_data = []
                    for seg in ocr_result["segments"]:
                        seg_data.append({
                            "Text": seg.get("text", ""),
                            "Confidence": f"{seg.get('conf', 0.0):.2%}",
                            "Bounding Box (X, Y, W, H)": str(seg.get("box", ""))
                        })
                    st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

                st.session_state['cleaned_sanskrit'] = cleaned_text

                dl_col1, dl_col2 = st.columns([1, 1])
                with dl_col1:
                    st.download_button("💾 Download Recognized Text", cleaned_text, file_name="sanskrit_ocr_output.txt")
                with dl_col2:
                    if st.button("➡️ Send to Translation Pipeline"):
                        st.session_state['text_to_translate'] = cleaned_text
                        st.success("Sent to Translation tab! Click on the '🌐 Translation' tab above.")


# =========================================================
# TAB 3: TRANSLATION PIPELINE
# =========================================================
with tabs[2]:
    st.header("🌐 Sanskrit to English Translation Pipeline")
    st.write("Translate classical Sanskrit verses, Vedic terms, or OCR-extracted text into English.")

    # Preset examples
    preset_options = [
        "Custom Input",
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥ (Bhagavad Gita 1.1)",
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥ (Bhagavad Gita 2.47)",
        "वृथा वृष्टिः समुद्रेषु वृथा तृप्तेषु भोजनम्। वृथा दानं धनाढ्येषु वृथा दीपो दिवापि च॥ (Chanakya Niti)",
        "सत्यमेव जयते नानृतम् (Mundaka Upanishad)",
        "विद्या ददाति विनयं विनयाद्याति पात्रताम्। (Hitopadesha)",
        "दूर्वा (Vedic Term - Sacred Grass)",
        "अश्वमेध (Vedic Term - Horse Sacrifice Ritual)"
    ]

    selected_preset = st.selectbox("Select Preset Verse / Vedic Term or type custom text:", preset_options)

    default_input = ""
    if selected_preset != "Custom Input":
        # Extract Sanskrit part before parenthesis
        default_input = selected_preset.split("(")[0].strip()
    elif 'text_to_translate' in st.session_state:
        default_input = st.session_state['text_to_translate']
    elif 'cleaned_sanskrit' in st.session_state:
        default_input = st.session_state['cleaned_sanskrit']

    sanskrit_input = st.text_area("Sanskrit Source Text (Devanagari):", value=default_input, height=140)

    if st.button("🌐 Translate to English", type="primary"):
        if sanskrit_input.strip():
            with st.spinner("Translating Sanskrit text..."):
                start_time = time.time()
                meta_res = translation_model.translate_with_metadata(sanskrit_input)
                elapsed = time.time() - start_time

                st.markdown("---")
                st.subheader("📖 Translation Results")

                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.markdown("**Sanskrit Input:**")
                    st.info(meta_res["cleaned_sanskrit"])
                with t_col2:
                    st.markdown(f"**English Translation:** ({meta_res['source']})")
                    st.success(meta_res["english"])

                res_m1, res_m2, res_m3 = st.columns(3)
                res_m1.metric("Translation Source", meta_res["source"])
                res_m2.metric("Match Method", meta_res["match_type"])
                res_m3.metric("Translation Latency", f"{elapsed:.3f}s")

                st.download_button("💾 Download Translation", meta_res["english"], file_name="sanskrit_english_translation.txt")
        else:
            st.warning("Please provide Sanskrit Devanagari text to translate.")


# =========================================================
# TAB 4: EVALUATION MODULE
# =========================================================
with tabs[3]:
    st.header("📊 Systematic Pipeline Evaluation")
    st.write("Compute industry standard error and similarity metrics for OCR (CER, WER) and Machine Translation (BLEU, chrF).")

    eval_tab1, eval_tab2 = st.tabs(["🧮 Interactive Metric Calculator", "📈 Batch Test Set Benchmark"])

    with eval_tab1:
        st.subheader("Interactive Single-Sample Evaluator")
        eval_mode = st.radio("Evaluation Type", ["OCR Recognition Evaluation (CER, WER)", "Machine Translation Evaluation (BLEU, chrF)"], horizontal=True)

        if eval_mode.startswith("OCR"):
            st.markdown("##### 🔍 OCR Evaluation (Character Error Rate & Word Error Rate)")
            c1, c2 = st.columns(2)
            with c1:
                ref_ocr = st.text_input("Ground Truth Devanagari Text (Reference):", value="धर्मक्षेत्रे कुरुक्षेत्रे")
            with c2:
                hyp_ocr = st.text_input("OCR Predicted Text (Hypothesis):", value="धर्मक्षेत्रे कुरूक्षेत्रे")

            if st.button("Calculate OCR Metrics"):
                ocr_eval = evaluate_ocr_sample(ref_ocr, hyp_ocr)
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Character Error Rate (CER)", f"{ocr_eval['CER']:.2%}")
                ec2.metric("Word Error Rate (WER)", f"{ocr_eval['WER']:.2%}")
                ec3.metric("Character Accuracy", f"{ocr_eval['Accuracy']:.2%}")
                st.caption("Lower CER and WER indicate higher recognition accuracy.")

        else:
            st.markdown("##### 🌐 Translation Evaluation (BLEU & chrF++ Score)")
            c1, c2 = st.columns(2)
            with c1:
                ref_tr = st.text_area("Reference English Translation (Ground Truth):", value="In the holy field of Kurukshetra, assembled together desiring to fight, what did my sons and the sons of Pandu do?")
            with c2:
                hyp_tr = st.text_area("Candidate / Predicted English Translation:", value="On the holy plain of Kurukshetra, gathered together eager for battle, what did my sons and Pandu's sons do?")

            if st.button("Calculate Translation Metrics"):
                tr_eval = evaluate_translation_sample(ref_tr, hyp_tr)
                tc1, tc2 = st.columns(2)
                tc1.metric("BLEU Score (0 - 100)", f"{tr_eval['BLEU']:.2f}")
                tc2.metric("chrF++ Score (0 - 100)", f"{tr_eval['chrF']:.2f}")
                st.caption("Higher BLEU and chrF scores indicate closer fidelity to the ground truth reference.")

    with eval_tab2:
        st.subheader("Batch Dataset Benchmark on Processed Test Split")
        if not sample_data["test_pairs"].empty:
            test_df = sample_data["test_pairs"].head(20)
            st.write(f"Sample test pairs loaded ({len(test_df)} entries from `data/processed/translation/test.csv`):")
            st.dataframe(test_df[["sanskrit", "english", "source"]].head(10), use_container_width=True)

            if st.button("🚀 Run Live Batch Translation Benchmark (First 10 Samples)"):
                with st.spinner("Running batch evaluation..."):
                    subset = test_df.head(10)
                    preds = []
                    for s in subset["sanskrit"]:
                        preds.append(translation_model.translate(s))
                    
                    refs = subset["english"].tolist()
                    bleu = compute_bleu(refs, preds)
                    chrf = compute_chrf(refs, preds)

                    b_col1, b_col2 = st.columns(2)
                    b_col1.metric("Corpus BLEU Score", f"{bleu:.2f}")
                    b_col2.metric("Corpus chrF++ Score", f"{chrf:.2f}")

                    # Detailed inspection table
                    bench_table = []
                    for i in range(len(preds)):
                        bench_table.append({
                            "Sanskrit Source": subset["sanskrit"].iloc[i],
                            "Reference English": refs[i],
                            "Model Output": preds[i]
                        })
                    st.markdown("#### Sample Predictions Inspection")
                    st.dataframe(pd.DataFrame(bench_table), use_container_width=True)
        else:
            st.info("Processed test dataset not found. Run `python src/translation_preprocessing.py` to generate test splits.")


# =========================================================
# TAB 5: DATASET EXPLORER
# =========================================================
with tabs[4]:
    st.header("📚 Sanskrit Corpus & Dataset Explorer")
    st.write("Explore the integrated datasets used for OCR character training and parallel translation.")

    ds_choice = st.selectbox("Choose Dataset to Explore:", [
        "Bhagavad Gita Verses (Shlokas & Meanings)",
        "Vedic Lexicon (Terms & English Definitions)",
        "Devanagari Handwritten Character Gallery (DHCD 46 Classes)",
        "Itihasa Parallel Corpus Split Statistics"
    ])

    if ds_choice.startswith("Bhagavad"):
        if not sample_data["gita"].empty:
            search_gita = st.text_input("Filter Gita verses by keyword (Sanskrit or English):", "")
            df = sample_data["gita"]
            if search_gita:
                df = df[df["Shloka"].str.contains(search_gita, na=False) | df["EngMeaning"].str.contains(search_gita, case=False, na=False)]
            st.dataframe(df[["Shloka", "EngMeaning"]].head(50), use_container_width=True)
        else:
            st.info("Bhagavad Gita dataset file not loaded.")

    elif ds_choice.startswith("Vedic"):
        if not sample_data["vedic"].empty:
            search_vedic = st.text_input("Filter Vedic terms by keyword:", "")
            vdf = sample_data["vedic"]
            if search_vedic:
                vdf = vdf[vdf["nagari"].str.contains(search_vedic, na=False) | vdf["description"].str.contains(search_vedic, case=False, na=False)]
            st.dataframe(vdf[["nagari", "word", "category", "description"]].head(50), use_container_width=True)
        else:
            st.info("Vedic Lexicon dataset file not loaded.")

    elif ds_choice.startswith("Devanagari"):
        st.markdown("#### 🕉️ DHCD 46 Devanagari Classes (36 Consonants/Conjuncts + 10 Digits)")
        char_rows = []
        for idx in sorted(DHCD_MAP.keys()):
            info = DHCD_MAP[idx]
            char_rows.append({
                "Class Index": idx,
                "Devanagari Glyph": info["devanagari"],
                "Transliteration": info["label"],
                "Folder Identifier": info["name"]
            })
        st.dataframe(pd.DataFrame(char_rows), use_container_width=True)

    else:
        st.markdown("#### 📈 Integrated Parallel Translation Corpus Statistics")
        st.markdown("""
        | Corpus Name | Source | Train Pairs | Validation Pairs | Test Pairs |
        | :--- | :--- | :--- | :--- | :--- |
        | **Itihasa Shlokas** | Classical Epics (Mahabharata, Ramayana) | 74,882 | 6,137 | 11,696 |
        | **Bhagavad Gita** | Chapter 1 to 18 Complete Shlokas | 560 | 70 | 71 |
        | **Vedic Lexicon** | Ancient Vedic Terms & Roots | 1,006 | 126 | 126 |
        | **Total Combined** | Unified Sanskrit-English Corpus | **76,448** | **6,333** | **11,893** |
        """)


# =========================================================
# TAB 6: ABOUT
# =========================================================
with tabs[5]:
    st.header("ℹ️ About Sanskrit Vision")
    st.markdown("""
    ### 🕉️ Systematic Framework for Sanskrit Character Recognition & Translation
    **Sanskrit Vision** is an AI system dedicated to preserving and digitizing ancient Sanskrit manuscripts, shlokas, and historical texts.

    #### 🧠 Technology Stack
    - **UI & Visualization**: Streamlit, Matplotlib, PIL
    - **Image Processing**: OpenCV (Otsu binarization, Adaptive thresholding, CLAHE, Shirorekha detection & zone extraction)
    - **OCR**: PaddleOCR Devanagari (`PP-OCRv5`)
    - **Machine Translation**: Hybrid Corpus Dictionary + `deep-translator`
    - **Evaluation Metrics**: `jiwer` (CER, WER), `sacrebleu` (BLEU, chrF++)

    ---
    *Preserving ancient wisdom with modern artificial intelligence.*
    """)
