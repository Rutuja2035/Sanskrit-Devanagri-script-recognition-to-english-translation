# ── Windows PyTorch DLL fix & MKLDNN CPU bypass ─────────────────────────────
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

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
try:
    import paddlex.inference.models.runners.paddle_static.config.blocklists as _bl
    for _m in ("PP-OCRv5_mobile_det", "devanagari_PP-OCRv5_mobile_rec", "PP-OCRv4_mobile_det", "devanagari_PP-OCRv4_mobile_rec"):
        if _m not in _bl.MKLDNN_BLOCKLIST:
            _bl.MKLDNN_BLOCKLIST.append(_m)
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import time
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
# Setup paths
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import our backend modules
from src.preprocessing import clean_image
from src.ocr import perform_ocr
from src.postprocess import clean_ocr_text
from src.translate import translate_sanskrit_to_english

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Page Configuration
st.set_page_config(
    page_title="Sanskrit Vision",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Noto+Serif+Devanagari:wght@400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
        }

        .main-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f59e0b, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            text-align: center;
        }
        .sub-title {
            font-size: 1.15rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 300;
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        }

        .sanskrit-text {
            font-family: 'Noto Serif Devanagari', serif;
            font-size: 1.5rem;
            color: #fcd34d;
            line-height: 1.8;
            padding: 1rem;
            background: rgba(0,0,0,0.25);
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
        }
        .english-text {
            font-size: 1.25rem;
            color: #6ee7b7;
            line-height: 1.6;
            padding: 1rem;
            background: rgba(0,0,0,0.25);
            border-radius: 8px;
            border-left: 4px solid #10b981;
        }
        
        .section-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Header
st.markdown('<div class="main-title">🕉️ Sanskrit Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Multi-Domain Sanskrit OCR & Neural Translation Engine</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3389/3389081.png", width=60)
    st.markdown("### Engine Settings")
    
    latency_mode = st.radio(
        "⚡ Latency / Inference Mode",
        ["Real-Time Fast (1.5s - 2.5s)", "Deep Precision Two-Pass (3s - 4.5s)"],
        index=0,
        help="Real-time mode uses single-pass inference to reduce latency while preserving high accuracy."
    )
    is_fast_mode = "Real-Time" in latency_mode

    domain_profile = st.selectbox(
        "📜 Document Domain Profile",
        [
            "Ancient Manuscript & Palm-Leaf (ताड़पत्र/भूर्जपत्र)",
            "Manual / Handwritten Script (हस्तलिखित)",
            "Digital & Modern Printed (मुद्रित ग्रन्थ)"
        ],
        index=0,
        help="Select your document style for optimal Sauvola thresholding and illumination correction."
    )
    is_manuscript = "Ancient" in domain_profile or "Manual" in domain_profile

    enable_lexical_correction = st.checkbox(
        "🕉️ Sanskrit Lexical & Corpus Auto-Correction",
        value=True,
        help="Applies Paninian orthographic repairs and aligns classical shlokas with verified Sanskrit corpus."
    )

    st.markdown("---")
    st.markdown("### Model Engine Status")
    st.success("🟢 **PP-OCRv5 + Fine-Tuned SanskritCRNN Booster**")

    st.markdown("---")
    use_hf_api = st.checkbox("Enable Hugging Face API", value=False)
    hf_token_input = None
    if use_hf_api:
        hf_token_input = st.text_input("HF API Token", type="password", value=HF_TOKEN if HF_TOKEN else "")
    
    st.markdown("---")
    st.markdown("### Pipeline Architecture")
    st.markdown("1. 🖼️ Sauvola & Illumination Enhancement\n2. 🔍 PP-OCRv5 Devanagari Detection\n3. 🧠 SanskritCRNN Sequence Booster\n4. 🧹 Unicode & Lexical Normalization\n5. 🌐 Hybrid Neural Translation")
    
    st.markdown("---")
    st.caption("Sanskrit Vision • Production Ready")

# PDF Generation Helper
def create_pdf(sanskrit_text: str, english_text: str) -> bytes:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        
        ws_root = Path(__file__).resolve().parents[2]
        font_devanagari = ws_root / "Text" / "sanskrit_ocr_translation" / "src" / "NotoSansDevanagari-Regular.ttf"
        if not font_devanagari.exists():
            font_devanagari = ws_root / "Text-Translation" / "sanskrit_ocr_translation" / "data" / "fonts" / "siddhanta.ttf"

        font_regular = ws_root / "Text" / "sanskrit_ocr_translation" / "src" / "NotoSans-Regular.ttf"
        
        if font_devanagari.exists():
            pdf.add_font("NotoSansDevanagari", style="", fname=str(font_devanagari))
            dev_font = "NotoSansDevanagari"
        else:
            dev_font = "helvetica"
            
        if font_regular.exists():
            pdf.add_font("NotoSans", style="", fname=str(font_regular))
            reg_font = "NotoSans"
        else:
            reg_font = "helvetica"

        pdf.set_font(reg_font, size=18)
        pdf.cell(0, 10, "Sanskrit Vision - OCR & Translation Report", 0, 1, 'C')
        pdf.ln(5)

        pdf.set_font(reg_font, size=14)
        pdf.cell(0, 10, "1. Extracted Sanskrit Text (Devanagari):", 0, 1, 'L')
        pdf.set_font(dev_font, size=12)
        pdf.multi_cell(0, 8, sanskrit_text)
        pdf.ln(8)

        pdf.set_font(reg_font, size=14)
        pdf.cell(0, 10, "2. English Translation:", 0, 1, 'L')
        pdf.set_font(reg_font, size=12)
        pdf.multi_cell(0, 8, english_text)

        return bytes(pdf.output())
    except Exception:
        # Fallback simple text-based export if fpdf font error
        return f"Sanskrit Vision Report\n\nSanskrit Text:\n{sanskrit_text}\n\nEnglish Translation:\n{english_text}".encode("utf-8")

# Main Content Area
st.markdown("### 📄 Document Ingestion")
uploaded_file = st.file_uploader("Upload Sanskrit Document (Palm-leaf, handwritten, or digital print)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📷 Original Document</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded File")
        
    with col2:
        prep_start = time.time()
        with st.spinner("Applying Sauvola Binarization & Enhancement..."):
            processed_image = clean_image(image, is_manuscript=is_manuscript)
        prep_duration = time.time() - prep_start
        st.markdown('<div class="section-header">⚙️ Preprocessed Image</div>', unsafe_allow_html=True)
        st.image(processed_image, use_container_width=True, caption=f"Enhanced ({prep_duration:.2f}s)")

    st.markdown("---")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_button = st.button("🚀 Run OCR & Translation", type="primary", use_container_width=True)

    if run_button or st.session_state.get('auto_run', False):
        st.session_state['auto_run'] = True

        # OCR Execution
        st.markdown('<div class="section-header">🔍 Sanskrit Recognition Output</div>', unsafe_allow_html=True)
        ocr_start = time.time()
        with st.spinner("Recognizing Devanagari script via PP-OCRv5 + SanskritCRNN..."):
            raw_text = perform_ocr(
                image,
                high_accuracy_mode=(not is_fast_mode),
                enable_corpus_alignment=enable_lexical_correction
            )
            if not raw_text.strip() and processed_image is not None:
                raw_text = perform_ocr(
                    processed_image,
                    high_accuracy_mode=(not is_fast_mode),
                    enable_corpus_alignment=enable_lexical_correction
                )
            cleaned_text = clean_ocr_text(raw_text)
        ocr_duration = time.time() - ocr_start

        st.markdown(f'<div class="glass-card"><div class="sanskrit-text">{cleaned_text if cleaned_text else "No Sanskrit text detected. Please inspect image quality or select a different domain profile."}</div></div>', unsafe_allow_html=True)

        # Translation Execution
        st.markdown('<div class="section-header">🌐 English Translation</div>', unsafe_allow_html=True)
        trans_start = time.time()
        with st.spinner("Translating via Sanskrit Hybrid MT..."):
            effective_token = hf_token_input if use_hf_api and hf_token_input else None
            translation = translate_sanskrit_to_english(cleaned_text, hf_token=effective_token)
        trans_duration = time.time() - trans_start

        st.markdown(f'<div class="glass-card"><div class="english-text">{translation if translation else "Translation unavailable."}</div></div>', unsafe_allow_html=True)

        # Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        total_time = prep_duration + ocr_duration + trans_duration
        m1.metric("⚡ Total Pipeline Time", f"{total_time:.2f}s")
        m2.metric("🔍 OCR Latency", f"{ocr_duration:.2f}s")
        m3.metric("🌐 Translation Latency", f"{trans_duration:.2f}s")
        m4.metric("📄 Words Extracted", f"{len(cleaned_text.split())} Words")

        # Export Options
        st.markdown("### 💾 Export Results")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            st.download_button(
                label="📥 Download Sanskrit (.txt)",
                data=cleaned_text,
                file_name="extracted_sanskrit.txt",
                mime="text/plain",
                use_container_width=True
            )
        with exp_col2:
            st.download_button(
                label="📥 Download Translation (.txt)",
                data=translation,
                file_name="english_translation.txt",
                mime="text/plain",
                use_container_width=True
            )
        with exp_col3:
            pdf_bytes = create_pdf(cleaned_text, translation)
            st.download_button(
                label="📑 Export Document (.pdf)",
                data=pdf_bytes,
                file_name="sanskrit_vision_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    st.info("👆 Please upload a Sanskrit document to run the pipeline.")
    
    st.markdown("""
        <div style="opacity: 0.75; margin-top: 1rem;">
            <div class="glass-card">
                <div class="section-header">💡 Quick Sample Testing</div>
                <p>You can test with built-in test manuscripts located at: <code>Text-Translation/sanskrit_ocr_translation/data/test_evaluation/</code></p>
                <ul>
                    <li><b>manuscript_1.jpg</b> — Ancient palm-leaf / paper manuscript</li>
                    <li><b>digital_1.jpg</b> — Digital printed Devanagari verse</li>
                    <li><b>handwritten_plain_1.jpg</b> — Handwritten Sanskrit text</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
