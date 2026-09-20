import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv

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
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Noto+Serif+Devanagari:wght@400;700&display=swap');
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
        }

        /* Title Styling */
        .main-title {
            font-size: 3rem;
            font-weight: 600;
            background: linear-gradient(135deg, #f59e0b, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        }

        /* Typography */
        .sanskrit-text {
            font-family: 'Noto Serif Devanagari', serif;
            font-size: 1.5rem;
            color: #fcd34d;
            line-height: 1.8;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
        }
        .english-text {
            font-size: 1.25rem;
            color: #6ee7b7;
            line-height: 1.6;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            border-left: 4px solid #10b981;
        }
        
        .section-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Sidebar styling override */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Header
st.markdown('<div class="main-title">Sanskrit Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Multi-Domain Sanskrit OCR & Translation Engine</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3389/3389081.png", width=60) # Placeholder icon
    st.markdown("### Settings")
    st.markdown("Configure the engine parameters.")
    
    use_hf_api = st.checkbox("Use Hugging Face API", value=True)
    if use_hf_api:
        hf_token_input = st.text_input("HF API Token", type="password", value=HF_TOKEN if HF_TOKEN else "")
    
    st.markdown("---")
    st.markdown("### Engine Layers")
    st.markdown("1. Ingestion & Preprocessing\n2. OCR Detection\n3. Sequence Recognition\n4. Text Cleaning\n5. Neural Translation")
    
    st.markdown("---")
    st.markdown("**Member 4 Deliverable**\n*UI & Integration*")

# Main Content Area
st.markdown("### Upload Manuscript")
uploaded_file = st.file_uploader("Choose a Sanskrit image (Palm-leaf, manual, or digital typography)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Image Upload
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📷 Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True, caption="Uploaded File")
        
    with col2:
        # 2. Preprocessing
        with st.spinner("Applying Sauvola Binarization & Despeckling..."):
            processed_image = clean_image(image)
        st.markdown('<div class="section-header">⚙️ Preprocessed Image</div>', unsafe_allow_html=True)
        st.image(processed_image, use_column_width=True, caption="Cleaned & Ready for OCR")

    st.markdown("---")

    # 3. OCR and Post-processing
    st.markdown('<div class="section-header">🔍 OCR Extraction</div>', unsafe_allow_html=True)
    with st.spinner("Extracting Devanagari text using DBNet & CRNN..."):
        raw_text = perform_ocr(processed_image)
        cleaned_text = clean_ocr_text(raw_text)
    
    st.markdown(f'<div class="glass-card"><div class="sanskrit-text">{cleaned_text}</div></div>', unsafe_allow_html=True)

    # 4. Translation
    st.markdown('<div class="section-header">🌐 English Translation</div>', unsafe_allow_html=True)
    with st.spinner("Translating via Neural Machine Translation..."):
        translation = translate_sanskrit_to_english(cleaned_text, hf_token=HF_TOKEN if use_hf_api else None)
    
    st.markdown(f'<div class="glass-card"><div class="english-text">{translation}</div></div>', unsafe_allow_html=True)

    # 5. Export / Download
    st.markdown("### Export Results")
    col3, col4 = st.columns(2)
    
    with col3:
        st.download_button(
            label="📥 Download Sanskrit Text",
            data=cleaned_text,
            file_name="extracted_sanskrit.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col4:
        st.download_button(
            label="📥 Download Translation",
            data=translation,
            file_name="english_translation.txt",
            mime="text/plain",
            use_container_width=True
        )
else:
    # Placeholder state when no image is uploaded
    st.info("👆 Please upload an image to start the Sanskrit Vision pipeline.")
    
    # Show a demo layout
    st.markdown("""
        <div style="opacity: 0.5;">
            <h4>Pipeline Preview</h4>
            <div class="glass-card" style="margin-bottom: 1rem;">
                <div class="section-header">1. Upload & Preprocess</div>
                <p>Enhances contrast and removes noise from historical documents.</p>
            </div>
            <div class="glass-card" style="margin-bottom: 1rem;">
                <div class="section-header">2. Text Extraction</div>
                <p>Accurately reads Devanagari script using advanced OCR.</p>
            </div>
            <div class="glass-card">
                <div class="section-header">3. Neural Translation</div>
                <p>Translates classical Sanskrit structures into English contextually.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
