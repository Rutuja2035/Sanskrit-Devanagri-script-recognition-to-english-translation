# Sanskrit Vision: OCR & Translation

This is the interactive Streamlit application for the Sanskrit Devanagari OCR and Translation project. 

## Features
- Upload Sanskrit manuscript images
- Preprocess the images to remove noise
- Extract Devanagari text using OCR
- Clean and normalize the extracted text
- Translate the text to English
- Download the results

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Copy `.env.example` to `.env` and configure your `HF_TOKEN`.
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
