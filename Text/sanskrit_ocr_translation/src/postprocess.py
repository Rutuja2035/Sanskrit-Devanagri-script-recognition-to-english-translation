import unicodedata
import re

def clean_ocr_text(raw_text: str) -> list[str]:
    """
    Cleans raw OCR output by normalizing Unicode, removing noise,
    fixing whitespace, and splitting into sentences.
    """
    # 1. Normalize Unicode (critical for Devanagari)
    text = unicodedata.normalize("NFC", raw_text)
    
    # 2. Remove stray OCR noise characters
    # Keep Devanagari block (\u0900-\u097F), whitespace (\s), danda (।), and double danda (॥)
    # Note: danda is \u0964 and double danda is \u0965, which are inside the Devanagari block,
    # but we explicitly keep spaces and standard punctuation if needed.
    # The regex below removes anything that is NOT Devanagari or whitespace.
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    
    # 3. Fix broken/extra whitespace around matras and conjuncts
    # Remove space before Devanagari vowel signs (matras) and virama (halant)
    text = re.sub(r'\s+([\u093E-\u094D\u0962\u0963])', r'\1', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Split text into sentences using danda marks as delimiters
    # We split by one or more dandas/double dandas.
    sentences = re.split(r'[\u0964\u0965]+', text)
    
    # Clean up empty strings from the split
    clean_sentences = [s.strip() for s in sentences if s.strip()]
    
    return clean_sentences

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Test block with sample noisy OCR strings
    sample_text_1 = "धर्मक्षेत्रे   कु  रुक्षेत्रे समवेता युयुत्सवः । मामकाः पा ण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    sample_text_2 = "यदा यदा हि ध र्मस्य ग्लानिर्भवति भा रत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥"
    
    print("Testing clean_ocr_text:")
    print("--- Sample 1 ---")
    print(f"Raw: {sample_text_1}")
    print(f"Cleaned: {clean_ocr_text(sample_text_1)}")
    
    print("\n--- Sample 2 ---")
    print(f"Raw: {sample_text_2}")
    print(f"Cleaned: {clean_ocr_text(sample_text_2)}")
