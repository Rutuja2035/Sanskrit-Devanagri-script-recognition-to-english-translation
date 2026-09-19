import sys
import argparse
from postprocess import clean_ocr_text
from translate import TranslatorAPI

def main():
    # Ensure UTF-8 output for console
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Clean Sanskrit OCR text and translate to English.")
    parser.add_argument("--text", type=str, help="Raw OCR text string to process.", default=None)
    parser.add_argument("--file", type=str, help="Path to a file containing raw OCR text.", default=None)
    
    args = parser.parse_args()
    
    raw_text = ""
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            sys.exit(1)
    elif args.text:
        raw_text = args.text
    else:
        # Fallback to a default noisy OCR example
        print("No input provided. Using default sample text.")
        raw_text = "धर्मक्षेत्रे   कु  रुक्षेत्रे समवेता युयुत्सवः । मामकाः पा ण्डवाश्चैव किमकुर्वत सञ्जय ॥"
        
    print(f"\n--- 1. Raw Input Text ---")
    print(raw_text)
    
    # Clean and split the OCR text
    print(f"\n--- 2. Cleaning and Segmenting OCR Text ---")
    sentences = clean_ocr_text(raw_text)
    
    if not sentences:
        print("No valid sentences found after cleaning.")
        return
        
    for i, s in enumerate(sentences):
        print(f"[{i+1}] {s}")
        
    # Translate
    print(f"\n--- 3. Translating to English ---")
    translator = TranslatorAPI()
    
    translations = translator.translate_sentences(sentences, src_lang="san_Deva", tgt_lang="eng_Latn")
    
    print(f"\n--- 4. Final Output ---")
    for i in range(len(sentences)):
        print(f"Sanskrit : {sentences[i]}")
        print(f"English  : {translations[i]}\n")

if __name__ == "__main__":
    main()
