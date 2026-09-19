import os
import requests
from typing import List

class TranslatorAPI:
    """
    A simple wrapper for the Hugging Face Inference API for machine translation.
    Defaults to NLLB-200 distilled 600M, which is good for Indic translation.
    """
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M"):
        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.hf_token = os.environ.get("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        if not self.hf_token:
            print("Warning: HF_TOKEN not set. Translation API may be rate-limited.")

    def translate_sentences(self, sentences: List[str], src_lang: str = "san_Deva", tgt_lang: str = "eng_Latn") -> List[str]:
        """
        Translate a list of sentences using the Hugging Face API.
        """
        translated = []
        for sentence in sentences:
            if not sentence.strip():
                translated.append("")
                continue
                
            payload = {
                "inputs": sentence,
                "parameters": {"src_lang": src_lang, "tgt_lang": tgt_lang}
            }
            
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and "translation_text" in result[0]:
                        translated.append(result[0]["translation_text"])
                    else:
                        translated.append(f"[API Format Error: {result}]")
                else:
                    translated.append(f"[API Error HTTP {response.status_code}: {response.text}]")
            except Exception as e:
                translated.append(f"[Request Failed: {str(e)}]")
                
        return translated

if __name__ == "__main__":
    # Quick test
    translator = TranslatorAPI()
    print(translator.translate_sentences(["धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।"]))
