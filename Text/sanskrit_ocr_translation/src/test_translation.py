import os
import requests
import sys

def main():
    # Ensure stdout handles utf-8 properly
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Due to strict local system security policies blocking 'numpy', we are using sample texts directly.")
    examples = [
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।",
        "मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    ]
    targets = [
        "O Sanjaya, what did my sons and the sons of Pandu do, when they gathered on the sacred field of Kurukshetra, eager for battle?",
        "What did they do, O Sanjaya?"
    ]

    # Hugging Face Inference API details
    # We use NLLB-200 because IndicTrans2 requires custom code and may not run on the free serverless Inference API.
    model_id = "facebook/nllb-200-distilled-600M"
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    # Try to get token from env
    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    print(f"\nTranslating using {model_id} via Hugging Face Inference API...")
    if not hf_token:
        print("Note: No HF_TOKEN environment variable found. API might be rate-limited.")
        print("If you encounter a rate limit error, you can set your token via 'set HF_TOKEN=your_token'.\n")
    
    print("=== Translation Results ===\n")
    for i in range(len(examples)):
        payload = {
            "inputs": examples[i],
            "parameters": {"src_lang": "san_Deva", "tgt_lang": "eng_Latn"}
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            result = response.json()
            if isinstance(result, list) and "translation_text" in result[0]:
                predicted = result[0]["translation_text"]
            else:
                predicted = f"API Error: {result}"
        except Exception as e:
            predicted = f"Request failed: {e}"
            
        print(f"--- Example {i+1} ---")
        print(f"Source (Sanskrit): {examples[i]}")
        print(f"Target (English) : {targets[i]}")
        print(f"Predicted        : {predicted}\n")

if __name__ == "__main__":
    main()
