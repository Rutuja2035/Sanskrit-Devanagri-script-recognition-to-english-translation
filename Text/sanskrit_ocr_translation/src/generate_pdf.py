import os
from fpdf import FPDF
from translate import TranslatorAPI
from postprocess import clean_ocr_text

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("NotoSansDevanagari", style="", fname="NotoSansDevanagari-Regular.ttf", uni=True)
        self.add_font("NotoSans", style="", fname="NotoSans-Regular.ttf", uni=True)

    def chapter_title(self, title):
        self.set_font("NotoSans", size=16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, text, font_family, size=12):
        self.set_font(font_family, size=size)
        self.multi_cell(0, 8, text)
        self.ln(10)

def main():
    # Hardcoded transcribed text from the images provided by the user
    images_text = {
        "Image 1": "निरन्तरान्धकारित-दिगन्तर-कन्दलदमन्द-सुधारस-बिन्दु-सान्द्रतर-घनाघन-वृन्द-सन्देहकर-स्यन्दमान-मकरन्द-बिन्दु-बन्धुरतर-माकन्द-तरु-कुल-तल्प-कल्प-मृदुल-सिकता-जाल-जटिल-मूल-तल-मरुवक-मिलदलघु-लघु-लय-कलित-रमणीय-पानीय-शालिका-बालिका-करार-विन्द-गलन्तिका-गलदेला-लवङ्ग-पाटल-घनसार-कस्तूरिकातिसौरभ-मेदुर-लघुतर-मधुर-शीतलतर-सलिलधारा-निराकरिष्णु-तदीय-विमल-विलोचन-मयू",
        "Image 2": "कश्चिद्धरिद्रो धीवरश्चिरं रात्रौ कस्यचिद्दीपस्य समीपे नौकया मत्स्यान्वेषी भ्रामि। तेन बहुद्योगं कृत्वापि जालानि वारिषु बहुवारं क्षिप्त्वापि केवलं कां-श्चित्क्षुद्रमत्स्यान्गृहीत्वा द्वीपं त्यक्त्वा नदीतीरं प्राप्य क्षुद्रवृक्षशाखां गृहीत्वा नौका तत्र दृढं बद्धा। जालानि नदीतीरे स्थापयित्वा मत्स्यांस्तृणानि च पिटके निक्षिपति। ततः पिटकं यष्टिं चादाय तीरमारुह्येवं च प्रभाते ग्रामं शीघ्रं प्रचलति। तत्र क्षेत्रपतिवाणिक्प्रतिवासिभृत्यजनसमूहं प्रविश्य मत्स्या मत्स्या इत्युच्चैः क्रोशति ॥ १ ॥",
        "Image 3": "वनसङ्घ... हन... काव्यकथा... कवीश्वरा...",
        "Image 4": "श्रीगणेशायनमः॥ ॥ यस्य स्मरणमात्रेण जन्म संसार बंधनात्॥ विमुच्यते नमस्त स्मै विष्णवे प्रभविष्णवे॥ १॥ वैशंपायन उवाच श्रुत्वा धर्मानशेषेण पावनानि च सर्वशः॥ युधिष्ठिरः शांतनवं पुनरेवाभ्यभाषत॥ २॥ युधिष्ठिर उवाच॥ किमेकं दैवतं लोके किं वा प्येकं परायणं॥ स्तुवंतः कं कमर्चंतः प्राप्नुयुर्मानवाः शुभं॥ ३॥ को धर्मः सर्वधर्माणां भवतः परमो मतः॥ किं जपन्मुच्यते जंतुर्जन्मसंसारबंध"
    }

    pdf = PDF()
    pdf.add_page()
    translator = TranslatorAPI()

    for img_name, raw_text in images_text.items():
        print(f"Processing {img_name}...")
        pdf.chapter_title(img_name)
        
        sentences = clean_ocr_text(raw_text)
        
        pdf.set_font("NotoSans", size=14)
        pdf.cell(0, 10, "Original Sanskrit Text:", 0, 1, 'L')
        
        full_original = "\n".join(sentences)
        pdf.chapter_body(full_original, font_family="NotoSansDevanagari")
        
        # WE ARE NOW USING THE USER'S MODEL FROM translate.py
        translations = translator.translate_sentences(sentences, src_lang="san_Deva", tgt_lang="eng_Latn")
        
        pdf.set_font("NotoSans", size=14)
        pdf.cell(0, 10, "English Translation:", 0, 1, 'L')
        
        full_translation = "\n".join(translations)
        pdf.chapter_body(full_translation, font_family="NotoSans")

    output_pdf_path = "translated_images.pdf"
    pdf.output(output_pdf_path)
    print(f"Saved translated text to {output_pdf_path}")

if __name__ == "__main__":
    main()
