import time
from PIL import Image

def perform_ocr(image: Image.Image) -> str:
    """
    Mock function for OCR.
    In the real implementation, this will extract text using EasyOCR/PaddleOCR. 
    (Member 2's responsibility)
    """
    time.sleep(1.5) # Simulate processing time
    return "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।\nमामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
