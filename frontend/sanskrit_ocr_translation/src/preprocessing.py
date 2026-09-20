import time
from PIL import Image, ImageOps

def clean_image(image: Image.Image) -> Image.Image:
    """
    Mock function for image preprocessing.
    In the real implementation, this will perform grayscale conversion, 
    denoising, thresholding, etc. (Member 2's responsibility)
    """
    time.sleep(1) # Simulate processing time
    # For the mock, just convert to grayscale to show a difference
    return ImageOps.grayscale(image)
