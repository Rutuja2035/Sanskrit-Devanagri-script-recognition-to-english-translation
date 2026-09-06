import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to single-channel grayscale.
    Handles images that are already grayscale or have an alpha channel.
    
    Args:
        image (np.ndarray): The input image.
        
    Returns:
        np.ndarray: Grayscale image.
    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return image

def denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply fast non-local means denoising to the image.
    
    Args:
        image (np.ndarray): The input image (grayscale or color).
        
    Returns:
        np.ndarray: Denoised image.
    """
    if len(image.shape) == 3 and image.shape[2] in (3, 4):
        # fastNlMeansDenoisingColored expects BGR or similar, but works for RGB just as well 
        # in terms of separating luma/chroma conceptually.
        if image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image
        return cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
    Expects and returns a grayscale image.
    
    Args:
        image (np.ndarray): The input image.
        
    Returns:
        np.ndarray: Contrast enhanced grayscale image.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_basic(image: np.ndarray) -> np.ndarray:
    """
    Basic preprocessing pipeline: Grayscale -> Denoise.
    
    Args:
        image (np.ndarray): The input image.
        
    Returns:
        np.ndarray: Basic preprocessed grayscale image.
    """
    gray = to_grayscale(image)
    return denoise(gray)

def preprocess_adaptive(image: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline: Grayscale -> Denoise -> Adaptive Gaussian Thresholding.
    
    Args:
        image (np.ndarray): The input image.
        
    Returns:
        np.ndarray: Binarized image using adaptive Gaussian thresholding.
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

def preprocess_otsu(image: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline: Grayscale -> Denoise -> Otsu's Binarization.
    
    Args:
        image (np.ndarray): The input image.
        
    Returns:
        np.ndarray: Binarized image using Otsu's method.
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)
    _, binarized = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binarized


def despeckle_image(binary_image: np.ndarray, min_size: int = 5, max_aspect_ratio: float = 12.0) -> np.ndarray:
    """
    Remove small pepper/speckle noise, wormholes, and isolated ink dots.
    
    Args:
        binary_image (np.ndarray): 2D uint8 binary image (0/255).
        min_size (int): Minimum pixel area of a legitimate stroke component.
        max_aspect_ratio (float): Maximum bounding aspect ratio to filter stray linear scratches.
        
    Returns:
        np.ndarray: Cleaned binary image without stray speckle noise.
    """
    if binary_image.ndim != 2:
        return binary_image

    # Ensure foreground is white (255) for connected components
    is_inverted = np.mean(binary_image) > 127
    target = cv2.bitwise_not(binary_image) if is_inverted else binary_image.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(target, connectivity=8)
    clean = np.zeros_like(target)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w / max(1, h), h / max(1, w))

        # Filter out tiny specks (< min_size) or extreme scratch noise (> max_aspect_ratio)
        if area >= min_size and aspect_ratio <= max_aspect_ratio:
            clean[labels == i] = 255

    return cv2.bitwise_not(clean) if is_inverted else clean


def preprocess_sauvola(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    despeckle: bool = True
) -> np.ndarray:
    """
    Preprocessing pipeline: Grayscale -> Denoise -> Sauvola Local Adaptive Thresholding.
    Excels at damaged, stained, and unevenly aged historical folios where global Otsu fails.
    
    Args:
        image (np.ndarray): The input image.
        window_size (int): Local neighborhood window size (must be odd).
        k (float): Sauvola parameter controlling dynamic threshold sensitivity.
        despeckle (bool): Whether to remove isolated speckle/wormhole noise.
        
    Returns:
        np.ndarray: Clean binarized image (0 and 255).
    """
    gray = to_grayscale(image)
    denoised = denoise(gray)

    try:
        from skimage.filters import threshold_sauvola
        # Window size must be odd
        win = window_size if window_size % 2 == 1 else window_size + 1
        thresh = threshold_sauvola(denoised, window_size=win, k=k)
        binary = ((denoised > thresh) * 255).astype(np.uint8)
    except Exception:
        # Graceful fallback to Gaussian adaptive if skimage unavailable
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
        )

    if despeckle:
        binary = despeckle_image(binary)

    return binary



def suppress_document_borders(image: np.ndarray, border_ratio: float = 0.035) -> np.ndarray:
    """
    Detect and neutralize outer framing lines, binding borders, and scanner margins.
    Fills outer border artifacts with the document's median paper background color
    to prevent OCR from hallucinating phantom initial/terminal characters like 'व', 'न', 'क'.
    
    Args:
        image (np.ndarray): The input document image (RGB, BGR, or grayscale).
        border_ratio (float): Approximate margin thickness ratio (default ~3.5% of dimension).
        
    Returns:
        np.ndarray: Cleaned image with outer borders suppressed.
    """
    out = image.copy()
    h, w = out.shape[:2]

    # Calculate median background color from the inner text area
    inner_y1, inner_y2 = max(0, int(h * 0.1)), min(h, int(h * 0.9))
    inner_x1, inner_x2 = max(0, int(w * 0.1)), min(w, int(w * 0.9))
    inner_crop = out[inner_y1:inner_y2, inner_x1:inner_x2]

    if out.ndim == 3:
        bg_color = np.median(inner_crop, axis=(0, 1)).astype(np.uint8).tolist()
    else:
        bg_color = int(np.median(inner_crop))

    # Margin widths
    pad_w = max(10, min(35, int(w * border_ratio)))
    pad_h = max(6, min(20, int(h * (border_ratio * 0.7))))

    # Mask outer borders with neutral background
    out[:, :pad_w] = bg_color
    out[:, -pad_w:] = bg_color
    out[:pad_h, :] = bg_color
    out[-pad_h:, :] = bg_color

    return out


def preprocess_manuscript(
    image: np.ndarray,
    suppress_borders: bool = True,
    auto_scale: bool = True,
) -> np.ndarray:
    """
    Advanced restoration pipeline tailored for ancient Sanskrit manuscripts,
    palm-leaf (ताड़पत्र) and birch-bark (भूर्जपत्र) documents.
    
    Pipeline:
    1. Outer border suppression to eliminate false margin characters.
    2. Adaptive super-resolution upscaling for low-DPI / dense manuscripts.
    3. Illumination normalization via morphological background division.
    4. Bilateral smoothing to remove fibrous papyrus grain.
    5. CLAHE + Unsharp masking for crisp, high-contrast character strokes.
    
    Args:
        image (np.ndarray): The input manuscript image array.
        suppress_borders (bool): Whether to mask out outer border framing lines.
        auto_scale (bool): Whether to upscale low-resolution scans (height < 600px).
        
    Returns:
        np.ndarray: Restored high-contrast grayscale image suitable for OCR.
    """
    processed = image
    if suppress_borders:
        processed = suppress_document_borders(processed)

    gray = to_grayscale(processed)
    h, w = gray.shape[:2]

    # 1. Super-resolution upscaling if textlines are small/dense (e.g. height < 600px)
    if auto_scale and h < 600:
        scale_factor = 1.5 if h >= 400 else 1.8
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Background illumination normalization
    # Estimate background with large morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Division normalization to flatten background discoloration & tea stains
    normalized = cv2.divide(gray, background, scale=255.0)

    # 3. Bilateral filter to suppress palm-leaf fibers while preserving ink boundaries
    smoothed = cv2.bilateralFilter(normalized.astype(np.uint8), d=5, sigmaColor=40, sigmaSpace=40)

    # 4. CLAHE contrast boost for faded ancient carbon ink
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    # 5. Gentle unsharp masking to sharpen character stroke edges
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)

    return sharpened
