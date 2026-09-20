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
    Detect and neutralize outer dark framing lines, scanner edge shadows, and binding margins.
    Only neutralizes borders when genuine dark framing artifacts or scanner borders are detected,
    ensuring that valid text characters near margins on cropped lines are never erased.
    
    Args:
        image (np.ndarray): The input document image (RGB, BGR, or grayscale).
        border_ratio (float): Approximate margin thickness ratio (default ~3.5% of dimension).
        
    Returns:
        np.ndarray: Cleaned image with outer borders suppressed.
    """
    h, w = image.shape[:2]
    # Safety check: do not suppress borders on cropped lines or small character patches
    if h < 200 or w < 200 or (w / max(1, h)) > 4.0:
        return image

    out = image.copy()

    # Calculate median background color from the inner text area
    inner_y1, inner_y2 = max(0, int(h * 0.15)), min(h, int(h * 0.85))
    inner_x1, inner_x2 = max(0, int(w * 0.15)), min(w, int(w * 0.85))
    inner_crop = out[inner_y1:inner_y2, inner_x1:inner_x2]

    if inner_crop.size == 0:
        return image

    if out.ndim == 3:
        bg_color = np.median(inner_crop, axis=(0, 1)).astype(np.uint8).tolist()
        inner_intensity = float(np.mean(inner_crop))
    else:
        bg_color = int(np.median(inner_crop))
        inner_intensity = float(np.median(inner_crop))

    # Margin widths
    pad_w = max(6, min(24, int(w * border_ratio)))
    pad_h = max(6, min(18, int(h * (border_ratio * 0.7))))

    # Only mask outer borders if the edge region is substantially darker than inner page background
    gray = to_grayscale(out)
    left_edge_mean = float(np.mean(gray[:, :pad_w]))
    right_edge_mean = float(np.mean(gray[:, -pad_w:]))
    top_edge_mean = float(np.mean(gray[:pad_h, :]))
    bottom_edge_mean = float(np.mean(gray[-pad_h:, :]))

    edge_thresh = inner_intensity - 45.0

    if left_edge_mean < edge_thresh:
        out[:, :pad_w] = bg_color
    if right_edge_mean < edge_thresh:
        out[:, -pad_w:] = bg_color
    if top_edge_mean < edge_thresh:
        out[:pad_h, :] = bg_color
    if bottom_edge_mean < edge_thresh:
        out[-pad_h:, :] = bg_color

    return out


def deskew_text_lines(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Detect and correct slight skew/rotation in handwritten and manuscript pages.
    
    Args:
        image: Input grayscale or color image.
        max_angle: Maximum allowable angle to correct (default ±15 degrees).
        
    Returns:
        np.ndarray: Deskewed image.
    """
    gray = to_grayscale(image)
    h, w = gray.shape[:2]
    
    # Fast edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=max(50, int(w * 0.2)), maxLineGap=15)
    
    if lines is None or len(lines) == 0:
        return image
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx != 0:
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) <= max_angle:
                angles.append(angle)
                
    if not angles:
        return image
        
    # Median angle is resilient to outliers
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.4:  # Negligible skew
        return image
        
    # Rotate around center
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    bg_val = int(np.median(gray))
    deskewed = cv2.warpAffine(
        image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(bg_val, bg_val, bg_val) if image.ndim == 3 else bg_val
    )
    return deskewed


def suppress_notebook_ruled_lines(image: np.ndarray) -> np.ndarray:
    """
    Detect and neutralize horizontal notebook ruled lines (lined paper)
    while strictly preserving Sanskrit character stems, shirorekha, and descenders.
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        return image

    h, w = image.shape[:2]
    if w < 100 or h < 50:
        return image

    out = image.copy()
    # 1. Color-based ruled line detection (blue or red notebook lines)
    if image.ndim == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(out)
        blue_diff = cv2.subtract(b, r)
        _, blue_mask = cv2.threshold(blue_diff, 28, 255, cv2.THRESH_BINARY)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 20), 1))
        blue_lines = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, h_kernel)
        if np.count_nonzero(blue_lines) > w * 2:
            v_dilate = cv2.dilate(blue_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
            out = cv2.inpaint(out, v_dilate, 3, cv2.INPAINT_TELEA)
            return out

    # 2. Grayscale morphological ruled line detection
    gray = to_grayscale(out)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Ruled lines span wide horizontal extents across the page (> 40% page width)
    line_min_w = max(40, int(w * 0.40))
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_w, 1))
    detected_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, line_kernel)

    # Protect vertical character strokes (do not sever consonant stems)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    vertical_strokes = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel)
    pure_lines = cv2.subtract(detected_lines, cv2.dilate(vertical_strokes, np.ones((3, 3), np.uint8)))

    if np.count_nonzero(pure_lines) > w:
        bg_val = int(np.median(gray))
        pure_lines_dil = cv2.dilate(pure_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)))
        if out.ndim == 3:
            out[pure_lines_dil > 0] = (bg_val, bg_val, bg_val)
        else:
            out[pure_lines_dil > 0] = bg_val

    return out


def preprocess_handwritten(
    image: np.ndarray,
    suppress_borders: bool = False,
    suppress_ruled_lines: bool = True,
    auto_scale: bool = True,
    deskew: bool = True,
) -> np.ndarray:
    """
    Dedicated restoration pipeline for manual and handwritten Sanskrit scripts (हस्तलिखित).
    
    Unlike harsh 1-bit binarization (which destroys stroke antialiasing), this
    retains full 8-bit grayscale stroke gradients for deep neural network extraction
    while eliminating notebook grids, ink blotches, and pen-pressure variations.
    
    Pipeline:
    1. Optional border suppression.
    2. Notebook ruled line neutralization.
    3. Document deskewing for natural horizontal shirorekha alignment.
    4. Background flattening via morphological opening/closing division.
    5. Adaptive bilateral filtering (preserves handwritten ink edges).
    6. Localized CLAHE contrast enhancement for faint strokes.
    7. Stroke healing: subtle morphological closing to connect faint disjointed ligatures.
    
    Args:
        image: Input document image array.
        suppress_borders: Whether to neutralize outer margin noise.
        suppress_ruled_lines: Whether to detect and remove notebook horizontal ruled lines.
        auto_scale: Whether to scale low-resolution lines to optimal neural net height.
        deskew: Whether to correct rotational skew.
        
    Returns:
        np.ndarray: Enhanced high-fidelity grayscale image optimized for OCR.
    """
    processed = image
    if suppress_borders:
        processed = suppress_document_borders(processed)
    if suppress_ruled_lines:
        processed = suppress_notebook_ruled_lines(processed)
    if deskew:
        processed = deskew_text_lines(processed)
        
    gray = to_grayscale(processed)
    h, w = gray.shape[:2]
    
    # 1. Optimal neural network line height scaling
    if auto_scale and h < 600:
        scale_factor = 1.5 if h >= 350 else 2.0
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
    # 2. Background illumination normalization
    k_size = max(25, (min(gray.shape[:2]) // 15) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255.0)
    
    # 3. Bilateral smoothing for pen jitter & paper grain
    smoothed = cv2.bilateralFilter(normalized.astype(np.uint8), d=5, sigmaColor=35, sigmaSpace=35)
    
    # 4. Adaptive CLAHE contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    
    # 5. Stroke healing (connect faint pen breaks in shirorekha / ligatures)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    healed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, close_kernel)
    
    # 6. Unsharp masking to sharpen character strokes
    blurred = cv2.GaussianBlur(healed, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(healed, 1.3, blurred, -0.3, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_manuscript(
    image: np.ndarray,
    suppress_borders: bool = False,
    auto_scale: bool = True,
    deskew: bool = True,
) -> np.ndarray:
    """
    Advanced restoration pipeline tailored for ancient Sanskrit manuscripts,
    palm-leaf (ताड़पत्र) and birch-bark (भूर्जपत्र) documents.
    
    Pipeline:
    1. Outer border suppression to eliminate false margin characters.
    2. Rotation deskewing for horizontal baseline alignment.
    3. Adaptive super-resolution upscaling for low-DPI / dense manuscripts.
    4. Multi-scale background illumination division (removes severe tea/water stains & mold).
    5. Bilateral smoothing to remove fibrous papyrus grain without blurring ink.
    6. CLAHE + Unsharp masking for crisp, high-contrast character strokes.
    
    Args:
        image (np.ndarray): The input manuscript image array.
        suppress_borders (bool): Whether to mask out outer border framing lines.
        auto_scale (bool): Whether to upscale low-resolution scans (height < 600px).
        deskew (bool): Whether to correct page skew.
        
    Returns:
        np.ndarray: Restored high-contrast grayscale image suitable for OCR.
    """
    processed = image
    if suppress_borders:
        processed = suppress_document_borders(processed)
    if deskew:
        processed = deskew_text_lines(processed)

    gray = to_grayscale(processed)
    h, w = gray.shape[:2]

    # 1. Super-resolution upscaling if textlines are small/dense (avoid over-scaling full wide folios)
    if auto_scale and h < 350 and w < 600:
        scale_factor = 1.5
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Background illumination normalization with dynamic kernel
    k_size = max(31, (min(gray.shape[:2]) // 12) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Division normalization to flatten background discoloration & tea stains
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255.0)

    # 3. Bilateral filter to suppress palm-leaf fibers while preserving ink boundaries
    smoothed = cv2.bilateralFilter(normalized.astype(np.uint8), d=5, sigmaColor=45, sigmaSpace=45)

    # 4. CLAHE contrast boost for faded ancient carbon ink
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)

    # 5. Gentle unsharp masking to sharpen character stroke edges
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_adaptive(
    image: np.ndarray,
    deskew: bool = True,
    suppress_ruled_lines: bool = True,
) -> np.ndarray:
    """
    Non-destructive adaptive preprocessing strategy for Sanskrit OCR.
    
    Principles:
    1. Preserves full continuous 8-bit gradients (never converts to 1-bit binary).
    2. Avoids aggressive morphological closing/erosion that washes out matras or thin strokes.
    3. Uses large-scale Gaussian background estimation to normalize illumination variations.
    4. Applies gentle CLAHE (clipLimit=1.6) to enhance faint ink without stroke bloat.
    5. Safely neutralizes horizontal notebook ruled lines while protecting vertical stems and shirorekha.
    6. Corrects minor rotational skew only if detected angle is noticeable (> 0.5 degrees).
    
    Args:
        image: Input document or line image (BGR, RGB, or Grayscale).
        deskew: Whether to check and correct rotational skew.
        suppress_ruled_lines: Whether to neutralize notebook ruled lines.
        
    Returns:
        np.ndarray: Enhanced high-fidelity BGR image optimized for PaddleOCR.
    """
    if image is None or image.size == 0:
        return image

    processed = image.copy()

    # 1. Non-destructive notebook line suppression if enabled
    if suppress_ruled_lines:
        processed = suppress_notebook_ruled_lines(processed)

    # 2. Rotation deskewing if skew is evident
    if deskew:
        processed = deskew_text_lines(processed, max_angle=12.0)

    # Convert to grayscale for illumination and contrast handling
    gray = to_grayscale(processed)
    h, w = gray.shape[:2]

    # 3. Gentle Gaussian illumination normalization (avoids destructive morphological closing)
    blur_k = max(31, (min(h, w) // 4) | 1)
    if blur_k % 2 == 0:
        blur_k += 1
    bg_est = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    mean_bg = max(1.0, float(np.mean(bg_est)))
    normalized = np.clip(
        (gray.astype(np.float32) / np.maximum(bg_est.astype(np.float32), 1.0)) * mean_bg,
        0,
        255
    ).astype(np.uint8)

    # 4. Gentle contrast lift (clipLimit=1.6 protects thin Sanskrit matras from blooming)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 5. Return as BGR for PaddleOCR input pipeline
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
