"""Devanagari Shirorekha (headline) detection and zone-based segmentation.

The Shirorekha is the continuous horizontal line that runs along the top of
Devanagari characters.  Detecting it allows the image to be split into three
structural zones used in traditional script analysis:

* **Upper zone** – matras and modifiers that appear above the headline.
* **Middle zone** – the main consonant bodies between the headline and the
  baseline.
* **Lower zone** – vowel markers, halant, and descenders that hang below the
  baseline.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shirorekha detection
# ---------------------------------------------------------------------------

def detect_shirorekha(image: np.ndarray) -> int:
    """Detect the Shirorekha (headline) Y-coordinate via horizontal projection.

    Parameters
    ----------
    image : np.ndarray
        A grayscale or binary image (2-D ``uint8`` array).  Colour images
        are automatically converted to grayscale first.

    Returns
    -------
    int
        The row index (Y-coordinate) that corresponds to the strongest
        horizontal line in the upper half of the image.  Returns ``0`` when
        the image is empty or too small for reliable detection.
    """
    if image is None or image.size == 0:
        return 0

    # Ensure single-channel grayscale.
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return 0

    # Binarize (invert so that ink = white = high values).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: sum of white pixels per row.
    projection = np.sum(binary, axis=1).astype(float)

    # The Shirorekha should be in the upper half of the image.
    upper_half = max(1, h // 2)
    search_region = projection[:upper_half]

    if search_region.size == 0:
        return 0

    # The row with the maximum projection is our best estimate.
    shirorekha_y = int(np.argmax(search_region))
    return shirorekha_y


# ---------------------------------------------------------------------------
# Shirorekha removal
# ---------------------------------------------------------------------------

def remove_shirorekha(image: np.ndarray) -> np.ndarray:
    """Remove the Shirorekha by blanking out the detected headline rows.

    A small band around the detected Y-coordinate is set to white (255).

    Parameters
    ----------
    image : np.ndarray
        Grayscale or binary image.

    Returns
    -------
    np.ndarray
        A copy of *image* with the Shirorekha rows set to white.
    """
    if image is None or image.size == 0:
        return image

    result = image.copy()
    y = detect_shirorekha(image)

    if y == 0:
        return result

    h = image.shape[0]
    # The Shirorekha typically spans a few pixels.  We blank a thin band.
    band = max(2, int(h * 0.03))
    y_start = max(0, y - band)
    y_end = min(h, y + band + 1)

    if result.ndim == 2:
        result[y_start:y_end, :] = 255
    else:
        result[y_start:y_end, :, :] = 255

    return result


# ---------------------------------------------------------------------------
# Zone segmentation
# ---------------------------------------------------------------------------

def segment_zones(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a Devanagari text image into upper, middle, and lower zones.

    The segmentation is based on the detected Shirorekha position:

    * **upper** – from the top of the image to the Shirorekha.
    * **middle** – from the Shirorekha to approximately two-thirds of the
      remaining height (the consonant body region).
    * **lower** – the rest of the image below the middle zone.

    When the Shirorekha cannot be reliably detected (e.g. the image is very
    small) the image is divided into three equal horizontal bands.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or binary image.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys ``"upper"``, ``"middle"``, and ``"lower"``,
        each mapped to the corresponding image region.  Regions that have
        zero height are returned as empty arrays (``np.array([])``) so that
        callers can safely check ``.size > 0``.
    """
    _empty = np.array([], dtype=np.uint8)

    if image is None or image.size == 0:
        return {"upper": _empty, "middle": _empty, "lower": _empty}

    h = image.shape[0]
    if h < 6:
        return {"upper": _empty, "middle": image, "lower": _empty}

    y = detect_shirorekha(image)

    if y <= 0 or y >= h - 2:
        # Fallback: equal thirds.
        t1 = h // 3
        t2 = 2 * h // 3
        return {
            "upper": image[:t1],
            "middle": image[t1:t2],
            "lower": image[t2:],
        }

    # Remaining height below the Shirorekha.
    remaining = h - y
    middle_end = y + max(1, int(remaining * 0.65))
    middle_end = min(middle_end, h)

    upper = image[:y] if y > 0 else _empty
    middle = image[y:middle_end] if middle_end > y else _empty
    lower = image[middle_end:] if middle_end < h else _empty

    return {"upper": upper, "middle": middle, "lower": lower}
