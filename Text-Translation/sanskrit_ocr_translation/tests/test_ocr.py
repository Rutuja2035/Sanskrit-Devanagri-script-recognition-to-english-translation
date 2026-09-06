import numpy as np
import pytest

from src.ocr import PaddleSanskritOCR


class FakePaddleEngine:
    def predict(self, _image):
        return [{
            "rec_texts": ["कर्म", "धर्म"],
            "rec_scores": [0.90, 0.80],
            "rec_polys": [
                [[40, 40], [90, 40], [90, 55], [40, 55]],
                [[10, 10], [70, 10], [70, 25], [10, 25]],
            ],
        }]


def test_paddle_ocr_returns_pipeline_contract_and_reading_order():
    model = PaddleSanskritOCR(engine=FakePaddleEngine())
    result = model.recognize(np.zeros((80, 120), dtype=np.uint8))

    assert result["text"] == "धर्म\nकर्म"
    assert result["confidence"] == pytest.approx(0.85)
    assert result["error"] is None
    assert result["segments"][0]["box"] == (10, 10, 60, 15)


def test_paddle_ocr_reports_unavailable_engine_without_text_error():
    model = PaddleSanskritOCR(engine=None)
    model.is_available = False
    model.initialization_error = "PaddleOCR is unavailable"

    result = model.recognize(np.zeros((20, 20), dtype=np.uint8))

    assert result["text"] == ""
    assert result["confidence"] == 0.0
    assert result["error"] == "PaddleOCR is unavailable"


def test_multi_column_topological_sorting():
    model = PaddleSanskritOCR(engine=object())
    # Two columns: Column 1 at x=10..60, Column 2 at x=200..250
    segments = [
        {"box": (200, 10, 50, 20), "text": "Col2_Line1", "conf": 0.9},
        {"box": (10, 50, 50, 20), "text": "Col1_Line2", "conf": 0.9},
        {"box": (10, 10, 50, 20), "text": "Col1_Line1", "conf": 0.9},
        {"box": (200, 50, 50, 20), "text": "Col2_Line2", "conf": 0.9},
    ]
    sorted_segs = model.sort_segments_topological(segments, multi_column=True, column_tol=0.3)
    order = [s["text"] for s in sorted_segs]
    assert order == ["Col1_Line1", "Col1_Line2", "Col2_Line1", "Col2_Line2"]


def test_sauvola_binarization_and_despeckle():
    from src.preprocessing import preprocess_sauvola, despeckle_image
    test_img = np.full((100, 100), 220, dtype=np.uint8)
    # Add fake ink stroke
    test_img[40:60, 40:60] = 30
    bin_img = preprocess_sauvola(test_img, window_size=15, despeckle=True)
    assert bin_img.shape == (100, 100)
    assert bin_img.dtype == np.uint8
    # Center stroke should be detected
    assert np.mean(bin_img[45:55, 45:55]) < 100


def test_sandhi_compound_splitting():
    from src.postprocess import split_sanskrit_compounds
    splits = split_sanskrit_compounds("धर्मात्मा", max_splits=3)
    assert len(splits) > 0
    # One of the recognized splits should contain root 'धर्म' or 'धर्मा'
    found_root = any("धर्म" in s[0] for s in splits)
    assert found_root

