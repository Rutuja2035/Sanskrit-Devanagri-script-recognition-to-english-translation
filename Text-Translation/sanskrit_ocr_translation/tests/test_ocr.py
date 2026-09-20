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
    assert result["raw_confidence"] == pytest.approx(0.85)
    assert result["confidence"] >= 0.85
    # Bounding box incorporates vertical matra padding (dy = int(15 * 0.18) = 2) -> y1=8, h=19
    assert result["segments"][0]["box"] == (10, 8, 60, 19)


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


def test_notebook_ruled_line_suppression():
    from src.preprocessing import suppress_notebook_ruled_lines
    img = np.full((120, 200, 3), 245, dtype=np.uint8)
    # Add a horizontal blue ruled notebook line
    img[60, :] = [240, 180, 140]  # BGR blueish line
    # Add a dark vertical Sanskrit consonant stroke
    img[40:80, 100:104] = [20, 20, 20]
    
    cleaned = suppress_notebook_ruled_lines(img)
    assert cleaned.shape == img.shape
    # Vertical stroke should be preserved
    assert np.mean(cleaned[40:80, 100:104]) < 50


def test_paninian_orthography_heals_conjuncts_and_avagraha():
    from src.postprocess import SanskritLexicalCorrector
    # Avagraha recovery from 'उ' / '5'
    res1, _, _ = SanskritLexicalCorrector.correct_text_and_calibrate_confidence("तस्मादपरहार्येउर्थे", 0.90)
    assert "ऽर्थे" in res1

    # Split conjunct recovery
    res2, _, _ = SanskritLexicalCorrector.correct_text_and_calibrate_confidence("न तवं शोचितुमरहसि", 0.90)
    assert "त्वं" in res2
    assert "शोचितुमर्हसि" in res2

    # Pre-base matra recovery
    res3, _, _ = SanskritLexicalCorrector.correct_text_and_calibrate_confidence("जनाधपाः", 0.90)
    assert "जनाधिपाः" in res3


def test_dynamic_crnn_handles_multiple_widths():
    import paddle
    from src.dl_model import get_sanskrit_crnn_model
    model = get_sanskrit_crnn_model(119)
    model.eval()
    with paddle.no_grad():
        for w in [320, 480, 640]:
            dummy = paddle.zeros([1, 3, 48, w], dtype="float32")
            out = model(dummy)
            assert out.shape[0] == 1
            assert out.shape[1] == w // 4
            assert out.shape[2] == 119


def test_two_pass_ocr_metadata_reporting():
    from src.ocr import PaddleSanskritOCR
    from tests.test_ocr import FakePaddleEngine
    model = PaddleSanskritOCR(engine=FakePaddleEngine())
    res = model.recognize(np.zeros((80, 120), dtype=np.uint8))

    assert "text" in res
    assert "confidence" in res
    assert "min_confidence" in res
    assert "max_confidence" in res
    assert "segment_count" in res
    assert "diagnostics" in res
    assert res["min_confidence"] <= res["confidence"] <= res["max_confidence"]
    assert res["segment_count"] == len(res["segments"])


def test_is_phantom_diacritic_filtering():
    from src.ocr import PaddleSanskritOCR
    # Spurious orphan diacritic strings should be identified as phantom
    assert PaddleSanskritOCR.is_phantom_diacritic_segment("ुुुुु॒ु॒रु॒रुर॒ु॒ुरु॒॒ु॒ु॒॒॒॒ु॒ुरुुरुु ।")
    assert PaddleSanskritOCR.is_phantom_diacritic_segment("ु॒ु॒ु॒ु॒ ॒ख॒ु॒ु॒ुरु॒े॒ु॒॒ख॒व॒जं॒॒ु॒ ॥")
    assert PaddleSanskritOCR.is_phantom_diacritic_segment("्ु॒ुरु॒ु॒ु॒ु॒ु॒ु॒रु॒ु॒ ॥")

    # Legitimate Sanskrit verses and characters should NEVER be flagged as phantom
    assert not PaddleSanskritOCR.is_phantom_diacritic_segment("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।")
    assert not PaddleSanskritOCR.is_phantom_diacritic_segment("सङ्कल्पप्रभवान् कामान् त्यक्त्वा सर्वानशेषतः ।")
    assert not PaddleSanskritOCR.is_phantom_diacritic_segment("क ।")
    assert not PaddleSanskritOCR.is_phantom_diacritic_segment("॥ १ ॥")



