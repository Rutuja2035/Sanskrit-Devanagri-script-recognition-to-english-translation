"""PaddleOCR adapter for Sanskrit text written in Devanagari."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:  # Allows the rest of the application and tests to load first.
    PaddleOCR = None


Box = Tuple[int, int, int, int]


class PaddleSanskritOCR:
    """Document-level Sanskrit Devanagari OCR backed by PaddleOCR.

    The public ``recognize`` method intentionally preserves the result contract
    used by the existing post-processing and translation pipeline.
    """

    def __init__(
        self,
        device: str = "cpu",
        detection_model: str = "PP-OCRv5_mobile_det",
        recognition_model: str = "devanagari_PP-OCRv5_mobile_rec",
        rec_model_dir: Optional[str | Path] = None,
        char_dict_path: Optional[str | Path] = None,
        engine: Optional[Any] = None,
    ) -> None:
        self.device = device
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.rec_model_dir = Path(rec_model_dir) if rec_model_dir else None
        self.char_dict_path = Path(char_dict_path) if char_dict_path else None
        self.engine: Optional[Any] = engine
        self.is_available = engine is not None
        self.initialization_error: Optional[str] = None
        self.model_variant = "Fine-Tuned Multi-Domain Sanskrit" if self.rec_model_dir else "Official PP-OCRv5 Devanagari"

        if engine is not None:
            return
        if PaddleOCR is None:
            self.initialization_error = (
                "PaddleOCR is not installed. Install dependencies from requirements.txt."
            )
            return

        # Check if fine-tuned weights exist and are requested
        if self.rec_model_dir and self.rec_model_dir.exists():
            try:
                kwargs: Dict[str, Any] = {
                    "text_detection_model_name": detection_model,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": False,
                    "device": device,
                }
                if hasattr(PaddleOCR, "__init__"):
                    # Pass custom recognition model directory if supported
                    try:
                        self.engine = PaddleOCR(
                            rec_model_dir=str(self.rec_model_dir),
                            rec_char_dict_path=str(self.char_dict_path) if self.char_dict_path else None,
                            use_gpu=device.startswith("gpu"),
                            **kwargs,
                        )
                        self.is_available = True
                        return
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            # PaddleOCR 3.x: explicitly choose the Sanskrit-capable Devanagari
            # recognition model instead of treating Sanskrit as generic Hindi.
            self.engine = PaddleOCR(
                text_detection_model_name=detection_model,
                text_recognition_model_name=recognition_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
            )
            self.is_available = True
        except TypeError:
            # Compatibility with the legacy 2.x inference API. ``sa`` selects
            # its bundled Sanskrit/Devanagari multilingual recognition model.
            try:
                self.engine = PaddleOCR(
                    lang="sa",
                    use_angle_cls=False,
                    use_gpu=device.startswith("gpu"),
                    show_log=False,
                )
                self.is_available = True
            except Exception as exc:  # pragma: no cover - runtime setup failure
                self.initialization_error = str(exc)
        except Exception as exc:  # pragma: no cover - runtime setup failure
            self.initialization_error = str(exc)

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        """Convert the existing grayscale/binary preprocessing output to BGR."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("OCR input must be a non-empty NumPy image array.")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(f"Unsupported OCR input shape: {image.shape}")

    @staticmethod
    def _box(points: Any, pad_v_ratio: float = 0.06, img_h: int = 0) -> Box:
        """Convert a Paddle quadrilateral/rectangle to (x, y, width, height) with vertical matra padding."""
        arr = np.asarray(points, dtype=float)
        if arr.size == 4 and arr.ndim == 1:
            x1, y1, x2, y2 = arr.tolist()
        else:
            arr = arr.reshape(-1, 2)
            x1, y1 = arr.min(axis=0)
            x2, y2 = arr.max(axis=0)

        w = max(0, int(x2 - x1))
        h = max(0, int(y2 - y1))
        if pad_v_ratio > 0 and h > 0:
            dy = int(h * pad_v_ratio)
            y1 = max(0, y1 - dy)
            if img_h > 0:
                y2 = min(img_h, y2 + dy)
            else:
                y2 = y2 + dy
            h = int(y2 - y1)
        return (int(x1), int(y1), w, h)

    @staticmethod
    def _as_dict(result: Any) -> Any:
        if hasattr(result, "to_dict"):
            d = result.to_dict()
        elif hasattr(result, "json"):
            d = result.json
        else:
            d = result
        if isinstance(d, dict) and "res" in d and isinstance(d["res"], dict):
            return d["res"]
        return d

    def _run_engine(self, image: np.ndarray) -> Iterable[Any]:
        if hasattr(self.engine, "predict"):
            return self.engine.predict(image)
        # PaddleOCR 2.x API.
        return self.engine.ocr(image, cls=False)

    def _segments_from_result(self, result: Any, img_h: int = 0) -> List[Dict[str, Any]]:
        """Read both PaddleOCR 3.x dictionaries and 2.x list results."""
        res_dict = self._as_dict(result)
        segments: List[Dict[str, Any]] = []

        if isinstance(res_dict, dict):
            texts = res_dict.get("rec_texts", [])
            scores = res_dict.get("rec_scores", [])
            boxes = (
                res_dict.get("rec_polys")
                if res_dict.get("rec_polys") is not None
                else (
                    res_dict.get("dt_polys")
                    if res_dict.get("dt_polys") is not None
                    else res_dict.get("rec_boxes", [])
                )
            )
            for text, score, points in zip(texts, scores, boxes):
                value = str(text).strip()
                if value:
                    segments.append({
                        "text": value,
                        "conf": float(score),
                        "box": self._box(points, pad_v_ratio=0.06, img_h=img_h),
                    })
            return segments

        # Older API: one page is a sequence of [quadrilateral, (text, score)].
        for line in result or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            points, recognition = line[0], line[1]
            if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                continue
            value = str(recognition[0]).strip()
            if value:
                segments.append({
                    "text": value,
                    "conf": float(recognition[1]),
                    "box": self._box(points, pad_v_ratio=0.06, img_h=img_h),
                })
        return segments

    @staticmethod
    def sort_segments_topological(
        segments: List[Dict[str, Any]],
        multi_column: bool = False,
        column_tol: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Sort detected line segments into natural reading order.
        
        Args:
            segments: List of dicts with 'box' = (x, y, w, h).
            multi_column: Whether the document contains multi-column / commentary layout.
            column_tol: Fraction of page width threshold to separate distinct columns.
            
        Returns:
            List[Dict[str, Any]]: Segments sorted by reading order.
        """
        if not segments:
            return []

        def group_horizontal_lines(segs: List[Dict[str, Any]], line_tol: float = 0.5) -> List[Dict[str, Any]]:
            """Group segments on the same horizontal line and order them left-to-right."""
            if not segs:
                return []
            avg_h = sum(s["box"][3] for s in segs) / len(segs)
            y_tol = max(6, int(avg_h * line_tol))
            
            # Sort by top coordinate first
            sorted_by_y = sorted(segs, key=lambda s: s["box"][1])
            lines: List[List[Dict[str, Any]]] = []
            
            for s in sorted_by_y:
                box = s["box"]
                cy = box[1] + box[3] / 2.0
                placed = False
                for line in lines:
                    line_cy = sum(item["box"][1] + item["box"][3] / 2.0 for item in line) / len(line)
                    if abs(cy - line_cy) <= y_tol:
                        line.append(s)
                        placed = True
                        break
                if not placed:
                    lines.append([s])
                    
            lines.sort(key=lambda l: min(item["box"][1] for item in l))
            result = []
            for l in lines:
                l.sort(key=lambda item: item["box"][0])
                result.extend(l)
            return result

        if not multi_column:
            # Standard horizontal reading order: group lines vertically and sort left-to-right
            return group_horizontal_lines(segments)

        # Multi-column / commentary layout sorting:
        # 1. Cluster bounding boxes into vertical column bands based on their horizontal midpoint/x1
        max_x = max(item["box"][0] + item["box"][2] for item in segments)
        col_width_thresh = max(100.0, max_x * column_tol)

        # Assign column index based on x coordinate
        def get_col_id(box: Box) -> int:
            return int(box[0] // col_width_thresh)

        # Group by column, then apply horizontal line grouping within each column
        columns_dict: Dict[int, List[Dict[str, Any]]] = {}
        for s in segments:
            cid = get_col_id(s["box"])
            columns_dict.setdefault(cid, []).append(s)
            
        ordered_result: List[Dict[str, Any]] = []
        for cid in sorted(columns_dict.keys()):
            col_segs = columns_dict[cid]
            ordered_result.extend(group_horizontal_lines(col_segs))
            
        return ordered_result

    def recognize(
        self,
        image: np.ndarray,
        suppress_borders: bool = False,
        multi_column: bool = False
    ) -> Dict[str, Any]:
        """Recognize a preprocessed Sanskrit image without altering its pipeline."""
        if not self.is_available or self.engine is None:
            return {
                "text": "",
                "confidence": 0.0,
                "segments": [],
                "error": self.initialization_error or "PaddleOCR is unavailable.",
            }

        try:
            target_image = image
            if suppress_borders:
                from src.preprocessing import suppress_document_borders
                target_image = suppress_document_borders(target_image)

            prepared = self._prepare_image(target_image)
            img_h = prepared.shape[0]
            segments: List[Dict[str, Any]] = []
            for page in self._run_engine(prepared):
                segments.extend(self._segments_from_result(page, img_h=img_h))

            # Topological reading order sorting (handles standard and multi-column folios)
            segments = self.sort_segments_topological(segments, multi_column=multi_column)
            text = "\n".join(item["text"] for item in segments)
            confidence = (
                sum(item["conf"] for item in segments) / len(segments)
                if segments else 0.0
            )
            return {"text": text, "confidence": confidence, "segments": segments, "error": None}
        except Exception as exc:
            return {"text": "", "confidence": 0.0, "segments": [], "error": str(exc)}

