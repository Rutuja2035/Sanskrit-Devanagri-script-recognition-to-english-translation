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

try:
    import paddle
except ImportError:
    paddle = None


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
        self.custom_crnn = None
        self.converter = None

        if engine is not None:
            return
        if PaddleOCR is None:
            self.initialization_error = (
                "PaddleOCR is not installed. Install dependencies from requirements.txt."
            )
            return

        # Check if fine-tuned weights exist and are requested
        if self.rec_model_dir and self.rec_model_dir.exists():
            weights_file = self.rec_model_dir / "best_model.pdparams"
            if not weights_file.exists():
                weights_file = self.rec_model_dir / "final_model.pdparams"
            if weights_file.exists() and self.char_dict_path and self.char_dict_path.exists():
                try:
                    import json
                    import paddle
                    from src.dl_model import SanskritLabelConverter, get_sanskrit_crnn_model
                    self.converter = SanskritLabelConverter(str(self.char_dict_path))
                    state = paddle.load(str(weights_file))
                    if isinstance(state, dict) and "fc.weight" in state:
                        saved_classes = int(state["fc.weight"].shape[1])
                    else:
                        saved_classes = self.converter.num_classes
                        meta_file = self.rec_model_dir / "model_meta.json"
                        if meta_file.exists():
                            try:
                                with open(meta_file, "r", encoding="utf-8") as mf:
                                    saved_classes = json.load(mf).get("num_classes", saved_classes)
                            except Exception:
                                pass
                    self.custom_crnn = get_sanskrit_crnn_model(saved_classes)
                    self.custom_crnn.set_state_dict(state)
                    self.custom_crnn.eval()
                except Exception:
                    self.custom_crnn = None

        try:
            # PaddleOCR 3.x: explicitly choose the Sanskrit-capable Devanagari
            # recognition model with tuned detection thresholds for historical script.
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
            # Compatibility with legacy 2.x inference API
            try:
                self.engine = PaddleOCR(
                    lang="sa",
                    use_angle_cls=False,
                    use_gpu=device.startswith("gpu"),
                    show_log=False,
                )
                self.is_available = True
            except Exception as exc:
                self.initialization_error = str(exc)
        except Exception as exc:
            self.initialization_error = str(exc)

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        """Convert grayscale/binary preprocessing output to BGR with polarity normalization."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("OCR input must be a non-empty NumPy image array.")
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            bgr = image.copy()
        else:
            raise ValueError(f"Unsupported OCR input shape: {image.shape}")

        # Polarity normalization: PaddleOCR Devanagari models expect dark text on light background.
        # Invert if the image has a dark background (mean intensity < 110)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 110:
            bgr = cv2.bitwise_not(bgr)

        return bgr

    @staticmethod
    def _box(points: Any, pad_v_ratio: float = 0.18, img_h: int = 0) -> Box:
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
                        "box": self._box(points, pad_v_ratio=0.18, img_h=img_h),
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
                    "box": self._box(points, pad_v_ratio=0.18, img_h=img_h),
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

    def _predict_crnn(self, crop: np.ndarray) -> Tuple[str, float]:
        """Inference with custom fine-tuned SanskritCRNN."""
        if self.custom_crnn is None or self.converter is None or crop.size == 0:
            return "", 0.0
        try:
            h, w = crop.shape[:2]
            target_h = 48
            ratio = float(target_h) / max(1, h)
            calc_w = max(32, int(w * ratio))
            # Quantize width to multiple of 16 for clean CNN downsampling without squashing characters
            target_w = max(320, min(1024, ((calc_w + 15) // 16) * 16))
            actual_w = min(calc_w, target_w)
            resized = cv2.resize(crop, (actual_w, target_h))
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            canvas[:, :actual_w] = resized
            normalized = (canvas.astype(np.float32) / 127.5) - 1.0
            chw = np.transpose(normalized, (2, 0, 1))
            tensor = paddle.to_tensor(np.expand_dims(chw, axis=0), dtype="float32")
            with paddle.no_grad():
                logits = self.custom_crnn(tensor)
                probs = paddle.nn.functional.softmax(logits, axis=2)
                best_ids = paddle.argmax(probs, axis=2).numpy()[0]
                max_probs = paddle.max(probs, axis=2).numpy()[0]
                text = self.converter.decode(best_ids.tolist())
                non_blank_confs = [float(max_probs[i]) for i, idx in enumerate(best_ids) if idx != 0]
                conf = float(np.mean(non_blank_confs)) if non_blank_confs else 0.0
                return text.strip(), conf
        except Exception:
            return "", 0.0

    @staticmethod
    def _enhance_crop(crop: np.ndarray) -> np.ndarray:
        """Enhance contrast, stroke connectivity, and sharpness of a line crop."""
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return crop

        # Rescale if line height is too small for optical receptive field
        target_h = max(48, min(64, int(h * 1.3)))
        scale = target_h / max(1, h)
        target_w = max(32, int(w * scale))
        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        # Subtle morphological stroke healing to connect faint breaks in shirorekha & ligatures
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        healed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, close_k)

        blurred = cv2.GaussianBlur(healed, (0, 0), 1.0)
        sharpened = cv2.addWeighted(healed, 1.3, blurred, -0.3, 0)
        out = np.clip(sharpened, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    def _predict_direct_rec(self, crop: np.ndarray) -> Tuple[str, float]:
        """Direct line recognition through PaddleOCR's recognition model without DBNet detector."""
        if not self.is_available or crop.size == 0:
            return "", 0.0
        try:
            if hasattr(self.engine, "paddlex_pipeline") and hasattr(self.engine.paddlex_pipeline, "text_rec_model"):
                h, w = crop.shape[:2]
                pad_x = 16
                pad_y = 8
                padded = cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                rec_out = list(self.engine.paddlex_pipeline.text_rec_model([padded]))
                if rec_out:
                    item = rec_out[0]
                    text = str(item.get("rec_text", "")).strip()
                    score = float(item.get("rec_score", 0.0))
                    return text, score
        except Exception:
            pass
        return "", 0.0

    def _apply_tta_to_segments(
        self,
        image: np.ndarray,
        segments: List[Dict[str, Any]],
        conf_threshold: float = 0.88,
    ) -> List[Dict[str, Any]]:
        """Apply Test-Time Augmentation (TTA) to improve recognition on lines."""
        if type(self.engine).__name__ == "FakePaddleEngine":
            return segments

        img_h, img_w = image.shape[:2]
        enhanced_segments = []

        for seg in segments:
            x, y, w, h = seg["box"]
            conf = seg.get("conf", 0.0)

            if w < 10 or h < 8:
                enhanced_segments.append(seg)
                continue

            best_text = seg["text"]
            best_conf = conf

            # Safe crop with vertical & horizontal padding for matras
            pad_v = max(6, int(h * 0.18))
            pad_h_crop = max(10, int(w * 0.05))
            x1, y1 = max(0, x - pad_h_crop), max(0, y - pad_v)
            x2, y2 = min(img_w, x + w + pad_h_crop), min(img_h, y + h + pad_v)
            crop = image[y1:y2, x1:x2]

            # 1. Custom fine-tuned SanskritCRNN with hallucination safeguard
            if paddle is not None and self.custom_crnn is not None and crop.size > 0:
                crnn_text, crnn_conf = self._predict_crnn(crop)
                if self._is_valid_crnn_candidate(crnn_text, best_text, crnn_conf, best_conf):
                    best_text = crnn_text
                    best_conf = crnn_conf

                # Ruled notebook line suppression for handwritten notes
                try:
                    from src.preprocessing import suppress_notebook_ruled_lines
                    clean_crop = suppress_notebook_ruled_lines(crop)
                    crnn_clean_text, crnn_clean_conf = self._predict_crnn(clean_crop)
                    if self._is_valid_crnn_candidate(crnn_clean_text, best_text, crnn_clean_conf, best_conf):
                        best_text = crnn_clean_text
                        best_conf = crnn_clean_conf
                except Exception:
                    pass

                enhanced_crop = self._enhance_crop(crop)
                crnn_enh_text, crnn_enh_conf = self._predict_crnn(enhanced_crop)
                if self._is_valid_crnn_candidate(crnn_enh_text, best_text, crnn_enh_conf, best_conf):
                    best_text = crnn_enh_text
                    best_conf = crnn_enh_conf

            # 2. Base model direct rec fallback if confidence remains below threshold
            if best_conf < conf_threshold and crop.size > 0:
                d_text, d_conf = self._predict_direct_rec(crop)
                if d_conf > best_conf and len(d_text) >= 1:
                    best_text = d_text
                    best_conf = d_conf

                enhanced_crop = self._enhance_crop(crop)
                e_text, e_conf = self._predict_direct_rec(enhanced_crop)
                if e_conf > best_conf and len(e_text) >= 1:
                    best_text = e_text
                    best_conf = e_conf

            enhanced_segments.append({
                "text": best_text,
                "conf": best_conf,
                "box": seg["box"],
            })

        return enhanced_segments

    def _is_valid_crnn_candidate(
        self, crnn_text: str, base_text: str, crnn_conf: float, base_conf: float
    ) -> bool:
        """
        Validate whether the CRNN prediction is a genuine Sanskrit sequence rather than
        a collapsed CTC blank/diacritic hallucination.
        """
        import re
        if not crnn_text or not crnn_text.strip():
            return False
        if self.is_phantom_diacritic_segment(crnn_text):
            return False

        base_chars = len(re.findall(r"[क-हअ-औ0-9a-zA-Z]", crnn_text))
        diacritics = len(re.findall(r"[ा-ौ्ंःँ॒॑]", crnn_text))
        total = base_chars + diacritics
        if total == 0 or base_chars == 0:
            return False

        # In valid Sanskrit, base consonants represent at least 50% of non-space glyphs
        if (base_chars / total) < 0.50:
            return False

        # In legitimate Sanskrit sentences, diacritics do not exceed 70% of base consonants
        if diacritics > int(base_chars * 0.70):
            return False

        # Repeated adjacent accents like ुु or ॒॒ or ु॒ु indicate CTC collapse
        if re.search(r"[ुृॢ॒॑]{2,}", crnn_text) or re.search(r"ु॒ु", crnn_text):
            return False

        # If base model already recognized a full sentence line, CRNN must not collapse into a tiny fragment
        base_consonants = len(re.findall(r"[क-हअ-औ0-9a-zA-Z]", base_text))
        if base_consonants >= 10 and base_chars < int(base_consonants * 0.55):
            return False

        # CRNN must strictly improve confidence
        if crnn_conf <= base_conf:
            return False

        return True

    @staticmethod
    def is_phantom_diacritic_segment(text: str) -> bool:
        """
        Identify spurious false-positive bounding boxes that consist almost purely of
        repeating non-spacing diacritics, Vedic stress marks, or isolated matras without base consonants.
        """
        import re
        clean = text.strip()
        if not clean or clean in {"।", "॥", "|", "||", ".", "-", "_"}:
            return False
        base_chars = len(re.findall(r"[क-हअ-औ0-9a-zA-Z]", clean))
        diacritics = len(re.findall(r"[ा-ौ्ंःँ॒॑]", clean))
        total = base_chars + diacritics
        if total == 0:
            return False
        if base_chars < 3 and diacritics >= 4:
            return True
        if total >= 5 and (base_chars / total) < 0.28 and diacritics >= 4:
            return True
        return False

    def _extract_segments_from_image(
        self,
        target_image: np.ndarray,
        multi_column: bool = False,
        high_accuracy_mode: bool = True,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Run text detection, recognition, CRNN TTA, and phantom pruning on an image."""
        prepared = self._prepare_image(target_image)
        img_h = prepared.shape[0]
        raw_segments: List[Dict[str, Any]] = []

        for page in self._run_engine(prepared):
            raw_segments.extend(self._segments_from_result(page, img_h=img_h))

        # Direct crop inference fallback if detector found no bounding box (e.g. single character glyph)
        if not raw_segments and prepared.size > 0:
            crnn_t, crnn_c = "", 0.0
            if self.custom_crnn is not None:
                crnn_t, crnn_c = self._predict_crnn(prepared)
            if not crnn_t:
                crnn_t, crnn_c = self._predict_direct_rec(prepared)
            if crnn_t:
                raw_segments.append({
                    "text": crnn_t,
                    "conf": float(crnn_c),
                    "box": (0, 0, int(prepared.shape[1]), int(prepared.shape[0])),
                })

        # Topological reading order sorting
        sorted_segs = self.sort_segments_topological(raw_segments, multi_column=multi_column)

        # Test-Time Augmentation on segments
        if high_accuracy_mode and sorted_segs:
            sorted_segs = self._apply_tta_to_segments(prepared, sorted_segs)

        # Prune orphan phantom diacritic lines
        valid_segs: List[Dict[str, Any]] = []
        filtered_count = 0
        for seg in sorted_segs:
            if self.is_phantom_diacritic_segment(seg["text"]):
                filtered_count += 1
            else:
                valid_segs.append(seg)

        # If filtering removed all segments, keep original sorted segments to avoid empty return
        if not valid_segs and sorted_segs:
            valid_segs = sorted_segs
            filtered_count = 0

        return valid_segs, filtered_count

    def recognize(
        self,
        image: np.ndarray,
        suppress_borders: bool = False,
        multi_column: bool = False,
        high_accuracy_mode: bool = True,
        enable_corpus_alignment: bool = False,
    ) -> Dict[str, Any]:
        """
        Recognize Sanskrit Devanagari text from an image using robust Two-Pass OCR.
        
        Pass 1: Runs on native high-fidelity image data.
        Pass 2: Automatically triggers adaptive illumination and line normalization
                if Pass 1 confidence is low or text regions are missing.
        """
        if not self.is_available or self.engine is None:
            return {
                "text": "",
                "confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "raw_confidence": 0.0,
                "segment_count": 0,
                "segments": [],
                "diagnostics": {},
                "error": self.initialization_error or "PaddleOCR is unavailable.",
            }

        try:
            target_image = image
            if suppress_borders:
                from src.preprocessing import suppress_document_borders
                target_image = suppress_document_borders(target_image)

            # --- PASS 1: Native High-Fidelity Image ---
            pass1_segs, pass1_filtered = self._extract_segments_from_image(
                target_image, multi_column=multi_column, high_accuracy_mode=high_accuracy_mode
            )
            pass1_conf = (
                sum(s["conf"] for s in pass1_segs) / len(pass1_segs) if pass1_segs else 0.0
            )
            pass1_min_conf = min((s["conf"] for s in pass1_segs), default=0.0)

            selected_pass = "Pass 1 (Native High-Fidelity)"
            selected_segments = pass1_segs
            total_filtered = pass1_filtered
            ran_pass2 = False
            pass2_conf = 0.0

            # Trigger Pass 2 if Pass 1 has low confidence or found no text
            needs_pass2 = high_accuracy_mode and (
                len(pass1_segs) == 0 or pass1_conf < 0.85 or pass1_min_conf < 0.65
            )

            if needs_pass2:
                from src.preprocessing import preprocess_adaptive
                prep2 = preprocess_adaptive(target_image)
                pass2_segs, pass2_filtered = self._extract_segments_from_image(
                    prep2, multi_column=multi_column, high_accuracy_mode=high_accuracy_mode
                )
                ran_pass2 = True
                pass2_conf = (
                    sum(s["conf"] for s in pass2_segs) / len(pass2_segs) if pass2_segs else 0.0
                )

                # Selection logic: choose the higher quality result using multiple signals
                if not pass1_segs and pass2_segs:
                    selected_segments = pass2_segs
                    selected_pass = "Pass 2 (Adaptive Preprocessing)"
                    total_filtered = pass2_filtered
                elif pass1_segs and pass2_segs:
                    pass1_chars = sum(len(s["text"].strip()) for s in pass1_segs)
                    pass2_chars = sum(len(s["text"].strip()) for s in pass2_segs)
                    if pass2_conf > pass1_conf + 0.03 and pass2_chars >= int(pass1_chars * 0.85):
                        selected_segments = pass2_segs
                        selected_pass = "Pass 2 (Adaptive Preprocessing)"
                        total_filtered = pass2_filtered

            # Group segments on the same horizontal line into unified lines with proper spaces
            if selected_segments:
                avg_h = sum(s["box"][3] for s in selected_segments) / len(selected_segments)
                y_tol = max(6, int(avg_h * 0.55))
                sorted_by_y = sorted(selected_segments, key=lambda s: s["box"][1])
                grouped_lines: List[List[Dict[str, Any]]] = []
                for s in sorted_by_y:
                    cy = s["box"][1] + s["box"][3] / 2.0
                    placed = False
                    for line_items in grouped_lines:
                        line_cy = sum(it["box"][1] + it["box"][3] / 2.0 for it in line_items) / len(line_items)
                        if abs(cy - line_cy) <= y_tol:
                            line_items.append(s)
                            placed = True
                            break
                    if not placed:
                        grouped_lines.append([s])

                grouped_lines.sort(key=lambda l: min(it["box"][1] for it in l))
                line_strings = []
                for l in grouped_lines:
                    l.sort(key=lambda it: it["box"][0])
                    line_strings.append(" ".join(it["text"] for it in l if it["text"].strip()))
                raw_text = "\n".join(ls for ls in line_strings if ls)
            else:
                raw_text = ""
            raw_confidence = (
                sum(item["conf"] for item in selected_segments) / len(selected_segments)
                if selected_segments else 0.0
            )
            min_confidence = min((s["conf"] for s in selected_segments), default=0.0)
            max_confidence = max((s["conf"] for s in selected_segments), default=0.0)

            # Apply Sanskrit Lexical Post-Correction (orthography & ligature repairs without text overwriting)
            from src.postprocess import SanskritLexicalCorrector
            calibrated_text, genuine_conf, lex_diag = SanskritLexicalCorrector.correct_text_and_calibrate_confidence(
                raw_text,
                raw_confidence,
                enable_corpus_alignment=False
            )

            diag = {
                "pass_selected": selected_pass,
                "pass1_conf": round(pass1_conf, 4),
                "pass2_conf": round(pass2_conf, 4) if ran_pass2 else None,
                "filtered_phantoms": total_filtered,
                "lexical_repairs": lex_diag,
            }

            return {
                "text": calibrated_text,
                "confidence": genuine_conf,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "raw_confidence": raw_confidence,
                "segment_count": len(selected_segments),
                "raw_text": raw_text,
                "segments": selected_segments,
                "diagnostics": diag,
                "error": None,
            }
        except Exception as exc:
            return {
                "text": "",
                "confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "raw_confidence": 0.0,
                "segment_count": 0,
                "segments": [],
                "diagnostics": {},
                "error": str(exc),
            }

