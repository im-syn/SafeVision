"""Face-level age estimation and gender classification for SafeVision.

The bundled ONNX model is ``onnx-community/age-gender-prediction-ONNX``.
It expects ImageNet-normalized 224x224 RGB face crops and returns two values
for every crop: estimated age and female probability.

Age is an estimate, not identity or legal-age verification.  The model does
not expose an age confidence value, so callers should use the configurable
review margin around their policy threshold.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_FILENAME = "onnx-communityage-gender-prediction-ONNX.onnx"
MODEL_ID = "onnx-community/age-gender-prediction-ONNX"
MODEL_INPUT_SIZE = 224
MIN_ONNX_RUNTIME_VERSION = (1, 18)
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
FACE_LABELS = {"FACE_FEMALE", "FACE_MALE"}
AGE_LABELS = {"CHILD", "ADULT", "AGE_REVIEW"}
GENDER_LABELS = {"GENDER_FEMALE", "GENDER_MALE"}


class AgeGenderModelMissingError(FileNotFoundError):
    """Raised when an enabled demographic check has no usable ONNX model."""


def _onnx_runtime_version_tuple():
    parts = []
    for token in str(getattr(ort, "__version__", "0")).split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 2:
            break
    return tuple((parts + [0, 0])[:2])


def default_model_path(base_dir=None):
    base_dir = Path(base_dir or Path(__file__).resolve().parent)
    return base_dir / "Models" / MODEL_FILENAME


def _parse_providers(providers):
    if isinstance(providers, str):
        providers = [item.strip() for item in providers.split(",") if item.strip()]
    if providers:
        available = set(ort.get_available_providers())
        selected = [provider for provider in providers if provider in available]
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        return selected or ["CPUExecutionProvider"]

    requested = os.environ.get("SAFEVISION_ONNX_PROVIDERS", "")
    if requested:
        return _parse_providers(requested)

    available = ort.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "DirectMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [provider for provider in preferred if provider in available]
    return selected or available or ["CPUExecutionProvider"]


def _clamp_box(box, width, height):
    if isinstance(box, dict):
        x = box.get("x", 0)
        y = box.get("y", 0)
        w = box.get("width", box.get("w", 0))
        h = box.get("height", box.get("h", 0))
    else:
        values = list([0, 0, 0, 0] if box is None else box)
        if len(values) < 4:
            values += [0] * (4 - len(values))
        x, y, w, h = values[:4]

    x1 = max(0, min(width, int(round(float(x)))))
    y1 = max(0, min(height, int(round(float(y)))))
    x2 = max(x1, min(width, int(round(float(x) + float(w)))))
    y2 = max(y1, min(height, int(round(float(y) + float(h)))))
    return [x1, y1, x2 - x1, y2 - y1]


def _intersection_over_union(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_boxes(boxes, iou_threshold=0.55):
    kept = []
    for box in sorted(boxes, key=lambda value: value[2] * value[3], reverse=True):
        if box[2] <= 0 or box[3] <= 0:
            continue
        if any(_intersection_over_union(box, other) >= iou_threshold for other in kept):
            continue
        kept.append(box)
    return kept


def face_boxes_from_detections(detections, width, height, minimum_score=0.2):
    boxes = []
    for detection in detections or []:
        label = str(detection.get("class") or detection.get("label") or "").upper()
        score = float(detection.get("score", detection.get("confidence", 0.0)) or 0.0)
        if label not in FACE_LABELS or score < float(minimum_score):
            continue
        box = detection.get("box")
        if box is None:
            box = detection.get("bounding_box")
        boxes.append(_clamp_box(box, width, height))
    return _deduplicate_boxes(boxes)


def face_result_to_detection(face):
    """Convert a face result into SafeVision's existing detection shape."""
    if face.get("age_enabled"):
        label = "CHILD" if face.get("is_underage") else ("AGE_REVIEW" if face.get("review_required") else "ADULT")
    else:
        label = f"GENDER_{str(face.get('gender') or 'UNKNOWN').upper()}"

    box = face.get("bounding_box") or {}
    score = face.get("gender_confidence")
    detection = {
        "class": label,
        "score": float(score if score is not None else 1.0),
        "box": [
            int(box.get("x", 0)),
            int(box.get("y", 0)),
            int(box.get("width", 0)),
            int(box.get("height", 0)),
        ],
        "category": "demographic",
        "source": "age_gender",
        "model": MODEL_ID,
        "censor": False,
        "age_estimate": face.get("age_estimate"),
        "age_confidence": None,
        "age_threshold": face.get("age_threshold"),
        "is_underage": face.get("is_underage"),
        "review_required": face.get("review_required", False),
        "gender": face.get("gender"),
        "gender_confidence": face.get("gender_confidence"),
        "score_semantics": "gender_confidence" if score is not None else "face_observation_marker",
        "face_source": face.get("face_source"),
    }
    parts = []
    if face.get("age_estimate") is not None:
        parts.append(f"age~{face['age_estimate']:.1f}")
    if face.get("gender"):
        parts.append(f"{face['gender']} {float(face.get('gender_confidence') or 0.0):.0%}")
    detection["display_label"] = f"{label}: " + ", ".join(parts) if parts else label
    return detection


def evaluate_protection_policy(
    nsfw_detected,
    demographics,
    *,
    block_if_nsfw_and_underage=True,
    block_if_underage=False,
    block_on_age_review=False,
):
    demographics = demographics or {}
    underage_detected = bool(demographics.get("underage_detected"))
    review_required = bool(demographics.get("review_required"))
    reasons = []
    if block_if_nsfw_and_underage and bool(nsfw_detected) and underage_detected:
        reasons.append("NSFW_CONTENT_WITH_ESTIMATED_UNDERAGE_PERSON")
    if block_if_underage and underage_detected:
        reasons.append("ESTIMATED_UNDERAGE_PERSON")
    if block_on_age_review and review_required:
        reasons.append("AGE_ESTIMATE_NEAR_POLICY_THRESHOLD")
    return {
        "blocked": bool(reasons),
        "verdict": "BLOCKED" if reasons else ("REVIEW" if review_required else "ALLOW"),
        "reasons": reasons,
        "nsfw_detected": bool(nsfw_detected),
        "underage_detected": underage_detected,
        "age_review_required": review_required,
    }


class AgeGenderDetector:
    """Lazy, batched age/gender inference over detected face regions."""

    def __init__(
        self,
        model_path=None,
        providers=None,
        session_options=None,
        min_face_size=32,
        face_padding=0.18,
        max_batch_size=8,
    ):
        self.model_path = Path(model_path or default_model_path())
        self.providers = providers
        self.session_options = session_options
        self.min_face_size = max(16, int(min_face_size or 32))
        self.face_padding = max(0.0, min(1.0, float(face_padding or 0.0)))
        self.max_batch_size = max(1, min(64, int(max_batch_size or 8)))
        self.session = None
        self.input_name = None
        self.output_name = None
        self.loaded_at = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._face_lock = threading.Lock()
        self._face_cascade = None

    @property
    def model_exists(self):
        return self.model_path.is_file()

    def load(self):
        if self.session is not None:
            return self
        with self._load_lock:
            if self.session is not None:
                return self
            if not self.model_exists:
                raise AgeGenderModelMissingError(
                    "Age/gender protection is enabled but its ONNX model is missing: "
                    f"{self.model_path}. Download onnx/model.onnx from {MODEL_ID} and place it at that path."
                )
            if _onnx_runtime_version_tuple() < MIN_ONNX_RUNTIME_VERSION:
                raise RuntimeError(
                    f"{MODEL_ID} uses ONNX IR 10 and requires onnxruntime>=1.18; "
                    f"installed version is {getattr(ort, '__version__', 'unknown')}."
                )
            providers = _parse_providers(self.providers)
            kwargs = {"providers": providers}
            if self.session_options is not None:
                kwargs["sess_options"] = self.session_options
            session = ort.InferenceSession(str(self.model_path), **kwargs)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            shape = list(input_info.shape)
            if len(shape) != 4 or int(shape[-2] or MODEL_INPUT_SIZE) != MODEL_INPUT_SIZE or int(shape[-1] or MODEL_INPUT_SIZE) != MODEL_INPUT_SIZE:
                raise ValueError(f"Unsupported age/gender model input shape: {shape}; expected [batch, 3, 224, 224]")
            self.session = session
            self.input_name = input_info.name
            self.output_name = output_info.name
            self.loaded_at = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        return self

    def _load_face_cascade(self):
        if self._face_cascade is not None:
            return self._face_cascade
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        self._face_cascade = None if cascade.empty() else cascade
        return self._face_cascade

    def detect_face_boxes(self, frame):
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        cascade = self._load_face_cascade()
        if cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        with self._face_lock:
            detected = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(self.min_face_size, self.min_face_size),
            )
        height, width = frame.shape[:2]
        boxes = [_clamp_box(box, width, height) for box in detected]
        return _deduplicate_boxes(boxes)

    def _expand_face_box(self, box, width, height):
        x, y, w, h = _clamp_box(box, width, height)
        side = max(w, h)
        padded_side = side * (1.0 + self.face_padding * 2.0)
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        expanded = [
            center_x - padded_side / 2.0,
            center_y - padded_side / 2.0,
            padded_side,
            padded_side,
        ]
        return _clamp_box(expanded, width, height)

    @staticmethod
    def preprocess_face(face_bgr):
        if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
            raise ValueError("face crop is empty")
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA if max(rgb.shape[:2]) > MODEL_INPUT_SIZE else cv2.INTER_CUBIC
        resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=interpolation)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGE_MEAN[0]) / IMAGE_STD[0]
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)

    def _predict_batch(self, batch):
        with self._inference_lock:
            outputs = self.session.run([self.output_name], {self.input_name: batch})[0]
        outputs = np.asarray(outputs, dtype=np.float32)
        if outputs.ndim != 2 or outputs.shape[1] < 2:
            raise ValueError(f"Unexpected age/gender model output shape: {outputs.shape}; expected [batch, 2]")
        return outputs

    def analyze_frame(
        self,
        frame,
        *,
        face_boxes=None,
        age_enabled=True,
        gender_enabled=True,
        age_threshold=18.0,
        review_margin=3.0,
        face_source=None,
    ):
        age_enabled = bool(age_enabled)
        gender_enabled = bool(gender_enabled)
        threshold = max(1.0, min(100.0, float(age_threshold or 18.0)))
        margin = max(0.0, min(25.0, float(review_margin or 0.0)))
        if not age_enabled and not gender_enabled:
            return {
                "enabled": False,
                "age_enabled": False,
                "gender_enabled": False,
                "faces_detected": 0,
                "faces": [],
                "underage_detected": None,
                "underage_count": None,
                "review_required": False,
            }
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("could not analyze an empty image")

        self.load()
        height, width = frame.shape[:2]
        # NumPy arrays deliberately reject truth-value testing when they contain
        # more than one element. Normalize every iterable once so callers can
        # pass the boxes returned directly by OpenCV/ONNX detectors.
        supplied_boxes = [] if face_boxes is None else list(face_boxes)
        supplied = bool(supplied_boxes)
        if supplied:
            boxes = _deduplicate_boxes([_clamp_box(box, width, height) for box in supplied_boxes])
            selected_source = face_source or "safevision_face_detection"
        else:
            boxes = self.detect_face_boxes(frame)
            selected_source = face_source or "opencv_haar"

        expanded_boxes, tensors = [], []
        for box in boxes:
            expanded = self._expand_face_box(box, width, height)
            x, y, w, h = expanded
            if w < self.min_face_size or h < self.min_face_size:
                continue
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            tensors.append(self.preprocess_face(crop))
            expanded_boxes.append(expanded)

        predictions = []
        for start in range(0, len(tensors), self.max_batch_size):
            batch = np.stack(tensors[start : start + self.max_batch_size], axis=0).astype(np.float32, copy=False)
            predictions.extend(self._predict_batch(batch).tolist())

        faces = []
        for index, (box, prediction) in enumerate(zip(expanded_boxes, predictions), start=1):
            raw_age = float(prediction[0])
            female_probability = max(0.0, min(1.0, float(prediction[1])))
            age = max(0.0, min(100.0, raw_age))
            is_underage = bool(age < threshold) if age_enabled else None
            review_required = bool(threshold <= age < threshold + margin) if age_enabled and margin > 0 else False
            gender = "female" if female_probability >= 0.5 else "male"
            gender_confidence = max(female_probability, 1.0 - female_probability)
            x, y, w, h = box
            face = {
                "id": f"face_{index}",
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "face_source": selected_source,
                "age_enabled": age_enabled,
                "gender_enabled": gender_enabled,
                "age_estimate": round(age, 2) if age_enabled else None,
                "age_confidence": None,
                "age_threshold": threshold if age_enabled else None,
                "is_underage": is_underage,
                "review_required": review_required,
                "gender": gender if gender_enabled else None,
                "gender_confidence": round(gender_confidence, 6) if gender_enabled else None,
                "gender_probabilities": (
                    {
                        "male": round(1.0 - female_probability, 6),
                        "female": round(female_probability, 6),
                    }
                    if gender_enabled
                    else None
                ),
            }
            faces.append(face)

        underage_count = sum(1 for face in faces if face.get("is_underage")) if age_enabled else None
        adult_count = sum(1 for face in faces if face.get("is_underage") is False) if age_enabled else None
        review_count = sum(1 for face in faces if face.get("review_required")) if age_enabled else None
        gender_counts = None
        if gender_enabled:
            gender_counts = {
                "female": sum(1 for face in faces if face.get("gender") == "female"),
                "male": sum(1 for face in faces if face.get("gender") == "male"),
            }
        return {
            "enabled": True,
            "age_enabled": age_enabled,
            "gender_enabled": gender_enabled,
            "model": {
                "id": MODEL_ID,
                "file": self.model_path.name,
                "loaded": self.session is not None,
                "input_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            },
            "face_detection": {
                "source": selected_source,
                "supplied_boxes": supplied,
                "minimum_face_size": self.min_face_size,
            },
            "age_threshold": threshold if age_enabled else None,
            "review_margin_years": margin if age_enabled else None,
            "faces_detected": len(faces),
            "underage_detected": bool(underage_count) if age_enabled else None,
            "underage_count": underage_count,
            "adult_count": adult_count,
            "review_required": bool(review_count) if age_enabled else False,
            "review_count": review_count,
            "gender_counts": gender_counts,
            "faces": faces,
            "limitations": [
                "Age is an estimate and must not be treated as proof of legal age.",
                "The model provides gender probability but no age-confidence score.",
                "Accuracy is reduced for children, occluded faces, side views, and poor lighting.",
            ],
        }

    def detect_frame(self, frame, **options):
        result = self.analyze_frame(frame, **options)
        return [face_result_to_detection(face) for face in result.get("faces", [])]

    def detect_image(self, image_path, **options):
        frame = cv2.imread(os.fspath(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.analyze_frame(frame, **options)
