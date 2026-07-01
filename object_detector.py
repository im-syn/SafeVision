import json
import math
import os

import cv2
import numpy as np

from safevision_utils import cv2_imread, create_onnx_session


DEFAULT_OBJECT_MODEL = os.path.join(os.path.dirname(__file__), "Models", "safety_objects.onnx")
DEFAULT_OBJECT_LABELS = os.path.join(os.path.dirname(__file__), "Models", "safety_objects.labels.json")


def _load_metadata(labels_path):
    labels_path = os.fspath(labels_path or DEFAULT_OBJECT_LABELS)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Safety object labels file not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as labels_file:
        metadata = json.load(labels_file)

    labels = list(metadata.get("labels") or [])
    if not labels:
        raise ValueError(f"Safety object labels file has no labels: {labels_path}")

    categories = dict(metadata.get("categories") or {})
    default_thresholds = dict(metadata.get("default_thresholds") or {})
    return labels, categories, default_thresholds, metadata


def _input_size(input_shape, fallback=640):
    if not input_shape or len(input_shape) < 4:
        return fallback
    value = input_shape[2]
    if value in (None, "None", "height", "width", -1):
        value = input_shape[3] if len(input_shape) > 3 else fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _preprocess_bgr(image, target_size):
    img_height, img_width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width**2 + img_height**2) / (new_width**2 + new_height**2)
    )
    resized = cv2.resize(rgb, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height
    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    padded = cv2.resize(padded, (target_size, target_size))
    image_data = padded.astype("float32") / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    return image_data, resize_factor, pad_left, pad_top


def _flatten_indices(indices):
    if indices is None:
        return []
    try:
        return np.array(indices).reshape(-1).astype(int).tolist()
    except Exception:
        return [int(item) for item in indices]


class ObjectContentDetector:
    def __init__(self, model_path=None, labels_path=None, providers=None, threshold=0.25, nms_threshold=0.45):
        self.model_path = os.fspath(model_path or DEFAULT_OBJECT_MODEL)
        self.labels_path = os.fspath(labels_path or DEFAULT_OBJECT_LABELS)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Safety object ONNX model not found: {self.model_path}")

        self.labels, self.categories, self.default_thresholds, self.metadata = _load_metadata(self.labels_path)
        self.threshold = float(threshold if threshold is not None else 0.25)
        self.nms_threshold = float(nms_threshold)

        self.onnx_session = create_onnx_session(self.model_path, providers=providers)
        input_info = self.onnx_session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_width = _input_size(input_info.shape, fallback=640)
        self.input_height = self.input_width
        print(
            f"Safety object detector loaded: {os.path.basename(self.model_path)} "
            f"({len(self.labels)} labels, threshold={self.threshold})"
        )

    def _threshold_for_label(self, label):
        category = self.categories.get(label, "safety_object")
        category_threshold = self.default_thresholds.get(category, self.threshold)
        try:
            return max(self.threshold, float(category_threshold))
        except (TypeError, ValueError):
            return self.threshold

    def _detection(self, label, score, box):
        return {
            "class": label,
            "score": float(score),
            "box": [int(value) for value in box],
            "category": self.categories.get(label, "safety_object"),
            "source": "objects",
            "model": self.metadata.get("model_name", "safety_objects"),
            "censor": True,
        }

    def _postprocess_raw_yolo(self, output, resize_factor, pad_left, pad_top):
        data = np.squeeze(output)
        if data.ndim != 2:
            return None

        label_count = len(self.labels)
        if data.shape[0] in {4 + label_count, 5 + label_count}:
            rows = data.T
        else:
            rows = data

        if rows.shape[1] < 4 + label_count:
            return None

        score_offset = rows.shape[1] - label_count
        boxes = []
        scores = []
        class_ids = []

        for row in rows:
            class_scores = row[score_offset:]
            if class_scores.size != label_count:
                continue
            class_id = int(np.argmax(class_scores))
            label = self.labels[class_id]
            score = float(class_scores[class_id])
            if score < self._threshold_for_label(label):
                continue

            x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            left = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top = int(round((y - h * 0.5 - pad_top) * resize_factor))
            width = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            boxes.append([left, top, width, height])
            scores.append(score)
            class_ids.append(class_id)

        keep = _flatten_indices(cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms_threshold))
        return [self._detection(self.labels[class_ids[index]], scores[index], boxes[index]) for index in keep]

    def _postprocess_nms(self, output, resize_factor, pad_left, pad_top):
        data = np.squeeze(output)
        if data.ndim != 2 or data.shape[1] < 6:
            return None

        boxes = []
        scores = []
        class_ids = []

        for row in data:
            score = float(row[4])
            class_id = int(row[5])
            if class_id < 0 or class_id >= len(self.labels):
                continue
            label = self.labels[class_id]
            if score < self._threshold_for_label(label):
                continue

            x1, y1, x2, y2 = [float(value) for value in row[:4]]
            left = int(round((x1 - pad_left) * resize_factor))
            top = int(round((y1 - pad_top) * resize_factor))
            width = int(round((x2 - x1) * resize_factor))
            height = int(round((y2 - y1) * resize_factor))
            boxes.append([left, top, width, height])
            scores.append(score)
            class_ids.append(class_id)

        keep = _flatten_indices(cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms_threshold))
        return [self._detection(self.labels[class_ids[index]], scores[index], boxes[index]) for index in keep]

    def _postprocess(self, outputs, resize_factor, pad_left, pad_top):
        output = outputs[0]
        detections = self._postprocess_raw_yolo(output, resize_factor, pad_left, pad_top)
        if detections is not None:
            return detections
        detections = self._postprocess_nms(output, resize_factor, pad_left, pad_top)
        if detections is not None:
            return detections
        raise ValueError(f"Unsupported safety object model output shape: {np.shape(output)}")

    def detect_frame(self, frame):
        image_data, resize_factor, pad_left, pad_top = _preprocess_bgr(frame, self.input_width)
        outputs = self.onnx_session.run(None, {self.input_name: image_data})
        return self._postprocess(outputs, resize_factor, pad_left, pad_top)

    def detect_image(self, image_path):
        image = cv2_imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_frame(image)
