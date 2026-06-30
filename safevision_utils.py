import os
import platform
import shutil
import tempfile

import cv2
import numpy as np


DEFAULT_CONTENT_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def label_group(label):
    label = str(label or "").upper()
    if label.startswith("FACE_"):
        return "face"
    if "COVERED" in label:
        return "covered"
    if "EXPOSED" in label:
        return "exposed"
    return "other"


def label_matches_filter(label, label_filter="exposed"):
    label_filter = str(label_filter or "exposed").lower()
    group = label_group(label)
    if label_filter == "all":
        return True
    if label_filter == "body":
        return group != "face"
    return group == "exposed"


def default_blur_rules(labels=None, blur=True):
    return {label: bool(blur) for label in (labels or DEFAULT_CONTENT_LABELS)}


def write_blur_exception_rules(path="BlurException.rule", rules=None, labels=None):
    path = os.fspath(path or "BlurException.rule")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    rules = rules or default_blur_rules(labels)
    with open(path, "w", encoding="utf-8") as rule_file:
        for label in (labels or DEFAULT_CONTENT_LABELS):
            rule_file.write(f"{label} = {'true' if rules.get(label, True) else 'false'}\n")
    return path


def ensure_blur_exception_rules(path="BlurException.rule", labels=None):
    path = os.fspath(path or "BlurException.rule")
    if not os.path.exists(path):
        write_blur_exception_rules(path, labels=labels)
        print(f"Created default blur exception rules at: {path}")
    return path


def load_blur_exception_rules(path="BlurException.rule", labels=None):
    path = ensure_blur_exception_rules(path, labels=labels)
    rules = default_blur_rules(labels)
    with open(path, "r", encoding="utf-8") as rule_file:
        for line in rule_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            label, blur = line.split("=", 1)
            rules[label.strip()] = blur.strip().lower() in {"1", "true", "yes", "on"}
    return rules


def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """Read an image through imdecode so Windows Unicode paths work."""
    path = os.fspath(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return cv2.imread(path, flags)


def cv2_imwrite(path, image, params=None):
    """Write an image through imencode so Windows Unicode paths work."""
    path = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    ext = os.path.splitext(path)[1] or ".jpg"
    try:
        ok, encoded = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return cv2.imwrite(path, image, params or [])


class ManagedVideoCapture:
    """cv2.VideoCapture wrapper with a Windows Unicode-path fallback."""

    def __init__(self, path):
        self.original_path = os.fspath(path)
        self.path_in_use = self.original_path
        self._temp_dir = None
        self._capture = cv2.VideoCapture(self.original_path)

        if not self._capture.isOpened() and platform.system() == "Windows" and os.path.exists(self.original_path):
            self._capture.release()
            suffix = os.path.splitext(self.original_path)[1] or ".mp4"
            self._temp_dir = tempfile.TemporaryDirectory(prefix="safevision_video_")
            self.path_in_use = os.path.join(self._temp_dir.name, f"input{suffix}")
            shutil.copy2(self.original_path, self.path_in_use)
            self._capture = cv2.VideoCapture(self.path_in_use)

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def release(self):
        self._capture.release()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def open_video_capture(path):
    return ManagedVideoCapture(path)


def parse_provider_list(value):
    if not value:
        env_value = os.environ.get("SAFEVISION_ONNX_PROVIDERS", "")
        value = env_value
    if not value:
        return None
    return [provider.strip() for provider in value.split(",") if provider.strip()]


def select_onnx_providers(requested=None):
    import onnxruntime

    available = onnxruntime.get_available_providers()
    if requested:
        selected = [provider for provider in requested if provider in available]
        missing = [provider for provider in requested if provider not in available]
        if missing:
            print(f"Requested ONNX providers are not available and will be skipped: {missing}")
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        return selected or ["CPUExecutionProvider"]

    allow_tensorrt = os.environ.get("SAFEVISION_ENABLE_TENSORRT", "").lower() in {"1", "true", "yes"}
    preferred = []
    if allow_tensorrt:
        preferred.append("TensorrtExecutionProvider")
    preferred.extend([
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "DirectMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ])

    selected = []
    for provider in preferred:
        if provider in available and provider not in selected:
            selected.append(provider)

    if not selected:
        selected = available or ["CPUExecutionProvider"]

    skipped_tensorrt = "TensorrtExecutionProvider" in available and not allow_tensorrt
    if skipped_tensorrt:
        print("TensorRT provider detected but disabled by default. Set SAFEVISION_ENABLE_TENSORRT=1 to opt in.")
    print(f"Using ONNX Runtime providers: {selected}")
    return selected


def create_onnx_session(model_path, providers=None, sess_options=None):
    import onnxruntime

    selected = select_onnx_providers(providers)
    try:
        if sess_options is not None:
            return onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=selected)
        return onnxruntime.InferenceSession(model_path, providers=selected)
    except Exception as exc:
        if selected != ["CPUExecutionProvider"] and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            print(f"ONNX provider initialization failed ({exc}). Falling back to CPUExecutionProvider.")
            if sess_options is not None:
                return onnxruntime.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
            return onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        raise
