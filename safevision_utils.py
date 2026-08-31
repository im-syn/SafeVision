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

SAFETY_OBJECT_LABELS = [
    "cigarette",
    "cigar",
    "vape",
    "smoking_pipe",
    "joint",
    "alcohol_bottle",
    "beer_bottle",
    "wine_glass",
    "beer_glass",
    "cocktail_glass",
    "pill",
    "pill_bottle",
    "syringe",
    "cannabis_leaf",
    "drug_bag",
]

ALL_CENSOR_LABELS = DEFAULT_CONTENT_LABELS + SAFETY_OBJECT_LABELS

# The nudity model deliberately reports broad body-context labels.  Those
# labels are useful as observations, but they are not equally explicit and
# must not all activate the child-protection compound rule.
CONTENT_RISK_LEVELS = {
    "SAFE": {
        "FACE_FEMALE",
        "FACE_MALE",
        "FEMALE_GENITALIA_COVERED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    },
    "LOW": {"MALE_BREAST_EXPOSED", "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEET_EXPOSED"},
    "MODERATE": {"BUTTOCKS_EXPOSED"},
    "HIGH": {"FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED"},
    "CRITICAL": {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"},
}
CONTENT_RISK_PRIORITY = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]

# An exposed armpit is common in ordinary photographs and is not censored by
# the balanced default profile. The strict profile can opt back in.
DEFAULT_UNCENSORED_LABELS = {"ARMPITS_EXPOSED"}

DEMOGRAPHIC_LABELS = [
    "CHILD",
    "ADULT",
    "AGE_REVIEW",
    "GENDER_FEMALE",
    "GENDER_MALE",
]

PROTECTION_RULE_DEFAULTS = {
    "BLOCK_IF_NSFW_AND_CHILD": True,
    "BLOCK_IF_CHILD": False,
    "BLOCK_ON_AGE_REVIEW": False,
    "PROTECTION_NSFW_MIN_RISK": "HIGH",
    "PROTECTION_NSFW_MIN_CONFIDENCE": 0.5,
    "UNDERAGE_AGE": 18.0,
    "AGE_REVIEW_MARGIN": 3.0,
    # Full-cover rendering.  Solid modes replace every source pixel; unlike a
    # blur, they do not preserve silhouettes or colors from the input.
    "FULL_COVER_MODE": "blur",
    "FULL_COVER_COLOR": "96,96,96",
    "FULL_COVER_TEXT_COLOR": "255,255,255",
    "FULL_COVER_SHOW_TEXT": True,
    "FULL_COVER_BLUR_STRENGTH": 99.0,
    "FULL_COVER_MESSAGE_NSFW": "Explicit content hidden",
    "FULL_COVER_MESSAGE_NSFW_AND_CHILD": "Possible illegal content - review required",
    "FULL_COVER_MESSAGE_CHILD": "Estimated underage person - review required",
    "FULL_COVER_MESSAGE_REVIEW": "Age review required",
    "FULL_COVER_MESSAGE_GENERIC": "Content hidden by SafeVision policy",
}

FULL_COVER_MODES = {"blur", "gray", "black", "color"}


def parse_bgr_color(value, default=(96, 96, 96)):
    """Parse ``B,G,R`` or ``#RRGGBB`` into an OpenCV BGR tuple."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(max(0, min(255, int(part))) for part in value)
        except (TypeError, ValueError):
            return tuple(default)
    text = str(value or "").strip()
    if text.startswith("#") and len(text) == 7:
        try:
            red, green, blue = int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
            return blue, green, red
        except ValueError:
            return tuple(default)
    try:
        parts = [int(part.strip()) for part in text.split(",")]
        if len(parts) == 3:
            return tuple(max(0, min(255, part)) for part in parts)
    except (TypeError, ValueError):
        pass
    return tuple(default)


def normalize_full_cover_mode(value, default="blur"):
    mode = str(value or default).strip().lower().replace("solid_", "")
    aliases = {"grey": "gray", "solid": "color", "opaque": "color", "gaussian": "blur"}
    mode = aliases.get(mode, mode)
    return mode if mode in FULL_COVER_MODES else default


def full_cover_options(rules=None, overrides=None):
    """Resolve rule-file defaults plus optional CLI/API overrides."""
    values = default_protection_rules()
    values.update(rules or {})
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value
    try:
        strength = max(3, min(999, int(float(values.get("FULL_COVER_BLUR_STRENGTH", 99)))))
    except (TypeError, ValueError):
        strength = 99
    if strength % 2 == 0:
        strength += 1
    return {
        "mode": normalize_full_cover_mode(values.get("FULL_COVER_MODE")),
        "color": parse_bgr_color(values.get("FULL_COVER_COLOR"), (96, 96, 96)),
        "text_color": parse_bgr_color(values.get("FULL_COVER_TEXT_COLOR"), (255, 255, 255)),
        "show_text": bool(values.get("FULL_COVER_SHOW_TEXT", True)),
        "blur_strength": strength,
        "messages": {
            "nsfw": str(values.get("FULL_COVER_MESSAGE_NSFW") or "Explicit content hidden"),
            "nsfw_and_child": str(
                values.get("FULL_COVER_MESSAGE_NSFW_AND_CHILD")
                or "Possible illegal content - review required"
            ),
            "child": str(
                values.get("FULL_COVER_MESSAGE_CHILD")
                or "Estimated underage person - review required"
            ),
            "review": str(values.get("FULL_COVER_MESSAGE_REVIEW") or "Age review required"),
            "generic": str(
                values.get("FULL_COVER_MESSAGE_GENERIC")
                or "Content hidden by SafeVision policy"
            ),
        },
    }


def full_cover_reason_kind(protection_policy=None, nsfw_triggered=False):
    reasons = set((protection_policy or {}).get("reasons") or [])
    if "NSFW_CONTENT_WITH_ESTIMATED_UNDERAGE_PERSON" in reasons:
        return "nsfw_and_child"
    if "ESTIMATED_UNDERAGE_PERSON" in reasons:
        return "child"
    if "AGE_ESTIMATE_NEAR_POLICY_THRESHOLD" in reasons:
        return "review"
    if nsfw_triggered or (protection_policy or {}).get("nsfw_detected"):
        return "nsfw"
    return "generic"


def full_cover_message(options, reason_kind="generic", override=None):
    if override is not None and str(override).strip():
        return str(override).strip()
    return str((options.get("messages") or {}).get(reason_kind) or (options.get("messages") or {}).get("generic") or "Content hidden")


def _wrap_center_text(text, max_chars):
    words = str(text or "").split()
    if not words:
        return []
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:4]


def draw_center_message(image, message, text_color=(255, 255, 255)):
    """Draw a readable, centered warning without depending on image size."""
    if image is None or not str(message or "").strip():
        return image
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.48, min(1.35, width / 900.0))
    thickness = max(1, int(round(scale * 2)))
    lines = _wrap_center_text(message, max(18, int(44 * 900 / max(400, width))))
    metrics = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_height = max((size[1] for size in metrics), default=18) + max(10, int(12 * scale))
    panel_width = min(width - 20, max((size[0] for size in metrics), default=0) + 36)
    panel_height = line_height * len(lines) + 22
    left = max(0, (width - panel_width) // 2)
    top = max(0, (height - panel_height) // 2)
    right = min(width - 1, left + panel_width)
    bottom = min(height - 1, top + panel_height)
    cv2.rectangle(image, (left, top), (right, bottom), (24, 24, 24), -1)
    baseline = top + 18 + max((size[1] for size in metrics), default=18)
    for line, (text_width, _text_height) in zip(lines, metrics):
        x = max(8, (width - text_width) // 2)
        cv2.putText(image, line, (x, baseline), font, scale, tuple(text_color), thickness, cv2.LINE_AA)
        baseline += line_height
    return image


def apply_full_cover(image, options, message=None):
    """Return a full-frame cover according to resolved render options."""
    if image is None:
        return image
    mode = normalize_full_cover_mode(options.get("mode"))
    if mode == "blur":
        strength = int(options.get("blur_strength") or 99)
        if strength % 2 == 0:
            strength += 1
        strength = max(3, min(999, strength))
        covered = cv2.GaussianBlur(image, (strength, strength), max(12.0, strength / 3.0))
        # A second pass prevents small high-contrast regions remaining legible.
        covered = cv2.GaussianBlur(covered, (strength, strength), max(12.0, strength / 3.0))
    else:
        if mode == "black":
            color = (0, 0, 0)
        elif mode == "gray":
            color = (96, 96, 96)
        else:
            color = parse_bgr_color(options.get("color"), (96, 96, 96))
        covered = np.full_like(image, color, dtype=np.uint8)
    if options.get("show_text", True) and message:
        draw_center_message(covered, message, options.get("text_color", (255, 255, 255)))
    return covered


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
    return {
        label: bool(blur) and label not in DEFAULT_UNCENSORED_LABELS
        for label in (labels or ALL_CENSOR_LABELS)
    }


def detection_is_censorable(detection):
    if isinstance(detection, dict):
        if "censor" in detection:
            return bool(detection.get("censor"))
        label = detection.get("class", "")
        category = str(detection.get("category", "")).lower()
        if category in {"smoking", "alcohol", "drugs"}:
            return True
    else:
        label = detection

    label_text = str(label or "")
    return "EXPOSED" in label_text.upper() or label_text in SAFETY_OBJECT_LABELS


def detection_is_nsfw(detection):
    """Return whether a detection represents exposed sexual content."""
    if isinstance(detection, dict):
        label = str(detection.get("class", "") or "")
        category = str(detection.get("category", "") or "").lower()
        source = str(detection.get("source", "nude") or "").lower()
        if source not in {"nude", "nudity", "safevision_nude"}:
            return False
        return category == "exposed" or "EXPOSED" in label.upper()
    return "EXPOSED" in str(detection or "").upper()


def get_content_risk_level(label):
    """Return the policy risk tier for one nudity-model label."""
    normalized = str(label or "").upper()
    for level, labels in CONTENT_RISK_LEVELS.items():
        if normalized in labels:
            return level
    return "UNKNOWN"


def content_risk_meets_minimum(candidate, minimum="HIGH"):
    candidate = str(candidate or "UNKNOWN").upper()
    minimum = str(minimum or "HIGH").upper()
    if candidate not in CONTENT_RISK_PRIORITY or minimum not in CONTENT_RISK_PRIORITY:
        return False
    return CONTENT_RISK_PRIORITY.index(candidate) >= CONTENT_RISK_PRIORITY.index(minimum)


def protection_nsfw_summary(detections, rules=None):
    """Return explicit-content evidence used only by child-protection policy.

    Regional detection/censoring remains independently controlled by label
    rules. This gate prevents weak or ordinary body-context detections from
    escalating an otherwise safe child image into a full-frame block.
    """
    rules = rules or {}
    minimum_risk = str(rules.get("PROTECTION_NSFW_MIN_RISK", "HIGH") or "HIGH").upper()
    if minimum_risk not in CONTENT_RISK_PRIORITY:
        minimum_risk = "HIGH"
    try:
        minimum_confidence = float(rules.get("PROTECTION_NSFW_MIN_CONFIDENCE", 0.5))
    except (TypeError, ValueError):
        minimum_confidence = 0.5
    minimum_confidence = max(0.0, min(1.0, minimum_confidence))

    evidence = []
    for detection in detections or []:
        if not isinstance(detection, dict):
            continue
        source = str(detection.get("source", "nude") or "").lower()
        if source not in {"nude", "nudity", "safevision_nude"}:
            continue
        label = str(detection.get("class") or detection.get("label") or "").upper()
        risk = get_content_risk_level(label)
        try:
            score = float(detection.get("score", detection.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if (
            risk not in {"SAFE", "UNKNOWN"}
            and content_risk_meets_minimum(risk, minimum_risk)
            and score >= minimum_confidence
        ):
            evidence.append({"label": label, "score": round(score, 6), "risk_level": risk})
    return {
        "detected": bool(evidence),
        "minimum_risk": minimum_risk,
        "minimum_confidence": minimum_confidence,
        "evidence": evidence,
    }


def parse_detector_selection(value=None, default="nude,age,gender"):
    raw_value = value
    if raw_value in (None, ""):
        raw_value = os.environ.get("SAFEVISION_DETECTORS", default)

    selected = []
    if isinstance(raw_value, (list, tuple, set)):
        tokens = raw_value
    else:
        tokens = str(raw_value or default).replace(";", ",").replace("+", ",").split(",")
    explicit_none = False
    for token in tokens:
        name = str(token or "").strip().lower()
        if name == "none":
            explicit_none = True
            continue
        if not name:
            continue
        if name == "all":
            selected.extend(["nude", "objects", "age", "gender"])
        elif name == "both":
            selected.extend(["nude", "objects"])
        elif name in {"combined", "protection", "default"}:
            selected.extend(["nude", "age", "gender"])
        elif name in {"nude", "nudity", "nsfw", "body", "safevision"}:
            selected.append("nude")
        elif name in {"object", "objects", "safety", "safety_objects", "cigarette", "smoking", "alcohol", "drugs"}:
            selected.append("objects")
        elif name in {"age", "ages", "child", "children", "minor", "minors", "underage"}:
            selected.append("age")
        elif name in {"gender", "sex"}:
            selected.append("gender")
        elif name in {"demographic", "demographics", "age_gender", "age-gender"}:
            selected.extend(["age", "gender"])

    selected = list(dict.fromkeys(selected))
    if not selected:
        if explicit_none:
            return []
        selected = ["nude", "age", "gender"]
    return selected


def default_protection_rules():
    return dict(PROTECTION_RULE_DEFAULTS)


def write_blur_exception_rules(path="BlurException.rule", rules=None, labels=None):
    path = os.fspath(path or "BlurException.rule")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)

    rules = rules or default_blur_rules(labels)
    with open(path, "w", encoding="utf-8") as rule_file:
        for label in (labels or ALL_CENSOR_LABELS):
            rule_file.write(f"{label} = {'true' if rules.get(label, True) else 'false'}\n")
        rule_file.write("\n# Child protection policy (age values are estimates)\n")
        protection = default_protection_rules()
        protection.update({key: rules[key] for key in protection if key in rules})
        for key, value in protection.items():
            if isinstance(value, bool):
                serialized = "true" if value else "false"
            else:
                serialized = str(value)
            rule_file.write(f"{key} = {serialized}\n")
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


def load_protection_rules(path="BlurException.rule"):
    """Load child-protection policy keys from the existing exception rule file."""
    path = ensure_blur_exception_rules(path)
    rules = default_protection_rules()
    with open(path, "r", encoding="utf-8") as rule_file:
        for line in rule_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = [part.strip() for part in line.split("=", 1)]
            key = key.upper()
            if key not in rules:
                continue
            if isinstance(rules[key], bool):
                rules[key] = raw_value.lower() in {"1", "true", "yes", "on"}
            elif isinstance(rules[key], (int, float)):
                try:
                    rules[key] = float(raw_value)
                except ValueError:
                    pass
            else:
                rules[key] = raw_value.upper() if key == "PROTECTION_NSFW_MIN_RISK" else raw_value
    rules["UNDERAGE_AGE"] = max(1.0, min(100.0, float(rules["UNDERAGE_AGE"])))
    rules["AGE_REVIEW_MARGIN"] = max(0.0, min(25.0, float(rules["AGE_REVIEW_MARGIN"])))
    rules["PROTECTION_NSFW_MIN_CONFIDENCE"] = max(
        0.0, min(1.0, float(rules["PROTECTION_NSFW_MIN_CONFIDENCE"]))
    )
    if rules["PROTECTION_NSFW_MIN_RISK"] not in CONTENT_RISK_PRIORITY:
        rules["PROTECTION_NSFW_MIN_RISK"] = PROTECTION_RULE_DEFAULTS["PROTECTION_NSFW_MIN_RISK"]
    rules["FULL_COVER_MODE"] = normalize_full_cover_mode(rules.get("FULL_COVER_MODE"))
    rules["FULL_COVER_BLUR_STRENGTH"] = max(
        3.0, min(999.0, float(rules.get("FULL_COVER_BLUR_STRENGTH", 99.0)))
    )
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


def normalize_mask_shape(mask_shape="rectangle"):
    value = str(mask_shape or "rectangle").strip().lower()
    if value in {"ellipse", "oval", "circle", "round"}:
        return "ellipse"
    return "rectangle"


def make_blur_kernel(strength=None, sigma=None, default=(23, 23, 30)):
    if strength in (None, ""):
        return default

    try:
        kernel = int(float(strength))
    except (TypeError, ValueError):
        return default

    kernel = max(3, min(kernel, 151))
    if kernel % 2 == 0:
        kernel += 1

    if sigma in (None, ""):
        sigma_value = max(1.0, float(kernel))
    else:
        try:
            sigma_value = max(1.0, float(sigma))
        except (TypeError, ValueError):
            sigma_value = max(1.0, float(kernel))

    return (kernel, kernel, sigma_value)


def apply_region_censor(
    image,
    x,
    y,
    w,
    h,
    blur_kernel=(23, 23, 30),
    use_solid_color=False,
    solid_color=(0, 0, 0),
    mask_shape="rectangle",
):
    image_height, image_width = image.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image_width, int(x + w))
    y2 = min(image_height, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return False

    roi = image[y1:y2, x1:x2]
    roi_height, roi_width = roi.shape[:2]

    if use_solid_color:
        censored_roi = np.full((roi_height, roi_width, 3), solid_color, dtype=np.uint8)
    else:
        kernel_x, kernel_y, kernel_sigma = blur_kernel
        censored_roi = cv2.GaussianBlur(roi, (int(kernel_x), int(kernel_y)), float(kernel_sigma))

    if normalize_mask_shape(mask_shape) == "ellipse":
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        center = (roi_width // 2, roi_height // 2)
        axes = (max(1, roi_width // 2), max(1, roi_height // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        roi[mask > 0] = censored_roi[mask > 0]
    else:
        image[y1:y2, x1:x2] = censored_roi

    return True


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
