import os
import math
import json
import cv2
import numpy as np
import onnx
from onnx import version_converter 
import onnxruntime
import argparse
from safevision_utils import (
    apply_full_cover,
    apply_region_censor,
    cv2_imread,
    cv2_imwrite,
    create_onnx_session,
    detection_is_censorable,
    detection_is_nsfw,
    full_cover_message,
    full_cover_options,
    full_cover_reason_kind,
    load_blur_exception_rules,
    load_protection_rules,
    make_blur_kernel,
    normalize_mask_shape,
    parse_provider_list,
    parse_detector_selection,
    protection_nsfw_summary,
)
from object_detector import DEFAULT_OBJECT_LABELS, DEFAULT_OBJECT_MODEL, ObjectContentDetector
from age_gender_detector import (
    AgeGenderDetector,
    AgeGenderModelMissingError,
    default_model_path as default_age_gender_model_path,
    evaluate_protection_policy,
    face_boxes_from_detections,
    face_result_to_detection,
)

__labels = [
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


def _read_image(image_path, target_size=320):
    img = cv2_imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width  = target_size  
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width**2 + img_height**2) / (new_width**2 + new_height**2)
    )

    img = cv2.resize(img, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height

    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    img = cv2.resize(img, (target_size, target_size))

    image_data = img.astype("float32") / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    return image_data, resize_factor, pad_left, pad_top


def _postprocess(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores) 

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top = int(round((y - h * 0.5 - pad_top) * resize_factor))
            width = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])  

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        label = __labels[class_id]
        detections.append(
            {
                "class": label,
                "score": float(score),
                "box": box,
                "category": "exposed" if "EXPOSED" in label else ("covered" if "COVERED" in label else "face" if label.startswith("FACE_") else "other"),
                "source": "nude",
                "model": "safevision_nude",
                "censor": "EXPOSED" in label,
            }
        )

    return detections



def _ensure_opset15(original_path: str) -> str:
    """
    Load the original ONNX model, convert it to opset 15 if needed,
    and save to a new file. Returns the path to the opset-15 model.
    """
    base, ext = os.path.splitext(original_path)
    conv_path = f"{base}_opset15{ext}"
    if not os.path.exists(conv_path):
        model = onnx.load(original_path)
        converted = version_converter.convert_version(model, 15)
        onnx.save(converted, conv_path)
    return conv_path

def download_model(url, save_path):
    """Download the ONNX model from the provided URL and save it to the specified path."""
    import urllib.request
    
    print(f"Downloading model from {url}...")
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download the file
        urllib.request.urlretrieve(url, save_path)
        print(f"Model downloaded successfully to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

class NudeDetector:
    def __init__(self, providers=None, model_path=None):
        # Set up model paths
        model_dir = os.path.join(os.path.dirname(__file__), "Models")
        model_orig = os.path.abspath(os.fspath(model_path or os.path.join(model_dir, "best.onnx")))
        
        # Check if model exists, if not download it
        if not os.path.exists(model_orig):
            print("Model file not found. Creating Models directory and downloading model...")
            model_url = "https://github.com/im-syn/SafeVision/raw/refs/heads/main/Models/best.onnx"
            success = download_model(model_url, model_orig)
            if not success:
                raise FileNotFoundError(f"Could not download model from {model_url}. Please download manually and place in {model_dir}")
        
        # Convert best.onnx → best_opset15.onnx at runtime
        model_to_load = _ensure_opset15(model_orig)

        self.onnx_session = create_onnx_session(model_to_load, providers=providers)
        inp = self.onnx_session.get_inputs()[0]
        self.input_name   = inp.name
        self.input_width  = inp.shape[2]
        self.input_height = inp.shape[3]

        self.blur_exception_rules = None
        self.full_blur_count      = 0
        
    def load_exception_rules(self, rule_file_path):
        if not rule_file_path:
            rule_file_path = "BlurException.rule"

        self.blur_exception_rules = load_blur_exception_rules(rule_file_path, labels=globals()["__labels"])
        print(f"Loaded {len(self.blur_exception_rules)} exception rules from {rule_file_path}")



    def should_apply_blur(self, label):
        should_blur = self.blur_exception_rules.get(label, True)
        if should_blur:
            self.full_blur_count += 1  # Increment the full blur count
        return should_blur

    def detect(self, image_path):
        preprocessed_image, resize_factor, pad_left, pad_top = _read_image(
            image_path, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs, resize_factor, pad_left, pad_top)

        return detections

    def censor(
        self,
        image_path,
        apply_blur=False,
        classes=[],
        output_path=None,
        full_blur_rule=0,
        blur_kernel=(23, 23, 30),
        mask_shape="rectangle",
        use_solid_color=False,
        solid_color=(0, 0, 0),
    ):
        detections = self.detect(image_path)
        if classes:
            detections = [
                detection for detection in detections if detection["class"] in classes
            ]

        img = cv2_imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_boxes = img.copy()
        img_combined = img.copy()

        if apply_blur:
            img_blur = img.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        log_data = []  # List to store information for the log file

        exposed_count = 0  # Counter for exposed labels

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]

            label = detection["class"]
            label_text = label if "EXPOSED" not in label else "Unsafe, " + label

            log_data.append({"label": label, "box": box})

            should_blur = self.should_apply_blur(label)
            print(f"Label: {label}, Should blur: {should_blur}")

            if apply_blur and "EXPOSED" in label and should_blur:
                print(f"Blur should be applied to: {label}")
                apply_region_censor(
                    img_blur,
                    x,
                    y,
                    w,
                    h,
                    blur_kernel=blur_kernel,
                    use_solid_color=use_solid_color,
                    solid_color=solid_color,
                    mask_shape=mask_shape,
                )
                exposed_count += 1

            else:
                # Draw boxes around NSFW regions
                cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add label near the box
                cv2.putText(img_boxes, label_text, (x, y - 5), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

            # Draw boxes on the combined image
            cv2.rectangle(img_combined, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label near the box
            cv2.putText(img_combined, label_text, (x, y - 5), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        if not output_path:
            input_path, ext = os.path.splitext(args.input)
            if apply_blur:
                output_path = f"output/{os.path.basename(input_path)}_Blur{ext}"
            else:
                output_path = f"output/{os.path.basename(input_path)}_Detect{ext}"

        if apply_blur:
            if full_blur_rule > 0 and exposed_count >= full_blur_rule:
                # Apply full blur to the whole image
                img_blur = cv2.GaussianBlur(img_blur, (blur_kernel[0], blur_kernel[1]), blur_kernel[2])

            cv2_imwrite(output_path, img_blur)
        else:
            # Save the image with boxes and labels
            cv2_imwrite(output_path, img_combined)
            # Save the boxes detection image with labels
            detect_path = f"Prosses/{os.path.basename(output_path)}"
            cv2_imwrite(detect_path, img_boxes)

        # Create a log file for the image
        log_file_path = f"Logs/{os.path.basename(output_path)}.log"
        with open(log_file_path, "w") as log_file:
            for data in log_data:
                log_file.write(f"Label: {data['label']}, Box: {data['box']}\n")

        return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Nude Detector")
    parser.add_argument("-i", "--input", type=str, help="Path to the input image", required=True)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save the censored image. If not provided, a default path will be used.",
    )
    parser.add_argument(
        "-b",
        "--blur",
        action="store_true",
        help="Apply blur to NSFW regions instead of drawing boxes",
    )
    parser.add_argument(
        "--boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw detection boxes on the final image (default: enabled; use --no-boxes to hide them).",
    )
    parser.add_argument(
        "--save-boxes-copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the unredacted debug image with boxes in Prosses/ (default: disabled for privacy).",
    )
    parser.add_argument(
        "--save-blur-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a clean regional-censor copy in Blur/ (default: enabled).",
    )
    parser.add_argument(
        "-e",
        "--exception",
        type=str,
        default=None,
        help="Path to the blur exception rules file",
    )
    parser.add_argument(
        "-fbr",
        "--full_blur_rule",
        type=int,
        default=0,
        help="Number of censorable NSFW exposed boxes that triggers the selected whole-image cover.",
    )
    parser.add_argument(
        "--mask-shape",
        type=str,
        choices=["rectangle", "ellipse", "oval"],
        default="rectangle",
        help="Shape used for regional censoring masks. Default is rectangle.",
    )
    parser.add_argument(
        "--blur-strength",
        type=int,
        default=23,
        help="Regional blur kernel strength. Larger odd numbers blur more. Default is 23.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=None,
        help="Gaussian blur sigma for regional blur. Defaults to the blur strength.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use a solid color instead of blur to mask detections.",
    )
    parser.add_argument(
        "--mask-color",
        type=str,
        default="0,0,0",
        help="Color to use for masking in BGR format. Default is black: '0,0,0'.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="Comma-separated ONNX Runtime providers, e.g. CUDAExecutionProvider,CPUExecutionProvider",
    )
    parser.add_argument(
        "--nsfw-model",
        type=str,
        default=None,
        help="Optional path to the SafeVision NSFW ONNX model (default: Models/best.onnx).",
    )
    parser.add_argument(
        "--detectors",
        type=str,
        default="nude,age,gender",
        help="Checks to run: nude, age, gender, objects, demographics, protection, or all.",
    )
    parser.add_argument(
        "--object-model",
        type=str,
        default=DEFAULT_OBJECT_MODEL,
        help="Path to the optional safety-object ONNX model.",
    )
    parser.add_argument(
        "--object-labels",
        type=str,
        default=DEFAULT_OBJECT_LABELS,
        help="Path to the safety-object labels JSON file.",
    )
    parser.add_argument(
        "--object-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence for safety-object detections.",
    )
    parser.add_argument(
        "--age-gender-model",
        type=str,
        default=str(default_age_gender_model_path()),
        help="Path to the age/gender ONNX model. An enabled age/gender check fails clearly when it is missing.",
    )
    parser.add_argument(
        "--underage-age",
        type=float,
        default=None,
        help="Estimated age below which a face is flagged as underage. Defaults to the rule file value (18).",
    )
    parser.add_argument(
        "--age-review-margin",
        type=float,
        default=None,
        help="Years above the underage threshold that require review. Defaults to the rule file value (3).",
    )
    parser.add_argument("--min-face-size", type=int, default=32, help="Minimum face size in pixels for fallback face detection.")
    parser.add_argument("--face-padding", type=float, default=0.18, help="Padding around face crops as a fraction of face size.")
    parser.add_argument(
        "--full-cover-mode",
        "--full-blur-mode",
        choices=["blur", "gray", "black", "color"],
        default=None,
        help="Whole-image cover used when policy or -fbr triggers. Solid modes reveal no source pixels.",
    )
    parser.add_argument(
        "--full-cover-color",
        default=None,
        help="Custom full-cover color as B,G,R or #RRGGBB (used by mode=color).",
    )
    parser.add_argument(
        "--full-cover-text-color",
        default=None,
        help="Centered warning text color as B,G,R or #RRGGBB.",
    )
    parser.add_argument(
        "--full-cover-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show or hide the centered full-cover warning.",
    )
    parser.add_argument(
        "--full-cover-message",
        default=None,
        help="Override the centered warning for this run.",
    )
    parser.add_argument(
        "--force-full-cover",
        action="store_true",
        help="Cover the whole image even when no automatic policy rule matched.",
    )
    parser.add_argument(
        "--block-if-nsfw-and-child",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override BLOCK_IF_NSFW_AND_CHILD from the rule file.",
    )
    parser.add_argument(
        "--block-if-child",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override BLOCK_IF_CHILD from the rule file.",
    )
    parser.add_argument(
        "--block-on-age-review",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override BLOCK_ON_AGE_REVIEW from the rule file.",
    )
    parser.add_argument(
        "--child-nsfw-min-risk",
        choices=["LOW", "MODERATE", "HIGH", "CRITICAL"],
        default=None,
        help="Minimum NSFW risk tier allowed to combine with an estimated child result.",
    )
    parser.add_argument(
        "--child-nsfw-min-confidence",
        type=float,
        default=None,
        help="Minimum NSFW confidence (0..1) for the compound child-protection rule.",
    )
    parser.add_argument(
        "--fail-on-policy",
        action="store_true",
        help="Exit with status 2 when the child-protection policy blocks the image (useful in CI).",
    )
    parser.add_argument(
        "--fail-on-underage",
        action="store_true",
        help="Exit with status 3 whenever an estimated underage face is found, even if policy allows it.",
    )
    return parser.parse_args()



def create_directories():
    # Create directories if they don't exist
    os.makedirs("Blur", exist_ok=True)
    os.makedirs("Prosses", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("Logs", exist_ok=True)


def _parse_bgr_color(value, default=(0, 0, 0)):
    try:
        parts = [int(part.strip()) for part in str(value or "").split(",")]
        if len(parts) == 3:
            return tuple(max(0, min(255, part)) for part in parts)
    except ValueError:
        pass
    print("Invalid mask color. Using default black.")
    return default


def _run_selected_detectors(
    image_path,
    enabled_detectors,
    providers,
    object_model,
    object_labels,
    object_threshold,
    age_gender_model,
    underage_age,
    age_review_margin,
    min_face_size,
    face_padding,
    nsfw_model=None,
):
    detections = []
    nude_detections = []
    if "nude" in enabled_detectors:
        detector = NudeDetector(providers=providers, model_path=nsfw_model)
        nude_detections = detector.detect(image_path)
        detections.extend(nude_detections)

    if "objects" in enabled_detectors:
        object_detector = ObjectContentDetector(
            model_path=object_model,
            labels_path=object_labels,
            providers=providers,
            threshold=object_threshold,
        )
        detections.extend(object_detector.detect_image(image_path))

    demographics = {
        "enabled": False,
        "age_enabled": False,
        "gender_enabled": False,
        "faces_detected": 0,
        "faces": [],
        "underage_detected": None,
        "review_required": False,
    }
    if "age" in enabled_detectors or "gender" in enabled_detectors:
        frame = cv2_imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        face_boxes = face_boxes_from_detections(
            nude_detections,
            width=frame.shape[1],
            height=frame.shape[0],
        )
        demographic_detector = AgeGenderDetector(
            model_path=age_gender_model,
            providers=providers,
            min_face_size=min_face_size,
            face_padding=face_padding,
        )
        demographics = demographic_detector.analyze_frame(
            frame,
            face_boxes=face_boxes or None,
            age_enabled="age" in enabled_detectors,
            gender_enabled="gender" in enabled_detectors,
            age_threshold=underage_age,
            review_margin=age_review_margin,
            face_source="safevision_nsfw_faces" if face_boxes else None,
        )
        detections.extend(face_result_to_detection(face) for face in demographics.get("faces", []))

    return detections, demographics


def _draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (x, max(12, y - 5)), font, 0.5, color, 1, cv2.LINE_AA)


def render_image_outputs(
    args,
    detections,
    rules,
    blur_kernel,
    mask_shape,
    solid_color,
    protection_policy=None,
    protection_rules=None,
):
    img = cv2_imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    img_blur = img.copy()
    img_boxes = img.copy()
    img_combined = img.copy()
    image_height, image_width = img.shape[:2]
    censorable_count = 0
    nsfw_censorable_count = 0
    log_data = []

    for detection in detections:
        x, y, w, h = [int(value) for value in detection.get("box", [0, 0, 0, 0])]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + w)
        y2 = min(image_height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        label = detection.get("class", "UNKNOWN")
        score = float(detection.get("score", 0.0))
        category = detection.get("category") or ("exposed" if "EXPOSED" in str(label).upper() else "other")
        source = detection.get("source", "nude")
        is_censorable = detection_is_censorable(detection)
        should_blur = rules.get(label, is_censorable)
        if label == "CHILD":
            color = (0, 165, 255)
        elif label == "AGE_REVIEW":
            color = (0, 215, 255)
        elif is_censorable and not should_blur:
            color = (160, 160, 160)
        else:
            color = (0, 0, 255) if is_censorable else (0, 255, 0)
        label_text = detection.get("display_label") or f"{label} {score:.2f}"
        if is_censorable and should_blur and "EXPOSED" in str(label).upper():
            label_text = f"Unsafe, {label} {score:.2f}"
        elif is_censorable and not should_blur:
            label_text = f"Allowed by rule, {label} {score:.2f}"

        log_data.append(
            {
                "label": label,
                "score": score,
                "category": category,
                "source": source,
                "box": [x1, y1, x2 - x1, y2 - y1],
                "censor": is_censorable,
                "blur": bool(is_censorable and should_blur),
            }
        )

        if is_censorable and should_blur:
            censorable_count += 1
            if detection_is_nsfw(detection):
                nsfw_censorable_count += 1

        if is_censorable and should_blur:
            apply_region_censor(
                img_blur,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                blur_kernel=blur_kernel,
                use_solid_color=args.color,
                solid_color=solid_color,
                mask_shape=mask_shape,
            )
            if args.blur:
                apply_region_censor(
                    img_combined,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    blur_kernel=blur_kernel,
                    use_solid_color=args.color,
                    solid_color=solid_color,
                    mask_shape=mask_shape,
                )

        if args.save_boxes_copy:
            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), color, 2)
            _draw_label(img_boxes, label_text, x1, y1, color)
        if args.boxes:
            cv2.rectangle(img_combined, (x1, y1), (x2, y2), color, 2)
            _draw_label(img_combined, label_text, x1, y1, color)

    protection_blocked = bool((protection_policy or {}).get("blocked"))
    count_rule_triggered = bool(args.full_blur_rule > 0 and nsfw_censorable_count >= args.full_blur_rule)
    full_cover_applied = bool(protection_blocked or count_rule_triggered or args.force_full_cover)
    cover_options = full_cover_options(
        protection_rules,
        {
            "FULL_COVER_MODE": args.full_cover_mode,
            "FULL_COVER_COLOR": args.full_cover_color,
            "FULL_COVER_TEXT_COLOR": args.full_cover_text_color,
            "FULL_COVER_SHOW_TEXT": args.full_cover_text,
        },
    )
    reason_kind = full_cover_reason_kind(
        protection_policy,
        nsfw_triggered=count_rule_triggered,
    )
    if args.force_full_cover and reason_kind == "generic":
        reason_kind = "generic"
    cover_message = full_cover_message(cover_options, reason_kind, args.full_cover_message)
    if full_cover_applied:
        img_blur = apply_full_cover(img_blur, cover_options, cover_message)
        img_combined = apply_full_cover(img_combined, cover_options, cover_message)

    output_path = args.output
    if not output_path:
        input_path, ext = os.path.splitext(args.input)
        suffix = "_Blur" if args.blur else "_Detect"
        output_path = f"output/{os.path.basename(input_path)}{suffix}{ext}"

    blur_path = f"Blur/{os.path.basename(output_path)}" if args.save_blur_copy else None
    detect_path = f"Prosses/{os.path.basename(output_path)}" if args.save_boxes_copy else None
    if full_cover_applied:
        final_image = img_combined
    elif args.blur:
        final_image = img_combined
    elif args.boxes:
        final_image = img_boxes if args.save_boxes_copy else img_combined
    else:
        final_image = img

    cv2_imwrite(output_path, final_image)
    if blur_path:
        cv2_imwrite(blur_path, img_blur)
    if detect_path:
        cv2_imwrite(detect_path, img_boxes)

    log_file_path = f"Logs/{os.path.basename(output_path)}.log"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for item in log_data:
            log_file.write(
                f"Label: {item['label']}, Score: {item['score']:.4f}, "
                f"Category: {item['category']}, Source: {item['source']}, "
                f"Censor: {item['censor']}, Blur: {item['blur']}, Box: {item['box']}\n"
            )

    rendering = {
        "boxes_on_final": bool(args.boxes and not full_cover_applied),
        "regional_censoring": bool(args.blur and not full_cover_applied),
        "censorable_regions": censorable_count,
        "nsfw_regions_for_full_cover_rule": nsfw_censorable_count,
        "boxes_copy_saved": bool(detect_path),
        "blur_copy_saved": bool(blur_path),
        "full_cover_applied": full_cover_applied,
        "full_cover_mode": cover_options["mode"] if full_cover_applied else None,
        "full_cover_reason": reason_kind if full_cover_applied else None,
        "full_cover_message": cover_message if full_cover_applied and cover_options["show_text"] else None,
        "solid_cover_reveals_source_pixels": False if full_cover_applied and cover_options["mode"] != "blur" else None,
    }
    return output_path, blur_path, detect_path, log_data, rendering

if __name__ == "__main__":
    create_directories()  # Create directories before processing

    args = parse_args()
    blur_kernel = make_blur_kernel(args.blur_strength, args.blur_sigma)
    mask_shape = normalize_mask_shape(args.mask_shape)
    solid_color = _parse_bgr_color(args.mask_color)

    print(f"Using {mask_shape} regional mask shape")
    print(f"Using regional blur kernel: {blur_kernel}")

    enabled_detectors = parse_detector_selection(args.detectors)
    print(f"Enabled detectors: {', '.join(enabled_detectors)}")

    exception_file_path = args.exception or "BlurException.rule"
    rules = load_blur_exception_rules(exception_file_path)
    protection_rules = load_protection_rules(exception_file_path)
    if args.block_if_nsfw_and_child is not None:
        protection_rules["BLOCK_IF_NSFW_AND_CHILD"] = args.block_if_nsfw_and_child
    if args.block_if_child is not None:
        protection_rules["BLOCK_IF_CHILD"] = args.block_if_child
    if args.block_on_age_review is not None:
        protection_rules["BLOCK_ON_AGE_REVIEW"] = args.block_on_age_review
    if args.child_nsfw_min_risk is not None:
        protection_rules["PROTECTION_NSFW_MIN_RISK"] = args.child_nsfw_min_risk
    if args.child_nsfw_min_confidence is not None:
        protection_rules["PROTECTION_NSFW_MIN_CONFIDENCE"] = max(
            0.0, min(1.0, args.child_nsfw_min_confidence)
        )
    providers = parse_provider_list(args.providers)

    underage_age = args.underage_age if args.underage_age is not None else protection_rules["UNDERAGE_AGE"]
    age_review_margin = args.age_review_margin if args.age_review_margin is not None else protection_rules["AGE_REVIEW_MARGIN"]

    try:
        detections, demographics = _run_selected_detectors(
            args.input,
            enabled_detectors,
            providers,
            args.object_model,
            args.object_labels,
            args.object_threshold,
            args.age_gender_model,
            underage_age,
            age_review_margin,
            args.min_face_size,
            args.face_padding,
            args.nsfw_model,
        )
    except AgeGenderModelMissingError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(f"Detections found: {len(detections)}")

    nsfw_gate = protection_nsfw_summary(detections, protection_rules)
    protection_policy = evaluate_protection_policy(
        nsfw_gate["detected"],
        demographics,
        block_if_nsfw_and_underage=protection_rules["BLOCK_IF_NSFW_AND_CHILD"],
        block_if_underage=protection_rules["BLOCK_IF_CHILD"],
        block_on_age_review=protection_rules["BLOCK_ON_AGE_REVIEW"],
    )
    protection_policy["nsfw_gate"] = nsfw_gate
    print(
        f"Protection verdict: {protection_policy['verdict']} "
        f"(underage={protection_policy['underage_detected']}, nsfw={protection_policy['nsfw_detected']})"
    )

    output_path, blur_path, detect_path, log_data, rendering = render_image_outputs(
        args,
        detections,
        rules,
        blur_kernel,
        mask_shape,
        solid_color,
        protection_policy,
        protection_rules,
    )
    analysis_path = f"Logs/{os.path.basename(output_path)}.analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as analysis_file:
        json.dump(
            {
                "input": os.path.abspath(args.input),
                "checks": enabled_detectors,
                "detections": log_data,
                "demographics": demographics,
                "protection_policy": protection_policy,
                "rendering": rendering,
            },
            analysis_file,
            indent=2,
        )
    print(f"Censored image saved at: {output_path}")
    if blur_path:
        print(f"Blur image saved at: {blur_path}")
    if detect_path:
        print(f"Boxes detection image saved at: {detect_path}")
    if rendering["full_cover_applied"]:
        print(
            f"Full cover applied: mode={rendering['full_cover_mode']}, "
            f"reason={rendering['full_cover_reason']}"
        )
    print(f"Analysis JSON saved at: {analysis_path}")
    if args.fail_on_policy and protection_policy["blocked"]:
        raise SystemExit(2)
    if args.fail_on_underage and demographics.get("underage_detected"):
        raise SystemExit(3)
