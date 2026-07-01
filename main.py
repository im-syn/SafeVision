import os
import math
import cv2
import numpy as np
import onnx
from onnx import version_converter 
import onnxruntime
import argparse
from safevision_utils import (
    apply_region_censor,
    cv2_imread,
    cv2_imwrite,
    create_onnx_session,
    detection_is_censorable,
    load_blur_exception_rules,
    make_blur_kernel,
    normalize_mask_shape,
    parse_provider_list,
    parse_detector_selection,
)
from object_detector import DEFAULT_OBJECT_LABELS, DEFAULT_OBJECT_MODEL, ObjectContentDetector

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
    def __init__(self, providers=None):
        # Set up model paths
        model_dir = os.path.join(os.path.dirname(__file__), "Models")
        model_orig = os.path.join(model_dir, "best.onnx")
        
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
        help="Number of exposed boxes to trigger full image blur",
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
        "--detectors",
        type=str,
        default="nude",
        help="Detector set to use: nude, objects, or both. Comma-separated values are accepted.",
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


def _run_selected_detectors(image_path, enabled_detectors, providers, object_model, object_labels, object_threshold):
    detections = []
    if "nude" in enabled_detectors:
        detector = NudeDetector(providers=providers)
        detections.extend(detector.detect(image_path))

    if "objects" in enabled_detectors:
        object_detector = ObjectContentDetector(
            model_path=object_model,
            labels_path=object_labels,
            providers=providers,
            threshold=object_threshold,
        )
        detections.extend(object_detector.detect_image(image_path))

    return detections


def _draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (x, max(12, y - 5)), font, 0.5, color, 1, cv2.LINE_AA)


def render_image_outputs(args, detections, rules, blur_kernel, mask_shape, solid_color):
    img = cv2_imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    img_blur = img.copy()
    img_boxes = img.copy()
    img_combined = img.copy()
    image_height, image_width = img.shape[:2]
    censorable_count = 0
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
        should_blur = rules.get(label, True)
        color = (0, 0, 255) if is_censorable else (0, 255, 0)
        label_text = f"{label} {score:.2f}"
        if is_censorable and "EXPOSED" in str(label).upper():
            label_text = f"Unsafe, {label} {score:.2f}"

        log_data.append(
            {
                "label": label,
                "score": score,
                "category": category,
                "source": source,
                "box": [x1, y1, x2 - x1, y2 - y1],
                "censor": is_censorable,
                "blur": bool(should_blur),
            }
        )

        if is_censorable:
            censorable_count += 1

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

        cv2.rectangle(img_boxes, (x1, y1), (x2, y2), color, 2)
        _draw_label(img_boxes, label_text, x1, y1, color)
        cv2.rectangle(img_combined, (x1, y1), (x2, y2), color, 2)
        _draw_label(img_combined, label_text, x1, y1, color)

    if args.full_blur_rule > 0 and censorable_count >= args.full_blur_rule:
        if args.color:
            img_blur[:, :] = solid_color
            if args.blur:
                img_combined[:, :] = solid_color
        else:
            img_blur = cv2.GaussianBlur(img_blur, (blur_kernel[0], blur_kernel[1]), blur_kernel[2])
            if args.blur:
                img_combined = cv2.GaussianBlur(img_combined, (blur_kernel[0], blur_kernel[1]), blur_kernel[2])

    output_path = args.output
    if not output_path:
        input_path, ext = os.path.splitext(args.input)
        suffix = "_Blur" if args.blur else "_Detect"
        output_path = f"output/{os.path.basename(input_path)}{suffix}{ext}"

    blur_path = f"Blur/{os.path.basename(output_path)}"
    detect_path = f"Prosses/{os.path.basename(output_path)}"
    final_image = img_combined if args.blur else img_boxes

    cv2_imwrite(output_path, final_image)
    cv2_imwrite(blur_path, img_blur)
    cv2_imwrite(detect_path, img_boxes)

    log_file_path = f"Logs/{os.path.basename(output_path)}.log"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        for item in log_data:
            log_file.write(
                f"Label: {item['label']}, Score: {item['score']:.4f}, "
                f"Category: {item['category']}, Source: {item['source']}, "
                f"Censor: {item['censor']}, Blur: {item['blur']}, Box: {item['box']}\n"
            )

    return output_path, blur_path, detect_path

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
    providers = parse_provider_list(args.providers)

    detections = _run_selected_detectors(
        args.input,
        enabled_detectors,
        providers,
        args.object_model,
        args.object_labels,
        args.object_threshold,
    )
    print(f"Detections found: {len(detections)}")

    output_path, blur_path, detect_path = render_image_outputs(
        args,
        detections,
        rules,
        blur_kernel,
        mask_shape,
        solid_color,
    )
    print(f"Censored image saved at: {output_path}")
    print(f"Blur image saved at: {blur_path}")
    print(f"Boxes detection image saved at: {detect_path}")
