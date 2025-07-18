import os
import math
import cv2
import numpy as np
import onnx
from onnx import version_converter 
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import argparse
import time
from tqdm import tqdm
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

def process_frames(video_path, detector, output_folder):
    """
    Read a video, run the detector on each frame,
    censor/save each frame to output_folder.
    """
    import os
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # detect + censor
        dets = detector.detect_frame(frame)
        out_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
        detector.censor_frame(frame, dets, out_path)

    cap.release()
def _read_frame(frame, target_size=320):
    img_height, img_width = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    image_data = img.astype("float32") / 255.0
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
        detections.append(
            {"class": __labels[class_id], "score": float(score), "box": box}
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
        model     = onnx.load(original_path)
        converted = version_converter.convert_version(model, 15)
        onnx.save(converted, conv_path)
    return conv_path


class NudeDetector:
    def __init__(self, providers=None):
        # 1) locate the shipped model
        model_orig   = os.path.join(os.path.dirname(__file__), "Models/best.onnx")
        # 2) convert/downgrade to opset15 on first run
        model_to_load = _ensure_opset15(model_orig)
        # 3) now load the compatible model
        self.onnx_session = onnxruntime.InferenceSession(
            model_to_load,
            providers=C.get_available_providers() if not providers else providers,
        )

        # 4) pull out input shape & name as before
        inp = self.onnx_session.get_inputs()[0]
        self.input_name   = inp.name
        self.input_width  = inp.shape[2]  # 320
        self.input_height = inp.shape[3]  # 320

        # Initialize exception rules to None
        self.blur_exception_rules = None

    def load_exception_rules(self, rule_file_path):
        if not rule_file_path:
            rule_file_path = "BlurException.rule"
            with open(rule_file_path, "w") as default_rule_file:
                for label in __labels:
                    default_rule_file.write(f"{label} = true\n")

        self.blur_exception_rules = {}
        with open(rule_file_path, "r") as rule_file:
            for line in rule_file:
                parts = line.strip().split("=")
                if len(parts) == 2:
                    label, blur = parts[0].strip(), parts[1].strip()
                    self.blur_exception_rules[label] = blur.lower() == "true"

    def should_apply_blur(self, label):
        return self.blur_exception_rules.get(label, True)

    def detect_frame(self, frame):
        preprocessed_image, resize_factor, pad_left, pad_top = _read_frame(
            frame, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs, resize_factor, pad_left, pad_top)

        return detections

    def censor_frame(self, frame, detections, output_path, nsfw_percentage=None):
        img_boxes = frame.copy()
        img_combined = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]

            label = detection["class"]
            label_text = label if "EXPOSED" not in label else "Unsafe, " + label

            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and 0 <= y + h < frame.shape[0] and 0 <= x + w < frame.shape[1]:
                if "EXPOSED" in label and self.should_apply_blur(label):
                    if nsfw_percentage is not None and nsfw_percentage >= 100:
                        frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (23, 23), 30)
                    else:
                        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img_boxes, label_text, (x, y - 5), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
                else:
                    cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img_boxes, label_text, (x, y - 5), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

                cv2.rectangle(img_combined, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_combined, label_text, (x, y - 5), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        # Save frames to the "output_frames" folder instead of the provided output path
        output_path = os.path.join("output_frames", f"{os.path.basename(output_path)}")
        cv2.imwrite(output_path, img_combined)
        cv2.imwrite(f"{output_path}_boxes.jpg", img_boxes)

       # print(f"Processed frame {output_path}")
        
    def blur_all_frames(self, frame_list, nsfw_percentage=None):
        exposed_frame_count = 0

        for _, detections, _ in frame_list:
            exposed_count = self.check_exposed_count(detections)
            if exposed_count >= 2:
                exposed_frame_count += 1

        total_frames = len(frame_list)
        exposed_percentage = (exposed_frame_count / total_frames) * 100

        for frame, detections, output_path in frame_list:
            if exposed_percentage >= nsfw_percentage:
                # Apply full blur to the whole image if the condition is met
                self.censor_frame(frame, detections, output_path, nsfw_percentage=100)
            else:
                # Blur individual frames based on the NSFW content
                self.censor_frame(frame, detections, output_path, nsfw_percentage=nsfw_percentage)

        print(f"Exposure percentage: {exposed_percentage}%")
        
    def check_exposed_count(self, detections):
        exposed_labels = [detection["class"] for detection in detections if "EXPOSED" in detection["class"]]
        exposed_count = len(exposed_labels)
        return exposed_count


class NudeVideoProcessor:
    def __init__(self, video_path, output_folder, task="video", providers=None, video_output_folder="video_output", blur_rule=0.5):
        self.task = task.lower()
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        self.detector = NudeDetector(providers)
        self.detector.load_exception_rules("BlurException.rule")

        self.output_folder = output_folder
        self.video_output_folder = video_output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.video_output_folder, exist_ok=True)

        self.blur_rule = blur_rule

    def process_video(self):
        frame_count = 0
        frame_list = []
        exposed_count = 0

        with tqdm(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Frames", unit="frames", ncols=100, mininterval=0.5) as pbar:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    break

                frame_count += 1
                detections = self.detector.detect_frame(frame)
                output_path = os.path.join(self.output_folder, f"frame_{frame_count}.jpg")

                if self.task == "frames":
                    self.detector.censor_frame(frame, detections, output_path)
                else:
                    frame_list.append((frame.copy(), detections, output_path))
                    exposed_count += self.check_exposed_count(detections)

                pbar.update(1)

        self.cap.release()

        if self.task == "video":
            self.create_video(frame_list, exposed_count, self.blur_rule)


 
    def create_video(self, frame_list, exposed_count, blur_rule):
        if not frame_list:
            return

        img_height, img_width = frame_list[0][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(self.video_output_folder, "output_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (img_width, img_height))

        total_frames = len(frame_list)
        with tqdm(total=total_frames, desc="Processing Video", unit="frames", ncols=100, mininterval=0.5) as pbar:
            for frame, detections, output_path in frame_list:
                self.detector.censor_frame(frame, detections, output_path, nsfw_percentage=exposed_count)
                out.write(frame)
                pbar.update(1)

        out.release()

        # Calculate the percentage of frames with NSFW content
        nsfw_percentage = exposed_count / total_frames * 100

        # Check if the conditions for blurring are met
        blur_rule_percentage, blur_rule_count = blur_rule
        if nsfw_percentage >= blur_rule_percentage or exposed_count >= blur_rule_count:
            # Blur all frames when either condition is met
            self.detector.blur_all_frames(frame_list, nsfw_percentage=nsfw_percentage)
        else:
            # Blur individual frames based on the NSFW content
            with tqdm(total=total_frames, desc="Censoring Frames", unit="frames", ncols=100, mininterval=0.5, leave=False) as pbar:
                for frame, detections, output_path in frame_list:
                    self.detector.censor_frame(frame, detections, output_path, nsfw_percentage=nsfw_percentage)
                    pbar.update(1)

        print(f"\nVideo saved at: {output_path}")

    # Add this method to NudeVideoProcessor class
    def check_exposed_count(self, detections):
        exposed_labels = [detection["class"] for detection in detections if "EXPOSED" in detection["class"]]
        exposed_count = len(exposed_labels)
        return exposed_count

            
def parse_blur_rule(value):
    parts = value.split('/')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Invalid rule format. Use 'percentage/count'.")
    
    try:
        percentage = float(parts[0])
        count = int(parts[1])
        return percentage, count
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid percentage or count value.")


def create_directories():
    # Create directories if they don't exist
    os.makedirs("video_outputs", exist_ok=True)
    os.makedirs("output_frames", exist_ok=True)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Nude Detector")
    parser.add_argument("-i", "--input", type=str, help="Path to the input video", required=True)
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save the censored frames or video. If not provided, a default path will be used.")
    parser.add_argument("-t", "--task", type=str, choices=["frames", "video"], default="video", help="Specify the task (frames or video)")
    parser.add_argument("-vo", "--video_output", type=str, default="video_output", help="Path to the video output folder. Default is 'video_output'")
    parser.add_argument("-r", "--rule", type=parse_blur_rule, default=(0, 0), help="Blur rule in the format 'percentage/count'")
    return parser.parse_args()

if __name__ == "__main__":
    create_directories()  # Create directories before processing
    args = parse_args()

    video_output_folder = args.video_output
    rule = args.rule

    if args.task == "video":
        video_processor = NudeVideoProcessor(args.input, args.output, task=args.task, video_output_folder=video_output_folder, blur_rule=rule)
        video_processor.process_video()
    elif args.task == "frames":
        detector = NudeDetector()
        detector.load_exception_rules("BlurException.rule")
        process_frames(args.input, detector, args.output)
