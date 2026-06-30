import os
import math
import cv2
import numpy as np
import onnx
from onnx import version_converter 
import onnxruntime
import argparse
import time
from tqdm import tqdm
import subprocess
import shutil
import sys
import platform
import zipfile
import urllib.request
import tempfile
from pathlib import Path
from safevision_utils import (
    cv2_imwrite,
    create_onnx_session,
    load_blur_exception_rules,
    open_video_capture,
    parse_provider_list,
)

# Configuration variables - adjust these for different visual effects
CONFIG = {
    # Blur settings
    'BLUR_STRENGTH_NORMAL': (23, 23, 30),  # (kernel_size_x, kernel_size_y, sigma)
    'BLUR_STRENGTH_HIGH': (31, 31, 50),    # Stronger blur for more sensitive content
    'FULL_BLUR_STRENGTH': (99, 99, 75),    # Very strong blur for full frame blurring
    'ENHANCED_BLUR': False,                # When True, applies stronger blur that completely obscures content
    
    # Box colors (BGR format)
    'BOX_COLOR_NORMAL': (0, 255, 0),       # Green for normal content
    'BOX_COLOR_EXPOSED': (0, 0, 255),      # Red for exposed content
    
    # Text settings
    'FONT_SCALE': 0.5,
    'FONT_THICKNESS': 1,
    'TEXT_COLOR_NORMAL': (0, 255, 0),      # Green for normal text
    'TEXT_COLOR_EXPOSED': (0, 0, 255),     # Red for exposed text
    
    # Detection threshold
    'DETECTION_THRESHOLD': 0.2,            # Minimum confidence score for detection
    
    # Monitoring settings
    'MONITOR_THRESHOLD_PERCENT': 10.0,     # Default percentage threshold for monitoring (overridden by -r)
    'MONITOR_THRESHOLD_COUNT': 5,          # Default count threshold for monitoring (overridden by -r)
    
    # Full blur trigger
    'FULL_BLUR_LABELS': 2,                 # Number of exposed labels to trigger full blur (overridden by -fbr)
    'FULL_BLUR_FRAMES': 10,                # Minimum frames with exposed content to trigger full blur
    
    # Solid color mask (alternative to blur)
    'USE_SOLID_COLOR': False,              # When True, uses solid color instead of blur
    'SOLID_COLOR': (0, 0, 0),              # BGR color for masking (black by default)
    'MASK_SHAPE': 'rectangle',             # Region mask shape: rectangle or ellipse
    
    # Output naming
    'OUTPUT_VIDEO_SUFFIX': '_processed.mp4',
    'OUTPUT_VIDEO_BOXES_SUFFIX': '_with_boxes.mp4',
    'OUTPUT_VIDEO_AUDIO_SUFFIX': '_with_audio.mp4',
    'OUTPUT_VIDEO_BOXES_AUDIO_SUFFIX': '_with_boxes_audio.mp4',
}

# Try to import ffmpeg, but don't fail if it's not available
try:
    import ffmpeg
except ImportError:
    ffmpeg = None
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
    output_folder = output_folder or "output_frames"
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
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

# Function to create a video writer with fallback codecs
def create_safe_video_writer(output_path, width, height, fps, codec_preference=None):
    """
    Create a VideoWriter with fallback options if the preferred codec fails
    """
    # Try the specified codec first
    if codec_preference:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_preference)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Using codec: {codec_preference}")
                return writer
        except Exception as e:
            print(f"Failed with preferred codec {codec_preference}: {str(e)}")
    
    # Try a list of codecs in order
    codecs = ["mp4v", "XVID", "MJPG", "DIVX"]
    for codec in codecs:
        if codec == codec_preference:
            continue  # Skip if we already tried it
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"Using codec: {codec}")
                return writer
        except Exception as e:
            print(f"Failed with codec {codec}: {str(e)}")
    
    # If all else fails, try with default codec
    print("All codecs failed, trying with default codec (0)")
    writer = cv2.VideoWriter(output_path, 0, fps, (width, height))
    if not writer.isOpened():
        print("ERROR: Could not create video writer with any codec.")
    
    return writer

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
        # 1) locate the shipped model
        model_dir = os.path.join(os.path.dirname(__file__), "Models")
        model_orig = os.path.join(model_dir, "best.onnx")
        
        # Check if model exists, if not download it
        if not os.path.exists(model_orig):
            print("Model file not found. Creating Models directory and downloading model...")
            model_url = "https://github.com/im-syn/SafeVision/raw/refs/heads/main/Models/best.onnx"
            success = download_model(model_url, model_orig)
            if not success:
                raise FileNotFoundError(f"Could not download model from {model_url}. Please download manually and place in {model_dir}")
        
        # 2) convert/downgrade to opset15 on first run
        model_to_load = _ensure_opset15(model_orig)
        # 3) now load the compatible model
        self.onnx_session = create_onnx_session(model_to_load, providers=providers)

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

        self.blur_exception_rules = load_blur_exception_rules(rule_file_path, labels=globals()["__labels"])
        print(f"Loaded {len(self.blur_exception_rules)} exception rules from {rule_file_path}")

    def should_apply_blur(self, label):
        return self.blur_exception_rules.get(label, True)

    def detect_frame(self, frame):
        preprocessed_image, resize_factor, pad_left, pad_top = _read_frame(
            frame, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(outputs, resize_factor, pad_left, pad_top)

        return detections

    def apply_region_censor(self, image, x, y, w, h, blur_kernel):
        image_height, image_width = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + w)
        y2 = min(image_height, y + h)
        if x2 <= x1 or y2 <= y1:
            return

        roi = image[y1:y2, x1:x2]
        roi_height, roi_width = roi.shape[:2]
        if CONFIG['USE_SOLID_COLOR']:
            censored_roi = np.full((roi_height, roi_width, 3), CONFIG['SOLID_COLOR'], dtype=np.uint8)
        else:
            censored_roi = cv2.GaussianBlur(roi, (blur_kernel[0], blur_kernel[1]), blur_kernel[2])

        if CONFIG.get('MASK_SHAPE') == 'ellipse':
            mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
            center = (roi_width // 2, roi_height // 2)
            axes = (max(1, roi_width // 2), max(1, roi_height // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            roi[mask > 0] = censored_roi[mask > 0]
        else:
            image[y1:y2, x1:x2] = censored_roi

    def censor_frame(self, frame, detections, output_path, nsfw_percentage=None, force_full_blur=False, save_frame=True):
        img_boxes = frame.copy()
        img_combined = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = CONFIG['FONT_SCALE']
        font_thickness = CONFIG['FONT_THICKNESS']

        # Check if we need to apply full frame blur based on the force_full_blur flag
        if force_full_blur:
            # Check if we should use solid color or blur
            if CONFIG['USE_SOLID_COLOR']:
                # Apply a solid color to the entire frame
                height, width = img_combined.shape[:2]
                img_combined = np.full((height, width, 3), CONFIG['SOLID_COLOR'], dtype=np.uint8)
                # Also update the original frame for video output
                frame[:] = np.full((height, width, 3), CONFIG['SOLID_COLOR'], dtype=np.uint8)
            else:
                # Apply a strong blur to the entire frame to fully conceal all content
                img_combined = cv2.GaussianBlur(img_combined, 
                                              (CONFIG['FULL_BLUR_STRENGTH'][0], CONFIG['FULL_BLUR_STRENGTH'][1]), 
                                              CONFIG['FULL_BLUR_STRENGTH'][2])
                # Also update the original frame for video output
                frame[:] = cv2.GaussianBlur(frame, 
                                          (CONFIG['FULL_BLUR_STRENGTH'][0], CONFIG['FULL_BLUR_STRENGTH'][1]), 
                                          CONFIG['FULL_BLUR_STRENGTH'][2])
            
            # Add a warning text overlay
            warning_text = "Content Filtered - Excessive NSFW Content"
            text_size = cv2.getTextSize(warning_text, font, 1.0, 2)[0]
            text_x = (img_combined.shape[1] - text_size[0]) // 2
            text_y = (img_combined.shape[0] + text_size[1]) // 2
            cv2.putText(img_combined, warning_text, (text_x, text_y), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            # Also add warning text to the original frame
            cv2.putText(frame, warning_text, (text_x, text_y), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Still draw boxes on the box image for reference
            for detection in detections:
                box = detection["box"]
                x, y, w, h = box[0], box[1], box[2], box[3]
                label = detection["class"]
                is_exposed = "EXPOSED" in label
                box_color = CONFIG['BOX_COLOR_EXPOSED'] if is_exposed else CONFIG['BOX_COLOR_NORMAL']
                cv2.rectangle(img_boxes, (x, y), (x + w, y + h), box_color, 2)
        else:
            # Normal processing for individual detections
            for detection in detections:
                box = detection["box"]
                x, y, w, h = box[0], box[1], box[2], box[3]

                label = detection["class"]
                label_text = label if "EXPOSED" not in label else "Unsafe, " + label
                
                # Select colors based on content type (exposed or normal)
                is_exposed = "EXPOSED" in label
                box_color = CONFIG['BOX_COLOR_EXPOSED'] if is_exposed else CONFIG['BOX_COLOR_NORMAL']
                text_color = CONFIG['TEXT_COLOR_EXPOSED'] if is_exposed else CONFIG['TEXT_COLOR_NORMAL']
                
                # Select blur strength based on content sensitivity and enhanced blur setting
                if CONFIG['ENHANCED_BLUR'] and is_exposed:
                    blur_kernel = CONFIG['FULL_BLUR_STRENGTH']  # Use the strongest blur for enhanced mode
                else:
                    blur_kernel = CONFIG['BLUR_STRENGTH_HIGH'] if is_exposed else CONFIG['BLUR_STRENGTH_NORMAL']

                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                if x2 > x1 and y2 > y1:
                    if is_exposed and self.should_apply_blur(label):
                        self.apply_region_censor(img_combined, x, y, w, h, blur_kernel)
                        self.apply_region_censor(frame, x, y, w, h, blur_kernel)
                    else:
                        cv2.rectangle(img_boxes, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(img_boxes, label_text, (x1, max(0, y1 - 5)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                else:
                    cv2.rectangle(img_boxes, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(img_boxes, label_text, (x, y - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                # Always draw boxes and labels on combined image
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(img_combined, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(img_combined, label_text, (x1, max(0, y1 - 5)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if save_frame and output_path:
            cv2_imwrite(output_path, img_combined)
            cv2_imwrite(f"{output_path}_boxes.jpg", img_boxes)

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
                self.censor_frame(frame, detections, output_path, nsfw_percentage=100, save_frame=False)
            else:
                # Blur individual frames based on the NSFW content
                self.censor_frame(frame, detections, output_path, nsfw_percentage=nsfw_percentage, save_frame=False)

        print(f"Exposure percentage: {exposed_percentage}%")
        
    def check_exposed_count(self, detections):
        exposed_labels = [detection["class"] for detection in detections if "EXPOSED" in detection["class"]]
        exposed_count = len(exposed_labels)
        return exposed_count


class NudeVideoProcessor:
    def __init__(self, video_path, output_folder, task="video", providers=None, video_output_folder="video_output", blur_rule=0.5):
        self.task = task.lower()
        self.video_path = video_path
        self.cap = open_video_capture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        # Extract the input filename without extension for output naming
        self.input_filename = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing input file: {self.input_filename}")

        # Get the original frame rate
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps <= 0:
            self.original_fps = 30.0  # Default to 30 fps if unable to determine
        print(f"Original video FPS: {self.original_fps}")

        self.detector = NudeDetector(providers)
        self.detector.load_exception_rules("BlurException.rule")

        self.output_folder = output_folder or "output_frames"
        self.video_output_folder = video_output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.video_output_folder, exist_ok=True)

        self.blur_rule = blur_rule
        # Store command line arguments for access within class methods
        global args

    def process_video(self):
        self.original_video_path = self.video_path

        if self.task == "frames":
            self._process_frames_task()
            return

        if args and hasattr(args, 'boxes') and args.boxes:
            self._process_boxes_video_stream(include_blur=hasattr(args, 'blur') and args.blur)
        else:
            self._process_video_stream()

    def _frame_output_path(self, frame_count):
        return os.path.join(self.output_folder, f"frame_{frame_count}.jpg")

    def _empty_video_stats(self):
        return {
            "total_frames": 0,
            "total_exposed_boxes": 0,
            "frames_with_exposed": 0,
            "frames_with_required_labels": 0,
        }

    def _update_video_stats(self, stats, detections):
        exposed_count = self.check_exposed_count(detections)
        stats["total_frames"] += 1
        stats["total_exposed_boxes"] += exposed_count
        if exposed_count > 0:
            stats["frames_with_exposed"] += 1
        if exposed_count >= CONFIG['FULL_BLUR_LABELS']:
            stats["frames_with_required_labels"] += 1
        return exposed_count

    def _should_apply_full_blur_from_stats(self, stats):
        total_frames = stats["total_frames"]
        if total_frames <= 0:
            return False, "", 0

        frames_with_exposed = stats["frames_with_exposed"]
        nsfw_percentage = frames_with_exposed / total_frames * 100
        blur_rule_percentage, blur_rule_count = self.blur_rule
        threshold_percentage = blur_rule_percentage if blur_rule_percentage > 0 else CONFIG['MONITOR_THRESHOLD_PERCENT']
        threshold_count = blur_rule_count if blur_rule_count > 0 else CONFIG['MONITOR_THRESHOLD_COUNT']

        print(f"\nFull blur analysis: {stats['frames_with_required_labels']} frames with {CONFIG['FULL_BLUR_LABELS']}+ exposed labels")
        print(f"Full blur threshold: {CONFIG['FULL_BLUR_FRAMES']} frames")

        if nsfw_percentage >= threshold_percentage:
            return True, f"NSFW content ({nsfw_percentage:.1f}%) exceeds threshold ({threshold_percentage}%)", nsfw_percentage
        if frames_with_exposed >= threshold_count:
            return True, f"Frames with exposed content ({frames_with_exposed}) exceeds threshold ({threshold_count})", nsfw_percentage
        if stats["frames_with_required_labels"] >= CONFIG['FULL_BLUR_FRAMES']:
            return True, (
                f"{stats['frames_with_required_labels']} frames with {CONFIG['FULL_BLUR_LABELS']}+ exposed labels "
                f"(threshold: {CONFIG['FULL_BLUR_FRAMES']} frames)"
            ), nsfw_percentage
        if CONFIG['FULL_BLUR_FRAMES'] == 1 and stats["frames_with_required_labels"] > 0:
            return True, f"Found {stats['frames_with_required_labels']} frames with {CONFIG['FULL_BLUR_LABELS']}+ exposed labels", nsfw_percentage

        return False, "", nsfw_percentage

    def _process_frames_task(self):
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            with tqdm(total=total_frames if total_frames > 0 else None, desc="Processing Frames", unit="frames", ncols=100, mininterval=0.5) as pbar:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    detections = self.detector.detect_frame(frame)
                    self.detector.censor_frame(frame, detections, self._frame_output_path(frame_count), save_frame=True)
                    pbar.update(1)
        finally:
            self.cap.release()

    def _process_video_stream(self):
        codec_preference = args.codec if args and hasattr(args, 'codec') else "mp4v"
        output_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_SUFFIX']}"
        output_path = os.path.join(self.video_output_folder, output_filename)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats = self._empty_video_stats()
        frame_count = 0
        out = None

        try:
            with tqdm(total=total_frames if total_frames > 0 else None, desc="Processing Video", unit="frames", ncols=100, mininterval=0.5) as pbar:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if out is None:
                        height, width = frame.shape[:2]
                        out = create_safe_video_writer(output_path, width, height, self.original_fps, codec_preference)
                        if not out.isOpened():
                            print("Failed to create video writer. Check your codec installation.")
                            return

                    detections = self.detector.detect_frame(frame)
                    frame_exposed_count = self._update_video_stats(stats, detections)

                    if frame_count % 50 == 0:
                        if frame_exposed_count > 0:
                            exposed_labels = [d["class"] for d in detections if "EXPOSED" in d["class"]]
                            print(f"Frame {frame_count}: {frame_exposed_count} exposed regions - {', '.join(exposed_labels)}")
                        if frame_exposed_count >= CONFIG['FULL_BLUR_LABELS']:
                            print(f"Frame {frame_count}: Has {frame_exposed_count} exposed labels (trigger threshold: {CONFIG['FULL_BLUR_LABELS']})")

                    frame_to_process = frame.copy()
                    self.detector.censor_frame(frame_to_process, detections, None, save_frame=False)
                    out.write(frame_to_process)
                    pbar.update(1)
        finally:
            if out is not None:
                out.release()
            self.cap.release()

        if frame_count == 0:
            print("No frames were read from the input video.")
            return

        apply_full_blur, blur_reason, nsfw_percentage = self._should_apply_full_blur_from_stats(stats)
        if apply_full_blur:
            print(f"\nWARNING: {blur_reason}")
            print("Applying full video blur as per monitoring rules")
            blurred_filename = f"{self.input_filename}_fully_blurred.mp4"
            blurred_output_path = os.path.join(self.video_output_folder, blurred_filename)
            if self._create_full_blur_video_stream(blurred_output_path, nsfw_percentage):
                print(f"Fully blurred video saved to: {blurred_output_path}")

                if args and hasattr(args, 'with_audio') and args.with_audio and os.path.exists(self.video_path):
                    blurred_audio_filename = f"{self.input_filename}_fully_blurred_with_audio.mp4"
                    blurred_with_audio = os.path.join(self.video_output_folder, blurred_audio_filename)
                    success = self.add_audio_to_video(blurred_output_path, self.video_path, blurred_with_audio)
                    if success:
                        print(f"Fully blurred video with audio saved to: {blurred_with_audio}")

        if args and hasattr(args, 'with_audio') and args.with_audio and os.path.exists(self.video_path):
            audio_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_AUDIO_SUFFIX']}"
            output_with_audio = os.path.join(self.video_output_folder, audio_filename)
            success = self.add_audio_to_video(output_path, self.video_path, output_with_audio)
            if success:
                print(f"\nVideo with audio saved at: {output_with_audio}")
            else:
                print(f"\nFailed to add audio. Video saved at: {output_path}")
        else:
            print(f"\nVideo saved at: {output_path}")

        if args and hasattr(args, 'delete_frames') and args.delete_frames:
            print("No intermediate frame images were written in video mode.")

    def _create_full_blur_video_stream(self, output_path, nsfw_percentage):
        source = open_video_capture(self.video_path)
        if not source.isOpened():
            print(f"Could not reopen video for full blur pass: {self.video_path}")
            return False

        codec_preference = args.codec if args and hasattr(args, 'codec') else "mp4v"
        total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        out = None

        try:
            with tqdm(total=total_frames if total_frames > 0 else None, desc="Applying Full Video Blur", unit="frames", ncols=100, mininterval=0.5) as pbar:
                while True:
                    ret, frame = source.read()
                    if not ret:
                        break
                    if out is None:
                        height, width = frame.shape[:2]
                        out = create_safe_video_writer(output_path, width, height, self.original_fps, codec_preference)
                        if not out.isOpened():
                            print("Failed to create blurred video writer. Continuing with regular output.")
                            return False

                    blurred_frame = frame.copy()
                    self.detector.censor_frame(
                        blurred_frame,
                        [],
                        None,
                        nsfw_percentage=nsfw_percentage,
                        force_full_blur=True,
                        save_frame=False,
                    )
                    out.write(blurred_frame)
                    pbar.update(1)
        finally:
            if out is not None:
                out.release()
            source.release()

        return True

    def _render_boxed_frame(self, frame, detections, include_blur=False):
        boxed_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = CONFIG['FONT_SCALE']
        font_thickness = CONFIG['FONT_THICKNESS']
        height, width = frame.shape[:2]

        for detection in detections:
            x, y, w, h = detection["box"]
            label = detection["class"]
            label_text = label if "EXPOSED" not in label else "Unsafe, " + label
            is_exposed = "EXPOSED" in label
            box_color = CONFIG['BOX_COLOR_EXPOSED'] if is_exposed else CONFIG['BOX_COLOR_NORMAL']
            text_color = CONFIG['TEXT_COLOR_EXPOSED'] if is_exposed else CONFIG['TEXT_COLOR_NORMAL']
            blur_kernel = CONFIG['BLUR_STRENGTH_HIGH'] if is_exposed else CONFIG['BLUR_STRENGTH_NORMAL']

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            if include_blur and is_exposed and self.detector.should_apply_blur(label):
                self.detector.apply_region_censor(boxed_frame, x, y, w, h, blur_kernel)

            cv2.rectangle(boxed_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(boxed_frame, label_text, (x1, max(0, y1 - 5)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return boxed_frame

    def _process_boxes_video_stream(self, include_blur=False):
        codec_preference = args.codec if args and hasattr(args, 'codec') else "mp4v"
        boxes_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_BOXES_SUFFIX']}"
        video_output_path = os.path.join(self.video_output_folder, boxes_filename)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        out = None

        try:
            with tqdm(total=total_frames if total_frames > 0 else None, desc="Generating Video with Boxes", unit="frames", ncols=100, mininterval=0.5) as pbar:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if out is None:
                        height, width = frame.shape[:2]
                        out = create_safe_video_writer(video_output_path, width, height, self.original_fps, codec_preference)
                        if not out.isOpened():
                            print("Failed to create video writer. Check your codec installation.")
                            return

                    detections = self.detector.detect_frame(frame)
                    out.write(self._render_boxed_frame(frame, detections, include_blur=include_blur))
                    pbar.update(1)
        finally:
            if out is not None:
                out.release()
            self.cap.release()

        if frame_count == 0:
            print("No frames were read from the input video.")
            return

        if args and hasattr(args, 'with_audio') and args.with_audio and os.path.exists(self.video_path):
            boxes_audio_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_BOXES_AUDIO_SUFFIX']}"
            output_with_audio = os.path.join(self.video_output_folder, boxes_audio_filename)
            success = self.add_audio_to_video(video_output_path, self.video_path, output_with_audio)
            if success:
                print(f"\nVideo with boxes and audio saved at: {output_with_audio}")
            else:
                print(f"\nFailed to add audio. Video with boxes saved at: {video_output_path}")
        else:
            print(f"\nVideo with boxes saved at: {video_output_path}")

        if args and hasattr(args, 'delete_frames') and args.delete_frames:
            print("No intermediate frame images were written in video mode.")


 
    def create_video(self, frame_list, exposed_count, blur_rule):
        if not frame_list:
            return

        img_height, img_width = frame_list[0][0].shape[:2]
        
        # Get codec preference from command line args
        codec_preference = args.codec if args and hasattr(args, 'codec') else "mp4v"
        
        # Use input filename for output naming
        output_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_SUFFIX']}"
        output_path = os.path.join(self.video_output_folder, output_filename)
        
        # Create a safe video writer with the original FPS and preferred codec
        out = create_safe_video_writer(output_path, img_width, img_height, self.original_fps, codec_preference)
        
        if not out.isOpened():
            print("Failed to create video writer. Check your codec installation.")
            return

        total_frames = len(frame_list)
        with tqdm(total=total_frames, desc="Processing Video", unit="frames", ncols=100, mininterval=0.5) as pbar:
            for frame, detections, output_path in frame_list:
                # Create a copy of the frame to avoid modifying the original in frame_list
                frame_to_process = frame.copy()
                self.detector.censor_frame(frame_to_process, detections, output_path, nsfw_percentage=exposed_count, save_frame=False)
                out.write(frame_to_process)
                pbar.update(1)

        out.release()

        # Calculate the percentage of frames with NSFW content
        total_exposed_boxes, frames_with_exposed, nsfw_percentage = self.check_exposed_regions(frame_list)
        
        # Get monitoring thresholds from CONFIG or command-line arguments
        # If blur_rule contains actual values (not zeros), use them
        # Otherwise, fall back to CONFIG values
        blur_rule_percentage, blur_rule_count = blur_rule
        
        # Use the rule values if provided (non-zero), otherwise use config defaults
        threshold_percentage = blur_rule_percentage if blur_rule_percentage > 0 else CONFIG['MONITOR_THRESHOLD_PERCENT']
        threshold_count = blur_rule_count if blur_rule_count > 0 else CONFIG['MONITOR_THRESHOLD_COUNT']
        
        # Determine if we need to apply full video blur based on monitoring rules
        apply_full_blur = False
        blur_reason = ""
        
        # Check -r rule (percentage or count based)
        if nsfw_percentage >= threshold_percentage:
            apply_full_blur = True
            blur_reason = f"NSFW content ({nsfw_percentage:.1f}%) exceeds threshold ({threshold_percentage}%)"
        elif frames_with_exposed >= threshold_count:
            apply_full_blur = True
            blur_reason = f"Frames with exposed content ({frames_with_exposed}) exceeds threshold ({threshold_count})"
            
        # Check -fbr rule (specific label count in frames)
        should_full_blur, full_blur_reason = self.should_apply_full_blur(frame_list)
        if should_full_blur:
            apply_full_blur = True
            blur_reason = full_blur_reason
        
        # Create a new video with appropriate blurring
        if apply_full_blur:
            print(f"\nWARNING: {blur_reason}")
            print(f"Applying full video blur as per monitoring rules (-r option)")
            
            # Create a new video file with full blur applied
            blurred_filename = f"{self.input_filename}_fully_blurred.mp4"
            blurred_output_path = os.path.join(self.video_output_folder, blurred_filename)
            blurred_out = create_safe_video_writer(blurred_output_path, img_width, img_height, self.original_fps, codec_preference)
            
            if not blurred_out.isOpened():
                print("Failed to create blurred video writer. Continuing with regular output.")
            else:
                # Process each frame with full blur
                with tqdm(total=total_frames, desc="Applying Full Video Blur", unit="frames", ncols=100, mininterval=0.5) as pbar:
                    for frame, detections, output_path in frame_list:
                        # Create a blurred copy of the frame
                        blurred_frame = frame.copy()
                        # Apply the censor_frame method with force_full_blur=True
                        self.detector.censor_frame(blurred_frame, detections, output_path,
                                                  nsfw_percentage=nsfw_percentage, force_full_blur=True, save_frame=False)
                        # Write the fully blurred frame to the output video
                        blurred_out.write(blurred_frame)
                        pbar.update(1)
                
                blurred_out.release()
                print(f"Fully blurred video saved to: {blurred_output_path}")
                
                # If audio is requested, add it to the blurred video too
                if args and hasattr(args, 'with_audio') and args.with_audio and os.path.exists(self.video_path):
                    blurred_audio_filename = f"{self.input_filename}_fully_blurred_with_audio.mp4"
                    blurred_with_audio = os.path.join(self.video_output_folder, blurred_audio_filename)
                    success = self.add_audio_to_video(blurred_output_path, self.video_path, blurred_with_audio)
                    if success:
                        print(f"Fully blurred video with audio saved to: {blurred_with_audio}")
        
        # Always create a standard processed video with individual blur areas
        print("\nCreating standard processed video with individual blur areas...")
        with tqdm(total=total_frames, desc="Censoring Frames", unit="frames", ncols=100, mininterval=0.5, leave=False) as pbar:
            for frame, detections, output_path in frame_list:
                self.detector.censor_frame(frame, detections, output_path, nsfw_percentage=nsfw_percentage, save_frame=False)
                pbar.update(1)

        # Add audio from the original video if available
        if args and hasattr(args, 'with_audio') and args.with_audio and os.path.exists(self.video_path):
            audio_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_AUDIO_SUFFIX']}"
            output_with_audio = os.path.join(self.video_output_folder, audio_filename)
            success = self.add_audio_to_video(output_path, self.video_path, output_with_audio)
            if success:
                print(f"\nVideo with audio saved at: {output_with_audio}")
            else:
                print(f"\nFailed to add audio. Video saved at: {output_path}")
        else:
            print(f"\nVideo saved at: {output_path}")
            
        # Delete frame images if requested
        if args and hasattr(args, 'delete_frames') and args.delete_frames:
            self.delete_processed_frames(frame_list)

    def create_video_with_boxes(self, frame_list, include_blur=False):
        """Create a video with detection boxes from processed frames"""
        if not frame_list:
            return

        img_height, img_width = frame_list[0][0].shape[:2]
        
        # Get codec preference from command line args
        codec_preference = args.codec if args and hasattr(args, 'codec') else "mp4v"
        
        # Use input filename for output naming
        boxes_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_BOXES_SUFFIX']}"
        video_output_path = os.path.join(self.video_output_folder, boxes_filename)
        
        # Create a safe video writer with the original FPS and preferred codec
        out = create_safe_video_writer(video_output_path, img_width, img_height, self.original_fps, codec_preference)
        
        if not out.isOpened():
            print("Failed to create video writer. Check your codec installation.")
            return

        total_frames = len(frame_list)
        with tqdm(total=total_frames, desc="Generating Video with Boxes", unit="frames", ncols=100, mininterval=0.5) as pbar:
            for frame, detections, output_path in frame_list:
                # Create a frame with boxes and labels
                boxed_frame = frame.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = CONFIG['FONT_SCALE']
                font_thickness = CONFIG['FONT_THICKNESS']

                # Always draw boxes and labels
                for detection in detections:
                    box = detection["box"]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    label = detection["class"]
                    label_text = label if "EXPOSED" not in label else "Unsafe, " + label
                    
                    # Select colors based on content type (exposed or normal)
                    is_exposed = "EXPOSED" in label
                    box_color = CONFIG['BOX_COLOR_EXPOSED'] if is_exposed else CONFIG['BOX_COLOR_NORMAL']
                    text_color = CONFIG['TEXT_COLOR_EXPOSED'] if is_exposed else CONFIG['TEXT_COLOR_NORMAL']
                    
                    # Select blur strength based on content sensitivity
                    blur_kernel = CONFIG['BLUR_STRENGTH_HIGH'] if is_exposed else CONFIG['BLUR_STRENGTH_NORMAL']

                    # Make sure coordinates are within bounds
                    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and 0 <= y + h < frame.shape[0] and 0 <= x + w < frame.shape[1]:
                        # Apply blur or solid color if needed
                        if include_blur and is_exposed and self.detector.should_apply_blur(label):
                            if CONFIG['USE_SOLID_COLOR']:
                                # Apply solid color mask
                                boxed_frame[y:y + h, x:x + w] = np.full(
                                    (h, w, 3), CONFIG['SOLID_COLOR'], dtype=np.uint8
                                )
                            else:
                                # Apply blur with configured strength
                                boxed_frame[y:y + h, x:x + w] = cv2.GaussianBlur(boxed_frame[y:y + h, x:x + w], 
                                                                               (blur_kernel[0], blur_kernel[1]), 
                                                                               blur_kernel[2])
                        
                        # Always draw the box and label
                        cv2.rectangle(boxed_frame, (x, y), (x + w, y + h), box_color, 2)
                        cv2.putText(boxed_frame, label_text, (x, y - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                
                # Write the frame to the output video
                out.write(boxed_frame)
                pbar.update(1)

        out.release()
        
        # Add audio from the original video if available
        if args and hasattr(args, 'with_audio') and args.with_audio and os.path.exists(self.video_path):
            boxes_audio_filename = f"{self.input_filename}{CONFIG['OUTPUT_VIDEO_BOXES_AUDIO_SUFFIX']}"
            output_with_audio = os.path.join(self.video_output_folder, boxes_audio_filename)
            success = self.add_audio_to_video(video_output_path, self.video_path, output_with_audio)
            if success:
                print(f"\nVideo with boxes and audio saved at: {output_with_audio}")
            else:
                print(f"\nFailed to add audio. Video with boxes saved at: {video_output_path}")
        else:
            print(f"\nVideo with boxes saved at: {video_output_path}")
            
        # Delete frame images if requested
        if args and hasattr(args, 'delete_frames') and args.delete_frames:
            self.delete_processed_frames(frame_list)

    def add_audio_to_video(self, video_path, audio_source_path, output_path):
        """Add audio from the original video to the processed video using multiple methods"""
        try:
            # Make sure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create temporary audio file
            temp_audio = os.path.join(self.video_output_folder, "temp_audio.wav")
            success = False
            
            print(f"\nAdding audio from {audio_source_path} to {video_path}...")
            
            # Check if ffmpeg is installed, first try directly then using our utility
            ffmpeg_installed = False
            ffmpeg_cmd = 'ffmpeg'
            
            # If a path was provided via command line args, use it
            if args and hasattr(args, 'ffmpeg_path') and args.ffmpeg_path:
                if os.path.exists(args.ffmpeg_path):
                    ffmpeg_cmd = args.ffmpeg_path
                    print(f"Using provided FFmpeg path: {ffmpeg_cmd}")
                    ffmpeg_installed = True
                else:
                    print(f"Provided FFmpeg path does not exist: {args.ffmpeg_path}")
            
            # If not using a custom path, check if ffmpeg is in PATH
            if not ffmpeg_installed:
                try:
                    # Test if ffmpeg is available
                    test_result = subprocess.run('ffmpeg -version', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ffmpeg_installed = (test_result.returncode == 0)
                except:
                    ffmpeg_installed = False
                
            # If not found directly, try using our download utility
            if not ffmpeg_installed:
                print("Attempting to use downloaded FFmpeg...")
                ffmpeg_installed = ensure_ffmpeg()
                
            if ffmpeg_installed:
                try:
                    print("Trying to extract audio with ffmpeg subprocess...")
                    extract_cmd = f'{ffmpeg_cmd} -i "{audio_source_path}" -q:a 0 -map a "{temp_audio}" -y'
                    extract_result = subprocess.run(extract_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    if extract_result.returncode == 0:  # If extraction succeeded
                        print("Audio extraction successful, now merging...")
                        merge_cmd = f'{ffmpeg_cmd} -i "{video_path}" -i "{temp_audio}" -c:v copy -c:a aac "{output_path}" -y'
                        merge_result = subprocess.run(merge_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        success = (merge_result.returncode == 0)
                        if success:
                            print(f"Successfully created video with audio using ffmpeg")
                    else:
                        print(f"Audio extraction failed with exit code {extract_result.returncode}")
                except Exception as e:
                    print(f"Subprocess method failed: {str(e)}")
            else:
                print("ffmpeg command-line tool not found. Trying Python libraries...")
                
            # Try ffmpeg-python if ffmpeg command failed or isn't installed
            if not success:
                try:
                    # First check if the ffmpeg-python module is properly installed
                    import ffmpeg
                    print("Trying ffmpeg-python method...")
                    
                    try:
                        # Extract audio to temporary file
                        (
                            ffmpeg
                            .input(audio_source_path)
                            .output(temp_audio, acodec='pcm_s16le', ac=2)
                            .overwrite_output()
                            .run(quiet=True, capture_stdout=True, capture_stderr=True)
                        )
                        
                        # Merge audio with video
                        input_video = ffmpeg.input(video_path)
                        input_audio = ffmpeg.input(temp_audio)
                        
                        (
                            ffmpeg
                            .output(
                                input_video.video, 
                                input_audio.audio, 
                                output_path, 
                                vcodec='copy',
                                acodec='aac',
                                strict='experimental'
                            )
                            .overwrite_output()
                            .run(quiet=True, capture_stdout=True, capture_stderr=True)
                        )
                        
                        if os.path.exists(output_path):
                            print(f"Successfully created video with audio using ffmpeg-python")
                            success = True
                    except Exception as e:
                        print(f"ffmpeg-python method failed: {str(e)}")
                except ImportError:
                    print("ffmpeg-python module is not installed or properly configured")
            
            # Clean up temporary files
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except:
                    pass
            
            # If all methods failed, try using a direct copy
            if not success:
                print("All audio addition methods failed. Creating output video without audio.")
                shutil.copy2(video_path, output_path)
                return False
                
            # Verify the output file exists and has a reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"Audio processing complete. Output file: {output_path}")
                return True
            else:
                print("Failed to create output file or file is too small.")
                if os.path.exists(video_path):
                    shutil.copy2(video_path, output_path)
                return False
        except Exception as e:
            print(f"Error adding audio to video: {str(e)}")
            # If all fails, copy the video without audio
            if os.path.exists(video_path):
                shutil.copy2(video_path, output_path)
            return False
            
    # Methods for analyzing exposed content in videos
    def check_exposed_count(self, detections):
        """
        Count the number of exposed regions in the detections.
        Returns the count of exposed labels.
        """
        exposed_labels = [detection["class"] for detection in detections if "EXPOSED" in detection["class"]]
        exposed_count = len(exposed_labels)
        return exposed_count
        
    def check_exposed_regions(self, frame_list):
        """
        Analyze all detections across frames to get statistics on exposed content.
        Returns:
        - total_exposed_boxes: Total count of all exposed boxes across all frames
        - frames_with_exposed: Number of frames containing any exposed content
        - exposed_percentage: Percentage of frames containing exposed content
        """
        total_exposed_boxes = 0
        frames_with_exposed = 0
        
        for _, detections, _ in frame_list:
            frame_exposed_count = self.check_exposed_count(detections)
            total_exposed_boxes += frame_exposed_count
            if frame_exposed_count > 0:
                frames_with_exposed += 1
        
        total_frames = len(frame_list)
        exposed_percentage = (frames_with_exposed / total_frames * 100) if total_frames > 0 else 0
        
        return total_exposed_boxes, frames_with_exposed, exposed_percentage
        
    def should_apply_full_blur(self, frame_list):
        """
        Check if the full blur rule conditions are met.
        Returns:
        - should_blur: Boolean indicating if full blur should be applied
        - reason: String explaining why full blur is being applied
        """
        # Count frames that have at least FULL_BLUR_LABELS exposed labels
        frames_with_required_labels = 0
        frames_with_any_exposed = 0
        
        # Analysis of each frame
        for _, detections, _ in frame_list:
            exposed_count = self.check_exposed_count(detections)
            if exposed_count >= CONFIG['FULL_BLUR_LABELS']:
                frames_with_required_labels += 1
            if exposed_count > 0:
                frames_with_any_exposed += 1
        
        # Calculate percentages
        total_frames = len(frame_list)
        percent_with_required = (frames_with_required_labels / total_frames * 100) if total_frames > 0 else 0
        
        # Debug output to help diagnose issues
        print(f"\nFull blur analysis: {frames_with_required_labels} frames with {CONFIG['FULL_BLUR_LABELS']}+ exposed labels")
        print(f"Full blur threshold: {CONFIG['FULL_BLUR_FRAMES']} frames")
        
        # Check if we should apply full blur based on the full blur rule
        if frames_with_required_labels >= CONFIG['FULL_BLUR_FRAMES']:
            return True, f"{frames_with_required_labels} frames with {CONFIG['FULL_BLUR_LABELS']}+ exposed labels (threshold: {CONFIG['FULL_BLUR_FRAMES']} frames)"
        
        # Special case: If only a label count was provided (not frames), trigger on any matching frame
        if CONFIG['FULL_BLUR_FRAMES'] == 1 and frames_with_required_labels > 0:
            return True, f"Found {frames_with_required_labels} frames with {CONFIG['FULL_BLUR_LABELS']}+ exposed labels"
        
        # Make sure we return False if no conditions are met
        return False, ""
        
    def delete_processed_frames(self, frame_list):
        """Delete all processed frame images to save disk space"""
        try:
            print("\nCleaning up frame files...")
            # Get list of frame files from frame_list
            frame_paths = set()
            box_frame_paths = set()
            
            # Extract paths from frame_list
            for _, _, output_path in frame_list:
                if output_path and os.path.exists(output_path):
                    frame_paths.add(output_path)
                    # Also add box images
                    box_path = f"{output_path}_boxes.jpg"
                    if os.path.exists(box_path):
                        box_frame_paths.add(box_path)
                        
            # Delete all frame files
            deleted_count = 0
            for path in frame_paths:
                try:
                    os.remove(path)
                    deleted_count += 1
                except:
                    pass
            
            # Delete all box files
            for path in box_frame_paths:
                try:
                    os.remove(path)
                    deleted_count += 1
                except:
                    pass
                    
            print(f"Deleted {deleted_count} frame files to save disk space")
        except Exception as e:
            print(f"Error while cleaning up frame files: {str(e)}")

            
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
    parser.add_argument("-r", "--rule", type=parse_blur_rule, default=(0, 0), 
                      help="Blur monitoring rule in format 'percentage/count'. If percentage of frames with NSFW content exceeds the percentage value OR if total frames with NSFW content exceeds count, a fully blurred version will be created.")
    parser.add_argument("-b", "--boxes", action="store_true", help="Create a video with detection boxes from frames")
    parser.add_argument("--blur", action="store_true", help="Apply blur to detected regions when using -b option")
    parser.add_argument("-a", "--with-audio", action="store_true", help="Include original audio in the output video")
    parser.add_argument("-c", "--codec", type=str, choices=["mp4v", "avc1", "xvid", "mjpg"], default="mp4v",
                      help="Video codec to use. Default is mp4v for better compatibility.")
    parser.add_argument("--ffmpeg-path", type=str, default=None, 
                      help="Full path to the ffmpeg executable (e.g., 'C:/ffmpeg/bin/ffmpeg.exe')")
    parser.add_argument("-df", "--delete-frames", action="store_true",
                      help="Delete frame images after building the video to save disk space")
    parser.add_argument("--enhanced-blur", action="store_true",
                      help="Use enhanced blur that completely obscures content (stronger blur effect)")
    parser.add_argument("-fbr", "--full-blur-rule", type=str, default=None,
                      help="Full blur rule in format 'labels/frames'. If exposed labels >= labels count in at least frames count, a fully blurred video will be created.")
    parser.add_argument("--color", action="store_true",
                      help="Use solid color instead of blur to mask NSFW content")
    parser.add_argument("--mask-color", type=str, default="0,0,0", 
                      help="Color to use for masking in BGR format (blue,green,red). Default is black: '0,0,0'")
    parser.add_argument("--mask-shape", type=str, choices=["rectangle", "ellipse"], default="rectangle",
                      help="Shape used for regional censoring masks. Default is rectangle.")
    parser.add_argument("--providers", type=str, default=None,
                      help="Comma-separated ONNX Runtime providers, e.g. CUDAExecutionProvider,CPUExecutionProvider")
    return parser.parse_args()

# Global args variable for access across classes
args = None

def download_ffmpeg_windows():
    """Download and extract FFmpeg for Windows"""
    print("Downloading FFmpeg for Windows...")
    
    # Create a directory for FFmpeg
    ffmpeg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg")
    os.makedirs(ffmpeg_dir, exist_ok=True)
    
    # Download the FFmpeg zip file
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    temp_zip = os.path.join(tempfile.gettempdir(), "ffmpeg.zip")
    
    try:
        print(f"Downloading FFmpeg from {ffmpeg_url}...")
        urllib.request.urlretrieve(ffmpeg_url, temp_zip)
        
        print("Extracting FFmpeg...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
            
        # Find the extracted directory (it may have a version number in the name)
        extracted_dirs = [d for d in os.listdir(ffmpeg_dir) if os.path.isdir(os.path.join(ffmpeg_dir, d)) and d.startswith("ffmpeg")]
        
        if extracted_dirs:
            extracted_dir = os.path.join(ffmpeg_dir, extracted_dirs[0])
            bin_dir = os.path.join(extracted_dir, "bin")
            
            # Add to PATH temporarily for this session
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
            
            print(f"FFmpeg has been downloaded and extracted to {bin_dir}")
            print("FFmpeg is now available for this session")
            
            # Test if ffmpeg works
            try:
                subprocess.run([os.path.join(bin_dir, "ffmpeg"), "-version"], check=True, stdout=subprocess.PIPE)
                print("FFmpeg is working correctly")
                return bin_dir
            except subprocess.CalledProcessError:
                print("FFmpeg was downloaded but failed to run")
        else:
            print("Could not find the extracted FFmpeg directory")
    except Exception as e:
        print(f"Error downloading or extracting FFmpeg: {str(e)}")
    
    return None

def find_ffmpeg_in_path():
    """Try to find ffmpeg in the system PATH"""
    # Check if ffmpeg is available in the PATH
    for path in os.environ["PATH"].split(os.pathsep):
        ffmpeg_exe = os.path.join(path, "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
        if os.path.exists(ffmpeg_exe) and os.access(ffmpeg_exe, os.X_OK):
            print(f"Found FFmpeg in PATH: {ffmpeg_exe}")
            return ffmpeg_exe
    return None

def ensure_ffmpeg(ffmpeg_path=None):
    """Ensure FFmpeg is available, downloading if necessary or using provided path"""
    # First check if ffmpeg is in the PATH
    path_ffmpeg = find_ffmpeg_in_path()
    if path_ffmpeg:
        return True
    
    # If a path is provided, check if it's valid
    if ffmpeg_path:
        try:
            if os.path.exists(ffmpeg_path):
                # Add the directory containing ffmpeg to the PATH
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
                print(f"Using provided FFmpeg path: {ffmpeg_path}")
                # Test if it works
                subprocess.run([ffmpeg_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return True
            else:
                print(f"Provided FFmpeg path does not exist: {ffmpeg_path}")
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"Provided FFmpeg path is invalid: {ffmpeg_path}")
    
    # If not found in PATH or provided path, try the default command
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg is available via command line")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("FFmpeg is not installed or not in PATH")
        
    # As a last resort, download FFmpeg based on platform
    if platform.system() == "Windows":
        print("FFmpeg not found in system. Attempting to download...")
        ffmpeg_path = download_ffmpeg_windows()
        return ffmpeg_path is not None
    else:
        print(f"Automatic FFmpeg installation is not supported on {platform.system()}")
        print("Please install FFmpeg manually. See FFMPEG_INSTALL.md for instructions.")
        return False

def check_ffmpeg_availability(ffmpeg_path=None):
    """Check and try to set up FFmpeg if needed"""
    ffmpeg_available = False
    
    # Try to run ffmpeg directly or use provided path
    ffmpeg_available = ensure_ffmpeg(ffmpeg_path)
    
    if not ffmpeg_available:
        print("\nWARNING: FFmpeg is not available. Audio processing will be limited.")
        print("To enable full audio support, install FFmpeg using the instructions in FFMPEG_INSTALL.md\n")
    
    return ffmpeg_available

if __name__ == "__main__":
    create_directories()  # Create directories before processing
    args = parse_args()

    video_output_folder = args.video_output
    rule = args.rule
    
    # Apply enhanced blur setting if requested
    if args.enhanced_blur:
        CONFIG['ENHANCED_BLUR'] = True
        print("Enhanced blur mode enabled - stronger blur will be applied")
    
    # Parse and set full blur rule if provided
    # The argument name uses hyphen, but Python converts to underscore
    if hasattr(args, 'full_blur_rule') and args.full_blur_rule:
        try:
            print(f"Processing full blur rule: {args.full_blur_rule}")
            parts = args.full_blur_rule.split('/')
            if len(parts) == 2:
                CONFIG['FULL_BLUR_LABELS'] = int(parts[0])
                CONFIG['FULL_BLUR_FRAMES'] = int(parts[1])
                print(f"Full blur rule set: {CONFIG['FULL_BLUR_LABELS']} exposed labels in at least {CONFIG['FULL_BLUR_FRAMES']} frames will trigger full blur")
            elif len(parts) == 1:
                CONFIG['FULL_BLUR_LABELS'] = int(parts[0])
                # When only label count is provided, set frames to 1 to trigger on any single frame
                CONFIG['FULL_BLUR_FRAMES'] = 1
                print(f"Full blur rule set: {CONFIG['FULL_BLUR_LABELS']} exposed labels in any frame will trigger full blur")
        except ValueError:
            print("Invalid full blur rule format. Using default values.")
    
    # Apply solid color masking if requested
    if args.color:
        CONFIG['USE_SOLID_COLOR'] = True
        try:
            color_parts = args.mask_color.split(',')
            if len(color_parts) == 3:
                CONFIG['SOLID_COLOR'] = (int(color_parts[0]), int(color_parts[1]), int(color_parts[2]))
                print(f"Using solid color masking with color: BGR={CONFIG['SOLID_COLOR']}")
        except ValueError:
            print("Invalid color format. Using default black color.")

    CONFIG['MASK_SHAPE'] = args.mask_shape
    print(f"Using {CONFIG['MASK_SHAPE']} regional mask shape")
    
    # Check if we need FFmpeg for this run
    if args.task == "video" and args.with_audio:
        check_ffmpeg_availability(args.ffmpeg_path)

    if args.task == "video":
        video_processor = NudeVideoProcessor(
            args.input,
            args.output,
            task=args.task,
            providers=parse_provider_list(args.providers),
            video_output_folder=video_output_folder,
            blur_rule=rule,
        )
        video_processor.process_video()
    elif args.task == "frames":
        detector = NudeDetector(providers=parse_provider_list(args.providers))
        detector.load_exception_rules("BlurException.rule")
        process_frames(args.input, detector, args.output)
