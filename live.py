#!/usr/bin/env python3
"""
Live Nudity Detection and Censoring
Real-time camera processing for nudity detection with live censoring capabilities.
"""

import cv2
import numpy as np
import os
import sys
import argparse
import time
import threading
import queue
from pathlib import Path
import urllib.request
import math
from safevision_utils import create_onnx_session, load_blur_exception_rules

# Global variables for ONNX Runtime components
onnxruntime = None
onnx = None
version_converter = None
_import_attempted = False
_import_successful = False

def _safe_import_onnx(force_retry=False):
    """Safely import ONNX Runtime with comprehensive error handling."""
    global onnxruntime, onnx, version_converter, _import_attempted, _import_successful
    
    if _import_attempted and not force_retry:
        print(f"DEBUG: Using cached ONNX import result: {_import_successful}")
        return _import_successful
    
    if force_retry:
        print("DEBUG: Forcing ONNX Runtime import retry...")
        _import_attempted = True  # Reset but allow retry
    else:
        _import_attempted = True
    
    try:
        # Try importing with different approaches
        import sys
        import importlib
        
        print("DEBUG: Starting ONNX Runtime import attempt...")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: Python executable: {sys.executable}")
        
        # Method 1: Direct import
        try:
            print("DEBUG: Method 1 - Direct import...")
            import onnxruntime as ort
            print(f"DEBUG: onnxruntime imported successfully, version: {ort.__version__}")
            
            import onnx as onnx_module
            print("DEBUG: onnx module imported successfully")
            
            from onnx import version_converter as vc
            print("DEBUG: version_converter imported successfully")
            
            # Test basic functionality
            providers = ort.get_available_providers()
            print(f"DEBUG: Available providers: {providers}")
            if not providers:
                raise Exception("No ONNX Runtime providers available")
            
            # Test session creation
            test_session = None
            try:
                # Create a minimal test to ensure DLL works
                model_path = None  # We'll skip this test for now
                if model_path and os.path.exists(model_path):
                    test_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            except Exception as session_e:
                print(f"DEBUG: Session test failed (expected): {session_e}")
                pass  # Skip session test if no model available
            finally:
                if test_session:
                    del test_session
            
            # If we get here, everything works
            onnxruntime = ort
            onnx = onnx_module
            version_converter = vc
            _import_successful = True
            print("DEBUG: Method 1 successful - ONNX Runtime fully imported!")
            return True
            
        except Exception as e1:
            print(f"DEBUG: Method 1 failed: {type(e1).__name__}: {e1}")
            # Method 2: Try with specific provider
            try:
                print("DEBUG: Method 2 - Trying with specific provider...")
                import onnxruntime as ort
                providers = ort.get_available_providers()
                print(f"DEBUG: Method 2 providers: {providers}")
                if 'CPUExecutionProvider' in providers:
                    import onnx as onnx_module
                    from onnx import version_converter as vc
                    onnxruntime = ort
                    onnx = onnx_module
                    version_converter = vc
                    _import_successful = True
                    print("DEBUG: Method 2 successful - ONNX Runtime imported with CPU provider!")
                    return True
                else:
                    raise Exception("CPU provider not available")
            except Exception as e2:
                print(f"DEBUG: Method 2 failed: {type(e2).__name__}: {e2}")
                # Both methods failed
                raise e1  # Raise the original error
                
    except ImportError as e:
        # Module not installed
        print(f"DEBUG: ImportError - ONNX Runtime not installed: {e}")
        _import_successful = False
        return False
    except Exception as e:
        # DLL or other errors
        print(f"DEBUG: Exception during ONNX import: {type(e).__name__}: {e}")
        import traceback
        print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
        _import_successful = False
        return False
    
    print("DEBUG: ONNX import failed - reached end of function")
    return False

def _get_onnx_status():
    """Get current ONNX Runtime status."""
    if not _import_attempted:
        print("DEBUG: _get_onnx_status calling _safe_import_onnx for first time")
        _safe_import_onnx()
    
    print(f"DEBUG: _get_onnx_status - _import_successful={_import_successful}, onnxruntime={onnxruntime is not None}")
    
    if _import_successful and onnxruntime is not None:
        try:
            providers = onnxruntime.get_available_providers()
            print(f"DEBUG: Got providers successfully: {providers}")
            return True, f"Available providers: {providers}"
        except Exception as e:
            print(f"DEBUG: Error getting providers: {e}")
            return False, "ONNX Runtime imported but providers not accessible"
    else:
        return False, "ONNX Runtime not available"

# Try initial import (silent)
_safe_import_onnx()

# Try to import PIL for image processing
try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not available, some features may not work")
    Image = None

# Configuration for live processing - tweak these if you want different behavior
LIVE_CONFIG = {
    # Blur settings - these are the kernel sizes for different blur levels
    'BLUR_STRENGTH': 25,                      # Default blur strength (single value for easier math)
    'BLUR_STRENGTH_CRITICAL': 35,            # Max blur for the really bad stuff
    'ENHANCED_BLUR': False,                   # Fancy blur mode if you want to get creative
    
    # Detection thresholds - mess with these to make detection more/less strict
    'DETECTION_THRESHOLD': 0.25,             # Base threshold - lower = more sensitive
    'EXPOSED_THRESHOLD': 0.4,                # Higher bar for calling something "exposed" 
    'GENITALIA_THRESHOLD': 0.6,              # Even higher bar for the explicit stuff
    'FACE_THRESHOLD': 0.7,                   # High threshold to avoid false face positives
    'TARGET_SIZE': 320,                       # Model input size - don't change this
    'FPS_TARGET': 30,                         # What we're aiming for performance-wise
    
    # UI colors (BGR format because OpenCV is weird like that)
    'BOX_COLOR_NORMAL': (0, 255, 0),         # Green boxes for normal detections
    'BOX_COLOR_EXPOSED': (0, 0, 255),        # Red boxes for the spicy content
    
    # Text styling
    'FONT_SCALE': 0.6,
    'FONT_THICKNESS': 2,
    'TEXT_COLOR_NORMAL': (0, 255, 0),
    'TEXT_COLOR_EXPOSED': (0, 0, 255),
    
    # Performance tweaks
    'SKIP_FRAMES': 2,                         # Skip frames to keep things running smooth
    'BUFFER_SIZE': 3,                         # How many frames to queue up
    
    # Alert system 
    'ALERT_THRESHOLD': 3,                     # How many frames in a row before we scream
    'ALERT_COOLDOWN': 5.0,                   # Seconds to chill between alerts
    
    # Recording stuff
    'AUTO_RECORD_ON_DETECTION': False,       # Auto-record when shit goes down
    'RECORD_BUFFER_SECONDS': 5,              # Buffer time for recordings
    
    # Privacy and masking options
    'PRIVACY_MODE': False,                    # Black screen mode for the paranoid
    'SOLID_COLOR_MASK': False,               # Use solid color instead of blur
    'MASK_COLOR': (0, 0, 0),                 # What color to use for masking
}

# Labels from the original model with severity classification
__labels = [
    "FEMALE_GENITALIA_COVERED",    # Safe
    "FACE_FEMALE",                 # Safe  
    "BUTTOCKS_EXPOSED",            # Moderate
    "FEMALE_BREAST_EXPOSED",       # High
    "FEMALE_GENITALIA_EXPOSED",    # Critical
    "MALE_BREAST_EXPOSED",         # Low
    "ANUS_EXPOSED",                # High
    "FEET_EXPOSED",                # Low
    "BELLY_COVERED",               # Safe
    "FEET_COVERED",                # Safe
    "ARMPITS_COVERED",             # Safe
    "ARMPITS_EXPOSED",             # Low
    "FACE_MALE",                   # Safe
    "BELLY_EXPOSED",               # Low
    "MALE_GENITALIA_EXPOSED",      # Critical
    "ANUS_COVERED",                # Safe
    "FEMALE_BREAST_COVERED",       # Safe
    "BUTTOCKS_COVERED",            # Safe
]

# Define severity levels for different content types
CONTENT_SEVERITY = {
    # Critical - Requires highest confidence and strongest censoring
    'CRITICAL': [
        'FEMALE_GENITALIA_EXPOSED',
        'MALE_GENITALIA_EXPOSED'
    ],
    # High - Significant nudity
    'HIGH': [
        'FEMALE_BREAST_EXPOSED', 
        'ANUS_EXPOSED'
    ],
    # Moderate - Partial nudity
    'MODERATE': [
        'BUTTOCKS_EXPOSED'
    ],
    # Low - Minor exposure
    'LOW': [
        'MALE_BREAST_EXPOSED',
        'BELLY_EXPOSED',
        'ARMPITS_EXPOSED',
        'FEET_EXPOSED'
    ],
    # Safe - Non-sexual content
    'SAFE': [
        'FACE_FEMALE',
        'FACE_MALE',
        'FEMALE_GENITALIA_COVERED',
        'BELLY_COVERED',
        'FEET_COVERED',
        'ARMPITS_COVERED',
        'ANUS_COVERED',
        'FEMALE_BREAST_COVERED',
        'BUTTOCKS_COVERED'
    ]
}

def get_content_severity(label):
    """Get the severity level of detected content."""
    for severity, labels in CONTENT_SEVERITY.items():
        if label in labels:
            return severity
    return 'UNKNOWN'

def get_confidence_threshold(label):
    """Get the appropriate confidence threshold based on content type."""
    severity = get_content_severity(label)
    
    if severity == 'CRITICAL':
        return LIVE_CONFIG['GENITALIA_THRESHOLD']
    elif label in ['FACE_FEMALE', 'FACE_MALE']:
        return LIVE_CONFIG['FACE_THRESHOLD']
    elif severity in ['HIGH', 'MODERATE']:
        return LIVE_CONFIG['EXPOSED_THRESHOLD']
    else:
        return LIVE_CONFIG['DETECTION_THRESHOLD']

def download_model(url, save_path):
    """Download the ONNX model from the provided URL and save it to the specified path."""
    print(f"Downloading model from {url}...")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        urllib.request.urlretrieve(url, save_path)
        print(f"Model downloaded successfully to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

def _ensure_opset15(original_path: str) -> str:
    """Convert ONNX model to opset 15 if needed."""
    base, ext = os.path.splitext(original_path)
    conv_path = f"{base}_opset15{ext}"
    if not os.path.exists(conv_path):
        if onnx is None or version_converter is None:
            if not _safe_import_onnx():
                raise RuntimeError("ONNX not available for model conversion")
        model = onnx.load(original_path)
        converted = version_converter.convert_version(model, 15)
        onnx.save(converted, conv_path)
    return conv_path

def _read_frame_live(frame, target_size=320):
    """Optimized frame preprocessing for live processing."""
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
        img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    img = cv2.resize(img, (target_size, target_size))
    
    image_data = img.astype("float32") / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    return image_data, resize_factor, pad_left, pad_top

def _postprocess_live(output, resize_factor, pad_left, pad_top):
    """Enhanced postprocessing for live detection with severity-based filtering."""
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        class_id = np.argmax(classes_scores)
        
        # Get the label for this detection
        if class_id < len(__labels):
            label = __labels[class_id]
            required_threshold = get_confidence_threshold(label)
            
            # Only accept detections that meet the severity-specific threshold
            if max_score >= required_threshold:
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int(round((x - w * 0.5 - pad_left) * resize_factor))
                top = int(round((y - h * 0.5 - pad_top) * resize_factor))
                width = int(round(w * resize_factor))
                height = int(round(h * resize_factor))
                
                # Additional validation for critical content
                severity = get_content_severity(label)
                if severity == 'CRITICAL':
                    # Extra strict validation for genitalia detection
                    # Require even higher confidence and reasonable bounding box size
                    min_box_area = 500  # Minimum area to avoid tiny false positives
                    box_area = width * height
                    if max_score >= 0.7 and box_area >= min_box_area:
                        class_ids.append(class_id)
                        scores.append(max_score)
                        boxes.append([left, top, width, height])
                elif severity == 'HIGH':
                    # Strict validation for high-severity content
                    if max_score >= 0.5:
                        class_ids.append(class_id)
                        scores.append(max_score)
                        boxes.append([left, top, width, height])
                else:
                    # Standard validation for other content
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([left, top, width, height])

    # Use stricter NMS parameters for better filtering
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.4)

    detections = []
    if len(indices) > 0:
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            label = __labels[class_id]
            severity = get_content_severity(label)
            
            detections.append({
                "class": label,
                "score": float(score),
                "box": box,
                "severity": severity
            })

    return detections

class LiveNudeDetector:
    def __init__(self, providers=None):
        print("Initializing Live Nude Detector...")
        
        # Force ONNX Runtime import with retry
        print("Attempting to import ONNX Runtime...")
        import_success = _safe_import_onnx(force_retry=True)
        print(f"ONNX import result: {import_success}")
        
        # Check ONNX Runtime availability
        success, status_msg = _get_onnx_status()
        print(f"ONNX status check: success={success}, msg={status_msg}")
        
        if not success:
            print(f"Warning: {status_msg}")
            print("Live detection will be disabled")
            self.onnx_session = None
            self.input_name = None
            self.input_width = 320
            self.input_height = 320
            self.blur_exception_rules = None
            return
        
        try:
            # Get script directory
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            
            print(f"Script directory: {script_dir}")
            
            # Model setup
            model_dir = os.path.join(script_dir, "Models")
            model_orig = os.path.join(model_dir, "best.onnx")
            
            print(f"Looking for model at: {model_orig}")
            print(f"Model exists: {os.path.exists(model_orig)}")
            
            # Download model if not exists
            if not os.path.exists(model_orig):
                print("Model file not found. Downloading...")
                model_url = "https://github.com/im-syn/SafeVision/raw/refs/heads/main/Models/best.onnx"
                success = download_model(model_url, model_orig)
                if not success:
                    print(f"Could not download model. Please download manually to {model_dir}")
                    self.onnx_session = None
                    return
            
            # Convert to opset 15 and load
            print("Converting model to opset 15...")
            model_to_load = _ensure_opset15(model_orig)
            print(f"Model to load: {model_to_load}")
            
            # Get available providers safely
            if onnxruntime is None:
                print("ONNX Runtime not available - detector disabled")
                self.onnx_session = None
                return
            
            print(f"ONNX Runtime available: {onnxruntime}")
            print("Creating ONNX session...")
            self.onnx_session = create_onnx_session(model_to_load, providers=providers)
            print("ONNX session created successfully!")

            # Get model input info
            inp = self.onnx_session.get_inputs()[0]
            self.input_name = inp.name
            self.input_width = inp.shape[2]
            self.input_height = inp.shape[3]
            
            print(f"Model input: {self.input_name}, shape: {inp.shape}")
            
            # Initialize exception rules to None (will be loaded separately)
            self.blur_exception_rules = None
            
            # Auto-load blur exception rules if they exist
            self.auto_load_blur_rules()
            
            print("Live Nude Detector initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Live Nude Detector: {e}")
            import traceback
            traceback.print_exc()
            self.onnx_session = None
            self.input_name = None
            self.input_width = 320
            self.input_height = 320
            self.blur_exception_rules = None

    def auto_load_blur_rules(self):
        """Automatically load blur exception rules if file exists in same directory."""
        rule_file_path = "BlurException.rule"
        
        if os.path.exists(rule_file_path) and os.path.getsize(rule_file_path) > 0:
            print(f"Found existing blur exception rules: {rule_file_path}")
            self.load_exception_rules(rule_file_path)
        else:
            print("No existing blur exception rules found. Creating default rules...")
            self.load_exception_rules(rule_file_path)  # This will create the default file

    def load_exception_rules(self, rule_file_path=None):
        """Load blur exception rules from file, creating default if needed."""
        if not rule_file_path:
            rule_file_path = "BlurException.rule"
            
        try:
            labels = globals()["__labels"]
            self.blur_exception_rules = load_blur_exception_rules(rule_file_path, labels=labels)
            print(f"Loaded {len(self.blur_exception_rules)} blur exception rules from {rule_file_path}")
        except Exception as e:
            print(f"Error loading blur exception rules: {e}")
            self.blur_exception_rules = {label: True for label in globals()["__labels"]}

    def should_apply_blur(self, label):
        """Check if blur should be applied to this label based on exception rules."""
        if self.blur_exception_rules is None:
            return True  # Default behavior if no rules loaded
        return self.blur_exception_rules.get(label, True)

    def detect_frame(self, frame):
        """Detect nudity in a single frame."""
        if self.onnx_session is None:
            # ONNX Runtime not available, return empty detections
            return []
            
        try:
            preprocessed_image, resize_factor, pad_left, pad_top = _read_frame_live(
                frame, self.input_width
            )
            outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
            detections = _postprocess_live(outputs, resize_factor, pad_left, pad_top)
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def apply_censoring(self, frame, detections):
        """Apply real-time censoring to detected regions with severity-based processing."""
        censored_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Define severity colors for visual feedback
        severity_colors = {
            'CRITICAL': (0, 0, 255),    # Red for critical content
            'HIGH': (0, 165, 255),      # Orange for high severity
            'MODERATE': (0, 255, 255),  # Yellow for moderate
            'LOW': (0, 255, 0),         # Green for low severity
            'SAFE': (255, 255, 255)     # White for safe content
        }
        
        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = detection["class"]
            score = detection["score"]
            severity = detection.get("severity", "MODERATE")
            
            # Check bounds
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            
            # Determine if censoring should be applied based on severity and rules
            should_censor = False
            
            # First check blur exception rules
            if not self.should_apply_blur(label):
                should_censor = False  # Skip if rules say not to blur this label
            else:
                # Apply severity-based logic if rules allow blurring
                if severity == 'CRITICAL':
                    should_censor = True
                elif severity == 'HIGH':
                    should_censor = True
                elif severity == 'MODERATE' and score >= 0.4:
                    should_censor = True
                elif severity == 'LOW' and score >= 0.6:
                    should_censor = True
            
            # Apply censoring if needed
            if should_censor:
                roi = censored_frame[y:y + h, x:x + w]
                
                if LIVE_CONFIG['SOLID_COLOR_MASK']:
                    # Just slam a solid color over it - simple and effective
                    censored_frame[y:y + h, x:x + w] = np.full(
                        (h, w, 3), LIVE_CONFIG['MASK_COLOR'], dtype=np.uint8
                    )
                else:
                    # Apply blur based on how bad the content is
                    if severity == 'CRITICAL':
                        blur_strength = LIVE_CONFIG['BLUR_STRENGTH_CRITICAL']
                    elif severity == 'HIGH':
                        blur_strength = int(LIVE_CONFIG['BLUR_STRENGTH'] * 1.5)  # Stronger blur
                    elif severity == 'MODERATE':
                        blur_strength = LIVE_CONFIG['BLUR_STRENGTH']
                    else:
                        blur_strength = int(LIVE_CONFIG['BLUR_STRENGTH'] * 0.7)  # Lighter blur
                    
                    # OpenCV needs odd numbers for blur kernels (because reasons)
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    
                    blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
                    censored_frame[y:y + h, x:x + w] = blurred_roi
            
            # Draw detection box and label with severity-based colors
            box_color = severity_colors.get(severity, (0, 255, 255))
            text_color = box_color
            
            cv2.rectangle(censored_frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Enhanced label with severity information
            label_text = f"{label} ({severity}) {score:.2f}"
            cv2.putText(censored_frame, label_text, (x, y - 10), font, 
                       LIVE_CONFIG['FONT_SCALE'], text_color, LIVE_CONFIG['FONT_THICKNESS'])
        
        return censored_frame
        
        return censored_frame

class GenderAgeDetector:
    def __init__(self, providers=None):
        """Initialize gender and age detection model."""
        print("Initializing Gender/Age Detector...")
        
        # Try to import ONNX Runtime if not already available
        if not _safe_import_onnx():
            print("Warning: ONNX Runtime not available - gender/age detection disabled")
            self.session = None
            return
        
        # Get script directory
        if getattr(sys, 'frozen', False):
            script_dir = os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model setup
        model_dir = os.path.join(script_dir, "Models")
        model_path = os.path.join(model_dir, "best_gender.onnx")
        
        if not os.path.exists(model_path):
            print(f"Gender/Age model not found at: {model_path}")
            print("Gender/Age detection will be disabled.")
            self.session = None
            return
        
        try:
            if onnxruntime is None:
                print("Warning: ONNX Runtime not available - gender/age detection disabled")
                self.session = None
                return
                
            self.session = create_onnx_session(model_path, providers=providers)
            
            # Get model input info
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.input_shape = inp.shape
            
            print(f"Gender/Age Detector initialized successfully!")
            print(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"Error loading gender/age model: {e}")
            self.session = None

    def softmax(self, x):
        """Apply softmax activation."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def preprocess_face(self, face_region):
        """Preprocess face region for gender/age detection."""
        if self.session is None:
            return None
            
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Resize to model input size (typically 224x224)
            target_size = (224, 224)
            pil_image = pil_image.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(pil_image).astype(np.float32)
            img_array = img_array / 255.0
            
            # Convert HWC to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None

    def predict_gender_age(self, face_region):
        """Predict gender and age from face region."""
        if self.session is None:
            return None, None, 0.0, 0.0
            
        try:
            # Preprocess the face
            input_tensor = self.preprocess_face(face_region)
            if input_tensor is None:
                return None, None, 0.0, 0.0
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # Process outputs (assuming gender is first output, age is second)
            if len(outputs) >= 2:
                gender_logits = outputs[0]
                age_prediction = outputs[1]
                
                # Process gender prediction
                gender_probs = self.softmax(gender_logits)
                gender_idx = int(np.argmax(gender_probs))
                gender_label = "Female" if gender_idx == 1 else "Male"
                gender_conf = float(gender_probs[0][gender_idx])
                
                # Process age prediction
                age_estimate = float(age_prediction[0][0])
                age_conf = 1.0  # Age is regression, so confidence is always high
                
                return gender_label, age_estimate, gender_conf, age_conf
            else:
                print("Unexpected model output format")
                return None, None, 0.0, 0.0
                
        except Exception as e:
            print(f"Error in gender/age prediction: {e}")
            return None, None, 0.0, 0.0

class LiveProcessor:
    def __init__(self, camera_id=0, show_boxes=True, privacy_mode=False, rules_file=None, enable_gender_detection=False):
        self.camera_id = camera_id
        self.show_boxes = show_boxes
        self.privacy_mode = privacy_mode
        self.enable_gender_detection = enable_gender_detection
        
        # Initialize detector
        self.detector = LiveNudeDetector()
        
        # Initialize gender/age detector if enabled
        self.gender_detector = None
        if enable_gender_detection:
            self.gender_detector = GenderAgeDetector()
        
        # Load blur exception rules (auto-load or from specified file)
        if rules_file:
            self.detector.load_exception_rules(rules_file)
        # Rules are auto-loaded during detector initialization
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera to a bigger resolution - because bigger is usually better
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get the actual resolution (sometimes cameras lie about what they support)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        
        # Processing state
        self.frame_count = 0
        self.last_detections = []
        self.exposed_streak = 0
        self.last_alert_time = 0
        self.recording = False
        self.video_writer = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0
        
        # Gender/Age detection results storage
        self.face_analysis_results = {}
        
        # Threading for async processing
        self.processing_queue = queue.Queue(maxsize=LIVE_CONFIG['BUFFER_SIZE'])
        self.result_queue = queue.Queue(maxsize=LIVE_CONFIG['BUFFER_SIZE'])
        self.processing_thread = None
        self.running = False

    def start_processing_thread(self):
        """Start background processing thread for better performance."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _processing_worker(self):
        """Background worker for frame processing - keeps things running smooth."""
        print("Background processing thread started")
        
        while self.running:
            try:
                # Get a frame to process (wait up to 1 second)
                frame = self.processing_queue.get(timeout=1.0)
                if frame is None:
                    print("Got shutdown signal in processing thread")
                    break
                
                # Do the actual detection work
                detections = self.detector.detect_frame(frame)
                
                # Put results back (non-blocking to avoid hanging)
                try:
                    self.result_queue.put((frame, detections), timeout=0.1)
                except queue.Full:
                    # Queue is full, just skip this result - no big deal
                    pass
                
            except queue.Empty:
                # No frames to process right now, just keep going
                continue
            except Exception as e:
                print(f"Error in background processing: {e}")
                # Don't crash the thread, just keep going
                continue
        
        print("Background processing thread stopped")

    def calculate_fps(self):
        """Calculate actual processing FPS."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.actual_fps = 30 / elapsed
            self.fps_start_time = current_time

    def analyze_faces_for_gender_age(self, frame, detections):
        """Analyze detected faces for gender and age."""
        if not self.enable_gender_detection or self.gender_detector is None or self.gender_detector.session is None:
            return {}
        
        face_results = {}
        
        for i, detection in enumerate(detections):
            label = detection["class"]
            
            # Only analyze face detections
            if label in ["FACE_FEMALE", "FACE_MALE"]:
                box = detection["box"]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                # Extract face region with some padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                # Extract face region
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size > 0:
                    # Predict gender and age
                    gender, age, gender_conf, age_conf = self.gender_detector.predict_gender_age(face_region)
                    
                    if gender is not None and age is not None:
                        face_results[f"face_{i}"] = {
                            "box": box,
                            "gender": gender,
                            "age": int(round(age)),
                            "gender_confidence": gender_conf,
                            "age_confidence": age_conf,
                            "detected_label": label
                        }
        
        return face_results

    def check_alert_conditions(self, detections):
        """Check if alert conditions are met based on severity."""
        # Count detections by severity
        critical_count = sum(1 for d in detections if d.get("severity") == "CRITICAL")
        high_count = sum(1 for d in detections if d.get("severity") == "HIGH")
        total_concerning = critical_count + high_count
        
        if total_concerning > 0:
            self.exposed_streak += 1
        else:
            self.exposed_streak = 0
        
        # Trigger alert based on severity and streak
        current_time = time.time()
        should_alert = False
        alert_msg = ""
        
        # Immediate alert for critical content
        if critical_count > 0:
            should_alert = True
            alert_msg = f"CRITICAL CONTENT DETECTED - {critical_count} explicit regions"
        elif (high_count > 0 and 
              self.exposed_streak >= max(1, LIVE_CONFIG['ALERT_THRESHOLD'] // 2)):
            should_alert = True
            alert_msg = f"HIGH SEVERITY CONTENT - {high_count} concerning regions"
        elif (total_concerning > 0 and 
              self.exposed_streak >= LIVE_CONFIG['ALERT_THRESHOLD']):
            should_alert = True
            alert_msg = f"SUSTAINED NUDITY DETECTION - {total_concerning} regions"
        
        # Apply cooldown
        if should_alert and current_time - self.last_alert_time > LIVE_CONFIG['ALERT_COOLDOWN']:
            self.last_alert_time = current_time
            return True, alert_msg
        
        return False, ""

    def start_recording(self, filename=None):
        """Start recording video."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"live_recording_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                          (self.frame_width, self.frame_height))
        self.recording = True
        print(f"Started recording: {filename}")

    def stop_recording(self):
        """Stop recording video."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Recording stopped")

    def draw_status_overlay(self, frame):
        """Draw status information overlay."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Status background
        overlay = frame.copy()
        overlay_height = 200 if self.enable_gender_detection else 150
        cv2.rectangle(overlay, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status_lines = [
            f"FPS: {self.actual_fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {len(self.last_detections)}",
            f"Exposed Streak: {self.exposed_streak}",
            f"Recording: {'ON' if self.recording else 'OFF'}"
        ]
        
        if self.enable_gender_detection:
            status_lines.append(f"Gender Detection: {'ON' if self.gender_detector and self.gender_detector.session else 'FAILED'}")
        
        for i, line in enumerate(status_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), font, 0.6, (0, 255, 0), 2)
        
        # Draw face analysis results
        if self.enable_gender_detection and hasattr(self, 'face_analysis_results'):
            y_offset = 35 + len(status_lines) * 25 + 10
            for face_id, result in self.face_analysis_results.items():
                face_info = f"{result['gender']}, {result['age']}y ({result['gender_confidence']:.2f})"
                cv2.putText(frame, face_info, (20, y_offset), font, 0.5, (255, 255, 0), 1)
                y_offset += 20
        
        # Instructions
        instructions = [
            "Controls:",
            "SPACE - Toggle Recording",
            "B - Toggle Boxes", 
            "P - Toggle Privacy Mode",
            "G - Toggle Gender Detection" if self.enable_gender_detection else "",
            "Q - Quit"
        ]
        
        # Filter out empty instructions
        instructions = [inst for inst in instructions if inst]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - len(instructions) * 20 + i * 20
            cv2.putText(frame, instruction, (20, y_pos), font, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main processing loop - this is where the magic happens."""
        print("Starting live processing...")
        print("Controls: SPACE=Record, B=Toggle Boxes, P=Privacy Mode, G=Gender Detection, Q=Quit")
        
        # Create a resizable window - much better than the tiny default
        cv2.namedWindow('Live Nudity Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Nudity Detection', 1280, 720)  # Start with a decent size
        
        # Start background processing thread
        self.start_processing_thread()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera - maybe it got unplugged?")
                    break
                
                self.frame_count += 1
                self.calculate_fps()
                
                # Only process every few frames to keep things smooth
                if self.frame_count % (LIVE_CONFIG['SKIP_FRAMES'] + 1) == 0:
                    # Add frame to processing queue (don't block if queue is full)
                    if not self.processing_queue.full():
                        self.processing_queue.put(frame.copy())
                
                # Try to get processed results (non-blocking)
                try:
                    processed_frame, detections = self.result_queue.get_nowait()
                    self.last_detections = detections
                except queue.Empty:
                    # No new results yet, use what we had before
                    detections = self.last_detections
                    processed_frame = frame
                
                # Handle the actual censoring/display logic
                try:
                    if not self.privacy_mode:
                        display_frame = self.detector.apply_censoring(frame, detections) if self.show_boxes else frame
                    else:
                        # Privacy mode - black screen with status only
                        display_frame = np.zeros_like(frame)
                        cv2.putText(display_frame, "PRIVACY MODE - Nothing to see here", 
                                  (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Do gender/age analysis if it's enabled and we have faces
                    if self.enable_gender_detection and detections:
                        self.face_analysis_results = self.analyze_faces_for_gender_age(frame, detections)
                    
                    # Check if we should freak out about what we detected
                    should_alert, alert_msg = self.check_alert_conditions(detections)
                    if should_alert:
                        print(f"ALERT: {alert_msg}")
                        # Flash a red border to get attention
                        cv2.rectangle(display_frame, (0, 0), 
                                    (display_frame.shape[1], display_frame.shape[0]), 
                                    (0, 0, 255), 10)
                    
                    # Auto-record if we're set up for that
                    if (LIVE_CONFIG['AUTO_RECORD_ON_DETECTION'] and 
                        len(detections) > 0 and not self.recording):
                        self.start_recording()
                    
                    # Draw all the status info on screen
                    self.draw_status_overlay(display_frame)
                    
                    # Record frame if we're recording
                    if self.recording and self.video_writer:
                        self.video_writer.write(display_frame)
                    
                    # Actually show the frame
                    cv2.imshow('Live Nudity Detection', display_frame)
                    
                except Exception as e:
                    print(f"Error in display processing: {e}")
                    # Show error frame instead of crashing
                    error_frame = np.zeros_like(frame)
                    cv2.putText(error_frame, f"Processing Error: {str(e)[:50]}", 
                              (20, frame.shape[0]//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Live Nudity Detection', error_frame)
                
                # Handle keyboard shortcuts
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space bar for recording
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('b'):  # Toggle detection boxes
                    self.show_boxes = not self.show_boxes
                    print(f"Detection boxes: {'ON' if self.show_boxes else 'OFF'}")
                elif key == ord('p'):  # Privacy mode toggle
                    self.privacy_mode = not self.privacy_mode
                    print(f"Privacy mode: {'ON' if self.privacy_mode else 'OFF'}")
                elif key == ord('g') and self.gender_detector:  # Gender detection toggle
                    self.enable_gender_detection = not self.enable_gender_detection
                    print(f"Gender detection: {'ON' if self.enable_gender_detection else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\nUser hit Ctrl+C - shutting down gracefully...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Shutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources properly - no more mysterious crashes."""
        print("Cleaning up...")
        self.running = False
        
        # Stop processing thread nicely
        if self.processing_thread:
            # Signal thread to stop
            try:
                self.processing_queue.put(None)  # This tells the thread to quit
                self.processing_thread.join(timeout=5)  # Wait for it to finish
                if self.processing_thread.is_alive():
                    print("Warning: Background thread didn't stop cleanly")
            except Exception as e:
                print(f"Error stopping processing thread: {e}")
        
        # Stop recording if it's running
        if self.recording:
            try:
                self.stop_recording()
            except Exception as e:
                print(f"Error stopping recording: {e}")
        
        # Release the camera properly
        if self.cap:
            try:
                self.cap.release()
                print("Camera released")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        
        # Close all windows
        try:
            cv2.destroyAllWindows()
            print("Windows closed")
        except Exception as e:
            print(f"Error closing windows: {e}")
        
        print("Cleanup complete - everything should be shut down now")
        
        # Give things a moment to settle
        time.sleep(0.5)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live Nudity Detection and Censoring")
    
    parser.add_argument("-c", "--camera", type=int, default=0,
                       help="Camera ID (default: 0)")
    parser.add_argument("-r", "--rules", type=str, default=None,
                       help="Path to blur exception rules file (optional - auto-loads BlurException.rule if exists)")
    parser.add_argument("-g", "--gender-detection", action="store_true",
                       help="Enable gender and age detection using best_gender.onnx model")
    parser.add_argument("--no-boxes", action="store_true",
                       help="Disable detection boxes")
    parser.add_argument("--privacy", action="store_true",
                       help="Start in privacy mode (no video display)")
    parser.add_argument("--enhanced-blur", action="store_true",
                       help="Use enhanced blur mode")
    parser.add_argument("--solid-color", action="store_true",
                       help="Use solid color masking instead of blur")
    parser.add_argument("--mask-color", type=str, default="0,0,0",
                       help="Color for solid masking (BGR format)")
    parser.add_argument("--auto-record", action="store_true",
                       help="Auto-record when nudity is detected")
    parser.add_argument("--alert-threshold", type=int, default=3,
                       help="Consecutive detections needed for alert")
    parser.add_argument("--skip-frames", type=int, default=2,
                       help="Process every nth frame (for performance)")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Show ONNX Runtime status for CLI usage
    success, status_msg = _get_onnx_status()
    if success:
        print("ONNX Runtime is available and working")
    else:
        print(f"ONNX Runtime status: {status_msg}")
    
    args = parse_args()
    
    # Apply configuration from arguments
    LIVE_CONFIG['ENHANCED_BLUR'] = args.enhanced_blur
    LIVE_CONFIG['SOLID_COLOR_MASK'] = args.solid_color
    LIVE_CONFIG['AUTO_RECORD_ON_DETECTION'] = args.auto_record
    LIVE_CONFIG['ALERT_THRESHOLD'] = args.alert_threshold
    LIVE_CONFIG['SKIP_FRAMES'] = args.skip_frames
    
    # Parse mask color
    if args.solid_color:
        try:
            color_parts = args.mask_color.split(',')
            if len(color_parts) == 3:
                LIVE_CONFIG['MASK_COLOR'] = (int(color_parts[0]), int(color_parts[1]), int(color_parts[2]))
        except ValueError:
            print("Invalid color format. Using default black.")
    
    try:
        print("Starting up live detection system...")
        print(f"Using camera {args.camera}")
        if args.rules:
            print(f"Using blur rules from: {args.rules}")
        if args.gender_detection:
            print("Gender/age detection enabled")
        
        # Create and run live processor in standalone mode
        processor = LiveProcessor(
            camera_id=args.camera,
            show_boxes=not args.no_boxes,
            privacy_mode=args.privacy,
            rules_file=args.rules,
            enable_gender_detection=args.gender_detection
        )
        
        print("System ready - starting main loop...")
        processor.run()
        
    except KeyboardInterrupt:
        print("\nUser interrupted - shutting down cleanly...")
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Check if your camera is working and models are available")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Main function complete")

if __name__ == "__main__":
    main()
