#!/usr/bin/env python3
"""
SafeVision Live Streamer Edition
Real-time nudity detection and censoring for streamers with OBS integration.
Supports camera input, screen capture, and virtual camera output.
"""

import cv2
import numpy as np
import sys
import os
import time
import threading
import queue
import argparse
import traceback
import math
import urllib.request
from pathlib import Path
import json
import socket
import subprocess
import platform
from collections import defaultdict
from datetime import datetime
from safevision_utils import create_onnx_session

# Try to import OBS WebSocket for integration
try:
    import obsws_python as obs
    OBS_AVAILABLE = True
    print("OBS WebSocket support available")
except ImportError:
    OBS_AVAILABLE = False
    print("Warning: obsws_python not available. Install with: pip install obsws-python")

# Try to import screen capture libraries
try:
    import mss
    MSS_AVAILABLE = True
    print("Screen capture support available")
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not available. Install with: pip install mss")

# Try to import virtual camera support
try:
    import pyvirtualcam
    PYVIRTUALCAM_AVAILABLE = True
    print("Virtual camera support available")
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False
    print("Warning: pyvirtualcam not available. Install with: pip install pyvirtualcam")

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
        return _import_successful
    
    _import_attempted = True
    
    try:
        import onnxruntime as ort
        import onnx as onnx_module
        from onnx import version_converter as vc
        
        # Test basic functionality
        providers = ort.get_available_providers()
        if not providers:
            raise RuntimeError("No ONNX providers available")
        
        onnxruntime = ort
        onnx = onnx_module
        version_converter = vc
        _import_successful = True
        print(f"ONNX Runtime imported successfully, version: {ort.__version__}")
        return True
        
    except ImportError as e:
        print(f"ONNX Runtime not installed: {e}")
        _import_successful = False
        return False
    except Exception as e:
        print(f"Error during ONNX import: {type(e).__name__}: {e}")
        _import_successful = False
        return False

def _get_onnx_status():
    """Get current ONNX Runtime status."""
    if not _import_attempted:
        _safe_import_onnx()
    
    if _import_successful and onnxruntime is not None:
        try:
            providers = onnxruntime.get_available_providers()
            return True, f"Available providers: {providers}"
        except Exception as e:
            return False, f"Error getting providers: {e}"
    else:
        return False, "ONNX Runtime not available"

# Try initial import
_safe_import_onnx()

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("Warning: PIL not available, some features may not work")
    PIL_AVAILABLE = False
    Image = None

# Streamer-optimized configuration
STREAMER_CONFIG = {
    # Performance settings for streaming
    'TARGET_FPS': 60,                         # High FPS for smooth streaming
    'PROCESSING_FPS': 30,                     # AI processing FPS (can be lower)
    'RESOLUTION': (1920, 1080),              # Default streaming resolution
    'QUALITY': 'HIGH',                       # HIGH, MEDIUM, LOW
    
    # Detection thresholds - matching live.py for accuracy
    'DETECTION_THRESHOLD': 0.25,             # Base threshold from live.py
    'EXPOSED_THRESHOLD': 0.4,                # Higher bar for exposed content
    'GENITALIA_THRESHOLD': 0.6,              # High confidence for explicit content
    'FACE_THRESHOLD': 0.7,                   # Avoid false face positives
    
    # Blur settings - matching live.py
    'BLUR_STRENGTH': 25,                     # Default blur kernel from live.py
    'BLUR_STRENGTH_CRITICAL': 35,           # Max blur for critical content
    'SOLID_COLOR_MASK': False,              # Use blur by default
    'MASK_COLOR': (0, 0, 0),                # Black color for masking
    'ADAPTIVE_BLUR': True,                   # Adjust blur based on content
    'MOTION_BLUR_REDUCTION': True,           # Reduce blur artifacts during motion
    
    # OBS Integration
    'OBS_HOST': 'localhost',
    'OBS_PORT': 4455,
    'OBS_PASSWORD': '',                      # Set your OBS WebSocket password
    'AUTO_SCENE_SWITCH': False,              # Auto switch to safe scene when needed
    'SAFE_SCENE_NAME': 'BRB Screen',         # Scene to switch to when content detected
    'MAIN_SCENE_NAME': 'Main Stream',       # Normal streaming scene
    
    # Virtual Camera
    'VIRTUAL_CAM_ENABLED': True,
    'VIRTUAL_CAM_NAME': 'SafeVision Cam',
    'VIRTUAL_CAM_FPS': 30,
    
    # Input Sources
    'INPUT_MODE': 'CAMERA',                  # CAMERA, SCREEN, WINDOW, OBS_SOURCE
    'CAMERA_ID': 0,
    'SCREEN_CAPTURE_MONITOR': 1,            # Monitor number for screen capture
    'WINDOW_TITLE': '',                     # Specific window to capture
    'OBS_SOURCE_NAME': '',                  # OBS source to monitor
    
    # Alert System for Streamers
    'INSTANT_BLUR': True,                   # Immediate blur without delay
    'ALERT_WEBHOOK': '',                    # Discord/Slack webhook for alerts
    'AUTO_PAUSE_STREAM': False,             # Auto pause stream on detection
    'SAFE_MODE_TIMEOUT': 10,                # Seconds in safe mode after detection
    
    # UI for Streamers
    'OVERLAY_ENABLED': True,                # Show detection overlay
    'STREAM_SAFE_OVERLAY': True,           # Show "Stream Safe" indicator
    'PERFORMANCE_OVERLAY': True,           # Show FPS and performance info
    'DETECTION_COUNTER': True,             # Show detection count
    
    # Privacy and Safety
    'PRIVACY_MODE': False,                 # Hide sensitive UI elements
    'LOG_DETECTIONS': True,                # Log all detections for review
    'SAVE_DETECTION_FRAMES': False,        # Save frames with detections
    'DETECTION_LOG_PATH': 'detection_log.json',
    
    # Performance Optimization
    'GPU_ACCELERATION': True,
    'MULTI_THREADING': True,
    'FRAME_SKIP_ON_LOAD': True,           # Skip frames when system is loaded
    'DYNAMIC_QUALITY': True,              # Lower quality when performance drops
}

# Content labels with streaming safety levels
STREAMING_LABELS = [
    "FEMALE_GENITALIA_COVERED",    # Stream Safe
    "FACE_FEMALE",                 # Stream Safe  
    "BUTTOCKS_EXPOSED",            # Stream Unsafe - Blur
    "FEMALE_BREAST_EXPOSED",       # Stream Unsafe - Strong Blur
    "FEMALE_GENITALIA_EXPOSED",    # Stream Banned - Immediate Action
    "MALE_BREAST_EXPOSED",         # Stream Safe (usually)
    "ANUS_EXPOSED",                # Stream Banned - Immediate Action
    "FEET_EXPOSED",                # Stream Safe
    "BELLY_COVERED",               # Stream Safe
    "FEET_COVERED",                # Stream Safe
    "ARMPITS_COVERED",             # Stream Safe
    "ARMPITS_EXPOSED",             # Stream Safe
    "FACE_MALE",                   # Stream Safe
    "BELLY_EXPOSED",               # Stream Safe (context dependent)
    "MALE_GENITALIA_EXPOSED",      # Stream Banned - Immediate Action
    "ANUS_COVERED",                # Stream Safe
    "FEMALE_BREAST_COVERED",       # Stream Safe
    "BUTTOCKS_COVERED",            # Stream Safe
]

# Streaming safety classification - matching live.py severity system
STREAMING_SAFETY = {
    'CRITICAL': [
        'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED'
    ],
    'HIGH': [
        'FEMALE_BREAST_EXPOSED', 'ANUS_EXPOSED'
    ],
    'MODERATE': [
        'BUTTOCKS_EXPOSED'
    ],
    'LOW': [
        'MALE_BREAST_EXPOSED', 'BELLY_EXPOSED', 'ARMPITS_EXPOSED', 'FEET_EXPOSED'
    ],
    'SAFE': [
        'FACE_FEMALE', 'FACE_MALE', 'FEMALE_GENITALIA_COVERED',
        'BELLY_COVERED', 'FEET_COVERED', 'ARMPITS_COVERED', 'ANUS_COVERED',
        'FEMALE_BREAST_COVERED', 'BUTTOCKS_COVERED'
    ]
}

def get_streaming_safety(label):
    """Get severity level for detected content (matching live.py)."""
    for severity, labels in STREAMING_SAFETY.items():
        if label in labels:
            return severity
    return 'UNKNOWN'

def get_streaming_threshold(label):
    """Get confidence threshold matching live.py implementation."""
    severity = get_streaming_safety(label)
    
    if severity == 'CRITICAL':
        return STREAMER_CONFIG['GENITALIA_THRESHOLD']
    elif label in ['FACE_FEMALE', 'FACE_MALE']:
        return STREAMER_CONFIG['FACE_THRESHOLD']
    elif severity in ['HIGH', 'MODERATE']:
        return STREAMER_CONFIG['EXPOSED_THRESHOLD']
    else:
        return STREAMER_CONFIG['DETECTION_THRESHOLD']

class DetectionTracker:
    """Tracks detections across frames to maintain consistent blur."""
    
    def __init__(self, max_age=30, iou_threshold=0.3, confidence_decay=0.95):
        self.max_age = max_age  # Maximum frames to keep a track alive
        self.iou_threshold = iou_threshold  # IoU threshold for matching detections
        self.confidence_decay = confidence_decay  # How much confidence decays per frame
        
        self.tracks = []  # List of active tracks
        self.next_id = 0  # Next track ID
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def update(self, detections):
        """Update tracks with new detections."""
        # Age existing tracks
        for track in self.tracks:
            track['age'] += 1
            track['confidence'] *= self.confidence_decay
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track['age'] < self.max_age]
        
        # Match new detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for i, detection in enumerate(detections):
            best_iou = 0
            best_track_idx = -1
            
            for j, track in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                    
                iou = self.calculate_iou(detection['box'], track['box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_idx = j
            
            if best_track_idx >= 0:
                # Update existing track
                track = self.tracks[best_track_idx]
                track['box'] = detection['box']
                track['confidence'] = max(track['confidence'], detection['score'])
                track['severity'] = detection['severity']
                track['class'] = detection['class']
                track['age'] = 0  # Reset age
                track['last_seen'] = time.time()
                
                matched_tracks.add(best_track_idx)
                matched_detections.add(i)
            
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                new_track = {
                    'id': self.next_id,
                    'box': detection['box'],
                    'confidence': detection['score'],
                    'severity': detection['severity'],
                    'class': detection['class'],
                    'age': 0,
                    'created': time.time(),
                    'last_seen': time.time()
                }
                self.tracks.append(new_track)
                self.next_id += 1
        
        return self.get_active_tracks()
    
    def get_active_tracks(self):
        """Get all active tracks that should be blurred."""
        active_tracks = []
        
        for track in self.tracks:
            # Only include tracks with sufficient confidence or critical severity
            if (track['confidence'] > 0.1 or 
                track['severity'] in ['CRITICAL', 'HIGH'] or
                track['age'] < 5):  # Keep recent tracks visible
                
                active_tracks.append({
                    'id': track['id'],
                    'box': track['box'],
                    'confidence': track['confidence'],
                    'severity': track['severity'],
                    'class': track['class'],
                    'age': track['age']
                })
        
        return active_tracks
    
    def clear(self):
        """Clear all tracks."""
        self.tracks = []

class OBSIntegration:
    """OBS WebSocket integration for streamers."""
    
    def __init__(self, host='localhost', port=4455, password=''):
        self.host = host
        self.port = port
        self.password = password
        self.client = None
        self.connected = False
        
        if OBS_AVAILABLE:
            self.connect()
        else:
            print("OBS integration disabled - obsws_python not available")
    
    def connect(self):
        """Connect to OBS WebSocket."""
        try:
            self.client = obs.ReqClient(host=self.host, port=self.port, password=self.password)
            self.connected = True
            print(f"Connected to OBS at {self.host}:{self.port}")
            
            # Get OBS version info
            version_info = self.client.get_version()
            print(f"OBS Version: {version_info.obs_version}")
            
        except Exception as e:
            print(f"Failed to connect to OBS: {e}")
            self.connected = False
    
    def switch_scene(self, scene_name):
        """Switch to a specific scene."""
        if not self.connected:
            return False
        
        try:
            self.client.set_current_program_scene(scene_name)
            print(f"Switched to scene: {scene_name}")
            return True
        except Exception as e:
            print(f"Failed to switch scene: {e}")
            return False
    
    def get_current_scene(self):
        """Get current active scene."""
        if not self.connected:
            return None
        
        try:
            response = self.client.get_current_program_scene()
            return response.current_program_scene_name
        except Exception as e:
            print(f"Failed to get current scene: {e}")
            return None
    
    def start_stop_streaming(self, start=True):
        """Start or stop streaming."""
        if not self.connected:
            return False
        
        try:
            if start:
                self.client.start_stream()
                print("Started streaming")
            else:
                self.client.stop_stream()
                print("Stopped streaming")
            return True
        except Exception as e:
            print(f"Failed to {'start' if start else 'stop'} streaming: {e}")
            return False
    
    def get_streaming_status(self):
        """Get current streaming status."""
        if not self.connected:
            return False
        
        try:
            response = self.client.get_stream_status()
            return response.output_active
        except Exception as e:
            print(f"Failed to get streaming status: {e}")
            return False
    
    def send_chat_message(self, message):
        """Send a message to chat (if supported)."""
        # This would need additional OBS plugins
        print(f"Chat message: {message}")

class ScreenCapture:
    """Screen capture for streaming."""
    
    def __init__(self, monitor=1):
        self.monitor = monitor
        self.sct = None
        
        if MSS_AVAILABLE:
            self.sct = mss.mss()
            self.monitors = self.sct.monitors
            print(f"Available monitors: {len(self.monitors) - 1}")
            for i, monitor in enumerate(self.monitors[1:], 1):
                print(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
        else:
            print("Screen capture disabled - mss not available")
    
    def capture_screen(self, monitor_num=None):
        """Capture screen or specific monitor."""
        if not MSS_AVAILABLE or not self.sct:
            return None
        
        try:
            if monitor_num is None:
                monitor_num = self.monitor
            
            if monitor_num >= len(self.monitors):
                monitor_num = 1
            
            monitor = self.monitors[monitor_num]
            screenshot = self.sct.grab(monitor)
            
            # Convert to OpenCV format
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            return frame
            
        except Exception as e:
            print(f"Screen capture error: {e}")
            return None
    
    def capture_window(self, window_title):
        """Capture specific window by title."""
        # This would need platform-specific implementation
        print(f"Window capture not implemented for: {window_title}")
        return None

class VirtualCamera:
    """Virtual camera output for OBS."""
    
    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.enabled = False
        
        if PYVIRTUALCAM_AVAILABLE:
            try:
                self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
                self.enabled = True
                print(f"Virtual camera initialized: {width}x{height} @ {fps}FPS")
            except Exception as e:
                print(f"Failed to initialize virtual camera: {e}")
        else:
            print("Virtual camera disabled - pyvirtualcam not available")
    
    def send_frame(self, frame):
        """Send frame to virtual camera."""
        if not self.enabled or not self.cam:
            return False
        
        try:
            # Ensure frame is the right size
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Send to virtual camera
            self.cam.send(frame_rgb)
            return True
            
        except Exception as e:
            print(f"Virtual camera error: {e}")
            return False
    
    def close(self):
        """Close virtual camera."""
        if self.cam:
            self.cam.close()
            self.enabled = False

class StreamerDetector:
    """Optimized detector for streaming."""
    
    def __init__(self):
        print("Initializing Streamer Detector...")
        
        # Import ONNX Runtime
        if not _safe_import_onnx(force_retry=True):
            print("ONNX Runtime not available - detector disabled")
            self.onnx_session = None
            return
        
        # Get script directory
        if getattr(sys, 'frozen', False):
            script_dir = os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Model setup
        model_dir = os.path.join(script_dir, "Models")
        model_path = os.path.join(model_dir, "best.onnx")
        
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            print("Please ensure the model file exists")
            self.onnx_session = None
            return
        
        try:
            provider_override = None if STREAMER_CONFIG['GPU_ACCELERATION'] else ['CPUExecutionProvider']
            
            # Session options for performance
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            if STREAMER_CONFIG['MULTI_THREADING']:
                sess_options.intra_op_num_threads = 0  # Use all available threads
                sess_options.inter_op_num_threads = 0
            
            self.onnx_session = create_onnx_session(
                model_path,
                providers=provider_override,
                sess_options=sess_options,
            )
            
            # Get model info
            inp = self.onnx_session.get_inputs()[0]
            self.input_name = inp.name
            self.input_size = inp.shape[2]  # Assuming square input
            
            print(f"Streamer Detector initialized successfully!")
            print(f"Model input: {self.input_name}, size: {self.input_size}")
            
        except Exception as e:
            print(f"Error initializing detector: {e}")
            traceback.print_exc()
            self.onnx_session = None
    
    def preprocess_frame(self, frame, target_size=320):
        """Optimized preprocessing for streaming."""
        img_height, img_width = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate aspect ratio
        aspect = img_width / img_height

        if img_height > img_width:
            new_height = target_size
            new_width = int(round(target_size * aspect))
        else:
            new_width = target_size
            new_height = int(round(target_size / aspect))

        # Resize
        img = cv2.resize(img, (new_width, new_height))

        # Padding
        pad_x = target_size - new_width
        pad_y = target_size - new_height
        pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
        pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        img = cv2.resize(img, (target_size, target_size))
        
        # Normalize and format for model
        image_data = img.astype("float32") / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0)

        # Calculate resize factor for postprocessing
        resize_factor = math.sqrt(
            (img_width**2 + img_height**2) / (new_width**2 + new_height**2)
        )

        return image_data, resize_factor, pad_left, pad_top
    
    def postprocess_detections(self, outputs, resize_factor, pad_left, pad_top):
        """Process model outputs with live.py compatible filtering."""
        outputs = np.transpose(np.squeeze(outputs[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            class_id = np.argmax(classes_scores)
            
            if class_id < len(STREAMING_LABELS):
                label = STREAMING_LABELS[class_id]
                required_threshold = get_streaming_threshold(label)
                
                # Only accept detections that meet the severity-specific threshold
                if max_score >= required_threshold:
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                    
                    # Scale back to original coordinates (matching live.py format)
                    left = int(round((x - w * 0.5 - pad_left) * resize_factor))
                    top = int(round((y - h * 0.5 - pad_top) * resize_factor))
                    width = int(round(w * resize_factor))
                    height = int(round(h * resize_factor))
                    
                    # Additional validation for critical content
                    severity = get_streaming_safety(label)
                    if severity == 'CRITICAL':
                        # Extra strict validation for genitalia detection
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

        # Use stricter NMS parameters matching live.py for better filtering
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.4)

        detections = []
        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                label = STREAMING_LABELS[class_id]
                severity = get_streaming_safety(label)
                
                detections.append({
                    "class": label,
                    "score": float(score),
                    "box": box,
                    "severity": severity
                })

        return detections
    
    def detect_frame(self, frame):
        """Detect content in frame optimized for streaming."""
        if self.onnx_session is None:
            return []
        
        try:
            # Preprocess
            input_data, resize_factor, pad_left, pad_top = self.preprocess_frame(
                frame, self.input_size
            )
            
            # Run inference
            outputs = self.onnx_session.run(None, {self.input_name: input_data})
            
            # Postprocess
            detections = self.postprocess_detections(outputs, resize_factor, pad_left, pad_top)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

class StreamProcessor:
    """Main streaming processor with all integrations."""
    
    def __init__(self, config=None):
        # Load configuration
        self.config = config or STREAMER_CONFIG.copy()
        
        # Initialize components
        self.detector = StreamerDetector()
        self.obs = None
        self.screen_capture = None
        self.virtual_camera = None
        self.camera = None
        
        # Initialize detection tracker for stable blur
        self.tracker = DetectionTracker(
            max_age=45,          # Keep tracks for 45 frames (1.5 seconds at 30fps)
            iou_threshold=0.3,   # IoU threshold for matching
            confidence_decay=0.98 # Slow confidence decay for stability
        )
        
        # State tracking
        self.running = False
        self.input_mode = self.config['INPUT_MODE']
        self.frame_count = 0
        self.detection_count = 0
        self.last_detections = []
        self.stable_tracks = []  # Tracks from previous frame
        self.safe_mode = False
        self.safe_mode_start = 0
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.actual_fps = 0
        self.processing_fps = 0
        
        # Detection logging
        self.detection_log = []
        
        # Initialize integrations
        self.init_integrations()
    
    def init_integrations(self):
        """Initialize OBS, virtual camera, and input sources."""
        
        # OBS Integration
        if OBS_AVAILABLE:
            self.obs = OBSIntegration(
                host=self.config['OBS_HOST'],
                port=self.config['OBS_PORT'],
                password=self.config['OBS_PASSWORD']
            )
        
        # Virtual Camera
        if PYVIRTUALCAM_AVAILABLE and self.config['VIRTUAL_CAM_ENABLED']:
            width, height = self.config['RESOLUTION']
            self.virtual_camera = VirtualCamera(
                width=width, 
                height=height, 
                fps=self.config['VIRTUAL_CAM_FPS']
            )
        
        # Screen Capture
        if MSS_AVAILABLE:
            self.screen_capture = ScreenCapture(
                monitor=self.config['SCREEN_CAPTURE_MONITOR']
            )
        
        # Camera Input
        if self.input_mode == 'CAMERA':
            self.init_camera()
    
    def init_camera(self):
        """Initialize camera input."""
        try:
            self.camera = cv2.VideoCapture(self.config['CAMERA_ID'])
            if not self.camera.isOpened():
                raise RuntimeError(f"Cannot open camera {self.config['CAMERA_ID']}")
            
            # Set camera properties
            width, height = self.config['RESOLUTION']
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config['TARGET_FPS'])
            
            # Get actual properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}FPS")
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.camera = None
    
    def get_frame(self):
        """Get frame from current input source."""
        frame = None
        
        if self.input_mode == 'CAMERA' and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                frame = None
                
        elif self.input_mode == 'SCREEN' and self.screen_capture:
            frame = self.screen_capture.capture_screen()
            
        elif self.input_mode == 'WINDOW' and self.screen_capture:
            frame = self.screen_capture.capture_window(self.config['WINDOW_TITLE'])
        
        # Resize frame to target resolution if needed
        if frame is not None:
            target_width, target_height = self.config['RESOLUTION']
            if frame.shape[:2] != (target_height, target_width):
                frame = cv2.resize(frame, (target_width, target_height))
        
        return frame
    
    def process_frame(self, frame):
        """Process frame with AI detection and apply safety measures with tracking."""
        if frame is None:
            return frame
        
        processed_frame = frame.copy()
        
        # Skip AI processing on some frames for performance
        should_process_ai = (self.frame_count % max(1, self.config['TARGET_FPS'] // self.config['PROCESSING_FPS']) == 0)
        
        raw_detections = []
        if should_process_ai and self.detector.onnx_session is not None:
            raw_detections = self.detector.detect_frame(frame)
        
        # Update tracker with new detections
        if raw_detections or should_process_ai:
            # Update tracker even with empty detections to age existing tracks
            self.stable_tracks = self.tracker.update(raw_detections)
        
        # Use stable tracks for consistent blur
        if self.stable_tracks:
            self.last_detections = self.stable_tracks
            self.detection_count += len([t for t in self.stable_tracks if t['age'] == 0])  # Only count new detections
            
            # Log detections
            if self.config['LOG_DETECTIONS'] and raw_detections:
                self.log_detections(raw_detections)
            
            # Apply safety measures using stable tracks
            processed_frame = self.apply_safety_measures(processed_frame, self.stable_tracks)
            
            # Handle streaming actions
            self.handle_streaming_actions(self.stable_tracks)
        
        # Apply overlays
        if self.config['OVERLAY_ENABLED']:
            processed_frame = self.apply_overlays(processed_frame, self.stable_tracks)
        
        # Check safe mode timeout
        if self.safe_mode and time.time() - self.safe_mode_start > self.config['SAFE_MODE_TIMEOUT']:
            self.exit_safe_mode()
        
        return processed_frame
    
    def apply_safety_measures(self, frame, tracks):
        """Apply real-time censoring to tracked regions with stable blur."""
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
        
        for track in tracks:
            box = track["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = track["class"]
            score = track["confidence"]  # Use tracked confidence
            severity = track.get("severity", "MODERATE")
            track_id = track["id"]
            age = track["age"]
            
            # Check bounds
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            
            # Determine if censoring should be applied based on severity
            should_censor = False
            
            # Apply severity-based logic with age consideration
            if severity == 'CRITICAL':
                should_censor = True
            elif severity == 'HIGH':
                should_censor = True
            elif severity == 'MODERATE' and score >= 0.4:
                should_censor = True
            elif severity == 'LOW' and score >= 0.6:
                should_censor = True
            
            # For older tracks, keep censoring but potentially reduce strength
            if age > 0 and should_censor:
                # Keep censoring for tracked objects to prevent flickering
                should_censor = True
            
            # Apply censoring if needed
            if should_censor:
                roi = censored_frame[y:y + h, x:x + w]
                
                if self.config.get('SOLID_COLOR_MASK', False):
                    # Solid color mask
                    censored_frame[y:y + h, x:x + w] = np.full(
                        (h, w, 3), self.config['MASK_COLOR'], dtype=np.uint8
                    )
                else:
                    # Apply blur based on severity with age consideration
                    if severity == 'CRITICAL':
                        blur_strength = self.config['BLUR_STRENGTH_CRITICAL']
                    elif severity == 'HIGH':
                        blur_strength = int(self.config['BLUR_STRENGTH'] * 1.5)  # Stronger blur
                    elif severity == 'MODERATE':
                        blur_strength = self.config['BLUR_STRENGTH']
                    else:
                        blur_strength = int(self.config['BLUR_STRENGTH'] * 0.7)  # Lighter blur
                    
                    # For older tracks, slightly reduce blur but keep consistency
                    if age > 10:
                        fade_factor = min(0.3, age / 100.0)  # Max 30% reduction over time
                        blur_strength = int(blur_strength * (1.0 - fade_factor))
                    
                    # OpenCV needs odd numbers for blur kernels
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    
                    blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
                    censored_frame[y:y + h, x:x + w] = blurred_roi
            
            # Draw detection box and label with severity-based colors
            if self.config.get('SHOW_DETECTION_BOXES', True):
                box_color = severity_colors.get(severity, (0, 255, 255))
                text_color = box_color
                
                # Adjust box thickness based on age (newer tracks get thicker boxes)
                thickness = 3 if age == 0 else 2 if age < 10 else 1
                cv2.rectangle(censored_frame, (x, y), (x + w, y + h), box_color, thickness)
                
                # Enhanced label with track information
                label_text = f"ID:{track_id} {label} ({severity}) {score:.2f}"
                if age > 0:
                    label_text += f" t:{age}"
                
                cv2.putText(censored_frame, label_text, (x, y - 10), font, 
                           self.config.get('FONT_SCALE', 0.6), text_color, 
                           self.config.get('FONT_THICKNESS', 2))
        
        return censored_frame
        
        return censored_frame
    
    def handle_streaming_actions(self, tracks):
        """Handle streaming-specific actions based on tracked detections."""
        # Check for critical and high severity content
        critical_tracks = [t for t in tracks if t['severity'] == 'CRITICAL']
        high_tracks = [t for t in tracks if t['severity'] == 'HIGH']
        
        if critical_tracks:
            self.enter_safe_mode('CRITICAL_CONTENT')
            
            # Auto scene switch if enabled
            if self.obs and self.config.get('AUTO_SCENE_SWITCH', False):
                self.obs.switch_scene(self.config.get('SAFE_SCENE_NAME', 'BRB Screen'))
            
            # Send alert if webhook configured
            if self.config.get('ALERT_WEBHOOK'):
                self.send_alert_webhook('CRITICAL: Explicit content detected!')
                
        elif high_tracks and not self.safe_mode:
            self.enter_safe_mode('HIGH_SEVERITY_CONTENT')
    
    def enter_safe_mode(self, reason):
        """Enter safe mode."""
        if not self.safe_mode:
            self.safe_mode = True
            self.safe_mode_start = time.time()
            print(f"Entering safe mode: {reason}")
    
    def exit_safe_mode(self):
        """Exit safe mode."""
        if self.safe_mode:
            self.safe_mode = False
            print("Exiting safe mode")
            
            # Return to main scene if auto-switching is enabled
            if self.obs and self.config['AUTO_SCENE_SWITCH']:
                self.obs.switch_scene(self.config['MAIN_SCENE_NAME'])
    
    def apply_overlays(self, frame, tracks):
        """Apply UI overlays for streamers."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Stream safe indicator
        if self.config['STREAM_SAFE_OVERLAY']:
            if self.safe_mode:
                status_text = "SAFE MODE"
                status_color = (0, 255, 255)  # Yellow
            elif any(track.get('severity') in ['CRITICAL', 'HIGH'] for track in tracks):
                status_text = "CONTENT DETECTED"
                status_color = (0, 0, 255)  # Red
            else:
                status_text = "STREAM SAFE"
                status_color = (0, 255, 0)  # Green
            
            cv2.putText(frame, status_text, (10, 30), font, 1, status_color, 2)
        
        # Performance overlay
        if self.config['PERFORMANCE_OVERLAY']:
            perf_text = f"FPS: {self.actual_fps:.1f} | AI: {self.processing_fps:.1f}"
            cv2.putText(frame, perf_text, (10, frame.shape[0] - 60), font, 0.6, (255, 255, 255), 2)
        
        # Detection counter
        if self.config['DETECTION_COUNTER']:
            det_text = f"Detections: {self.detection_count}"
            cv2.putText(frame, det_text, (10, frame.shape[0] - 30), font, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def log_detections(self, detections):
        """Log detections for review."""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'frame': self.frame_count,
            'detections': detections
        }
        
        self.detection_log.append(log_entry)
        
        # Save to file periodically
        if len(self.detection_log) % 100 == 0:
            self.save_detection_log()
    
    def save_detection_log(self):
        """Save detection log to file."""
        try:
            with open(self.config['DETECTION_LOG_PATH'], 'w') as f:
                json.dump(self.detection_log, f, indent=2)
        except Exception as e:
            print(f"Failed to save detection log: {e}")
    
    def send_alert_webhook(self, message):
        """Send alert via webhook."""
        # Implementation would depend on webhook service (Discord, Slack, etc.)
        print(f"ALERT: {message}")
    
    def calculate_fps(self):
        """Calculate actual FPS."""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.actual_fps = 30 / (current_time - self.fps_start)
            self.fps_start = current_time
    
    def run(self):
        """Main streaming loop."""
        print("Starting SafeVision Streamer...")
        print(f"Input mode: {self.input_mode}")
        print(f"Target resolution: {self.config['RESOLUTION']}")
        print(f"AI Detection: {'Enabled' if self.detector.onnx_session else 'Disabled'}")
        print(f"OBS Integration: {'Enabled' if self.obs and self.obs.connected else 'Disabled'}")
        print(f"Virtual Camera: {'Enabled' if self.virtual_camera and self.virtual_camera.enabled else 'Disabled'}")
        
        self.running = True
        
        # Create display window
        if not self.config['PRIVACY_MODE']:
            cv2.namedWindow('SafeVision Streamer', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SafeVision Streamer', 1280, 720)
        
        try:
            while self.running:
                start_time = time.time()
                
                # Get frame from input source
                frame = self.get_frame()
                
                if frame is None:
                    print("No frame received, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Send to virtual camera
                if self.virtual_camera and self.virtual_camera.enabled:
                    self.virtual_camera.send_frame(processed_frame)
                
                # Display frame (if not in privacy mode)
                if not self.config['PRIVACY_MODE']:
                    display_frame = cv2.resize(processed_frame, (1280, 720))
                    cv2.imshow('SafeVision Streamer', display_frame)
                
                # Handle keyboard input
                if not self.config['PRIVACY_MODE']:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Toggle safe mode manually
                        if self.safe_mode:
                            self.exit_safe_mode()
                        else:
                            self.enter_safe_mode('MANUAL')
                    elif key == ord('c'):
                        # Switch input mode
                        self.cycle_input_mode()
                    elif key == ord('r'):
                        # Reset detection counter
                        self.detection_count = 0
                
                # Update counters
                self.frame_count += 1
                self.calculate_fps()
                
                # Frame rate limiting
                target_frame_time = 1.0 / self.config['TARGET_FPS']
                elapsed = time.time() - start_time
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                
        except KeyboardInterrupt:
            print("\nStopping streamer...")
        except Exception as e:
            print(f"Error in streaming loop: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cycle_input_mode(self):
        """Cycle through available input modes."""
        modes = ['CAMERA', 'SCREEN']
        current_index = modes.index(self.input_mode) if self.input_mode in modes else 0
        next_index = (current_index + 1) % len(modes)
        
        self.input_mode = modes[next_index]
        print(f"Switched to input mode: {self.input_mode}")
        
        # Reinitialize input source
        if self.input_mode == 'CAMERA':
            self.init_camera()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        
        self.running = False
        
        # Save final detection log
        if self.detection_log:
            self.save_detection_log()
        
        # Close camera
        if self.camera:
            self.camera.release()
        
        # Close virtual camera
        if self.virtual_camera:
            self.virtual_camera.close()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("Cleanup complete")

def parse_streamer_args():
    """Parse command line arguments for streamer version."""
    parser = argparse.ArgumentParser(description="SafeVision Live Streamer Edition")
    
    # Input options
    parser.add_argument("-i", "--input", choices=['camera', 'screen', 'window'], 
                       default='camera', help="Input source")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("-m", "--monitor", type=int, default=1, help="Monitor for screen capture")
    parser.add_argument("-w", "--window", type=str, help="Window title to capture")
    
    # Resolution and performance
    parser.add_argument("--resolution", type=str, default="1920x1080", 
                       help="Target resolution (WIDTHxHEIGHT)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--ai-fps", type=int, default=30, help="AI processing FPS")
    
    # OBS integration
    parser.add_argument("--obs-host", type=str, default="localhost", help="OBS WebSocket host")
    parser.add_argument("--obs-port", type=int, default=4455, help="OBS WebSocket port")
    parser.add_argument("--obs-password", type=str, default="", help="OBS WebSocket password")
    parser.add_argument("--auto-scene-switch", action="store_true", 
                       help="Auto switch OBS scenes on detection")
    
    # Virtual camera
    parser.add_argument("--virtual-cam", action="store_true", help="Enable virtual camera output")
    parser.add_argument("--vcam-fps", type=int, default=30, help="Virtual camera FPS")
    
    # Detection settings
    parser.add_argument("--sensitivity", type=float, default=0.2, 
                       help="Detection sensitivity (0.1-0.9)")
    parser.add_argument("--blur-strength", type=int, default=30, help="Blur strength")
    
    # Privacy and safety
    parser.add_argument("--privacy", action="store_true", help="Privacy mode (no display)")
    parser.add_argument("--safe-timeout", type=int, default=10, 
                       help="Safe mode timeout in seconds")
    
    # Performance
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--quality", choices=['low', 'medium', 'high'], default='high',
                       help="Processing quality")
    
    return parser.parse_args()

def main():
    """Main function for streamer edition."""
    print("SafeVision Live Streamer Edition")
    print("================================")
    
    # Check ONNX status
    success, status_msg = _get_onnx_status()
    print(f"AI Detection: {'Available' if success else 'Unavailable'}")
    if not success:
        print(f"Warning: {status_msg}")
    
    # Parse arguments
    args = parse_streamer_args()
    
    # Create configuration
    config = STREAMER_CONFIG.copy()
    
    # Apply command line arguments
    config['INPUT_MODE'] = args.input.upper()
    config['CAMERA_ID'] = args.camera
    config['SCREEN_CAPTURE_MONITOR'] = args.monitor
    config['WINDOW_TITLE'] = args.window or ''
    config['TARGET_FPS'] = args.fps
    config['PROCESSING_FPS'] = args.ai_fps
    config['DETECTION_THRESHOLD'] = args.sensitivity
    config['BLUR_STRENGTH'] = args.blur_strength
    config['PRIVACY_MODE'] = args.privacy
    config['SAFE_MODE_TIMEOUT'] = args.safe_timeout
    config['GPU_ACCELERATION'] = args.gpu
    config['VIRTUAL_CAM_ENABLED'] = args.virtual_cam
    config['VIRTUAL_CAM_FPS'] = args.vcam_fps
    config['OBS_HOST'] = args.obs_host
    config['OBS_PORT'] = args.obs_port
    config['OBS_PASSWORD'] = args.obs_password
    config['AUTO_SCENE_SWITCH'] = args.auto_scene_switch
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        config['RESOLUTION'] = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}, using default")
    
    # Quality settings
    if args.quality == 'low':
        config['BLUR_STRENGTH'] = 20
        config['TARGET_FPS'] = 30
    elif args.quality == 'medium':
        config['BLUR_STRENGTH'] = 25
        config['TARGET_FPS'] = 45
    
    try:
        # Create and run processor
        processor = StreamProcessor(config)
        processor.run()
        
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print("Streamer session ended")

if __name__ == "__main__":
    main()
