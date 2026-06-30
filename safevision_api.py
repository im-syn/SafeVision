#!/usr/bin/env python3
"""
SafeVision API Server
RESTful API endpoint for nudity detection using ONNX model.
Provides HTTP endpoints for image and video content analysis.
"""

import os
import sys
import json
import time
import uuid
import base64
import mimetypes
import threading
from datetime import datetime
from pathlib import Path
from io import BytesIO
import traceback
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Web framework imports
try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask not available. Install with: pip install flask flask-cors")
    FLASK_AVAILABLE = False

try:
    from werkzeug.utils import secure_filename
    WERKZEUG_AVAILABLE = True
except ImportError:
    print("Werkzeug not available. Install with: pip install werkzeug")
    WERKZEUG_AVAILABLE = False

# Image processing imports
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    print("OpenCV not available. Install with: pip install opencv-python")
    OPENCV_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("PIL not available. Install with: pip install pillow")
    PIL_AVAILABLE = False

# ONNX Runtime imports with error handling
try:
    import onnxruntime
    import onnx
    from onnx import version_converter
    ONNX_AVAILABLE = True
except ImportError as e:
    print(f"ONNX Runtime not available: {e}")
    print("Install with: pip install onnxruntime onnx")
    ONNX_AVAILABLE = False

# Import the detectors if available
try:
    from main import NudeDetector as ImageNudeDetector, __labels
    IMAGE_DETECTOR_AVAILABLE = True
except ImportError:
    print("Image detector not available")
    IMAGE_DETECTOR_AVAILABLE = False

try:
    from video import NudeDetector as VideoNudeDetector
    VIDEO_DETECTOR_AVAILABLE = True
except ImportError:
    print("Video detector not available")
    VIDEO_DETECTOR_AVAILABLE = False

DETECTOR_AVAILABLE = IMAGE_DETECTOR_AVAILABLE

# API Configuration
API_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'DEBUG': False,
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max upload
    'MAX_URL_DOWNLOAD_SIZE': 50 * 1024 * 1024,
    'URL_TIMEOUT': 20,
    'UPLOAD_FOLDER': 'api_uploads',
    'OUTPUT_FOLDER': 'api_outputs',
    'TEMP_FOLDER': 'api_temp',
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'},
    'IMAGE_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'},
    'VIDEO_EXTENSIONS': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'},
    'DEFAULT_VIDEO_MAX_FRAMES': 60,
    'DEFAULT_VIDEO_SAMPLE_SECONDS': 1.0,
    'DEFAULT_THRESHOLD': 0.25,
    'CLEANUP_INTERVAL': 3600,  # 1 hour
    'MAX_FILE_AGE': 86400,     # 24 hours
}

# Content labels with API response formatting
CONTENT_LABELS = [
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

# Risk levels for API responses
RISK_LEVELS = {
    'SAFE': ['FACE_FEMALE', 'FACE_MALE', 'FEMALE_GENITALIA_COVERED', 'BELLY_COVERED', 
             'FEET_COVERED', 'ARMPITS_COVERED', 'ANUS_COVERED', 'FEMALE_BREAST_COVERED', 
             'BUTTOCKS_COVERED'],
    'LOW': ['MALE_BREAST_EXPOSED', 'BELLY_EXPOSED', 'ARMPITS_EXPOSED', 'FEET_EXPOSED'],
    'MODERATE': ['BUTTOCKS_EXPOSED'],
    'HIGH': ['FEMALE_BREAST_EXPOSED', 'ANUS_EXPOSED'],
    'CRITICAL': ['FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED']
}

RISK_PRIORITY = ['SAFE', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL']

def get_risk_level(label):
    """Get risk level for a detected label."""
    for level, labels in RISK_LEVELS.items():
        if label in labels:
            return level
    return 'UNKNOWN'

def parse_bool_param(value, default=False):
    """Parse common truthy/falsy strings from query/form data."""
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on', 'y'}

def risk_is_higher(candidate, current):
    """Return True when candidate is a higher risk level than current."""
    if candidate not in RISK_PRIORITY:
        return False
    if current not in RISK_PRIORITY:
        return True
    return RISK_PRIORITY.index(candidate) > RISK_PRIORITY.index(current)

def format_detection(detection):
    """Format a raw model detection for JSON responses."""
    label = detection['class']
    return {
        'label': label,
        'confidence': round(float(detection['score']), 4),
        'risk_level': get_risk_level(label),
        'bounding_box': {
            'x': int(detection['box'][0]),
            'y': int(detection['box'][1]),
            'width': int(detection['box'][2]),
            'height': int(detection['box'][3])
        }
    }

def summarize_detections(detections, threshold):
    """Build consistent risk summary data for image and video responses."""
    risk_scores = {}
    highest_risk = 'SAFE'
    for detection in detections:
        risk = get_risk_level(detection['class'])
        risk_scores[risk] = risk_scores.get(risk, 0) + 1
        if risk_is_higher(risk, highest_risk):
            highest_risk = risk
    return {
        'total_detections': len(detections),
        'highest_risk_level': highest_risk,
        'risk_distribution': risk_scores,
        'is_safe': highest_risk in ['SAFE', 'LOW'],
        'threshold_used': threshold
    }

# Initialize Flask app
if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)
    app.config['MAX_CONTENT_LENGTH'] = API_CONFIG['MAX_CONTENT_LENGTH']
    
    # Create necessary directories
    for folder in [API_CONFIG['UPLOAD_FOLDER'], API_CONFIG['OUTPUT_FOLDER'], API_CONFIG['TEMP_FOLDER']]:
        os.makedirs(folder, exist_ok=True)

class SafeVisionAPI:
    """Main API class for SafeVision nudity detection."""
    
    def __init__(self):
        self.detector = None
        self.video_detector = None
        self.model_loaded = False
        self.video_model_loaded = False
        self.request_count = 0
        self.start_time = time.time()
        self.active_sessions = {}
        
        # Initialize detector if available
        if DETECTOR_AVAILABLE and ONNX_AVAILABLE:
            try:
                self.detector = ImageNudeDetector()
                self.model_loaded = True
                print("✅ SafeVision detector loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load detector: {e}")
                self.model_loaded = False
        else:
            print("❌ Dependencies not available for detector initialization")
    
    def allowed_file(self, filename):
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in API_CONFIG['ALLOWED_EXTENSIONS']

    def get_extension(self, filename):
        """Return a lowercase extension without the dot."""
        return Path(filename or "").suffix.lower().lstrip(".")

    def get_media_type(self, file_path, content_type=None):
        """Detect whether a saved file should be handled as image or video."""
        extension = self.get_extension(file_path)
        if extension in API_CONFIG['IMAGE_EXTENSIONS']:
            return 'image'
        if extension in API_CONFIG['VIDEO_EXTENSIONS']:
            return 'video'

        content_type = (content_type or mimetypes.guess_type(str(file_path))[0] or "").lower()
        if content_type.startswith("image/"):
            return 'image'
        if content_type.startswith("video/"):
            return 'video'

        if OPENCV_AVAILABLE:
            image = cv2.imread(str(file_path))
            if image is not None:
                return 'image'
            capture = cv2.VideoCapture(str(file_path))
            try:
                if capture.isOpened():
                    return 'video'
            finally:
                capture.release()
        return 'unknown'

    def extension_from_url(self, media_url, content_type=None):
        """Pick a safe local file extension from a URL or Content-Type header."""
        parsed = urlparse(media_url)
        path_extension = self.get_extension(unquote(parsed.path))
        if path_extension in API_CONFIG['ALLOWED_EXTENSIONS']:
            return path_extension

        guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
        guessed_extension = self.get_extension(guessed)
        if guessed_extension == "jpe":
            guessed_extension = "jpg"
        if guessed_extension in API_CONFIG['ALLOWED_EXTENSIONS']:
            return guessed_extension
        return "bin"

    def download_media_url(self, media_url):
        """Download an HTTP/HTTPS media URL to the API temp folder with size limits."""
        parsed = urlparse(media_url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {
                'status': 'error',
                'code': 400,
                'error': 'url must be an absolute http:// or https:// media URL'
            }

        request_headers = {
            'User-Agent': 'SafeVisionAPI/1.0',
            'Accept': 'image/*,video/*,*/*;q=0.5'
        }
        request_obj = Request(media_url, headers=request_headers)
        temp_path = None

        try:
            with urlopen(request_obj, timeout=API_CONFIG['URL_TIMEOUT']) as response:
                content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
                content_length = response.headers.get('Content-Length')
                try:
                    content_length = int(content_length) if content_length else None
                except ValueError:
                    content_length = None
                if content_length and content_length > API_CONFIG['MAX_URL_DOWNLOAD_SIZE']:
                    return {
                        'status': 'error',
                        'code': 413,
                        'error': 'Remote media is too large',
                        'max_size_mb': API_CONFIG['MAX_URL_DOWNLOAD_SIZE'] // (1024 * 1024)
                    }

                extension = self.extension_from_url(response.geturl(), content_type)
                filename = f"url_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.{extension}"
                temp_path = os.path.join(API_CONFIG['TEMP_FOLDER'], filename)
                os.makedirs(API_CONFIG['TEMP_FOLDER'], exist_ok=True)

                total_read = 0
                with open(temp_path, 'wb') as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total_read += len(chunk)
                        if total_read > API_CONFIG['MAX_URL_DOWNLOAD_SIZE']:
                            output_file.close()
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass
                            return {
                                'status': 'error',
                                'code': 413,
                                'error': 'Remote media is too large',
                                'max_size_mb': API_CONFIG['MAX_URL_DOWNLOAD_SIZE'] // (1024 * 1024)
                            }
                        output_file.write(chunk)

                return {
                    'status': 'success',
                    'path': temp_path,
                    'bytes': total_read,
                    'content_type': content_type,
                    'final_url': response.geturl()
                }
        except HTTPError as exc:
            return {'status': 'error', 'code': exc.code, 'error': f'Failed to download media URL: HTTP {exc.code}'}
        except URLError as exc:
            return {'status': 'error', 'code': 400, 'error': f'Failed to download media URL: {exc.reason}'}
        except Exception as exc:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return {'status': 'error', 'code': 500, 'error': f'Failed to download media URL: {exc}'}
    
    def generate_session_id(self):
        """Generate unique session ID."""
        return str(uuid.uuid4())
    
    def process_image(self, image_path, threshold=None, blur=False, session_id=None):
        """Process image for nudity detection."""
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'status': 'error',
                'code': 500
            }
        
        try:
            # Run detection
            detections = self.detector.detect(image_path)
            
            # Apply threshold filter
            if threshold is None:
                threshold = API_CONFIG['DEFAULT_THRESHOLD']
            
            filtered_detections = [
                d for d in detections if d['score'] >= threshold
            ]
            
            # Process censored version if requested
            censored_path = None
            if blur:
                try:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    censored_path = os.path.join(API_CONFIG['OUTPUT_FOLDER'], f"{base_name}_censored.jpg")
                    self.detector.censor(image_path, apply_blur=True, output_path=censored_path)
                except Exception as e:
                    print(f"Warning: Failed to create censored version: {e}")
            
            # Update request count
            self.request_count += 1
            
            # Prepare response
            response = {
                'status': 'success',
                'session_id': session_id or self.generate_session_id(),
                'timestamp': datetime.now().isoformat(),
                'media': {
                    'type': 'image',
                    'is_video': False
                },
                'analysis': summarize_detections(filtered_detections, threshold),
                'detections': [format_detection(d) for d in filtered_detections],
                'censored_available': censored_path is not None,
                'processing_info': {
                    'model_version': 'best.onnx',
                    'total_requests': self.request_count,
                    'uptime_seconds': int(time.time() - self.start_time)
                }
            }
            
            return response
            
        except Exception as e:
            return {
                'error': f'Processing failed: {str(e)}',
                'status': 'error',
                'code': 500,
                'traceback': traceback.format_exc() if API_CONFIG['DEBUG'] else None
            }

    def ensure_video_detector(self):
        """Load the video frame detector only when a video URL is actually processed."""
        if self.video_model_loaded and self.video_detector is not None:
            return True
        if not VIDEO_DETECTOR_AVAILABLE or not ONNX_AVAILABLE:
            return False
        try:
            self.video_detector = VideoNudeDetector()
            self.video_model_loaded = True
            print("✅ SafeVision video detector loaded successfully")
            return True
        except Exception as exc:
            print(f"❌ Failed to load video detector: {exc}")
            self.video_detector = None
            self.video_model_loaded = False
            return False

    def process_video(self, video_path, threshold=None, session_id=None, max_frames=None, sample_seconds=None, full_scan=False):
        """Process sampled video frames for nudity detection."""
        if not self.ensure_video_detector():
            return {
                'error': 'Video detection model not loaded',
                'status': 'error',
                'code': 500
            }
        if not OPENCV_AVAILABLE:
            return {
                'error': 'OpenCV is required for video URL detection',
                'status': 'error',
                'code': 500
            }

        if threshold is None:
            threshold = API_CONFIG['DEFAULT_THRESHOLD']
        max_frames = max(1, int(max_frames or API_CONFIG['DEFAULT_VIDEO_MAX_FRAMES']))
        sample_seconds = max(0.1, float(sample_seconds or API_CONFIG['DEFAULT_VIDEO_SAMPLE_SECONDS']))

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return {
                'error': 'Could not open downloaded video',
                'status': 'error',
                'code': 415
            }

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_seconds = total_frames / fps if fps > 0 and total_frames > 0 else None

            if full_scan:
                frame_step = 1
            elif fps > 0:
                frame_step = max(1, int(round(fps * sample_seconds)))
            elif total_frames > max_frames:
                frame_step = max(1, total_frames // max_frames)
            else:
                frame_step = 1

            frames_analyzed = 0
            frames_with_detections = 0
            all_detections = []
            frame_results = []

            def analyze_frame(frame, analyzed_frame_index):
                raw_detections = self.video_detector.detect_frame(frame)
                filtered_detections = [
                    detection for detection in raw_detections if detection['score'] >= threshold
                ]
                timestamp = analyzed_frame_index / fps if fps > 0 else None
                frame_results.append({
                    'frame_index': analyzed_frame_index,
                    'timestamp_seconds': round(timestamp, 3) if timestamp is not None else None,
                    'detections': [format_detection(detection) for detection in filtered_detections]
                })
                return filtered_detections

            if not full_scan and total_frames > 0:
                sample_indices = list(range(0, total_frames, frame_step))[:max_frames]
                for frame_index in sample_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    filtered_detections = analyze_frame(frame, frame_index)
                    if filtered_detections:
                        frames_with_detections += 1
                        all_detections.extend(filtered_detections)
                    frames_analyzed += 1
            else:
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break

                    if frame_index % frame_step == 0:
                        filtered_detections = analyze_frame(frame, frame_index)
                        if filtered_detections:
                            frames_with_detections += 1
                            all_detections.extend(filtered_detections)
                        frames_analyzed += 1
                        if not full_scan and frames_analyzed >= max_frames:
                            break

                    frame_index += 1

            self.request_count += 1
            analysis = summarize_detections(all_detections, threshold)
            analysis.update({
                'frames_analyzed': frames_analyzed,
                'frames_with_detections': frames_with_detections,
                'sampled': not full_scan,
                'sample_seconds': sample_seconds if not full_scan else None,
                'frame_step': frame_step
            })

            return {
                'status': 'success',
                'session_id': session_id or self.generate_session_id(),
                'timestamp': datetime.now().isoformat(),
                'media': {
                    'type': 'video',
                    'is_video': True,
                    'width': width,
                    'height': height,
                    'fps': round(fps, 3) if fps else None,
                    'total_frames': total_frames,
                    'duration_seconds': round(duration_seconds, 3) if duration_seconds is not None else None
                },
                'analysis': analysis,
                'frames': frame_results,
                'processing_info': {
                    'model_version': 'best.onnx',
                    'total_requests': self.request_count,
                    'uptime_seconds': int(time.time() - self.start_time)
                }
            }
        except Exception as exc:
            return {
                'error': f'Video processing failed: {exc}',
                'status': 'error',
                'code': 500,
                'traceback': traceback.format_exc() if API_CONFIG['DEBUG'] else None
            }
        finally:
            capture.release()
    
    def get_stats(self):
        """Get API statistics."""
        return {
            'status': 'online',
            'model_loaded': self.model_loaded,
            'total_requests': self.request_count,
            'uptime_seconds': int(time.time() - self.start_time),
            'active_sessions': len(self.active_sessions),
            'supported_formats': list(API_CONFIG['ALLOWED_EXTENSIONS']),
            'image_formats': list(API_CONFIG['IMAGE_EXTENSIONS']),
            'video_formats': list(API_CONFIG['VIDEO_EXTENSIONS']),
            'max_file_size_mb': API_CONFIG['MAX_CONTENT_LENGTH'] // (1024 * 1024),
            'max_url_download_mb': API_CONFIG['MAX_URL_DOWNLOAD_SIZE'] // (1024 * 1024),
            'version': '1.0.0'
        }
    
    def cleanup_old_files(self):
        """Clean up old temporary files."""
        try:
            current_time = time.time()
            for folder in [API_CONFIG['UPLOAD_FOLDER'], API_CONFIG['OUTPUT_FOLDER'], API_CONFIG['TEMP_FOLDER']]:
                for file_path in Path(folder).glob('*'):
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > API_CONFIG['MAX_FILE_AGE']:
                            file_path.unlink()
                            print(f"Cleaned up old file: {file_path}")
        except Exception as e:
            print(f"Cleanup error: {e}")

# Initialize API instance
api_instance = SafeVisionAPI()

# Flask routes
if FLASK_AVAILABLE:
    
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify(api_instance.get_stats())
    
    @app.route('/api/v1/detect', methods=['POST'])
    def detect_image():
        """Main detection endpoint for images."""
        try:
            # Check if model is loaded
            if not api_instance.model_loaded:
                return jsonify({
                    'error': 'Detection model not available',
                    'status': 'error'
                }), 503
            
            # Validate request
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No file provided',
                    'status': 'error'
                }), 400
            
            file = request.files['file']
            if file.filename == '' or not api_instance.allowed_file(file.filename):
                return jsonify({
                    'error': 'Invalid file type',
                    'status': 'error',
                    'allowed_types': list(API_CONFIG['ALLOWED_EXTENSIONS'])
                }), 400
            
            # Get parameters
            threshold = request.form.get('threshold', API_CONFIG['DEFAULT_THRESHOLD'], type=float)
            blur = request.form.get('blur', 'false').lower() == 'true'
            session_id = request.form.get('session_id')
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(API_CONFIG['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Process image
            result = api_instance.process_image(file_path, threshold, blur, session_id)
            
            if result.get('status') == 'error':
                return jsonify(result), result.get('code', 500)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'error': f'Unexpected error: {str(e)}',
                'status': 'error'
            }), 500
    
    @app.route('/api/v1/detect/base64', methods=['POST'])
    def detect_base64():
        """Detection endpoint for base64 encoded images."""
        try:
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({
                    'error': 'No base64 image data provided',
                    'status': 'error'
                }), 400
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(data['image'])
                
                # Save to temporary file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                temp_filename = f"temp_{timestamp}.jpg"
                temp_path = os.path.join(API_CONFIG['TEMP_FOLDER'], temp_filename)
                
                with open(temp_path, 'wb') as f:
                    f.write(image_data)
                
            except Exception as e:
                return jsonify({
                    'error': f'Invalid base64 image data: {str(e)}',
                    'status': 'error'
                }), 400
            
            # Get parameters
            threshold = data.get('threshold', API_CONFIG['DEFAULT_THRESHOLD'])
            blur = data.get('blur', False)
            session_id = data.get('session_id')
            
            # Process image
            result = api_instance.process_image(temp_path, threshold, blur, session_id)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if result.get('status') == 'error':
                return jsonify(result), result.get('code', 500)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'error': f'Unexpected error: {str(e)}',
                'status': 'error'
            }), 500

    @app.route('/api/v1/detect/url', methods=['GET'])
    @app.route('/api/v1/detect/media', methods=['GET'])
    def detect_media_url():
        """Detection endpoint for image/video media URLs."""
        downloaded_path = None
        keep_file = False
        try:
            if not api_instance.model_loaded:
                return jsonify({
                    'error': 'Detection model not available',
                    'status': 'error'
                }), 503

            media_url = request.args.get('url') or request.args.get('media_url')
            if not media_url:
                return jsonify({
                    'error': 'Missing required query parameter: url',
                    'status': 'error',
                    'example': '/api/v1/detect/url?url=https://example.com/media.jpg'
                }), 400

            threshold = request.args.get('threshold', API_CONFIG['DEFAULT_THRESHOLD'], type=float)
            blur = parse_bool_param(request.args.get('blur'), default=False)
            keep_file = parse_bool_param(request.args.get('keep'), default=False)
            full_scan = parse_bool_param(request.args.get('full_scan'), default=False)
            max_frames = request.args.get('max_frames', API_CONFIG['DEFAULT_VIDEO_MAX_FRAMES'], type=int)
            sample_seconds = request.args.get('sample_seconds', API_CONFIG['DEFAULT_VIDEO_SAMPLE_SECONDS'], type=float)
            session_id = request.args.get('session_id')

            download_result = api_instance.download_media_url(media_url)
            if download_result.get('status') == 'error':
                return jsonify(download_result), download_result.get('code', 500)

            downloaded_path = download_result['path']
            media_type = api_instance.get_media_type(downloaded_path, download_result.get('content_type'))

            if media_type == 'image':
                result = api_instance.process_image(downloaded_path, threshold, blur, session_id)
            elif media_type == 'video':
                result = api_instance.process_video(
                    downloaded_path,
                    threshold=threshold,
                    session_id=session_id,
                    max_frames=max_frames,
                    sample_seconds=sample_seconds,
                    full_scan=full_scan
                )
            else:
                return jsonify({
                    'error': 'Downloaded URL is not a supported image or video',
                    'status': 'error',
                    'content_type': download_result.get('content_type'),
                    'allowed_types': list(API_CONFIG['ALLOWED_EXTENSIONS'])
                }), 415

            if result.get('status') == 'error':
                return jsonify(result), result.get('code', 500)

            result['source'] = {
                'kind': 'url',
                'url': media_url,
                'final_url': download_result.get('final_url'),
                'content_type': download_result.get('content_type'),
                'bytes_downloaded': download_result.get('bytes'),
                'kept_file': keep_file,
                'local_path': downloaded_path if keep_file else None
            }
            result['media']['is_video'] = media_type == 'video'
            result['media']['type'] = media_type
            return jsonify(result)

        except Exception as exc:
            return jsonify({
                'error': f'Unexpected error: {exc}',
                'status': 'error',
                'traceback': traceback.format_exc() if API_CONFIG['DEBUG'] else None
            }), 500
        finally:
            if downloaded_path and not keep_file:
                try:
                    os.remove(downloaded_path)
                except OSError:
                    pass
    
    @app.route('/api/v1/labels', methods=['GET'])
    def get_labels():
        """Get available detection labels."""
        return jsonify({
            'labels': CONTENT_LABELS,
            'risk_levels': RISK_LEVELS,
            'total_labels': len(CONTENT_LABELS)
        })
    
    @app.route('/api/v1/stats', methods=['GET'])
    def get_statistics():
        """Get detailed API statistics."""
        return jsonify(api_instance.get_stats())
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({
            'error': 'File too large',
            'max_size_mb': API_CONFIG['MAX_CONTENT_LENGTH'] // (1024 * 1024),
            'status': 'error'
        }), 413

def start_cleanup_scheduler():
    """Start background cleanup scheduler."""
    def cleanup_worker():
        while True:
            time.sleep(API_CONFIG['CLEANUP_INTERVAL'])
            api_instance.cleanup_old_files()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

def main():
    """Main function to start the API server."""
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. Cannot start API server.")
        print("Install with: pip install flask flask-cors")
        return
    
    if not api_instance.model_loaded:
        print("⚠️  Warning: Detection model not loaded. API will have limited functionality.")
    
    print(f"🚀 Starting SafeVision API Server...")
    print(f"📡 Host: {API_CONFIG['HOST']}")
    print(f"🔌 Port: {API_CONFIG['PORT']}")
    print(f"🔍 Model loaded: {api_instance.model_loaded}")
    print(f"📁 Upload folder: {API_CONFIG['UPLOAD_FOLDER']}")
    print(f"📁 Output folder: {API_CONFIG['OUTPUT_FOLDER']}")
    
    # Start cleanup scheduler
    start_cleanup_scheduler()
    
    # Available endpoints
    print("\n📋 Available Endpoints:")
    print("  GET  /api/v1/health       - Health check")
    print("  POST /api/v1/detect       - Image detection (multipart/form-data)")
    print("  POST /api/v1/detect/base64 - Image detection (base64 JSON)")
    print("  GET  /api/v1/detect/url?url=... - Image/video detection from media URL")
    print("  GET  /api/v1/labels       - Available labels")
    print("  GET  /api/v1/stats        - API statistics")
    
    print(f"\n🌐 API Documentation: http://{API_CONFIG['HOST']}:{API_CONFIG['PORT']}/api/v1/health")
    print("="*60)
    
    try:
        app.run(
            host=API_CONFIG['HOST'],
            port=API_CONFIG['PORT'],
            debug=API_CONFIG['DEBUG'],
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 API Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == '__main__':
    main()
