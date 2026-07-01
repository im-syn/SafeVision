
# ![SafeVision Logo](https://i.ibb.co/d4LqhX4/Safe-Vision-2.png)

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-yellow)
![License](https://img.shields.io/github/license/im-syn/safevision)
![Status](https://img.shields.io/badge/stability-stable-brightgreen)
![API](https://img.shields.io/badge/API-REST%20Endpoint-green)
![GUI](https://img.shields.io/badge/GUI-Multiple%20Interfaces-purple)

---

## Overview

**SafeVision** is a comprehensive, multi-modal content safety suite powered by advanced ONNX deep learning models. This sophisticated system provides real-time nudity detection, content filtering, and automated censoring across images, videos, live streams, and camera feeds. With multiple interfaces including CLI tools, modern GUI applications, live monitoring, streaming integration, and RESTful API endpoints, SafeVision offers enterprise-grade content moderation capabilities for various use cases.

### 🎯 Key Capabilities
- **Multi-Format Support**: Images, videos, live camera feeds, screen capture
- **Real-time Processing**: Live detection with sub-second response times
- **Multiple Interfaces**: CLI, GUI, API server, screen overlay, streaming integration
- **Advanced AI Models**: ONNX-optimized nudity/body detection plus optional safety-object detection
- **Professional Features**: Batch processing, custom rules, alert systems, logging
- **Cross-Platform**: Windows, Linux, macOS support with optimized performance

---

## 📑 Table of Contents

### 🚀 Getting Started
* [🔧 Installation & Setup](#-installation--setup)
* [📁 Project Structure](#-project-structure)
* [⚙️ Configuration](#️-configuration)
* [🔧 Command-Line Arguments Reference](#-command-line-arguments-reference)

### 🛠️ Core Applications
* [🖼️ Image Processing (main.py)](#️-image-processing-mainpy)
* [🎥 Video Processing (video.py)](#-video-processing-videopy)
* [📺 Live Camera Detection (live.py)](#-live-camera-detection-livepy)
* [🎮 Live Streaming (live_streamer.py)](#-live-streaming-live_streamerpy)

### 🎨 User Interfaces
* [🖼️ Modern GUI (safevision_gui.py)](#️-modern-gui-safevision_guipy)
* [🌐 REST API Server (safevision_api.py)](#-rest-api-server-safevision_apipy)

### ⚙️ Advanced Features
* [📋 Blur Exception Rules](#-blur-exception-rules)
* [🚬 Safety Object Model](#-safety-object-model-smoking-alcohol-drugs)
* [🔍 Detection Models & Labels](#-detection-models--labels)
* [📊 Logging & Monitoring](#-logging--monitoring)

### 📖 Documentation
* [🏗️ How It Works](#️-how-it-works)
* [📂 Output Directory Structure](#-output-directory-structure)
* [🔧 Troubleshooting](#-troubleshooting)
* [📷 Examples & Demos](#-examples--demos)

---

## ⭐ Features

### 🤖 AI-Powered Detection
- **Advanced ONNX Models**: Optimized deep learning models for accurate content detection
- **Dual Model Detection**: Run nudity/body detection, safety-object detection, or both in one pass
- **33 Censor Labels**: 18 nudity/body labels plus 15 smoking/alcohol/drug object labels
- **Risk Assessment**: Automatic severity classification (Safe, Low, Moderate, High, Critical)
- **Real-time Processing**: Sub-second analysis with GPU acceleration support
- **Confidence Scoring**: Adjustable detection thresholds for different use cases

### 🎯 Content Processing
- **Multi-Format Support**: Images (JPG, PNG, BMP, TIFF), Videos (MP4, AVI, MOV, MKV)
- **Live Camera Feeds**: Real-time webcam and USB camera processing
- **Screen Capture**: Monitor desktop activity with overlay detection
- **Batch Processing**: Process multiple files simultaneously
- **Audio Preservation**: Maintain original audio in processed videos

### 🛡️ Censoring & Safety
- **Intelligent Blurring**: Selective or full-frame blur with adjustable strength
- **Color Masking**: Solid color overlay as alternative to blur
- **Custom Exception Rules**: Define what content to blur or ignore
- **Protection Modes**: Kids Safety, Streamer, Nudity Fighter, and custom modes
- **Alert Systems**: Real-time notifications and emergency actions

### 🖥️ User Interfaces
- **Modern PyQt5 GUI**: Professional interface with drag-drop, themes, and live preview
- **Command Line Tools**: Full CLI support for automation and scripting
- **Screen Overlay**: Transparent overlay for monitoring any application
- **REST API Server**: HTTP endpoints for integration with other applications
- **Streaming Integration**: OBS and live streaming platform support

### 📊 Professional Features
- **Detailed Logging**: Comprehensive detection logs with timestamps and metadata
- **Performance Monitoring**: FPS tracking, memory usage, and system optimization
- **Configuration Management**: Persistent settings and customizable parameters
- **Report Generation**: Export detection reports in multiple formats
- **Multi-threading**: Optimized performance with parallel processing

### 🔧 Development & Integration
- **RESTful API**: Complete API server with JSON responses and file upload support
- **SDK Components**: Reusable classes for custom integration
- **Event Hooks**: Callback system for custom actions on detection
- **Plugin Architecture**: Extensible design for custom detection rules
- **Cross-platform**: Windows, Linux, macOS support with native installers

---

## 🔧 Installation & Setup

### 📋 System Requirements

**Minimum Requirements:**
- Python 3.8+ (3.9+ recommended)
- 4GB RAM (8GB+ recommended)
- 2GB free disk space
- CPU with AVX2 support (Intel 2013+, AMD 2017+)

**Recommended for Optimal Performance:**
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with CUDA support
- SSD storage for faster model loading

### 🚀 Quick Installation

```bash
# Clone the repository
git clone https://github.com/im-syn/safevision.git
cd safevision

# Install core dependencies
pip install -r requirements.txt

# For GUI applications (PyQt5 interface)
pip install -r requirements_gui.txt

# For API server functionality
pip install -r requirements_api.txt

# For live streaming features
pip install -r requirements_streaming.txt
```

### 📦 Required Models

SafeVision requires ONNX model files in the `Models/` directory:

```bash
# Create Models directory
mkdir Models

# Place your models (obtain from official source):
# Models/best.onnx                    - Main nudity/body detection model
# Models/best_gender.onnx             - Gender/age detection model (optional)
# Models/safety_objects.onnx          - Optional smoking/alcohol/drug object model
# Models/safety_objects.labels.json   - Labels/categories for safety_objects.onnx
```

> **📥 Model Download**: Contact the maintainer or check releases for official model files.

The safety-object model is optional. If it is present, image and video processing can use `--detectors objects` or `--detectors both`. If it is missing, keep the default `--detectors nude` mode.

`safety_objects.labels.json` must stay in sync with the ONNX class order. The current labels are:

```text
cigarette, cigar, vape, smoking_pipe, joint,
alcohol_bottle, beer_bottle, wine_glass, beer_glass, cocktail_glass,
pill, pill_bottle, syringe, cannabis_leaf, drug_bag
```

### Safety Object Model Training

The current `Models/safety_objects.onnx` model was trained as a second YOLO object-detection model. It does not replace the nudity/body model; it adds visible smoking, alcohol, and drug-related object detection.

Training workflow used:

```powershell
# From the training toolkit folder
cd C:\path\to\SafeText\traine

# Check/install training dependencies
python train_safety_objects.py check

# Import a Roboflow/YOLOv8 dataset zip
python train_safety_objects.py import-yolo datasets\cigarette.v3i.yolov8.zip

# Validate the merged YOLO dataset
python train_safety_objects.py validate

# Train. Use auto so it falls back to CPU when CUDA is unavailable.
python train_safety_objects.py train --base-model yolov8s.pt --imgsz 640 --epochs 100 --batch 16 --device auto

# Export the best checkpoint to SafeVision ONNX files
python train_safety_objects.py export --imgsz 640 --opset 15
```

The export writes:

```text
SafeVision/Models/safety_objects.onnx
SafeVision/Models/safety_objects.labels.json
```

The first dataset imported for this model was a YOLOv8 cigarette dataset from Roboflow. The training script is designed to merge more YOLO datasets later, so alcohol and drug classes can be improved by importing additional datasets with matching or aliased class names. Keep `safety_labels.json` stable once a model is trained because ONNX output class indexes depend on label order.

### 🔧 Advanced Installation Options

#### GPU Acceleration (Recommended)
```bash
# For NVIDIA GPU support
pip install onnxruntime-gpu

# For Intel GPU support  
pip install onnxruntime-openvino

# For AMD GPU support
pip install onnxruntime-directml
```

SafeVision now chooses ONNX Runtime providers conservatively. CUDA is preferred when available, CPU is kept as a fallback, and TensorRT is disabled by default because it requires extra CUDA/TensorRT shared libraries. To request a provider explicitly:

```bash
python video.py -i input.mp4 --providers CUDAExecutionProvider,CPUExecutionProvider
python main.py -i input.jpg --providers CUDAExecutionProvider,CPUExecutionProvider
```

To opt into TensorRT, install the matching CUDA/TensorRT runtime libraries in the environment and set `SAFEVISION_ENABLE_TENSORRT=1`.

#### FFmpeg Installation (Required for Video Processing)
```bash
# Windows (using chocolatey)
choco install ffmpeg

# macOS (using homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Manual installation
python download_ffmpeg.py
```

#### Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv safevision_env

# Activate (Windows)
safevision_env\Scripts\activate

# Activate (Linux/macOS)
source safevision_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### ✅ Installation Verification

```bash
# Test core functionality
python main.py --help

# Test GUI interface
python safevision_gui.py

# Test API server
python safevision_api.py

# Test live detection
python live.py --help

# Test video processing
python video.py --help

# Test the all-in-one console CLI. Run without args for the interactive menu.
python safeVisionCLI.py --help
python safeVisionCLI.py

# Test live streaming
python live_streamer.py --help
```

### SafeVision Console CLI

`safeVisionCLI.py` is the recommended control console for non-developers and power users. Run it with no arguments to open a looped row-based menu. It creates `settings/configs.json`, keeps rule profiles, scans media folders, launches GUI/API tools, and routes images/videos to the right processing script.

```bash
# Open the interactive looped menu
python safeVisionCLI.py

# First-time setup: create settings/configs.json, folders, and BlurException.rule
python safeVisionCLI.py init

# Check scripts, folders, active rule profile, and ONNX providers
python safeVisionCLI.py status

# Scan the input folder for videos and images
python safeVisionCLI.py scan input --recursive

# Process a video with the active rule profile
python safeVisionCLI.py process input/1.mp4 --providers CUDAExecutionProvider,CPUExecutionProvider

# Use the new safety-object model only
python safeVisionCLI.py process input/1.mp4 --detectors objects --boxes --blur

# Run both nudity/body and safety-object models together
python safeVisionCLI.py process input/1.mp4 --detectors both --boxes --blur --object-threshold 0.25

# Persist the default detector mode in settings/configs.json
python safeVisionCLI.py settings set processing.detectors both
python safeVisionCLI.py settings set processing.object_threshold 0.25

# Launch the desktop GUI or API/web server when those files exist
python safeVisionCLI.py launch gui
python safeVisionCLI.py launch web

# Launch Screen Guard from saved settings/configs.json
python safeVisionCLI.py screen

# Override saved screen settings for one run
python safeVisionCLI.py screen -- --mode blur --show-boxes --show-labels --label-filter body
```

Rule profiles are stored in `settings/configs.json`; activating a profile writes the working `BlurException.rule` file used by the processing scripts:

```bash
python safeVisionCLI.py rules list
python safeVisionCLI.py rules use faces_allowed
python safeVisionCLI.py rules set default FACE_MALE false
python safeVisionCLI.py rules export default my_profile.rule
```

### 🚨 Common Installation Issues

#### ONNX Runtime Issues
```bash
# If you get ONNX import errors:
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime

# For GPU support:
pip install onnxruntime-gpu
```

If a GPU provider cannot load inside Docker or another isolated environment, SafeVision logs the provider error and falls back to CPU instead of crashing. Check `nvidia-smi`, CUDA library availability, and the selected provider list printed at startup.

#### OpenCV Issues
```bash
# If OpenCV fails to import:
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.1.78
```

#### PyQt5 Issues
```bash
# If GUI fails to start:
pip uninstall PyQt5
pip install PyQt5==5.15.9
```

---

## 📁 Project Structure

```
SafeVision/
├── 🎯 Core Applications
│   ├── main.py                    # Image processing CLI
│   ├── video.py                   # Video processing CLI
│   ├── live.py                    # Live camera detection
│   ├── live_streamer.py           # Live streaming integration
│   ├── safeVisionScreenGuard.py   # Desktop screen protection overlay
│   ├── safeVisionCLI.py           # Interactive console and settings manager
│   ├── object_detector.py         # Optional safety-object ONNX detector
│   └── safevision_utils.py        # Shared rules, providers, and IO helpers
│
├── 🎨 User Interfaces
│   ├── safevision_gui.py          # Modern PyQt5 GUI
│   └── safevision_api.py          # REST API server
│
├── 📁 Models & Configuration
│   ├── Models/
│   │   ├── best.onnx                       # Main nudity/body detection model
│   │   ├── best_gender.onnx                # Gender/age model (optional)
│   │   ├── safety_objects.onnx             # Smoking/alcohol/drug object detector
│   │   └── safety_objects.labels.json      # Safety-object labels and thresholds
│   ├── BlurException.rule         # Default blur rules, auto-created if missing
│   ├── settings/configs.json      # Console settings and rule profiles
│   └── custom_rules.rule          # Custom rule examples
│
├── 📁 Output Directories (Auto-created)
│   ├── output/                    # Final processed content
│   ├── Blur/                     # Blurred versions
│   ├── Prosses/                  # Detection visualizations
│   ├── video_output/             # Processed videos
│   ├── Logs/                     # Application logs
│   └── detection_screenshots/    # Detection captures
│
└── 📁 Configuration Files (Auto-generated)
    ├── safevision_settings.json  # GUI settings
    ├── detection_log.json        # Detection history
    └── violation_log.json        # Violation tracking
```

### 📄 Key Files Explained

#### 🎯 Core Processing Files
- **`main.py`**: Single image processing with CLI interface
- **`video.py`**: Batch video processing with advanced options
- **`object_detector.py`**: Reusable YOLO ONNX detector for smoking/alcohol/drug objects
- **`live.py`**: Real-time camera feed analysis
- **`live_streamer.py`**: Live streaming integration with OBS support

#### 🎨 User Interface Files
- **`safevision_gui.py`**: Modern PyQt5 interface with advanced features
- **`safevision_api.py`**: RESTful API server for integration

#### ⚙️ Configuration Files
- **`BlurException.rule`**: Rules defining what content to blur
- **`*.json`**: Runtime settings and detection logs
- **`safevision_gui.py`**: Modern PyQt5 interface with advanced features
- **`safevision_api_server.py`**: RESTful API server for integration

#### ⚙️ Configuration Files
- **`config.py`**: Global settings and model configuration
- **`BlurException.rule`**: Rules defining what content to blur
- **`*.json`**: Runtime settings and detection logs

---

## ⚙️ Configuration

### 🔧 Global Configuration (config.py)

The main configuration file controls detection behavior, performance settings, and output options:

```python
# Detection Settings
DETECTION_THRESHOLD = 0.25      # Minimum confidence for detection
ENHANCED_BLUR = False           # Enable stronger blur effects
FULL_BLUR_STRENGTH = (99, 99, 75)  # Full-frame blur parameters

# Performance Settings
GPU_ACCELERATION = True         # Enable GPU processing
MULTI_THREADING = True         # Use multiple CPU cores
TARGET_FPS = 30                # Target processing FPS

# Output Settings
OUTPUT_VIDEO_SUFFIX = '_processed.mp4'
AUTO_CLEANUP = False           # Auto-delete temporary files
SAVE_DETECTION_LOGS = True     # Enable detection logging
```

### 📋 Blur Exception Rules

Create custom rules in `BlurException.rule` or `custom_rules.rule`:

```ini
# Content-specific rules (true = blur, false = ignore)
FEMALE_GENITALIA_EXPOSED = true
MALE_GENITALIA_EXPOSED = true
FEMALE_BREAST_EXPOSED = true
BUTTOCKS_EXPOSED = true
ANUS_EXPOSED = true
MALE_BREAST_EXPOSED = false
BELLY_EXPOSED = false
FEET_EXPOSED = false
ARMPITS_EXPOSED = false
FACE_FEMALE = false
FACE_MALE = false

# Covered content (usually safe)
FEMALE_GENITALIA_COVERED = false
FEMALE_BREAST_COVERED = false
BUTTOCKS_COVERED = false
ANUS_COVERED = false
BELLY_COVERED = false
FEET_COVERED = false
ARMPITS_COVERED = false

# Safety object model labels
cigarette = true
vape = true
alcohol_bottle = true
wine_glass = true
pill = true
syringe = true
cannabis_leaf = true
```

The console rule profiles now include both model families. `true` means a matching detection is blurred/masked; `false` means the detection can still be reported or boxed but the blur/mask is skipped.

Detector settings are stored in `settings/configs.json`:

```json
{
  "processing": {
    "detectors": "nude",
    "object_model": "Models/safety_objects.onnx",
    "object_labels": "Models/safety_objects.labels.json",
    "object_threshold": 0.25
  }
}
```

Valid detector modes are `nude`, `objects`, and `both`.

---

## 🚬 Safety Object Model (Smoking, Alcohol, Drugs)

SafeVision now supports a second optional ONNX model: `Models/safety_objects.onnx`.

This model is a YOLO-style object detector trained for visible safety-related objects:

| Category | Labels |
|----------|--------|
| Smoking | `cigarette`, `cigar`, `vape`, `smoking_pipe`, `joint` |
| Alcohol | `alcohol_bottle`, `beer_bottle`, `wine_glass`, `beer_glass`, `cocktail_glass` |
| Drugs | `pill`, `pill_bottle`, `syringe`, `cannabis_leaf`, `drug_bag` |

The model metadata lives in `Models/safety_objects.labels.json`. It defines label order, category mapping, aliases, and default category thresholds. Do not edit label order unless you also retrain/export the ONNX model.

### Detector Modes

The image, video, CLI, and GUI paths can choose which model family to run:

| Mode | Meaning |
|------|---------|
| `nude` | Run only the original nudity/body model |
| `objects` | Run only `safety_objects.onnx` |
| `both` | Run both models and merge detections |

Examples:

```bash
# Image: object model only
python main.py -i input.jpg -b --detectors objects --object-threshold 0.25

# Image: both models
python main.py -i input.jpg -b --detectors both

# Video: both models with boxes and blur
python video.py -i input.mp4 -b --blur --detectors both --object-threshold 0.25

# CLI: persist both models as the default
python safeVisionCLI.py settings set processing.detectors both
```

When a safety-object label is detected, SafeVision treats it as censorable. It can be blurred, solid-color masked, boxed, logged, exported to JSON/CSV reports, and exported to EDL/FCPXML marker files.

### Training Summary

The current model was trained as a separate YOLOv8 object detector:

1. A YOLOv8-format cigarette dataset was imported from Roboflow.
2. The training tool normalized dataset structure into `datasets/safety_objects`.
3. Labels were mapped to the canonical `safety_labels.json` order.
4. Training used Ultralytics YOLO with `imgsz 640`.
5. The best checkpoint was exported to ONNX opset 15.
6. The exported files were copied into `SafeVision/Models/`.

Recommended command flow:

```powershell
cd C:\path\to\SafeText\traine
python train_safety_objects.py check
python train_safety_objects.py import-yolo C:\path\to\dataset.yolov8.zip
python train_safety_objects.py validate
python train_safety_objects.py train --base-model yolov8s.pt --imgsz 640 --epochs 100 --batch 16 --device auto
python train_safety_objects.py export --imgsz 640 --opset 15
```

If PyTorch reports `torch.cuda.is_available(): False`, use `--device auto` or `--device cpu`. Use `--device 0 --strict-device` only when a real CUDA GPU is visible to PyTorch.

For better coverage, import more datasets for alcohol/drug classes before retraining. The first model is strongest on cigarette-like objects because that was the first dataset added.

---

## 🔍 Detection Models & Labels

| File | Purpose | Labels |
|------|---------|--------|
| `Models/best.onnx` | Main nudity/body detector | 18 body/coverage labels |
| `Models/best_gender.onnx` | Optional gender/age helper for live tools | Gender/age outputs |
| `Models/safety_objects.onnx` | Optional smoking/alcohol/drug object detector | 15 object labels |
| `Models/safety_objects.labels.json` | Metadata for the safety-object detector | Label order, categories, aliases, thresholds |

Detection outputs now include source metadata when reports are enabled:

```json
{
  "class": "cigarette",
  "category": "smoking",
  "source": "objects",
  "model": "safety_objects",
  "censor": true
}
```

This lets JSON/CSV reports and editor marker exports identify which model produced each detection.

### 🎛️ Application-Specific Settings

#### GUI Settings (`safevision_settings.json`)
```json
{
  "window_geometry": {
    "width": 1200,
    "height": 800,
    "x": 100,
    "y": 100
  },
  "theme": "dark",
  "last_directory": "",
  "auto_preview": true,
  "default_codec": "mp4v"
}
```

#### API Server Configuration
```python
API_CONFIG = {
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB
    'DEFAULT_THRESHOLD': 0.25,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
}
```

---

## 🖼️ Image Processing (main.py)

**Purpose**: Process single images with nudity/body and optional safety-object detection, then apply censoring/blurring effects.

**Key Features**:
- Single image analysis with selectable ONNX model inference
- Optional smoking/alcohol/drug object detection with `safety_objects.onnx`
- Customizable blur strength and masking options
- Bounding box visualization with confidence scores
- Multiple output formats (original, blurred, detection overlay)
- Custom exception rules for selective censoring

---

## �️ Image Processing (main.py)

**Purpose**: Process single images with selected SafeVision detector models and apply censoring/blurring effects.

**Key Features**:
- Single image analysis with ONNX model inference
- Customizable blur strength and masking options
- Bounding box visualization with confidence scores
- Multiple output formats (original, blurred, detection overlay)
- Custom exception rules for selective censoring

### 🚀 Basic Usage

```bash
# Simple detection with default settings
python main.py -i path/to/image.jpg

# Detection with custom output path
python main.py -i input.jpg -o custom_output.jpg

# Apply blur to detected regions
python main.py -i input.jpg -b

# Run only the safety-object model
python main.py -i input.jpg -b --detectors objects

# Run both the nudity/body model and safety-object model
python main.py -i input.jpg -b --detectors both --object-threshold 0.25

# Use custom exception rules
python main.py -i input.jpg -b -e custom_rules.rule

# Trigger full blur when 2+ exposed regions detected
python main.py -i input.jpg -b -fbr 2
```

### 📋 Command Line Arguments

| Argument | Long Form | Type | Description | Default |
|----------|-----------|------|-------------|---------|
| `-i` | `--input` | `str` | **Required.** Input image path | None |
| `-o` | `--output` | `str` | Output image path | Auto-generated |
| `-b` | `--blur` | `flag` | Apply blur to detected regions | False |
| `-e` | `--exception` | `str` | Path to blur exception rules file | `BlurException.rule` |
| `-fbr` | `--full_blur_rule` | `int` | Exposed regions count to trigger full blur | 0 (disabled) |
| N/A | `--detectors` | `str` | Detector mode: `nude`, `objects`, or `both` | `nude` |
| N/A | `--object-model` | `str` | Path to safety-object ONNX model | `Models/safety_objects.onnx` |
| N/A | `--object-labels` | `str` | Path to safety-object labels JSON | `Models/safety_objects.labels.json` |
| N/A | `--object-threshold` | `float` | Minimum confidence for safety-object detections | `0.25` |

### 📁 Output Structure

When processing `example.jpg`, the following files are created:

```
output/
├── example.jpg              # Final processed image (boxes + selective blur)
Blur/
├── example.jpg              # Fully blurred version
Prosses/
├── example.jpg              # Detection boxes only (no blur)
Logs/
├── detection_YYYYMMDD.log   # Processing log with timestamps
```

### 🔍 Detection Categories

The model detects 18 different content categories with confidence scores:

**Safe Content:**
- `FACE_FEMALE`, `FACE_MALE` - Facial detection
- `*_COVERED` variants - Clothed body parts

**Risk Categories:**
- **Low Risk**: `MALE_BREAST_EXPOSED`, `BELLY_EXPOSED`, `FEET_EXPOSED`, `ARMPITS_EXPOSED`
- **Moderate Risk**: `BUTTOCKS_EXPOSED`  
- **High Risk**: `FEMALE_BREAST_EXPOSED`, `ANUS_EXPOSED`
- **Critical Risk**: `FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`

**Safety Object Categories (optional `--detectors objects|both`):**
- **Smoking**: `cigarette`, `cigar`, `vape`, `smoking_pipe`, `joint`
- **Alcohol**: `alcohol_bottle`, `beer_bottle`, `wine_glass`, `beer_glass`, `cocktail_glass`
- **Drugs**: `pill`, `pill_bottle`, `syringe`, `cannabis_leaf`, `drug_bag`

---

## 🎥 Video Processing (video.py)

**Purpose**: Process video files with frame-by-frame nudity/body and optional safety-object detection, then apply censoring effects.

**Key Features**:
- Frame-by-frame analysis with selectable ONNX model inference
- Optional smoking/alcohol/drug object detection with `safety_objects.onnx`
- Audio preservation during processing
- Multiple output formats (original, blurred, with detection boxes)
- Batch processing with progress tracking
- Custom codec support and FFmpeg integration
- Advanced blur rules and full-frame triggers

### 🚀 Basic Usage

```bash
# Simple video detection
python video.py -i path/to/video.mp4 -t video

# Blur detected areas with audio preservation
python video.py -i input.mp4 -b --blur -a

# Safety-object model only
python video.py -i input.mp4 -b --blur --detectors objects

# Both models with object threshold control
python video.py -i input.mp4 -b --blur --detectors both --object-threshold 0.25

# Custom output location
python video.py -i input.mp4 -o output.mp4 -t video

# Enhanced blur with custom rules
python video.py -i input.mp4 -b --blur --enhanced-blur -fbr 2/10

# Solid color masking with ellipse regions instead of rectangular blur
python video.py -i input.mp4 -b --color --mask-color 255,0,0 --mask-shape ellipse
```

### 📋 Command Line Arguments

| Argument | Long Form | Type | Description | Default |
|----------|-----------|------|-------------|---------|
| `-i` | `--input` | `str` | **Required.** Input video path | None |
| `-o` | `--output` | `str` | Output video path | Auto-generated |
| `-t` | `--task` | `str` | Task type: `video` or `frames` | `video` |
| `-vo` | `--video_output` | `str` | Output folder | `video_output` |
| `-r` | `--rule` | `str` | Rule format: `percentage/count` | `50.0/5` |
| `-b` | `--boxes` | `flag` | Draw detection boxes | False |
| `--blur` | N/A | `flag` | Blur detected areas (requires `-b`) | False |
| `-a` | `--with-audio` | `flag` | Include original audio | False |
| `-c` | `--codec` | `str` | Video codec (`mp4v`, `xvid`, etc.) | `mp4v` |
| `--ffmpeg-path` | N/A | `str` | Custom FFmpeg path | Auto-detect |
| `-df` | `--delete-frames` | `flag` | Auto-delete temporary frames | False |
| `--enhanced-blur` | N/A | `flag` | Stronger censorship blur | False |
| `--color` | N/A | `flag` | Use solid color masking | False |
| `--mask-color` | N/A | `str` | Color for masking (BGR: `0,0,255`) | `0,0,0` |
| `--mask-shape` | N/A | `str` | Regional mask shape: `rectangle` or `ellipse` | `rectangle` |
| `-fbr` | `--full-blur-rule` | `str` | Full blur trigger: `labels/frames` | `0` |
| N/A | `--providers` | `str` | Comma-separated ONNX Runtime providers | Auto-select |
| N/A | `--detectors` | `str` | Detector mode: `nude`, `objects`, or `both` | `nude` |
| N/A | `--object-model` | `str` | Path to safety-object ONNX model | `Models/safety_objects.onnx` |
| N/A | `--object-labels` | `str` | Path to safety-object labels JSON | `Models/safety_objects.labels.json` |
| N/A | `--object-threshold` | `float` | Minimum confidence for safety-object detections | `0.25` |

### 🎛️ Processing Modes

#### Frame Processing Mode (`-t frames`)
```bash
# Extract and process individual frames
python video.py -i video.mp4 -t frames
```
- Extracts video frames for individual analysis
- Useful for detailed frame inspection
- Outputs processed frames to `output_frames/`

#### Video Processing Mode (`-t video`)
```bash
# Process entire video with censoring
python video.py -i video.mp4 -t video -b --blur -a
```
- Processes entire video with applied censoring
- Generates final output video file
- Preserves audio when `-a` flag is used

### 🔧 Advanced Features

#### Blur Rules (`-r percentage/count`)
```bash
# Trigger full blur when 30% of frames or 10 frames have detections
python video.py -i video.mp4 -r 30.0/10 -b --blur
```

#### Full Blur Rules (`-fbr labels/frames`)
```bash
# Full blur when 2+ exposed labels detected for 5+ consecutive frames
python video.py -i video.mp4 -fbr 2/5 -b --blur
```

#### Custom Codecs
```bash
# Use specific codec for output
python video.py -i input.mp4 -c xvid -b --blur -a
```

### 📁 Output Structure

When processing `example.mp4`, the following files are created:

```
video_output/
├── example_processed.mp4         # Final processed video
├── example_with_boxes.mp4        # Video with detection boxes
├── example_with_audio.mp4        # Audio-preserved version
└── example_with_boxes_audio.mp4  # Boxes + audio version

output_frames/                    # If using frames mode
├── frame_001.jpg                 # Individual processed frames
├── frame_002.jpg
└── ...

Logs/
├── video_processing_YYYYMMDD.log # Processing log
└── detection_report.json         # Detailed detection data
```

### ⚙️ Configuration Options

The video processor uses several configuration parameters from `CONFIG` dictionary:

```python
CONFIG = {
    'BLUR_STRENGTH_NORMAL': (23, 23, 30),     # Normal blur parameters
    'BLUR_STRENGTH_HIGH': (31, 31, 50),       # High blur parameters
    'FULL_BLUR_STRENGTH': (99, 99, 75),       # Full frame blur
    'ENHANCED_BLUR': False,                    # Enhanced blur mode
    'DETECTION_THRESHOLD': 0.2,               # Detection confidence
    'MONITOR_THRESHOLD_PERCENT': 10.0,        # Monitoring threshold %
    'MONITOR_THRESHOLD_COUNT': 5,             # Monitoring frame count
    'FULL_BLUR_LABELS': 2,                    # Labels to trigger full blur
    'OUTPUT_VIDEO_SUFFIX': '_processed.mp4'   # Output file suffix
}
```

### 🚨 Common Issues & Solutions

#### FFmpeg Not Found
```bash
# Install FFmpeg or specify path
python video.py -i video.mp4 --ffmpeg-path /path/to/ffmpeg
```

#### Large Video Files
```bash
# Video mode streams directly to the output video and avoids frame JPGs
python video.py -i large_video.mp4 -df --enhanced-blur
```

`-df` no longer waits until the end to clean up frame JPGs in video mode; intermediate frame images are not written in the first place. Use `-t frames` when you intentionally want per-frame image output.

#### Windows Unicode Paths

Image reads and writes use Unicode-safe OpenCV helpers, and video preview/processing has a Windows fallback for paths containing non-ASCII characters or special symbols. If OpenCV cannot open the original video path directly, SafeVision copies the input to a temporary ASCII path for the capture session.

#### Audio Sync Issues
```bash
# Use specific codec for better compatibility
python video.py -i video.mp4 -c mp4v -a
```

---

##  Blur Exception Rules

Create a file named `BlurException.rule` and define what labels to blur:

```ini
FACE_MALE = false
FEMALE_BREAST_EXPOSED = true
ANUS_EXPOSED = true
...
```

* `true` → Blur this label.
* `false` → Skip blurring for this label.

---

##  GUI Application

A modern desktop GUI is available in `SafeVisionGUI.py`.

###  Features

* Drag & drop images/videos
* Blurring / Masking / Bounding Box mode
* FFmpeg-based audio merging
* Codec & frame settings
* Real-time log panel and live preview
* Theme toggle (dark/light)

###  Launch
```bash
python SafeVisionGUI.py
```
---

## 🔧 Command-Line Arguments Reference

This section provides comprehensive documentation for all command-line arguments available in SafeVision's core components.

### 🖼️ main.py - Image Processing

**Purpose**: Process individual images with nudity/body and optional safety-object detection.

**Basic Usage**: `python main.py -i input.jpg [options]`

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | `str` | **Required** | Path to the input image file |
| `--output` | `-o` | `str` | Auto-generated | Path to save the censored image. If not provided, creates default path |
| `--blur` | `-b` | `flag` | `False` | Apply blur to NSFW regions instead of drawing detection boxes |
| `--exception` | `-e` | `str` | `BlurException.rule` | Path to the blur exception rules file for custom filtering |
| `--full_blur_rule` | `-fbr` | `int` | `0` | Number of exposed regions that trigger full image blur |
| `--detectors` | | `str` | `nude` | Detector mode: `nude`, `objects`, or `both` |
| `--object-model` | | `str` | `Models/safety_objects.onnx` | Safety-object ONNX model path |
| `--object-labels` | | `str` | `Models/safety_objects.labels.json` | Safety-object labels JSON path |
| `--object-threshold` | | `float` | `0.25` | Minimum safety-object confidence |

**Examples**:
```bash
# Basic detection with boxes
python main.py -i image.jpg

# Apply blur to detected regions
python main.py -i image.jpg -b

# Use custom blur rules
python main.py -i image.jpg -e custom_rules.rule

# Full blur if 2+ exposed regions found
python main.py -i image.jpg -fbr 2

# Detect cigarettes/alcohol/drugs only
python main.py -i image.jpg -b --detectors objects

# Detect both model families
python main.py -i image.jpg -b --detectors both
```

### 🎬 video.py - Video Processing

**Purpose**: Process video files with frame-by-frame nudity/body and optional safety-object detection.

**Basic Usage**: `python video.py -i input.mp4 [options]`

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | `str` | **Required** | Path to the input video file |
| `--output` | `-o` | `str` | Auto-generated | Path to save the processed video/frames |
| `--task` | `-t` | `str` | `video` | Processing mode: `frames` or `video` |
| `--video_output` | `-vo` | `str` | `video_output` | Directory for video output files |
| `--rule` | `-r` | `str` | `0/0` | Blur monitoring rule: `percentage/count` format |
| `--boxes` | `-b` | `flag` | `False` | Create video with detection boxes overlay |
| `--blur` | | `flag` | `False` | Apply blur when using boxes mode |
| `--with-audio` | `-a` | `flag` | `False` | Include original audio in output video |
| `--codec` | `-c` | `str` | `mp4v` | Video codec: `mp4v`, `avc1`, `xvid`, `mjpg` |
| `--ffmpeg-path` | | `str` | Auto-detect | Custom path to FFmpeg executable |
| `--delete-frames` | `-df` | `flag` | `False` | Delete frame images after video creation |
| `--enhanced-blur` | | `flag` | `False` | Use stronger blur that completely obscures content |
| `--full-blur-rule` | `-fbr` | `str` | None | Full blur rule: `labels/frames` format |
| `--color` | | `flag` | `False` | Use solid color instead of blur |
| `--mask-color` | | `str` | `0,0,0` | BGR color for masking (blue,green,red) |
| `--detectors` | | `str` | `nude` | Detector mode: `nude`, `objects`, or `both` |
| `--object-model` | | `str` | `Models/safety_objects.onnx` | Safety-object ONNX model path |
| `--object-labels` | | `str` | `Models/safety_objects.labels.json` | Safety-object labels JSON path |
| `--object-threshold` | | `float` | `0.25` | Minimum safety-object confidence |

**Examples**:
```bash
# Basic video processing
python video.py -i video.mp4

# Process only frames (no video creation)
python video.py -i video.mp4 -t frames

# Create video with boxes and audio
python video.py -i video.mp4 -b -a

# Apply blur monitoring (10% threshold or 50 frames)
python video.py -i video.mp4 -r 10/50

# Use enhanced blur with custom codec
python video.py -i video.mp4 --enhanced-blur -c avc1

# Full blur if 2+ exposed labels in 5+ frames
python video.py -i video.mp4 -fbr 2/5

# Detect safety objects only
python video.py -i video.mp4 -b --blur --detectors objects

# Detect nudity/body labels plus safety objects
python video.py -i video.mp4 -b --blur --detectors both
```

### 📹 live.py - Live Camera Processing

**Purpose**: Real-time nudity detection and censoring from camera input with advanced features.

**Basic Usage**: `python live.py [options]`

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--camera` | `-c` | `int` | `0` | Camera ID to use for input (0 = default camera) |
| `--rules` | `-r` | `str` | Auto-detect | Path to blur exception rules file |
| `--gender-detection` | `-g` | `flag` | `False` | Enable gender and age detection using best_gender.onnx |
| `--no-boxes` | | `flag` | `False` | Disable detection boxes display |
| `--privacy` | | `flag` | `False` | Start in privacy mode (no video display) |
| `--enhanced-blur` | | `flag` | `False` | Use enhanced blur mode for stronger effect |
| `--solid-color` | | `flag` | `False` | Use solid color masking instead of blur |
| `--mask-color` | | `str` | `0,0,0` | Color for solid masking in BGR format |
| `--auto-record` | | `flag` | `False` | Auto-record when nudity is detected |
| `--alert-threshold` | | `int` | `3` | Consecutive detections needed for alert |
| `--skip-frames` | | `int` | `2` | Process every nth frame for performance |

**Examples**:
```bash
# Basic live detection
python live.py

# Use camera 1 with gender detection
python live.py -c 1 -g

# Privacy mode with custom rules
python live.py --privacy -r custom.rule

# Enhanced blur with auto-recording
python live.py --enhanced-blur --auto-record

# Solid color masking (red)
python live.py --solid-color --mask-color 0,0,255
```

### 🖥️ safeVisionScreenGuard.py - Desktop Screen Guard

**Purpose**: Protect what is currently visible on your desktop without recording. It captures monitor pixels locally with `mss`, runs SafeVision detection, and draws a transparent always-on-top overlay that blocks or outlines detected unsafe regions.

**Basic Usage**: `python safeVisionScreenGuard.py [options]`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--monitor` | `int` | `1` | Monitor number from `--list-monitors` |
| `--list-monitors` | `flag` | `False` | Print available monitors and exit |
| `--mode` | `str` | `box` | Base mode: `box`, `blur`, `block`, `both`, or `privacy` |
| `--fps` | `float` | `5.0` | Detection frames per second |
| `--threshold` | `float` | `0.35` | Minimum confidence to display/block |
| `--providers` | `str` | Auto | Comma-separated ONNX Runtime providers |
| `--rules` | `str` | `BlurException.rule` | Rule file used to decide what is blocked |
| `--capture-backend` | `str` | `auto` | Pixel capture backend: `auto`, `gdi`, or `mss` |
| `--smooth-overlay` / `--no-smooth-overlay` | `flag` | `True` | Merge duplicate boxes, smooth jitter, cache blur patches, and skip unchanged redraws |
| `--smooth-iou` | `float` | `0.45` | Overlap threshold for duplicate merge and tracking |
| `--smooth-alpha` | `float` | `0.65` | Box follow speed; lower is steadier, higher follows faster |
| `--track-hold-ms` | `int` | `350` | Keep the last stable region visible through short missed detections |
| `--merge-nearby` / `--no-merge-nearby` | `flag` | `True` | Combine nearby body-part detections into one continuous protected region |
| `--merge-distance` | `int` | `140` | Pixel distance for merging nearby detections |
| `--merge-overlap` | `float` | `0.35` | Merge nested boxes when this much of the smaller box overlaps |
| `--feedback-safe-capture` / `--no-feedback-safe-capture` | `flag` | `False` | Briefly hide the overlay during screenshots; useful for diagnostics but can visibly blink |
| `--capture-hide-ms` | `int` | `20` | Milliseconds to hide the overlay during feedback-safe capture |
| `--drop-stale-on-screen-change` / `--keep-stale-regions` | `flag` | Drop stale | Remove held boxes immediately when the screen under them changes |
| `--stale-region-delta` | `float` | `10.0` | Pixel-change threshold used to release stale held boxes |
| `--screen-change-delta` | `float` | `28.0` | Whole-screen change threshold used to release stale held boxes |
| `--exclude-overlay-capture` / `--allow-overlay-capture` | `flag` | Exclude | Prevent the guard capture from seeing its own overlay on Windows |
| `--label-filter` | `str` | `exposed` | `exposed`, `body`, or `all` labels |
| `--respect-rules` / `--ignore-rules` | `flag` | Respect | Apply or bypass `BlurException.rule` |
| `--show-boxes` / `--no-boxes` | `flag` | Mode-based | Show or hide detection outlines |
| `--labels`, `--show-labels`, `--no-labels` | `flag` | `False` | Show or hide labels and confidence |
| `--blur` / `--no-blur` | `flag` | Mode-based | Draw localized blurred live screen patches |
| `--block-enabled` / `--no-block` | `flag` | Mode-based | Fill detected regions with a solid color |
| `--privacy-on-detection` / `--no-privacy` | `flag` | Mode-based | Cover the whole monitor on detection |
| `--blur-style` | `str` | `gaussian` | `gaussian` or `pixelate` localized blur |
| `--blur-strength` | `int` | `45` | Local blur or pixelation strength |
| `--mask-shape` | `str` | `rectangle` | `rectangle` or `ellipse` overlays |
| `--show-status` | `flag` | `False` | Show a small status HUD |
| `--no-click-through` | `flag` | `False` | Let overlay receive Escape/key input |

**Examples**:
```bash
# Show available monitors
python safeVisionScreenGuard.py --list-monitors

# Draw boxes around detected exposed regions on monitor 1
python safeVisionScreenGuard.py --monitor 1 --mode box

# Draw boxes and labels for exposed and covered body labels
python safeVisionScreenGuard.py --monitor 1 --mode box --label-filter body --show-labels --show-status

# Blur only detected regions while keeping the rest of the screen normal
python safeVisionScreenGuard.py --monitor 1 --mode blur --show-boxes --label-filter exposed

# Make one continuous stable box for body detections
python safeVisionScreenGuard.py --monitor 1 --mode blur --show-boxes --show-labels --label-filter body --track-hold-ms 350 --merge-distance 140

# Disable the default smoothing/debug the raw overlay behavior
python safeVisionScreenGuard.py --monitor 1 --mode blur --show-boxes --no-smooth-overlay

# Test detections without BlurException.rule hiding skipped labels
python safeVisionScreenGuard.py --monitor 1 --mode box --all-labels --ignore-rules --show-labels

# Cover the whole screen whenever unsafe content is detected
python safeVisionScreenGuard.py --monitor 1 --mode privacy

# Launch from the console app using settings/configs.json
python safeVisionCLI.py screen

# Change persistent Screen Guard settings from CLI
python safeVisionCLI.py settings set screen_guard.mode blur
python safeVisionCLI.py settings set screen_guard.show_labels true
python safeVisionCLI.py settings set screen_guard.label_filter body
python safeVisionCLI.py settings set screen_guard.smooth_overlay false
python safeVisionCLI.py settings set screen_guard.track_hold_ms 350
python safeVisionCLI.py settings set screen_guard.merge_distance 140
python safeVisionCLI.py settings set screen_guard.feedback_safe_capture false
python safeVisionCLI.py settings set screen_guard.drop_stale_on_screen_change true
```

OBS is not required for screen guard mode. If OBS WebSocket refuses connection in `live_streamer.py`, start OBS and enable WebSocket, or use `safeVisionScreenGuard.py` for direct desktop protection.

### 🎮 live_streamer.py - Streaming Edition

**Purpose**: Professional streaming solution with OBS integration, virtual camera, and advanced streaming features.

**Basic Usage**: `python live_streamer.py [options]`

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | `str` | `camera` | Input source: `camera`, `screen`, `window` |
| `--camera` | `-c` | `int` | `0` | Camera ID for camera input |
| `--monitor` | `-m` | `int` | `1` | Monitor number for screen capture |
| `--window` | `-w` | `str` | None | Window title to capture |
| `--resolution` | | `str` | `1920x1080` | Target resolution (WIDTHxHEIGHT) |
| `--fps` | | `int` | `60` | Target frames per second |
| `--ai-fps` | | `int` | `30` | AI processing frames per second |
| `--obs-host` | | `str` | `localhost` | OBS WebSocket host address |
| `--obs-port` | | `int` | `4455` | OBS WebSocket port |
| `--obs-password` | | `str` | Empty | OBS WebSocket password |
| `--auto-scene-switch` | | `flag` | `False` | Auto switch OBS scenes on detection |
| `--virtual-cam` | | `flag` | `False` | Enable virtual camera output |
| `--vcam-fps` | | `int` | `30` | Virtual camera frames per second |
| `--sensitivity` | | `float` | `0.2` | Detection sensitivity (0.1-0.9) |
| `--blur-strength` | | `int` | `30` | Blur strength for censoring |
| `--privacy` | | `flag` | `False` | Privacy mode (no display) |
| `--safe-timeout` | | `int` | `10` | Safe mode timeout in seconds |
| `--gpu` | | `flag` | `False` | Enable GPU acceleration |
| `--quality` | | `str` | `high` | Processing quality: `low`, `medium`, `high` |

**Examples**:
```bash
# Basic camera streaming
python live_streamer.py

# Screen capture for streaming
python live_streamer.py -i screen -m 1

# OBS integration with virtual camera
python live_streamer.py --virtual-cam --obs-host localhost

# High sensitivity with GPU acceleration
python live_streamer.py --sensitivity 0.1 --gpu

# Custom resolution for streaming
python live_streamer.py --resolution 1280x720 --fps 30
```

### 🌐 safevision_api.py - REST API Server

**Purpose**: RESTful API server for nudity detection with HTTP endpoints for web integration.

**Basic Usage**: `python safevision_api.py`

**Configuration**: The API server uses configuration constants defined in the file:

| Setting | Default | Description |
|---------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address (0.0.0.0 = all interfaces) |
| `PORT` | `5000` | Server port number |
| `MAX_CONTENT_LENGTH` | `50MB` | Maximum file upload size |
| `MAX_URL_DOWNLOAD_SIZE` | `50MB` | Maximum remote media URL download size |
| `DEFAULT_THRESHOLD` | `0.25` | Default detection confidence threshold |
| `DEFAULT_VIDEO_MAX_FRAMES` | `60` | Maximum sampled frames for video URL analysis |
| `DEFAULT_VIDEO_SAMPLE_SECONDS` | `1.0` | Sample one video frame every N seconds by default |

**Available Endpoints**:
- `GET /api/v1/health` - Health check and status
- `POST /api/v1/detect` - Image detection (multipart/form-data)
- `POST /api/v1/detect/base64` - Image detection (base64 JSON)
- `GET /api/v1/detect/url?url=...` - Detect image or video from an HTTP/HTTPS media URL
- `GET /api/v1/detect/media?url=...` - Alias for URL-based media detection
- `GET /api/v1/labels` - Available detection labels
- `GET /api/v1/stats` - API usage statistics

**Examples**:
```bash
# Start API server
python safevision_api.py

# Test with curl
curl -X GET http://localhost:5000/api/v1/health
curl -X POST -F "file=@test.jpg" http://localhost:5000/api/v1/detect

# Detect an image URL
curl "http://localhost:5000/api/v1/detect/url?url=https://example.com/image.jpg&threshold=0.3"

# Detect a video URL. Default scans sampled frames and returns media.is_video=true.
curl "http://localhost:5000/api/v1/detect/url?url=https://example.com/video.mp4&max_frames=30&sample_seconds=1"

# Force a full video frame scan when you need maximum coverage.
curl "http://localhost:5000/api/v1/detect/url?url=https://example.com/video.mp4&full_scan=true"
```

### 🖥️ safevision_gui.py - GUI Application

**Purpose**: Advanced PyQt5 desktop application with comprehensive UI for all SafeVision features.

**Basic Usage**: `python safevision_gui.py`

**Features**:
- No command-line arguments (fully GUI-driven)
- Drag & drop file interface
- Real-time processing preview
- Multi-tabbed workspace with split views
- Built-in file browser and media viewer
- Settings management and theme switching
- Integration with all core SafeVision components

**Examples**:
```bash
# Launch GUI application
python safevision_gui.py

# Note: All configuration is done through the GUI interface
# No command-line arguments are supported
```

### 🔗 Common Argument Patterns

**File Paths**:
- Use absolute paths for reliability: `C:\path\to\file.jpg`
- Relative paths work from script directory: `./images/test.jpg`
- Supports various image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Video formats: `.mp4`, `.avi`, `.mov`, `.mkv`

**Blur Rules Format**:
- Single number: `5` (count threshold)
- Percentage/Count: `10/50` (10% or 50 frames)
- Labels/Frames: `2/5` (2+ labels in 5+ frames)

**Color Format**:
- BGR format: `blue,green,red`
- Examples: `0,0,0` (black), `255,0,0` (blue), `0,255,0` (green), `0,0,255` (red)

**Performance Tips**:
- Use `--gpu` flag when available for faster processing
- Adjust `--skip-frames` for real-time performance
- Lower `--ai-fps` for better system responsiveness
- Use `--delete-frames` to save disk space in video processing

### 📚 Comprehensive Usage Examples

**Workflow 1: Complete Image Analysis**
```bash
# Process image with all options
python main.py -i input.jpg -b -e custom.rule -fbr 3 -o results/output.jpg
```

**Workflow 2: Professional Video Processing**
```bash
# High-quality video with audio and monitoring
python video.py -i stream.mp4 -a -r 5/20 --enhanced-blur -c avc1 --delete-frames
```

**Workflow 3: Live Streaming Setup**
```bash
# Complete streaming solution
python live_streamer.py -i camera --virtual-cam --obs-host localhost --gpu --sensitivity 0.15
```

**Workflow 4: Surveillance & Monitoring**
```bash
# Privacy-focused live detection with recording
python live.py -c 1 --privacy --auto-record --alert-threshold 2 -r surveillance.rule
```

**Workflow 5: API-Based Integration**
```bash
# Start API server and test detection
python safevision_api.py
# In another terminal:
curl -X POST -F "image=@test.jpg" -F "threshold=0.3" http://localhost:5000/api/v1/detect
```

---

# Live Nudity Detection with Blur Exception Rules & Gender/Age Detection

## Usage Examples

### 1. Basic Usage (Auto-loads blur rules)
```bash
python live.py
```
- Automatically checks for `BlurException.rule` in the same directory
- If found and not empty, uses existing rules
- If not found or empty, creates default rules with all labels set to `true`

### 2. Custom Rules File
```bash
python live.py -r custom_rules.rule
```
or
```bash
python live.py --rules custom_rules.rule
```

### 3. Enable Gender and Age Detection
```bash
python live.py -g
```
or
```bash
python live.py --gender-detection
```
- Requires `best_gender.onnx` model in the `Models/` folder
- Displays gender (Male/Female) and estimated age for detected faces
- Shows confidence scores for predictions

### 4. Complete Example with All Features
```bash
python live.py -c 0 -r my_rules.rule -g --auto-record --alert-threshold 2
```

## Blur Exception Rules

### Auto-Loading Behavior:
- **On startup**: Checks for `BlurException.rule` in same directory
- **If exists and not empty**: Loads existing rules
- **If missing or empty**: Creates default rules automatically
- **Optional -r parameter**: Override auto-loading with specific file

### Rules File Format:
The rules file uses the format: `LABEL = true/false`
- `true`: Apply blur/censoring to this detection type
- `false`: Skip blur/censoring for this detection type

### Available Labels:
- FEMALE_GENITALIA_EXPOSED
- MALE_GENITALIA_EXPOSED  
- FEMALE_BREAST_EXPOSED
- MALE_BREAST_EXPOSED
- BUTTOCKS_EXPOSED
- ANUS_EXPOSED
- BELLY_EXPOSED
- FEET_EXPOSED
- ARMPITS_EXPOSED
- FACE_FEMALE
- FACE_MALE
- FEMALE_GENITALIA_COVERED
- FEMALE_BREAST_COVERED
- BUTTOCKS_COVERED
- ANUS_COVERED
- BELLY_COVERED
- FEET_COVERED
- ARMPITS_COVERED

## Gender/Age Detection

### Requirements:
- `best_gender.onnx` model in `Models/` directory
- Model format should match the example in `myai/run.py`
- Input: 224x224 RGB images
- Output: Gender classification + Age regression

### Features:
- **Real-time Analysis**: Analyzes detected faces in real-time
- **Gender Classification**: Male/Female with confidence scores
- **Age Estimation**: Estimated age in years
- **Visual Feedback**: Shows results on status overlay
- **Runtime Toggle**: Press 'G' key to toggle on/off during runtime

### Status Display:
When enabled, shows:
- `Gender Detection: ON/FAILED` in status overlay
- For each detected face: `Female, 25y (0.95)` format
- Gender, estimated age, and confidence score

## Command Line Arguments

- `-c, --camera`: Camera ID (default: 0)
- `-r, --rules`: Path to blur exception rules file (optional - auto-loads if exists)
- `-g, --gender-detection`: Enable gender and age detection
- `--no-boxes`: Disable detection boxes
- `--privacy`: Start in privacy mode
- `--enhanced-blur`: Use enhanced blur mode
- `--solid-color`: Use solid color masking
- `--mask-color`: Color for solid masking (BGR format)
- `--auto-record`: Auto-record when nudity detected
- `--alert-threshold`: Consecutive detections for alert
- `--skip-frames`: Process every nth frame for performance

## Runtime Controls

- **SPACE**: Toggle recording
- **B**: Toggle detection boxes
- **P**: Toggle privacy mode  
- **G**: Toggle gender detection (if enabled)
- **Q**: Quit application

## Example Rule Configurations

### Strict Mode (Only explicit content)
```
FEMALE_GENITALIA_EXPOSED = true
MALE_GENITALIA_EXPOSED = true
FEMALE_BREAST_EXPOSED = true
BUTTOCKS_EXPOSED = true
ANUS_EXPOSED = true
MALE_BREAST_EXPOSED = false
BELLY_EXPOSED = false
FEET_EXPOSED = false
ARMPITS_EXPOSED = false
FACE_FEMALE = false
FACE_MALE = false
# All COVERED labels = false
```

### Conservative Mode (Maximum privacy)
```
# All EXPOSED labels = true
# Covered areas also censored for privacy
FEMALE_GENITALIA_COVERED = true
FEMALE_BREAST_COVERED = true
BUTTOCKS_COVERED = true
FACE_FEMALE = false
FACE_MALE = false
```

### Face Analysis Mode (Skip censoring, focus on gender/age)
```
# All labels = false (no censoring)
# Use with -g flag for gender/age analysis only
```

---

##  How It Works

###  Pipeline

1. **Preprocessing** – Resize and normalize input image or video frames.
2. **Inference** – Use `ONNXRuntime` to run the selected detector model(s): `best.onnx`, `safety_objects.onnx`, or both.
3. **Postprocessing** – Convert YOLO outputs into bounding boxes, labels, scores, categories, and source model metadata.
4. **Censorship** – Apply blur/mask/box per user rules, including `BlurException.rule` object-label rules.
5. **Rendering** – Save censored images/videos to output folders.

---

## 📂 Output Directory Structure

| Folder          | Description                      |
| --------------- | -------------------------------- |
| `output/`       | Final censored images/videos     |
| `blur/`         | Full blurred content             |
| `prosses/`      | Detection-only visuals (no blur) |
| `video_output/` | Rendered final videos            |

---

## 📷 NSFW Demo (Spoiler Warning)

<details>
  <summary>⚠️ Click to Show Example Output Image using SafeVisionGUI (Contains NSFW Examples with Blurring)</summary>
  <p>

![Blurred Output](https://github.com/user-attachments/assets/a62d64d1-199c-4d28-a34f-46c53ba056e6)

*Example showing SafeVision blurring applied on exposed content. using the SafeVisionGUI*

  </p>
</details>
<details>
  <summary>⚠️ Click to Show Example Output using CLI (main.py) (Contains NSFW Examples with Blurring)</summary>
  <p>

![Blurred Output](https://github.com/user-attachments/assets/5a9b362b-e103-427c-b10d-8f6157578f10)

*Example showing SafeVision blurring applied on exposed content.*

  </p>
</details>

---
### Conclusion
SafeVision provides a robust solution for detecting and blurring nudity in images and videos, making it a valuable tool for content moderation and safe media sharing. Follow the instructions in this README to set up and use SafeVision effectively. 

---

## 🛠 Maintainer & Support

Maintained by [@im-syn](https://github.com/im-syn)
Pull requests, issues, and contributions are welcome!

---

> **Note:** This project is intended for ethical and responsible use only. Always follow legal and platform-specific content handling policies.


---
## ☕ Like It?

If this helped you, consider giving the repo a 🌟 or forking it to your toolkit.
Thank you for using **SafeContentText**! Feel free to open issues or PRs for improvements.
