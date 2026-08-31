"""Environment-backed settings for the standalone SafeVision Web API."""

import os
from pathlib import Path


API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent


def _load_local_env(path=API_DIR / ".env"):
    """Load simple KEY=VALUE entries without overriding service variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _bool(name, default=False):
    return str(os.environ.get(name, str(default))).strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _path(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return str(value.resolve())


def build_api_config(default_age_model):
    _load_local_env()
    runtime = API_DIR / "runtime"
    return {
        "HOST": os.environ.get("SAFEVISION_API_HOST", "127.0.0.1"),
        "PORT": _int("SAFEVISION_API_PORT", 5000),
        "DEBUG": _bool("SAFEVISION_API_DEBUG", False),
        "MAX_CONTENT_LENGTH": _int("SAFEVISION_API_MAX_UPLOAD_MB", 50) * 1024 * 1024,
        "MAX_URL_DOWNLOAD_SIZE": _int("SAFEVISION_API_MAX_URL_MB", 50) * 1024 * 1024,
        "URL_TIMEOUT": _int("SAFEVISION_API_URL_TIMEOUT", 20),
        "UPLOAD_FOLDER": _path("SAFEVISION_API_UPLOAD_FOLDER", runtime / "uploads"),
        "OUTPUT_FOLDER": _path("SAFEVISION_API_OUTPUT_FOLDER", runtime / "outputs"),
        "TEMP_FOLDER": _path("SAFEVISION_API_TEMP_FOLDER", runtime / "temp"),
        "RULE_FILE": _path("SAFEVISION_RULE_FILE", PROJECT_ROOT / "BlurException.rule"),
        "NSFW_MODEL": _path("SAFEVISION_NSFW_MODEL", PROJECT_ROOT / "Models" / "best.onnx"),
        "AGE_GENDER_MODEL": _path("SAFEVISION_AGE_GENDER_MODEL", default_age_model),
        "ALLOWED_EXTENSIONS": {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "mp4", "avi", "mov", "mkv", "webm", "m4v"},
        "IMAGE_EXTENSIONS": {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"},
        "VIDEO_EXTENSIONS": {"mp4", "avi", "mov", "mkv", "webm", "m4v"},
        "DEFAULT_VIDEO_MAX_FRAMES": _int("SAFEVISION_API_VIDEO_MAX_FRAMES", 60),
        "DEFAULT_VIDEO_SAMPLE_SECONDS": _float("SAFEVISION_API_VIDEO_SAMPLE_SECONDS", 1.0),
        "DEFAULT_THRESHOLD": _float("SAFEVISION_API_THRESHOLD", 0.25),
        "DEFAULT_CHECKS": os.environ.get("SAFEVISION_DETECTORS", "nude,age,gender"),
        "UNDERAGE_AGE": _float("SAFEVISION_UNDERAGE_AGE", 18),
        "AGE_REVIEW_MARGIN": _float("SAFEVISION_AGE_REVIEW_MARGIN", 3),
        "BLOCK_IF_NSFW_AND_CHILD": _bool("SAFEVISION_BLOCK_IF_NSFW_AND_CHILD", True),
        "PROTECTION_FORCES_FULL_COVER": _bool("SAFEVISION_PROTECTION_FORCES_FULL_COVER", True),
        "FULL_COVER_MODE": os.environ.get("SAFEVISION_FULL_COVER_MODE", ""),
        "FULL_COVER_COLOR": os.environ.get("SAFEVISION_FULL_COVER_COLOR", ""),
        "FULL_COVER_TEXT_COLOR": os.environ.get("SAFEVISION_FULL_COVER_TEXT_COLOR", ""),
        "FULL_COVER_SHOW_TEXT": os.environ.get("SAFEVISION_FULL_COVER_SHOW_TEXT", ""),
        "CLEANUP_INTERVAL": _int("SAFEVISION_API_CLEANUP_INTERVAL", 3600),
        "MAX_FILE_AGE": _int("SAFEVISION_API_MAX_FILE_AGE", 86400),
    }
