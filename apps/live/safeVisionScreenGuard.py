#!/usr/bin/env python3
"""
SafeVision Screen Guard

Real-time local screen protection. Captures the current monitor with mss,
runs SafeVision detection, and draws a transparent topmost overlay that blocks
or outlines detected unsafe screen regions. No frames are recorded or saved.
"""

import argparse
import ctypes
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

try:
    import mss
except ImportError:
    mss = None

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:
    Image = None
    ImageDraw = None
    ImageTk = None

from safevision_utils import (
    label_group,
    label_matches_filter,
    load_blur_exception_rules,
    load_protection_rules,
    parse_detector_selection,
    parse_provider_list,
    protection_nsfw_summary,
)
from age_gender_detector import (
    AgeGenderDetector,
    default_model_path as default_age_gender_model_path,
    evaluate_protection_policy,
    face_boxes_from_detections,
    face_result_to_detection,
)
from video import NudeDetector


APP_DIR = PROJECT_ROOT
TRANSPARENT_COLOR = "#010203"
DEFAULT_BLOCK_COLOR = "0,0,0"
DEFAULT_LABEL_BG = "#111111"

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
LWA_COLORKEY = 0x00000001
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
SW_HIDE = 0
SW_SHOWNA = 8
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


def set_dpi_awareness():
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def hex_to_colorref(value):
    value = value.lstrip("#")
    if len(value) != 6:
        value = "010203"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return red | (green << 8) | (blue << 16)


def configure_transparent_window(window, click_through=True, exclude_from_capture=True):
    if os.name != "nt":
        return
    try:
        window.update_idletasks()
        hwnd = window.winfo_id()
        user32 = ctypes.windll.user32
        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ex_style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW
        if click_through:
            ex_style |= WS_EX_TRANSPARENT
        else:
            ex_style &= ~WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)
        user32.SetLayeredWindowAttributes(hwnd, hex_to_colorref(TRANSPARENT_COLOR), 0, LWA_COLORKEY)
        affinity = WDA_EXCLUDEFROMCAPTURE if exclude_from_capture else WDA_NONE
        try:
            user32.SetWindowDisplayAffinity(hwnd, affinity)
        except Exception:
            pass
    except Exception:
        pass


def bgr_to_hex(value):
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        parts = DEFAULT_BLOCK_COLOR.split(",")
    try:
        b, g, r = [max(0, min(255, int(part))) for part in parts]
    except ValueError:
        b, g, r = 0, 0, 0
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_color(value, default):
    value = str(value or "").strip()
    if value.startswith("#") and len(value) == 7:
        return value
    if "," in value:
        return bgr_to_hex(value)
    return default


def normalize_args(args):
    if args.show_boxes is None:
        args.show_boxes = args.mode in {"box", "both", "block"}
    if args.block_enabled is None:
        args.block_enabled = args.mode in {"block", "both"}
    if args.blur_enabled is None:
        args.blur_enabled = args.mode == "blur"
    if args.privacy_on_detection is None:
        args.privacy_on_detection = args.mode == "privacy"

    args.line_width = max(1, int(args.line_width))
    args.box_padding = max(0, int(args.box_padding))
    args.min_box_area = max(0, int(args.min_box_area))
    args.blur_strength = max(3, int(args.blur_strength))
    args.hold_ms = max(0, int(args.hold_ms))
    args.track_hold_ms = max(0, int(args.track_hold_ms))
    args.fps = max(0.2, float(args.fps))
    args.overlay_fps = max(1.0, float(args.overlay_fps))
    args.threshold = max(0.0, min(1.0, float(args.threshold)))
    if args.underage_age is not None:
        args.underage_age = max(1.0, min(100.0, float(args.underage_age)))
    if args.age_review_margin is not None:
        args.age_review_margin = max(0.0, min(25.0, float(args.age_review_margin)))
    args.min_face_size = max(16, int(args.min_face_size))
    args.face_padding = max(0.0, min(1.0, float(args.face_padding)))
    args.smooth_iou = max(0.05, min(0.95, float(args.smooth_iou)))
    args.smooth_alpha = max(0.05, min(1.0, float(args.smooth_alpha)))
    args.stable_score_alpha = max(0.0, min(1.0, float(args.stable_score_alpha)))
    args.merge_overlap = max(0.05, min(1.0, float(args.merge_overlap)))
    args.merge_distance = max(0, int(args.merge_distance))
    args.feedback_delta = max(0.0, float(args.feedback_delta))
    args.capture_hide_ms = max(0, int(args.capture_hide_ms))
    args.stale_region_delta = max(0.0, float(args.stale_region_delta))
    args.screen_change_delta = max(0.0, float(args.screen_change_delta))
    args.label_filter = str(args.label_filter or "exposed").lower()
    return args


def list_monitors():
    if mss is None:
        print("mss is not installed. Install requirements.txt first.")
        return 2
    with mss.mss() as sct:
        for index, monitor in enumerate(sct.monitors[1:], start=1):
            print(
                f"{index}: {monitor['width']}x{monitor['height']} "
                f"at ({monitor['left']}, {monitor['top']})"
            )


class MssCaptureBackend:
    def __init__(self, monitor):
        self.monitor = monitor
        self.sct = None

    def __enter__(self):
        self.sct = mss.mss()
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    def grab_bgr(self):
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


class WindowsGdiCaptureBackend:
    def __init__(self, monitor):
        if os.name != "nt":
            raise RuntimeError("Windows GDI capture is only available on Windows.")
        self.monitor = monitor
        self.width = int(monitor["width"])
        self.height = int(monitor["height"])
        self.left = int(monitor["left"])
        self.top = int(monitor["top"])
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self.user32.GetDC.argtypes = [ctypes.c_void_p]
        self.user32.GetDC.restype = ctypes.c_void_p
        self.user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.user32.ReleaseDC.restype = ctypes.c_int
        self.gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        self.gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
        self.gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
        self.gdi32.CreateDIBSection.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.gdi32.CreateDIBSection.restype = ctypes.c_void_p
        self.gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.gdi32.SelectObject.restype = ctypes.c_void_p
        self.gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        self.gdi32.DeleteObject.restype = ctypes.c_int
        self.gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
        self.gdi32.DeleteDC.restype = ctypes.c_int
        self.gdi32.BitBlt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        self.gdi32.BitBlt.restype = ctypes.c_int
        self.gdi32.GetDIBits.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.gdi32.GetDIBits.restype = ctypes.c_int
        self.screen_dc = None
        self.memory_dc = None
        self.bitmap = None
        self.old_bitmap = None
        self.bits = ctypes.c_void_p()
        self.bitmap_info = BITMAPINFO()
        self.bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self.bitmap_info.bmiHeader.biWidth = self.width
        self.bitmap_info.bmiHeader.biHeight = -self.height
        self.bitmap_info.bmiHeader.biPlanes = 1
        self.bitmap_info.bmiHeader.biBitCount = 32
        self.bitmap_info.bmiHeader.biCompression = 0

    def __enter__(self):
        self.screen_dc = self.user32.GetDC(0)
        if not self.screen_dc:
            raise RuntimeError("Could not get Windows screen DC.")
        self.memory_dc = self.gdi32.CreateCompatibleDC(self.screen_dc)
        self.bitmap = self.gdi32.CreateDIBSection(
            self.screen_dc,
            ctypes.byref(self.bitmap_info),
            DIB_RGB_COLORS,
            ctypes.byref(self.bits),
            None,
            0,
        )
        if not self.memory_dc or not self.bitmap:
            raise RuntimeError("Could not create Windows GDI capture objects.")
        self.old_bitmap = self.gdi32.SelectObject(self.memory_dc, self.bitmap)
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.memory_dc and self.old_bitmap:
            self.gdi32.SelectObject(self.memory_dc, self.old_bitmap)
        if self.bitmap:
            self.gdi32.DeleteObject(self.bitmap)
        if self.memory_dc:
            self.gdi32.DeleteDC(self.memory_dc)
        if self.screen_dc:
            self.user32.ReleaseDC(0, self.screen_dc)
        self.screen_dc = None
        self.memory_dc = None
        self.bitmap = None
        self.old_bitmap = None

    def grab_bgr(self):
        ok = self.gdi32.BitBlt(
            self.memory_dc,
            0,
            0,
            self.width,
            self.height,
            self.screen_dc,
            self.left,
            self.top,
            SRCCOPY,
        )
        if not ok:
            error = ctypes.get_last_error()
            raise RuntimeError(f"Windows GDI BitBlt capture failed (error {error}).")
        if not self.bits:
            raise RuntimeError("Windows GDI capture did not return a DIB buffer.")
        size = self.width * self.height * 4
        array_type = ctypes.c_ubyte * size
        bgra = np.ctypeslib.as_array(array_type.from_address(self.bits.value)).reshape((self.height, self.width, 4))
        return bgra[:, :, :3].copy()


class AutoCaptureBackend:
    def __init__(self, monitor):
        self.monitor = monitor
        self.backend = None
        self.backend_name = None

    def __enter__(self):
        if os.name == "nt":
            try:
                self.backend = WindowsGdiCaptureBackend(self.monitor)
                self.backend.__enter__()
                self.backend_name = "gdi"
                return self
            except Exception as exc:
                print(f"Windows GDI capture unavailable ({exc}). Falling back to mss.")
                try:
                    if self.backend is not None:
                        self.backend.__exit__(None, None, None)
                except Exception:
                    pass
        self.backend = MssCaptureBackend(self.monitor)
        self.backend.__enter__()
        self.backend_name = "mss"
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.backend is not None:
            self.backend.__exit__(exc_type, exc, tb)
            self.backend = None

    def switch_to_mss(self, reason):
        print(f"Capture backend {self.backend_name} failed ({reason}). Falling back to mss.")
        try:
            if self.backend is not None:
                self.backend.__exit__(None, None, None)
        except Exception:
            pass
        self.backend = MssCaptureBackend(self.monitor)
        self.backend.__enter__()
        self.backend_name = "mss"

    def grab_bgr(self):
        try:
            return self.backend.grab_bgr()
        except Exception as exc:
            if self.backend_name != "mss":
                self.switch_to_mss(exc)
                return self.backend.grab_bgr()
            raise


def create_capture_backend(args, monitor):
    backend = args.capture_backend
    if backend == "auto":
        return AutoCaptureBackend(monitor)
    if backend == "gdi":
        return WindowsGdiCaptureBackend(monitor)
    return MssCaptureBackend(monitor)


class ScreenGuard:
    def __init__(self, args):
        if mss is None:
            raise RuntimeError("mss is not installed. Run: pip install -r requirements.txt")

        self.args = args
        self.running = threading.Event()
        self.running.set()
        self.result_queue = queue.Queue(maxsize=1)
        self.latest_detections = []
        self.latest_frame = None
        self.latest_frame_signature = None
        self.latest_demographics = {
            "enabled": False,
            "underage_detected": None,
            "review_required": False,
            "faces_detected": 0,
            "faces": [],
        }
        self.latest_protection_policy = {
            "blocked": False,
            "verdict": "ALLOW",
            "reasons": [],
            "nsfw_detected": False,
            "underage_detected": False,
            "age_review_required": False,
        }
        self.last_detection_time = 0.0
        self.last_positive_time = 0.0
        self.last_positive_screen_hash = None
        self.previous_smoothed_detections = []
        self.overlay_visible = True
        self.block_color = bgr_to_hex(args.block_color)
        self.outline_color = normalize_color(args.outline_color, "#ff3333")
        self.safe_outline_color = normalize_color(args.safe_outline_color, "#3388ff")
        self.rule_skipped_outline_color = normalize_color(args.rule_skipped_outline_color, "#aaaaaa")
        self.label_bg = normalize_color(args.label_bg, DEFAULT_LABEL_BG)
        self.label_color = args.label_color or "white"
        self._tk_images = []
        self._blur_cache = {}
        self._last_draw_signature = None
        self._last_status_draw = 0.0
        self.stats = {
            "frames": 0,
            "detections": 0,
            "last_fps_time": time.time(),
            "fps": 0.0,
        }

        if args.blur_enabled and (Image is None or ImageTk is None):
            raise RuntimeError("Pillow is required for screen blur mode. Run: pip install pillow")

        with mss.mss() as sct:
            if args.monitor < 1 or args.monitor >= len(sct.monitors):
                raise ValueError(f"Monitor {args.monitor} is not available. Use --list-monitors.")
            self.monitor = dict(sct.monitors[args.monitor])

        providers = parse_provider_list(args.providers)
        self.enabled_detectors = [
            name for name in parse_detector_selection(args.detectors) if name in {"nude", "age", "gender"}
        ]
        if not self.enabled_detectors:
            raise ValueError("No checks enabled. Use --detectors nsfw,age,gender or --detectors none only for non-monitoring commands.")

        self.blur_exception_rules = load_blur_exception_rules(args.rules)
        self.protection_rules = load_protection_rules(args.rules)
        if args.underage_age is None:
            args.underage_age = float(self.protection_rules["UNDERAGE_AGE"])
        if args.age_review_margin is None:
            args.age_review_margin = float(self.protection_rules["AGE_REVIEW_MARGIN"])
        self.detector = None
        if "nude" in self.enabled_detectors:
            self.detector = NudeDetector(providers=providers)
            self.detector.load_exception_rules(args.rules)

        self.demographic_detector = None
        if "age" in self.enabled_detectors or "gender" in self.enabled_detectors:
            self.demographic_detector = AgeGenderDetector(
                model_path=args.age_gender_model,
                providers=providers,
                min_face_size=args.min_face_size,
                face_padding=args.face_padding,
            )
            self.demographic_detector.load()

        set_dpi_awareness()
        self.root = tk.Tk()
        self.root.title("SafeVision Screen Guard")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=TRANSPARENT_COLOR)
        try:
            self.root.attributes("-transparentcolor", TRANSPARENT_COLOR)
        except tk.TclError:
            pass
        self.root.update_idletasks()
        self.hwnd = self.root.winfo_id()

        geometry = (
            f"{self.monitor['width']}x{self.monitor['height']}"
            f"+{self.monitor['left']}+{self.monitor['top']}"
        )
        self.root.geometry(geometry)
        self.canvas = tk.Canvas(
            self.root,
            width=self.monitor["width"],
            height=self.monitor["height"],
            highlightthickness=0,
            bg=TRANSPARENT_COLOR,
        )
        self.canvas.pack(fill="both", expand=True)

        self.root.after(
            50,
            lambda: configure_transparent_window(
                self.root,
                args.click_through,
                args.exclude_overlay_capture,
            ),
        )
        self.root.after(150, self.hide_overlay)

        if not args.click_through:
            self.root.bind("<Escape>", lambda _event: self.stop())

        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def stop(self):
        self.running.clear()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def hide_overlay(self):
        if not self.overlay_visible:
            return
        try:
            self.root.withdraw()
            self.overlay_visible = False
        except tk.TclError:
            pass

    def show_overlay(self):
        if self.overlay_visible:
            return
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            configure_transparent_window(
                self.root,
                self.args.click_through,
                self.args.exclude_overlay_capture,
            )
            self.overlay_visible = True
        except tk.TclError:
            pass

    def hide_window_for_capture(self):
        if (
            os.name != "nt"
            or not self.args.feedback_safe_capture
            or not self.overlay_visible
            or not self.hwnd
        ):
            return False
        try:
            ctypes.windll.user32.ShowWindow(self.hwnd, SW_HIDE)
            if self.args.capture_hide_ms:
                time.sleep(self.args.capture_hide_ms / 1000.0)
            return True
        except Exception:
            return False

    def restore_window_after_capture(self, was_hidden):
        if not was_hidden or os.name != "nt" or not self.hwnd:
            return
        try:
            ctypes.windll.user32.ShowWindow(self.hwnd, SW_SHOWNA)
        except Exception:
            pass

    def capture_loop(self):
        interval = 1.0 / max(1, self.args.fps)
        with create_capture_backend(self.args, self.monitor) as capture:
            while self.running.is_set():
                start = time.time()
                try:
                    hidden_for_capture = self.hide_window_for_capture()
                    frame = capture.grab_bgr()
                    self.restore_window_after_capture(hidden_for_capture)
                    screen_hash = self.screen_hash(frame)
                    raw_nsfw_detections = self.detector.detect_frame(frame) if self.detector else []
                    demographics = self.analyze_demographics(frame, raw_nsfw_detections)
                    detections = self.filter_detections(raw_nsfw_detections, frame, screen_hash)
                    nsfw_gate = protection_nsfw_summary(detections, self.protection_rules)
                    protection_policy = evaluate_protection_policy(
                        nsfw_gate["detected"],
                        demographics,
                        block_if_nsfw_and_underage=self.protection_rules["BLOCK_IF_NSFW_AND_CHILD"],
                        block_if_underage=self.protection_rules["BLOCK_IF_CHILD"],
                        block_on_age_review=self.protection_rules["BLOCK_ON_AGE_REVIEW"],
                    )
                    protection_policy["nsfw_gate"] = nsfw_gate
                    self.latest_demographics = demographics
                    self.latest_protection_policy = protection_policy
                    if self.args.show_demographics:
                        detections.extend(self.demographic_overlay_detections(demographics))
                    frame_signature = self.attach_region_signatures(frame, detections)
                    self.publish_detections(
                        detections,
                        frame if self.args.blur_enabled else None,
                        frame_signature,
                    )
                    self.update_stats(detections)
                except Exception as exc:
                    self.restore_window_after_capture(locals().get("hidden_for_capture", False))
                    print(f"Screen guard detection error: {exc}")
                    time.sleep(1.0)

                elapsed = time.time() - start
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    def analyze_demographics(self, frame, raw_nsfw_detections):
        if self.demographic_detector is None:
            return {
                "enabled": False,
                "age_enabled": False,
                "gender_enabled": False,
                "faces_detected": 0,
                "faces": [],
                "underage_detected": None,
                "review_required": False,
            }
        face_boxes = face_boxes_from_detections(
            raw_nsfw_detections,
            width=frame.shape[1],
            height=frame.shape[0],
        )
        return self.demographic_detector.analyze_frame(
            frame,
            face_boxes=face_boxes or None,
            age_enabled="age" in self.enabled_detectors,
            gender_enabled="gender" in self.enabled_detectors,
            age_threshold=self.args.underage_age,
            review_margin=self.args.age_review_margin,
            face_source="safevision_nsfw_faces" if face_boxes else None,
        )

    def demographic_overlay_detections(self, demographics):
        overlays = []
        for face in demographics.get("faces", []):
            detection = face_result_to_detection(face)
            detection["label_group"] = "demographic"
            detection["rule_allowed"] = True
            detection["censor"] = False
            overlays.append(detection)
        return overlays

    def filter_detections(self, detections, frame=None, screen_hash=None):
        filtered = []
        for detection in detections:
            label = detection["class"]
            score = detection["score"]
            if score < self.args.threshold:
                continue
            if not label_matches_filter(label, self.args.label_filter):
                continue
            rule_allowed = bool(self.blur_exception_rules.get(label, True))
            if self.args.respect_rules and not rule_allowed:
                continue
            x, y, w, h = detection["box"]
            padding = self.args.box_padding
            x1 = max(0, min(self.monitor["width"], x - padding))
            y1 = max(0, min(self.monitor["height"], y - padding))
            x2 = max(0, min(self.monitor["width"], x + w + padding))
            y2 = max(0, min(self.monitor["height"], y + h + padding))
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < self.args.min_box_area:
                continue
            cleaned = dict(detection)
            cleaned["box"] = [x1, y1, x2 - x1, y2 - y1]
            cleaned["rule_allowed"] = rule_allowed
            cleaned["label_group"] = label_group(label)
            filtered.append(cleaned)
        if self.args.smooth_overlay:
            filtered = self.smooth_detections(filtered, frame, screen_hash)
        elif self.args.blur_enabled:
            for detection in filtered:
                self.attach_blur_source(detection, frame)
        return filtered

    @staticmethod
    def detection_iou(first, second):
        ax, ay, aw, ah = first["box"]
        bx, by, bw, bh = second["box"]
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def union_box(first_box, second_box):
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def intersection_area(first_box, second_box):
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    def expanded_intersects(self, first_box, second_box):
        distance = self.args.merge_distance
        if distance <= 0:
            return False
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box
        ax1 = ax - distance
        ay1 = ay - distance
        ax2 = ax + aw + distance
        ay2 = ay + ah + distance
        bx1 = bx - distance
        by1 = by - distance
        bx2 = bx + bw + distance
        by2 = by + bh + distance
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    def should_merge_detections(self, first, second):
        if self.detection_iou(first, second) >= self.args.smooth_iou:
            return True
        intersection = self.intersection_area(first["box"], second["box"])
        if intersection > 0:
            first_area = max(1, first["box"][2] * first["box"][3])
            second_area = max(1, second["box"][2] * second["box"][3])
            if intersection / min(first_area, second_area) >= self.args.merge_overlap:
                return True
        if self.args.merge_nearby and self.expanded_intersects(first["box"], second["box"]):
            return True
        if self.args.merge_nearby and self.same_visual_target(first["box"], second["box"]):
            return True
        return False

    @staticmethod
    def box_center(box):
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    def same_visual_target(self, first_box, second_box):
        ax, ay = self.box_center(first_box)
        bx, by = self.box_center(second_box)
        distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        first_size = max(first_box[2], first_box[3])
        second_size = max(second_box[2], second_box[3])
        dynamic_distance = max(self.args.merge_distance, int(max(first_size, second_size) * 0.9))
        return distance <= dynamic_distance

    def merge_overlapping_detections(self, detections):
        merged = []
        for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
            target = None
            for existing in merged:
                if self.should_merge_detections(existing, detection):
                    target = existing
                    break
            if target is None:
                copy = dict(detection)
                copy["merged_labels"] = [detection["class"]]
                merged.append(copy)
                continue

            target["box"] = self.union_box(target["box"], detection["box"])
            if detection["score"] > target["score"]:
                target["class"] = detection["class"]
                target["score"] = detection["score"]
                target["label_group"] = detection.get("label_group", target.get("label_group"))
            target["rule_allowed"] = target.get("rule_allowed", True) and detection.get("rule_allowed", True)
            labels = target.setdefault("merged_labels", [target["class"]])
            if detection["class"] not in labels:
                labels.append(detection["class"])
        return merged

    def interpolate_box(self, previous_box, current_box):
        alpha = self.args.smooth_alpha
        return [
            int(round(previous_box[index] + (current_box[index] - previous_box[index]) * alpha))
            for index in range(4)
        ]

    def stabilize_detection_metadata(self, detection, previous, feedback_only=False):
        if previous is None:
            detection["stable_score"] = float(detection.get("score", 0.0))
            return detection

        if feedback_only:
            detection["class"] = previous.get("class", detection.get("class"))
            detection["score"] = previous.get("score", detection.get("score", 0.0))
            detection["stable_score"] = previous.get("stable_score", detection.get("score", 0.0))
            detection["label_group"] = previous.get("label_group", detection.get("label_group"))
            detection["merged_labels"] = previous.get("merged_labels", detection.get("merged_labels", []))
            return detection

        previous_score = float(previous.get("stable_score", previous.get("score", detection.get("score", 0.0))))
        current_score = float(detection.get("score", 0.0))
        alpha = self.args.stable_score_alpha
        stable_score = previous_score + (current_score - previous_score) * alpha
        detection["stable_score"] = stable_score
        detection["score"] = stable_score
        if previous.get("label_group") == detection.get("label_group") and current_score <= previous_score + 0.08:
            detection["class"] = previous.get("class", detection.get("class"))
        return detection

    def screen_changed_since_positive(self, screen_hash):
        if not self.args.drop_stale_on_screen_change:
            return False
        delta = self.region_delta(self.last_positive_screen_hash, screen_hash)
        return delta is not None and delta > self.args.screen_change_delta

    def smooth_detections(self, detections, frame=None, screen_hash=None):
        now = time.time()
        merged = self.merge_overlapping_detections(detections)
        if not merged:
            if self.screen_changed_since_positive(screen_hash):
                self.previous_smoothed_detections = []
                return []
            if (
                self.previous_smoothed_detections
                and now - self.last_positive_time <= self.args.track_hold_ms / 1000.0
            ):
                held = []
                for detection in self.previous_smoothed_detections:
                    copy = dict(detection)
                    if not self.held_detection_is_still_valid(copy, frame):
                        continue
                    copy["held"] = True
                    held.append(copy)
                if not held:
                    self.previous_smoothed_detections = []
                return held
            self.previous_smoothed_detections = []
            return []

        smoothed = []
        real_detection_seen = False
        for detection in merged:
            best_previous = None
            best_iou = 0.0
            for previous in self.previous_smoothed_detections:
                overlap = self.detection_iou(previous, detection)
                if overlap > best_iou or self.should_merge_detections(previous, detection):
                    best_iou = overlap
                    best_previous = previous
            copy = dict(detection)
            feedback_only = self.detection_looks_like_overlay_feedback(copy, frame, best_previous)
            if feedback_only and self.screen_changed_since_positive(screen_hash):
                continue
            if feedback_only:
                copy["box"] = list(best_previous["box"])
                copy["feedback_only"] = True
            elif best_previous is not None and (best_iou >= self.args.smooth_iou or self.should_merge_detections(best_previous, detection)):
                copy["box"] = self.interpolate_box(best_previous["box"], detection["box"])
            if not feedback_only:
                real_detection_seen = True
            self.stabilize_detection_metadata(copy, best_previous, feedback_only)
            if self.args.blur_enabled:
                self.attach_blur_source(copy, frame, best_previous, use_previous=feedback_only)
            smoothed.append(copy)
        if not smoothed:
            self.previous_smoothed_detections = []
            return []
        self.previous_smoothed_detections = [dict(detection) for detection in smoothed]
        self.last_positive_time = now
        if real_detection_seen:
            self.last_positive_screen_hash = screen_hash
        return smoothed

    def hash_region_array(self, region):
        if region is None or region.size == 0:
            return None
        sample = cv2.resize(region, (8, 8), interpolation=cv2.INTER_AREA)
        return sample.tobytes()

    def extract_region(self, frame, box, inset=0):
        if frame is None:
            return None
        x, y, w, h = [int(value) for value in box]
        inset = max(0, int(inset))
        x1 = max(0, x + inset)
        y1 = max(0, y + inset)
        x2 = min(frame.shape[1], x + w - inset)
        y2 = min(frame.shape[0], y + h - inset)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def region_hash(self, frame, box, inset=0):
        return self.hash_region_array(self.extract_region(frame, box, inset=inset))

    def process_censor_region(self, region):
        if region is None or region.size == 0:
            return None
        if self.args.blur_style == "pixelate":
            scale = max(4, min(40, self.args.blur_strength))
            small_w = max(1, region.shape[1] // scale)
            small_h = max(1, region.shape[0] // scale)
            processed = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(processed, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)

        kernel = self.args.blur_strength
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(region, (kernel, kernel), 0)

    def attach_blur_source(self, detection, frame, previous=None, use_previous=False):
        if use_previous and previous is not None and previous.get("source_patch") is not None:
            detection["source_patch"] = previous.get("source_patch")
            detection["source_hash"] = previous.get("source_hash")
            detection["censor_hash"] = previous.get("censor_hash")
            detection["region_hash"] = previous.get("source_hash")
            return

        region = self.extract_region(frame, detection["box"])
        if region is None or region.size == 0:
            if previous is not None and previous.get("source_patch") is not None:
                self.attach_blur_source(detection, frame, previous, use_previous=True)
            return

        source_patch = region.copy()
        censored_patch = self.process_censor_region(source_patch)
        detection["source_patch"] = source_patch
        detection["source_hash"] = self.hash_region_array(source_patch)
        detection["censor_hash"] = self.hash_region_array(censored_patch)
        detection["region_hash"] = detection["source_hash"]

    def detection_looks_like_overlay_feedback(self, detection, frame, previous):
        if previous is None or frame is None or not self.args.blur_enabled:
            return False
        if previous.get("censor_hash") is None:
            return False
        if self.detection_iou(previous, detection) < 0.15 and not self.should_merge_detections(previous, detection):
            return False
        inset = max(self.args.line_width + 2, 4)
        current_hash = self.region_hash(frame, detection["box"], inset=inset)
        delta = self.region_delta(previous.get("censor_hash"), current_hash)
        return delta is not None and delta <= self.args.feedback_delta

    @staticmethod
    def screen_hash(frame):
        if frame is None:
            return None
        sample = cv2.resize(frame, (24, 14), interpolation=cv2.INTER_AREA)
        return sample.tobytes()

    @staticmethod
    def region_delta(first_hash, second_hash):
        if not first_hash or not second_hash or len(first_hash) != len(second_hash):
            return None
        first = np.frombuffer(first_hash, dtype=np.uint8).astype(np.int16)
        second = np.frombuffer(second_hash, dtype=np.uint8).astype(np.int16)
        return float(np.mean(np.abs(first - second)))

    def held_detection_is_still_valid(self, detection, frame):
        if not self.args.drop_stale_on_screen_change:
            return True
        if frame is None:
            return True
        if self.region_looks_like_censor_feedback(detection, frame):
            return True
        previous_hash = detection.get("source_hash") or detection.get("region_hash")
        current_hash = self.region_hash(frame, detection["box"])
        delta = self.region_delta(previous_hash, current_hash)
        if delta is None:
            return False
        if delta > self.args.stale_region_delta:
            return False
        detection["region_hash"] = current_hash
        return True

    def region_looks_like_censor_feedback(self, detection, frame):
        if frame is None or not self.args.blur_enabled or detection.get("censor_hash") is None:
            return False
        inset = max(self.args.line_width + 2, 4)
        current_hash = self.region_hash(frame, detection["box"], inset=inset)
        delta = self.region_delta(detection.get("censor_hash"), current_hash)
        return delta is not None and delta <= self.args.feedback_delta

    def attach_region_signatures(self, frame, detections):
        if not detections or not self.args.blur_enabled:
            return None
        signatures = []
        for detection in detections:
            signature = detection.get("source_hash") if self.args.smooth_overlay else self.region_hash(frame, detection["box"])
            detection["region_hash"] = signature
            signatures.append(signature)
        return tuple(signatures)

    def publish_detections(self, detections, frame=None, frame_signature=None):
        if self.result_queue.full():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        self.result_queue.put_nowait((time.time(), detections, frame, frame_signature))

    def update_stats(self, detections):
        self.stats["frames"] += 1
        self.stats["detections"] += len(detections)
        now = time.time()
        delta = now - self.stats["last_fps_time"]
        if delta >= 1.0:
            self.stats["fps"] = self.stats["frames"] / delta
            self.stats["frames"] = 0
            self.stats["last_fps_time"] = now

    def draw_loop(self):
        try:
            while True:
                (
                    self.last_detection_time,
                    self.latest_detections,
                    self.latest_frame,
                    self.latest_frame_signature,
                ) = self.result_queue.get_nowait()
        except queue.Empty:
            pass

        detections = self.latest_detections
        frame = self.latest_frame
        frame_signature = self.latest_frame_signature

        draw_signature = self.overlay_signature(detections, frame_signature)
        status_due = self.args.show_status and time.time() - self._last_status_draw >= 0.5
        if not self.args.smooth_overlay or draw_signature != self._last_draw_signature or status_due:
            self.draw_overlay(detections, frame)
            self._last_draw_signature = draw_signature
            if self.args.show_status:
                self._last_status_draw = time.time()
        if self.running.is_set():
            self.root.after(max(15, int(1000 / max(1, self.args.overlay_fps))), self.draw_loop)

    def overlay_signature(self, detections, frame_signature=None):
        if not detections:
            return (
                "empty",
                bool(self.args.show_status),
                bool(self.latest_protection_policy.get("blocked")),
                self.latest_demographics.get("underage_detected"),
            )
        detection_parts = []
        for detection in detections:
            x, y, w, h = detection["box"]
            detection_parts.append(
                (
                    detection.get("class"),
                    round(float(detection.get("score", 0)), 2),
                    int(x) // 3,
                    int(y) // 3,
                    int(w) // 3,
                    int(h) // 3,
                    detection.get("label_group"),
                    detection.get("rule_allowed", True),
                )
            )
        return (
            tuple(detection_parts),
            frame_signature if self.args.blur_enabled else None,
            self.args.show_boxes,
            self.args.show_labels,
            self.args.block_enabled,
            self.args.blur_enabled,
            self.args.privacy_on_detection,
            bool(self.latest_protection_policy.get("blocked")),
            bool(self.latest_demographics.get("underage_detected")),
        )

    def detection_outline_color(self, detection):
        if not detection.get("rule_allowed", True):
            return self.rule_skipped_outline_color
        if detection.get("label_group") != "exposed":
            return self.safe_outline_color
        return self.outline_color

    def draw_shape(self, x, y, x2, y2, *, fill="", outline="", width=1):
        if self.args.mask_shape == "ellipse":
            return self.canvas.create_oval(x, y, x2, y2, fill=fill, outline=outline, width=width)
        return self.canvas.create_rectangle(x, y, x2, y2, fill=fill, outline=outline, width=width)

    def blur_cache_key(self, detection):
        if not self.args.smooth_overlay:
            return None
        x, y, w, h = [int(value) for value in detection["box"]]
        return (
            x,
            y,
            w,
            h,
            detection.get("source_hash") or detection.get("region_hash"),
            self.args.blur_style,
            self.args.blur_strength,
            self.args.mask_shape,
        )

    def prune_blur_cache(self):
        if len(self._blur_cache) <= 24:
            return
        for key in list(self._blur_cache.keys())[:-16]:
            self._blur_cache.pop(key, None)

    def build_blurred_region(self, detection, frame, cache_key=None):
        if frame is None or Image is None or ImageTk is None:
            if detection.get("source_patch") is None or Image is None or ImageTk is None:
                return None
        if cache_key is not None and cache_key in self._blur_cache:
            return self._blur_cache[cache_key]

        region = detection.get("source_patch")
        if region is None:
            x, y, w, h = detection["box"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(frame.shape[1], int(x + w))
            y2 = min(frame.shape[0], int(y + h))
            if x2 <= x1 or y2 <= y1:
                return None
            region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None

        processed = self.process_censor_region(region)
        if processed is None:
            return None
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGBA")
        if self.args.mask_shape == "ellipse" and ImageDraw is not None:
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, image.size[0] - 1, image.size[1] - 1), fill=255)
            image.putalpha(mask)
        photo = ImageTk.PhotoImage(image)
        if cache_key is not None:
            self._blur_cache[cache_key] = photo
            self.prune_blur_cache()
        return photo

    def draw_overlay(self, detections, frame=None):
        self.canvas.delete("all")
        self._tk_images = []
        protection_blocked = bool(self.latest_protection_policy.get("blocked"))
        if not detections and not self.args.show_status and not protection_blocked:
            self.hide_overlay()
            return

        self.show_overlay()

        if protection_blocked or (detections and self.args.privacy_on_detection):
            self.canvas.create_rectangle(
                0,
                0,
                self.monitor["width"],
                self.monitor["height"],
                fill=self.block_color,
                outline="",
            )
            self.canvas.create_text(
                self.monitor["width"] // 2,
                self.monitor["height"] // 2,
                text=(
                    "SafeVision child-protection policy blocked this screen"
                    if protection_blocked
                    else "SafeVision blocked unsafe screen content"
                ),
                fill="white",
                font=("Segoe UI", 24, "bold"),
            )
            return

        for detection in detections:
            x, y, w, h = detection["box"]
            x2 = x + w
            y2 = y + h
            label = detection["class"]
            score = detection["score"]
            should_censor = bool(detection.get("censor", True))

            if self.args.blur_enabled and should_censor:
                image = self.build_blurred_region(detection, frame, self.blur_cache_key(detection))
                if image is not None:
                    self._tk_images.append(image)
                    self.canvas.create_image(x, y, image=image, anchor="nw")
            if self.args.block_enabled and should_censor:
                self.draw_shape(x, y, x2, y2, fill=self.block_color, outline="")
            if self.args.show_boxes:
                self.draw_shape(
                    x,
                    y,
                    x2,
                    y2,
                    outline=self.detection_outline_color(detection),
                    width=self.args.line_width,
                )
            if self.args.show_labels:
                merged_count = len(detection.get("merged_labels", []))
                suffix = f" +{merged_count - 1}" if merged_count > 1 else ""
                text = detection.get("display_label") or f"{label}{suffix} {score:.2f}"
                text_width = max(160, len(text) * 8)
                text_y = max(0, y - 22)
                self.canvas.create_rectangle(x, text_y, x + text_width, text_y + 20, fill=self.label_bg, outline="")
                self.canvas.create_text(
                    x + 5,
                    text_y + 10,
                    text=text,
                    fill=self.label_color,
                    anchor="w",
                    font=("Segoe UI", 9, "bold"),
                )

        if self.args.show_status:
            held_count = sum(1 for detection in detections if detection.get("held"))
            underage = self.latest_demographics.get("underage_detected")
            underage_status = "off" if underage is None else ("yes" if underage else "no")
            status = (
                f"SafeVision Screen Guard | mode={self.args.mode} | labels={self.args.label_filter} "
                f"| rules={'on' if self.args.respect_rules else 'off'} | detections={len(detections)} "
                f"| underage~={underage_status} | policy={self.latest_protection_policy.get('verdict', 'ALLOW')} "
                f"| held={held_count} | fps={self.stats['fps']:.1f}"
            )
            self.canvas.create_rectangle(8, 8, 980, 34, fill=self.label_bg, outline="")
            self.canvas.create_text(16, 21, text=status, fill="white", anchor="w", font=("Segoe UI", 10, "bold"))

    def run(self):
        print("SafeVision Screen Guard running.")
        print("No frames are recorded or saved.")
        print(
            f"Checks: {', '.join(self.enabled_detectors)} | estimated-underage threshold: "
            f"{self.args.underage_age:g} years"
        )
        print(
            "Render: "
            f"mode={self.args.mode}, boxes={self.args.show_boxes}, "
            f"blur={self.args.blur_enabled}, block={self.args.block_enabled}, "
            f"labels={self.args.show_labels}, label_filter={self.args.label_filter}, "
            f"respect_rules={self.args.respect_rules}, smooth={self.args.smooth_overlay}, "
            f"track_hold_ms={self.args.track_hold_ms}, merge_nearby={self.args.merge_nearby}, "
            f"feedback_delta={self.args.feedback_delta}, "
            f"feedback_safe_capture={self.args.feedback_safe_capture}"
        )
        print("Press Ctrl+C in this terminal to stop.")
        if not self.args.click_through:
            print("Press Escape on the overlay to close it.")

        worker = threading.Thread(target=self.capture_loop, daemon=True)
        worker.start()
        self.root.after(50, self.draw_loop)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.running.clear()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time SafeVision desktop screen guard overlay.")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor number from --list-monitors.")
    parser.add_argument("--list-monitors", action="store_true", help="Print available monitors and exit.")
    parser.add_argument("--mode", choices=["box", "blur", "block", "both", "privacy"], default="box",
                        help="Base overlay behavior. Fine-grained flags below can override this.")
    parser.add_argument("--fps", type=float, default=5.0, help="Detection FPS. Lower values use less CPU/GPU.")
    parser.add_argument("--overlay-fps", type=float, default=20.0, help="Overlay redraw FPS.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum detection confidence.")
    parser.add_argument("--hold-ms", type=int, default=650, help="Keep boxes visible this long after detection.")
    parser.add_argument("--providers", type=str, default=None, help="Comma-separated ONNX providers.")
    parser.add_argument("--rules", type=str, default=str(APP_DIR / "BlurException.rule"), help="BlurException.rule file path.")
    parser.add_argument("--detectors", type=str, default="nude,age,gender",
                        help="Checks to run: nsfw/nude, age, gender, demographics, or all.")
    parser.add_argument("--age-gender-model", type=str, default=str(default_age_gender_model_path()),
                        help="Path to the estimated-age/gender ONNX model.")
    parser.add_argument("--underage-age", type=float, default=None,
                        help="Estimated underage threshold. Defaults to BlurException.rule (18).")
    parser.add_argument("--age-review-margin", type=float, default=None,
                        help="Review band above the threshold. Defaults to BlurException.rule (3).")
    parser.add_argument("--min-face-size", type=int, default=32,
                        help="Minimum fallback face size in pixels.")
    parser.add_argument("--face-padding", type=float, default=0.18,
                        help="Padding around face crops as a fraction of face size.")
    parser.add_argument("--show-demographics", dest="show_demographics", action="store_true",
                        help="Draw non-censoring estimated-age/gender face boxes.")
    parser.add_argument("--hide-demographics", dest="show_demographics", action="store_false",
                        help="Run child protection without drawing demographic face boxes.")
    parser.set_defaults(show_demographics=False)
    parser.add_argument("--capture-backend", choices=["auto", "gdi", "mss"], default="auto",
                        help="Screen pixel capture backend. auto uses Windows GDI on Windows, mss elsewhere.")
    parser.add_argument("--smooth-overlay", dest="smooth_overlay", action="store_true",
                        help="Smooth boxes, merge duplicates, cache blur patches, and skip unchanged redraws.")
    parser.add_argument("--no-smooth-overlay", dest="smooth_overlay", action="store_false",
                        help="Disable overlay smoothing and redraw every overlay tick.")
    parser.set_defaults(smooth_overlay=True)
    parser.add_argument("--smooth-iou", type=float, default=0.45,
                        help="Overlap threshold used to merge/track boxes when smoothing is enabled.")
    parser.add_argument("--smooth-alpha", type=float, default=0.65,
                        help="Smoothing follow speed. Higher follows movement faster; lower is steadier.")
    parser.add_argument("--track-hold-ms", type=int, default=1600,
                        help="Keep the last stable detection visible through short missed detections.")
    parser.add_argument("--stable-score-alpha", type=float, default=0.2,
                        help="How quickly label confidence changes on a tracked box. Lower is steadier.")
    parser.add_argument("--merge-nearby", dest="merge_nearby", action="store_true",
                        help="Merge nearby detections into one continuous protected region.")
    parser.add_argument("--no-merge-nearby", dest="merge_nearby", action="store_false",
                        help="Only merge detections that overlap.")
    parser.set_defaults(merge_nearby=True)
    parser.add_argument("--merge-distance", type=int, default=260,
                        help="Pixel distance used by --merge-nearby.")
    parser.add_argument("--merge-overlap", type=float, default=0.35,
                        help="Intersection-over-smaller-box threshold used to merge nested boxes.")
    parser.add_argument("--feedback-delta", type=float, default=18.0,
                        help="Low-res pixel delta used to reject detections caused by the overlay itself.")
    parser.add_argument("--feedback-safe-capture", dest="feedback_safe_capture", action="store_true",
                        help="Briefly hide the overlay during screenshots so detection cannot see its own blur/boxes.")
    parser.add_argument("--no-feedback-safe-capture", dest="feedback_safe_capture", action="store_false",
                        help="Do not hide the overlay while capturing screenshots.")
    parser.set_defaults(feedback_safe_capture=False)
    parser.add_argument("--capture-hide-ms", type=int, default=20,
                        help="Milliseconds to hide the overlay before each feedback-safe capture.")
    parser.add_argument("--drop-stale-on-screen-change", dest="drop_stale_on_screen_change", action="store_true",
                        help="Remove held boxes immediately when the screen under them changes.")
    parser.add_argument("--keep-stale-regions", dest="drop_stale_on_screen_change", action="store_false",
                        help="Keep held boxes for track-hold-ms even when the screen under them changes.")
    parser.set_defaults(drop_stale_on_screen_change=True)
    parser.add_argument("--stale-region-delta", type=float, default=10.0,
                        help="Pixel-change threshold used to release stale held boxes.")
    parser.add_argument("--screen-change-delta", type=float, default=28.0,
                        help="Whole-screen change threshold used to release stale held boxes.")
    parser.add_argument("--exclude-overlay-capture", dest="exclude_overlay_capture", action="store_true",
                        help="Ask Windows to exclude the overlay from screen captures.")
    parser.add_argument("--allow-overlay-capture", dest="exclude_overlay_capture", action="store_false",
                        help="Allow screen capture tools to see the overlay.")
    parser.set_defaults(exclude_overlay_capture=True)
    parser.add_argument("--label-filter", choices=["exposed", "body", "all"], default="exposed",
                        help="Detection labels to protect: exposed only, all body labels, or every label including faces.")
    parser.add_argument("--exposed-only", dest="label_filter", action="store_const", const="exposed",
                        help="Protect only labels containing EXPOSED.")
    parser.add_argument("--body-labels", dest="label_filter", action="store_const", const="body",
                        help="Protect exposed and covered body labels, but not face labels.")
    parser.add_argument("--all-labels", dest="label_filter", action="store_const", const="all",
                        help="Protect every detected label, including covered and face labels.")
    parser.add_argument("--respect-rules", dest="respect_rules", action="store_true",
                        help="Use BlurException.rule to skip labels marked false.")
    parser.add_argument("--ignore-rules", dest="respect_rules", action="store_false",
                        help="Show/protect detections even when BlurException.rule marks the label false.")
    parser.set_defaults(respect_rules=True)
    parser.add_argument("--show-boxes", dest="show_boxes", action="store_true", default=None,
                        help="Draw outline boxes around detections.")
    parser.add_argument("--no-boxes", dest="show_boxes", action="store_false",
                        help="Do not draw outline boxes.")
    parser.add_argument("--labels", "--show-labels", dest="show_labels", action="store_true",
                        help="Show label/confidence text on the overlay.")
    parser.add_argument("--no-labels", dest="show_labels", action="store_false",
                        help="Hide label/confidence text on the overlay.")
    parser.set_defaults(show_labels=False)
    parser.add_argument("--show-status", action="store_true", help="Show a small status HUD.")
    parser.add_argument("--block-enabled", dest="block_enabled", action="store_true", default=None,
                        help="Fill detected regions with the block color.")
    parser.add_argument("--no-block", dest="block_enabled", action="store_false",
                        help="Do not fill detected regions with the block color.")
    parser.add_argument("--blur", dest="blur_enabled", action="store_true", default=None,
                        help="Draw blurred live screen patches over detected regions.")
    parser.add_argument("--no-blur", dest="blur_enabled", action="store_false",
                        help="Do not draw blurred live screen patches.")
    parser.add_argument("--privacy-on-detection", dest="privacy_on_detection", action="store_true", default=None,
                        help="Cover the whole monitor whenever any matching detection is found.")
    parser.add_argument("--no-privacy", dest="privacy_on_detection", action="store_false",
                        help="Disable whole-monitor privacy cover.")
    parser.add_argument("--blur-style", choices=["gaussian", "pixelate"], default="gaussian",
                        help="How localized blur patches are rendered.")
    parser.add_argument("--blur-strength", type=int, default=45,
                        help="Gaussian kernel size or pixelation scale. Higher values censor more.")
    parser.add_argument("--mask-shape", choices=["rectangle", "ellipse"], default="rectangle",
                        help="Shape used for boxes, solid blocks, and blur patches.")
    parser.add_argument("--block-color", default=DEFAULT_BLOCK_COLOR, help="BGR block color, e.g. 0,0,0 or 0,0,255.")
    parser.add_argument("--outline-color", default="#ff3333", help="Overlay outline color.")
    parser.add_argument("--safe-outline-color", default="#3388ff", help="Outline color for covered/face/non-exposed labels.")
    parser.add_argument("--rule-skipped-outline-color", default="#aaaaaa",
                        help="Outline color when --ignore-rules displays a rule-skipped label.")
    parser.add_argument("--label-bg", default=DEFAULT_LABEL_BG, help="Label background color.")
    parser.add_argument("--label-color", default="white", help="Label text color.")
    parser.add_argument("--line-width", type=int, default=4, help="Overlay box line width.")
    parser.add_argument("--box-padding", type=int, default=0, help="Extra pixels added around detection regions.")
    parser.add_argument("--min-box-area", type=int, default=0, help="Ignore detections smaller than this pixel area.")
    parser.add_argument("--click-through", dest="click_through", action="store_true",
                        help="Let mouse clicks pass through the overlay.")
    parser.add_argument("--no-click-through", dest="click_through", action="store_false",
                        help="Overlay captures mouse/keyboard; Escape closes it.")
    parser.set_defaults(click_through=True)
    return normalize_args(parser.parse_args())


def main():
    args = parse_args()
    if args.list_monitors:
        return list_monitors() or 0

    try:
        guard = ScreenGuard(args)
        guard.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Failed to start screen guard: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
