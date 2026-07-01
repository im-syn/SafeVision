#!/usr/bin/env python3
"""
SafeVision CLI

A user-friendly command console for SafeVision settings, rule profiles, media
scanning, and launching image/video/API/GUI workflows.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from safevision_utils import (
    ALL_CENSOR_LABELS,
    default_blur_rules,
    ensure_blur_exception_rules,
    load_blur_exception_rules,
    parse_provider_list,
    select_onnx_providers,
    write_blur_exception_rules,
)


APP_DIR = Path(__file__).resolve().parent
SETTINGS_DIR = APP_DIR / "settings"
CONFIG_PATH = SETTINGS_DIR / "configs.json"
DEFAULT_RULE_PATH = APP_DIR / "BlurException.rule"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}

SCRIPT_TARGETS = {
    "image": "main.py",
    "video": "video.py",
    "gui": "SafeVisionGUI.py",
    "api": "safevision_api.py",
    "web": "safevision_api.py",
    "live": "live.py",
    "streamer": "live_streamer.py",
    "screen": "safeVisionScreenGuard.py",
}


COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def color(text, name=None):
    if COLORAMA_AVAILABLE:
        mapping = {
            "red": Fore.RED,
            "green": Fore.GREEN,
            "yellow": Fore.YELLOW,
            "blue": Fore.BLUE,
            "magenta": Fore.MAGENTA,
            "cyan": Fore.CYAN,
            "bold": Style.BRIGHT,
        }
        return f"{mapping.get(name, '')}{text}{Style.RESET_ALL}"
    if name in COLORS:
        return f"{COLORS[name]}{text}{COLORS['reset']}"
    return text


def print_header(title):
    print(color(f"\n{title}", "bold"))
    print(color("=" * len(title), "cyan"))


def print_ok(message):
    print(color(f"[OK] {message}", "green"))


def print_warn(message):
    print(color(f"[WARN] {message}", "yellow"))


def print_error(message):
    print(color(f"[ERROR] {message}", "red"))


def default_profiles():
    return {
        "default": default_blur_rules(),
        "strict": default_blur_rules(),
        "faces_allowed": {
            **default_blur_rules(),
            "FACE_FEMALE": False,
            "FACE_MALE": False,
        },
        "covered_allowed": {
            **default_blur_rules(),
            "FEMALE_GENITALIA_COVERED": False,
            "BELLY_COVERED": False,
            "FEET_COVERED": False,
            "ARMPITS_COVERED": False,
            "ANUS_COVERED": False,
            "FEMALE_BREAST_COVERED": False,
            "BUTTOCKS_COVERED": False,
        },
    }


def default_config():
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "active_rule_profile": "default",
        "paths": {
            "input": "input",
            "output": "output",
            "video_output": "video_output",
            "blur": "Blur",
            "process": "Prosses",
            "logs": "Logs",
            "models": "Models",
        },
        "processing": {
            "providers": "",
            "detectors": "nude",
            "object_model": "Models/safety_objects.onnx",
            "object_labels": "Models/safety_objects.labels.json",
            "object_threshold": 0.25,
            "codec": "mp4v",
            "mask_shape": "rectangle",
            "mask_color": "0,0,0",
            "blur_strength": 23,
            "blur_sigma": 0.0,
            "enhanced_blur": False,
            "with_audio": False,
            "delete_frames": True,
            "rule": "0/0",
            "full_blur_rule": "",
            "save_report": False,
            "report_formats": "json,csv",
            "export_markers": "",
            "marker_gap": 1.0,
        },
        "screen_guard": {
            "monitor": 1,
            "mode": "box",
            "fps": 5.0,
            "overlay_fps": 20.0,
            "threshold": 0.35,
            "hold_ms": 650,
            "providers": "",
            "rules": "BlurException.rule",
            "capture_backend": "auto",
            "smooth_overlay": True,
            "smooth_iou": 0.45,
            "smooth_alpha": 0.65,
            "stable_score_alpha": 0.2,
            "track_hold_ms": 1600,
            "merge_nearby": True,
            "merge_distance": 260,
            "merge_overlap": 0.35,
            "feedback_delta": 18.0,
            "feedback_safe_capture": False,
            "capture_hide_ms": 20,
            "drop_stale_on_screen_change": True,
            "stale_region_delta": 10.0,
            "screen_change_delta": 28.0,
            "exclude_overlay_capture": True,
            "label_filter": "exposed",
            "respect_rules": True,
            "show_boxes": True,
            "show_labels": False,
            "show_status": False,
            "block_enabled": False,
            "blur_enabled": False,
            "privacy_on_detection": False,
            "blur_style": "gaussian",
            "blur_strength": 45,
            "mask_shape": "rectangle",
            "block_color": "0,0,0",
            "outline_color": "#ff3333",
            "safe_outline_color": "#3388ff",
            "rule_skipped_outline_color": "#aaaaaa",
            "label_bg": "#111111",
            "label_color": "white",
            "line_width": 4,
            "box_padding": 0,
            "min_box_area": 0,
            "click_through": True,
        },
        "rule_profiles": default_profiles(),
    }


def deep_merge(base, incoming):
    result = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        config = deep_merge(default_config(), loaded)
    else:
        config = default_config()
    save_config(config)
    return config


def save_config(config):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


def resolve_app_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = APP_DIR / path
    return path


def script_path(name):
    return APP_DIR / SCRIPT_TARGETS[name]


def script_exists(name):
    return script_path(name).exists()


def python_command():
    return sys.executable or "python"


def bool_text(value):
    return color("true", "green") if value else color("false", "yellow")


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def set_nested(config, dotted_key, value):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]

    old_value = target.get(parts[-1])
    if isinstance(old_value, bool):
        value = parse_bool(value)
    elif isinstance(old_value, int):
        value = int(value)
    elif isinstance(old_value, float):
        value = float(value)
    target[parts[-1]] = value


def get_nested(config, dotted_key):
    target = config
    for part in dotted_key.split("."):
        if not isinstance(target, dict) or part not in target:
            raise KeyError(dotted_key)
        target = target[part]
    return target


def rule_profile(config, name=None):
    name = name or config.get("active_rule_profile", "default")
    profiles = config.setdefault("rule_profiles", default_profiles())
    if name not in profiles:
        raise KeyError(f"Rule profile does not exist: {name}")
    return name, profiles[name]


def normalize_rules(rules):
    normalized = default_blur_rules()
    for label, value in rules.items():
        normalized[label] = bool(value)
    return normalized


def activate_rule_profile(config, name):
    name, rules = rule_profile(config, name)
    rules = normalize_rules(rules)
    config["rule_profiles"][name] = rules
    config["active_rule_profile"] = name
    write_blur_exception_rules(DEFAULT_RULE_PATH, rules=rules)
    save_config(config)
    print_ok(f"Activated rule profile '{name}' and wrote {DEFAULT_RULE_PATH.name}")


def run_subprocess(command, cwd=APP_DIR):
    print(color("Command:", "cyan"), " ".join(str(part) for part in command))
    return subprocess.call(command, cwd=str(cwd))


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause(message="Press Enter to continue..."):
    try:
        input(color(f"\n{message}", "cyan"))
    except EOFError:
        pass


def prompt(message, default=None, required=False):
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        try:
            value = input(f"{message}{suffix}: ").strip()
        except EOFError:
            return default or ""
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print_warn("A value is required.")


def prompt_bool(message, default=False):
    default_text = "Y/n" if default else "y/N"
    value = prompt(f"{message} ({default_text})", default="")
    if not value:
        return default
    return parse_bool(value)


def selected_row(text):
    return color(f"> {text}", "cyan")


def unselected_row(text):
    return f"  {text}"


def read_menu_key():
    if os.name == "nt" and sys.stdin.isatty():
        import msvcrt

        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            arrow = msvcrt.getwch()
            if arrow == "H":
                return "up"
            if arrow == "P":
                return "down"
            return ""
        if key == "\r":
            return "enter"
        if key == "\x1b":
            return "escape"
        if key.isdigit():
            return key
        return key.lower()
    return input(color("Select number: ", "cyan")).strip()


def menu_select(title, options, *, subtitle=None, allow_back=True):
    """
    Show an interactive row selector.

    options: list of (label, value) tuples.
    """
    rows = list(options)
    if allow_back:
        rows.append(("Back", "__back__"))
    index = 0

    if not sys.stdin.isatty():
        print_header(title)
        for row_index, (label, _) in enumerate(rows, start=1):
            print(f"{row_index}. {label}")
        return rows[0][1] if rows else "__back__"

    while True:
        clear_screen()
        print_header(title)
        if subtitle:
            print(subtitle)
        print(color("Use Up/Down + Enter, or press a number.", "yellow"))
        print()
        for row_index, (label, _) in enumerate(rows, start=1):
            text = f"{row_index}. {label}"
            print(selected_row(text) if row_index - 1 == index else unselected_row(text))

        key = read_menu_key()
        if key == "up":
            index = (index - 1) % len(rows)
        elif key == "down":
            index = (index + 1) % len(rows)
        elif key == "enter":
            return rows[index][1]
        elif key == "escape":
            return "__back__"
        elif key.isdigit():
            selected = int(key)
            if 1 <= selected <= len(rows):
                return rows[selected - 1][1]


def choose_from_values(title, values, default=None, allow_back=True):
    options = [(str(value), value) for value in values]
    if default is not None:
        subtitle = f"Current: {default}"
    else:
        subtitle = None
    return menu_select(title, options, subtitle=subtitle, allow_back=allow_back)


def media_files_for_menu(media_type="all"):
    config = load_config()
    input_folder = resolve_app_path(config["paths"]["input"])
    return list(iter_media(input_folder, recursive=True, media_type=media_type))


def choose_media_file(media_type="all"):
    files = media_files_for_menu(media_type)
    options = []
    for path in files[:100]:
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            label = f"{path.relative_to(APP_DIR)} ({size_mb:.1f} MB)"
        except ValueError:
            label = str(path)
        options.append((label, str(path)))
    options.append(("Type a custom path", "__custom__"))

    selected = menu_select("Select Media File", options, allow_back=True)
    if selected == "__custom__":
        return prompt("Input file path", required=True)
    if selected == "__back__":
        return None
    return selected


def command_init(args):
    config = load_config()
    for relative in config["paths"].values():
        resolve_app_path(relative).mkdir(parents=True, exist_ok=True)
    activate_rule_profile(config, config.get("active_rule_profile", "default"))
    print_ok(f"Settings ready at {CONFIG_PATH}")


def command_status(args):
    config = load_config()
    print_header("SafeVision Status")
    print(f"App folder: {APP_DIR}")
    print(f"Config: {CONFIG_PATH} ({'exists' if CONFIG_PATH.exists() else 'missing'})")
    print(f"Rule file: {DEFAULT_RULE_PATH} ({'exists' if DEFAULT_RULE_PATH.exists() else 'missing'})")
    print(f"Active rule profile: {config.get('active_rule_profile')}")

    print_header("Scripts")
    for name in ["image", "video", "gui", "api", "live", "streamer", "screen"]:
        path = script_path(name)
        print(f"{name:8} {bool_text(path.exists())}  {path.name}")

    print_header("Folders")
    for key, value in config["paths"].items():
        path = resolve_app_path(value)
        print(f"{key:12} {bool_text(path.exists())}  {path}")

    print_header("ONNX Providers")
    try:
        selected = select_onnx_providers(parse_provider_list(config["processing"].get("providers", "")))
        print(", ".join(selected))
    except Exception as exc:
        print_warn(f"Could not inspect ONNX Runtime providers: {exc}")

    screen_guard = config.get("screen_guard", {})
    print_header("Screen Guard")
    for key in [
        "mode",
        "label_filter",
        "respect_rules",
        "capture_backend",
        "smooth_overlay",
        "smooth_iou",
        "smooth_alpha",
        "stable_score_alpha",
        "track_hold_ms",
        "merge_nearby",
        "merge_distance",
        "merge_overlap",
        "feedback_delta",
        "feedback_safe_capture",
        "capture_hide_ms",
        "drop_stale_on_screen_change",
        "stale_region_delta",
        "screen_change_delta",
        "exclude_overlay_capture",
        "show_boxes",
        "show_labels",
        "block_enabled",
        "blur_enabled",
        "privacy_on_detection",
        "threshold",
    ]:
        print(f"{key:22} {screen_guard.get(key)}")


def command_folders(args):
    config = load_config()
    print_header("SafeVision Folders")
    for key, value in config["paths"].items():
        path = resolve_app_path(value)
        if args.create:
            path.mkdir(parents=True, exist_ok=True)
        size = ""
        if path.exists():
            file_count = sum(1 for item in path.rglob("*") if item.is_file()) if args.recursive else sum(1 for item in path.iterdir() if item.is_file())
            size = f"files={file_count}"
        print(f"{key:12} {bool_text(path.exists())}  {path} {size}")


def iter_media(folder, recursive=False, media_type="all"):
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        is_image = suffix in IMAGE_EXTENSIONS
        is_video = suffix in VIDEO_EXTENSIONS
        if media_type == "image" and not is_image:
            continue
        if media_type == "video" and not is_video:
            continue
        if media_type == "all" and not (is_image or is_video):
            continue
        yield path


def command_scan(args):
    config = load_config()
    folder = Path(args.folder or resolve_app_path(config["paths"]["input"]))
    if not folder.exists():
        print_error(f"Folder does not exist: {folder}")
        return 2

    files = list(iter_media(folder, recursive=args.recursive, media_type=args.type))
    print_header(f"Media Scan: {folder}")
    iterator = tqdm(files, desc="Scanning", unit="file") if tqdm and args.progress else files
    images = 0
    videos = 0
    for path in iterator:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            images += 1
            kind = color("image", "green")
        else:
            videos += 1
            kind = color("video", "magenta")
        if not args.names_only:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{kind:14} {size_mb:8.2f} MB  {path}")
        else:
            print(path)

    print_ok(f"Found {images} images and {videos} videos")


def command_settings(args):
    config = load_config()
    if args.settings_action == "show":
        print(json.dumps(config, indent=2))
    elif args.settings_action == "get":
        try:
            print(get_nested(config, args.key))
        except KeyError:
            print_error(f"Unknown setting: {args.key}")
            return 2
    elif args.settings_action == "set":
        set_nested(config, args.key, args.value)
        save_config(config)
        print_ok(f"Set {args.key} = {args.value}")
    elif args.settings_action == "reset":
        config = default_config()
        save_config(config)
        print_ok(f"Reset settings at {CONFIG_PATH}")


def command_rules(args):
    config = load_config()
    profiles = config.setdefault("rule_profiles", default_profiles())

    if args.rules_action == "init":
        ensure_blur_exception_rules(DEFAULT_RULE_PATH, labels=ALL_CENSOR_LABELS)
        activate_rule_profile(config, config.get("active_rule_profile", "default"))
        return

    if args.rules_action == "list":
        print_header("Rule Profiles")
        active = config.get("active_rule_profile", "default")
        for name in sorted(profiles):
            marker = "*" if name == active else " "
            normalized = normalize_rules(profiles[name])
            enabled = sum(1 for value in normalized.values() if value)
            disabled = len(ALL_CENSOR_LABELS) - enabled
            print(f"{marker} {name:18} blur={enabled:2} skip={disabled:2}")
        return

    if args.rules_action == "show":
        name, rules = rule_profile(config, args.profile)
        rules = normalize_rules(rules)
        print_header(f"Rule Profile: {name}")
        for label in ALL_CENSOR_LABELS:
            print(f"{label:30} {'blur' if rules.get(label, True) else 'skip'}")
        return

    if args.rules_action == "use":
        activate_rule_profile(config, args.profile)
        return

    if args.rules_action == "create":
        if args.profile in profiles and not args.force:
            print_error(f"Profile already exists: {args.profile}. Use --force to overwrite.")
            return 2
        if args.from_file:
            profiles[args.profile] = load_blur_exception_rules(args.from_file, labels=ALL_CENSOR_LABELS)
        elif args.base:
            _, base_rules = rule_profile(config, args.base)
            profiles[args.profile] = dict(base_rules)
        else:
            profiles[args.profile] = default_blur_rules()
        save_config(config)
        print_ok(f"Created rule profile '{args.profile}'")
        return

    if args.rules_action == "set":
        name, rules = rule_profile(config, args.profile)
        if args.label not in ALL_CENSOR_LABELS:
            print_error(f"Unknown label: {args.label}")
            return 2
        rules[args.label] = parse_bool(args.value)
        profiles[name] = normalize_rules(rules)
        save_config(config)
        if name == config.get("active_rule_profile"):
            write_blur_exception_rules(DEFAULT_RULE_PATH, rules=profiles[name])
        print_ok(f"{name}: {args.label} = {'true' if rules[args.label] else 'false'}")
        return

    if args.rules_action == "delete":
        if args.profile == "default":
            print_error("The default profile cannot be deleted.")
            return 2
        if args.profile not in profiles:
            print_error(f"Profile does not exist: {args.profile}")
            return 2
        del profiles[args.profile]
        if config.get("active_rule_profile") == args.profile:
            config["active_rule_profile"] = "default"
            activate_rule_profile(config, "default")
        save_config(config)
        print_ok(f"Deleted rule profile '{args.profile}'")
        return

    if args.rules_action == "export":
        name, rules = rule_profile(config, args.profile)
        write_blur_exception_rules(args.output, rules=normalize_rules(rules))
        print_ok(f"Exported '{name}' to {args.output}")
        return


def append_common_processing_options(command, config, args):
    processing = config["processing"]
    providers = args.providers if getattr(args, "providers", None) is not None else processing.get("providers", "")
    if providers:
        command.extend(["--providers", providers])
    detectors = getattr(args, "detectors", None)
    if detectors is None:
        detectors = processing.get("detectors", "nude")
    if detectors:
        command.extend(["--detectors", str(detectors)])
    object_model = getattr(args, "object_model", None) or processing.get("object_model")
    if object_model:
        command.extend(["--object-model", str(object_model)])
    object_labels = getattr(args, "object_labels", None) or processing.get("object_labels")
    if object_labels:
        command.extend(["--object-labels", str(object_labels)])
    object_threshold = getattr(args, "object_threshold", None)
    if object_threshold is None:
        object_threshold = processing.get("object_threshold")
    if object_threshold not in (None, ""):
        command.extend(["--object-threshold", str(object_threshold)])
    return command


def clean_extra_args(extra):
    extra = list(extra or [])
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def screen_guard_args_from_config(config):
    settings = config.get("screen_guard", {})
    args = []
    value_options = [
        ("monitor", "--monitor"),
        ("mode", "--mode"),
        ("fps", "--fps"),
        ("overlay_fps", "--overlay-fps"),
        ("threshold", "--threshold"),
        ("hold_ms", "--hold-ms"),
        ("rules", "--rules"),
        ("capture_backend", "--capture-backend"),
        ("smooth_iou", "--smooth-iou"),
        ("smooth_alpha", "--smooth-alpha"),
        ("stable_score_alpha", "--stable-score-alpha"),
        ("track_hold_ms", "--track-hold-ms"),
        ("merge_distance", "--merge-distance"),
        ("merge_overlap", "--merge-overlap"),
        ("feedback_delta", "--feedback-delta"),
        ("capture_hide_ms", "--capture-hide-ms"),
        ("stale_region_delta", "--stale-region-delta"),
        ("screen_change_delta", "--screen-change-delta"),
        ("label_filter", "--label-filter"),
        ("blur_style", "--blur-style"),
        ("blur_strength", "--blur-strength"),
        ("mask_shape", "--mask-shape"),
        ("block_color", "--block-color"),
        ("outline_color", "--outline-color"),
        ("safe_outline_color", "--safe-outline-color"),
        ("rule_skipped_outline_color", "--rule-skipped-outline-color"),
        ("label_bg", "--label-bg"),
        ("label_color", "--label-color"),
        ("line_width", "--line-width"),
        ("box_padding", "--box-padding"),
        ("min_box_area", "--min-box-area"),
    ]
    for key, flag in value_options:
        value = settings.get(key)
        if value not in (None, ""):
            args.extend([flag, str(value)])

    providers = settings.get("providers") or config.get("processing", {}).get("providers", "")
    if providers:
        args.extend(["--providers", str(providers)])

    mode = settings.get("mode", "box")
    mode_defaults = {
        "show_boxes": mode in {"box", "both", "block"},
        "block_enabled": mode in {"block", "both"},
        "blur_enabled": mode == "blur",
        "privacy_on_detection": mode == "privacy",
    }
    render_bool_options = [
        ("show_boxes", "--show-boxes", "--no-boxes"),
        ("block_enabled", "--block-enabled", "--no-block"),
        ("blur_enabled", "--blur", "--no-blur"),
        ("privacy_on_detection", "--privacy-on-detection", "--no-privacy"),
    ]
    for key, enabled_flag, disabled_flag in render_bool_options:
        if key in settings and bool(settings.get(key)) != mode_defaults[key]:
            args.append(enabled_flag if settings.get(key) else disabled_flag)

    if settings.get("respect_rules") is False:
        args.append("--ignore-rules")
    if settings.get("smooth_overlay") is False:
        args.append("--no-smooth-overlay")
    if settings.get("merge_nearby") is False:
        args.append("--no-merge-nearby")
    if settings.get("feedback_safe_capture"):
        args.append("--feedback-safe-capture")
    if settings.get("drop_stale_on_screen_change") is False:
        args.append("--keep-stale-regions")
    if settings.get("exclude_overlay_capture") is False:
        args.append("--allow-overlay-capture")
    if settings.get("show_labels"):
        args.append("--show-labels")
    if settings.get("click_through") is False:
        args.append("--no-click-through")

    if settings.get("show_status"):
        args.append("--show-status")

    return args


def command_process(args):
    config = load_config()
    activate_rule_profile(config, config.get("active_rule_profile", "default"))

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input does not exist: {input_path}")
        return 2

    suffix = input_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        target = "image"
    elif suffix in VIDEO_EXTENSIONS:
        target = "video"
    else:
        print_error(f"Unsupported input type: {input_path.suffix}")
        return 2

    if not script_exists(target):
        print_error(f"Required script is missing: {script_path(target)}")
        return 2

    if target == "image":
        processing = config["processing"]
        command = [python_command(), str(script_path("image")), "-i", str(input_path)]
        if args.output:
            command.extend(["-o", args.output])
        if args.blur:
            command.append("-b")
        if args.full_blur_rule:
            command.extend(["-fbr", args.full_blur_rule])
        if getattr(args, "color", False):
            command.append("--color")
        mask_color = getattr(args, "mask_color", None) or processing.get("mask_color")
        if mask_color:
            command.extend(["--mask-color", mask_color])
        mask_shape = getattr(args, "mask_shape", None) or processing.get("mask_shape")
        if mask_shape:
            command.extend(["--mask-shape", mask_shape])
        blur_strength = getattr(args, "blur_strength", None)
        if blur_strength is None:
            blur_strength = processing.get("blur_strength")
        if blur_strength:
            command.extend(["--blur-strength", str(blur_strength)])
        blur_sigma = getattr(args, "blur_sigma", None)
        if blur_sigma is None:
            blur_sigma = processing.get("blur_sigma")
        if blur_sigma:
            command.extend(["--blur-sigma", str(blur_sigma)])
        append_common_processing_options(command, config, args)
    else:
        processing = config["processing"]
        command = [python_command(), str(script_path("video")), "-i", str(input_path), "-t", "video"]
        video_output = args.output_dir or processing.get("video_output") or config["paths"].get("video_output")
        if video_output:
            command.extend(["-vo", video_output])
        if getattr(args, "analyze_only", False):
            command.append("--analyze-only")
        if args.boxes:
            command.append("-b")
        if args.blur:
            command.append("--blur")
        if args.with_audio or processing.get("with_audio"):
            command.append("-a")
        codec = args.codec or processing.get("codec")
        if codec:
            command.extend(["-c", codec])
        if args.delete_frames or processing.get("delete_frames"):
            command.append("-df")
        if args.enhanced_blur or processing.get("enhanced_blur"):
            command.append("--enhanced-blur")
        if args.color:
            command.append("--color")
        mask_color = args.mask_color or processing.get("mask_color")
        if mask_color:
            command.extend(["--mask-color", mask_color])
        mask_shape = args.mask_shape or processing.get("mask_shape")
        if mask_shape:
            command.extend(["--mask-shape", mask_shape])
        blur_strength = getattr(args, "blur_strength", None)
        if blur_strength is None:
            blur_strength = processing.get("blur_strength")
        if blur_strength:
            command.extend(["--blur-strength", str(blur_strength)])
        blur_sigma = getattr(args, "blur_sigma", None)
        if blur_sigma is None:
            blur_sigma = processing.get("blur_sigma")
        if blur_sigma:
            command.extend(["--blur-sigma", str(blur_sigma)])
        rule = args.rule or processing.get("rule")
        if rule and rule != "0/0":
            command.extend(["-r", rule])
        fbr = args.full_blur_rule or processing.get("full_blur_rule")
        if fbr:
            command.extend(["-fbr", fbr])
        if getattr(args, "save_report", False) or processing.get("save_report"):
            command.append("--save-report")
        report_formats = getattr(args, "report_formats", None) or processing.get("report_formats")
        if report_formats:
            command.extend(["--report-formats", str(report_formats)])
        export_markers = getattr(args, "export_markers", None)
        if export_markers is None:
            export_markers = processing.get("export_markers")
        if export_markers:
            command.extend(["--export-markers", str(export_markers)])
        marker_gap = getattr(args, "marker_gap", None)
        if marker_gap is None:
            marker_gap = processing.get("marker_gap")
        if marker_gap:
            command.extend(["--marker-gap", str(marker_gap)])
        if args.ffmpeg_path:
            command.extend(["--ffmpeg-path", args.ffmpeg_path])
        append_common_processing_options(command, config, args)

    return run_subprocess(command)


def command_launch(args):
    name = args.target
    if not script_exists(name):
        print_error(f"{name} script is missing: {script_path(name)}")
        return 2
    command = [python_command(), str(script_path(name))]
    extra = clean_extra_args(args.extra)
    if name == "screen":
        config = load_config()
        activate_rule_profile(config, config.get("active_rule_profile", "default"))
        command.extend(screen_guard_args_from_config(config))
    if extra:
        command.extend(extra)
    return run_subprocess(command)


def command_screen(args):
    return command_launch(argparse.Namespace(target="screen", extra=args.extra))


def command_providers(args):
    config = load_config()
    requested = parse_provider_list(args.providers or config["processing"].get("providers", ""))
    try:
        providers = select_onnx_providers(requested)
        print_header("Selected ONNX Providers")
        for provider in providers:
            print(provider)
    except Exception as exc:
        print_error(f"Could not inspect providers: {exc}")
        return 2


def interactive_scan():
    while True:
        choice = menu_select(
            "Scan Media",
            [
                ("Scan all media in input folder", "all"),
                ("Scan images only", "image"),
                ("Scan videos only", "video"),
                ("Scan a custom folder", "custom"),
            ],
        )
        if choice == "__back__":
            return

        config = load_config()
        folder = None
        media_type = choice
        if choice == "custom":
            folder = prompt("Folder to scan", default=str(resolve_app_path(config["paths"]["input"])))
            media_type = choose_from_values("Media Type", ["all", "image", "video"], default="all", allow_back=False)
        recursive = prompt_bool("Scan recursively", default=True)
        clear_screen()
        command_scan(argparse.Namespace(folder=folder, recursive=recursive, type=media_type, names_only=False, progress=True))
        pause()


def interactive_process():
    selected = menu_select(
        "Process Media",
        [
            ("Choose from input folder", "choose"),
            ("Type custom image/video path", "custom"),
        ],
    )
    if selected == "__back__":
        return
    input_path = choose_media_file("all") if selected == "choose" else prompt("Input file path", required=True)
    if not input_path:
        return

    path = Path(input_path)
    suffix = path.suffix.lower()
    config = load_config()
    processing = config["processing"]

    args = argparse.Namespace(
        input=input_path,
        output=None,
        output_dir=None,
        providers=processing.get("providers", ""),
        detectors=processing.get("detectors", "nude"),
        object_model=processing.get("object_model", "Models/safety_objects.onnx"),
        object_labels=processing.get("object_labels", "Models/safety_objects.labels.json"),
        object_threshold=processing.get("object_threshold", 0.25),
        blur=False,
        boxes=False,
        with_audio=False,
        codec=processing.get("codec", "mp4v"),
        delete_frames=processing.get("delete_frames", True),
        enhanced_blur=processing.get("enhanced_blur", False),
        color=False,
        mask_color=processing.get("mask_color", "0,0,0"),
        mask_shape=processing.get("mask_shape", "rectangle"),
        blur_strength=processing.get("blur_strength", 23),
        blur_sigma=processing.get("blur_sigma", 0.0),
        analyze_only=False,
        save_report=processing.get("save_report", False),
        report_formats=processing.get("report_formats", "json,csv"),
        export_markers=processing.get("export_markers", ""),
        marker_gap=processing.get("marker_gap", 1.0),
        rule=processing.get("rule", "0/0"),
        full_blur_rule=processing.get("full_blur_rule", ""),
        ffmpeg_path=None,
    )

    args.detectors = choose_from_values(
        "Detector Models",
        ["nude", "objects", "both"],
        default=processing.get("detectors", "nude"),
        allow_back=False,
    )
    if args.detectors in {"objects", "both"}:
        args.object_threshold = float(prompt("Safety-object threshold", default=str(args.object_threshold or 0.25)))

    if suffix in IMAGE_EXTENSIONS:
        args.blur = prompt_bool("Apply blur/mask to detected regions", default=True)
        output = prompt("Output image path (blank for default)", default="")
        args.output = output or None
        args.mask_shape = choose_from_values("Mask Shape", ["rectangle", "ellipse", "oval"], default=args.mask_shape, allow_back=False)
        args.blur_strength = int(prompt("Blur strength", default=str(args.blur_strength or 23)))
        args.color = prompt_bool("Use solid color instead of blur", default=False)
        args.mask_color = prompt("Mask color BGR", default=args.mask_color)
        fbr = prompt("Full blur rule count for image (blank for none)", default=args.full_blur_rule)
        args.full_blur_rule = fbr or None
    elif suffix in VIDEO_EXTENSIONS:
        mode = menu_select(
            "Video Output Mode",
            [
                ("Processed censored video", "processed"),
                ("Detection-box video", "boxes"),
                ("Detection-box video with blur", "boxes_blur"),
                ("Analyze only, no rendered video", "analyze"),
            ],
            allow_back=False,
        )
        args.analyze_only = mode == "analyze"
        args.boxes = mode in {"boxes", "boxes_blur"}
        args.blur = mode in {"processed", "boxes_blur"}
        args.with_audio = prompt_bool("Include original audio", default=processing.get("with_audio", False))
        args.delete_frames = prompt_bool("Avoid/delete intermediate frame files", default=True)
        args.enhanced_blur = prompt_bool("Use enhanced blur", default=processing.get("enhanced_blur", False))
        args.color = prompt_bool("Use solid color instead of blur", default=False)
        args.mask_shape = choose_from_values("Mask Shape", ["rectangle", "ellipse", "oval"], default=args.mask_shape, allow_back=False)
        args.mask_color = prompt("Mask color BGR", default=args.mask_color)
        args.blur_strength = int(prompt("Blur strength", default=str(args.blur_strength or 23)))
        args.codec = choose_from_values("Video Codec", ["mp4v", "avc1", "xvid", "mjpg"], default=args.codec, allow_back=False)
        args.output_dir = prompt("Video output folder", default=processing.get("video_output") or config["paths"].get("video_output", "video_output"))
        args.rule = prompt("Monitor rule percentage/count", default=args.rule)
        args.full_blur_rule = prompt("Full blur rule labels/frames (blank for default)", default=args.full_blur_rule)
        args.save_report = args.analyze_only or prompt_bool("Write JSON/CSV detection reports", default=processing.get("save_report", False))
        args.report_formats = prompt("Report formats", default=args.report_formats)
        args.export_markers = choose_from_values("Marker Export", ["none", "edl", "fcpxml", "both"], default=args.export_markers or "none", allow_back=False)
        if args.export_markers == "none":
            args.export_markers = ""
        if args.export_markers:
            args.marker_gap = float(prompt("Marker gap seconds", default=str(args.marker_gap or 1.0)))
    else:
        print_error(f"Unsupported input type: {suffix}")
        pause()
        return

    providers = prompt("ONNX providers (blank = auto/default)", default=args.providers)
    args.providers = providers or None
    clear_screen()
    command_process(args)
    pause()


def interactive_settings():
    common_settings = [
        ("Active rule profile", "active_rule_profile"),
        ("ONNX providers", "processing.providers"),
        ("Detector models", "processing.detectors"),
        ("Safety object model", "processing.object_model"),
        ("Safety object labels", "processing.object_labels"),
        ("Safety object threshold", "processing.object_threshold"),
        ("Video codec", "processing.codec"),
        ("Mask shape", "processing.mask_shape"),
        ("Mask color", "processing.mask_color"),
        ("Blur strength", "processing.blur_strength"),
        ("Blur sigma", "processing.blur_sigma"),
        ("Delete frames", "processing.delete_frames"),
        ("Include audio", "processing.with_audio"),
        ("Enhanced blur", "processing.enhanced_blur"),
        ("Save reports", "processing.save_report"),
        ("Report formats", "processing.report_formats"),
        ("Marker export", "processing.export_markers"),
        ("Marker gap seconds", "processing.marker_gap"),
        ("Input folder", "paths.input"),
        ("Video output folder", "paths.video_output"),
        ("Screen guard monitor", "screen_guard.monitor"),
        ("Screen guard mode", "screen_guard.mode"),
        ("Screen guard label filter", "screen_guard.label_filter"),
        ("Screen guard respect rules", "screen_guard.respect_rules"),
        ("Screen guard capture backend", "screen_guard.capture_backend"),
        ("Screen guard smooth overlay", "screen_guard.smooth_overlay"),
        ("Screen guard smooth IoU", "screen_guard.smooth_iou"),
        ("Screen guard smooth alpha", "screen_guard.smooth_alpha"),
        ("Screen guard stable score alpha", "screen_guard.stable_score_alpha"),
        ("Screen guard track hold ms", "screen_guard.track_hold_ms"),
        ("Screen guard merge nearby", "screen_guard.merge_nearby"),
        ("Screen guard merge distance", "screen_guard.merge_distance"),
        ("Screen guard merge overlap", "screen_guard.merge_overlap"),
        ("Screen guard feedback delta", "screen_guard.feedback_delta"),
        ("Screen guard feedback-safe capture", "screen_guard.feedback_safe_capture"),
        ("Screen guard capture hide ms", "screen_guard.capture_hide_ms"),
        ("Screen guard drop stale on screen change", "screen_guard.drop_stale_on_screen_change"),
        ("Screen guard stale region delta", "screen_guard.stale_region_delta"),
        ("Screen guard screen change delta", "screen_guard.screen_change_delta"),
        ("Screen guard exclude overlay from capture", "screen_guard.exclude_overlay_capture"),
        ("Screen guard show boxes", "screen_guard.show_boxes"),
        ("Screen guard show labels", "screen_guard.show_labels"),
        ("Screen guard show status", "screen_guard.show_status"),
        ("Screen guard block regions", "screen_guard.block_enabled"),
        ("Screen guard blur regions", "screen_guard.blur_enabled"),
        ("Screen guard privacy cover", "screen_guard.privacy_on_detection"),
        ("Screen guard blur style", "screen_guard.blur_style"),
        ("Screen guard blur strength", "screen_guard.blur_strength"),
        ("Screen guard mask shape", "screen_guard.mask_shape"),
        ("Screen guard threshold", "screen_guard.threshold"),
        ("Screen guard FPS", "screen_guard.fps"),
        ("Screen guard hold ms", "screen_guard.hold_ms"),
        ("Screen guard box padding", "screen_guard.box_padding"),
        ("Screen guard min box area", "screen_guard.min_box_area"),
        ("Screen guard outline color", "screen_guard.outline_color"),
        ("Screen guard safe outline", "screen_guard.safe_outline_color"),
        ("Screen guard block color", "screen_guard.block_color"),
        ("Screen guard label background", "screen_guard.label_bg"),
        ("Screen guard label color", "screen_guard.label_color"),
        ("Screen guard click-through", "screen_guard.click_through"),
    ]

    while True:
        choice = menu_select(
            "Settings",
            [
                ("Show full config", "show"),
                ("Edit common setting", "edit_common"),
                ("Set any dotted key", "set_any"),
                ("Reset settings to defaults", "reset"),
            ],
        )
        if choice == "__back__":
            return
        clear_screen()
        if choice == "show":
            command_settings(argparse.Namespace(settings_action="show"))
            pause()
        elif choice == "edit_common":
            config = load_config()
            options = []
            for label, key in common_settings:
                try:
                    current = get_nested(config, key)
                except KeyError:
                    current = ""
                options.append((f"{label}: {current}", key))
            key = menu_select("Choose Setting", options)
            if key == "__back__":
                continue
            current = get_nested(config, key)
            if isinstance(current, bool):
                value = str(prompt_bool(f"Set {key}", default=current)).lower()
            elif key in {"processing.mask_shape", "screen_guard.mask_shape"}:
                value = choose_from_values("Mask Shape", ["rectangle", "ellipse", "oval"], default=current, allow_back=False)
            elif key == "processing.codec":
                value = choose_from_values("Video Codec", ["mp4v", "avc1", "xvid", "mjpg"], default=current, allow_back=False)
            elif key == "processing.detectors":
                value = choose_from_values("Detector Models", ["nude", "objects", "both"], default=current, allow_back=False)
            elif key == "processing.export_markers":
                value = choose_from_values("Marker Export", ["none", "edl", "fcpxml", "both"], default=current or "none", allow_back=False)
                if value == "none":
                    value = ""
            elif key == "processing.report_formats":
                value = choose_from_values("Report Formats", ["json,csv", "json", "csv"], default=current, allow_back=False)
            elif key == "screen_guard.mode":
                value = choose_from_values("Screen Guard Mode", ["box", "blur", "block", "both", "privacy"], default=current, allow_back=False)
            elif key == "screen_guard.label_filter":
                value = choose_from_values("Label Filter", ["exposed", "body", "all"], default=current, allow_back=False)
            elif key == "screen_guard.blur_style":
                value = choose_from_values("Blur Style", ["gaussian", "pixelate"], default=current, allow_back=False)
            elif key == "screen_guard.capture_backend":
                value = choose_from_values("Capture Backend", ["auto", "gdi", "mss"], default=current, allow_back=False)
            else:
                value = prompt(f"Set {key}", default=str(current))
            command_settings(argparse.Namespace(settings_action="set", key=key, value=value))
            pause()
        elif choice == "set_any":
            key = prompt("Dotted setting key", required=True)
            value = prompt("Value", required=True)
            command_settings(argparse.Namespace(settings_action="set", key=key, value=value))
            pause()
        elif choice == "reset":
            if prompt_bool("Reset all settings", default=False):
                command_settings(argparse.Namespace(settings_action="reset"))
                pause()


def interactive_rules():
    while True:
        choice = menu_select(
            "Rule Profiles",
            [
                ("List profiles", "list"),
                ("Use/activate profile", "use"),
                ("Show active profile", "show_active"),
                ("Create profile", "create"),
                ("Change one label", "set_label"),
                ("Export profile to .rule", "export"),
                ("Rebuild BlurException.rule", "init"),
            ],
        )
        if choice == "__back__":
            return
        config = load_config()
        profiles = sorted(config.setdefault("rule_profiles", default_profiles()).keys())
        clear_screen()
        if choice == "list":
            command_rules(argparse.Namespace(rules_action="list"))
            pause()
        elif choice == "use":
            profile = choose_from_values("Activate Profile", profiles, default=config.get("active_rule_profile"), allow_back=True)
            if profile != "__back__":
                clear_screen()
                command_rules(argparse.Namespace(rules_action="use", profile=profile))
                pause()
        elif choice == "show_active":
            command_rules(argparse.Namespace(rules_action="show", profile=config.get("active_rule_profile")))
            pause()
        elif choice == "create":
            name = prompt("New profile name", required=True)
            base = choose_from_values("Base Profile", profiles, default=config.get("active_rule_profile"), allow_back=True)
            if base == "__back__":
                base = None
            command_rules(argparse.Namespace(rules_action="create", profile=name, base=base, from_file=None, force=False))
            pause()
        elif choice == "set_label":
            profile = choose_from_values("Profile", profiles, default=config.get("active_rule_profile"), allow_back=True)
            if profile == "__back__":
                continue
            label = choose_from_values("Label", ALL_CENSOR_LABELS, allow_back=True)
            if label == "__back__":
                continue
            value = prompt_bool(f"Blur {label}", default=True)
            clear_screen()
            command_rules(argparse.Namespace(rules_action="set", profile=profile, label=label, value=str(value).lower()))
            pause()
        elif choice == "export":
            profile = choose_from_values("Profile", profiles, default=config.get("active_rule_profile"), allow_back=True)
            if profile == "__back__":
                continue
            output = prompt("Output .rule file", default=f"{profile}.rule")
            command_rules(argparse.Namespace(rules_action="export", profile=profile, output=output))
            pause()
        elif choice == "init":
            command_rules(argparse.Namespace(rules_action="init"))
            pause()


def interactive_folders():
    while True:
        choice = menu_select(
            "Folders",
            [
                ("Show folders", "show"),
                ("Create missing folders", "create"),
                ("Scan input folder", "scan"),
            ],
        )
        if choice == "__back__":
            return
        clear_screen()
        if choice == "show":
            command_folders(argparse.Namespace(create=False, recursive=False))
        elif choice == "create":
            command_folders(argparse.Namespace(create=True, recursive=False))
        elif choice == "scan":
            command_scan(argparse.Namespace(folder=None, recursive=True, type="all", names_only=False, progress=True))
        pause()


def interactive_launch():
    while True:
        choice = menu_select(
            "Launch App",
            [
                ("Desktop GUI", "gui"),
                ("API/Web server", "web"),
                ("Live camera detector", "live"),
                ("Streamer mode", "streamer"),
                ("Screen guard overlay", "screen"),
            ],
        )
        if choice == "__back__":
            return
        clear_screen()
        if choice == "screen" and prompt_bool("Use saved screen guard settings", default=True):
            extra = []
        elif choice == "screen":
            config = load_config()
            screen_guard = config.get("screen_guard", {})
            monitor = prompt("Monitor", default=str(screen_guard.get("monitor", 1)))
            mode = choose_from_values(
                "Screen Guard Mode",
                ["box", "blur", "block", "both", "privacy"],
                default=screen_guard.get("mode", "box"),
                allow_back=False,
            )
            label_filter = choose_from_values(
                "Label Filter",
                ["exposed", "body", "all"],
                default=screen_guard.get("label_filter", "exposed"),
                allow_back=False,
            )
            show_boxes = prompt_bool("Show outline boxes", default=screen_guard.get("show_boxes", True))
            show_labels = prompt_bool("Show text labels", default=screen_guard.get("show_labels", False))
            respect_rules = prompt_bool("Respect active BlurException.rule", default=screen_guard.get("respect_rules", True))
            extra = [
                "--monitor", monitor,
                "--mode", mode,
                "--label-filter", label_filter,
                "--show-boxes" if show_boxes else "--no-boxes",
                "--show-labels" if show_labels else "--no-labels",
                "--respect-rules" if respect_rules else "--ignore-rules",
            ]
        else:
            extra_text = prompt("Extra script arguments (blank for none)", default="")
            extra = extra_text.split() if extra_text else []
        command_launch(argparse.Namespace(target=choice, extra=extra))
        pause()


def interactive_status():
    while True:
        choice = menu_select(
            "Status & Providers",
            [
                ("Show full status", "status"),
                ("Show selected ONNX providers", "providers"),
            ],
        )
        if choice == "__back__":
            return
        clear_screen()
        if choice == "status":
            command_status(argparse.Namespace())
        elif choice == "providers":
            providers = prompt("Provider override (blank = config/default)", default="")
            command_providers(argparse.Namespace(providers=providers or None))
        pause()


def interactive_menu():
    load_config()
    while True:
        config = load_config()
        active = config.get("active_rule_profile", "default")
        provider_text = config.get("processing", {}).get("providers") or "auto"
        choice = menu_select(
            "SafeVision Console",
            [
                ("Scan images/videos", "scan"),
                ("Process image/video", "process"),
                ("Settings", "settings"),
                ("Rule profiles", "rules"),
                ("Folders", "folders"),
                ("Launch GUI/API/live tools", "launch"),
                ("Status and providers", "status"),
                ("Exit", "exit"),
            ],
            subtitle=f"Active profile: {active} | Providers: {provider_text}",
            allow_back=False,
        )
        if choice == "scan":
            interactive_scan()
        elif choice == "process":
            interactive_process()
        elif choice == "settings":
            interactive_settings()
        elif choice == "rules":
            interactive_rules()
        elif choice == "folders":
            interactive_folders()
        elif choice == "launch":
            interactive_launch()
        elif choice == "status":
            interactive_status()
        elif choice == "exit":
            clear_screen()
            print_ok("SafeVision CLI closed.")
            return 0


def command_menu(args):
    if not sys.stdin.isatty():
        print_error("Interactive menu needs a terminal. Use subcommands such as 'scan', 'process', or 'settings'.")
        return 2
    return interactive_menu()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="safeVisionCLI.py",
        description="SafeVision command console for processing, settings, rules, scans, GUI, and API.",
    )
    subparsers = parser.add_subparsers(dest="command")

    menu_parser = subparsers.add_parser("menu", help="Open the interactive looped console menu")
    menu_parser.set_defaults(func=command_menu)

    init_parser = subparsers.add_parser("init", help="Create settings folders, config, and default rule file")
    init_parser.set_defaults(func=command_init)

    status_parser = subparsers.add_parser("status", help="Show SafeVision installation status")
    status_parser.set_defaults(func=command_status)

    folders_parser = subparsers.add_parser("folders", help="Show configured project folders")
    folders_parser.add_argument("--create", action="store_true", help="Create missing configured folders")
    folders_parser.add_argument("--recursive", action="store_true", help="Count files recursively")
    folders_parser.set_defaults(func=command_folders)

    scan_parser = subparsers.add_parser("scan", help="Scan a folder for image/video inputs")
    scan_parser.add_argument("folder", nargs="?", help="Folder to scan. Defaults to configured input folder.")
    scan_parser.add_argument("-r", "--recursive", action="store_true", help="Scan recursively")
    scan_parser.add_argument("-t", "--type", choices=["all", "image", "video"], default="all")
    scan_parser.add_argument("--names-only", action="store_true", help="Print only paths")
    scan_parser.add_argument("--progress", action="store_true", help="Show tqdm progress while listing")
    scan_parser.set_defaults(func=command_scan)

    settings_parser = subparsers.add_parser("settings", help="Show or update settings/configs.json")
    settings_sub = settings_parser.add_subparsers(dest="settings_action", required=True)
    settings_sub.add_parser("show", help="Print full config").set_defaults(func=command_settings)
    settings_get = settings_sub.add_parser("get", help="Get a dotted setting key")
    settings_get.add_argument("key")
    settings_get.set_defaults(func=command_settings)
    settings_set = settings_sub.add_parser("set", help="Set a dotted setting key")
    settings_set.add_argument("key")
    settings_set.add_argument("value")
    settings_set.set_defaults(func=command_settings)
    settings_sub.add_parser("reset", help="Reset config to defaults").set_defaults(func=command_settings)

    rules_parser = subparsers.add_parser("rules", help="Manage BlurException.rule profiles")
    rules_sub = rules_parser.add_subparsers(dest="rules_action", required=True)
    rules_sub.add_parser("init", help="Create BlurException.rule from active profile").set_defaults(func=command_rules)
    rules_sub.add_parser("list", help="List rule profiles").set_defaults(func=command_rules)
    rules_show = rules_sub.add_parser("show", help="Show a rule profile")
    rules_show.add_argument("profile", nargs="?")
    rules_show.set_defaults(func=command_rules)
    rules_use = rules_sub.add_parser("use", help="Activate a profile and write BlurException.rule")
    rules_use.add_argument("profile")
    rules_use.set_defaults(func=command_rules)
    rules_create = rules_sub.add_parser("create", help="Create a new rule profile")
    rules_create.add_argument("profile")
    rules_create.add_argument("--base", help="Copy from an existing profile")
    rules_create.add_argument("--from-file", help="Import rules from a .rule file")
    rules_create.add_argument("--force", action="store_true", help="Overwrite if profile exists")
    rules_create.set_defaults(func=command_rules)
    rules_set = rules_sub.add_parser("set", help="Set one label in a profile")
    rules_set.add_argument("profile")
    rules_set.add_argument("label")
    rules_set.add_argument("value", help="true/false")
    rules_set.set_defaults(func=command_rules)
    rules_delete = rules_sub.add_parser("delete", help="Delete a rule profile")
    rules_delete.add_argument("profile")
    rules_delete.set_defaults(func=command_rules)
    rules_export = rules_sub.add_parser("export", help="Export a profile to a .rule file")
    rules_export.add_argument("profile")
    rules_export.add_argument("output")
    rules_export.set_defaults(func=command_rules)

    process_parser = subparsers.add_parser("process", help="Process an image or video using the right SafeVision engine")
    process_parser.add_argument("input")
    process_parser.add_argument("-o", "--output", help="Image output file")
    process_parser.add_argument("--output-dir", help="Video output folder")
    process_parser.add_argument("--providers", help="Comma-separated ONNX providers")
    process_parser.add_argument("--detectors", choices=["nude", "objects", "both"], help="Detector set to use")
    process_parser.add_argument("--object-model", help="Path to safety-object ONNX model")
    process_parser.add_argument("--object-labels", help="Path to safety-object labels JSON")
    process_parser.add_argument("--object-threshold", type=float, help="Minimum confidence for safety-object detections")
    process_parser.add_argument("--blur", action="store_true", help="Apply blur/mask to detections")
    process_parser.add_argument("--boxes", action="store_true", help="For video, generate detection-box output")
    process_parser.add_argument("-a", "--with-audio", action="store_true")
    process_parser.add_argument("-c", "--codec", choices=["mp4v", "avc1", "xvid", "mjpg"])
    process_parser.add_argument("-df", "--delete-frames", action="store_true")
    process_parser.add_argument("--enhanced-blur", action="store_true")
    process_parser.add_argument("--color", action="store_true", help="Use solid color instead of blur")
    process_parser.add_argument("--mask-color", help="BGR color, e.g. 0,0,0")
    process_parser.add_argument("--mask-shape", choices=["rectangle", "ellipse", "oval"])
    process_parser.add_argument("--blur-strength", type=int, help="Regional blur kernel strength")
    process_parser.add_argument("--blur-sigma", type=float, help="Regional Gaussian blur sigma")
    process_parser.add_argument("--analyze-only", action="store_true", help="Analyze video without rendering a censored output")
    process_parser.add_argument("--save-report", action="store_true", help="Write JSON/CSV detection reports")
    process_parser.add_argument("--report-formats", help="Comma-separated report formats: json,csv")
    process_parser.add_argument("--export-markers", help="Comma-separated marker formats: edl,fcpxml,both")
    process_parser.add_argument("--marker-gap", type=float, help="Seconds between detections before a new marker")
    process_parser.add_argument("-r", "--rule", help="Video full blur monitor rule percentage/count")
    process_parser.add_argument("-fbr", "--full-blur-rule", help="Full blur rule")
    process_parser.add_argument("--ffmpeg-path")
    process_parser.set_defaults(func=command_process)

    launch_parser = subparsers.add_parser("launch", help="Launch GUI, API/web, screen guard, live, streamer, image, or video scripts")
    launch_parser.add_argument("target", choices=["gui", "api", "web", "screen", "live", "streamer", "image", "video"])
    launch_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to the target script")
    launch_parser.set_defaults(func=command_launch)

    screen_parser = subparsers.add_parser("screen", help="Launch Screen Guard using settings/configs.json")
    screen_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args overriding saved screen guard settings")
    screen_parser.set_defaults(func=command_screen)

    providers_parser = subparsers.add_parser("providers", help="Show selected ONNX providers")
    providers_parser.add_argument("--providers", help="Provider override list")
    providers_parser.set_defaults(func=command_providers)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1 and sys.stdin.isatty():
        return interactive_menu()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
