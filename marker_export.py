import csv
import json
import os
from collections import Counter
from xml.etree import ElementTree as ET


def parse_format_list(value, default=None):
    if value in (None, "", False):
        return list(default or [])
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).replace(";", ",").split(",")

    formats = []
    for item in raw_values:
        name = str(item).strip().lower()
        if not name or name == "none":
            continue
        if name in {"all", "both"}:
            formats.extend(["edl", "fcpxml"])
        elif name in {"json", "csv", "edl", "fcpxml", "fcp", "xml"}:
            formats.append("fcpxml" if name in {"fcp", "xml"} else name)
    return list(dict.fromkeys(formats))


def detection_category(label):
    label = str(label or "").upper()
    if "EXPOSED" in label:
        return "exposed"
    if "COVERED" in label:
        return "covered"
    if label.startswith("FACE_"):
        return "face"
    return "other"


def detection_events_from_frame(detections, frame_index, fps):
    events = []
    time_seconds = (max(1, int(frame_index)) - 1) / fps if fps else 0.0
    for detection in detections:
        label = detection.get("class", "UNKNOWN")
        box = detection.get("box", [0, 0, 0, 0])
        score = float(detection.get("score", 0.0))
        events.append(
            {
                "frame": int(frame_index),
                "time": round(time_seconds, 6),
                "class": label,
                "category": detection_category(label),
                "score": score,
                "box": [int(value) for value in box],
            }
        )
    return events


def build_analysis_payload(events, metadata=None, stats=None):
    metadata = dict(metadata or {})
    stats = dict(stats or {})
    class_counts = Counter(event["class"] for event in events)
    category_counts = Counter(event.get("category", "other") for event in events)
    frames_with_detections = sorted({event["frame"] for event in events})

    summary = {
        "total_detections": len(events),
        "frames_with_detections": len(frames_with_detections),
        "class_counts": dict(sorted(class_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
    }
    summary.update(stats)

    return {
        "metadata": metadata,
        "summary": summary,
        "detections": events,
    }


def write_json_report(path, events, metadata=None, stats=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = build_analysis_payload(events, metadata=metadata, stats=stats)
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump(payload, report_file, indent=2)
    return path


def write_csv_report(path, events):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["frame", "time", "class", "category", "score", "x", "y", "w", "h"]
    with open(path, "w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            x, y, w, h = event.get("box", [0, 0, 0, 0])
            writer.writerow(
                {
                    "frame": event.get("frame", 0),
                    "time": event.get("time", 0),
                    "class": event.get("class", ""),
                    "category": event.get("category", ""),
                    "score": f"{float(event.get('score', 0.0)):.6f}",
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )
    return path


def write_detection_reports(output_dir, base_name, events, metadata=None, stats=None, formats=None):
    formats = parse_format_list(formats, default=["json", "csv"])
    written = []
    if "json" in formats:
        written.append(write_json_report(os.path.join(output_dir, f"{base_name}_analysis.json"), events, metadata, stats))
    if "csv" in formats:
        written.append(write_csv_report(os.path.join(output_dir, f"{base_name}_analysis.csv"), events))
    return written


def frame_to_timecode(frame_index, fps):
    frame_rate = max(1, int(round(fps or 30)))
    total_frames = max(0, int(frame_index) - 1)
    frames = total_frames % frame_rate
    total_seconds = total_frames // frame_rate
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _duration_frames(start_frame, end_frame):
    return max(1, int(end_frame) - int(start_frame) + 1)


def group_marker_events(events, fps, gap_seconds=1.0):
    fps = fps or 30.0
    max_gap_frames = max(1, int(round(float(gap_seconds or 1.0) * fps)))
    by_label = {}
    for event in sorted(events, key=lambda item: (item["class"], item["frame"])):
        by_label.setdefault(event["class"], []).append(event)

    groups = []
    for label, label_events in by_label.items():
        current = None
        for event in label_events:
            frame = event["frame"]
            if current is None or frame - current["end_frame"] > max_gap_frames:
                if current is not None:
                    groups.append(current)
                current = {
                    "label": label,
                    "category": event.get("category", "other"),
                    "start_frame": frame,
                    "end_frame": frame,
                    "max_score": float(event.get("score", 0.0)),
                    "detections": 1,
                }
            else:
                current["end_frame"] = frame
                current["max_score"] = max(current["max_score"], float(event.get("score", 0.0)))
                current["detections"] += 1
        if current is not None:
            groups.append(current)

    return sorted(groups, key=lambda group: (group["start_frame"], group["label"]))


def export_edl(path, marker_groups, fps, title="SafeVision Markers", source_name="SafeVision"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as edl_file:
        edl_file.write(f"TITLE: {title}\n")
        edl_file.write("FCM: NON-DROP FRAME\n\n")

        if not marker_groups:
            edl_file.write("* No detections found.\n")
            return path

        for index, group in enumerate(marker_groups, start=1):
            start_tc = frame_to_timecode(group["start_frame"], fps)
            end_tc = frame_to_timecode(group["end_frame"] + 1, fps)
            label = group["label"]
            score = group["max_score"]
            edl_file.write(
                f"{index:03d}  AX       V     C        {start_tc} {end_tc} {start_tc} {end_tc}\n"
            )
            edl_file.write(f"* FROM CLIP NAME: {source_name}\n")
            edl_file.write(f"* LOC: {start_tc} RED SafeVision: {label} score={score:.3f}\n\n")
    return path


def _fraction_seconds(frame, fps):
    fps_int = max(1, int(round(fps or 30)))
    frame = max(0, int(frame))
    return f"{frame}/{fps_int}s"


def export_fcpxml(
    path,
    marker_groups,
    fps,
    total_frames=1,
    width=1920,
    height=1080,
    title="SafeVision Markers",
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fps_int = max(1, int(round(fps or 30)))
    total_frames = max(1, int(total_frames or 1))

    fcpxml = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        name=f"FFVideoFormat{width}x{height}p{fps_int}",
        frameDuration=f"1/{fps_int}s",
        width=str(int(width or 1920)),
        height=str(int(height or 1080)),
    )
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="SafeVision")
    project = ET.SubElement(event, "project", name=title)
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=_fraction_seconds(total_frames, fps),
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")
    gap = ET.SubElement(
        spine,
        "gap",
        name="SafeVision markers",
        offset="0s",
        duration=_fraction_seconds(total_frames, fps),
    )
    for group in marker_groups:
        start_frame = max(0, int(group["start_frame"]) - 1)
        duration = _duration_frames(group["start_frame"], group["end_frame"])
        ET.SubElement(
            gap,
            "marker",
            start=_fraction_seconds(start_frame, fps),
            duration=_fraction_seconds(duration, fps),
            value=f"SafeVision: {group['label']} score={group['max_score']:.3f}",
        )

    tree = ET.ElementTree(fcpxml)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def export_marker_files(output_dir, base_name, events, metadata=None, formats=None, gap_seconds=1.0):
    metadata = dict(metadata or {})
    marker_formats = parse_format_list(formats)
    if not marker_formats:
        return []

    fps = float(metadata.get("fps") or 30.0)
    marker_groups = group_marker_events(events, fps=fps, gap_seconds=gap_seconds)
    title = f"SafeVision Markers - {base_name}"
    source_name = metadata.get("source_name") or base_name
    written = []

    if "edl" in marker_formats:
        written.append(
            export_edl(
                os.path.join(output_dir, f"{base_name}_markers.edl"),
                marker_groups,
                fps=fps,
                title=title,
                source_name=source_name,
            )
        )
    if "fcpxml" in marker_formats:
        written.append(
            export_fcpxml(
                os.path.join(output_dir, f"{base_name}_markers.fcpxml"),
                marker_groups,
                fps=fps,
                total_frames=metadata.get("total_frames", 1),
                width=metadata.get("width", 1920),
                height=metadata.get("height", 1080),
                title=title,
            )
        )

    return written
