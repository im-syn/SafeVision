<a id="top"></a>

<div align="center">

# 🏗️ SafeVision Project Structure

![Architecture](https://img.shields.io/badge/Architecture-Layered-2563EB?style=for-the-badge)
![Compatibility](https://img.shields.io/badge/Compatibility-Stable_Launchers-22C55E?style=for-the-badge)
![Docs](https://img.shields.io/badge/Docs-Folder_Specific-7C3AED?style=for-the-badge)

[Documentation center](README.md) · [Project home](../README.md) · [Applications](../apps/README.md)

</div>

---

## 🎯 Organization goals

SafeVision separates stable user commands from maintained implementations:

- public commands stay short and familiar;
- large GUI/live implementations live in purpose-specific packages;
- core detector and policy modules remain reusable;
- API deployment files stay isolated from desktop runtime artifacts;
- every important folder explains its ownership and privacy behavior;
- model licensing remains separate from source-code licensing.

## 🗂️ Repository tree

```text
SafeVision/
├── main.py                         Stable image engine CLI
├── video.py                        Stable video engine CLI
├── safeVisionCLI.py                Settings, rules, launch, and process console
├── safevision_api.py               Compatibility launcher for local API
│
├── SafeVisionGUI.py                Compatibility launcher
├── live.py                         Compatibility launcher
├── safeVisionScreenGuard.py        Compatibility launcher
├── live_streamer.py                Compatibility launcher
│
├── apps/
│   ├── desktop/SafeVisionGUI.py    Maintained PyQt5 implementation
│   └── live/
│       ├── live.py                 Maintained camera implementation
│       ├── safeVisionScreenGuard.py
│       └── live_streamer.py
│
├── age_gender_detector.py          Demographic ONNX adapter
├── object_detector.py              Optional safety-object adapter
├── safevision_utils.py             Shared rules, providers, policy, rendering
├── marker_export.py                Reports and editor marker exports
│
├── Models/                         Model files and provenance registry
├── rule_templates/                 50 portable policy presets
├── settings/                       Persistent console/Screen Guard settings
├── tests/                          Focused synthetic regressions
├── SafeVision Web API/             Flask/Waitress HTTP service
├── docs/                            Architecture and licensing documents
│
├── input/                           Ignored sensitive sources
├── output/                          Ignored requested final output
├── Blur/                            Ignored clean censor copies
├── Prosses/                         Ignored unredacted reviewer copies
├── Logs/                            Ignored analysis/log files
│
├── BlurException.rule              Active policy
├── CHILD_PROTECTION.md              Protection contract
├── CHANGELOG.md                     Release migration notes
├── LICENSE                          Exact software license
├── NOTICE                           Required notices
└── README.md                        Main project homepage
```

## 🧩 Layer ownership

| Layer | Owns | Does not own |
|---|---|---|
| Detector adapters | ONNX inputs/outputs, preprocessing, normalized evidence | Policy decisions or user interface |
| Shared utilities | Providers, rules, evidence gates, cover rendering | Camera/HTTP lifecycle |
| Image/video engines | Media I/O, per-run options, reporting | GUI state or API authentication |
| Applications | User interaction, capture, preview, overlay, OBS | Duplicate model-policy logic |
| Web API | HTTP validation, request overrides, result lifecycle | Desktop settings |
| Rules/settings | Configured behavior and defaults | Model truth or legal conclusions |
| Documentation | Contracts, examples, limits, provenance | Granting third-party rights not owned by SafeVision |

## 🔁 Compatibility launcher pattern

The root launcher is intentionally tiny:

```python
from apps.live.live import *

if __name__ == "__main__":
    main()
```

This keeps existing commands and imports working while GitHub presents the
large implementations in organized folders. Launchers must not add different
defaults or silently transform arguments.

## 📍 Repository-root resolution

Nested applications compute:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

They add that path for shared-module imports and use it for models, rules,
engine scripts, GUI preferences, and optional FFmpeg files. This avoids the
classic refactor bug where moving a script makes it look for
`apps/live/Models/best.onnx`.

## ➕ Adding a new application

1. Choose `apps/desktop/`, `apps/live/`, or create a clearly scoped sibling.
2. Add a package `__init__.py`.
3. Resolve root resources through `PROJECT_ROOT`.
4. Keep detection/policy logic in shared modules.
5. Add a root compatibility launcher only if the command is user-facing.
6. Add the target to `safeVisionCLI.py` when appropriate.
7. Create a detailed folder README.
8. Add import/help tests for root and package paths.
9. Update the main project tree and changelog.

## 🔒 Generated-data boundary

Tracked READMEs keep runtime folders visible in Git without tracking their
contents. `.gitignore` allows only each folder's README. Do not weaken those
rules to make demos easier; use safe remote/synthetic examples instead.

## ✅ Refactor validation

```powershell
python -m py_compile `
  SafeVisionGUI.py live.py safeVisionScreenGuard.py live_streamer.py `
  apps\desktop\SafeVisionGUI.py `
  apps\live\live.py `
  apps\live\safeVisionScreenGuard.py `
  apps\live\live_streamer.py

python live.py --help
python -m apps.live.live --help
python safeVisionScreenGuard.py --help
python -m apps.live.safeVisionScreenGuard --help
python live_streamer.py --help
python -m apps.live.live_streamer --help
```

The desktop GUI is validated through compilation/import and a manual launch;
automated commands should not leave a window running in CI.

<p align="right"><a href="#top">⬆️ Back to top</a></p>
