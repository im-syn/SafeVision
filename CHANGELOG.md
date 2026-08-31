<a id="top"></a>

<div align="center">
  <a href="README.md">
    <img src="https://i.ibb.co/d4LqhX4/Safe-Vision-2.png" alt="SafeVision logo" width="500">
  </a>

  <h1>SafeVision Changes</h1>

  <p><strong>Migration notes for the current development release.</strong></p>

  <p>
    <img alt="From" src="https://img.shields.io/badge/Upgrade-From_6.8-2563EB?style=for-the-badge">
    <img alt="Status" src="https://img.shields.io/badge/Status-Development-F59E0B?style=for-the-badge">
    <img alt="Compatibility" src="https://img.shields.io/badge/Commands-Compatible-22C55E?style=for-the-badge">
  </p>

  <p>
    <a href="README.md">Project home</a> ·
    <a href="docs/PROJECT_STRUCTURE.md">Project structure</a> ·
    <a href="docs/LICENSING.md">Licensing</a> ·
    <a href="tests/README.md">Validation</a>
  </p>
</div>

---

## Current development version (from 6.8)

This is the practical migration list for users of the older public SafeVision repository.

### Detection

- Added `age_gender_detector.py` around `onnx-community/age-gender-prediction-ONNX`.
- The default check set is now `nude,age,gender` for images and videos.
- Age and gender can run independently: `--detectors age`, `gender`, or `demographics`.
- Added explicit missing-model errors. NSFW-only runs still work when the age/gender model is absent, as long as age and gender are disabled.
- Video reports count sampled face observations; they do not claim to count unique people.

### Child-protection policy

- Added `BLOCK_IF_NSFW_AND_CHILD`, `BLOCK_IF_CHILD`, and `BLOCK_ON_AGE_REVIEW`.
- Added configurable `UNDERAGE_AGE` and `AGE_REVIEW_MARGIN`.
- Added `PROTECTION_NSFW_MIN_RISK` and `PROTECTION_NSFW_MIN_CONFIDENCE` so common body-context detections do not automatically become child + NSFW blocks.
- The balanced default excludes `ARMPITS_EXPOSED` from regional censoring and requires HIGH/CRITICAL evidence at confidence 0.5 for the compound rule.
- Analysis JSON/API responses now include policy reasons and the exact NSFW evidence that activated the compound gate.

### Rendering

- Added whole-media modes `blur`, `gray`, `black`, and `color`.
- Solid modes replace every source pixel. They are intended for outputs where visual information must not remain underneath.
- Added automatic centered messages for NSFW, NSFW + estimated underage, underage-only, and age-review triggers.
- Added `--boxes` / `--no-boxes`, `--save-boxes-copy` / `--no-save-boxes-copy`, and `--save-blur-copy` / `--no-save-blur-copy` for images.
- The separate unredacted boxes copy is now off by default. This is a privacy-oriented default change.
- Video monitoring now writes `_fully_covered.mp4` instead of describing every whole-media result as “fully blurred.”

### Rules and command line

- `video.py` now accepts `-e/--exception`, matching `main.py`.
- Added per-run child-policy overrides and full-cover overrides to `main.py`, `video.py`, and `safeVisionCLI.py process`.
- Expanded the desktop GUI with the same policy thresholds, `.rule` selection,
  final-box/copy switches, and full-cover mode/color/text controls.
- Added 50 complete presets under `rule_templates/`, plus a catalog that explains the intended workflow for each file.
- `BlurException.rule` now includes full-cover mode, color, text, and message settings.

### Web API

- Moved the deployable local API to `SafeVision Web API/`.
- Kept `safevision_api.py` as a compatibility launcher.
- Added `.env`-based server, model, rule, and runtime path configuration.
- Added image and video render controls for blur, boxes, forced full cover, cover mode/color, and warning text.
- Added `GET /api/v1/results/<filename>` for generated result downloads.
- Added Waitress and reverse-proxy deployment instructions.

### `vision2` live service

- Added age/gender checks and child-protection configuration to the live API/admin codebase.
- Added admin defaults and request parameters for solid/blur full cover, automatic policy cover, colors, and reason-specific messages.
- Child-protection blocks can force a complete cover even when the original request asked for regional rendering.
- Image and video output JSON records the applied cover mode and message.

### Repository cleanup

- Generated inputs/outputs, logs, API runtime files, Python caches, local environments, and converted models are ignored.
- The large age/gender model stays local by default and is no longer expected in a normal Git commit.
- The legacy `Models/best_gender.onnx` file is no longer used. Deployments should point at `Models/onnx-communityage-gender-prediction-ONNX.onnx`.

### Application organization

- Moved the maintained PyQt5 implementation to `apps/desktop/SafeVisionGUI.py`.
- Moved live camera, Screen Guard, and OBS/virtual-camera implementations to
  `apps/live/`.
- Kept `SafeVisionGUI.py`, `live.py`, `safeVisionScreenGuard.py`, and
  `live_streamer.py` as small root compatibility launchers.
- Added repository-root path resolution so models, rules, engines, settings,
  and FFmpeg discovery keep working from nested modules.
- Added direct package launch support with `python -m apps...`.

### Documentation and licensing audit

- Added folder-specific manuals for applications, GUI, live tools, models,
  settings, tests, API runtime, input/output, censor copies, reviewer copies,
  and logs.
- Expanded the Web API and 50-template rule catalog with visual navigation,
  architecture, request/policy flow, troubleshooting, and operational checks.
- Added `docs/README.md`, `docs/PROJECT_STRUCTURE.md`, and
  `docs/LICENSING.md`.
- Added a root `NOTICE` and a model registry with provenance, embedded metadata,
  hashes, and separate license status.
- Documented that `best.onnx` and `safety_objects.onnx` declare Ultralytics
  AGPL-3.0, the new age/gender model card declares Apache-2.0, and the legacy
  `best_gender.onnx` has no established license.
- Replaced `.gitkeep` files in user/runtime folders with privacy-focused
  READMEs while continuing to ignore generated contents.

### Compatibility notes

- Existing `python main.py`, `python video.py`, and `python safevision_api.py` entry points remain available.
- `full_blur` request/option names remain accepted where they existed; new documentation calls the operation “full cover” because it may be a solid replacement rather than a blur.
- Existing regional `--color` / `--mask-color` behavior is unchanged. Use `--full-cover-mode color` / `--full-cover-color` for the whole image or video.

<p align="right"><a href="#top">⬆️ Back to top</a></p>
