<a id="top"></a>

<div align="center">
  <a href="README.md">
    <img src="https://i.ibb.co/d4LqhX4/Safe-Vision-2.png" alt="SafeVision logo" width="520">
  </a>

  <h1>Age, Underage, and Gender Checks</h1>

  <p><strong>The SafeVision demographic and compound child-protection contract.</strong></p>

  <p>
    <img alt="Age" src="https://img.shields.io/badge/Age-Estimate_Only-F59E0B?style=for-the-badge">
    <img alt="Gender" src="https://img.shields.io/badge/Gender-Model_Output-7C3AED?style=for-the-badge">
    <img alt="Policy" src="https://img.shields.io/badge/Policy-Compound_Evidence-EF4444?style=for-the-badge">
    <img alt="Review" src="https://img.shields.io/badge/Outcome-Human_Review-2563EB?style=for-the-badge">
  </p>

  <p>
    <a href="README.md">Project home</a> ·
    <a href="rule_templates/README.md">Rule library</a> ·
    <a href="Models/README.md">Model registry</a> ·
    <a href="docs/LICENSING.md">Licensing</a>
  </p>
</div>

---

> [!IMPORTANT]
> Estimated age is not proof of legal age, and model-reported gender is not a
> person's identity. SafeVision produces review signals and configured policy
> outcomes, not legal findings.

SafeVision can run three independent checks in one pass: `nude`, `age`, and
`gender`. They are enabled by default for images and videos. The optional
`objects` check remains available.

The age/gender model is
[`onnx-community/age-gender-prediction-ONNX`](https://huggingface.co/onnx-community/age-gender-prediction-ONNX).
Place `onnx/model.onnx` at:

```text
Models/onnx-communityage-gender-prediction-ONNX.onnx
```

When `age` or `gender` is enabled and this file is missing, SafeVision stops
with an actionable missing-model error. Disabled checks do not load the model.
The supplied model is ONNX IR 10, so SafeVision requires ONNX Runtime 1.18 or
newer; `requirements.txt` enforces this before deployment.

## What the result means

The model is run on every detected face, in batches. Existing SafeVision
`FACE_FEMALE` and `FACE_MALE` boxes are reused when the NSFW check runs. An
OpenCV face detector is used when age/gender runs without the NSFW model.

The model returns an estimated age and a binary female probability. SafeVision
adds:

- `is_underage`: estimated age is below `UNDERAGE_AGE` (18 by default).
- `review_required`: estimated age is from 18 up to the configured review
  margin (three years by default).
- `gender`: `female` or `male`, plus `gender_confidence`.
- face coordinates and the face-detection source.

Age is an estimate, not proof of identity or legal age. The upstream model
reports roughly 4.5 years mean absolute age error and reduced accuracy for
children. It does not provide an age-confidence score. Borderline results
should be reviewed by a person.

### Decision-state matrix

| Face state | Qualified NSFW evidence | Balanced result |
|---|---:|---|
| `ADULT` | No | Normal detector/rendering flow |
| `ADULT` | Yes | NSFW flow; child compound rule does not apply |
| `AGE_REVIEW` | No | Review signal; allowed unless review blocking is enabled |
| `CHILD` | No | Informational child estimate; allowed when `BLOCK_IF_CHILD=false` |
| `CHILD` | Yes | Compound policy block when enabled |

“Qualified” means the NSFW evidence reaches both
`PROTECTION_NSFW_MIN_RISK` and `PROTECTION_NSFW_MIN_CONFIDENCE`.

## CLI and CI

```powershell
# Default: NSFW + estimated age + gender
python main.py -i input/photo.jpg

# Child check only
python main.py -i input/photo.jpg --detectors age

# All checks, including safety objects
python main.py -i input/photo.jpg --detectors all

# Video (face inference is batched within each analyzed frame)
python video.py -i input/video.mp4 --detectors nude,age,gender --save-report

# CI: exit 2 when the compound policy blocks the image
python main.py -i input/photo.jpg --fail-on-policy

# CI: exit 3 on any estimated-underage face
python main.py -i input/photo.jpg --fail-on-underage
```

Every image run writes `Logs/<output>.analysis.json`. Video JSON/CSV reports
include age and gender fields. Video counts are face observations across
frames, not unique people.

The PyQt GUI exposes independent age and gender toggles and the estimated
underage threshold in **Detector Models**. Its Basic tab also includes the
compound/underage/review policy switches, qualified NSFW risk and confidence,
box/copy privacy switches, and full-cover mode/color/text controls. The
Advanced tab selects the `.rule` preset. `settings/configs.json` stores the
same defaults under `processing` for `safeVisionCLI.py`.

Screen Guard, live camera, and Live Streamer also default to NSFW + age +
gender. They reuse NSFW face boxes and apply the same rule file. A matching
child-protection policy forces a full-screen/stream safety block; demographic
face boxes are informational and never censored by themselves.

```powershell
python safeVisionScreenGuard.py --detectors nsfw,age,gender --show-status
python live.py --demographics
python live_streamer.py --detectors nsfw,age,gender --show-demographics
```

## Protection rule file

`BlurException.rule` now supports these policy keys:

```ini
BLOCK_IF_NSFW_AND_CHILD = true
BLOCK_IF_CHILD = false
BLOCK_ON_AGE_REVIEW = false
PROTECTION_NSFW_MIN_RISK = HIGH
PROTECTION_NSFW_MIN_CONFIDENCE = 0.5
UNDERAGE_AGE = 18
AGE_REVIEW_MARGIN = 3

FULL_COVER_MODE = gray
FULL_COVER_SHOW_TEXT = true
FULL_COVER_MESSAGE_NSFW_AND_CHILD = Possible illegal content - review required
```

`BLOCK_IF_NSFW_AND_CHILD` is the default compound rule. A `CHILD` result by
itself is informational unless `BLOCK_IF_CHILD` is enabled. `CHILD`, `ADULT`,
and `AGE_REVIEW` regions are not blurred as ordinary NSFW detections; a blocked
compound policy can trigger full-media protection.

The compound rule intentionally uses a stronger gate than ordinary body-label
detection. The balanced default requires a `HIGH` or `CRITICAL` label at 0.5
confidence or above. `ARMPITS_EXPOSED`, `BELLY_EXPOSED`, `FEET_EXPOSED`, and
`MALE_BREAST_EXPOSED` are `LOW`; `BUTTOCKS_EXPOSED` is `MODERATE`; exposed
female breasts or anus are `HIGH`; and exposed genitalia are `CRITICAL`.
Consequently an ordinary child/family photo with only an armpit observation
does not activate a full-image block. `ARMPITS_EXPOSED` is also disabled for
regional censoring in the balanced default profile. The `strict` profile uses
`MODERATE`, 0.35, and enables armpit censoring.

Rule profiles can edit these values, for example:

```powershell
python safeVisionCLI.py rules set default BLOCK_IF_NSFW_AND_CHILD true
python safeVisionCLI.py rules set default PROTECTION_NSFW_MIN_RISK HIGH
python safeVisionCLI.py rules set default PROTECTION_NSFW_MIN_CONFIDENCE 0.5
python safeVisionCLI.py rules set default UNDERAGE_AGE 18
```

## Local API

The regular endpoints accept `checks=nude,age,gender`, plus `age_check`,
`gender_check`, and `nsfw_check` boolean overrides.

Dedicated demographic endpoints are also available:

```text
POST /api/v1/detect/demographics
POST /api/v1/detect/demographics/base64
GET  /api/v1/detect/demographics/url?url=https://example.com/image.jpg
```

Responses contain `checks`, `demographics`, and `protection_policy`. Videos
contain per-frame demographic results plus a sampled-observation summary.
When rendering is requested, a blocking policy applies the configured
whole-media cover. `gray`, `black`, and `color` replace every source pixel;
`blur` retains only heavily obscured visual structure. The API accepts
`full_cover`, `full_cover_mode`, `full_cover_color`,
`full_cover_show_text`, and `full_cover_message` for per-request rendering.

## Performance notes

- The ONNX session is lazy and reused.
- Multiple faces in one frame are inferred as a batch (default maximum 8).
- NSFW face boxes are reused, avoiding a second face-detection pass.
- Video API checks run only on the already selected sample frames.
- Configure ONNX providers with `--providers` or `SAFEVISION_ONNX_PROVIDERS`.

## Privacy and review requirements

- Keep the unredacted `Prosses/` reviewer copy disabled unless it is required
  by an authorized workflow.
- Do not publish demographic boxes as a censored output.
- Store analysis JSON as potentially sensitive decision data.
- Use a solid cover when the blocked output must contain no source pixels.
- Record model hash, rule file, thresholds, and human-review outcome.
- Validate the policy across representative groups and image conditions.
- Provide a correction or appeal path for consequential decisions.

For model provenance and license boundaries, read
[Models/README.md](Models/README.md). For implementation and commercial-use
questions, read [docs/LICENSING.md](docs/LICENSING.md).

<p align="right"><a href="#top">⬆️ Back to top</a></p>
