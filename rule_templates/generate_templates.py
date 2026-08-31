"""Generate the supported SafeVision .rule preset library and its catalog."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent

NUDE_LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]
OBJECT_LABELS = [
    "cigarette", "cigar", "vape", "smoking_pipe", "joint", "alcohol_bottle",
    "beer_bottle", "wine_glass", "beer_glass", "cocktail_glass", "pill",
    "pill_bottle", "syringe", "cannabis_leaf", "drug_bag",
]
LOW_CONTEXT = ["MALE_BREAST_EXPOSED", "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEET_EXPOSED"]
COVERED = [label for label in NUDE_LABELS if "COVERED" in label]
EXPLICIT = ["FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"]
SMOKING = ["cigarette", "cigar", "vape", "smoking_pipe", "joint"]
ALCOHOL = ["alcohol_bottle", "beer_bottle", "wine_glass", "beer_glass", "cocktail_glass"]
DRUGS = ["pill", "pill_bottle", "syringe", "cannabis_leaf", "drug_bag"]


def base_rules():
    rules = {label: True for label in NUDE_LABELS + OBJECT_LABELS}
    rules["ARMPITS_EXPOSED"] = False
    rules.update(
        {
            "BLOCK_IF_NSFW_AND_CHILD": True,
            "BLOCK_IF_CHILD": False,
            "BLOCK_ON_AGE_REVIEW": False,
            "PROTECTION_NSFW_MIN_RISK": "HIGH",
            "PROTECTION_NSFW_MIN_CONFIDENCE": 0.5,
            "UNDERAGE_AGE": 18,
            "AGE_REVIEW_MARGIN": 3,
            "FULL_COVER_MODE": "blur",
            "FULL_COVER_COLOR": "96,96,96",
            "FULL_COVER_TEXT_COLOR": "255,255,255",
            "FULL_COVER_SHOW_TEXT": True,
            "FULL_COVER_BLUR_STRENGTH": 99,
            "FULL_COVER_MESSAGE_NSFW": "Explicit content hidden",
            "FULL_COVER_MESSAGE_NSFW_AND_CHILD": "Possible illegal content - review required",
            "FULL_COVER_MESSAGE_CHILD": "Estimated underage person - review required",
            "FULL_COVER_MESSAGE_REVIEW": "Age review required",
            "FULL_COVER_MESSAGE_GENERIC": "Content hidden by SafeVision policy",
        }
    )
    return rules


def flags(labels, enabled):
    return {label: enabled for label in labels}


PRESETS = [
    ("01_balanced_default", "Balanced default", "Good first choice. Ignores exposed-armpit false positives and only combines HIGH+ NSFW evidence with estimated underage faces.", {}, "nude,age,gender"),
    ("02_balanced_gray_cover", "Balanced with solid gray cover", "Balanced detection with an opaque gray result whenever a full-cover rule fires.", {"FULL_COVER_MODE": "gray"}, "nude,age,gender"),
    ("03_balanced_black_cover", "Balanced with solid black cover", "Balanced detection with a fully black result and centered reason text.", {"FULL_COVER_MODE": "black"}, "nude,age,gender"),
    ("04_balanced_custom_blue_cover", "Balanced with custom blue cover", "Example of a custom solid cover color for branded or kiosk output.", {"FULL_COVER_MODE": "color", "FULL_COVER_COLOR": "110,64,28"}, "nude,age,gender"),
    ("05_balanced_no_cover_text", "Balanced without cover text", "Creates a clean full cover without writing policy text onto the media.", {"FULL_COVER_MODE": "gray", "FULL_COVER_SHOW_TEXT": False}, "nude,age,gender"),
    ("06_strict_moderate", "Strict moderate-risk policy", "Lets MODERATE detections participate in the NSFW + underage compound rule.", {"PROTECTION_NSFW_MIN_RISK": "MODERATE", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.35}, "nude,age,gender"),
    ("07_critical_only", "Critical-only compound policy", "Only genital detections can combine with an estimated underage face to block.", {"PROTECTION_NSFW_MIN_RISK": "CRITICAL", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.55}, "nude,age,gender"),
    ("08_high_confidence", "High-confidence moderation", "Reduces false positives by requiring 75% confidence for the compound policy.", {"PROTECTION_NSFW_MIN_CONFIDENCE": 0.75}, "nude,age,gender"),
    ("09_sensitive_confidence", "Sensitive-confidence moderation", "Raises recall by accepting HIGH+ compound evidence from 30% confidence.", {"PROTECTION_NSFW_MIN_CONFIDENCE": 0.30}, "nude,age,gender"),
    ("10_armpits_included", "Strict regional armpit censoring", "Opts exposed armpits back into regional censoring; useful only where that is intentional.", {"ARMPITS_EXPOSED": True}, "nude,age,gender"),
    ("11_compound_child_standard", "NSFW plus estimated child", "Blocks only when qualified NSFW evidence and an estimated underage face occur together.", {}, "nude,age,gender"),
    ("12_compound_child_gray", "NSFW plus child, gray cover", "Compound child-protection rule with a no-source-pixels gray cover.", {"FULL_COVER_MODE": "gray"}, "nude,age,gender"),
    ("13_compound_child_black", "NSFW plus child, black cover", "Compound child-protection rule with a no-source-pixels black cover.", {"FULL_COVER_MODE": "black"}, "nude,age,gender"),
    ("14_block_any_estimated_child", "Block any estimated child", "Blocks whenever an estimated underage face is found, even when NSFW is absent.", {"BLOCK_IF_CHILD": True, "FULL_COVER_MODE": "gray"}, "age,gender"),
    ("15_block_any_child_black", "Block any estimated child, black", "Underage-only blocking with an opaque black result.", {"BLOCK_IF_CHILD": True, "FULL_COVER_MODE": "black"}, "age,gender"),
    ("16_block_age_review_band", "Block age review band", "Blocks estimates inside the near-threshold review band as well as the normal compound rule.", {"BLOCK_ON_AGE_REVIEW": True, "FULL_COVER_MODE": "gray"}, "nude,age,gender"),
    ("17_review_only_workflow", "Review-band workflow", "Turns off compound blocking and blocks only estimates near the configured threshold.", {"BLOCK_IF_NSFW_AND_CHILD": False, "BLOCK_ON_AGE_REVIEW": True, "FULL_COVER_MODE": "gray"}, "age,gender"),
    ("18_underage_threshold_16", "Estimated age threshold 16", "Example jurisdiction/workflow threshold of estimated age below 16.", {"UNDERAGE_AGE": 16, "AGE_REVIEW_MARGIN": 2}, "nude,age,gender"),
    ("19_underage_threshold_21", "Estimated age threshold 21", "Conservative adult-content access policy using an estimated age threshold of 21.", {"UNDERAGE_AGE": 21, "AGE_REVIEW_MARGIN": 3, "BLOCK_IF_CHILD": True, "FULL_COVER_MODE": "black"}, "age,gender"),
    ("20_wide_review_margin", "Wide age review margin", "Flags a five-year band above the underage threshold for human review.", {"AGE_REVIEW_MARGIN": 5, "BLOCK_ON_AGE_REVIEW": True}, "nude,age,gender"),
    ("21_explicit_regions_only", "Explicit regions only", "Censors only HIGH and CRITICAL exposed regions; common body context stays visible.", {**flags(NUDE_LABELS, False), **flags(EXPLICIT, True)}, "nude,age,gender"),
    ("22_genitals_only", "Genitals only", "Regional censoring is limited to exposed genital detections.", {**flags(NUDE_LABELS, False), "FEMALE_GENITALIA_EXPOSED": True, "MALE_GENITALIA_EXPOSED": True}, "nude"),
    ("23_breast_and_genitals", "Breast and genitals", "Regional censoring for female breast and genital detections only.", {**flags(NUDE_LABELS, False), "FEMALE_BREAST_EXPOSED": True, "FEMALE_GENITALIA_EXPOSED": True, "MALE_GENITALIA_EXPOSED": True}, "nude"),
    ("24_moderate_and_above", "Moderate and above", "Censors buttocks plus all HIGH and CRITICAL regions.", {**flags(NUDE_LABELS, False), **flags(EXPLICIT, True), "BUTTOCKS_EXPOSED": True}, "nude,age,gender"),
    ("25_no_common_body_context", "No common body-context censor", "Leaves feet, belly, male chest, and armpits alone while retaining explicit-region censoring.", {**flags(LOW_CONTEXT, False)}, "nude,age,gender"),
    ("26_covered_labels_allowed", "Covered labels allowed", "Does not regional-censor covered clothing/body-context detections.", {**flags(COVERED, False)}, "nude,age,gender"),
    ("27_all_safety_objects", "All safety objects", "Censors smoking, alcohol, and drug-object detections alongside nudity checks.", {}, "all"),
    ("28_smoking_objects_only", "Smoking objects only", "Use with --detectors objects to censor smoking-related objects while leaving alcohol/drug labels excepted.", {**flags(NUDE_LABELS, False), **flags(OBJECT_LABELS, False), **flags(SMOKING, True)}, "objects"),
    ("29_alcohol_objects_only", "Alcohol objects only", "Use with --detectors objects to censor bottles and drinking vessels.", {**flags(NUDE_LABELS, False), **flags(OBJECT_LABELS, False), **flags(ALCOHOL, True)}, "objects"),
    ("30_drug_objects_only", "Drug objects only", "Use with --detectors objects to censor configured drug and syringe labels.", {**flags(NUDE_LABELS, False), **flags(OBJECT_LABELS, False), **flags(DRUGS, True)}, "objects"),
    ("31_nudity_no_objects", "Nudity without object censoring", "All optional safety-object labels are excepted; use nude/age/gender checks only.", {**flags(OBJECT_LABELS, False)}, "nude,age,gender"),
    ("32_objects_without_nudity", "Objects without nudity censoring", "Object-moderation preset; run with the objects detector to skip the NSFW model.", {**flags(NUDE_LABELS, False)}, "objects"),
    ("33_family_photo_low_false_positive", "Family-photo low false positive", "Explicit-only regional censoring with the balanced HIGH compound child policy.", {**flags(NUDE_LABELS, False), **flags(EXPLICIT, True), "PROTECTION_NSFW_MIN_CONFIDENCE": 0.65, "FULL_COVER_MODE": "gray"}, "nude,age,gender"),
    ("34_workplace_filter", "Workplace filter", "Censors MODERATE+ nudity plus smoking/drug objects; ordinary low-risk body context is ignored.", {**flags(LOW_CONTEXT, False), **flags(ALCOHOL, False), "PROTECTION_NSFW_MIN_RISK": "MODERATE"}, "all"),
    ("35_education_filter", "Education filter", "Conservative child-aware filter with opaque gray cover and age-review blocking.", {"BLOCK_ON_AGE_REVIEW": True, "PROTECTION_NSFW_MIN_RISK": "MODERATE", "FULL_COVER_MODE": "gray"}, "all"),
    ("36_medical_human_review", "Medical human-review workflow", "Avoids low-risk escalation, keeps full blur reversible for authorized review, and widens the age review band.", {**flags(LOW_CONTEXT, False), "AGE_REVIEW_MARGIN": 5, "BLOCK_ON_AGE_REVIEW": True}, "nude,age,gender"),
    ("37_art_moderation", "Art moderation", "Uses CRITICAL-only compound escalation and explicit-region censoring to reduce overblocking in art collections.", {**flags(NUDE_LABELS, False), **flags(EXPLICIT, True), "PROTECTION_NSFW_MIN_RISK": "CRITICAL"}, "nude,age,gender"),
    ("38_social_media_balanced", "Social-media balanced", "Balanced regional censoring plus an opaque gray policy block suitable for generated previews.", {"FULL_COVER_MODE": "gray", "FULL_COVER_COLOR": "88,88,88"}, "nude,age,gender"),
    ("39_dating_app_adult_gate", "Adult-gate workflow", "Blocks any face estimated below 21 and covers the result in black.", {"UNDERAGE_AGE": 21, "BLOCK_IF_CHILD": True, "BLOCK_ON_AGE_REVIEW": True, "FULL_COVER_MODE": "black"}, "age,gender"),
    ("40_live_stream_balanced", "Live-stream balanced", "Moderate sensitivity and short review margin for live/video monitoring.", {"PROTECTION_NSFW_MIN_RISK": "MODERATE", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.45, "AGE_REVIEW_MARGIN": 2}, "nude,age,gender"),
    ("41_video_archive_review", "Video archive review", "High-confidence compound blocking with visible blur for internal archive review.", {"PROTECTION_NSFW_MIN_CONFIDENCE": 0.70, "FULL_COVER_MODE": "blur", "FULL_COVER_BLUR_STRENGTH": 151}, "nude,age,gender"),
    ("42_ci_compound_block", "CI compound-block policy", "Balanced policy intended with --fail-on-policy for automated pipelines.", {"FULL_COVER_MODE": "gray"}, "nude,age,gender"),
    ("43_ci_any_underage", "CI any-underage policy", "Intended with --fail-on-underage or BLOCK_IF_CHILD for strict upload gates.", {"BLOCK_IF_CHILD": True, "FULL_COVER_MODE": "black"}, "age"),
    ("44_manual_review_queue", "Manual review queue", "Blocks the age review band and uses a neutral gray result with clear reviewer text.", {"BLOCK_ON_AGE_REVIEW": True, "FULL_COVER_MODE": "gray", "FULL_COVER_MESSAGE_REVIEW": "Age estimate needs human review"}, "nude,age,gender"),
    ("45_zero_tolerance", "Zero-tolerance policy", "Blocks any estimated child/review result and accepts LOW-risk NSFW compound evidence.", {"BLOCK_IF_CHILD": True, "BLOCK_ON_AGE_REVIEW": True, "PROTECTION_NSFW_MIN_RISK": "LOW", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.20, "ARMPITS_EXPOSED": True, "FULL_COVER_MODE": "black"}, "all"),
    ("46_low_false_positive", "Maximum false-positive resistance", "Critical-only, high-confidence compound policy with common context and covered labels excepted.", {**flags(LOW_CONTEXT + COVERED, False), "PROTECTION_NSFW_MIN_RISK": "CRITICAL", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.85}, "nude,age,gender"),
    ("47_high_recall", "High-recall moderation", "Sensitive MODERATE+ compound gate with lower confidence; requires a review process.", {"PROTECTION_NSFW_MIN_RISK": "MODERATE", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.20, "BLOCK_ON_AGE_REVIEW": True, "ARMPITS_EXPOSED": True}, "all"),
    ("48_solid_gray_no_text", "Silent solid gray cover", "A no-source-pixels gray cover with no text overlay.", {"FULL_COVER_MODE": "gray", "FULL_COVER_SHOW_TEXT": False}, "nude,age,gender"),
    ("49_black_legal_review", "Black legal-review cover", "Opaque black output with cautious review wording for compound child-protection matches.", {"FULL_COVER_MODE": "black", "FULL_COVER_MESSAGE_NSFW_AND_CHILD": "Possible illegal content - preserve evidence and escalate for review"}, "nude,age,gender"),
    ("50_custom_brand_cover", "Custom branded cover", "Example custom-color cover and neutral moderation message; change the BGR values to match your product.", {"FULL_COVER_MODE": "color", "FULL_COVER_COLOR": "74,47,112", "FULL_COVER_TEXT_COLOR": "255,255,255", "FULL_COVER_MESSAGE_GENERIC": "Unavailable under this content policy"}, "nude,age,gender"),
]


ORDER = NUDE_LABELS + OBJECT_LABELS + [
    "BLOCK_IF_NSFW_AND_CHILD", "BLOCK_IF_CHILD", "BLOCK_ON_AGE_REVIEW",
    "PROTECTION_NSFW_MIN_RISK", "PROTECTION_NSFW_MIN_CONFIDENCE", "UNDERAGE_AGE",
    "AGE_REVIEW_MARGIN", "FULL_COVER_MODE", "FULL_COVER_COLOR",
    "FULL_COVER_TEXT_COLOR", "FULL_COVER_SHOW_TEXT", "FULL_COVER_BLUR_STRENGTH",
    "FULL_COVER_MESSAGE_NSFW", "FULL_COVER_MESSAGE_NSFW_AND_CHILD",
    "FULL_COVER_MESSAGE_CHILD", "FULL_COVER_MESSAGE_REVIEW",
    "FULL_COVER_MESSAGE_GENERIC",
]


def serialize(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_presets():
    for slug, title, purpose, overrides, detectors in PRESETS:
        rules = base_rules()
        rules.update(overrides)
        lines = [
            f"# {title}",
            f"# {purpose}",
            f"# Recommended detectors: {detectors}",
            "# Age and gender values are model estimates and require human review for high-stakes decisions.",
            "",
        ]
        for index, key in enumerate(ORDER):
            if index == len(NUDE_LABELS):
                lines.extend(["", "# Optional safety-object labels"])
            if index == len(NUDE_LABELS) + len(OBJECT_LABELS):
                lines.extend(["", "# Child-protection and full-cover policy"])
            lines.append(f"{key} = {serialize(rules[key])}")
        (ROOT / f"{slug}.rule").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_catalog():
    rows = []
    for index, (slug, title, purpose, _overrides, detectors) in enumerate(PRESETS, start=1):
        rows.append(f"| {index} | `{slug}.rule` | {title} | `{detectors}` | {purpose} |")
    readme = """# Ready-to-use rule templates

This folder contains 50 complete `BlurException.rule` presets. Start with `01_balanced_default.rule`; move to a stricter preset only after testing it against media representative of your own workload.

Use a template with an image:

```powershell
python main.py -i .\\input\\photo.jpg -b --no-boxes -e .\\rule_templates\\01_balanced_default.rule
```

Use the same template with a video:

```powershell
python video.py -i .\\input\\clip.mp4 -e .\\rule_templates\\12_compound_child_gray.rule --detectors nude,age,gender -r 10/5 --with-audio
```

Use a template through the wrapper CLI:

```powershell
python safeVisionCLI.py process .\\input\\photo.jpg --blur -e .\\rule_templates\\03_balanced_black_cover.rule
```

`true` on a detection label means the region may be censored. The `BLOCK_*` keys decide when the age estimate changes the policy verdict. `FULL_COVER_MODE` controls only the whole-media result: `blur` obscures the source, while `gray`, `black`, and `color` replace every source pixel. A solid cover is the correct choice when the underlying media must never be visible.

The templates do not turn models on by themselves. Use the recommended `--detectors` value, or set `processing.detectors` in `settings/configs.json`. Age estimates are uncertain; never use them as proof of legal age or identity.

## Template catalog

| # | File | Intended use | Recommended checks | What changes |
|---:|---|---|---|---|
""" + "\n".join(rows) + """

## Editing a preset

Copy a preset to a new filename before changing it. Keep one `KEY = value` per line. Messages do not need quotes. Colors in `.rule` files use OpenCV BGR order (`blue,green,red`); the web API and `vision2` also accept web-style `#RRGGBB` request colors.

Regenerate the shipped presets after editing this generator:

```powershell
python .\\rule_templates\\generate_templates.py
```
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    write_presets()
    write_catalog()
    print(f"Wrote {len(PRESETS)} templates and README.md to {ROOT}")
