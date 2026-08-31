import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import age_gender_detector as age_module

from age_gender_detector import (
    AgeGenderDetector,
    AgeGenderModelMissingError,
    evaluate_protection_policy,
    face_boxes_from_detections,
)
from safevision_utils import (
    apply_full_cover,
    default_blur_rules,
    full_cover_message,
    full_cover_options,
    full_cover_reason_kind,
    parse_detector_selection,
    protection_nsfw_summary,
)


class FakeSession:
    def run(self, _outputs, inputs):
        batch = next(iter(inputs.values()))
        rows = np.asarray([[16.5, 0.8], [19.0, 0.2], [35.0, 0.1]], dtype=np.float32)
        return [rows[: batch.shape[0]]]


class AgeGenderDetectorTests(unittest.TestCase):
    def detector_with_fake_session(self):
        detector = AgeGenderDetector("unused.onnx", min_face_size=16, face_padding=0, max_batch_size=8)
        detector.session = FakeSession()
        detector.input_name = "pixel_values"
        detector.output_name = "logits"
        return detector

    def test_multi_face_age_gender_and_review_results(self):
        detector = self.detector_with_fake_session()
        frame = np.full((160, 360, 3), 128, dtype=np.uint8)
        result = detector.analyze_frame(
            frame,
            face_boxes=[[10, 20, 80, 80], [130, 20, 80, 80], [250, 20, 80, 80]],
            age_enabled=True,
            gender_enabled=True,
            age_threshold=18,
            review_margin=3,
        )

        self.assertEqual(result["faces_detected"], 3)
        self.assertTrue(result["underage_detected"])
        self.assertEqual(result["underage_count"], 1)
        self.assertTrue(result["review_required"])
        self.assertEqual(result["gender_counts"], {"female": 1, "male": 2})
        self.assertEqual(result["faces"][0]["gender"], "female")
        self.assertIsNone(result["faces"][0].get("age_confidence"))

    def test_missing_model_errors_only_when_enabled_check_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            detector = AgeGenderDetector(Path(folder) / "missing.onnx")
            disabled = detector.analyze_frame(
                np.zeros((32, 32, 3), dtype=np.uint8),
                age_enabled=False,
                gender_enabled=False,
            )
            self.assertFalse(disabled["enabled"])
            with self.assertRaises(AgeGenderModelMissingError):
                detector.analyze_frame(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    face_boxes=[[0, 0, 32, 32]],
                    age_enabled=True,
                    gender_enabled=False,
                )

    def test_numpy_face_boxes_are_supported_and_deduplicated(self):
        detections = [
            {"class": "FACE_FEMALE", "score": 0.9, "box": np.asarray([1, 2, 30, 30])},
            {"class": "FACE_FEMALE", "score": 0.8, "box": np.asarray([2, 3, 30, 30])},
        ]
        boxes = face_boxes_from_detections(detections, width=100, height=100)
        self.assertEqual(len(boxes), 1)

        result = self.detector_with_fake_session().analyze_frame(
            np.full((100, 100, 3), 128, dtype=np.uint8),
            face_boxes=np.asarray([[10, 10, 60, 60]], dtype=np.int32),
        )
        self.assertEqual(result["faces_detected"], 1)
        self.assertTrue(result["face_detection"]["supplied_boxes"])

    def test_compound_child_protection_policy(self):
        demographics = {"underage_detected": True, "review_required": False}
        blocked = evaluate_protection_policy(True, demographics)
        allowed = evaluate_protection_policy(False, demographics)
        self.assertTrue(blocked["blocked"])
        self.assertEqual(blocked["verdict"], "BLOCKED")
        self.assertFalse(allowed["blocked"])

    def test_child_policy_gate_ignores_ordinary_body_context(self):
        armpits = [{"class": "ARMPITS_EXPOSED", "score": 0.99, "source": "nude"}]
        self.assertFalse(protection_nsfw_summary(armpits)["detected"])
        self.assertFalse(default_blur_rules()["ARMPITS_EXPOSED"])

        weak_explicit = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.49, "source": "nude"}]
        self.assertFalse(protection_nsfw_summary(weak_explicit)["detected"])

        explicit = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.8, "source": "nude"}]
        summary = protection_nsfw_summary(explicit)
        self.assertTrue(summary["detected"])
        self.assertEqual(summary["evidence"][0]["risk_level"], "HIGH")

        moderate = [{"class": "BUTTOCKS_EXPOSED", "score": 0.8, "source": "nude"}]
        self.assertFalse(protection_nsfw_summary(moderate)["detected"])
        strict = protection_nsfw_summary(
            moderate,
            {"PROTECTION_NSFW_MIN_RISK": "MODERATE", "PROTECTION_NSFW_MIN_CONFIDENCE": 0.35},
        )
        self.assertTrue(strict["detected"])

    def test_check_selection_supports_nsfw_alias_and_explicit_none(self):
        self.assertEqual(parse_detector_selection("nsfw,age,gender"), ["nude", "age", "gender"])
        self.assertEqual(parse_detector_selection("none"), [])

    def test_age_and_gender_fields_can_be_disabled_independently(self):
        frame = np.full((80, 80, 3), 128, dtype=np.uint8)
        gender_only = self.detector_with_fake_session().analyze_frame(
            frame,
            face_boxes=[[0, 0, 80, 80]],
            age_enabled=False,
            gender_enabled=True,
        )
        self.assertIsNone(gender_only["underage_detected"])
        self.assertIsNone(gender_only["faces"][0]["age_estimate"])
        self.assertEqual(gender_only["faces"][0]["gender"], "female")

        age_only = self.detector_with_fake_session().analyze_frame(
            frame,
            face_boxes=[[0, 0, 80, 80]],
            age_enabled=True,
            gender_enabled=False,
        )
        self.assertTrue(age_only["underage_detected"])
        self.assertIsNone(age_only["faces"][0]["gender"])
        self.assertIsNone(age_only["faces"][0]["gender_confidence"])

    def test_old_onnx_runtime_gets_actionable_compatibility_error(self):
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "model.onnx"
            model_path.touch()
            detector = AgeGenderDetector(model_path)
            with patch.object(age_module.ort, "__version__", "1.15.1"):
                with self.assertRaisesRegex(RuntimeError, "onnxruntime>=1.18"):
                    detector.load()

    def test_solid_full_cover_replaces_every_source_pixel(self):
        source = np.random.default_rng(7).integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
        options = full_cover_options(
            overrides={"FULL_COVER_MODE": "gray", "FULL_COVER_SHOW_TEXT": False}
        )
        covered = apply_full_cover(source, options, "hidden")
        self.assertTrue(np.all(covered == np.asarray([96, 96, 96], dtype=np.uint8)))
        self.assertFalse(np.array_equal(source, covered))

    def test_full_cover_message_follows_policy_reason(self):
        policy = {
            "reasons": ["NSFW_CONTENT_WITH_ESTIMATED_UNDERAGE_PERSON"],
            "nsfw_detected": True,
        }
        kind = full_cover_reason_kind(policy)
        self.assertEqual(kind, "nsfw_and_child")
        self.assertEqual(
            full_cover_message(full_cover_options(), kind),
            "Possible illegal content - review required",
        )


if __name__ == "__main__":
    unittest.main()
