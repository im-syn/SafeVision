<a id="top"></a>

<div align="center">
  <a href="../README.md">
    <img src="https://i.ibb.co/d4LqhX4/Safe-Vision-2.png" alt="SafeVision logo" width="480">
  </a>

  <h1>SafeVision Tests</h1>

  <p><strong>Focused regression coverage for demographics, policy gates, model errors, and solid covers.</strong></p>

  <p>
    <img alt="Framework" src="https://img.shields.io/badge/Framework-unittest-3776AB?style=for-the-badge&logo=python&logoColor=white">
    <img alt="Tests" src="https://img.shields.io/badge/Focused_Tests-10-22C55E?style=for-the-badge">
    <img alt="Fixtures" src="https://img.shields.io/badge/Fixtures-Synthetic-7C3AED?style=for-the-badge">
  </p>

  <p>
    <a href="../README.md">Project home</a> ·
    <a href="../CHILD_PROTECTION.md">Policy contract</a> ·
    <a href="../Models/README.md">Model registry</a>
  </p>
</div>

---

## 🚀 Run the suite

From the repository root:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Run only solid-cover tests without importing an unrelated installed `tests`
package:

```powershell
python -m unittest discover -s tests -p "test_age_gender_detector.py" `
  -k "solid_full_cover" -v
```

> [!TIP]
> Use discovery from the repository root. Importing `tests.*` directly can
> collide with a third-party package that also uses the name `tests`.

## ✅ Current coverage

| Test area | Contract protected |
|---|---|
| Multi-face batch | One model call can return several face results |
| Missing model | Error occurs only when age/gender is requested |
| NumPy face boxes | Shared NSFW boxes are accepted and deduplicated |
| Compound policy | Qualified NSFW + estimated child can block |
| Family-photo false positive | Ordinary low-risk body context does not activate the compound gate |
| Detector aliases | `nsfw`, `nude`, demographics, and explicit `none` normalize correctly |
| Independent fields | Age and gender outputs can be disabled separately |
| Runtime version | Older ONNX Runtime receives an actionable compatibility error |
| Solid cover | Gray/black/color output contains no source pixels |
| Reason message | Full-cover text follows the policy reason |

The suite uses small synthetic NumPy frames and fake inference sessions. It
does not need to load the 345 MB age/gender model.

## 🧪 Test design rules

1. Keep fixtures synthetic and safe for a public repository.
2. Test policy evidence separately from visual rendering.
3. Assert missing-model behavior for enabled and disabled checks.
4. Never make tests depend on a camera, monitor, OBS, or a private database.
5. Use deterministic arrays and scores rather than unstable real-world model
   output for unit tests.
6. Add a regression before changing a safety-sensitive threshold or gate.
7. Confirm opaque covers by comparing every output pixel, not by appearance.

## 🧭 Manual release checks

Unit tests are necessary but not sufficient. Before a release, also verify:

- `python safeVisionCLI.py status` with the intended model files;
- one safe family-style synthetic image under the balanced policy;
- one synthetic compound-policy fixture;
- public output with boxes/reviewer copy disabled;
- a short video with reports, full cover, and FFmpeg audio;
- root and package launchers for the reorganized GUI/live tools;
- API health, upload, render, and result download;
- model hashes and licensing records.

## ➕ Adding a test

Use `unittest.TestCase`, keep the filename `test_*.py`, and avoid writing into
tracked output folders. Temporary artifacts belong in `tempfile.TemporaryDirectory`.

```python
import unittest


class ExampleTests(unittest.TestCase):
    def test_behavior_is_explicit(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
```

## 🌐 Companion service tests

If `vision2` is checked out beside SafeVision:

```powershell
Push-Location "..\vision2"
python -m unittest discover -s tests -p "test_*.py" -v
Pop-Location
```

Those service-policy tests may print a local MySQL connection warning when a
development database is not running. Database integration must be tested
separately.

<p align="right"><a href="#top">⬆️ Back to top</a></p>
