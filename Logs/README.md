<div align="center">

# 🧾 SafeVision Logs and Analysis

![Analysis](https://img.shields.io/badge/Data-Policy_Evidence-2563EB?style=for-the-badge)
![Sensitive](https://img.shields.io/badge/Privacy-Potentially_Sensitive-F59E0B?style=for-the-badge)

[Project home](../README.md) · [Protection policy](../CHILD_PROTECTION.md) · [Testing](../tests/README.md)

</div>

Image processing can write text logs and `<output>.analysis.json` files here.
The analysis record separates detector selection, observations, qualified NSFW
evidence, policy reasons, and rendering decisions.

Logs may contain:

- filenames and local paths;
- detector labels and scores;
- face boxes and estimated ages;
- model-reported gender values;
- policy thresholds and reasons;
- output filenames and copy decisions.

> [!WARNING]
> A log can remain sensitive even when it does not contain image pixels. Treat
> demographic estimates, paths, timestamps, and moderation decisions as
> protected operational data.

Before sharing a log, remove personal paths and identifiers, confirm the media
owner's expectations, and preserve enough configuration context to interpret
the decision accurately. Contents are ignored by Git; only this README is
tracked.
