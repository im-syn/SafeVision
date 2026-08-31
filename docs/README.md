<a id="top"></a>

<div align="center">
  <a href="../README.md">
    <img src="https://i.ibb.co/d4LqhX4/Safe-Vision-2.png" alt="SafeVision logo" width="540">
  </a>

  <h1>SafeVision Documentation Center</h1>

  <p><strong>Architecture, applications, rules, deployment, privacy, models, testing, and licensing.</strong></p>

  <p>
    <img alt="Guides" src="https://img.shields.io/badge/Guides-Task_Focused-2563EB?style=for-the-badge">
    <img alt="Examples" src="https://img.shields.io/badge/Examples-Copy_Ready-7C3AED?style=for-the-badge">
    <img alt="Safety" src="https://img.shields.io/badge/Safety-Policy_Aware-10B981?style=for-the-badge">
  </p>

  <p>
    <a href="../README.md">Project home</a> ·
    <a href="PROJECT_STRUCTURE.md">Project structure</a> ·
    <a href="LICENSING.md">Licensing</a> ·
    <a href="../Models/README.md">Model registry</a>
  </p>
</div>

---

## 🧭 Find the right guide

<table>
<tr>
<td width="50%" valign="top">

### 🚀 Start and process

- [Main README](../README.md) — complete installation and usage
- [Desktop GUI](../apps/desktop/README.md) — visual image/video workflow
- [Live tools](../apps/live/README.md) — camera, screen, OBS, virtual camera
- [Web API](../SafeVision%20Web%20API/README.md) — HTTP and deployment

</td>
<td width="50%" valign="top">

### 🛡️ Configure and protect

- [Rule templates](../rule_templates/README.md) — 50 policy presets
- [Child protection](../CHILD_PROTECTION.md) — evidence and decision contract
- [Settings](../settings/README.md) — persistent CLI/Screen Guard defaults
- [Output privacy](../README.md#output-privacy) — boxes and generated copies

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🧩 Build and test

- [Project structure](PROJECT_STRUCTURE.md) — folder and ownership map
- [Applications package](../apps/README.md) — compatibility launchers
- [Test guide](../tests/README.md) — automated and manual checks
- [Changelog](../CHANGELOG.md) — migration notes

</td>
<td width="50%" valign="top">

### ⚖️ License and release

- [Licensing guide](LICENSING.md) — code/model boundaries
- [Model registry](../Models/README.md) — provenance, metadata, hashes
- [Repository license](../LICENSE) — exact software terms
- [Required notices](../NOTICE) — attribution and third-party notices

</td>
</tr>
</table>

## 🗺️ Documentation principles

Every SafeVision guide should:

1. lead with a working command or decision;
2. distinguish detector evidence, policy verdict, and rendering;
3. identify whether an output can expose source pixels;
4. describe age and gender as model estimates, not facts;
5. link to model/license provenance instead of implying all artifacts share one
   license;
6. preserve stable root commands when implementation paths move;
7. use synthetic examples and never embed sensitive media;
8. state platform or dependency limits honestly.

## 🔗 Folder README map

| Folder | README purpose |
|---|---|
| `apps/` | Application organization and compatibility |
| `apps/desktop/` | Full GUI operations and development |
| `apps/live/` | Camera, Screen Guard, OBS, and performance |
| `Models/` | Model registry, hashes, metadata, and licensing |
| `rule_templates/` | 50 policy presets and rule semantics |
| `SafeVision Web API/` | Local service setup, requests, and deployment |
| `settings/` | Persistent settings schema and precedence |
| `tests/` | Regression suite and release checks |
| `input/` | Sensitive input handling |
| `output/` | Final-output interpretation |
| `Blur/` | Clean regional-censor copies |
| `Prosses/` | Restricted unredacted reviewer copies |
| `Logs/` | Analysis-record privacy |

## ✍️ Updating documentation

When behavior changes:

- update the closest folder README first;
- update the main README command/capability map;
- update `CHANGELOG.md` for release-visible changes;
- update `Models/README.md` and `LICENSING.md` for any binary or provenance
  change;
- run the link/anchor checks described in the test/release guide;
- verify copied commands against the actual `--help` output.

<p align="right"><a href="#top">⬆️ Back to top</a></p>
