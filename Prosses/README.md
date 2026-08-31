<div align="center">

# 🔍 Restricted Reviewer Copies

![Unredacted](https://img.shields.io/badge/Artifact-Unredacted-EF4444?style=for-the-badge)
![Access](https://img.shields.io/badge/Access-Authorized_Reviewers_Only-7F1D1D?style=for-the-badge)

[Project home](../README.md) · [GUI privacy](../apps/desktop/README.md#rendering-output-copies) · [Responsible use](../README.md#responsible-use)

</div>

`Prosses/` is the legacy-compatible folder for optional detection-box reviewer
copies. The spelling is retained because older SafeVision commands and user
workflows refer to it.

> [!CAUTION]
> Files here intentionally preserve original pixels and add detection labels.
> They are **not censored outputs** and must not be published as safe copies.

The copy is disabled by default. Enable it only for an access-controlled review
workflow:

```powershell
python main.py -i ".\input\photo.jpg" -b `
  --save-boxes-copy --no-boxes
```

Recommended controls:

- documented reviewer purpose and authorization;
- least-privilege filesystem access;
- encryption at rest and in transit;
- short retention and auditable deletion;
- no public issue attachments;
- separate storage from shareable results.

Contents are ignored by Git; only this warning document is tracked.
