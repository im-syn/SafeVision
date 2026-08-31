<div align="center">

# 🎭 Clean Censor Copies

![Censored](https://img.shields.io/badge/Artifact-Regional_Censor_Copy-7C3AED?style=for-the-badge)
![Boxes](https://img.shields.io/badge/Boxes-Removed-10B981?style=for-the-badge)

[Project home](../README.md) · [Image commands](../README.md#image-processing) · [Output privacy](../README.md#output-privacy)

</div>

When `--save-blur-copy` is enabled, image processing writes a separate clean
regional-censor copy here. Detection boxes are not added to this copy.

```powershell
python main.py -i ".\input\photo.jpg" -b `
  --no-boxes --save-blur-copy
```

> [!IMPORTANT]
> “Clean” means no reviewer boxes. It does not mean that every original pixel
> is hidden: regional censoring leaves areas outside matched detections visible.

If a policy block activates, SafeVision applies the selected whole-media cover
to the final output and safe copy. Solid gray, black, and color modes replace
all source pixels; full blur retains derived visual structure.

Contents are ignored by Git. Review files before sharing and delete them under
the same retention policy as other generated media.
