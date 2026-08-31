<div align="center">

# 📤 SafeVision Final Output

![Final result](https://img.shields.io/badge/Purpose-Requested_Final_Result-2563EB?style=for-the-badge)
![Review first](https://img.shields.io/badge/Publish-Review_First-F59E0B?style=for-the-badge)

[Project home](../README.md) · [Rendering guide](../README.md#full-cover-reference) · [Output privacy](../README.md#output-privacy)

</div>

`main.py` writes the explicitly requested final image here when no other output
path is supplied. Generated contents are ignored by Git.

An item in this folder is not automatically safe to publish. Its visibility
depends on the command:

| Result | Source visibility |
|---|---|
| Boxes only | Original pixels remain visible |
| Regional blur/mask | Original remains outside protected regions |
| Full blur | Derived colors and broad shapes remain |
| Solid gray/black/color cover | Source pixels are replaced |

For public output, use `--no-boxes --no-save-boxes-copy`, inspect the analysis
JSON, and prefer a solid full cover when policy blocks the item.

```powershell
python main.py -i ".\input\photo.jpg" `
  -o ".\output\photo_checked.jpg" `
  -b --no-boxes --no-save-boxes-copy `
  --full-cover-mode gray
```

Only this README is intended for version control.
