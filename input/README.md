<div align="center">

# 📥 SafeVision Input

![Private media](https://img.shields.io/badge/Contents-Private_Media-EF4444?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-Contents_Ignored-10B981?style=for-the-badge&logo=git)

[Project home](../README.md) · [Output privacy](../README.md#output-privacy) · [Responsible use](../README.md#responsible-use)

</div>

This optional folder is a convenient place for local source images and videos.
SafeVision does not require inputs to live here; `-i` can point to any readable
file.

```powershell
python main.py -i ".\input\photo.jpg" -b --no-boxes
python video.py -i ".\input\clip.mp4" --save-report
```

> [!CAUTION]
> Treat every source file as sensitive. Folder contents are ignored by Git, but
> that is not encryption or access control. Do not commit, attach, or publish
> private or illegal media.

Operational checklist:

- use synthetic or lawfully obtained fixtures for development;
- restrict filesystem access;
- remove inputs when the approved retention period ends;
- avoid meaningful personal names in filenames;
- remember that backups, shell history, and external sync tools are separate
  data flows;
- never assume that deleting a Git-ignored file removes other copies.

Only this README is intended for version control.
