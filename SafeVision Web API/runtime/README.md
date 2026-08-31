<div align="center">

# 🧹 SafeVision API Runtime

![Uploads](https://img.shields.io/badge/Data-Uploads%20%7C%20Results%20%7C%20Temp-F59E0B?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-Contents_Ignored-10B981?style=for-the-badge&logo=git)
![Access](https://img.shields.io/badge/Access-Service_Only-EF4444?style=for-the-badge)

[Web API guide](../README.md) · [Project home](../../README.md) · [Licensing](../../docs/LICENSING.md)

</div>

The local API creates three working areas below this folder:

```text
runtime/
├── uploads/    Accepted request bodies while processing
├── outputs/    Rendered files available through /api/v1/results
└── temp/       URL downloads and intermediate media
```

Paths can be moved through `.env`:

```dotenv
SAFEVISION_API_UPLOAD_FOLDER=SafeVision Web API/runtime/uploads
SAFEVISION_API_OUTPUT_FOLDER=SafeVision Web API/runtime/outputs
SAFEVISION_API_TEMP_FOLDER=SafeVision Web API/runtime/temp
```

> [!CAUTION]
> These folders can contain private source media, rendered results, filenames,
> and temporary downloads. Git ignores their contents, but the host filesystem,
> backups, antivirus, and monitoring systems may still retain copies.

Production controls should include:

- a dedicated low-privilege service account;
- non-public filesystem permissions;
- separate volume/partition where appropriate;
- upload and URL-download size limits;
- automatic expiration through `SAFEVISION_API_MAX_FILE_AGE`;
- cleanup monitoring and disk-capacity alerts;
- no direct web-server static mapping to uploads/temp;
- encrypted storage where the risk assessment requires it;
- tested incident deletion and backup-retention procedures.

Only this README is intended for version control.
