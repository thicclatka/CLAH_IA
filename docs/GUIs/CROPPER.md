# 1Photon Cropping Utility

## Web App

A web-based utility for cropping 1-photon imaging data (.isxd files) before processing. Built with FastAPI backend and React frontend.

### Features

- Interactive cropping interface for ISXD files
- Real-time frame preview
- Directory navigation and file management
- Auto import cropping dimensions if cropping was done previously

### Output

- `crop_dims.json`: JSON file containing crop coordinates
    - Format: `[(x1, y1), (x2, y2)]`
    - Saved in the same directory as the ISXD file

### Access

The web app can be accessed at: `http://[server-address]:[port]`. Based on what is in [Application settings](https://github.com/thicclatka/CLAH_IA/blob/main/SystemdServices/app_settings.json), the default port is `8001`.
