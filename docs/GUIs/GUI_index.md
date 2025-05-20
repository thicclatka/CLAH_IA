# GUI Tools

This directory contains documentation for the various GUI tools available in CLAH Image Analysis.

## Available GUIs

### [Motion Correction GUI](MOCOGUI.md)

A graphical interface for motion correction of calcium imaging data. Provides tools for:

- Motion correction parameter adjustment
- Real-time preview of correction results
- Batch processing capabilities

### [SegDict Utility](SDGUI.md)

A web-based interface for viewing and evaluating segments from the segDict. Features include:

- Component evaluation and classification
- Spatial and temporal visualization
- Neural network integration
- Export capabilities

## Access

All web-based GUIs can be accessed through your server at their respective ports. Default ports are configured in [Application settings](https://github.com/thicclatka/CLAH_IA/blob/main/SystemdServices/app_settings.json).

## Requirements

- System service setup (via systemd)
- Available ports for web access
- GPU recommended for optimal performance
