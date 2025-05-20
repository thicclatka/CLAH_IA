# Command Line Tools

This directory contains documentation for the various command-line interface (CLI) tools available in CLAH Image Analysis.

## CLI Categories

### [tifStackFunc](tifStackFunc.md)

Core tools for processing raw imaging data:

- Motion correction of calcium imaging movies
- Segmentation of cells using CNMF
- Generation of segmented dictionaries (segDict)
- Output includes motion-corrected movies, component evaluations, and spatial maps

### [Unit Analysis](unitAnalysis.md)

Tools for analyzing individual units and their responses (set up mostly for 2P experiments currently):

- Quick Tuning: Analysis of cell responses to behavioral cues
- Multi-session Structure Wrapper: Combines data across sessions
- Post Cell Registrar Cue Cell Finder: Identifies cells responding to specific cues

### [Cell Registration](cellRegistration.md)

Tools for registering cells across sessions:

- Cell Registrar with ROICaT: Aligns cells across multiple sessions
- Cluster Info Collater: Aggregates and analyzes registration results
