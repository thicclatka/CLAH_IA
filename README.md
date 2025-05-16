# Calcium Imaging Analysis Pipeline for 1- and 2-Photon Microscopy

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://thicclatka.github.io/CLAH_IA)
[![Python 3.11](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![build for base install](https://github.com/thicclatka/CLAH_IA/actions/workflows/build.yml/badge.svg)](https://github.com/thicclatka/CLAH_IA/actions/workflows/build.yml)

A comprehensive suite of Python programs for 1- & 2-photon calcium imaging analysis. Best to use on a local server, with a GPU for faster processing and data stored on a drive(s) separate from the root mount.

Documentation: [https://thicclatka.github.io/CLAH_IA](https://thicclatka.github.io/CLAH_IA)

## Overview

CLAH Image Analysis provides tools for:

- 1- and 2-photon imaging analysis
- Motion correction and segmentation
- Cell registration
- Unit analysis
- Calcium imaging processing

## Quick Navigation

- [Installation Guide](docs/1_installation.md)
- [Order of Operations](docs/2_order-of-operations.md)
- [Structure for path names](docs/3_structure-folder-path-names.md)
- [CLIs](docs/CLIs/tifStackFunc.md)
- [GUIs](docs/GUIs/MOCOGUI.md)

## TODO

- [ ] Flesh out start up & set up scripts
- [ ] Add GUI/WA details to Documentation
- [ ] Change segDict_WA to FastAPI + typescript from streamlit
- [ ] Create docker file
