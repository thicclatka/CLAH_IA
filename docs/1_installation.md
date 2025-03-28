# Installation Guide

## Prerequisites

- [Anaconda](https://docs.anaconda.com/)
    - still testing on more lightweight python environment/package managers
- [TMUX](https://github.com/tmux/tmux/wiki)
- To use a GPU:
    - a machine running Ubuntu or Windows
        - can use Windows Subsystem for Linux (WSL)
        - would recommend [Pop!_OS](https://system76.com/pop/download/) for NVIDIA drivers
        - use a non-Debian-based Linux distro at your own risk
    - CUDA (required if using GPU)
        - [for Ubuntu (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
        - [for Windows (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
        - [for WSL (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl)
        - [for Ubuntu/WSL (handy guide on github)](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

## Dependencies

- [CaImAn](https://github.com/thicclatka/CaImAn)
- [ROICaT](https://github.com/RichieHakim/ROICaT)
- [scikit-cuda (required if using GPU)](https://github.com/lebedov/scikit-cuda)
- [SQLJobScheduler](https://github.com/thicclatka/SQLJobScheduler)

## Installation

### Development Mode (Recommended)

```bash
# Clone CLAH Image Analysis repository
git clone git@github.com:thicclatka/CLAH_IA.git

# cd to CLAH IA directory
cd /path/to/CLAH_IA

# create conda environment
conda env create -f environment.yml -n caiman
conda activate caiman

# Install package
pip install -e "/path/to/CLAH_IA[all]"
```

### Library Mode

```bash
# to use as a library without GPU support
pip install git+https://github.com/thicclatka/CLAH_IA

# to use as a library with GPU support
pip install git+https://github.com/thicclatka/CLAH_IA[all]
```
