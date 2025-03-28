# Installation Guide

## Prerequisites

- a machine running Ubuntu or Windows Subsystem for Linux (WSL)
    - use MacOS or a non-Debian-based Linux distro at your own risk
- [Anaconda](https://docs.anaconda.com/)
    - still testing on more lightweight python environment/package managers
- [TMUX](https://github.com/tmux/tmux/wiki)
- CUDA (required if using GPU)
    - [for Ubuntu (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
    - [for WSL (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl)
    - [for Ubuntu/WSL (handy guide on github)](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

## Dependencies

- [CaImAn](https://github.com/thicclatka/CaImAn)
- [ROICaT](https://github.com/RichieHakim/ROICaT)
- [scikit-cuda (required if using GPU)](https://github.com/lebedov/scikit-cuda)
- [SQLJobScheduler](https://github.com/thicclatka/SQLJobScheduler)

## Step-by-Step Installation

### 1. Install CaImAn

```bash
# Clone CaImAn repository (using CLAH fork)
git clone git@github.com:thicclatka/CaImAn.git

# Navigate to CaImAn directory
cd /path/to/CaImAn/

# Create and activate conda environment
conda env create -f environment.yml -n caiman
conda activate caiman

# Install CaImAn
pip install -e .  # Use -e for editable mode, but this is optional
```

### 2. Install [scikit-cuda](https://github.com/lebedov/scikit-cuda) (needed for GPU, otherwise optional)

```bash
conda activate caiman

# install version on github, not pypi
pip install git+https://github.com/lebedov/scikit-cuda
```

### 3. Install [ROICaT](https://github.com/RichieHakim/ROICaT)

```bash
conda activate caiman
pip install roicat[all]==1.3.6
pip install git+https://github.com/RichieHakim/roiextractors
```

### 4. Install [SQLJobScheduler](https://github.com/thicclatka/SQLJobScheduler) (optional)

```bash
conda activate caiman
pip install git+https://github.com/thicclatka/SQLJobScheduler
```

### 5. Install CLAH Image Analysis

```bash
conda activate caiman

# to actively develop
# Clone CLAH Image Analysis repository
git clone https://github.com/thicclatka/CLAH_IA.git

# Install package
pip install -e /path/to/CLAH_IA  # -e is optional here as well

# to use as a library
pip install git+https://github.com/thicclatka/CLAH_IA
```
