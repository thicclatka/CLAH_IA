# Installation Guide

Can be used as a library on all OS's, but to properly use the package entirely, best to use on a machine running a Debian-based distribution with systemd and NVIDIA drivers.

## Prerequisites

- [Anaconda](https://docs.anaconda.com/)
    - still testing on more lightweight python environment/package managers
    - base install works with [uv](https://github.com/astral-sh/uv)
- To use with a GPU:
    - a machine running Ubuntu or Windows
        - can use Windows Subsystem for Linux (WSL)
        - would recommend [Pop!\_OS](https://system76.com/pop/download/) for NVIDIA drivers
        - use a non-Debian-based Linux distro at your own risk
    - CUDA
        - [for Ubuntu (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
        - [for Windows (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
        - [for WSL (CUDA website)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl)
        - [for Ubuntu/WSL (handy guide on github)](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)
- To run web applications as persistent background services:
    - [systemd](https://systemd.io/)
    - [TMUX](https://github.com/tmux/tmux/wiki)

## Dependencies

- [CaImAn](https://github.com/thicclatka/CaImAn)
- [ROICaT](https://github.com/RichieHakim/ROICaT)
- [scikit-cuda (required for GPU)](https://github.com/lebedov/scikit-cuda)
- [SQLJobScheduler](https://github.com/thicclatka/SQLJobScheduler)

## Installation

### Development Mode (Recommended)

```bash

# Clone CLAH Image Analysis repository
git clone https://github.com/thicclatka/CLAH_IA.git

# cd to CLAH IA directory
cd /path/to/CLAH_IA

# create conda environment
# yaml here is based on caiman's
conda env create -f environment.yml -n caiman
conda activate caiman

# Install package
# will also install dependencies listed above
pip install -e "/path/to/CLAH_IA[all]"
```

### Library Mode

```bash
# to use as a library without GPU support
pip install git+https://github.com/thicclatka/CLAH_IA

# to use as a library with GPU support
pip install git+https://github.com/thicclatka/CLAH_IA[all]
```

### Persistent web application setup

If program is running on a machine running Linux with systemd,
you can set up the web applications to run as system services.
This allows them to start automatically on boot and be managed by systemd.

To do this, navigate to the root directory of the `CLAH_IA` repository and run the [`setup.sh`](https://github.com/thicclatka/CLAH_IA/blob/main/SystemdServices/setup.sh) script:

```bash
cd /path/to/CLAH_IA
chmod +x setup.sh
./setup.sh
```

This script will typically perform the following actions:

- Create systemd service files for the web applications.
- Store files in config directory (`~/.clah_ia/`)
- Provide instructions on:
    - How to reload systemd daemon
    - Start the services
    - Enable services to start on boot up

Make sure to inspect the [SystemdServices directory in the repo](https://github.com/thicclatka/CLAH_IA/blob/main/SystemdServices) script if you want to understand the exact commands being run or if you need to customize the setup.
