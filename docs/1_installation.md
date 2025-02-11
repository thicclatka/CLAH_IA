# Installation Guide

## Prerequisites

- [Anaconda](https://docs.anaconda.com/)

## Dependencies

- [CaImAn](https://github.com/thicclatka/CaImAn)
- [ROICaT](https://github.com/RichieHakim/ROICaT)

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

### 2. Install CLAH Image Analysis

```bash
# Clone CLAH Image Analysis repository
git clone git@github.com:thicclatka/CLAH_IA.git

# Install package
pip install -e /path/to/CLAH_IA  # -e is optional here as well
```

### 3. Install ROICaT

With the caiman environment activated:

```bash
conda activate caiman
pip install roicat[all]==1.3.3
pip install git+https://github.com/RichieHakim/roiextractors
```
