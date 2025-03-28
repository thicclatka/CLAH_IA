"""
CLAH_ImageAnalysis - Suite of Python programs used for 2-photon calcium imaging analysis
"""

import tomli

try:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
        __version__ = pyproject["project"]["version"]
except Exception:
    __version__ = "unknown"
