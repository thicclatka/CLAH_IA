from setuptools import find_packages
from setuptools import setup

setup(
    name="CLAH_ImageAnalysis",
    author="Alex Hurowitz",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "SimpleITK",
        "aenum",
        "easygui",
        "h5py",
        "inquirer",
        "libsvm",
        "lxml",
        "pandas",
        "plotly",
        "prettytable",
        "rasterio",
        "rich",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "seaborn",
        "tqdm",
        "ttkthemes",
    ],
)
