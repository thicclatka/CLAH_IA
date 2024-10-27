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
        "inquirer",
        "libsvm",
        "lxml",
        "plotly",
        "plotly",
        "prettytable",
        "rasterio",
        "rich",
        "scikit-learn",
        "seaborn",
        "tqdm",
        "ttkthemes",
    ],
)
