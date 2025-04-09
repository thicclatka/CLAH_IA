from pathlib import Path
import os
# import sys


def get_directory_of_repo_from_file() -> Path:
    """
    Get the directory of the repository from the file

    Returns:
        Path: Path to the repository, which is the repo_dir, which is the parent of the CLAH_ImageAnalysis directory
    """
    return Path(__file__).parent.parent.parent


def get_directory_of_modules_from_file() -> Path:
    """
    Get the directory of the modules from the file

    Returns:
        Path: Path to the modules, which is usually at repo_dir/CLAH_ImageAnalysis
    """
    return Path(__file__).parent.parent


def get_code_dir_path(dir_name: str) -> Path:
    """
    Get the path to the code directory

    Args:
        dir_name (str): The name of the code directory to get the path to, i.e. "utils", "GUI", "decoder", etc.

    Returns:
        Path: Path to the code directory
    """

    root_dir = get_directory_of_modules_from_file()

    code_dirs = os.listdir(root_dir)

    code_dirs = [dir for dir in code_dirs if not dir.startswith(".")]

    if dir_name not in code_dirs:
        raise ValueError(f"Directory {dir_name} not found in {root_dir}")

    return Path(root_dir, dir_name)


def get_config_dir_path() -> Path:
    """
    Get the path to the config directory
    """
    home_dir = Path.home()
    config_dir = Path(home_dir, ".clah_ia")
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_path2dbs() -> Path:
    """
    Get the path to the databases

    Returns:
        Path: Path to the databases, which is usually at repo_dir/dbs
    """
    config_dir = get_config_dir_path()
    path2dbs = Path(config_dir, "dbs")
    path2dbs.mkdir(parents=True, exist_ok=True)
    return path2dbs


def get_path2NNmodels() -> Path:
    """
    Get the path to the neural network models (.pth files)

    Returns:
        Path: Path to the neural network models, which is usually at config_dir/NNmodels
    """
    config_dir = get_config_dir_path()
    path2NNmodels = Path(config_dir, "NNmodels")
    path2NNmodels.mkdir(parents=True, exist_ok=True)
    return path2NNmodels


def get_conda_prefix() -> Path:
    """
    Get the path to the conda prefix

    Returns:
        Path: Path to the conda prefix
    """
    return Path(os.environ["CONDA_PREFIX"])


def get_python_exec_path() -> Path:
    """
    Get the path to the python executable

    Returns:
        Path: Path to the python executable
    """
    conda_prefix = get_conda_prefix()
    python_exec_path = Path(conda_prefix, "bin", "python")
    return python_exec_path


def get_python_env() -> str:
    """
    Get the python environment name

    Returns:
        str: The name of the python environment
    """
    conda_prefix = get_conda_prefix()
    python_env = str(conda_prefix).split("/")[-1]
    return python_env
