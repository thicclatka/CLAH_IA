import grp
import os
import pwd
import shutil
import stat
from typing import Literal

import easygui

from CLAH_ImageAnalysis.utils import print_wFrame, text_dict


def move_back_dir(levels: int = 1) -> None:
    """
    Move back a specified number of directories.

    Parameters:
        levels (int): The number of directories to move back. Default is 1.
    """

    os.chdir("../" * levels)


def find_eligible_folders_w_ftag(ftag: str, dir_of_session_folders: str) -> list[str]:
    """
    Find all folders with a given file tag in the current directory.
    """
    if os.path.exists(dir_of_session_folders):
        os.chdir(dir_of_session_folders)
    else:
        print(f"Directory {dir_of_session_folders} does not exist.")
        raise FileNotFoundError(f"Directory {dir_of_session_folders} does not exist.")

    folder_contents = os.listdir(dir_of_session_folders)
    foldersWftag = []
    for folder in folder_contents:
        if os.path.isdir(os.path.join(dir_of_session_folders, folder)):
            if _find_ftag(ftag, folder):
                foldersWftag.append(os.path.abspath(folder))

    foldersWftag.sort()
    return foldersWftag


def _find_ftag(ftag: str, folder: str = None) -> str:
    """Find the most recent file with the specified file tag in the specified folder.

    Args:
        folder (str, optional): Path to folder to search. If None, searches current directory.

    Returns:
        str: Path to most recent matching CSV file.
    """
    if folder is not None:
        folder = os.path.abspath(folder)
        os.chdir(folder)

    ftag_file = findLatest(ftag)

    if folder is not None:
        os.chdir("..")

    return ftag_file


def check_folder_path(
    fpath: str | None | list, msg: str | None = None
) -> str | list[str]:
    """
    Check if the folder path is valid and if not, open a dialog to select a folder.

    Parameters:
        fpath (str | None | list): The folder path to check. Fpath can be a single string, a list with no elements, or None.
        msg (str, optional): The message to display in the dialog. Default is "Select folder".
        multiple_folders (bool, optional): If True, allow multiple folders to be selected. Default is False.
    Returns:
        str: The folder path.
    """
    if msg is None:
        msg = "Select folder"
    if fpath is None or fpath == []:
        fpath = easygui.diropenbox(msg=msg)
    return fpath


def os_splitterNcheck(
    file_path: str, baseORpath: Literal["base", "path"]
) -> str | None:
    """
    Split the given file path and return either the base or the path component. If the path component is empty, suggest file_path is a folder or file in the current directory and returns the input unchanged.

    Parameters:
        file_path (str): The file path to split.
        baseORpath (Literal["base", "path"]): Specifies whether to return the base or the path component.

    Returns:
        str: The base component of the file path if `baseORpath` is "base", or the path component if `baseORpath` is "path".
    """

    if os.path.split(file_path)[0] == "":
        return file_path
    elif baseORpath == "base":
        return os.path.split(file_path)[1]
    elif baseORpath == "path":
        return os.path.split(file_path)[0]


def create_folder(folder_path: str, verbose: bool = True) -> None:
    """
    Create a folder at the specified path if it doesn't already exist.

    Parameters:
        folder_path (str): The path of the folder to be created.
        verbose (bool): Whether to print a message when the folder is created or already exists. Default is True.

    Returns:
        None
    """
    if not check_folder(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.") if verbose else None
    else:
        print(f"Folder '{folder_path}' already exists.") if verbose else None
    return


def delete_folder(folder_path: str) -> None:
    """
    Delete a folder at the specified path.

    Parameters:
        folder_path (str): The path of the folder to be deleted.
    """

    if check_folder(folder_path):
        shutil.rmtree(folder_path)
        print_wFrame(f"Deleted {folder_path} folder")


def check_folder(folder_path: str) -> bool:
    """
    Check if the specified folder exists.

    Parameters:
        folder_path (str): The path of the folder to be checked.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    return os.path.exists(folder_path)


def chdir_check_folder(
    folder_path: str, create: bool = True, verbose: bool = True
) -> None:
    """
    Change the current working directory to the specified folder path.
    If the folder doesn't exist, create it.

    Parameters:
        folder_path (str): The path of the folder to be checked and changed to.
        create (bool): Whether to create the folder if it doesn't exist. Default is True.
        verbose (bool): Whether to print a message when the folder is created or already exists. Default is True.
    """

    if create:
        create_folder(folder_path, verbose=verbose)
    else:
        if not check_folder(folder_path):
            folder_path = easygui.diropenbox()
    os.chdir(folder_path)


def findLatest(
    filetags: str | list[str],
    notInclude: list[str] = [],
    path2check: str | None = None,
    full_path: bool = False,
) -> str | None:
    """
    Find the latest file in the current directory that matches any of the given filetags.
    Exclude files that contain any of the specified strings in the notInclude list.

    Parameters:
        filetags (str | list[str]): The tag(s) to match in the file names.
        notInclude (list[str], optional): List of strings to exclude from the search. Default is [].
        full_path (bool, optional): If True, return the full path of the latest file. Default is False.

    Returns:
        str: The path of the latest file that matches any of the filetags, or an empty string if no match is found.
    """

    # Convert filetags to a list if it's a single string
    if isinstance(filetags, str):
        filetags = [filetags]

    # Ensure notInclude is a list
    if isinstance(notInclude, str):
        notInclude = [notInclude]

    if path2check is None:
        path2use = os.getcwd()
    else:
        path2use = path2check

    currDir = os.listdir(path2use)
    filetagFinder = [
        match
        for match in currDir
        if all(filetag in match for filetag in filetags)
        and all(exclude not in match for exclude in notInclude)
    ]

    if path2check is not None:
        filetagFinder = [os.path.join(path2check, file) for file in filetagFinder]

    if filetagFinder:
        latest_file = max(filetagFinder, key=os.path.getctime)
    else:
        latest_file = ""

    if full_path and latest_file:
        latest_file = os.path.abspath(latest_file)

    return latest_file


def setNprint_folder_path(
    dayPath: str, dayDir: list[str], sess_num: int | tuple[int, ...]
) -> tuple[str | list[str], str | list[str]]:
    """
    Set the current working directory to the specified folder path and return the folder path.

    Parameters:
        dayPath (str): The path of the day.
        dayDir (list[str]): List of directory names for each session.
        sess_num (int | tuple[int, ...]): The session number or a tuple of session numbers.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - The folder paths.
            - The folder names.
    """

    folder_path = []
    folder_name = []
    if isinstance(sess_num, int):
        folder_path = f"{dayPath}/{dayDir[sess_num - 1]}"
        os.chdir(folder_path)
        folder_name = dayDir[sess_num - 1]
    elif isinstance(sess_num, tuple):
        for idx, sess in enumerate(sess_num):
            folder_path.append(f"{dayPath}/{dayDir[sess - 1]}")
            # os.chdir(folder_path[idx])
            folder_name.append(dayDir[sess - 1])
            #! NOTE YOU WILL BE IN PARENT DIRECTORY NOT OS.CHDIR INTO SESSION DIRECTORY
    return folder_path, folder_name


def basename_finder(filename: str, file_tag: str) -> str | None:
    """
    Finds the basename of a filename by splitting it at the specified file tag.

    Parameters:
        filename (str): The full filename.
        file_tag (str): The tag used to split the filename.

    Returns:
        str | None: The basename of the filename.
    """
    return filename.split(file_tag)[0]


def get_basename(file_path: str) -> str:
    """
    Get the basename of a file path.

    Parameters:
        file_path (str): The file path to get the basename of.

    Returns:
        str: The basename of the file path.
    """
    return os.path.basename(file_path)


def get_parent_dir(file_path: str) -> str:
    """
    Get the parent directory of a file path.

    Parameters:
        file_path (str): The file path to get the parent directory of.

    Returns:
        str: The parent directory of the file path.
    """

    return os.path.dirname(file_path)


def file_checker_via_tag(
    file_tag: str | list[str], file_end: str = "PKL", full_path: bool = False
) -> list[str]:
    """
    Check if a file exists for each tag and return a list of the files.

    Parameters:
        file_tag (str | list[str]): The tag(s) to match in the file names.
        file_end (str, optional): The file extension to match. Default is "PKL".
        full_path (bool, optional): If True, return the full path of the files. Default is False.

    Returns:
        list[str]: A list of the files that exist for each tag.
    """

    if isinstance(file_tag, str):
        file_tag = [file_tag]

    # Use list comprehension to find the latest file for each tag
    file_list = [findLatest([ftag, file_end], full_path=full_path) for ftag in file_tag]

    # Create a list of files that are not None or False
    eligible_file_list = [file for file in file_list if file]

    return file_list, eligible_file_list


def delete_pkl_then_mat(pkl_fname: str | list[str]) -> None:
    """
    Delete a PKL file and its corresponding MAT file.

    Parameters:
        pkl_fname (str | list[str]): The PKL file name or a list of PKL file names.
    """

    file_tag = text_dict()["file_tag"]
    if isinstance(pkl_fname, str):
        pkl_fname = [pkl_fname]

    for pkl2delete in pkl_fname:
        if check_folder(pkl2delete):
            # remove pkl first
            os.remove(pkl2delete)
            print_wFrame(f"Deleted {pkl2delete}")

            mat = pkl2delete.replace(file_tag["PKL"], file_tag["MAT"])
            if os.path.exists(mat):
                os.remove(mat)

        else:
            print_wFrame(f"{pkl2delete} does not exist.")


def print_folder_contents(
    file_list: str | list[str], new_line: bool = True, pre_file_msg: str | None = None
) -> None:
    """
    Print the contents of a folder.

    Parameters:
        file_list (str | list[str]): The file or list of files to print.
        new_line (bool, optional): Whether to print a new line after printing the files. Default is True.
        pre_file_msg (str, optional): The message to print before each file. Default is None.
    """

    if isinstance(file_list, str):
        file_list = [file_list]
    for file2print in file_list:
        if check_folder(file2print):
            msg2print = (
                file2print if pre_file_msg is None else f"{pre_file_msg} {file2print}"
            )
            print_wFrame(f"{msg2print}")
    if new_line:
        print()


def fileNfolderChecker(file: str) -> bool:
    """
    Check if a file exists and is a folder.
    """

    return file and check_folder(file)


def remove_file(file: str) -> None:
    """
    Remove a file if it exists and is a folder.
    """
    if fileNfolderChecker(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)


def create_symlink(src: str, link_name: str) -> None:
    """
    Create a symbolic link to a file.
    """
    try:
        # Remove existing symlink if it exists
        if os.path.islink(link_name):
            os.unlink(link_name)

        # Create the symlink
        os.symlink(src, link_name)

    except FileExistsError:
        print(f"Link already exists: {link_name}")
    except OSError as e:
        print(f"Error creating symlink: {e}")


def get_dirname(file_path: str) -> str | None:
    """
    Get the directory name of a file path.
    """
    if "/" in file_path:
        return os.path.dirname(file_path)
    else:
        return None


def check_path_permissions(path):
    # Split path into components
    parts = path.split("/")
    current_path = "/"

    print("Checking permissions along path:")
    for part in parts:
        if part:  # Skip empty parts
            current_path = os.path.join(current_path, part)
            try:
                stat_info = os.stat(current_path)
                owner = pwd.getpwuid(stat_info.st_uid).pw_name
                group = grp.getgrgid(stat_info.st_gid).gr_name
                mode = stat.filemode(stat_info.st_mode)

                print(f"\nPath: {current_path}")
                print(f"Owner: {owner}")
                print(f"Group: {group}")
                print(f"Mode: {mode}")
                print(f"Current process can read: {os.access(current_path, os.R_OK)}")
                print(f"Current process can write: {os.access(current_path, os.W_OK)}")
                print(
                    f"Current process can execute: {os.access(current_path, os.X_OK)}"
                )
            except Exception as e:
                print(f"\nError checking {current_path}: {e}")


def print_permissions_info():
    current_user = pwd.getpwuid(os.getuid()).pw_name
    current_groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]

    print(f"Current user: {current_user}")
    print(f"Current groups: {current_groups}")
    print(f"Process UID: {os.getuid()}")
    print(f"Process GID: {os.getgid()}")
