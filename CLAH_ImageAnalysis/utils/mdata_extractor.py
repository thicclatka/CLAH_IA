import glob
import os

import easygui
from rich import print
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from CLAH_ImageAnalysis.utils import color_bool, findLatest, text_dict

text_lib = text_dict()
file_tag = text_lib["file_tag"]
break_lean = text_lib["breaker"]["lean_half"]


def path_selector(path: str | list = []) -> str:
    """
    Selects a directory path using a GUI dialog box.

    Parameters:
        path (list): A list containing the path. If provided, the function returns the provided path.

    Returns:
        str: The selected directory path.
    """
    if path:
        return path
    else:
        return easygui.diropenbox()


def file_selector(filename: str | list = [], default: str = "*") -> str:
    """
    Selects a file using a GUI dialog box.

    Parameters:
        filename (list): A list containing the filename. If provided, the function returns the provided filename.

    Returns:
        str: The selected file.
    """
    if filename:
        return filename
    else:
        return easygui.fileopenbox(default=default)


def mouseExtrctr(dir_to_extract: str) -> tuple[str, str, str, str, str, bool]:
    """
    Extracts metadata from the given directory path.

    Parameters:
        dir_to_extract (str): The directory path to extract metadata from.

    Returns:
        tuple: A tuple containing the extracted metadata (date, mouseid, test_type, ch, numSess).
    """

    delim = "_"
    dir_split = dir_to_extract.split(delim)
    mouseid, date, test_type, ch, numSess = [], [], [], [], []

    skip_folder = False

    if len(dir_split) == 3 or len(dir_split) == 4:
        date = dir_split[0]
        mouseid = dir_split[1]
        test_type = dir_split[-1]
        if len(dir_split) == 3:
            ch = "1ch"
        elif len(dir_split) == 4:
            ch = dir_split[2]
    elif len(dir_split) == 2:
        mouseid = dir_split[0]
        numSess = dir_split[-1]
    elif len(dir_split) == 1:
        skip_folder = True
    else:
        print(
            f"ERROR: Cannot read metadata from filename correctly. Please check {dir_to_extract}"
        )
        skip_folder = True
    return date, mouseid, test_type, ch, numSess, skip_folder


def file_finder(
    session: str, filetag: list | str, notInclude: list = []
) -> tuple[str, str]:
    """
    Finds the latest file in the given session directory that matches the specified filetag.

    Parameters:
        session (str): The session directory path.
        filetag (str): The filetag to match.
        notInclude (list): A list of filetags to exclude from the search.

    Returns:
        tuple: A tuple containing the path of the latest file and a flag indicating if a file was found ("Yes" or "No").
    """

    os.chdir(session)
    # if isinstance(filetag, str):
    last_file = findLatest(filetag, notInclude)
    # else:
    #     new_filetag = []
    #     for tag in filetag:
    #         new_filetag.append(f"*{tag}")
    #     new_filetag = "".join(new_filetag)
    #     last_file = glob.glob(new_filetag)

    if last_file:
        return last_file, "Yes"
    else:
        return last_file, "No"


def _create_metadata_dict(dayPath: str, dayDir: list) -> dict:
    """
    Create a metadata dictionary for the given dayPath and dayDir.

    Parameters:
        dayPath (str): The path of the day directory.
        dayDir (list): The list of session directories in the day directory.

    Returns:
        dict: The metadata dictionary containing information about each session.
    """

    metadata_dict = {}
    # start idx at 1 for filling in metadata_dict
    # do separate idx given potential to skip folder
    sess_idx = 1
    for _, sess in enumerate(dayDir, start=1):
        # print(break_lean)
        current_sess = f"{dayPath}/{sess}"
        # print(f"{sess_num:02}| {sess}")
        date, mouseid, test_type, ch, numSess, skip_folder = mouseExtrctr(sess)
        if not skip_folder:
            _, emc_bool = file_finder(
                current_sess, [file_tag["COMP_EMCFNAME"], file_tag["H5"]]
            )
            # _, h5_bool = file_finder(current_sess, file_tag["PRE_EMC_H5"])
            _, h5_bool = file_finder(current_sess, [file_tag["H5"], file_tag["CYCLE"]])
            _, isxd_bool = file_finder(current_sess, file_tag["ISXD"])
            _, sD_bool = file_finder(current_sess, file_tag["SD"])
            _, tdml_bool = file_finder(current_sess, file_tag["TDML"])
            _, CSS_bool = file_finder(current_sess, file_tag["CSS"])
            _, MSS_bool = file_finder(current_sess, file_tag["MSS"])
            _, CI_bool = file_finder(current_sess, file_tag["CLUSTER_INFO"])

            MSS_cond = MSS_bool == "Yes"
            h5_cond = h5_bool == "Yes"
            isxd_cond = isxd_bool == "Yes"

            metadata_dict[f"M{sess_idx}"] = {}
            metadata_dict[f"M{sess_idx}"] = {
                "Num": sess_idx,
                "Folder Name": sess,
                "Date": date,
                "Mouse": mouseid,
                "Test Type": test_type,
                "Ch": ch,
                "Sessions": int(numSess.split("Sess")[0]) if numSess else [],
                "H5 present?": color_bool(h5_bool)
                if not MSS_cond and not isxd_cond
                else [],
                "ISXD present?": color_bool(isxd_bool)
                if not MSS_cond and not h5_cond
                else [],
                "Motion Corrected?": color_bool(emc_bool) if not MSS_cond else [],
                "segDict?": color_bool(sD_bool) if not MSS_cond else [],
                "TDML?": color_bool(tdml_bool) if not MSS_cond else [],
                "cueShiftstruc?": color_bool(CSS_bool) if not MSS_cond else [],
                "MSS?": color_bool(MSS_bool) if not h5_cond else [],
                "Cluster Info?": color_bool(CI_bool) if not h5_cond else [],
            }
            sess_idx += 1
    return metadata_dict


def _createNprint_sess_per_subj_dict(metadata_dict: dict, console: Console) -> None:
    """
    Create and print a session summary per subject.

    Parameters:
        metadata_dict (dict): A dictionary containing metadata information.
        console (Console): An instance of the Console class for printing the session summary.
    """

    sess_per_subj_dict = {}
    for mid, labels in metadata_dict.items():
        mouseid = labels["Mouse"]
        if mouseid not in sess_per_subj_dict:
            sess_per_subj_dict[mouseid] = {
                "TotSess:": 0,
                "SessList": [],
                "DateBySess": [],
            }
        sess_per_subj_dict[mouseid]["TotSess:"] += 1
        sess_per_subj_dict[mouseid]["SessList"].append(labels["Num"])
        if labels["Date"]:
            sess_per_subj_dict[mouseid]["DateBySess"].append(int(labels["Date"]))

    sess_per_subj_dict = dict(sorted(sess_per_subj_dict.items()))

    for mouseid, data in sess_per_subj_dict.items():
        if len(data["SessList"]) > 1:
            make_sess_table = True
            break
        else:
            make_sess_table = False

    if make_sess_table:
        print("\n-- Session summary per subject:")
        sess_table = Table(
            show_header=True, header_style="bold magenta", box=ROUNDED, show_lines=True
        )
        sess_table.add_column("Subj")
        sess_table.add_column("Total")
        sess_table.add_column("Session List")
        sess_table.add_column("Date by Session")
        for mouseid, data in sess_per_subj_dict.items():
            row = [
                str(mouseid),
                str(data["TotSess:"]),
                str(data["SessList"]).strip("[]"),
                str(data["DateBySess"]).strip("[]"),
            ]
            sess_table.add_row(*row)
        console.print(sess_table)


def dayExtrctr(path: str | list = []) -> tuple[str, list]:
    """
    Extracts sessions from the given day/folder path.

    Parameters:
        path (list): A list containing the day/folder path. If provided, the function uses the provided path.

    Returns:
        tuple: A tuple containing the day/folder path and a list of session directories found within the day/folder.
    """

    dayPath = path_selector(path)

    os.chdir(dayPath)
    dayDir = [f for f in os.listdir() if os.path.isdir(f)]
    dayDir.sort()
    metadata_dict = {}
    print("Extracting sessions from day/folder:", dayPath)
    print("-- W/in selected day/folder found these sessions:")
    metadata_dict = _create_metadata_dict(dayPath, dayDir)

    # print results of metadata table
    console = Console()
    mdata_table = Table(
        show_header=True, header_style="bold magenta", box=ROUNDED, show_lines=True
    )
    # Add columns to the table
    for label, values in metadata_dict[next(iter(metadata_dict))].items():
        if values:
            mdata_table.add_column(label)

    # Add rows to the table
    for mid, labels in metadata_dict.items():
        row = [str(labels[column.header]) for column in mdata_table.columns]
        mdata_table.add_row(*row)

    # print results
    console.print(mdata_table)

    # create session summary per subject
    # will print if there are more than 1 session per subject
    _createNprint_sess_per_subj_dict(metadata_dict, console)

    return dayPath, dayDir
