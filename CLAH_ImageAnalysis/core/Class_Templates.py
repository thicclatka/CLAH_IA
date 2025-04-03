"""
BaseClass and Utility Functions for CLAH Scripts

This script provides a foundational `BaseClass` and utility functions for running CLAH scripts. It includes decorators, utility functions, and a base class to facilitate the processing and analysis of data.

Functions:
- run_CLAH_script: Runs a CLAH script with the specified script class and parser enum.
- check_mode4BC: Decorator to check if the expected mode matches the current mode of the object.

Classes:
- BaseClass: A foundational class for CLAH scripts, providing essential methods and utilities for data processing and analysis.

Dependencies:
- numpy: For numerical operations.
- CLAH_ImageAnalysis.utils: Contains various utility functions for CLAH analysis.
- CLAH_ImageAnalysis.dependencies: Additional dependencies for CLAH analysis.
- rich: For enhanced printing of messages.

Usage:
1. Inherit from `BaseClass` to create custom script classes for specific CLAH analyses.
2. Use the `run_CLAH_script` function to run the script with the necessary arguments.
3. Utilize the provided methods and decorators to streamline the processing workflow.

Example:
    from CLAH_ImageAnalysis.example import MyParserEnum
    from CLAH_ImageAnalysis.core import run_CLAH_script, BaseClass

    class MyCustomScript(BaseClass):
        def __init__(self, **kwargs):
            super().__init__(program_name="MyCustomScript", mode="manager", **kwargs)

        def process_order(self):
            # Custom processing logic
            pass

    if __name__ == "__main__":
        run_CLAH_script(MyCustomScript, parser_enum=MyParserEnum)

Note: Ensure that the necessary input data structures and dependencies are correctly formatted and available before running the script.

"""

import os
import tomli
import numpy as np
from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis import dependencies
from typing import Any
from rich import print as rprint
from pathlib import Path
import functools
import sys
# from contextlib import contextmanager

# from functools import wraps


# TODO: ADD SELF.CONFIG DICT?


def get_project_version() -> str:
    """
    Read version from pyproject.toml file.

    Returns:
        str: Version string from pyproject.toml
    """
    try:
        pyproject_path = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)
        return pyproject["project"]["version"]
    except Exception as e:
        print(f"Warning: Could not read version from pyproject.toml: {e}")
        return "0.0.0"


def extract_args_preParser(
    parser_enum: Any, flag2find: str, clear_terminal: bool = True
) -> Any:
    """
    Extracts specific flag arguments from the given arguments without needing to create a parser. i.e. "--from_sql" or "--help"
    """
    return flag2find in sys.argv


def run_CLAH_script(
    script_class: Any,
    parser_enum: Any,
    alt_run: bool = False,
    clear_terminal: bool = True,
) -> None:
    """
    Runs a CLAH script.

    Args:
        script_class: The class of the script to run.
        parser_enum: The parser enum for creating arguments.
        alt_run: A boolean indicating whether to run the alternative run method.
        clear_terminal: A boolean indicating whether to clear the terminal before running the script.
    Returns:
        None
    """

    args = utils.parser_utils.createParserNextractArgs(
        parser_enum=parser_enum, clear_terminal=clear_terminal
    )
    # convert args to dictionary
    args_dict = vars(args).copy()
    # remove 'path' and 'sess2process' from the dictionary
    args_dict.pop("path", None)
    args_dict.pop("sess2process", None)

    # create an instance of the script class
    script_instance = script_class(
        path=args.path, sess2process=args.sess2process, **args_dict
    )

    # run the script
    if not alt_run:
        script_instance.run()
    else:
        script_instance.alt_run()


def check_mode4BC(expected_mode: str) -> Any:
    """
    Decorator that checks if the expected mode matches the current mode of the object.

    Args:
        expected_mode: The expected mode to be checked against.

    Returns:
        The decorated function.

    Raises:
        ModeError: If the expected mode does not match the current mode.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            self.check_mode(expected_mode)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class BaseClass:
    def __init__(
        self,
        program_name: str,
        mode: str,
        steps: bool = False,
        sess2process: list = [],
    ) -> None:
        """
        Initializes an instance of the class.

        Args:
            program_name (str): The name of the program.
            mode (str): The mode of operation. Should be either 'manager' or 'utils'.
            sess2process (list, optional): A list of sessions to process. Defaults to an empty list.

        Raises:
            ValueError: If an invalid mode is provided.

        """
        self.mode = mode
        self.manager_OR_utils = ["manager", "utils"]
        self.__version__ = get_project_version()

        if self.mode == self.manager_OR_utils[0]:
            self.program_name = program_name
            self.steps = steps
            self.sess2process = sess2process
        elif self.mode == self.manager_OR_utils[1]:
            self.program_name = f"Utils_for_{program_name}"
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Expected '{self.manager_OR_utils[0]}' or '{self.manager_OR_utils[1]}'."
            )

        # initiate utils funcs
        self.initiate_utils()

    def __repr__(self) -> str:
        """
        Return a string representation of this instance.
        """
        return f"BaseClass(program_name={self.program_name!r})"

    def check_mode(self, expected_mode: str) -> None:
        if self.mode != expected_mode:
            raise RuntimeError(
                f"The current operation is only available in '{expected_mode}' mode."
            )

    def initiate_utils(self) -> None:
        """
        Initializes the utils and other commonly used utilities.

        This method stores various utility functions and objects into the class instance for easy access.

        Args:
            None

        Returns:
            None
        """
        # store utils into self
        self.utils = utils

        # store other utils into self that are often used
        self.text_lib = utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]
        self.dict_name = self.text_lib["dict_name"]
        self.print_wFrm = utils.print_wFrame
        self.StatusPrinter = utils.ProcessStatusPrinter
        self.findLatest = utils.findLatest

        self.folder_tools = utils.folder_tools
        self.saveNloadUtils = utils.saveNloadUtils
        self.load_segDict = self.saveNloadUtils.load_segDict
        self.load_file = self.saveNloadUtils.load_file
        self.savedict2file = self.saveNloadUtils.savedict2file

        self.enum_utils = utils.enum_utils
        self.enum2dict = self.enum_utils.enum2dict
        self.color_dict = utils.color_dict()
        self.print_header = utils.print_header

        self.time_utils = utils.time_utils

        if self.mode == self.manager_OR_utils[0]:
            self.print_end_wholeProc = self.StatusPrinter.print_end_wholeProc(
                self.program_name, self.steps
            )
        if self.mode == self.manager_OR_utils[1]:
            self.image_utils = utils.image_utils
            self.fig_tools = utils.fig_tools

        # store dependencies into
        self.dep = dependencies

        # rich print
        self.rprint = rprint

    def print_done_small_proc(self, new_line: bool = True) -> None:
        """
        Prints the completion message for small processing tasks.

        Parameters:
            new_line (bool): Whether to print a new line after the completion message. Default is True.
        """
        self.rprint(self.text_lib["completion"]["small_proc"])
        if new_line:
            print()

    @check_mode4BC("manager")
    def run(self, with_wrapper: bool = True) -> None:
        """
        Run the whole process.
        """
        if with_wrapper:

            @self.print_end_wholeProc
            def run_handler():
                self.run_whole_proc()

            run_handler()
        else:
            self.run_whole_proc()

    @check_mode4BC("manager")
    def run_whole_proc(self) -> None:
        """
        Runs the whole processing procedure.

        This method executes the processing steps in the order specified by `process_order`.

        Parameters:
            None

        Returns:
            None
        """
        self.sessBysess_processing(self.process_order)
        self.post_proc_run()

    @check_mode4BC("manager")
    def process_order(self) -> Any:
        """
        Placeholder for the order of processing logic. Override this method in subclasses.
        """
        pass

    @check_mode4BC("manager")
    def endIter_funcs(self) -> Any:
        """
        Placeholder for the end of iteration logic. Override this method in subclasses.
        """
        pass

    @check_mode4BC("manager")
    def overwrite_check(self) -> Any:
        """
        Placeholder for the overwrite check logic. Override this method in subclasses.
        """
        pass

    @check_mode4BC("manager")
    def post_proc_run(self) -> Any:
        """
        Placeholder for the post processing logic. Override this method in subclasses.
        """
        pass

    @check_mode4BC("manager")
    def sessBysess_processing(self, func) -> None:
        """
        Perform session-by-session processing. Runs through all the files in the 'sess2process' list, runs the 'forLoop_var_init' method, which should initiate the variables necessarily for running in each iteration, and then runs the "endIter_funcs" which depending on the class, should generally clear any variables needed on a per iteration basis to be cleared. Further, this method prints the status of the processing at the start and end of each loop iteration.

        Args:
            sess2process (list): List of session numbers to process.

        Yields:
            None
        """
        for sess_idx, sess_num in enumerate(self.sess2process):
            self.forLoop_var_init(sess_idx, sess_num)
            with self.StatusPrinter.print_status_forLoop(
                self.program_name, sess_idx, self.sess2process, self.folder_name
            ):
                try:
                    self.overwrite_check()
                    func()
                finally:
                    self.endIter_funcs()

    @check_mode4BC("manager")
    def static_class_var_init(
        self,
        folder_path: list | str,
        file_of_interest: str,
        selection_made: list | bool,
        select_by_ID: bool = False,
        noTDML4SD: bool = False,
    ) -> None:
        """
        Initializes static class variables based on the provided inputs.

        Args:
            folder_path (str): The path to the folder.
            file_tag (str): The file tag.
            selection_made (bool): Indicates whether a selection has been made.
            select_by_ID (bool): Indicates whether to select by ID.
            noTDML4SD (bool): Indicates whether to exclude TDML from the selection specifically when selecting for SD (segDict).

        Returns:
            None
        """
        PFS = "path_fold_sel"

        self.print_header(self.text_lib["headers"][PFS])

        self.print_header(self.text_lib["steps"][PFS]["s1"], subhead=True)

        if not folder_path:
            path_strings = self.utils.create_multiline_string(
                [
                    self.text_lib["steps"][PFS]["no_path1"],
                    self.text_lib["steps"][PFS]["no_path2"],
                ]
            )
            self.rprint(path_strings)

        # if folder_path is empty, expects user to select path
        # will show what is inside the selected path
        self.dayPath, self.dayDir = self.utils.dayExtrctr(folder_path)

        self.print_header(self.text_lib["steps"][PFS]["s2"], subhead=True)
        # self.rprint(self.text_lib["steps"][PFS][f"note_{self.program_name}"])

        selector_result, self.multSessIDDict = self.utils.subj_selector(
            dayPath=self.dayPath,
            dayDir=self.dayDir,
            file_of_interest=file_of_interest,
            selection_made=selection_made,
            select_by_ID=select_by_ID,
            noTDML4SD=noTDML4SD,
        )

        if self.multSessIDDict is None:
            self.sess2process = selector_result
        else:
            # for CSS or select by ID option, we need to store the ID and the session index
            # to get sess2process
            self.ID_arr = []
            self.sess2process = []
            for ID in self.multSessIDDict.keys():
                for idx in self.multSessIDDict[ID]["IDX"]:
                    self.ID_arr.append(ID)
                    self.sess2process.append(idx)
            self.sess2process = np.array(self.sess2process)
            self.ID_arr = np.array(self.ID_arr)

    @check_mode4BC("manager")
    def static_class_var_init_non_select(self, folder_path: list | str) -> None:
        """
        Initializes the static class variables 'dayDir' and 'dayPath' using the 'utils.dayExtrctr' function.

        Parameters:
        - folder_path (str): The path to the folder.

        Returns:
        None
        """
        self.dayDir, self.dayPath = self.utils.dayExtrctr(folder_path)

    @check_mode4BC("manager")
    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        """
        Initialize variables for the loop. Add to this method in subclasses. Always call super().forLoop_var_init(). By default, will store the session index and number, set the folder path and name, and the subjected ID.

        Parameters:
        - sess_idx (int): The index of the session.
        - sess_num (int): The total number of sessions.

        Returns:
        None
        """
        # store into self
        self.sess_idx = sess_idx
        self.sess_num = sess_num

        # set folder name & path
        self.folder_path, self.folder_name = self.utils.setNprint_folder_path(
            self.dayPath, self.dayDir, self.sess_num
        )

        if isinstance(self.folder_name, list):
            split_fname = self.folder_name[0].split("_")
        else:
            split_fname = self.folder_name.split("_")
        if len(split_fname) == 2:
            self.ID = split_fname[0]
        elif len(split_fname) == 3:
            self.date = split_fname[0]
            self.ID = split_fname[1]
            self.etype = split_fname[2]

    @check_mode4BC("manager")
    def create_conCat_outputFolder(self) -> None:
        print(
            "Given concatenation parameter being set to True will check/create concat output folder:"
        )
        path = self.folder_path[0]
        parts = path.split(os.sep)

        #! USING A & B FOR PRE VS POST FOR NOW, TEMPORARY FOR NOW
        if "A-" in self.etype:
            etype2use = self.etype.replace("A-", "-")
        else:
            etype2use = self.etype.replace("-001", "")

        if len(parts) > 2:
            self.output_path = os.sep.join(parts[:-2])
            self.output_path = os.path.join(self.output_path, f"{parts[-2]}_Concat")
        self.folder_tools.create_folder(self.output_path)

        self.output_pathByID = os.path.join(
            self.output_path, f"{self.date}_{self.ID}_{etype2use}"
        )
        self.folder_tools.create_folder(self.output_pathByID)
        self.print_done_small_proc()

    @check_mode4BC("manager")
    def group_sessions_by_id4concat(self) -> list:
        """
        Groups sessions by their IDs if the ID appears exactly twice.

        This method processes two lists: `self.ID_arr` and `self.sess2process`.
        It counts the occurrences of each ID in `self.ID_arr` and groups the
        corresponding sessions from `self.sess2process` if an ID appears exactly
        twice. The result is a list of tuples, where each tuple contains two
        sessions corresponding to an ID that appears twice.

        Returns:
            list of tuple: A list of tuples, where each tuple contains two
            sessions corresponding to an ID that appears exactly twice.
        """
        from collections import defaultdict

        id_count = defaultdict(int)
        for id in self.ID_arr:
            id_count[id] += 1

        grouped_sessions = []
        id_to_sessions = defaultdict(list)
        for id, sess in zip(self.ID_arr, self.sess2process):
            id_to_sessions[id].append(sess)

        for id, sess in id_to_sessions.items():
            count = id_count[id]
            if count == 2:
                grouped_sessions.append(tuple(id_to_sessions[id]))
            elif count % 2 == 0:
                # if count is even &  > 2, group in pairs
                for i in range(0, count, 2):
                    grouped_sessions.append(tuple(sess[i : i + 2]))
            elif count % 2 == 1:
                # same logic as above, but ignore the last session
                for i in range(0, count - 1, 2):
                    grouped_sessions.append(tuple(sess[i : i + 2]))

        return grouped_sessions

    @check_mode4BC("manager")
    def enable_fig_tools(self) -> None:
        """
        Enables the figure tools for plotting and visualization. Needed if not using a helper class to plot for a script.
        """
        self.fig_tools = utils.fig_tools

    @check_mode4BC("manager")
    def is_var_none_check(self, var2check: Any, replacement_var: Any) -> Any:
        """
        Checks if a variable is None and returns either the original variable or a replacement variable.

        Parameters:
        var2check (any): The variable to check for None.
        replacement_var (any): The replacement variable to use if var2check is None.

        Returns:
        any: The original variable if it is not None, otherwise the replacement variable.
        """
        var = replacement_var if var2check is None else var2check
        return var

    @check_mode4BC("manager")
    def create_quickLinks_folder(self, fpath: str | None = None) -> None:
        """
        Creates a hidden quickLinks folder.

        Parameters:
        - fpath (str | None, optional): Base folder path where the quickLinks folder will be created.
                                        If None, uses the default quickLinks folder path. Defaults to None.

        Returns:
        None
        """
        if fpath is None:
            fpath = self.text_lib["Folders"]["QL"]
        else:
            fpath = os.path.join(fpath, self.text_lib["Folders"]["QL"])
        self.utils.create_folder(fpath, verbose=False)

    @check_mode4BC("manager")
    def create_symlink4QL(
        self,
        src: str,
        link_name: str,
        fpath4link: str | None = None,
    ) -> None:
        """
        Create a symbolic link to a file in the quickLinks folder.

        Parameters:
            src (str): Source file path to create symbolic link from.
            link_name (str): Name of the symbolic link to create.
            fpath4link (str | None, optional): Base folder path where the quickLinks folder and symbolic link will be created.
                                                If None, uses the default quickLinks folder path. Defaults to None.

        The symbolic link will be created in a quickLinks subfolder, either at the default location or under the specified fpath4link.
        The link_name parameter should be just the name of the link, not the full path.
        """
        if fpath4link is None:
            fpath4link = self.text_lib["Folders"]["QL"]
        else:
            fpath4link = os.path.join(fpath4link, self.text_lib["Folders"]["QL"])

        link_name = os.path.join(fpath4link, link_name)
        self.folder_tools.create_symlink(src, link_name)

    @check_mode4BC("manager")
    def create_FullFolderPath4file(self, fname: str) -> str:
        """
        Create a full folder path for a file.

        Parameters:
        - fname (str): The file name.

        Returns:
        - str: The full folder path for the file. If the file name contains a '/', this implies the file name already includes the folder path and will return the file name as is.
        """
        if "/" not in fname:
            return os.path.join(self.folder_path, fname)
        else:
            return fname
