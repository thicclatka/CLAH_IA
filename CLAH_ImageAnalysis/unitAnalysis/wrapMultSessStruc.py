"""
This script defines the `wrapMultSessStruc` class, which manages the creation of a multiple session segmentation structure by extending the BaseClass (BC). It processes multiple sessions, loads various data structures, and saves the combined results.

Functions:
    alt_run(): Executes an alternative run of the method, ensuring the parser works correctly.
    static_class_var_init(path, sess2process, output_folder): Initializes the static class variables for the class.
    loop_byID_bySession(): Loops through the sessions and fills the multiple session segmentation structure.
    sBYs_perID_processing(s2p_perID, ID): Processes the sessions by IDs for a specific ID.
    _fill_multSessSegStruc_overall(ID, sess_idx): Fills the multiple session segmentation structure for a specific subject ID and session index.
    _fill_DSImage(): Fills the multiple session segmentation structure with the downsampled image.
    _fill_mSSD_other(ftag): Fills the multiple session segmentation structure with other major dictionaries.
    _fill_mSSD_segDict(numSess): Fills the multiple session segmentation structure with the segmentation dictionary.
    _fill_mSSD_H5_metadata(numSess, tolerance=1e-9): Fills the metadata for the multi-session segmentation structure from an H5 file.
    print_headers(folder_anlz_str, compl_msg_fLoop, sess_num): Prints the headers for folder analysis and completion message.

Classes:
    wrapMultSessStruc: Manages the creation of a multiple session segmentation structure by extending the BaseClass (BC).

Main Execution:
    If the script is run directly, it will execute the `run_CLAH_script` function to create an instance of `wrapMultSessStruc`, run the parser, and execute the script.

Dependencies:
    - numpy: For numerical operations on arrays.
    - contextlib: For context management.
    - typing: For type hints.
    - inquirer: For prompting user input.
    - CLAH_ImageAnalysis.core: Base class and script running utilities.
    - CLAH_ImageAnalysis.tifStackFunc: Utilities for handling TIFF stack functions.
    - CLAH_ImageAnalysis.unitAnalysis: Enums for unit analysis.

Usage:
    This script is designed to be executed directly or imported as a module. When run directly, it uses the `run_CLAH_script` function to handle argument parsing and execution flow.

Example:
    To run the script directly:
    ```bash
    python wrapMultSessStruc.py --path /path/to/data --sess2process '1,2,3' --output_folder /path/to/output
    ```

    To import and use within another script:
    ```python
    from CLAH_ImageAnalysis.core import wrapMultSessStruc

    wmss = wrapMultSessStruc(path='/path/to/data', sess2process=[1, 2, 3], output_folder='/path/to/output')
    wmss.loop_byID_bySession()
    ```

Parser Arguments:
    The script uses the following parser arguments defined in `UA_enum.Parser4WMSS`:
        --path, -p: Path to the data folder (default: []).
        --sess2process, -s2p: Sessions to process (default: []), can be a list of session numbers.
        --output_folder, -out: Output folder path (default: None).
"""

from contextlib import contextmanager
import inquirer
import numpy as np
from typing import Any

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.tifStackFunc import H5_Utils
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


class wrapMultSessStruc(BC):
    def __init__(self, path: str | list, sess2process: str, output_folder: str) -> None:
        """
        Initialize the WrapMultSessStruc class.

        Parameters:
            path (str | list): The path to the data folder.
            sess2process (str): The sessions to process.
            output_folder (str): The output folder path.
        """

        self.program_name = "WMSS"
        self.class_type = "manager"
        BC.__init__(self, self.program_name, mode=self.class_type)
        # initiate H5_Utils
        self.H5_Utils = H5_Utils()

        self.static_class_var_init(path, sess2process, output_folder)

    def alt_run(self) -> None:
        """
        Executes an alternative run of the method.

        This method is used to perform an alternative run of the process. It is called `alt_run` because it doesn't follow through the usual `process_order` with respect to `sessBySess_processing`. It is specifically designed to ensure that the script iterates through the subjects and respective sessions correctly in creating the wrapMultSessStruc.

        Parameters:
            self (object): The instance of the class.
        """

        self.loop_byID_bySession()

    def static_class_var_init(self, path, sess2process, output_folder) -> None:
        """
        Initializes the static class variables for the WrapMultSessStruc class.

        Parameters:
        - path (str): The folder path.
        - sess2process (str): The selection made.
        - output_folder (str): The output folder path.
        """

        def _setExpName(output_folder: str) -> str:
            """
            Prompts the user to enter a name for the experiment and processes the input.

            Parameters:
            - output_folder (str): The initial output folder name.

            Returns:
            - str: The updated output folder name with the user-provided experiment name.
            """

            exp_name = ""
            while exp_name == "":
                input_str = self.utils.create_multiline_string(
                    [
                        "Enter experiment name or keywords (an be written w/ or w/out underscores)",
                        f"NOTE: Important keywords {self.text_lib['IMP_FILE_KW']}: ",
                    ]
                )
                exp_name = input(input_str)
                if exp_name == "":
                    print("Empty input. Please enter a valid experiment name.")
            exp_name = exp_name.replace(" ", "_")
            return f"{output_folder}_{exp_name}"

        def _choose_brain_region(output_folder: str) -> str:
            """
            Prompts the user to choose a brain region and appends it to the output folder name.

            Parameters:
            - output_folder (str): The initial output folder name.

            Returns:
            - str: The updated output folder name with the chosen brain region.
            """

            question = [
                inquirer.List(
                    "brain_region",
                    message="Choose brain region the recording to place in",
                    choices=[reg for reg in self.text_lib["brain_regions"]],
                )
            ]
            answer = inquirer.prompt(question)
            chosen_region = answer["brain_region"]
            return f"{output_folder}_{chosen_region}"

        # initiate BC static_class_var_init
        # given CSS, creates multSessIDDict, sess2process, ID_arr
        BC.static_class_var_init(
            self,
            folder_path=path,
            file_of_interest=self.text_lib["selector"]["tags"]["CSS"],
            selection_made=sess2process,
            select_by_ID=True,
        )

        # initiate keys for segDict
        self.SDkey = self.enum2dict(TSF_enum.segDict_Txt)

        # check if output_folder is given,
        # if not, result to default which is _MultSessComb
        if output_folder:
            output_folder_base = self.folder_tools.get_basename(output_folder)
            output_folder_parent = self.folder_tools.get_parent_dir(output_folder)
            if not output_folder_base.startswith("_MS_"):
                output_folder_base = f"_MS_{output_folder_base}"
            if "DG" not in output_folder_base and "CA3" not in output_folder_base:
                output_folder_base = _choose_brain_region(output_folder_base)
            output_folder = f"{output_folder_parent}/{output_folder_base}"
            self.COMB_FOLDER = output_folder
        else:
            output_folder = "_MS"
            output_folder = _setExpName(output_folder)
            output_folder = _choose_brain_region(output_folder)

            self.COMB_FOLDER = (
                f"{self.folder_tools.get_parent_dir(self.dayPath)}/{output_folder}"
            )

    def loop_byID_bySession(self) -> None:
        """
        Loops through the sessions and fills the multiple session segmentation structure.
        """

        for ID in np.unique(self.ID_arr):
            # initialize empty multSessSegStruc for each subject ID
            self.multSessSegStruc = {}

            self.print_header(self.text_lib["headers"]["start_msg_WMSS"].format(ID))

            # session to process per ID
            s2p_perID = self.sess2process[self.ID_arr == ID]

            # go through each session per ID
            # fill mSSS accordingly
            # print which subject is being processed & how it is being processed
            self.sBYs_perID_processing(
                s2p_perID=s2p_perID,
                ID=ID,
            )

            # once mSSS for ID is filled, save to comb folder
            # move back 2 directories
            # self.folder_tools.move_back_dir(levels=2)

            # create COMB_FOLDER if it doesnt exist, changes directory to folder after
            self.folder_tools.chdir_check_folder(self.COMB_FOLDER)

            # create ID_fname if it doesnt exist, change directory to folder after
            self.folder_tools.chdir_check_folder(f"{ID}_{len(s2p_perID)}Sess")
            self.saveNloadUtils.savedict2file(
                dict_to_save=self.multSessSegStruc,
                dict_name=self.dict_name["MSS"],
                filename=ID,
                file_suffix=self.dict_name["MSS"],
                date=True,
                filetype_to_save=[self.file_tag["MAT"], self.file_tag["PKL"]],
            )

            # print completion for whole script
            self.print_header(self.text_lib["completion"]["whole_proc_MSS"].format(ID))

    @contextmanager
    def print_headers(
        self, folder_anlz_str: str, compl_msg_fLoop: str, sess_num: int
    ) -> Any:
        """
        Prints the headers for folder analysis and completion message.

        Parameters:
            folder_anlz_str (str): The folder analysis string to be printed.
            compl_msg_fLoop (str): The completion message for the folder loop.
            sess_num (int): The session number.
        """
        # set & print current folder
        self.folder_path, _ = self.utils.setNprint_folder_path(
            self.dayPath, self.dayDir, sess_num
        )
        # print folder analysis string
        self.print_header(folder_anlz_str, subhead=True)
        try:
            yield
        finally:
            self.print_header(compl_msg_fLoop, subhead=True)

    def sBYs_perID_processing(self, s2p_perID: list, ID: str) -> None:
        """
        Process the sBYs (sessions by IDs) for a specific ID.

        Parameters:
            s2p_perID (list): A list of session numbers for the given ID.
            ID (str): The ID for which the sBYs are being processed.
        """

        for sess_idx, sess_num in enumerate(s2p_perID):
            sess_num = int(sess_num)
            # set & print current folder
            self.folder_path, _ = self.utils.setNprint_folder_path(
                self.dayPath, self.dayDir, sess_num
            )
            # folder analysis string
            folder_anlz_str = self.text_lib["headers"]["sess_extrct"].format(
                sess_idx + 1, len(s2p_perID), ID
            )
            # completion message for for loop
            compl_msg_fLoop = self.text_lib["completion"]["forLoop_CSS"].format(
                sess_idx + 1, len(s2p_perID), ID
            )

            with self.print_headers(folder_anlz_str, compl_msg_fLoop, sess_num):
                # fill mSSS accordingly
                self._fill_multSessSegStruc_overall(ID, sess_idx)

    def _fill_multSessSegStruc_overall(self, ID: str, sess_idx: int) -> None:
        """
        Fills the multiple session segmentation structure for a specific subject ID and session index.

        Parameters:
            ID (str): The subject ID.
            sess_idx (int): The session index.
        """
        # define numSess
        numSess = f"numSess{sess_idx}"
        self.multSessSegStruc[numSess] = {}

        # metadata for specific session
        self.multSessSegStruc[numSess]["PATH"] = self.folder_path
        self.multSessSegStruc[numSess]["DAYNAME"] = self.multSessIDDict[ID]["DATE"][
            sess_idx
        ]
        self.multSessSegStruc[numSess]["SESSNAME"] = self.multSessIDDict[ID]["TYPE"][
            sess_idx
        ]

        # loading & filling multSessSegStruc w/image
        self.multSessSegStruc[numSess]["IMG"] = self._fill_DSImage()

        # loading & filling multSessSegStruc w/major dicts created from previous scripts
        # cueShiftStruc
        self.rprint("cueShiftStruc:")
        self.multSessSegStruc[numSess][self.dict_name["CSS"]] = self._fill_mSSD_other(
            "CSS"
        )
        # treadBehDict
        self.rprint("treadBehDict:")
        self.multSessSegStruc[numSess][self.dict_name["TREADBEHDICT"]] = (
            self._fill_mSSD_other("TBD")
        )
        # segDict
        # load in segDict differently from other mats
        self.rprint("segDict:")
        self._fill_mSSD_segDict(numSess)

        if self.findLatest([self.file_tag["SD"], "prevName", self.file_tag["MAT"]]):
            self.rprint("segDict_prevNameVar:")
            self._fill_mSSD_segDict(numSess, prevNameVar=True)

        # CueCellFinder Struct
        self.rprint("CCFStruct:")
        self.multSessSegStruc[numSess][self.dict_name["CCF"]] = self._fill_mSSD_other(
            "CCF"
        )

    def _fill_DSImage(self) -> np.ndarray:
        """
        Fills the multiple session segmentation structure with the downsampled image.
        """

        # get latest DS image
        latest_DSImage = self.findLatest(
            [
                self.file_tag["DOWNSAMPLE"],
                self.file_tag["AVGCA"],
                self.file_tag["TEMPFILT"],
                self.file_tag["IMG"],
            ]
        )
        # read image
        DSImage = self.utils.image_utils.read_image(latest_DSImage)
        # normalize image
        DSImage = self.utils.image_utils.normalize_image(
            DSImage, output_dtype=np.float64
        )
        return DSImage

    def _fill_mSSD_other(self, ftag: str) -> object:
        """
        Fills the multiple session segmentation structure with other major dictionaries.

        Parameters:
            ftag (str): The file tag.

        Returns:
            object: The loaded object.
        """

        latest_file = self.findLatest([self.file_tag[ftag], self.file_tag["PKL"]])
        try:
            loaded_pkl = self.saveNloadUtils.load_file(latest_file, previous=True)
            # loaded_pkl = convert_lists_to_arrays(loaded_pkl)
        except Exception as e:
            self.rprint(f"Problem with loading {ftag}: {e}")
        return loaded_pkl

    def _fill_mSSD_segDict(self, numSess: str, prevNameVar: bool = False) -> None:
        """
        Fills the multiple session segmentation structure with the segDict.

        Parameters:
            numSess (str): The session number.
            prevNameVar (bool, optional): Whether to use the previous name variable (C instead of C_Temporal). Defaults to False.
        """
        if prevNameVar:
            fileTags2Find = [self.file_tag["SD"], "prevName", self.file_tag["MAT"]]
        else:
            fileTags2Find = [self.file_tag["SD"], self.file_tag["PKL"]]

        try:
            latest_file = self.findLatest(fileTags2Find)
        except Exception as e:
            self.rprint(f"Problem with finding latest segDict: {e}")
            self.rprint("Will try to find latest h5")
            latest_file = self.findLatest([self.file_tag["SD"], self.file_tag["H5"]])
            if not latest_file:
                self.utils.debug_utils.raiseVE_SysExit1("No usable segDict or h5 found")

        try:
            C_Temporal, A_Spatial, dx, dy = self.saveNloadUtils.load_segDict(
                latest_file,
                keep_A_sparse=True,
                C_Temporal=True,
                A_Spatial=True,
                dx=True,
                dy=True,
                prevNameVar=prevNameVar,
            )
        except Exception as e:
            print(f"Problem with loading segDict: {e}")

        if not prevNameVar:
            self.multSessSegStruc[numSess][f"{self.dict_name['SD']}Name"] = latest_file
            self.multSessSegStruc[numSess][self.SDkey["C_TEMPORAL"]] = C_Temporal
            self.multSessSegStruc[numSess][self.SDkey["A_SPATIAL"]] = A_Spatial
            self.multSessSegStruc[numSess][self.SDkey["DX"]] = dx
            self.multSessSegStruc[numSess][self.SDkey["DY"]] = dy
        else:
            self.multSessSegStruc[numSess][
                f"{self.dict_name['SD']}Name_prevNameVar"
            ] = latest_file
            self.multSessSegStruc[numSess]["C"] = C_Temporal
            self.multSessSegStruc[numSess]["A"] = A_Spatial
            self.multSessSegStruc[numSess]["d1"] = dx
            self.multSessSegStruc[numSess]["d2"] = dy

    # !THIS ISNT BEING USED FOR NOW, KEEP JUST IN CASE
    def _fill_mSSD_H5_metadata(self, numSess: str, tolerance: float = 1e-9) -> None:
        """
        Fill the metadata for the multi-session segmentation structure from an H5 file.

        Parameters:
            numSess (int): The session number.
            tolerance (float, optional): The tolerance for comparing pixel per um values. Defaults to 1e-9.
        """

        latest_file = self.findLatest(
            self.file_tag["COMP_EMCFNAME"], notInclude=self.file_tag["SD"]
        )
        self.print_wFrm(f"Found H5 file: {latest_file}")
        try:
            element_size_um = self.H5_Utils.read_file4wrapMultSess(latest_file)
        except Exception as e:
            print(f"Problem with loading h5: {e}")
        if abs(element_size_um[1] - element_size_um[-1]) > tolerance:
            self.utils.debug_utils.raiseVE_SysExit1(
                "WARNING! Pixel per um not equal in x and y. Check H5 file for any errors"
            )
        else:
            element_size_um = element_size_um[1]
        self.multSessSegStruc[numSess]["ELEMENT_SIZE_UM"] = element_size_um


if __name__ == "__main__":
    from CLAH_ImageAnalysis.unitAnalysis import UA_enum

    run_CLAH_script(wrapMultSessStruc, parser_enum=UA_enum.Parser4WMSS, alt_run=True)
