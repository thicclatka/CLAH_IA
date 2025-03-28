import os

import numpy as np
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_enum
from CLAH_ImageAnalysis.unitAnalysis import CueCellFinder
from CLAH_ImageAnalysis.unitAnalysis import pks_utils
from CLAH_ImageAnalysis.unitAnalysis import QT_Plotters
from CLAH_ImageAnalysis.unitAnalysis import UA_enum
from CLAH_ImageAnalysis.unitAnalysis import wrapCueShiftTuning

# from CLAH_ImageAnalysis.unitAnalysis import RidgeWalker

######################################################
#  QT_manager class funcs
######################################################


class QT_manager(BC):
    """
    The QT_manager class is responsible for managing the quick tuning analysis of unit data. It extends the BaseClass (BC) and provides various methods for initializing, processing, and analyzing data, including handling segmentation, creating peak dictionaries, and managing cue shift tuning.

    Attributes:
        program_name (str): The name of the program.
        path (list): The path to the folder containing the data.
        sess2process (list): List of sessions to process.
        frameWindow (int): Window size (in frames) used for smoothing for the event/peak detection. Default is 15
        sdThresh (float): Threshold multiplier for event/peak detection based on the standard deviation of the signal's derivative. Default is 3
        timeout (int): Minimum distance between detected peaks/events in seconds. Default is 3
        overwrite (bool): Flag indicating whether to overwrite existing files.
        toPlotPks (bool): Whether to plot results from pks_utils. Default is False
        forPres (bool): Whether to export svgs for presentation. Default is False
        concatCheck (bool): Whether to check for concatenated sessions. Default is False
        __version__ (str): Version of the QT_manager class.
        class_type (str): Type of the class.

    Methods:
        __init__(self, program_name, path=[], sess2process=[], frameWindow=15, sdThresh=3, timeout=3, overwrite=None, rewOmit=False):
            Initializes the QT_manager class with the provided parameters.

        static_class_var_init(self, folder_path, sess2process, frameWindow, sdThresh, timeout, overwrite):
            Initializes the static class variables.

        forLoop_var_init(self, sess_idx, sess_num):
            Initializes the variables and classes required for the for loop iteration.

        _reset_forLoop_var(self):
            Resets the variables used in the for loop to None or default values.

        overwrite_check(self):
            Checks if pkl files exist and prompts the user to overwrite them if necessary.

        init_N_create_pksDict(self):
            Initializes and creates the pksDict dictionary.

        import_segDict(self):
            Imports the segmentation dictionary from the latest file.

        wrapCueShift2cueShiftStrucExporter(self):
            Wraps the CueShiftTuning function to export the cue shift structure.

        _print_css_details(self):
            Prints the details of cueShiftStruc.

        Plot_cueShiftTuning(self):
            Plots the cue shift tuning for the given parameters.

        _create_lapType_name_arr(self):
            Creates an array of lap type names based on the given lap type array and omit laps.

        _find_refLapType(self):
            Finds the reference lap type based on the frequency of each lap type.

        endIter_funcs(self):
            Performs end-of-loop operations, including resetting variables and garbage collection.
    """

    def __init__(
        self,
        program_name: str,
        path: str | list = [],
        sess2process: list = [],
        frameWindow: float = 15,
        sdThresh: float = 3,
        timeout: int = 3,
        overwrite: bool | None = None,
        toPlotPks=False,
        forPres=False,
        # concatCheck=False,  #! THIS PARAMETER IS STILL A WORK IN PROGRESS
        # rewOmit=False,
    ) -> None:
        """
        Initializes the QT_manager class.

        Parameters:
            program_name (str): The name of the program.
            path (str | list): The path to the folder containing the data.
            sess2process (list): The session to process.
            frameWindow (float): Window size (in frames) used for smoothing for the event/peak detection. Default is 15
            sdThresh (float): Threshold multiplier for event/peak detection based on the standard deviation of the signal's derivative. Default is 3
            timeout (int): Minimum distance between detected peaks/events in seconds. Default is 3
            overwrite (bool | None): Whether to overwrite existing files.
            toPlotPks (bool): Whether to plot results from pks_utils. Default is False
            forPres (bool): Whether to plot for presentation. Default is False
        """

        self.program_name = program_name
        self.class_type = "manager"

        #! SINCE CONCATCHECK IS STILL A WORK IN PROGRESS, IT IS NOT YET IMPLEMENTED
        #! SETTING TO FALSE FOR NOW
        concatCheck = False
        print("ConcatCheck is still a work in progress. Setting to False for now.")

        BC.__init__(
            self,
            program_name=self.program_name,
            mode=self.class_type,
            sess2process=sess2process,
        )

        self.static_class_var_init(
            folder_path=path,
            sess2process=sess2process,
            frameWindow=frameWindow,
            sdThresh=sdThresh,
            timeout=timeout,
            overwrite=overwrite,
            toPlotPks=toPlotPks,
            forPres=forPres,
            concatCheck=concatCheck,
        )

    def static_class_var_init(
        self,
        folder_path: str | list,
        sess2process: list,
        frameWindow: float,
        sdThresh: float,
        timeout: int,
        overwrite: bool | None,
        toPlotPks: bool,
        forPres: bool,
        concatCheck: bool,
    ) -> None:
        """
        Initializes the static class variables.

        Parameters:
            folder_path (str): The path to the folder.
            sess2process (str): The session to process.
            frameWindow (float): Window size (in frames) used for smoothing for the event/peak detection. Default is 15
            sdThresh (float): Threshold multiplier for event/peak detection based on the standard deviation of the signal's derivative. Default is 3
            timeout (int): Minimum distance between detected peaks/events in seconds. Default is 3
            overwrite (bool): Whether to overwrite existing files.
            toPlotPks (bool): Whether to plot results from pks_utils. Default is False
            forPres (bool): Whether to plot for presentation. Default is False
            concatCheck (bool): Whether to check for concatenated sessions. Default is False
        """

        BC.static_class_var_init(
            self,
            folder_path=folder_path,
            file_of_interest=self.text_lib["selector"]["tags"]["SD"],
            selection_made=sess2process,
            select_by_ID=concatCheck,
        )
        self.frameWindow = frameWindow
        self.sdThresh = sdThresh
        self.timeout = timeout
        self.overwrite = overwrite
        self.toPlotPks = toPlotPks
        self.forPres = forPres
        self.concatCheck = concatCheck
        self.fig_save_path = self.text_lib["FIGSAVE"]["DEFAULT"]

        self.PKSkey = self.enum2dict(UA_enum.PKS)
        self.PCLAkey = self.enum2dict(PCLA_enum.TXT)
        self.CSSkey = self.enum2dict(UA_enum.CSS)
        self.LCDkey = self.enum2dict(UA_enum.LCD)

        # DS image name
        self.DS_image = self.utils.image_utils.get_DSImage_filename()

        # change sess2process to a tuple of session numbers grouped by ID
        # for when sessions need/were concatenated
        if self.concatCheck:
            self.sess2process = self.group_sessions_by_id4concat()

    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        """
        Initializes the variables and classes required for the for loop iteration.

        Parameters:
            sess_idx (int): The index of the current session.
            sess_num (int): The total number of sessions.
        """

        # initiate forLoop_var_init from BaseClass
        BC.forLoop_var_init(self, sess_idx, sess_num)

        # initiate plotting class
        if self.concatCheck:
            sess_name = (self.dayDir[sess_num[0] - 1], self.dayDir[sess_num[-1] - 1])
            self.folder_name_concat = [
                os.path.abspath(fold) for fold in self.folder_name
            ]
            for idx, folder in enumerate(self.folder_name):
                self.folder_name_concat[idx] = os.path.abspath(folder)
        else:
            sess_name = self.dayDir[sess_num - 1]

        self.plotting = QT_Plotters(sess_name=sess_name, forPres=self.forPres)
        # initiate PKSUtils class
        self.PKSUtils = pks_utils(
            frameWindow=self.frameWindow,
            sdThresh=self.sdThresh,
            timeout=self.timeout,
            sess_name=self.folder_name,
        )

        if self.concatCheck:
            self.create_conCat_outputFolder()

        # initiate variables as None
        self._reset_forLoop_var()

    def _reset_forLoop_var(self) -> None:
        """
        Resets the variables used in the for loop to None or default values.
        """

        # initiate variables as None
        self.pksDict = None
        self.C_Temporal = None
        self.A_Spatial = None
        self.CCF = None
        self.cueShiftStruc = None
        self.treadBehDict = None
        self.lapCueDict = None
        self.loaded_css = False
        self.refLapType = None

    def overwrite_check(self) -> None:
        """
        Checks if pkl files exist and prompts the user to overwrite them if necessary.

        This method checks for the existence of pkl files and provides a warning if any of the files
        are found in the specified folder. If the `overwrite` and `overwrite_choice` attributes are
        both False, the user is prompted to choose whether to delete the existing files and start
        from scratch. If the user chooses to delete the files, the `overwrite` attribute is set to
        True and the overwrite process is initiated.

        If the `overwrite_choice` attribute is already set to 'y', the overwrite process is initiated
        without prompting the user again.

        If the `overwrite_choice` attribute is set to 'n', the existing files are used for analysis.

        Parameters:
            self: The instance of the class.
        """

        # check to see if pkl files exist
        if isinstance(self.folder_name, str):
            folder_name2use = [self.folder_name]
        else:
            folder_name2use = self.folder_name_concat

        # TBD_file, LAPD_file, CSS_file = [], [], []
        eligible_file_list = []
        # eligibile_file_list_bySess = []
        QTcheck = []
        fileChecker = self.folder_tools.fileNfolderChecker

        for folder in folder_name2use:
            self.folder_tools.chdir_check_folder(
                folder, verbose=False
            ) if self.concatCheck else None
            (tbd, lpd, css), efl = self.folder_tools.file_checker_via_tag(
                file_tag=[
                    self.file_tag["TBD"],
                    self.file_tag["LAPD"],
                    self.file_tag["CSS"],
                ],
                file_end=self.file_tag["PKL"],
                full_path=self.concatCheck,
            )
            eligible_file_list.extend(efl)

            # determine if any of the files exist
            QTcheck.append(fileChecker(css) or fileChecker(lpd) and fileChecker(tbd))

        # convert list of bools to single bool
        # if only 1 bool, remains as bool
        # if 2 bools, means concat session, so only need 1 session to have pkl
        QTcheck = any(QTcheck)

        if self.overwrite is None:
            self.overwrite = self.utils.overwrite_selector(
                file_check=QTcheck,
                file_list=eligible_file_list,
                init_prompt=f"WARNING: quickTuning output files were found in {self.folder_path}:",
                user_confirm_prompt="Would you like to delete these files & start from scratch?",
            )
        if self.overwrite and eligible_file_list:
            self.rprint("OVERWRITING PROCESS INITIATED:")
            self.folder_tools.delete_pkl_then_mat(eligible_file_list)
            self.folder_tools.delete_folder(self.fig_save_path)
            self.rprint(f"Starting anew for {self.folder_path}")
            print()
        elif not self.overwrite and eligible_file_list:
            self.rprint("***Using existing files for analysis***")
            self.folder_tools.print_folder_contents(
                eligible_file_list, pre_file_msg="Using:"
            )
        elif not eligible_file_list:
            print_statement = "Starting anew for:"
            if isinstance(self.folder_path, str):
                self.rprint(f"{print_statement} {self.folder_path}")
            elif isinstance(self.folder_path, list):
                folders = [os.path.basename(folder) for folder in self.folder_path] + [
                    f"Concat Folder: {os.path.basename(self.output_pathByID)}"
                ]
                self.rprint("Starting anew for:")
                for folder in folders:
                    self.print_wFrm(folder)
            print()

    def init_N_create_pksDict(self) -> None:
        """
        Initializes and creates the pksDict dictionary.

        This method iterates over each row in the C_Temporal array and finds Ca transients using the PKSUtils class.
        It then stores the peaks, amplitudes, and waveforms in the pksDict dictionary.
        """

        print("Finding Ca transients & creating pksDict")

        self.pksDict = {}
        for seg in range(self.C_Temporal.shape[0]):
            # # !TESTING OUT RIDGE
            # RW_ALGO = RidgeWalker(
            #     cell_num=seg,
            #     Ca_arr=self.C_Temporal[seg, :],
            #     beta=2,
            #     gamma=3,
            #     window_size=100,
            #     min_scale_length=41,
            #     neighborhood_size=(1, 3),
            #     total_scales=100,
            #     # frameWindow=self.frameWindow,
            # )
            # RW_ALGO.showPeaksOverTrace()
            # if seg == self.C_Temporal.shape[0] - 1:
            #     RW_ALGO.export_params2JSON()

            # Iterate over each row in C_Temporal
            peaks, amps, waveform = self.PKSUtils.find_CaTransients(
                self.C_Temporal[seg, :].copy(), toPlot=self.toPlotPks, cell_num=seg
            )
            key = f"{self.PKSkey['SEG']}{seg}"
            self.pksDict[key] = {
                self.PKSkey["PEAKS"]: peaks,
                self.PKSkey["AMPS"]: amps,
                self.PKSkey["WAVEFORM"]: waveform,
            }
        self.print_done_small_proc()

    def import_segDict(self):
        """
        Imports the segmentation dictionary from the latest file.

        This method finds the latest file with the specified file tags for the segmentation dictionary,
        loads the segmentation dictionary from that file, and assigns the loaded values to the
        `C_Temporal` and `A_Spatial` attributes of the object.

        Returns:
            None
        """
        self.latest_file_segDict = self.findLatest(
            [self.file_tag["SD"], self.file_tag["PKL"]]
        )
        self.C_Temporal, self.A_Spatial = self.load_segDict(
            self.latest_file_segDict, C=True, A=True
        )

        # print("Creating image of cell numbers overlayed onto downsampled image")
        # self.plotting.plot_cellNum_overDSImage(
        #     A_Spat=self.A_Spatial, DS_image=self.DS_image
        # )
        # self.print_done_small_proc()

    def wrapCueShift2cueShiftStrucExporter(self) -> tuple:
        """
        Wraps the `CueShiftTuning` function to export the cue shift structure.

        This method calls the `CueShiftTuning` function to generate the cue shift structure,
        tread behavior dictionary, lap cue dictionary, and loaded CSS. If the CSS was not
        loaded from a previous file, it adds the segment dictionary name, path, and pksDict
        to the cue shift structure. It then sets up the `CueCellFinder` class for later plotting,
        and processes to find the reference lap type. It also creates an array of lap type names
        for figures and visibility. Finally, it prints CSS details and saves the CSS.

        Returns:
            tuple: A tuple containing the cue shift structure and tread behavior dictionary.
        """

        (
            self.cueShiftStruc,
            self.treadBehDict,
            self.lapCueDict,
            self.loaded_css,
        ) = wrapCueShiftTuning(self.pksDict, self.folder_path)

        if not self.loaded_css:
            # adding segdict name, path, & pksDict to cueShiftStruc
            # if css wasn't loaded from prev file
            self.cueShiftStruc[self.CSSkey["SD"]] = self.latest_file_segDict
            self.cueShiftStruc[self.CSSkey["PATH"]] = self.folder_path
            self.cueShiftStruc[self.CSSkey["PKS"]] = self.pksDict

        # set up ccf class for later plotting
        # also provides omitlaps & switch laps
        self.CCF = CueCellFinder(
            cueShiftStruc=self.cueShiftStruc,
            treadBehDict=self.treadBehDict,
            C_Temporal=self.C_Temporal,
            A_Spatial=self.A_Spatial,
            PKSUtils=self.PKSUtils,
            pksDict=self.pksDict,
            forPres=self.forPres,
        )

        # processing to find refLapType
        self._find_refLapType()
        # create array of laptype names for figs/visibility
        # uses omitlaps & switch laps from ccf
        self._create_lapType_name_arr()

        # print css details
        self._print_css_details()

        if not self.loaded_css:
            # find latest file to use for filename for cueShiftStruc
            if self.findLatest(self.file_tag["XML"]):
                ftag2remove = self.file_tag["XML"]
                fname = self.findLatest(self.file_tag["XML"])
            elif self.findLatest([self.file_tag["GPIO_SUFFIX"], self.file_tag["CSV"]]):
                ftag2remove = []
                fname = self.findLatest(
                    [self.file_tag["GPIO_SUFFIX"], self.file_tag["CSV"]]
                ).split(self.file_tag["GPIO_SUFFIX"])[0]

            # fill in css with refLapType & laptypename
            # if css wasn't loaded from prev file
            self.cueShiftStruc[self.CSSkey["LAP_CUE"]][self.LCDkey["LAP_KEY"]][
                self.LCDkey["REFLAP"]
            ] = self.refLapType
            self.cueShiftStruc[self.CSSkey["LAP_CUE"]][self.LCDkey["LAP_KEY"]][
                self.LCDkey["LAP_TANAME"]
            ] = self.lapTypeNameArr

            # saving css to mat & pkl & h5
            self.savedict2file(
                dict_to_save=self.cueShiftStruc,
                dict_name=self.dict_name["CSS"],
                filename=fname,
                file_tag_to_remove=ftag2remove,
                file_suffix=self.dict_name["CSS"],
                date=True,
                filetype_to_save=[self.file_tag["MAT"], self.file_tag["PKL"]],
            )

    def _print_css_details(self) -> None:
        """
        Prints the details of cueShiftStruc.

        This method prints the details of cueShiftStruc, including the LapType numbers and names,
        the reference LapType, and the number of PC cells found in the reference LapType.
        """

        self.rprint("cueShiftStruc Details:")
        self.print_wFrm("LapType Number & Names:")
        for idx, name in enumerate(self.lapTypeNameArr, start=1):
            laptype2print = f"LapType {idx:02} - {name}"
            if idx == self.refLapType + 1:
                # print refLapType num & name
                ref_hl = self.color_dict["red"]
                laptype2print = f"{laptype2print} (REFERENCE)"
            elif name is np.nan:
                ref_hl = self.color_dict["orange"]
                laptype2print = (
                    f"{laptype2print} [Omitted due to low lap count (n = 1)]"
                )
            else:
                ref_hl = None
            self.print_wFrm(laptype2print, frame_num=1, color=ref_hl)
        # procures PC number from refLapType
        PC_refLapType = self.cueShiftStruc[self.CSSkey["PCLS"]][
            f"lapType{self.refLapType + 1}"
        ][self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]]
        # prints sum PC_refLapType
        self.print_wFrm(
            f"Number of PC cells found in reference LapType: {int(np.sum(PC_refLapType))}\n"
        )

    def Plot_cueShiftTuning(self) -> None:
        """
        Plot the cue shift tuning for the given parameters.

        Parameters:
        - self.cueShiftStruc: The cue shift structure.
        - self.refLapType: The reference lap type.
        - self.C_Temporal: The temporal parameter.
        - self.lapTypeNameArr: The array of lap type names.
        """

        self.rprint("Plotting cueShiftStruc ...")
        self.plotting.cueShiftTuning(
            self.cueShiftStruc, self.refLapType, self.C_Temporal, self.lapTypeNameArr
        )

        # to plot start & mid cells need to add refLapType & lapTypeNameArr
        # for it to work
        self.CCF.plot_startNmidCueCellTuning(self.refLapType, self.lapTypeNameArr)
        self.CCF.plot_midCueCells_via_2xMethod()
        self.CCF.plot_midCueCells_viaStatTest()
        self.CCF.createNexport_CCFdict()
        # self.print_done_small_proc()

    def _create_lapType_name_arr(self) -> None:
        """
        Creates an array of lap type names based on the given lap type array and omit laps.
        """

        omitLaps = self.CCF.omitLaps

        switchLaps = self.CCF.switchLaps
        lapTypeArr = self.lapCueDict[self.LCDkey["LAP_KEY"]][
            self.LCDkey["LAP_TYPEARR"]
        ].copy()

        # lapTypeNameArr = np.full(len(np.unique(lapTypeArr)), np.nan)
        lapTypeNameArr = [None] * len(np.unique(lapTypeArr))

        OL_keys_to_use = [key for key, value in omitLaps.items() if len(value) > 0]
        all_omit_laps = np.concatenate(
            [value for key, value in omitLaps.items() if len(value) > 0]
        )
        non_omit_laps = []
        for lt_idx, lapType in enumerate(lapTypeArr):
            if lt_idx not in all_omit_laps and lt_idx not in switchLaps:
                non_omit_laps.append(lt_idx)
        non_omit_laps = np.array(non_omit_laps)

        if any(key in OL_keys_to_use for key in ["OMITBOTH", "OMITCUE2"]) and not any(
            key == "OMITOPTO" for key in OL_keys_to_use
        ):
            non_omit_type = "BOTHCUES"
        elif any(key in OL_keys_to_use for key in ["OMITTONE", "OMITLED"]):
            non_omit_type = "CUE/LED/TONE"
        elif any(key in OL_keys_to_use for key in ["OMITOPTO"]):
            non_omit_type = "CUEwOPTO"
        else:
            non_omit_type = "CUE1"

        # sometimes at end of multi cue laps, there is a weird lapType
        # shows up once, this NaNs it out
        # no PC results will result given low amount of info
        unique_lapTypes, counts = np.unique(lapTypeArr, return_counts=True)
        unique_lapTypes = unique_lapTypes[counts > 1]

        for lap_idx, lapType in enumerate(lapTypeArr):
            if lapType in unique_lapTypes:
                for omit_type in OL_keys_to_use:
                    if lap_idx in omitLaps[omit_type]:
                        if omit_type == "OMITCUE1_L1":
                            lapTypeNameArr[lapType - 1] = "CUE1"
                        elif omit_type == "OMITCUE1_L2":
                            lapTypeNameArr[lapType - 1] = "SHIFT"
                        elif omit_type == "OMITTONE":
                            lapTypeNameArr[lapType - 1] = "CUE/LED"
                        elif omit_type == "OMITLED":
                            lapTypeNameArr[lapType - 1] = "CUE/TONE"
                        elif omit_type == "OMITCUE1" and any(
                            key in OL_keys_to_use for key in ["OMITTONE", "OMITLED"]
                        ):
                            lapTypeNameArr[lapType - 1] = "LED/TONE"
                        elif omit_type == "OMITCUE1" and any(
                            key in OL_keys_to_use for key in ["OMITOPTO"]
                        ):
                            lapTypeNameArr[lapType - 1] = "OPTO"
                        elif omit_type == "OMITOPTO":
                            lapTypeNameArr[lapType - 1] = "CUE"
                        else:
                            lapTypeNameArr[lapType - 1] = omit_type
                if switchLaps is not None:
                    if lap_idx in switchLaps:
                        lapTypeNameArr[lapType - 1] = "SWITCH"
                        if "OMITCUE1_L1" in OL_keys_to_use:
                            lapTypeNameArr[lapType - 1] = "SHIFT"
                if lap_idx in non_omit_laps:
                    lapTypeNameArr[lapType - 1] = non_omit_type
            else:
                lapTypeNameArr[lapType - 1] = np.nan
        # rename laptype for 2 odor at same location
        exp_2odor_vals = {"OMITCUE2", "OMITCUE1", "OMITBOTH"}
        if set(lapTypeNameArr) == exp_2odor_vals:
            lapTypeNameArr = [
                "CUE1" if name == "OMITCUE2" else "CUE2" if name == "OMITCUE1" else name
                for name in lapTypeNameArr
            ]
            # change refLapType to CUE1 otherwise it would be OMITBOTH
            self.refLapType = lapTypeNameArr.index("CUE1")

        # fill in self.lapTypeNameArr
        self.lapTypeNameArr = lapTypeNameArr

    def _find_refLapType(self) -> None:
        """
        Finds the reference lap type based on the frequency of each lap type.
        """

        lapTypeArr = self.lapCueDict[self.LCDkey["LAP_KEY"]][
            self.LCDkey["LAP_TYPEARR"]
        ].copy()
        # lapTypeArr = self.cueShiftStruc[self.CSSkey["LAP_CUE"]][self.CSSkey["LAP_KEY"]][
        #     self.CSSkey["LAP_TYPEARR"]
        # ]

        # Count the frequency of each lap type
        numLapType = [
            len(np.where(lapTypeArr == i)[0])
            for i in range(1, int(np.max(lapTypeArr)) + 1)
        ]
        # Determine the lap type with the maximum count
        # turn into string to use as key for cueShiftStruc["PCLappedSess"]
        self.refLapType = np.argmax(numLapType)

    def endIter_funcs(self) -> None:
        """
        Perform end-of-loop operations.

        This method resets variables and performs garbage collection.

        Parameters:
            self (object): The instance of the class.
        """

        # reset variables & run garbage collector
        with self.StatusPrinter.garbage_collector():
            self._reset_forLoop_var()
