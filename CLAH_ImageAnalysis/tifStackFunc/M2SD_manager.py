import os
import numpy as np
from tqdm import tqdm
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.tifStackFunc import CNMF_Utils
from CLAH_ImageAnalysis.tifStackFunc import H5_Utils
from CLAH_ImageAnalysis.tifStackFunc import ImageStack_Utils
from CLAH_ImageAnalysis.tifStackFunc import MoCoPreprocessing
from CLAH_ImageAnalysis.tifStackFunc import Movie_Utils as movie_utils
from CLAH_ImageAnalysis.tifStackFunc import NoRMCorre
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


class M2SD_manager(BC):
    """
    M2SD_Utils class provides utility functions for image analysis and processing.

    Parameters:
        folder_path (list, optional): The path to the folder. Defaults to [].
        sess2process (list, optional): List of sessions to process. Defaults to [].
        motion_correct (bool, optional): Whether to perform motion correction. Defaults to False.
        segment (bool, optional): Whether to perform segmentation. Defaults to False.
        n_proc4MOCO (int | None, optional): Number of processes for motion correction. Defaults to None.
        n_proc4CNMF (int | None, optional): Number of processes for CNMF. Defaults to None.
        concatCheck (bool, optional): Whether to concatenate multiple sessions. Defaults to False.
        mc_iter (int, optional): Number of motion correction iterations. Defaults to 1.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        compute_metrics (bool, optional): Whether to compute motion correction metrics. Defaults to False.
        folder_path (str | list): Path to the folder containing data
        sess2process (list): List of sessions to process
        motion_correct (bool): Whether to perform motion correction
        segment (bool): Whether to perform segmentation
        n_proc4MOCO (int | None): Number of processes for motion correction
        n_proc4CNMF (int | None): Number of processes for CNMF
        concatCheck (bool): Whether to concatenate multiple sessions
        mc_iter (int): Number of motion correction iterations
        overwrite (bool): Whether to overwrite existing files
        compute_metrics (bool): Whether to compute motion correction metrics
        use_cropper (bool): Whether to use the cropping utility. Defaults to False.
        separate_channels (bool): Whether to motion correct channels separately. Defaults to False.
        export_postseg_residuals (bool): Whether to export the post-segmentation residuals as a video file. Defaults to False.

    Attributes:
        H5U (H5_Utils): H5 utilities instance
        ISXU (ISX_Utils): ISX utilities instance
        folder_path (str | list): Path to data folder
        sess2process (list): Sessions to process
        motion_correct (bool): Motion correction flag
        segment (bool): Segmentation flag
        n_proc4MOCO (int | None): Processes for motion correction
        n_proc4CNMF (int | None): Processes for CNMF
        concatCheck (bool): Concatenation flag
        mc_iter (int): Motion correction iterations
        overwrite (bool): Overwrite flag
        compute_metrics (bool): Compute metrics flag
        use_cropper (bool): Whether to use the cropping utility. Defaults to False.
        separate_channels (bool): Whether to motion correct channels separately. Only applicable for 2photon data with 2 channels. Defaults to False.
        onePhotonCheck (bool): One photon imaging flag
        dview: Parallel processing view
        CNMFpar (dict): CNMF parameters
        ISUtils (ImageStack_Utils): Image stack utilities
        CNMFU (CNMF_Utils): CNMF utilities
        array_for_cnmf (list): Arrays for CNMF processing
        dx (float): X dimension pixel size
        dy (float): Y dimension pixel size
        dims (tuple): Image dimensions
        basename (str): Base filename
        fname_mmap_postproc (str): Memory mapped filename
        output_pathByID (str): Output path by ID

    Methods:
        __init__: Initialize the M2SD_manager class
        static_class_var_init: Initialize static class variables
        _init_vars4Iter: Initialize variables for each iteration
        _init_CNMF_params: Initialize CNMF parameters
        _init_parallel_processing: Initialize parallel processing
        _init_ISUtils: Initialize ImageStack utilities
        _init_CNMF_vars: Initialize CNMF variables
        _init_output_paths: Initialize output paths
        _ISU_chan2use_utils: Utility for handling two-channel data
        _ISU_downsample_array: Downsample array using ISUtils
        _ISU_normalizeNcorrect_tempfiltDSArr: Normalize and correct temporally filtered downsampled array
        _ISU_saveImage: Save image using ISUtils
        clear_ISUtils: Clear ISUtils and perform garbage collection
        initializeCNMF: Initialize CNMF utilities
        _onePhotonCheck_utils: Check if data is one-photon
        find_init_patches_viaCNMF: Find initial patches using CNMF
        evaluate_found_patches: Evaluate found patches
        plot_contours: Plot component contours
        refine_patches_using_accepted_patches: Refine patches using accepted ones
        play_movie: Play movie with options
        _create_recontructed_movie: Create reconstructed movie
        _save_residual_movie: Save residual movie
        endIter_funcs: Functions to run at end of iteration
    """

    ######################################################
    #  funcs run at init
    ######################################################
    def __init__(
        self,
        program_name: str,
        path: str | list = [],
        sess2process: list[int] | str = [],
        motion_correct: bool = False,
        segment: bool = False,
        n_proc4MOCO: int | None = None,
        n_proc4CNMF: int | None = None,
        concatCheck: bool = False,
        mc_iter: int = 1,
        overwrite: bool = False,
        compute_metrics: bool = False,
        use_cropper: bool = False,
        separate_channels: bool = False,
        export_postseg_residuals: bool = False,
    ) -> None:
        """
        Initialize the M2SD_manager class.

        Parameters:
            program_name (str): Name of the program
            path (str | list): Path or list of paths to the data files
            sess2process (list[int] | str): List of sessions to process. If str, it must be 'all' & it will convert to list of integers representing all available sessions.
            motion_correct (bool): Whether to perform motion correction
            segment (bool): Whether to perform segmentation
            n_proc4MOCO (int | None): Number of processes for motion correction
            n_proc4CNMF (int | None): Number of processes for CNMF
            concatCheck (bool): Whether to check for concatenated files
            mc_iter (int): Number of motion correction iterations
            overwrite (bool): Whether to overwrite existing files
            compute_metrics (bool): Whether to compute performance metrics
            use_cropper (bool): Whether to use the cropping utility. Defaults to False.
            separate_channels (bool): Whether to motion correct channels separately. Only applicable for 2photon data with 2 channels. Defaults to False.
            export_postseg_residuals (bool): Whether to export the post-segmentation residuals as a video file. Defaults to False.
        """
        self.program_name = program_name
        self.class_type = "manager"

        BC.__init__(
            self,
            program_name=self.program_name,
            mode=self.class_type,
            steps=True,
            sess2process=sess2process,
        )

        # initiate H5 Utils
        self.H5U = H5_Utils()

        # initiate MoCoPreprocessing for 1photon data
        self.MCPP = MoCoPreprocessing()

        # init global vars
        self.static_class_var_init(
            folder_path=path,
            sess2process=sess2process,
            motion_correct=motion_correct,
            segment=segment,
            n_proc4MOCO=n_proc4MOCO,
            n_proc4CNMF=n_proc4CNMF,
            concatCheck=concatCheck,
            mc_iter=mc_iter,
            overwrite=overwrite,
            compute_metrics=compute_metrics,
            use_cropper=use_cropper,
            separate_channels=separate_channels,
            export_postseg_residuals=export_postseg_residuals,
        )

    def static_class_var_init(
        self,
        folder_path: str | list,
        sess2process: list[int] | str,
        motion_correct: bool,
        segment: bool,
        n_proc4MOCO: int | None,
        n_proc4CNMF: int | None,
        concatCheck: bool,
        mc_iter: int,
        overwrite: bool,
        compute_metrics: bool,
        use_cropper: bool,
        separate_channels: bool,
        export_postseg_residuals: bool,
    ) -> None:
        """
        Initializes the static class variables.

        Parameters:
            folder_path (str | list): The path to the folder.
            sess2process (list[int] | str): The sessions to process. If str, it must be 'all' & it will convert to list of integers representing all available sessions.
            motion_correct (bool): Whether to motion correct the data.
            segment (bool): Whether to segment the data.
            n_proc4MOCO (int | None): The number of processes to use for motion correction.
            n_proc4CNMF (int | None): The number of processes to use for CNMF.
            concatCheck (bool): Whether to check for concatenation.
            mc_iter (int): Number of iterations for motion correction.
            overwrite (bool): Flag indicating whether to overwrite existing files.
            compute_metrics (bool): Whether to compute metrics for motion correction.
            use_cropper (bool): Whether to use the cropping utility. Defaults to False.
            separate_channels (bool): Whether to motion correct channels separately. Only applicable for 2photon data with 2 channels. Defaults to False.
            export_postseg_residuals (bool): Whether to export the post-segmentation residuals as a video file. Defaults to False.
        """
        BC.static_class_var_init(
            self,
            folder_path=folder_path,
            file_of_interest=self.text_lib["selector"]["tags"]["EMC"],
            selection_made=sess2process,
            select_by_ID=concatCheck,
        )

        self.CNMFpar = self.enum2dict(TSF_enum.CNMF_PARAMS)
        self.motion_correct = motion_correct
        self.segment = segment
        self.n_proc4MOCO = n_proc4MOCO
        self.n_proc4CNMF = n_proc4CNMF
        self.concatCheck = concatCheck
        self.mc_iter = mc_iter
        self.overwrite = overwrite
        self.compute_metrics = compute_metrics
        self.use_cropper = use_cropper
        self.separate_channels = separate_channels
        self.export_postseg_residuals = export_postseg_residuals

        # change sess2process to a tuple of session numbers grouped by ID
        # for when sessions need/were concatenated
        if self.concatCheck:
            self.sess2process = self.group_sessions_by_id4concat()

        # initiate vars to fill during iterations
        self._init_vars4Iter()

    def _init_vars4Iter(self):
        """
        Initializes the variables for the iterations.
        """
        # Cluster
        self.c = None
        self.dview = None
        self.n_processes = None
        # arrays
        self.array_for_cnmf = []
        self.array_to_trim = []
        self.basename = None
        # Class Utils
        self.CNMFU = None
        self.ISUtils = None
        # Memoray Map
        # self.mmap_fname = None
        self.mmap_of_moco = None
        self.mmap_of_moco_ch1 = None
        self.mmap_of_moco_ch2 = None

        # H5
        self.chan_idx = None
        self.element_size_um = None
        self.hfsiz = None
        self.h5filename = None
        self.h5fname_sqz = None
        self.h5fname_sqz_ch1 = None
        self.h5fname_sqz_ch2 = None
        self.h5filename_postproc = None
        self.h5filename_postproc_ch1 = None
        self.h5filename_postproc_ch2 = None
        self.dimension_labels = ["t", "y", "x"]

        # isxd (1photon data)
        self.isxd_fname = None
        self.latest_isxd = None
        self.latest_tiff = None

        # latest files
        self.latest_eMC = None
        self.latest_eMC_concat = None
        self.latest_h5 = None

        self.latest_h5sqz = None
        self.latest_h5sqz_ch1 = None
        self.latest_h5sqz_ch2 = None

        self.latest_mmap_moco = None
        self.latest_mmap_moco_ch1 = None
        self.latest_mmap_moco_ch2 = None

        # list of latest files
        self.list_latest_sqz = []
        self.list_latest_mmap_moco = []

        self.fname_mmap_postproc = []

        # onePhotonCheck
        self.onePhotonCheck = False

        # certain pre-processes applied (boolean)
        self.bandpass_applied = False
        self.high_pass_applied = False
        self.CLAHE_applied = False
        self.crop_applied = False

    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        """
        Initializes the variables for the for loop.

        Parameters:
            sess_idx (int): The session index.
            sess_num (int): The session number.
        """
        # initiate forLoop_var_init from BaseClass
        BC.forLoop_var_init(self, sess_idx, sess_num)

        # create output folder for concat if concatCheck is True
        if self.concatCheck:
            self.create_conCat_outputFolder()

        # find eMC file, h5 file, and h5 sqz file
        # init ImageStack Utils after procuring basename from h5 file
        self._find_related_eMC_filesNinit_vars()

        # if overwrite is True, delete existing files and find related eMC files again
        if self.overwrite:
            self._overwrite_files()
            self._find_related_eMC_filesNinit_vars()

    def _find_related_eMC_filesNinit_vars(self) -> None:
        """
        Finds the related eMC files and initializes the variables.
        """

        def _findFiles(full_path: bool = False) -> tuple:
            eMC_file = self.findLatest(self.file_tag["COMP_EMCFNAME"])
            h5_file = self.findLatest(
                [self.file_tag["H5"], self.file_tag["CYCLE"], self.file_tag["ELEMENT"]],
                notInclude=[
                    self.file_tag["SQZ"],
                    self.file_tag["EMC"],
                    self.file_tag["PS_CON_RES_MOV"],
                    self.file_tag["PS_NOISE_MOV"],
                    self.file_tag["PS_CDA_MOV"],
                    self.file_tag["PS_CDA_BDF_MOV"],
                ],
                full_path=full_path,
            )
            isxd_file = self.findLatest(self.file_tag["ISXD"], full_path=full_path)
            tiff_file = self.findLatest(self.file_tag["TIFF"], full_path=full_path)

            # find latest h5 sqz file
            # latest_h5sqz will be present for non 2Ch files
            # - this is generally the norm
            # latest_h5sqz_ch1 & latest_h5sqz_ch2 will be present for 2Ch files
            h5sqz_file = self.findLatest(
                self.file_tag["SQZ"] + self.file_tag["H5"], full_path=full_path
            )
            h5sqz_ch1 = self.findLatest(
                f"{self.file_tag['SQZ']}_Ch1{self.file_tag['H5']}", full_path=full_path
            )
            h5sqz_ch2 = self.findLatest(
                f"{self.file_tag['SQZ']}_Ch2{self.file_tag['H5']}", full_path=full_path
            )

            # find latest memmap from motion correction
            # for non 2Ch files, latest_mmap_moco will be present
            mmap_moco_file = self.findLatest(
                self.file_tag["MMAP"],
                notInclude=[self.file_tag["MMAP_MC_PROC"], "_Ch1", "_Ch2"],
                full_path=full_path,
            )
            # for 2Ch sessions
            mmap_moco_ch1 = self.findLatest(
                [self.file_tag["MMAP"], self.file_tag["SQZ"] + "_Ch1"],
                notInclude=[self.file_tag["MMAP_MC_PROC"], "_Ch2"],
                full_path=full_path,
            )
            mmap_moco_ch2 = self.findLatest(
                [self.file_tag["MMAP"], self.file_tag["SQZ"] + "_Ch2"],
                notInclude=[self.file_tag["MMAP_MC_PROC"], "_Ch1"],
                full_path=full_path,
            )
            return (
                eMC_file,
                h5_file,
                isxd_file,
                tiff_file,
                h5sqz_file,
                h5sqz_ch1,
                h5sqz_ch2,
                mmap_moco_file,
                mmap_moco_ch1,
                mmap_moco_ch2,
            )

        if self.concatCheck:
            self.latest_eMC = []
            self.latest_h5 = []
            # self.latest_isxd = []
            # self.latest_tiff = []
            for directory in self.folder_path:
                os.chdir(directory)
                latest_eMC, latest_h5, _, _, _, _, _, _, _, _ = _findFiles(
                    full_path=self.concatCheck
                )

                self.latest_eMC.append(latest_eMC)
                self.latest_h5.append(latest_h5)
                # self.latest_isxd.append(latest_isxd)
                # self.latest_tiff.append(latest_tiff)
            os.chdir(self.output_pathByID)
            (
                self.latest_eMC_concat,
                self.latest_h5concat,
                _,
                _,
                self.latest_h5sqz,
                self.latest_h5sqz_ch1,
                self.latest_h5sqz_ch2,
                self.latest_mmap_moco,
                self.latest_mmap_moco_ch1,
                self.latest_mmap_moco_ch2,
            ) = _findFiles(full_path=self.concatCheck)

            if self.latest_h5sqz:
                self.list_latest_sqz.append(self.latest_h5sqz)
            # os.chdir(self.dayPath)
        else:
            (
                self.latest_eMC,
                self.latest_h5,
                self.latest_isxd,
                self.latest_tiff,
                self.latest_h5sqz,
                self.latest_h5sqz_ch1,
                self.latest_h5sqz_ch2,
                self.latest_mmap_moco,
                self.latest_mmap_moco_ch1,
                self.latest_mmap_moco_ch2,
            ) = _findFiles(full_path=True)

            if self.latest_h5sqz:
                self.list_latest_sqz.append(self.latest_h5sqz)
            if self.latest_h5sqz_ch1:
                self.list_latest_sqz.append(self.latest_h5sqz_ch1)
            if self.latest_h5sqz_ch2:
                self.list_latest_sqz.append(self.latest_h5sqz_ch2)

        # set basename
        if self.concatCheck:
            h52use = self.latest_h5[0]
            h52use = h52use.split("/")[-1]
        else:
            h52use = self.latest_h5

        if self.file_tag["CYCLE"] in h52use:
            self.basename = self.folder_tools.basename_finder(
                h52use, self.file_tag["CYCLE"]
            )
        else:
            self.basename = self.folder_tools.basename_finder(
                h52use, self.file_tag["H5"]
            )

        if self.latest_isxd:
            self.basename = os.path.basename(os.path.dirname(self.latest_isxd))
        elif self.latest_tiff:
            self.basename = os.path.basename(os.path.dirname(self.latest_tiff))

        if self.concatCheck:
            if "preK" in self.basename:
                self.basename = self.basename.replace("preK", "")
            else:
                self.basename = self.basename.replace("pre", "")

    def _overwrite_files(self) -> None:
        """
        Deletes existing files.
        """

        def _find_files(ftag):
            return [os.path.abspath(f) for f in os.listdir() if ftag in f]

        print(f"Running overwrite for {self.basename} before processing")

        # find files
        ftags2search = [
            "A",
            "SQZ",
            "EMC",
            "MMAP",
            "SD",
            "HTML",
            "NPZ",
            "PARAMS",
            "C_EVAL",
            "POST_SG",
        ]
        file_patterns = [self.file_tag[ftag] for ftag in ftags2search]

        files2remove = []
        for pattern in file_patterns:
            files2remove.extend(_find_files(pattern))
        files2remove.sort()

        # find folders
        folders = [
            self.text_lib["Folders"][key] for key in self.text_lib["Folders"].keys()
        ]
        folders = [os.path.abspath(f) for f in os.listdir() if f in folders]
        folders.sort()

        # combine files and folders
        files2remove += folders
        if files2remove:
            print(
                "--overwrite was set to True & eligible M2SD output files were found that can be removed:"
            )
            if len(files2remove) > 5:
                for file in tqdm(files2remove, desc="Removing files"):
                    self.folder_tools.remove_file(file)
            else:
                for file in files2remove:
                    self.print_wFrm(f"Removing: {file}")
                    self.folder_tools.remove_file(file)
            self.print_done_small_proc()
        else:
            print(
                "--overwrite was set to True but no eligible M2SD output files were found... skipping"
            )

        # reset variables
        self._init_vars4Iter()

    def init_vars_from_unproc_h5(self) -> None:
        """
        Initiates global variables that require chan_idx output from unprocessed h5 file.

        This method sets various global variables based on the `chan_idx` attribute and the latest_h5 file.
        It initializes the `basename` attribute, sets the name for h5 post processing, and creates a list of h5 filenames for post processing.

        Returns:
            None
        """
        # create variable to determine if 1Ch or 2Ch session
        self.oneCh = len(self.chan_idx) == 1
        self.twoCh = len(self.chan_idx) == 2

        if self.concatCheck:
            basename2use = self.fname_concat.split(self.file_tag["CYCLE"])[0]
        else:
            basename2use = self.basename

        # set name for h5 post processing
        self.h5filename_postproc = basename2use
        if self.oneCh:
            self.h5filename_postproc += self.file_tag["COMP_EMCFNAME"]
            # convert to list for consistency
            self.h5filename_postproc = [self.h5filename_postproc]
            self.h5filename_postproc_ch2 = self.h5filename_postproc[0]
        elif self.twoCh:
            self.h5filename_postproc = [
                f"{self.h5filename_postproc}_Ch{idx + 1}{self.file_tag['COMP_EMCFNAME']}"
                for idx in self.chan_idx
            ]
            self.h5filename_postproc_ch1 = self.h5filename_postproc[0]
            self.h5filename_postproc_ch2 = self.h5filename_postproc[1]
        # create list of h5 filenames for post processing
        self.list_h5fn_pproc = [self.h5filename_postproc_ch2]
        if self.h5filename_postproc_ch1 is not None:
            self.list_h5fn_pproc.append(self.h5filename_postproc_ch1)

        basename4ISU = [basename2use]

        if self.concatCheck:
            sess_folders_base = [f"{f}/{os.path.basename(f)}" for f in self.folder_path]
            # sess_folders_base = [
            #     f.replace("A-", "pre-").replace("B-", "post-")
            #     for f in sess_folders_base
            # ]
            basename4ISU += sess_folders_base
        # if self.segCh is None:
        #     self.onePhotonCheck = True

        # init ImageStack Utils
        self.ISUtils = ImageStack_Utils(
            basename4ISU, onePhotonCheck=self.onePhotonCheck
        )

    ######################################################
    #  H5 funcs
    ######################################################

    def load_unproc_H5(self) -> None:
        """
        Loads the unprocessed H5 file.
        """
        if self.concatCheck:
            print("Concatenating H5 files")
            if self.latest_h5concat:
                self.print_wFrm("Found already concatenated H5")
                self.print_wFrm(f"Using: {self.latest_h5concat}")

                # store concatenated H5 filename into self.fname_concat
                self.fname_concat = self.latest_h5concat
                file2use = self.latest_h5concat
            else:
                self.print_wFrm("Using files:")
                for h5 in self.latest_h5:
                    self.create_quickLinks_folder(fpath=os.path.dirname(h5))
                    self.create_symlink4QL(
                        src=h5,
                        link_name=self.text_lib["QL_LNAMES"]["RAW_H5"],
                        fpath4link=self.folder_tools.get_dirname(h5),
                    )
                    self.print_wFrm(f"{h5}", frame_num=1)
                concath5name = f"{self.basename}{self.file_tag['CYCLE']}{self.file_tag['CODE']}{self.file_tag['ELEMENT']}{self.file_tag['CODE']}"
                self.fname_concat = os.path.join(self.output_pathByID, concath5name)
                file2use = self.H5U.concatH5s(
                    H5s=self.latest_h5, fname_concat=self.fname_concat
                )
                self.create_symlink4QL(
                    src=file2use,
                    link_name=self.text_lib["QL_LNAMES"]["RAW_H5"],
                    fpath4link=self.folder_tools.get_dirname(file2use),
                )
            self.print_done_small_proc()
        else:
            self.create_quickLinks_folder()
            if self.latest_isxd:
                file2use = self.latest_isxd
                qlName = self.text_lib["QL_LNAMES"]["RAW_ISXD"]
            elif self.latest_tiff:
                file2use = self.latest_tiff
                qlName = self.text_lib["QL_LNAMES"]["RAW_TIFF"]
            else:
                file2use = self.latest_h5
                qlName = self.text_lib["QL_LNAMES"]["RAW_H5"]

            self.create_symlink4QL(
                src=file2use,
                link_name=qlName,
                fpath4link=self.folder_tools.get_dirname(file2use),
            )

        if file2use.endswith(self.file_tag["ISXD"]) or file2use.endswith(
            self.file_tag["TIFF"]
        ):
            self.onePhotonCheck = True
            self.segCh = None
            self.chan_idx = [1]
            self.isxd_fname = file2use
            self.total_frames, self.isxsiz = self.MCPP.get_movie_data(file2use)

            print(f"Reading isxd file: {file2use}")
            self.print_wFrm(f"Total frames: {self.total_frames}")
            self.print_wFrm(f"Dimensions: {self.isxsiz}")
            print()

        elif file2use.endswith(self.file_tag["H5"]):
            self.onePhotonCheck = False
            (
                self.h5filename,
                self.hfsiz,
                self.info,
                self.segCh,
                self.chan_idx,
                self.element_size_um,
            ) = self.H5U.read_file4MOCO(file2use)

            # set total frames
            self.total_frames = self.info[0]

    def write_procImageStack2H5(self, twoChan: bool = False) -> None:
        """
        Writes the post-processed image stack into the H5 file.
        """
        print("Writing post-processed image stack(s) into h5:")
        for idx, array in enumerate(self.norm_uint_tempfilteredDS_arr):
            # init h5fname variable
            h5fname = self.list_h5fn_pproc[idx]
            # set chan for 2Ch sessions
            if idx == 0:
                chan = 1
            elif idx == 1:
                chan = 0
            if self.twoCh:
                self.print_wFrm(
                    f"Writing post-processed image stack for Ch{chan} into h5"
                )
                twoChan = True
            h5fname = self.H5U.write_to_file(
                array_to_write=array,
                filename=h5fname,
                chan_idx=[chan],
                element_size_um=self.element_size_um,
                dimension_labels=self.dimension_labels,
                date=True,
                return_fname=True,
                twoChan=twoChan,
            )
            self.list_h5fn_pproc[idx] = h5fname

            if idx == 0:
                if not self.concatCheck:
                    self.create_symlink4QL(
                        src=os.path.join(self.folder_path, h5fname),
                        link_name=self.text_lib["QL_LNAMES"]["NORM_TFDS_H5"],
                        fpath4link=self.folder_tools.get_dirname(h5fname),
                    )

            print("Exporting image stack as colormapped AVI:")
            avi_fname = self.ISUtils.apply_colormap2stack_export2avi(
                array_to_use=array,
                fname_save=h5fname,
            )
            if not self.concatCheck:
                self.create_symlink4QL(
                    src=os.path.join(self.folder_path, avi_fname),
                    link_name=self.text_lib["QL_LNAMES"]["NORM_TFDS_AVI"],
                    fpath4link=self.folder_tools.get_dirname(avi_fname),
                )
            self.print_done_small_proc()

        # clear self.norm_uint_tempfilteredDS_arr for memory
        with self.StatusPrinter.garbage_collector():
            self.norm_uint_tempfilteredDS_arr = None

    def squeeze_h5_write2file(self) -> None:
        """
        Squeezes the H5 file and writes it to a file.
        """
        if not self.onePhotonCheck:
            print(f"Loading {self.h5filename}")
            self.print_wFrm(
                "squeezing array from 5 to 3 dimensions (for Motion Correction)"
            )

            twoChan = False
            if self.twoCh:
                twoChan = True
                self.print_wFrm(
                    "With 2 channels, will squeeze each channel separately:"
                )
            print()

            for chan in self.chan_idx:
                # for 1Ch, will only have 1 entry in chan_idx
                # so chan will be 0 to index from H5 properly
                if self.oneCh:
                    chan = 0
                h5fname_sqz = self.H5U.squeeze_fileNwrite(
                    file2read=self.h5filename,
                    chan_idx=[chan],
                    element_size_um=self.element_size_um,
                    dimension_labels=self.dimension_labels,
                    remove_Cycle=True,
                    twoChan=twoChan,  # if 2Ch, will add Ch1/Ch2 to filename
                )
                h5fname_sqz = self.create_FullFolderPath4file(h5fname_sqz)

                if self.oneCh:
                    self.h5fname_sqz = h5fname_sqz
                elif self.twoCh:
                    if chan == 1:
                        self.h5fname_sqz_ch2 = h5fname_sqz
                    elif chan == 0:
                        self.h5fname_sqz_ch1 = h5fname_sqz
                if self.oneCh or (self.twoCh and chan == 0):
                    if not self.concatCheck:
                        self.create_symlink4QL(
                            src=h5fname_sqz,
                            link_name=self.text_lib["QL_LNAMES"]["SQZ_H5"],
                            fpath4link=self.folder_tools.get_dirname(h5fname_sqz),
                        )
        else:
            print(
                f"Loading: {self.latest_isxd if self.latest_isxd else self.latest_tiff}"
            )
            self.print_wFrm(
                "Squeezing + additional preprocessing required before exporting to h5 necessary for motion correction"
            )
            if self.latest_isxd:
                file2use = self.latest_isxd
            elif self.latest_tiff:
                file2use = self.latest_tiff

            (
                filtered_arr,
                self.bandpass_applied,
                self.high_pass_applied,
                self.CLAHE_applied,
                self.crop_applied,
            ) = self.MCPP.preprocessing_movie(
                file2use,
                output_fname=self.basename,
                use_cropper=self.use_cropper,
            )

            self.h5fname_sqz = self.H5U.squeeze_fileNwrite(
                file2read=self.basename,
                array2use=filtered_arr,
                chan_idx=self.chan_idx,
                element_size_um=self.element_size_um,
                dimension_labels=self.dimension_labels,
                remove_Cycle=True,
                export_sample=True,
                high_pass_applied=self.high_pass_applied,
                CLAHE_applied=self.CLAHE_applied,
                bandpass_applied=self.bandpass_applied,
                crop_applied=self.crop_applied,
            )
            if not self.concatCheck:
                self.create_symlink4QL(
                    src=os.path.join(self.folder_path, self.h5fname_sqz),
                    link_name=self.text_lib["QL_LNAMES"]["SQZ_H5"],
                    fpath4link=self.folder_tools.get_dirname(self.h5fname_sqz),
                )
                self.create_symlink4QL(
                    src=os.path.join(
                        self.folder_path,
                        self.findLatest(self.file_tag["ABBR_DS"]),
                    ),
                    link_name=self.text_lib["QL_LNAMES"]["SMP_SQZ_H5"],
                    fpath4link=self.folder_tools.get_dirname(
                        os.path.join(
                            self.folder_path, self.findLatest(self.file_tag["ABBR_DS"])
                        )
                    ),
                )

    def pre_moco_h5_tools(self) -> None:
        """
        Performs pre-motion correction H5 tools.
        """
        # load unprocessed h5
        self.load_unproc_H5()

        # initiate global vars that require chan_idx output from unprocessed h5 file
        self.init_vars_from_unproc_h5()

        # check if mmap exists for 1Ch sessions
        # or if both mmaps exist for 2Ch sessions
        if not self.latest_mmap_moco or (
            not self.latest_mmap_moco_ch2 and self.latest_mmap_moco_ch1
        ):
            if not self.onePhotonCheck:
                # remove previous h5 sqz file if exists
                for sqzfile in self.list_latest_sqz:
                    self.utils.folder_tools.remove_file(sqzfile)
                # squeeze h5 file & write to file before motion correction
                self.squeeze_h5_write2file()
            elif self.onePhotonCheck and not self.list_latest_sqz:
                # squeeze h5 file & write to file before motion correction
                # will apply preprocessing for miniscope h5 files
                self.squeeze_h5_write2file()
            elif self.onePhotonCheck and self.list_latest_sqz:
                print(
                    f"Pre-processed h5 file detected, will use this for motion correction: {self.list_latest_sqz[0]}"
                )
                self.print_wFrm("Skipping H5 squeeze & preprocessing")

                self.h5fname_sqz = self.list_latest_sqz[0]
                print()
        else:
            print(
                "Found memmap from previous motion correction, so skipping H5 squeeze\n"
            )

    ######################################################
    #  cluster funcs
    ######################################################

    def start_cluster(self, n_processes: int = None) -> None:
        """
        Starts the cluster.

        Parameters:
            n_processes (int, optional): The number of processes. Defaults to None.
        """
        self.c, self.dview, self.n_processes = self.utils.caiman_utils.start_cluster(
            N_PROC=n_processes
        )
        self.print_wFrm(f"{self.n_processes} processes started")

    def restart_cluster(self, n_processes: int = None):
        """
        Restarts the cluster specifically for segmentation

        Parameters:
            n_processes (int, optional): The number of processes. Defaults to None.
        """
        # check if dview is running, will stop otherwise
        if self.dview:
            self.stop_cluster()
        # start cluster
        self.start_cluster(n_processes=n_processes)

    def stop_cluster(self, final=False):
        """
        Stops the cluster.

        Parameters:
            final (bool, optional): Whether it's the final stop. Defaults to False.
        """
        if final:
            # stop server
            self.utils.caiman_utils.stop_cluster(dview=self.dview, remove_log=True)
            print("Cluster stopped")
            print()
        else:
            # stop dview cluster
            self.dview.terminate()

    ######################################################
    #  mmap funcs
    ######################################################

    def load_mmap_ImageStack(
        self,
        mmap_fname_to_load: str | list = [],
        store4Trim: bool = False,
        store4CNMF: bool = False,
    ):
        """
        Loads the memmap file.

        Parameters:
            mmap_fname_to_load (list, optional): The mmap filename to load. Defaults to [].
            store4Trim (bool, optional): Whether to store for trimming. Defaults to False.
            store4CNMF (bool, optional): Whether to store for CNMF. Defaults to False.
        """
        # if mmap_fname_to_load is empty, will use self.mmap_of_moco
        # which is filename of the mmap of motion corrected image
        if not mmap_fname_to_load:
            mmap_fname_to_load = self.list_latest_mmap_moco
        # store mmap filename to load into self
        # self.mmap_fname = mmap_fname_to_load

        # check if string, change to list if so
        if isinstance(mmap_fname_to_load, str):
            mmap_fname_to_load = [mmap_fname_to_load]

        for idx, mmap in enumerate(mmap_fname_to_load):
            # print fname for confirmation
            self.print_wFrm(
                f"Memory Map file to load: {self.folder_tools.os_splitterNcheck(mmap, 'base')}"
            )

            if self.concatCheck:
                mmap2load = mmap
            else:
                mmap2load = os.path.join(self.folder_path, mmap)

            # load mmap file & reshapes into [frames, y, x]
            array, dims, dy, dx = self.utils.caiman_utils.load_mmap(
                mmap2load, reshape_post_moco=True
            )
            # store to self
            if store4Trim:
                self.array_to_trim.append(array)
                self.dx = dx
                self.dy = dy
                self.dims = dims
            if store4CNMF:
                self.array_for_cnmf.append(array)
                if idx == 0:
                    # whether 1 or 2Ch, will store dy, dx, dims just for 1st entry
                    # will be used for CNMF
                    # dx, dy, dims are the same for all Ch
                    self.dx = dx
                    self.dy = dy
                    self.dims = dims
        if store4Trim:
            with self.StatusPrinter.garbage_collector():
                self.moco = None

    def save_memmap_postproc(self) -> None:
        """
        Saves the post-processed mmap.
        """
        list_fname_mmap_pp = []
        if self.oneCh:
            chan_arr = [1]
        elif self.twoCh:
            chan_arr = [1, 0]
        for idx, h5fn in enumerate(self.list_h5fn_pproc):
            chan2use = chan_arr[idx]

            # make sure to write out full path to H5 to save given changes to CaImAn (07092024)
            fname2save = (
                os.path.join(self.folder_path, h5fn) if not self.concatCheck else h5fn
            )
            if self.twoCh:
                print(f"...Ch{chan2use + 1}", end="", flush=True)
            fname_mmap_postproc = self.utils.caiman_utils.save_mmap(
                fname2save=fname2save,
                chan_num=chan2use,
                base_name=self.file_tag["MMAP_MC_PROC"],
            )

            self.print_wFrm(f"Filename: {fname_mmap_postproc}")

            list_fname_mmap_pp.append(fname_mmap_postproc)

        # store fname_mmapp_postproc for segmentation
        self.fname_mmap_postproc = list_fname_mmap_pp

        # if not self.concatCheck:
        #     for fname in self.fname_mmap_postproc:
        #         self.create_symlink4QL(
        #             src=os.path.join(self.folder_path, fname),
        #             link_name=self.text_lib["QL_LNAMES"]["NORM_TFDS_MMAP"],
        #             fpath4link=self.folder_tools.get_dirname(fname),
        #         )

    def find_mmap_fname(self) -> str:
        """
        Finds the mmap filename.

        Returns:
            str: The mmap filename.
        """
        mmaps = []
        # extract latest mmap file (not Ch1)
        if self.concatCheck:
            os.chdir(self.output_pathByID)
        mmap_fname_ch2 = self.findLatest(
            filetags=self.file_tag["MMAP_MC_PROC"],
            notInclude="Ch1",
            full_path=self.concatCheck,
        )
        mmap_fname_ch1 = self.findLatest(
            filetags=self.file_tag["MMAP_MC_PROC"],
            notInclude="Ch2",
            full_path=self.concatCheck,
        )
        # store latest mmap into self
        if mmap_fname_ch2:
            self.fname_mmap_postproc.append(mmap_fname_ch2)
            mmaps.append(mmap_fname_ch2)
        if mmap_fname_ch1:
            self.fname_mmap_postproc.append(mmap_fname_ch1)
            mmaps.append(mmap_fname_ch1)

        return mmaps

    ######################################################
    #  motion correction
    ######################################################

    def motion_correction(self) -> None:
        """
        Performs motion correction.
        """
        print("Motion Correction:")
        self.print_wFrm("WARNING: This could take awhile")

        oneCh_mmap_checker = self.latest_mmap_moco != ""
        twoCh_mmap_checker = (
            self.latest_mmap_moco_ch2 != "" and self.latest_mmap_moco_ch1 != ""
        )

        # be default check is True, so motion correction will be performed
        no_moco_mmap = True

        # if either mmap for 1Ch or mmaps for 2Ch are present, will skip motion correction
        if oneCh_mmap_checker or twoCh_mmap_checker:
            no_moco_mmap = False

        if no_moco_mmap:
            self.print_wFrm("Starting cluster for parallel proc motion correction")
            # use less than default # of CPU Cores
            # otherwise, will overload GPU
            # # raise number of processes for 1Photon given smaller number of frame batches motion corrected at a time
            # if self.onePhotonCheck:
            #     self.n_proc4MOCO = 20
            self.start_cluster(n_processes=self.n_proc4MOCO)

            # prepending folder path to h5fname_sqz given CaImAn changes (07092024)
            if self.concatCheck:
                file_to_correct = [self.h5fname_sqz]
            else:
                if self.oneCh:
                    file_to_correct = [os.path.join(self.folder_path, self.h5fname_sqz)]
                elif self.twoCh:
                    file_to_correct = [
                        os.path.join(self.folder_path, h5sqz)
                        for h5sqz in [self.h5fname_sqz_ch2, self.h5fname_sqz_ch1]
                    ]

            # if self.onePhotonCheck:
            #     file_to_correct = [
            #         os.path.join(self.folder_path, self.findLatest("ABBREV"))
            #     ]

            # store dview to use
            dview_to_use = self.dview

            user_set_params = self.findLatest(
                filetags=[self.file_tag["USER_MC_PARAMS"], self.file_tag["JSON"]],
                full_path=self.concatCheck,
            )
            if user_set_params:
                self.print_wFrm(f"USING USER SET MC PARAMS: {user_set_params}")
                params = self.saveNloadUtils.load_file(fname=user_set_params)
            else:
                params = None

            self.moco = NoRMCorre(
                h5filename=file_to_correct,
                dview=dview_to_use,
                params=params,
                onePhotonCheck=self.onePhotonCheck,
                mc_iter=self.mc_iter,
                compute_metrics=self.compute_metrics,
                separate_channels=self.separate_channels,
            )
            # order of list entries in moco
            # - 1) Ch2 (green)
            # - 2) Ch1 (red)

            # store fname of motion corrected memmap
            # turn list entries into strings
            # will be reconverted into a list when placed into list_latest_mmap_moco
            if self.oneCh:
                self.mmap_of_moco = self.moco[0].fname_tot_els.copy()
                self.mmap_of_moco = self.mmap_of_moco[0]
                mmap4ql = self.mmap_of_moco
            elif self.twoCh:
                self.mmap_of_moco_ch2 = self.moco[0].fname_tot_els.copy()
                self.mmap_of_moco_ch1 = self.moco[1].fname_tot_els.copy()
                # index 0th entry
                self.mmap_of_moco_ch2 = self.mmap_of_moco_ch2[0]
                self.mmap_of_moco_ch1 = self.mmap_of_moco_ch1[0]
                mmap4ql = self.mmap_of_moco_ch2

            if not self.concatCheck:
                mmap4ql = os.path.join(self.folder_path, mmap4ql)

                self.create_symlink4QL(
                    src=mmap4ql,
                    link_name=self.text_lib["QL_LNAMES"]["NORM_MMAP"],
                    fpath4link=self.folder_tools.get_dirname(mmap4ql),
                )
            # stop cluster to clear memory
            self.print_wFrm("Stopping cluster", end="", flush=True)
            self.stop_cluster()
            self.print_done_small_proc(new_line=False)

            for idx, _ in enumerate(self.chan_idx):
                if idx == 0:
                    # for 1Ch, will use self.mmap_of_moco
                    # for 2Ch, will use self.mmap_of_moco_ch2
                    mmap = self.mmap_of_moco if self.oneCh else self.mmap_of_moco_ch2
                    channel = 2
                elif idx == 1:
                    # only present for 2Ch
                    mmap = self.mmap_of_moco_ch1
                    channel = 1
                # print chan_str if 2Ch, otherwise, empty string
                chan_str = "" if self.oneCh else f" for Ch{channel}"
                self.print_wFrm(
                    f"Motion corrected image stack{chan_str} saved to: {self.folder_tools.os_splitterNcheck(mmap, 'base')}"
                )
        else:
            self.print_wFrm("Found memmap from previous motion correction")
            self.print_wFrm("Skipping motion correction step")
            if self.oneCh:
                # store into self
                self.mmap_of_moco = self.latest_mmap_moco
                self.print_wFrm(f"Will use this memmap: {self.mmap_of_moco}")
            elif self.twoCh:
                # store into self
                self.mmap_of_moco_ch2 = self.latest_mmap_moco_ch2
                self.mmap_of_moco_ch1 = self.latest_mmap_moco_ch1
                # print results
                self.print_wFrm(
                    f"Will use this memmap for Ch2: {self.mmap_of_moco_ch2}"
                )
                self.print_wFrm(
                    f"Will use this memmap for Ch1: {self.mmap_of_moco_ch1}"
                )
        # append to list_latest_mmap_moco
        if self.oneCh:
            self.list_latest_mmap_moco.append(self.mmap_of_moco)
        elif self.twoCh:
            self.list_latest_mmap_moco.append(self.mmap_of_moco_ch2)
            self.list_latest_mmap_moco.append(self.mmap_of_moco_ch1)
        print()

    ######################################################
    #  image stack funcs
    ######################################################

    def run_ISUtils4Segmentation(self, frame_threshold: int = 20000) -> None:
        """
        Runs Image Stack Utils to prepare for Segmentation.

        Parameters:
            frame_threshold (int): The threshold value for the total number of frames. If the total number of frames
                                    is greater than this threshold, downsampling will occur before temporal filtering. Otherwise, temporal filtering will occur first.

        Returns:
            None
        """
        # set twoChanArr to [2, 1] if len of chan_idx is 2
        # otherwise, set to empty list
        self.twoChanArr = []
        if self.twoCh:
            self.twoChanArr = [2, 1]

        print("Running Image Stack Utils to prepare for Segmentation:")
        # trim stack & save avg as tif
        self._ISU_trimStack()

        # check if total frames is greater than frame_threshold
        # if so, will downsample before temporal filtering
        # otherwise, will temporal filter first
        DS_array, TF_array, TFDS_array = [], [], []
        if self.total_frames > frame_threshold:
            print("Total frames greater than frame threshold")
            print("Downsampling before applying Temporal Filter")
            DS_array = self._ISU_downsample(array2ds=self.trimmed_array, Temp_Exp=False)
            TFDS_array = self._ISU_tempFilter(array2tf=DS_array, Downsample=True)
        else:
            TF_array = self._ISU_tempFilter(
                array2tf=self.trimmed_array,
                Downsample=False,
            )
            TFDS_array = self._ISU_downsample(array2ds=TF_array, Temp_Exp=True)

        # normalize & correct image stack
        # this result is written as a h5 in a separate func
        # resulting h5 is saved as mmap in a separate func as well
        self._ISU_normalizeNcorrect_tempfiltDSArr(array2norm=TFDS_array)
        # clear vars
        with self.StatusPrinter.garbage_collector():
            DS_array, TF_array, TFDS_array = None, None, None

    def _ISU_chan2use_utils(self, idx: int, str: str) -> list:
        """
        Returns the channel to use based on the given index, twoChanArr.
        twoChanArr[0] = 2, twoChanArr[1] = 1

        Parameters:
            idx (int): The index to determine the channel(s) to use.
            str (str): The string to print before the channel information.

        Returns:
            chan2use (list): The channel(s) to use.

        Raises:
            ValueError: If the index is greater than 2.
        """
        if idx > 2:
            raise ValueError("Index must be less than 2")
        if self.oneCh:
            chan2use = []
        elif self.twoCh:
            chan2use = self.twoChanArr[idx]
            print(f"{str} Calcium Channel {chan2use}:")
        return chan2use

    def _ISU_trimStack(self) -> None:
        """
        Trims the stack of images and saves the trimmed array as well as the trim coordinates.

        Returns:
            None
        """
        # trim stack & save avg as tif
        self.trimmed_array = []
        self.trimYX = []
        for idx, array in enumerate(self.array_to_trim):
            # set chan2use & print 2Ch str if so
            # chan2use = self._ISU_chan2use_utils(idx, "Trimming")

            if not self.onePhotonCheck:
                # trim stack
                # trimmed_array, trimYX = self.ISUtils.trim2pStack(
                #     array_to_trim=array, store=False
                # )
                trimmed_array = array
                trimYX = None
            # elif self.onePhotonCheck and self.high_pass_applied:
            #     print(
            #         "---Skipping trimming & min-z projection removal step for 1Photon data given high-pass filter was applied---"
            #     )
            #     trimmed_array = array
            #     trimYX = None
            # elif self.onePhotonCheck and not self.high_pass_applied:
            elif self.onePhotonCheck:
                # trimmed_array = self.ISUtils.min_zProj_removal(array_to_use=array)
                trimmed_array = array
                trimYX = None

            self.trimmed_array.append(trimmed_array)
            self.trimYX.append(trimYX)

            # save as tif
            # print("| Saving Calcium Channel average tif")
            # self._ISU_saveImage(array_to_use=trimmed_array, twoChanInt=chan2use)

    def _ISU_tempFilter(self, array2tf: list[np.ndarray], Downsample=False) -> list:
        """
        Applies temporal filter to the array.

        Parameters:
            array2tf (list[np.ndarray]): The list of arrays to apply the temporal filter to.
            Downsample (bool): Whether to downsample the array.

        Returns:
            TF_array (list[np.ndarray]): The array after applying the temporal filter.
        """
        TF_array = []
        for idx, array in enumerate(array2tf):
            # set chan2use & print 2Ch str if so
            chan2use = self._ISU_chan2use_utils(idx, "Applying Temporal Filter to")

            # apply temporal filter
            TFD = self.ISUtils.caTempFilter(array_to_use=array, store=False)
            TF_array.append(TFD)

            # save as tif
            print("| Saving average of temporally filtered image stack")
            self._ISU_saveImage(
                array_to_use=TFD,
                twoChanInt=chan2use,
                Temp_Exp=True,
                Downsample=Downsample,
            )
        with self.StatusPrinter.garbage_collector():
            array2tf = None
        return TF_array

    def _ISU_downsample(
        self, array2ds: list[np.ndarray], Temp_Exp: bool = False
    ) -> list:
        """
        Downsamples the array.

        Parameters:
            array2ds (list[np.ndarray]): The list of arrays to downsample.
            Temp_Exp (bool): Whether to apply temporal filter before downsampling.

        Returns:
            DS_array (list[np.ndarray]): The list of downsampled arrays.
        """
        DS_array = []
        for idx, array in enumerate(array2ds):
            # set chan2use & print 2Ch str if so
            chan2use = self._ISU_chan2use_utils(idx, "Downsampling")

            # downsample stack
            DSD = self.ISUtils.downsampleStack(
                array_to_ds=array,
            )

            DS_array.append(DSD)

            # save as tif
            print("| Saving average of downsampled image stack")
            self._ISU_saveImage(
                array_to_use=DSD,
                twoChanInt=chan2use,
                Temp_Exp=Temp_Exp,
                Downsample=True,
            )
        with self.StatusPrinter.garbage_collector():
            array2ds = None
        return DS_array

    def _ISU_normalizeNcorrect_tempfiltDSArr(
        self, array2norm: list[np.ndarray]
    ) -> None:
        """
        Normalize and correct the tempfilteredDS_array.

        This method applies normalization and correction to the tempfilteredDS_array using the ISUtils.normalizeNcorrect_ImageStack method.

        Parameters:
            array2norm (list[np.ndarray]): The list of arrays to be normalized and corrected.
        """
        self.norm_uint_tempfilteredDS_arr = []
        for idx, array in enumerate(array2norm):
            # don't need chan2use, but print 2Ch str if so
            _ = self._ISU_chan2use_utils(
                idx, "Normalizing & Correcting Temporally Filtered"
            )
            # normalize & correct image stack
            # - stack at this point is temporally filtered & downsampled stack
            norm_uint_tempfilteredDS_arr = self.ISUtils.normalizeNcorrect_ImageStack(
                array_to_use=array
            )
            self.norm_uint_tempfilteredDS_arr.append(norm_uint_tempfilteredDS_arr)
            print()

    def _ISU_saveImage(
        self,
        array_to_use: np.ndarray,
        twoChanInt: int,
        Temp_Exp: bool = False,
        Downsample: bool = False,
        bit_range: list[int] = [8, 16],
    ) -> None:
        """
        Save the average of the image stack as a TIFF file.

        This method uses the ISUtils.avg_CaCh_tifwriter to save the average of the given image stack
        as a TIFF file. It supports various options such as temporal filtering, downsampling, and
        bit range conversion.

        Parameters:
            array_to_use (ndarray): The image stack to be saved.
            twoChanInt (int): The channel integer to use for the filename.
            Temp_Exp (bool, optional): Whether to adjust the filename to account for exponential decay filter. Defaults to False.
            Downsample (bool, optional): Whether to adjust the filename to account for downsampling. Defaults to False.
            bit_range (list[int], optional): List of bit ranges to convert the normalized array to. Defaults to [16].

        Returns:
            None

        Note:
            This method will save one TIFF file for each bit range specified in the bit_range list.
        """

        for bit in bit_range:
            self.print_wFrm(f"Saving average of image stack as {bit}-bit TIFF")
            self.ISUtils.avg_CaCh_tifwriter(
                array_to_use=array_to_use,
                Temp_Exp=Temp_Exp,
                Downsample=Downsample,
                twoChanInt=twoChanInt,
                bit_range=bit,
            )
            print()

    def clear_ISUtils(self) -> None:
        """
        Clears the ISUtils attribute and performs garbage collection.

        This method sets the ISUtils attribute to None and then performs garbage collection to free up memory.
        """
        with self.StatusPrinter.garbage_collector():
            self.ISUtils = None

    ######################################################
    #  CNMF funcs
    ######################################################

    def run_CNMF(self) -> None:
        """
        Initializes the CNMF_Utils object.
        """

        def _onePhotonCheck_utils() -> None:
            """
            Sets the onePhotonCheck attribute to True. If MotionCorr parameters file contains onePhoton in title. This is a failsafe mechanism.
            """
            paramsFile = self.findLatest(
                "onePhoton",
                path2check=os.path.join(
                    self.folder_path, self.text_lib["Folders"]["PARAMS"]
                ),
            )
            if paramsFile:
                self.onePhotonCheck = True

        if not self.onePhotonCheck and not self.concatCheck:
            _onePhotonCheck_utils()

        for idx, array in enumerate(self.array_for_cnmf):
            if (
                "Ch2" in self.fname_mmap_postproc[idx]
                and len(self.fname_mmap_postproc) > 1
            ):
                basename2use = self.basename + "_Ch2"
            elif (
                "Ch1" in self.fname_mmap_postproc[idx]
                and len(self.fname_mmap_postproc) > 1
            ):
                basename2use = self.basename + "_Ch1"
            else:
                basename2use = self.basename

            print(
                f"Running CNMF for {self.fname_mmap_postproc[idx]} - {idx + 1} of {len(self.array_for_cnmf)}"
            )

            user_set_params = self.findLatest(
                filetags=[self.file_tag["USER_CNMF_PARAMS"], self.file_tag["JSON"]],
                full_path=self.concatCheck,
            )
            if user_set_params:
                self.print_wFrm(f"USING USER-SET CNMF PARAMETERS: {user_set_params}")
                params = self.saveNloadUtils.load_file(fname=user_set_params)
                rf = params["RF"]
                stride = params["STRIDE"]
            else:
                params = None
                par_idx = 0 if not self.onePhotonCheck else 1
                self.CNMFpar2use = {k: v[par_idx] for k, v in self.CNMFpar.items()}
                rf = self.CNMFpar2use["RF"]
                stride = self.CNMFpar2use["STRIDE"]

            self.CNMFU = CNMF_Utils(
                basename=basename2use,
                Ca_Array=array,
                dview=self.dview,
                dx=self.dx,
                dy=self.dy,
                dims=self.dims,
                n_processes=self.n_proc4CNMF,
                onePhotonCheck=self.onePhotonCheck,
                params=params,
            )

            # find initial patches
            self.CNMFU.find_initial_patches(rf=rf, stride=stride)

            # evaluate found patches
            self.CNMFU.comp_evaluator()

            # plot contours
            folder_path = None
            if self.concatCheck:
                folder_path = self.output_pathByID
            self.CNMFU.comp_contour_plotter(folder_path=folder_path)

            # RE-RUN seeded CNMF on accepted patches to refine & perform deconvolution
            self.CNMFU.fill_in_accepted_patches()
            # refine patches
            self.CNMFU.refine_patches(stride=None, rf=None)

            if self.concatCheck:
                fname_mmap = self.fname_mmap_postproc[idx]
            else:
                fname_mmap = os.path.join(
                    self.folder_path, self.fname_mmap_postproc[idx]
                )

            Yr, _, _ = self.utils.caiman_utils.load_mmap(fname_mmap=fname_mmap)
            self.CNMFU.calc_F_dff(Yr=Yr)

            folder_path_concat = None
            folder_path_subj = []

            if self.concatCheck:
                folder_path_concat = self.output_pathByID
                folder_path_subj = self.folder_path

            self.CNMFU.createNexport_segDict(
                concatCheck=self.concatCheck,
                folder_path_concat=folder_path_concat,
                folder_path_subj=folder_path_subj,
            )

            if not self.concatCheck:
                for ftype in ["H5", "MAT", "PKL"]:
                    latest_sd = self.findLatest(
                        [self.file_tag["SD"], self.file_tag[ftype]]
                    )
                    self.create_symlink4QL(
                        src=os.path.join(self.folder_path, latest_sd),
                        link_name=self.text_lib["QL_LNAMES"][f"SD_{ftype}"],
                    )

    ######################################################
    # view motion correction with denoised
    ######################################################

    def find_residual_postSeg(self, downsample_ratio: float = 0.2) -> None:
        """
        Finds the residuals of post-segmentation by comparing the original movie with the denoised movie reconstructed from CNMF.

        Parameters:
            downsample_ratio (float): The ratio by which to downsample the concatenated residual movie (default is 0.2).
        """
        TKEEPER = self.time_utils.TimeKeeper()
        # extract element size if not already done
        # esp if segment is only done
        if self.element_size_um is None and not self.latest_isxd:
            if self.concatCheck:
                h52use = self.latest_h5[0]
            else:
                h52use = self.latest_h5
            self.element_size_um = self.H5U.extract_element_size(h52use)
        else:
            self.element_size_um = None

        print("Finding Residuals of Post-Segmentation")
        self.print_wFrm("Preparing reconstructed movie/residuals")
        # load original movie, normalized & converted to 16-bit unsigned

        fname2load = []
        if self.concatCheck:
            fname2load = self.fname_mmap_postproc
        else:
            fname2load = [
                os.path.join(self.folder_path, f) for f in self.fname_mmap_postproc
            ]

        for idx, fname in enumerate(fname2load):
            wCH = None
            if "Ch2" in fname:
                wCH = "Ch2"
            elif "Ch1" in fname:
                wCH = "Ch1"
            wCH_tag = f"_{wCH}" if wCH is not None else ""

            m_els = movie_utils.load_movie(fname)

            # extract denoised movies, with & without background
            denoised_noback = self._create_recontructed_movie()
            denoised_wback = self._create_recontructed_movie(wBackground=True)

            self.print_wFrm("Creating reconstructed movie (C dot A + b dot F)")
            # normalize & convert to uint
            CdotA_bDotF = movie_utils.add_caption_to_movie(
                movie=movie_utils.normNconvert2uint(denoised_wback),
                text=None,
            )
            movies2do = [(CdotA_bDotF, self.file_tag["PS_CDA_BDF_MOV"])]

            if self.export_postseg_residuals:
                self.print_wFrm("Creating reconstructed movie (C dot A)")
                # normalize & convert to uint
                # add caption to first 30 frames
                CdotA = movie_utils.add_caption_to_movie(
                    movie=movie_utils.normNconvert2uint(denoised_noback),
                    text="Accepted Components",
                )

                self.print_wFrm("Creating residuals (C dot A vs non-segmented noise)")

                # noncap_noise = full film - (C dot A + b dot f)
                # add caption to first 30 frames
                noncap_noise = movie_utils.add_caption_to_movie(
                    movie=movie_utils.normNconvert2uint(m_els - denoised_wback),
                    text="Non-segmented Noise",
                )

                # concatenate C dot A & noncap_noise
                concatenated_residual_movie = movie_utils.concatenate_movies(
                    [CdotA, noncap_noise], axis=2, use_caiman=False
                )

                movies2do.append(
                    (concatenated_residual_movie, self.file_tag["PS_CON_RES_MOV"])
                )
            else:
                noncap_noise = None
                concatenated_residual_movie = None

            print()

            str2print = "Exporting reconstructed movie (C dot A + b dot F)"

            if self.export_postseg_residuals:
                str2print += (
                    " and residuals post-segmentation (C dot A vs non-segmented noise)"
                )
            print(str2print)
            # save residual movie
            for movies2save, ftag in movies2do:
                moviefname = ftag + wCH_tag
                for ftype in [self.file_tag["AVI"], self.file_tag["H5"]]:
                    if self.concatCheck:
                        moviefname = os.path.join(self.output_pathByID, moviefname)
                    else:
                        moviefname = os.path.join(self.folder_path, moviefname)

                    self._save_residual_movie(movies2save, moviefname, ftype)

                    if (
                        movies2save is concatenated_residual_movie
                        and ftype == self.file_tag["AVI"]
                    ):
                        if not self.concatCheck:
                            self.create_symlink4QL(
                                src=moviefname + self.file_tag["AVI"],
                                link_name=self.text_lib["QL_LNAMES"][
                                    "RES_MOV_CCAT_AVI"
                                ],
                                fpath4link=self.folder_tools.get_dirname(moviefname),
                            )
            TKEEPER.setEndNprintDuration()

        print()
        # clear for memory
        with self.StatusPrinter.garbage_collector():
            CdotA, noncap_noise, m_els = None, None, None
            concatenated_residual_movie = None
            denoised_noback, denoised_wback = None, None

    def view_movie(
        self,
        fname: list[str] = [],
        downsample_ratio: float = 0.2,
        magnification: int = 2,
        fr: int = 30,
        gain: int = 3,
        memory_mapped: bool = False,
        postSeg: bool = False,
    ) -> None:
        """
        View a movie.

        Parameters:
            fname (list): List of file names of the movie frames.
            downsample_ratio (float): Ratio by which to downsample the movie frames.
            magnification (int): Magnification factor for displaying the movie.
            fr (int): Frame rate of the movie playback.
            gain (int): Gain value for adjusting the movie brightness.
            memory_mapped (bool): Flag indicating whether the movie is memory-mapped.
            postSeg (bool): Flag indicating whether the movie is post-segmented.
        """
        if postSeg:
            mov = self._create_recontructed_movie()
        else:
            mov = movie_utils.load_movie(fname, is_memory_mapped=memory_mapped)
        movie_utils.process_and_play_movie(
            mov, downsample_ratio, fr, gain, magnification
        )

    def _create_recontructed_movie(self, wBackground: bool = False) -> np.ndarray:
        """
        Creates a reconstructed movie by applying Non-Negative Matrix Factorization (NMF) to the data.

        Parameters:
            wBackground (bool, optional): Whether to include the background in the reconstructed movie. Defaults to False.

        Returns:
            numpy.ndarray: The reconstructed movie.
        """

        denoised = movie_utils.create_denoised_movie(
            cnm_estimates=self.CNMFU.NonNegMatrix_post_refining.estimates,
            dims=self.dims,
            wBackground=wBackground,
        )
        return denoised

    def _save_residual_movie(
        self, residual_movie: np.ndarray, fname: str, ftag: str
    ) -> None:
        """
        Save the residual movie to a file.

        Parameters:
            residual_movie (numpy.ndarray): The residual movie to be saved.
            fname (str): The base filename for the saved movie.
            ftag (str): The file tag to be appended to the filename.
        """

        movie_utils.save_movie(
            residual_movie,
            fname,
            ftag,
            element_size_um=self.element_size_um,
            use_caiman=False,
        )

        fname2print = self.folder_tools.os_splitterNcheck(fname, baseORpath="base")

        if "CdotA" in fname2print and "NonCapNoise" not in fname2print:
            mtype = "C dot A"
        elif "NonSegNoise" in fname2print and "CdotA" not in fname2print:
            mtype = "Non-segmented Noise"
        elif "CdotA" in fname2print and "NonSegNoise" in fname2print:
            mtype = "C dot A & Non-segmented Noise"

        self.print_wFrm(f"Residual movie ({mtype}) saved as: {fname2print}{ftag}")

    ######################################################
    #  for loop ending func
    ######################################################

    def endIter_funcs(self) -> None:
        """
        Clears variables to save space for the next iteration.

        This method clears variables by initializing global variables and manually activating the garbage collector.
        """

        print("Clearing variables to save space for next iteration", end="", flush=True)
        # clears variables and activates garbage collector
        with self.StatusPrinter.garbage_collector():
            # clear vars via init iterable vars
            self._init_vars4Iter()
        self.print_done_small_proc()
