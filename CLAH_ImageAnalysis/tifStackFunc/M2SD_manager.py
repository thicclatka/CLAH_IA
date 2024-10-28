import os
import numpy as np
from tqdm import tqdm
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.tifStackFunc import CNMF_Utils
from CLAH_ImageAnalysis.tifStackFunc import H5_Utils
from CLAH_ImageAnalysis.tifStackFunc import ImageStack_Utils
from CLAH_ImageAnalysis.tifStackFunc import Movie_Utils as movie_utils
from CLAH_ImageAnalysis.tifStackFunc import NoRMCorre
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


class M2SD_manager(BC):
    """
    M2SD_Utils class provides utility functions for image analysis and processing.

    Parameters:
        folder_path (list, optional): The path to the folder. Defaults to [].

    Attributes:
        dayPath (str): The path to the day.
        dayDir (str): The day directory.
        sess2process (list): The sessions to process.
        H5U (H5_Utils): An instance of the H5_Utils class.
        c (CaimanCluster): The Caiman cluster object.
        dview (CaimanDView): The Caiman distributed view object.
        n_processes (int): The number of processes.
        array_for_cnmf (ndarray): The array for CNMF.
        array_to_trim (ndarray): The array to trim.
        trimmed_array (ndarray): The trimmed array.
        basename (str): The basename.
        trimYX (tuple): The trim YX coordinates.
        CNMFU (CNMF_Utils): An instance of the CNMF_Utils class.
        ISUtils (ImageStack_Utils): An instance of the ImageStack_Utils class.
        mmap_fname (str): The mmap filename.
        mmap_of_moco (str): The mmap filename of motion corrected image.
        chan_idx (int): The channel index.
        element_size_um (float): The element size in micrometers.
        hfsiz (int): The h5 file size.
        h5filename (str): The h5 filename.
        h5fname_sqz (str): The squeezed h5 filename.
        h5filename_postproc (str): The post-processed h5 filename.
        dimension_labels (list): The dimension labels.
        latest_eMC (str): The latest eMC file.
        latest_h5 (str): The latest h5 file.
        latest_h5sqz (str): The latest squeezed h5 file.
        latest_mmap_moco (str): The latest mmap file for motion correction.
        sess_idx (int): The session index.
        sess_num (int): The session number.
        folder_path (str): The folder path.
        folder_name (str): The folder name.
        norm_uint_tempfilteredDS_arr (ndarray): The normalized and temporally filtered array.
        dx (int): The x dimension.
        dy (int): The y dimension.
        dims (tuple): The dimensions.
        moco (NoRMCorre): An instance of the NoRMCorre class.
        fname_mmap_postproc (str): The filename of the post-processed mmap.

    """

    ######################################################
    #  funcs run at init
    ######################################################
    def __init__(
        self,
        program_name: str,
        path: str | list = [],
        sess2process: list = [],
        motion_correct: bool = False,
        segment: bool = False,
        n_proc4MOCO: int | None = None,
        n_proc4CNMF: int | None = None,
        concatCheck: bool = False,
        prev_sd_varnames: bool = False,
        kernel_window_size: int | None = None,
        # method_init=None,
        # meth_deconv=None,
    ):
        self.program_name = program_name
        self.__version__ = "0.1.0"
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

        # init global vars
        self.static_class_var_init(
            folder_path=path,
            sess2process=sess2process,
            motion_correct=motion_correct,
            segment=segment,
            n_proc4MOCO=n_proc4MOCO,
            n_proc4CNMF=n_proc4CNMF,
            concatCheck=concatCheck,
            prev_sd_varnames=prev_sd_varnames,
            kernel_window_size=kernel_window_size,
            # method_init,
            # meth_deconv,
        )

    def static_class_var_init(
        self,
        folder_path: str | list,
        sess2process: list,
        motion_correct: bool,
        segment: bool,
        n_proc4MOCO: int | None,
        n_proc4CNMF: int | None,
        concatCheck: bool,
        prev_sd_varnames: bool,
        kernel_window_size: int | None,
        # method_init,
        # meth_deconv,
    ) -> None:
        """
        Initializes the static class variables.

        Parameters:
            folder_path (str | list): The path to the folder.
            sess2process (list): The sessions to process.
            motion_correct (bool): Whether to motion correct the data.
            segment (bool): Whether to segment the data.
            n_proc4MOCO (int | None): The number of processes to use for motion correction.
            n_proc4CNMF (int | None): The number of processes to use for CNMF.
            concatCheck (bool): Whether to check for concatenation.
            prev_sd_varnames (bool): Whether to use previous SD variable names.
            kernel_window_size (int | None): The kernel window size.
        """
        BC.static_class_var_init(
            self,
            folder_path=folder_path,
            file_of_interest=self.text_lib["selector"]["tags"]["EMC"],
            selection_made=sess2process,
            select_by_ID=concatCheck,
        )

        self.CNMFpar = self.enum2dict(TSF_enum.CNMF_Params)
        self.motion_correct = motion_correct
        self.segment = segment
        self.n_proc4MOCO = n_proc4MOCO
        self.n_proc4CNMF = n_proc4CNMF
        self.concatCheck = concatCheck
        self.prev_sd_varnames = prev_sd_varnames
        self.kernel_window_size = kernel_window_size
        # self.method_init = method_init
        # self.meth_deconv = meth_deconv

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

        # latest files
        self.latest_eMC = None
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

        # onePhotonCheck
        self.onePhotonCheck = False

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
        # # print analysis header to denote which folder is being analyzed
        # self._print_analysis_header()

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
                    self.file_tag["CON_RES_MOV"],
                    self.file_tag["NOISE_MOV"],
                    self.file_tag["CDA_MOV"],
                ],
                full_path=full_path,
            )

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
                notInclude=[self.file_tag["MMAP_BASE"], "_Ch1", "_Ch2"],
                full_path=full_path,
            )
            # for 2Ch sessions
            mmap_moco_ch1 = self.findLatest(
                [self.file_tag["MMAP"], self.file_tag["SQZ"] + "_Ch1"],
                notInclude=[self.file_tag["MMAP_BASE"], "_Ch2"],
                full_path=full_path,
            )
            mmap_moco_ch2 = self.findLatest(
                [self.file_tag["MMAP"], self.file_tag["SQZ"] + "_Ch2"],
                notInclude=[self.file_tag["MMAP_BASE"], "_Ch1"],
                full_path=full_path,
            )
            return (
                eMC_file,
                h5_file,
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
            self.latest_h5concat = []
            for directory in self.folder_path:
                os.chdir(directory)
                latest_eMC, latest_h5, _, _, _, _, _, _ = _findFiles(
                    full_path=self.concatCheck
                )

                self.latest_eMC.append(latest_eMC)
                self.latest_h5.append(latest_h5)
            os.chdir(self.output_pathByID)
            (
                self.latest_eMC,
                self.latest_h5concat,
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

        if self.concatCheck:
            self.basename = self.basename.replace("pre", "")

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
        if self.segCh is None:
            self.onePhotonCheck = True

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
                h52use = self.latest_h5concat
            else:
                self.print_wFrm("Using files:")
                for h5 in self.latest_h5:
                    self.print_wFrm(f"{h5}", frame_num=1)
                concath5name = f"{self.basename}{self.file_tag['CYCLE']}{self.file_tag['CODE']}{self.file_tag['ELEMENT']}{self.file_tag['CODE']}"
                self.fname_concat = os.path.join(self.output_pathByID, concath5name)
                h52use = self.H5U.concatH5s(
                    H5s=self.latest_h5, fname_concat=self.fname_concat
                )
            self.print_done_small_proc()
        else:
            h52use = self.latest_h5

        (
            self.h5filename,
            self.hfsiz,
            self.info,
            self.segCh,
            self.chan_idx,
            self.element_size_um,
        ) = self.H5U.read_file4MOCO(h52use)

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
            self.list_h5fn_pproc[idx] = self.H5U.write_to_file(
                array_to_write=array,
                filename=h5fname,
                chan_idx=[chan],
                element_size_um=self.element_size_um,
                dimension_labels=self.dimension_labels,
                date=True,
                return_fname=True,
                twoChan=twoChan,
            )
        # clear self.norm_uint_tempfilteredDS_arr for memory
        with self.StatusPrinter.garbage_collector():
            self.norm_uint_tempfilteredDS_arr = None

    def squeeze_h5_write2file(self) -> None:
        """
        Squeezes the H5 file and writes it to a file.
        """
        print(f"Loading {self.h5filename}")
        if not self.onePhotonCheck:
            self.print_wFrm(
                "squeezing array from 5 to 3 dimensions (for Motion Correction)"
            )
            # self.dimension_labels = ["t", "y", "x"]

            twoChan = False
            if self.twoCh:
                self.print_wFrm(
                    "With 2 channels, will squeeze each channel separately:"
                )
            print()
            twoChan = True

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
                if self.oneCh:
                    self.h5fname_sqz = h5fname_sqz
                elif self.twoCh:
                    if chan == 0:
                        self.h5fname_sqz_ch2 = h5fname_sqz
                    elif chan == 1:
                        self.h5fname_sqz_ch1 = h5fname_sqz
        else:
            # import necessary libraries for miniscope h5 file preprocessing
            from skimage import img_as_uint

            self.print_wFrm(
                "Miniscope h5 file detected, squeezing + additional preprocessing will be done"
            )
            hf_arr, _ = self.H5U.read_file(self.h5filename)
            filtered_arr = []
            if self.kernel_window_size is None:
                default_kws = 20
                self.kernel_window_size = input(
                    f"Please enter a kernel window size necessary for background subtraction, in pixels (By default, will set it to {default_kws}): "
                )
                if self.kernel_window_size == "":
                    self.kernel_window_size = default_kws
                else:
                    self.kernel_window_size = int(self.kernel_window_size)
            self.print_wFrm(
                f"Using window size of {self.kernel_window_size} pixels for background subtraction"
            )

            # apply morphology tophat filter to each frame
            for frame in tqdm(hf_arr["imaging"], desc="Preprocessing 1-photon data"):
                # # subtract min value to remove glow/vignette effect
                frame2use = frame - frame.min()
                # denoise frame
                frame2use = self.dep.filter_utils.apply_median_blur_filter(
                    array_stack=frame2use, window_size=3
                )
                # frame2use = self.dep.filter_utils.apply_morphology_tophat_filter(
                #     array_stack=frame2use, window_size=self.kernel_window_size
                # )

                # use high-pass filter to remove low-frequency signal
                # via caiman funcs
                frame2use = self.utils.caiman_utils.apply_high_pass_filter_space(
                    frame2use, gSig_filt=(2, 2)
                )
                frame2use = (frame2use - frame2use.min()) / (
                    frame2use.max() - frame2use.min()
                )
                frame2use = img_as_uint(frame2use)
                frame2use = self.utils.image_utils.resize_to_square(frame2use)
                filtered_arr.append(frame2use)

            filtered_arr = np.array(filtered_arr)
            # give artifact in the beginning of the array
            # make first 5 frames the same as the 5th frame
            filtered_arr[:5] = filtered_arr[5]

            self.h5fname_sqz = self.H5U.squeeze_fileNwrite(
                file2read=self.h5filename,
                array2use=filtered_arr,
                chan_idx=self.chan_idx,
                element_size_um=self.element_size_um,
                dimension_labels=self.dimension_labels,
                remove_Cycle=True,
                window_size=self.kernel_window_size,
                export_sample=True,
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
                fname2save=fname2save, chan_num=chan2use
            )

            self.print_wFrm(f"Filename: {fname_mmap_postproc}")

            list_fname_mmap_pp.append(fname_mmap_postproc)

        # store first entry into fname_mmapp_postproc for segmentation
        self.fname_mmap_postproc = list_fname_mmap_pp[0]

    def find_mmap_fname(self) -> str:
        """
        Finds the mmap filename.

        Returns:
            str: The mmap filename.
        """
        # extract latest mmap file (not Ch1)
        if self.concatCheck:
            os.chdir(self.output_pathByID)
        mmap_fname = self.findLatest(
            self.file_tag["MMAP_BASE"], notInclude="Ch1", full_path=self.concatCheck
        )
        # store latest mmap into self
        self.fname_mmap_postproc = mmap_fname
        return mmap_fname

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

            self.moco = NoRMCorre(
                h5filename=file_to_correct,
                dview=dview_to_use,
                onePhotonCheck=self.onePhotonCheck,
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
            elif self.twoCh:
                self.mmap_of_moco_ch2 = self.moco[0].fname_tot_els.copy()
                self.mmap_of_moco_ch1 = self.moco[1].fname_tot_els.copy()
                # index 0th entry
                self.mmap_of_moco_ch2 = self.mmap_of_moco_ch2[0]
                self.mmap_of_moco_ch1 = self.mmap_of_moco_ch1[0]

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
            chan2use = self._ISU_chan2use_utils(idx, "Trimming")

            if not self.onePhotonCheck:
                # trim stack
                trimmed_array, trimYX = self.ISUtils.trim2pStack(
                    array_to_trim=array, store=False
                )
            else:
                trimmed_array = array
                trimYX = None

            self.trimmed_array.append(trimmed_array)
            self.trimYX.append(trimYX)

            # save as tif
            print("| Saving Calcium Channel average tif")
            self._ISU_saveImage(array_to_use=trimmed_array, twoChanInt=chan2use)

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

            if not self.onePhotonCheck:
                # downsample stack
                DSD = self.ISUtils.downsampleStack(array_to_ds=array, DS_factor=2)
            else:
                DSD = array

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
    ):
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

    def initializeCNMF(self) -> None:
        """
        Initializes the CNMF_Utils object.
        """
        self.CNMFU = CNMF_Utils(
            basename=self.basename,
            Ca_Array=self.array_for_cnmf[0],
            dview=self.dview,
            dx=self.dx,
            dy=self.dy,
            dims=self.dims,
            n_processes=self.n_proc4CNMF,
            # method_init=self.method_init,
            # meth_deconv=self.meth_deconv,
        )

    def find_init_patches_viaCNMF(self) -> None:
        """
        Finds initial patches using CNMF.

        This method uses CNMF to find initial patches based on the specified receptive field (rf) and stride.
        """
        self.CNMFU.find_initial_patches(
            rf=self.CNMFpar["RF"], stride=self.CNMFpar["STRIDE"]
        )

    def evaluate_found_patches(self) -> None:
        """
        Evaluates the found patches using the comp_evaluator method of the CNMFU object.

        This method is responsible for evaluating the patches found by the CNMFU object.
        It calls the comp_evaluator method to perform the evaluation.
        """
        self.CNMFU.comp_evaluator()

    def plot_contours(self) -> None:
        """
        Plots the contours of the components using the comp_contour_plotter method.
        """
        folder_path = None
        if self.concatCheck:
            folder_path = self.output_pathByID

        self.CNMFU.comp_contour_plotter(folder_path=folder_path)

    def refine_patches_using_accepted_patches(self) -> None:
        """
        Refines the patches using the accepted patches.

        This method re-runs seeded CNMF on the accepted patches to refine and perform deconvolution.
        It fills in the accepted patches and then refines them.
        """
        # RE-RUN seeded CNMF on accepted patches to refine & perform deconvolution
        self.CNMFU.fill_in_accepted_patches()
        # refine patches
        self.CNMFU.refine_patches(stride=None, rf=None)

    def calc_CaTransients(self) -> None:
        """
        Calculates calcium transients.

        Loads the mmap file, calculates the fluorescence dF/F using CNMF algorithm.
        """
        if self.concatCheck:
            fname_mmap = self.fname_mmap_postproc
        else:
            fname_mmap = os.path.join(self.folder_path, self.fname_mmap_postproc)

        Yr, _, _ = self.utils.caiman_utils.load_mmap(fname_mmap=fname_mmap)
        self.CNMFU.calc_F_dff(Yr=Yr)

    def save_segDict(self) -> None:
        """
        Saves the segmentation dictionary using the CNMFU object's createNexport_segDict method.

        This method is responsible for saving the segmentation dictionary by calling the createNexport_segDict method
        of the CNMFU object.
        """
        folder_path_concat = None
        folder_path_subj = []

        if self.concatCheck:
            folder_path_concat = self.output_pathByID
            folder_path_subj = self.folder_path

        self.CNMFU.createNexport_segDict(
            concatCheck=self.concatCheck,
            folder_path_concat=folder_path_concat,
            folder_path_subj=folder_path_subj,
            prev_sd_varnames=self.prev_sd_varnames,
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
        if self.element_size_um is None:
            if self.concatCheck:
                h52use = self.latest_h5[0]
            else:
                h52use = self.latest_h5

            self.element_size_um = self.H5U.extract_element_size(h52use)
        print("Finding Residuals of Post-Segmentation")
        self.print_wFrm(
            "Finding difference between original & denoised movie (reconstructed from CNMF)"
        )
        # load original movie, normalized & converted to 16-bit unsigned
        if self.concatCheck:
            fname2load = self.fname_mmap_postproc
        else:
            fname2load = os.path.join(self.folder_path, self.fname_mmap_postproc)

        m_els = movie_utils.load_movie(fname2load)

        # extract denoised movies, with & without background
        denoised_noback = self._create_recontructed_movie()
        denoised_wback = self._create_recontructed_movie(wBackground=True)

        # normalize & convert to uint
        CdotA = movie_utils.normNconvert2uint(denoised_noback)
        # noncap_noise = full film - (C dot A + b dot f)
        noncap_noise = movie_utils.normNconvert2uint(m_els - denoised_wback)
        # concatenate C dot A & noncap_noise
        concatenated_residual_movie = movie_utils.concatenate_movies(
            [CdotA, noncap_noise], axis=2
        )
        # movie_utils.process_and_play_movie(concatenated_movie, downsample_ratio)

        # save residual movie
        for movies2save in [
            CdotA,
            noncap_noise,
            concatenated_residual_movie,
        ]:
            if movies2save is CdotA:
                moviefname = self.file_tag["CDA_MOV"]
            elif movies2save is noncap_noise:
                moviefname = self.file_tag["NOISE_MOV"]
            elif movies2save is concatenated_residual_movie:
                moviefname = self.file_tag["CON_RES_MOV"]
            for ftype in [self.file_tag["AVI"], self.file_tag["H5"]]:
                if self.concatCheck:
                    moviefname = os.path.join(self.output_pathByID, moviefname)
                else:
                    moviefname = os.path.join(self.folder_path, moviefname)
                self._save_residual_movie(movies2save, moviefname, ftype)
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

        cnm = self.CNMFU.NonNegMatrix_post_refining.estimates
        denoised = movie_utils.create_denoised_movie(
            cnm, self.dims, wBackground=wBackground
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
            residual_movie, fname, ftag, element_size_um=self.element_size_um
        )

        fname2print = self.folder_tools.os_splitterNcheck(fname, baseORpath="base")

        if "CdotA" in fname2print:
            mtype = "C dot A"
        elif "NonSeg_Noise" in fname2print:
            mtype = "Non-segmented Noise"
        elif "Concatenated" in fname2print:
            mtype = "Concatenated"

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
