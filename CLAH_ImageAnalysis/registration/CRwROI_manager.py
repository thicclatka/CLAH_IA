import copy
import tempfile
from enum import Enum

# from typing import List, Optional
import numpy as np
import roicat.visualization
from aenum import extend_enum
from roicat.data_importing import Data_roicat

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.registration import CRwROI_enum, CRwROI_plots, CRwROI_utils
from CLAH_ImageAnalysis.tifStackFunc.TSF_enum import segDict_Txt


class CRwROI_manager(BC, Data_roicat):
    """
    Manager class for Cell Registration with ROI Categorization using the ROICaT algorithm.
    This class provides methods to load, process, and analyze multi-session segmentation
    structures (MSSS), aligning and blurring ROIs, running neural networks, clustering, and
    exporting data.

    Parameters:
        program_name (str): The name of the program.
        path (list, optional): The path to the data directory. Defaults to an empty list.
        sess2process (list, optional): A list of sessions to process. Defaults to an empty list.
        useGPU (bool, optional): Flag indicating whether to use GPU. Defaults to True.
        verbose (bool, optional): Flag indicating whether to enable verbose mode. Defaults to True.

    Attributes:
        CRTOOLS (CRwROI_utils): Utility functions for CRwROI.
        dir_temp (str): Directory for temporary files.
        SDstr (dict): Dictionary of segmentation text strings.
        CRpar (dict): Dictionary of CRwROI parameters.
        CRkey (dict): Dictionary of CRwROI keys.
        CRprt (dict): Dictionary of CRwROI print strings.
        clusterer (object): Clusterer object for ROICaT.
        aligner (object): Aligner object for ROICaT.
        blurrer (object): Blurrer object for ROICaT.
        roinet (object): ROInet object for ROICaT.
        swt (object): Scattering Wavelet Transform object for ROICaT.
        sim (object): Similarity computation object for ROICaT.
        multSessSegStruc (dict): Multi-session segmentation structure.
        subj_sessions (list): List of subject sessions.
        refLapType (list): List of reference lap types.
        isTC (list): List of isTC arrays for each session.
        DEVICE (str): Device used for PyTorch (CPU or GPU).
        kwargs_mcdm_tmp (dict): Temporary keyword arguments for the MCDM.

    Methods:
        static_class_var_init(folder_path, sess2process, useGPU, verbose):
            Initialize static class variables.

        forLoop_var_init(sess_idx, sess_num):
            Initializes variables for the for loop iteration and CRwROI plots.

        load_multSessSegStruc():
            Loads the multi-session segmentation structure from a file.

        ROICaT_0_start():
            Starts the ROICaT process by preparing the data and aligning & blurring the ROIs.

        ROICaT_1_runNN():
            Runs the ROICaT-1 algorithm using a neural network.

        ROICaT_2_clustering():
            Performs clustering on the ROI data using the ROICaT algorithm.

        ROICaT_3_plotting():
            Plots the distribution of similarity measures, quality metrics histograms, and prints the results.

        ROICaT_4_exportData():
            Saves the results of the CRwROI analysis.

        _prepare4ROICaT(include_discarded=True, um_per_pixel=None, out_height_width4ROI=None, centroid_method="median", class_labels=None):
            Prepares the data for ROICaT analysis.

        _aligner_viaROICaT():
            Performs image alignment using the aligner class from ROICaT.

        _setup_ROInet():
            Sets up the ROInet for image analysis.

        _generate_dataloaderNlatents():
            Generates the dataloader and latents for the ROI network.

        _SWT_transform():
            Applies the scattering wavelet transform to the ROI images.

        _compute_similarity():
            Computes the similarity between spatial footprints and features using blockwise computation.

        _init_clustererNfindOptimalParams():
            Initializes the clusterer and finds optimal parameters for pruning.

        _plot_distNsimRel():
            Plots the distance and similarity relationships.

        _pruneDataNextract_labels():
            Prunes the data based on a distance cutoff and extracts labels via fitting.

        _plot_quality_metrics_histosNprint_results(sp_fp_exp=None, cluster_silhouette_thresh=None, cluster_intra_means_thresh=None, rejected_label=-1):
            Plots quality metrics histograms and prints the results.

        _create_ROI_PC_box(ROIS_aligned, isTC):
            Create contours to highlight ROIs that are PC/cue cells.

        _save_results():
            Saves the results of the CRwROI analysis.

        check4prev_kwargs():
            Checks for previous optimal parameters and prompts the user to use them if available.
    """

    def __init__(
        self,
        program_name: str,
        path: str | list = [],
        sess2process: list = [],
        sessFocus: int | None = None,
        useGPU: bool = True,
        verbose: bool = True,
    ) -> None:
        self.program_name = program_name
        self.class_type = "manager"

        BC.__init__(
            self,
            program_name=self.program_name,
            mode=self.class_type,
            sess2process=sess2process,
        )
        # initiate depencies as CRTOOLS w/in self
        # self.CRTOOLS = CRwROI_utils(self.program_name)
        # init global vars
        self.static_class_var_init(path, sess2process, sessFocus, useGPU, verbose)

    ######################################################
    #  setting up global vars
    ######################################################

    def static_class_var_init(
        self,
        folder_path: str | list,
        sess2process: list,
        sessFocus: int | None,
        useGPU: bool,
        verbose: bool,
    ) -> None:
        """
        Initialize static class variables.

        Parameters:
            folder_path (str): The folder path.
            sess2process (str): The session to process.
            useGPU (bool): Whether to use GPU.
            verbose (bool): Whether to enable verbose mode.
        """
        # By default, will provide dayDir, dayPath, sess2process
        BC.static_class_var_init(
            self,
            folder_path=folder_path,
            file_of_interest=self.text_lib["selector"]["tags"]["MSS"],
            selection_made=sess2process,
        )

        self.sessFocus = sessFocus
        self._verbose = verbose
        self.useGPU = useGPU
        # find temp file to download model for neural net
        self.dir_temp = tempfile.gettempdir()

        self.SDstr = self.enum2dict(segDict_Txt)
        self.CRpar = self.enum2dict(CRwROI_enum.Params)
        # self.CRkey = self.enum2dict(CRwROI_enum.Txt)
        # self.CRprt = self.enum2dict(CRwROI_enum.TxtPrint)

    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        """
        Initializes variables for the for loop iteration. This is where CRwROI_plots is initialized.

        Parameters:
            sess_idx (int): The index of the session.
            sess_num (int): The total number of sessions.

        Returns:
            None
        """
        # initiate forLoop_var_init from BaseClass
        BC.forLoop_var_init(self, sess_idx, sess_num)

        # additions to forLoop_var_init for CRwROI_Utils
        # # initiate plotting funcs for CRwROI
        # self.CRPLOTTERS = CRwROI_plots(self.ID, self.program_name)

        # set ROICaT classes to None
        self.clusterer = None
        self.aligner = None
        self.blurrer = None
        self.roinet = None
        self.swt = None
        self.sim = None

        # dicts from enums clear before each run
        self.CRkey = {}
        self.CRprt = {}

    ######################################################
    # func for generating constants for enum
    ######################################################

    def add_SessBasedConst2enum(self, enum_class: Enum) -> None:
        """
        Dynamically add generated constants to the given enum class.

        Parameters:
            enum_class (Enum): The enum class to add constants to.

        Returns:
            None
        """

        def add_enum_constant(enum_class, name, value):
            if not hasattr(enum_class, name):
                extend_enum(enum_class, name, value)

        base_names = ["CC", "TC"]
        suffixes = ["", "_QC"]

        for sess in range(1, self.numSess + 1):
            for base in base_names:
                for suffix in suffixes:
                    attrName2add = f"{base}_S{sess}{suffix}"
                    enumAttr4form = f"{base}_SX{suffix}"
                    formVal = getattr(enum_class, enumAttr4form).value.format(sess)
                    add_enum_constant(enum_class, attrName2add, formVal)

    ######################################################
    # func for handling MSSS to be used in ROICaT
    ######################################################

    def load_multSessSegStruc(self) -> None:
        """
        Loads the multi-session segmentation structure.

        This method loads the multi-session segmentation structure from a file and initializes
        the `multSessSegStruc` attribute of the class instance. It also sets the `subj_sessions`
        attribute to a list of session keys within the `multSessSegStruc` dictionary.

        Returns:
            None
        """
        print("Loading multSessSegStruc")
        # load the multi-session segmentation structure
        # specifically load the pkl file
        findLastest_MSS = self.findLatest([self.file_tag["MSS"], self.file_tag["PKL"]])
        self.print_wFrm(f"Found: {findLastest_MSS}")
        # load the file
        self.multSessSegStruc = self.load_file(findLastest_MSS)
        self.subj_sessions = list(self.multSessSegStruc.keys())
        numSess = len(self.subj_sessions)

        if self.sessFocus is not None:
            if self.sessFocus > numSess:
                self.numSess = numSess
            elif self.sessFocus <= numSess:
                self.numSess = self.sessFocus
                self.subj_sessions = self.subj_sessions[: self.numSess]
        else:
            self.numSess = numSess

        # generate constants for enum based on session num

        self.add_SessBasedConst2enum(CRwROI_enum.Txt)
        self.add_SessBasedConst2enum(CRwROI_enum.TxtPrint)

        # load into self as dicts
        self.CRkey = self.enum2dict(CRwROI_enum.Txt)
        self.CRprt = self.enum2dict(CRwROI_enum.TxtPrint)

        # initiate CRwROI_Utils & CRwROI_plots
        self.CRTOOLS = CRwROI_utils(
            self.program_name,
            numSessFound=numSess,
            numSess2use=self.numSess,
            CRkey=self.CRkey,
            CRprt=self.CRprt,
        )
        self.CRPLOTTERS = CRwROI_plots(
            self.ID,
            self.program_name,
            numSess=self.numSess,
            CRkey=self.CRkey,
            CRpar=self.CRpar,
        )

        # print loaded results
        self.CRTOOLS.print_loaded_mSSS_results(ID=self.ID)

        # find refLapType per session

        try:
            self.refLapType = []
            for sess in self.subj_sessions:
                self.refLapType.append(
                    self.multSessSegStruc[sess]["cueShiftStruc"]["lapCue"]["lap"][
                        "refLapType"
                    ]
                )

            # extract isTC array per session
            self.CueCellTable = []
            for sess in self.subj_sessions:
                CCT = self.multSessSegStruc[sess]["CueCellFinderDict"]["CueCellTable"]
            self.CueCellTable.append(CCT)
            self.isCell = self.CRTOOLS.organize_cellTypes_fromCCT(self.CueCellTable)

        except Exception as e:
            self.rprint(f"Problem with importing cell type info: {e}")
            self.rprint("Skipping cell type info, continuing...")
            self.CueCellTable = None
            self.refLapType = None
            self.isCell = None

    ######################################################
    #  ROICaT funcs
    ######################################################

    def ROICaT_0_start(self) -> None:
        """
        Starts the ROICaT process by preparing the data and aligning & blurring the ROIs.

        This method prepares the data for the ROICaT process by calling the _prepare4ROICaT method.
        It then aligns and blurs the ROIs to enable better clustering by calling the _aligner_viaROICaT method.

        Returns:
            None
        """
        # prepare data for ROICaT
        self._prepare4ROICaT()
        # align & blur ROIs to enable better clustering
        self.aligner, self.blurrer = self._aligner_viaROICaT()

    def ROICaT_1_runNN(self) -> None:
        """
        Runs the ROICaT-1 algorithm using a neural network.

        This method performs the following steps:
        1. Sets up the ROInet (neural network).
        2. Generates dataloader and latents from ROInet.
        3. Performs scattering wavelet transform.
        4. Computes similarity using ROIs, neural network, and scattering wavelet transform.
        """
        # setup ROInet (neural network)
        self._setup_ROInet()
        # generate dataloader & latents from ROInet
        self._generate_dataloaderNlatents()
        # perform scattering wavelet transform
        self._SWT_transform()
        # compute similarity using ROIs, NN, and SWT
        self._compute_similarity()

    def ROICaT_2_clustering(self) -> None:
        """
        Performs clustering on the ROI data using the ROICaT algorithm.

        This method initializes the clusterer and finds the optimal parameters for clustering.
        It then prunes the data and extracts the labels for each cluster.

        Returns:
            None
        """
        self._init_clustererNfindOptimalParams()
        self._pruneDataNextract_labels()

    def ROICaT_3_plotting(self) -> None:
        """
        This method is responsible for plotting the distribution of similarity measures,
        plotting quality metrics histograms, and printing the results.

        Parameters:
            self: The object instance.

        Returns:
            None
        """
        self._plot_distNsimRel()
        self._plot_quality_metrics_histosNprint_results()

    def ROICaT_4_exportData(self) -> None:
        self._save_results()

    def _prepare4ROICaT(
        self,
        include_discarded: bool = True,
        um_per_pixel: float = None,
        out_height_width4ROI: list = None,
        centroid_method: str = "median",
        class_labels: str | None = None,
    ) -> None:
        """
        Prepare the data for ROICaT analysis.

        Parameters:
            include_discarded (bool, optional): Whether to include discarded components. Defaults to True.
            um_per_pixel (float, optional): The micrometers per pixel. Defaults to self.CRpar["UM_PER_PXL"].
            out_height_width4ROI (List[int], optional): The output height and width for ROI. Defaults to self.CRpar["ROI_HEIGHT_WIDTH"].
            centroid_method (str, optional): The method for calculating centroid. Defaults to "median".
            class_labels (Optional[str], optional): The class labels. Defaults to None.
        """
        # set defaults for self vars
        um_per_pixel = self.is_var_none_check(um_per_pixel, self.CRpar["UM_PER_PXL"])
        out_height_width4ROI = self.is_var_none_check(
            out_height_width4ROI, self.CRpar["ROI_HEIGHT_WIDTH"]
        )
        out_height_width4ROI = (
            self.CRpar["ROI_HEIGHT_WIDTH"]
            if out_height_width4ROI is None
            else out_height_width4ROI
        )
        # init ROICaT
        Data_roicat.__init__(self)

        self.n_session = len(self.subj_sessions)
        # init vars
        spatialFootprints = []
        FOV_height, FOV_width = [], []

        # Find FOV Height & width
        FOV_height, FOV_width = self.CRTOOLS.extract_FOV_heightNwidth(
            self.multSessSegStruc, self.subj_sessions, self.SDstr
        )

        # output dimensions for spatial footprints & FOV (not ROI)
        self.FOV_height_width = (FOV_height[0], FOV_width[0])

        spatialFootprints = [
            self.CRTOOLS.reshape_spatial4ROICaT(
                array=self.multSessSegStruc[session][self.SDstr["A_SPATIAL"]],
                out_height_width=[FOV_height[session_index], FOV_width[session_index]],
                transpose=True,
                threeD=True,
                sparse_output=True,
            )
            for session_index, session in enumerate(self.subj_sessions)
        ]

        FOV_images = self.CRTOOLS.create_FOV_images(
            self.multSessSegStruc, self.subj_sessions, self.FOV_height_width
        )

        print(f"um_per_pixel is set to: {um_per_pixel}")
        ## Using ROICaT functions necessary for further analysis
        # based on what is written in Data_caiman class in roicat/data_importing.py
        print("Setting ROICaT vars & funcs:")
        self.set_spatialFootprints(
            spatialFootprints=spatialFootprints, um_per_pixel=um_per_pixel
        )
        self.set_FOV_images(FOV_images=FOV_images)
        self._make_spatialFootprintCentroids(method=centroid_method)
        self._make_session_bool()
        self.transform_spatialFootprints_to_ROIImages(
            out_height_width=out_height_width4ROI
        )
        self.set_class_labels(labels=class_labels) if class_labels is not None else None

        # plot pre-shift
        self.CRPLOTTERS._plot_footprints(
            footprints=self.get_maxIntensityProjection_spatialFootprints(),
            title="Pre_all",
        )

    def _aligner_viaROICaT(self) -> tuple:
        """
        Performs image alignment using the aligner class from ROIcat.

        Returns:
            aligner: An instance of the aligner class after performing image alignment.
            blurrer: A blurrer object that contains the blurred regions of interest (ROIs).

        """

        def _augment_FOV(aligner) -> np.ndarray:
            """
            Augments the field of view (FOV) images by blending ROIs with the FOV images.

            Parameters:
                aligner: An instance of the aligner class.

            Returns:
                augmented_FOV: The augmented field of view images.

            """
            augmented_FOV = aligner.augment_FOV_images(
                FOV_images=self.FOV_images,
                spatialFootprints=self.spatialFootprints,
                roi_FOV_mixing_factor=self.CRpar["ROI_FOV_MIX_FACTOR"],
                use_CLAHE=self.CRpar["USE_CLAHE"],
                CLAHE_grid_size=int(self.CRpar["CLAHE_GRID_SIZE"]),
                CLAHE_clipLimit=int(self.CRpar["CLAHE_CLIPLIMIT"]),
                CLAHE_normalize=self.CRpar["CLAHE_NORM"],
            )
            return augmented_FOV

        def _fit_geometric(
            aligner: roicat.tracking.alignment.Aligner,
        ) -> roicat.tracking.alignment.Aligner:
            """
            Fits the geometric transformation to align the images.

            Parameters:
                aligner: An instance of the aligner class.

            Returns:
                The aligner object after fitting the geometric transformation and transforming the images.
            """
            if "CA3" in self.dayPath and "KET" not in self.dayPath:
                mask2use = self.CRpar["MASK_BORDER_CA3"]
            elif "DG" in self.dayPath:
                mask2use = self.CRpar["MASK_BORDER_DG"]
            elif "KET" in self.dayPath:
                mask2use = self.CRpar["MASK_BORDER_KET"]

            aligner.fit_geometric(
                template=self.aligned_FOV[0],
                ims_moving=self.aligned_FOV,
                template_method=self.CRpar["TEMP_METHOD"],
                mode_transform=self.CRpar["MODE_TRANS_GEO"],
                mask_borders=mask2use,
                n_iter=self.CRpar["NUM_ITER"],
                termination_eps=self.CRpar["TERMINATION_EPS"],
                gaussFiltSize=self.CRpar["GAUSSFILT_SIZE"],
                auto_fix_gaussFilt_step=self.CRpar["AUTOFIX_GAUSS"],
            )
            aligner.transform_images_geometric(self.aligned_FOV)
            return aligner

        def _fit_nonrigid(
            aligner: roicat.tracking.alignment.Aligner,
        ) -> roicat.tracking.alignment.Aligner:
            """
            Fits a non-rigid transformation to align the moving images with the template image.

            Parameters:
            - aligner: An instance of the aligner class.

            Returns:
            - aligner: The aligner instance with the non-rigid transformation applied.
            """
            aligner.fit_nonrigid(
                template=self.aligned_FOV[0],
                ims_moving=aligner.ims_registered_geo,
                remappingIdx_init=aligner.remappingIdx_geo,
                template_method=self.CRpar["TEMP_METHOD"],
                mode_transform=self.CRpar["MODE_TRANS_NONRIGID"],
                kwargs_mode_transform=self.CRpar["KWARGS_MODE_TRANS"],
            )
            aligner.transform_images_nonrigid(self.aligned_FOV)
            return aligner

        def _transform_ROIs(
            aligner: roicat.tracking.alignment.Aligner,
        ) -> roicat.tracking.alignment.Aligner:
            """
            Transforms the spatial footprints of the ROIs using the given aligner.

            Parameters:
                aligner: An instance of the aligner class used for transformation.

            Returns:
                The updated aligner object after transforming the ROIs.
            """
            aligner.transform_ROIs(
                ROIs=self.spatialFootprints,
                remappingIdx=aligner.remappingIdx_nonrigid,
                normalize=self.CRpar["ROI_NORM"],
            )
            # plot aligned ROIs as a footprint
            self.CRPLOTTERS._plot_footprints(
                footprints=self.get_maxIntensityProjection_spatialFootprints(
                    sf=aligner.ROIs_aligned
                ),
                title="Post_all",
            )
            return aligner

        def _blur_ROIs(
            aligner: roicat.tracking.alignment.Aligner,
        ) -> roicat.tracking.blurring.ROI_Blurrer:
            """
            Blurs the regions of interest (ROIs) using a specified blurring kernel.

            Parameters:
                aligner: An instance of the aligner class.

            Returns:
                A blurrer object that contains the blurred ROIs.

            """
            blurrer = roicat.tracking.blurring.ROI_Blurrer(
                frame_shape=self.FOV_height_width,
                kernel_halfWidth=self.CRpar["KERNEL_HALFWIDTH"],
                plot_kernel=self.CRpar["PLOT_KERNEL"],
            )
            blurrer.blur_ROIs(spatialFootprints=aligner.ROIs_aligned[:])
            # plot blurred ROIs as a footprint
            self.CRPLOTTERS._plot_footprints(
                footprints=blurrer.get_ROIsBlurred_maxIntensityProjection(),
                title="Blur_post_all",
            )
            return blurrer

        # init aligner
        aligner = roicat.tracking.alignment.Aligner(verbose=self._verbose)

        # augment FOV w/ROIs
        self.aligned_FOV = _augment_FOV(aligner)

        # geometric fit
        aligner = _fit_geometric(aligner)
        # non rigid fit
        aligner = _fit_nonrigid(aligner)
        # apply to ROIs
        aligner = _transform_ROIs(aligner)
        # blurring ROIs
        blurrer = _blur_ROIs(aligner)

        return aligner, blurrer

    def _setup_ROInet(self) -> None:
        """
        Sets up the ROInet for image analysis.

        This method sets the device for PyTorch and initializes the ROInet_embedder
        object for performing image analysis using the ROInet network.

        Parameters:
            None

        Returns:
            None
        """

        @self.StatusPrinter.output_btw_dots(
            pre_msg="Setting device for PyTorch.", pre_msg_append=True
        )
        def _set_device() -> str:
            """
            Sets the device for image analysis.

            This method sets the device for image analysis based on the specified parameters.

            Parameters:
            - use_GPU (bool): Flag indicating whether to use GPU for image analysis.
            - verbose (bool): Flag indicating whether to display verbose output.

            Returns:
            - DEVICE: The selected device for image analysis.
            """
            self.DEVICE = roicat.helpers.set_device(
                use_GPU=self.useGPU, verbose=self._verbose
            )

        # setting device
        _set_device()
        self.roinet = roicat.ROInet.ROInet_embedder(
            device=self.DEVICE,
            dir_networkFiles=self.dir_temp,
            download_method=self.CRpar["DL_METH"],
            download_url=self.CRpar["DL_URL"],
            download_hash=self.CRpar["DL_HASH"],
            forward_pass_version=self.CRpar["FP_VERS"],
            verbose=self._verbose,
        )

    def _generate_dataloaderNlatents(self) -> None:
        """
        Generates the dataloader and latents for the ROI network.

        Parameters:
            None

        Returns:
            None
        """
        # dataloader
        self.roinet.generate_dataloader(
            ROI_images=self.ROI_images,
            um_per_pixel=self.um_per_pixel,
            pref_plot=self.CRpar["PREF_PLOT"],
            jit_script_transforms=self.CRpar["JIT_SCRIPT_TRANS"],
            batchSize_dataloader=self.CRpar["BATCHSIZE"],
            pinMemory_dataloader=self.CRpar["PINMEMORY"],
            numWorkers_dataloader=self.CRpar["NUMWORKERS"],
            persistentWorkers_dataloader=self.CRpar["PERSISTENTWORKERS"],
            prefetchFactor_dataloader=self.CRpar["PREFETCHFACTOR"],
        )
        # latents
        self.roinet.generate_latents()

    def _SWT_transform(self) -> None:
        """
        Applies the scattering wavelet transform to the ROI images.

        This method initializes the scattering wavelet transformer class and applies the transform
        to the ROI images using the specified batch size.

        Parameters:
            None

        Returns:
            None
        """
        # init scattering wavelet transformer class
        self.swt = roicat.tracking.scatteringWaveletTransformer.SWT(
            kwargs_Scattering2D=self.CRpar["KWARGS_SCAT2D"],
            image_shape=self.ROI_images[0].shape[1:3],
            device=self.DEVICE,
        )
        # transform
        self.swt.transform(
            ROI_images=self.roinet.ROI_images_rs,
            batch_size=self.CRpar["SWT_BATCH_SIZE"],
        )

    def _compute_similarity(self) -> tuple:
        """
        Computes the similarity between spatial footprints and features using blockwise computation.

        Returns:
            Tuple: A tuple containing the computed similarities for spatial footprints (s_sf),
            nearest neighbors (s_NN), SWT (s_SWT), and session (s_sesh).
        """
        # init sim class
        self.sim = roicat.tracking.similarity_graph.ROI_graph(
            n_workers=self.CRpar["N_WORKERS"],
            frame_height=self.FOV_height_width[0],
            frame_width=self.FOV_height_width[1],
            block_height=self.CRpar["BLOCK_HEIGHT"],
            block_width=self.CRpar["BLOCK_WIDTH"],
            algorithm_nearestNeigbors_spatialFootprints=self.CRpar["ALGO_NN_SF"],
            verbose=self._verbose,
        )
        # compute similarity
        self.s_sf, self.s_NN, self.s_SWT, self.s_sesh = (
            self.sim.compute_similarity_blockwise(
                spatialFootprints=self.blurrer.ROIs_blurred,
                features_NN=self.roinet.latents,
                features_SWT=self.swt.latents,
                ROI_session_bool=self.session_bool,
                spatialFootprint_maskPower=self.CRpar["SF_MASKPOWER"],
            )
        )
        # normalize
        self.sim.make_normalized_similarities(
            centers_of_mass=self.centroids,
            features_NN=self.roinet.latents,  ## ROInet latents
            features_SWT=self.swt.latents,  ## SWT latents
            k_max=self.n_session * self.CRpar["KMAX"],
            k_min=self.n_session * self.CRpar["KMIN"],
            algo_NN=self.CRpar["ALGO_NN_NORM"],
            device=self.DEVICE,
        )

    def _init_clustererNfindOptimalParams(self) -> None:
        """
        Initializes the clusterer and finds optimal parameters for pruning.

        This method initializes the clusterer object with the specified parameters and then finds the optimal parameters
        for pruning by calling the `find_optimal_parameters_for_pruning` method of the clusterer. If the `kwargs_mcdm_tmp`
        attribute is None, it saves the optimal parameters to `kwargs_mcdm_tmp` attribute and also saves them to a file.
        If `kwargs_mcdm_tmp` is not None, it fills in the variables into the clusterer object to enable later functions.

        Returns:
            None
        """
        # initiate clusterer
        self.clusterer = roicat.tracking.clustering.Clusterer(
            s_sf=self.sim.s_sf,
            s_NN_z=self.sim.s_NN_z,
            s_SWT_z=self.sim.s_SWT_z,
            s_sesh=self.sim.s_sesh,
        )

        if self.kwargs_mcdm_tmp is None:
            # find optimal parameters for pruning
            kwargs_makeConjunctiveDistanceMatrix_best = (
                self.clusterer.find_optimal_parameters_for_pruning(
                    n_bins=None,
                    smoothing_window_bins=None,
                    kwargs_findParameters=self.CRpar["KW_FP"],
                    bounds_findParameters=self.CRpar["BOUND_FP"],
                    n_jobs_findParameters=self.CRpar["N_JOBS_FP"],
                )
            )
            self.kwargs_mcdm_tmp = copy.deepcopy(
                kwargs_makeConjunctiveDistanceMatrix_best
            )

            # save optimization params for easy access
            self.savedict2file(
                dict_to_save=kwargs_makeConjunctiveDistanceMatrix_best,
                dict_name=self.CRkey["OP_KW"],
                filename=self.CRkey["OP_KW_SAVE"],
                date=True,
                filetype_to_save=self.file_tag["PKL"],
            )
        else:
            # fill in vars into clusterer to enable later funcs given import
            self.clusterer.n_bins, self.clusterer.smooth_window = (
                self.CRTOOLS.find_nbinsNsmooth_window(s_sf_nnz=self.clusterer.s_sf.nnz)
            )
            self.clusterer.kwargs_makeConjunctiveDistanceMatrix = self.kwargs_mcdm_tmp
            print("Using previously found optimal parameters")
            print("Best value found:")
            for key, val in self.kwargs_mcdm_tmp.items():
                self.print_wFrm(f"{key}: {val}")

    def _plot_distNsimRel(self) -> None:
        """
        Plot the distance and similarity relationships.

        This method plots the clustering results by calling the `_plot_distSame_crwr` and `_plot_sim_relationships_crwr`
        methods from the `CRPLOTTERS` object. It takes the following arguments:

        - `clusterer`: The clusterer object used for clustering.
        - `kwargs_makeConjunctiveDistanceMatrix`: Additional keyword arguments for the `kwargs_makeConjunctiveDistanceMatrix` method

        The method also uses the following parameters from the `CRpar` dictionary:

        - `MAX_SAMPLES`: The maximum number of samples.
        - `SCAT_SZ`: The size of the scatter plot points.
        - `SCAT_ALPHA`: The transparency of the scatter plot points.
        """
        # plotting clustering results
        self.CRPLOTTERS._plot_distSame_crwr(
            clusterer=self.clusterer,
            kwargs_makeConjunctiveDistanceMatrix=self.kwargs_mcdm_tmp,
        )
        self.CRPLOTTERS._plot_sim_relationships_crwr(
            clusterer=self.clusterer,
            max_samples=self.CRpar["MAX_SAMPLES"],
            kwargs_scatter={
                "s": self.CRpar["SCAT_SZ"],
                "alpha": self.CRpar["SCAT_ALPHA"],
            },
            kwargs_makeConjunctiveDistanceMatrix=self.kwargs_mcdm_tmp,
        )

    def _pruneDataNextract_labels(self) -> None:
        """
        Prunes the data based on a distance cutoff and extracts labels via fitting.

        This method prunes the data based on a distance cutoff value and then extracts labels
        by fitting the pruned data. It also computes quality metrics and creates results and
        run_data based on the labels, spatial footprints, and other variables.

        Parameters:
            None

        Returns:
            None
        """
        # prune data based on d_cutoff
        self.clusterer.make_pruned_similarity_graphs(
            d_cutoff=self.CRpar["D_CUTOFF"],
            kwargs_makeConjunctiveDistanceMatrix=self.kwargs_mcdm_tmp,
            stringency=self.CRpar["STRINGENCY"],
            convert_to_probability=self.CRpar["CONV2PROB"],
        )
        # extract labels via fitting
        self.labels = self.clusterer.fit(
            d_conj=self.clusterer.dConj_pruned,
            session_bool=self.session_bool,
            min_cluster_size=self.CRpar["MIN_CLUSTER_SIZE"],
            n_iter_violationCorrection=self.CRpar["N_ITER_VC"],
            split_intraSession_clusters=self.CRpar["SPLIT_IS_CLUSTER"],
            cluster_selection_method=self.CRpar["CLUSTER_SEL_METH"],
            d_clusterMerge=self.CRpar["D_CLUSTMRG"],
            alpha=self.CRpar["FIT_ALPHA"],
            discard_failed_pruning=self.CRpar["DISCARD_FAIL"],
            n_steps_clusterSplit=self.CRpar["NSTEPS_CLSTER_SPLT"],
        )
        # self.labels = self.clusterer.fit_sequentialHungarian(
        #     d_conj=self.clusterer.dConj_pruned,
        #     session_bool=self.session_bool,
        #     thresh_cost=self.CRpar["THRESH_COST"],
        # )

        # extract quality metrics
        self.quality_metrics = self.clusterer.compute_quality_metrics()

        # create results & run_data
        # based on labels, spatial footprints, and other vars
        # see create_resultsNrun_data in CRwROI_dependencies
        # for more info
        self.results, self.run_data = self.CRTOOLS.create_resultsNrun_data(
            roicat=roicat,
            labels2use=self.labels,
            n_roi=self.n_roi,
            aligner=self.aligner,
            spatialFootprints=self.spatialFootprints,
            FOV_height_width=self.FOV_height_width,
            session_bool=self.session_bool,
            n_session=self.n_session,
            folder_path=self.folder_path,
            kwargs_mcdm_tmp=self.kwargs_mcdm_tmp,
            self_sdict=self.serializable_dict,
            blurrer_sdict=self.blurrer.serializable_dict,
            roinet_sdict=self.roinet.serializable_dict,
            swt_sdict=self.swt.serializable_dict,
            sim_sdict=self.sim.serializable_dict,
            clusterer=self.clusterer,
        )

    def _plot_quality_metrics_histosNprint_results(
        self,
        sp_fp_exp: float | None = None,
        cluster_silhouette_thresh: int | None = None,
        cluster_intra_means_thresh: int | None = None,
        rejected_label: int = -1,
    ) -> None:
        """
        Plot quality metrics histograms and print the results.

        Parameters:
            sp_fp_exp (float, optional): Exponent for spatial footprint power calculation. Defaults to None.
            cluster_silhouette_thresh (int, optional): Threshold for cluster silhouette. Defaults to None.
            cluster_intra_means_thresh (int, optional): Threshold for cluster intra-means. Defaults to None.
            rejected_label (int, optional): Label for rejected clusters. Defaults to -1.

        Returns:
            None
        """
        # set up default vars with self
        sp_fp_exp = self.is_var_none_check(sp_fp_exp, self.CRpar["SP_FP_EXP"])
        cluster_silhouette_thresh = self.is_var_none_check(
            cluster_silhouette_thresh, int(self.CRpar["CL_SILHOUETTE"])
        )
        cluster_intra_means_thresh = self.is_var_none_check(
            cluster_intra_means_thresh, int(self.CRpar["CL_INTRA_MEANS"])
        )
        # init count dict to fill
        count_dict = {
            self.CRkey["D2D"]: 0,
            self.CRkey["D2D_QC"]: 0,
            self.CRkey["D2D_CC"]: 0,
            self.CRkey["D2D_CC_QC"]: 0,
            self.CRkey["D2D_TC"]: 0,
            self.CRkey["D2D_TC_QC"]: 0,
            self.CRkey["W2W"]: 0,
            self.CRkey["W2W_QC"]: 0,
            self.CRkey["W2W_CC"]: 0,
            self.CRkey["W2W_CC_QC"]: 0,
            self.CRkey["W2W_TC"]: 0,
            self.CRkey["W2W_TC_QC"]: 0,
            self.CRkey["W2W2"]: 0,
            self.CRkey["W2W2_QC"]: 0,
            self.CRkey["W2W2_CC"]: 0,
            self.CRkey["W2W2_CC_QC"]: 0,
            self.CRkey["W2W2_TC"]: 0,
            self.CRkey["W2W2_TC_QC"]: 0,
            self.CRkey["ALLSESS"]: 0,
            self.CRkey["ALLSESS_QC"]: 0,
        }

        # init ROIsbyLabel vars
        allROISbyLabel = self.results[self.CRkey["CLUSTERS"]][self.CRkey["LABELS"]]
        # init ROIbsyLabelbySession vars
        allROISbyLabelbySession = self.results[self.CRkey["CLUSTERS"]][
            "labels_bySession"
        ]
        # get cluster_id & counts across all sesions
        cluster_id, cid_counts = np.unique(allROISbyLabel, return_counts=True)
        # get cluster_id by session
        cluster_id_bySess = []
        for sess in allROISbyLabelbySession:
            cluster_id_bySess.append(np.unique(sess))

        # fill in count for cells tracked in all sessions
        # wherever cid_counts == n_sesssion is a cell tracked across all sessions
        count_dict[self.CRkey["ALLSESS"]] = len(
            cluster_id[cid_counts == self.n_session]
        )

        # create labels based on basing certain thresholds
        # refer to this as quality control (QC)
        alpha_labels = self.CRTOOLS.extract_alpha_labels(
            cl_silhouette=self.clusterer.quality_metrics["cluster_silhouette"],
            cl_intra_means=self.clusterer.quality_metrics["cluster_intra_means"],
            cl_silhouette_thresh=cluster_silhouette_thresh,
            cl_intra_means_thresh=cluster_intra_means_thresh,
        )

        # find day 2 day or week 2 week tracked cells
        # apply QC as well
        # TODO: FIX THIS GIVEN ARRAY MISMATCH THAT OCCURS
        # count_dict = self.CRTOOLS.d2dNw2w_incQC_counter(
        #     cid_bySess=cluster_id_bySess,
        #     count_dict=count_dict,
        #     alpha_labels=alpha_labels,
        #     rejected_label=rejected_label,
        #     isCell=self.isCell,
        #     labelBySess=self.results[self.CRkey["CLUSTERS"]]["labels_bySession"],
        # )

        # create isCell boolean arrays for both post and pre QC
        # returns None if no cell types are found
        isCell_post_cluster = self.CRTOOLS.isTC_incQC_counter(
            results=self.results,
            isCell=self.isCell,
            alpha_labels=alpha_labels,
            rejected_label=rejected_label,
        )

        # Creates cluster_info dict
        # prints results accordingly into terminal
        self.cluster_info = self.CRTOOLS.create_cluster_info_dict(
            cluster_id=cluster_id,
            allROIS=allROISbyLabel,
            alpha_labels=alpha_labels,
            count_dict=count_dict,
            isCell_pre_cluster=self.isCell,
            isCell_post_cluster=isCell_post_cluster,
            rejected_label=rejected_label,
        )

        # init ROIS_aligned for legibility
        aligned_ROIs = self.results[self.CRkey["ROIS"]][self.CRkey["ROI_ALIGN"]]

        # create colored FOV clusters
        footprints4coloredclusters = [r.power(sp_fp_exp) for r in aligned_ROIs]

        if isCell_post_cluster is not None:
            self.PC_dict = {}
            for cellType in isCell_post_cluster.keys():
                if cellType == "NON":
                    continue
                self.PC_dict[cellType] = self._create_ROI_PC_box(
                    footprints4coloredclusters, isCell_post_cluster[cellType]["POST_QC"]
                )
        else:
            self.PC_dict = None

        # get FOVs post QC
        self.FOV_clusters = roicat.visualization.compute_colored_FOV(
            spatialFootprints=footprints4coloredclusters,
            FOV_height=self.results[self.CRkey["ROIS"]]["frame_height"],
            FOV_width=self.results[self.CRkey["ROIS"]]["frame_width"],
            labels=self.results[self.CRkey["CLUSTERS"]]["labels_bySession"],
            alphas_labels=alpha_labels,
            alphas_sf=self.clusterer.quality_metrics["sample_silhouette"],
        )

        # plot histograms
        self.CRPLOTTERS._plot_confidence_histograms(results=self.results)
        # plot colored FOV clusters
        self.CRPLOTTERS._plot_FOV_clusters(
            subj_id=self.ID,
            FOV_clusters=self.FOV_clusters,
            cluster_info=self.cluster_info,
            isCell2plot=isCell_post_cluster,
            PC_dict=self.PC_dict,
            contour=True,
        )

        # fill in results dict with FOV Clusters
        self.results[self.CRkey["CLUSTERS"]]["FOV_CLUSTERS"] = self.FOV_clusters

        # redo for pre QC, if pre QC discarded < post QC
        if (
            self.cluster_info[self.CRkey["DISCARD"]]
            < self.cluster_info[self.CRkey["DISCARD_QC"]]
        ):
            if isCell_post_cluster is not None:
                self.PC_dict_preQC = {}
                for cellType in isCell_post_cluster.keys():
                    if cellType == "NON":
                        continue
                    self.PC_dict_preQC[cellType] = self._create_ROI_PC_box(
                        footprints4coloredclusters,
                        isCell_post_cluster[cellType]["PRE_QC"],
                    )
            else:
                self.PC_dict_preQC = None

            self.FOV_clusters_preQC = roicat.visualization.compute_colored_FOV(
                spatialFootprints=footprints4coloredclusters,
                FOV_height=self.results[self.CRkey["ROIS"]]["frame_height"],
                FOV_width=self.results[self.CRkey["ROIS"]]["frame_width"],
                labels=self.results[self.CRkey["CLUSTERS"]]["labels_bySession"],
            )

            self.results[self.CRkey["CLUSTERS"]]["FOV_CLUSTERS_PREQC"] = (
                self.FOV_clusters_preQC
            )

            self.CRPLOTTERS._plot_FOV_clusters(
                subj_id=self.ID,
                FOV_clusters=self.FOV_clusters_preQC,
                cluster_info=self.cluster_info,
                isCell2plot=isCell_post_cluster,
                PC_dict=self.PC_dict_preQC,
                contour=True,
                preQC=True,
            )

    def _create_ROI_PC_box(self, ROIS_aligned: list, isTC: list) -> dict:
        """
        Create contours to highlight ROIs that are PC/cue cells.

        Parameters:
            ROIS_aligned (list): List of aligned ROIs.
            isTC (list): List indicating whether each ROI is a PC.

        Returns:
            None

        """
        ROIS_aligned_ofPC = [roia[PCS, :] for roia, PCS in zip(ROIS_aligned, isTC)]
        PC_dict = {
            self.CRkey["CENT"]: [],
            self.CRkey["BOUND"]: [],
            self.CRkey["MIN_D"]: [],
            self.CRkey["CONTOUR"]: [],
        }
        for sparse_ROIS in ROIS_aligned_ofPC:
            centroids4sess = []
            bounds4sess = []
            mindist4sess = []
            contours4sess = []
            for roi_sparse in sparse_ROIS:
                # Calculate centroid and bounds without converting to dense
                centroid, bounds, min_dist = (
                    self.CRTOOLS.find_centroidNbounds_fromROIsparse(
                        roi_sparse, self.FOV_height_width
                    )
                )
                contours = self.CRTOOLS.find_contours_of_nonzero(
                    roi_sparse, self.FOV_height_width
                )
                # append to appropriate list
                centroids4sess.append(centroid)
                bounds4sess.append(bounds)
                mindist4sess.append(min_dist)
                contours4sess.append(contours)
            PC_dict[self.CRkey["CENT"]].append(centroids4sess)
            PC_dict[self.CRkey["BOUND"]].append(bounds4sess)
            PC_dict[self.CRkey["MIN_D"]].append(mindist4sess)
            PC_dict[self.CRkey["CONTOUR"]].append(contours4sess)
        return PC_dict

    def _save_results(self) -> None:
        """
        Save the results of the CRwROI analysis.

        This method exports the parameters as JSON, saves the cluster count information as JSON,
        and saves the results and run_data dictionaries as pkl and mat files.
        """
        # export params as JSON
        self.enum_utils.export_param_from_enum(
            CRwROI_enum.Params,
            "CRwROIparams",
            self.file_tag["JSON"],
        )

        # save cluster count info as JSON
        self.savedict2file(
            dict_to_save=self.cluster_info,
            dict_name=self.CRkey["C_INFO"],
            filename=self.CRkey["C_INFO_SAVE"],
            date=False,
            filetype_to_save=[self.file_tag["JSON"]],
        )

        # save results as pkl & mat
        self.savedict2file(
            dict_to_save=self.results,
            dict_name=self.CRkey["RESULTS"],
            filename=self.CRkey["RESULTS_SAVE"],
            date=False,
            filetype_to_save=[self.file_tag["MAT"], self.file_tag["PKL"]],
        )

        # save run_data as pkl
        self.savedict2file(
            dict_to_save=self.run_data,
            dict_name=self.CRkey["RUN_DATA"],
            filename=self.CRkey["RUN_DATA_SAVE"],
            date=False,
            filetype_to_save=[self.file_tag["PKL"]],
        )

        self.savedict2file(
            dict_to_save=self.results[self.CRkey["CLUSTERS"]],
            dict_name=self.CRkey["CLUSTERS"],
            filename=self.CRkey["CLUSTERS"],
            date=False,
            filetype_to_save=[self.file_tag["MAT"]],
        )

    def check4prev_kwargs(self) -> None:
        """
        Check for previous optimal parameters and prompt the user to use them if available.

        This method checks if previous optimal parameters exist for Conjunctive Distance Matrix (CDM) analysis.
        If previous parameters are found, it prompts the user to confirm whether to use them.
        If confirmed, it loads the previous parameters and converts any NaN values to None.
        If not confirmed or no previous parameters are found, it sets the parameters to None.

        Returns:
            None
        """

        def _convertNaNs2None(kwargs: dict) -> dict:
            """
            Convert NaN values in a dictionary to None.

            Parameters:
                kwargs (dict): A dictionary containing key-value pairs.

            Returns:
                dict: A dictionary with NaN values replaced by None.

            """
            for key, val in kwargs.items():
                if isinstance(val, float) and np.isnan(val):
                    kwargs[key] = None
            return kwargs

        # init self.kwargs_mcdm_tmp as None
        self.kwargs_mcdm_tmp = None
        # init str var for new analysis
        new_analysis = "Proceeding with new analysis."

        # check to see if previous optimal parameters exist
        prev_kwargs = self.findLatest([self.CRkey["OP_KW_SAVE"], self.file_tag["PKL"]])
        if prev_kwargs:
            self.print_wFrm(
                f"Found previous optimal parameters for Conjunctive Distance Matrix: {prev_kwargs}"
            )
            ui_res = self.StatusPrinter.get_user_confirmation(
                prompt="Do you wish to use these parameters?"
            )
            if ui_res:
                self.print_wFrm(
                    "Loading previous optimal parameters", end="", flush=True
                )
                # set kwargs to previous optimal parameters
                self.kwargs_mcdm_tmp = self.load_file(prev_kwargs)
                self.kwargs_mcdm_tmp = _convertNaNs2None(self.kwargs_mcdm_tmp)
                self.print_done_small_proc(new_line=False)
            else:
                self.print_wFrm(new_analysis)
                # kwargs stays as None
        else:
            # no previous kwargs found, print accordingly
            # kwargs stays as None
            self.print_wFrm("No previous optimal parameters found.")
            self.print_wFrm(new_analysis)
