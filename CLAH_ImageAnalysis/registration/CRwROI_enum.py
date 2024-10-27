from enum import Enum
from CLAH_ImageAnalysis.utils import text_dict

# import multiprocessing as mp

######################################################
# Global vars - options for ROICat funcs
######################################################
# which template is used for alignment
# - image = use static image from which to align
# - sequential = look at changes in image over time
TEMPLATE_METHOD = ["image", "sequential"]
# Geometric transformation methods
# - translation = moves image in x and y
# - euclidean = translation + rotation
# - affine = translation + rotation + scaling
# - homography = translation + rotation + scaling + skewing
MDT_GEO = ["translation", "euclidean", "affine", "homography"]
# Non-rigid transformation methods
# - createOptFlow_DeepFlow = creates an instance of the DeepFlow optical flow algorithm, which is effective for large displacements in fluid-like motion
# - calcOptFlowFarneback = computes a dense optical flow using the Gunnar Farneback’s algorithm, which is robust to noise and provides spatial and temporal image derivatives
MDT_NR = ["createOptFlow_DeepFlow", "calcOptFlowFarneback"]
# Nearest neighbors algorithms
# - brute = uses a brute-force search, not efficient for large datasets
# - kd_tree = uses a k-d tree data structure, efficient for small to medium-sized datasets
# - balltree = uses a ball tree data structure, efficient for high-dimensional data
# - auto = automatically chooses the best algorithm based on the input data
ALGO_NN = ["brute", "kd_tree", "balltree", "auto"]
# Centroid methods aka "center" point of ROI
# - median
# - centerOfMass = uses mean to find centroid
CENTROID_METH = ["median", "centerOfMass"]


def _get_mask_borders(mask_border: int | tuple) -> tuple:
    """
    Convert the mask_border parameter to a tuple of 4 integers.

    Args:
        mask_border (int or tuple): The mask border value(s) to be converted.

    Returns:
        tuple: A tuple of 4 integers representing the mask border.

    """
    # check to see if mask_border is a single int or a tuple of 4 ints
    # if it is a single int, convert it to a tuple
    # otherwise, return it as is
    if isinstance(mask_border, int):
        mask_border = (mask_border,) * 4
    return mask_border


def _get_kwargs_scat2d(J: int, L: int) -> dict:
    """
    Returns a dictionary containing the keyword arguments for the scat2d function.

    Parameters:
    J (int): the number of convolutional layers
    L (int): the number of wavelet angles

    Returns:
    dict: A dictionary containing the keyword arguments for the swt function.
    """
    return {"J": J, "L": L}


def _get_kwargsNbounds_find_params() -> tuple:
    """
    Set the kwargs, bounds, and n_jobs for the find_optimal_parameters_for_pruning function within the clusterer initiated class (from roicat.tracking.cluser.Clusterer).

    Returns:
        tuple: A tuple containing the kwargs, bounds, and n_jobs (first 2 are dicts, latter is int) for the find_optimal_parameters_for_pruning function.

    """
    kwargs_findParameters = {
        "n_patience": 300,  ## Number of optimization epoch to wait for tol_frac to converge
        "tol_frac": 0.001,  ## Fractional change below which optimization will conclude
        "max_trials": 1200,  ## Max number of optimization epochs
        "max_duration": 60
        * 10,  ## Max amount of time (in seconds) to allow optimization to proceed for
    }
    bounds_findParameters = {
        "power_NN": (4.49, 5.0),  ## Bounds for the exponent applied to s_NN
        "power_SWT": (0.0, 0.5),  ## Bounds for the exponent applied to s_SWT
        "p_norm": (
            -5,
            0,
        ),  ## Bounds for the p-norm p value (Minkowski) applied to mix the matrices
        "sig_NN_kwargs_mu": (
            0.0,
            1.0,
        ),  ## Bounds for the sigmoid center for s_NN
        "sig_NN_kwargs_b": (
            0.00,
            1.5,
        ),  ## Bounds for the sigmoid slope for s_NN
        "sig_SWT_kwargs_mu": (
            0.0,
            1.0,
        ),  ## Bounds for the sigmoid center for s_SWT
        "sig_SWT_kwargs_b": (
            0.00,
            1.5,
        ),  ## Bounds for the sigmoid slope for s_SWT
    }
    n_jobs_findParameters = -1  # number of core to use (-1 is all cores)
    return kwargs_findParameters, bounds_findParameters, n_jobs_findParameters


class Params(Enum):
    """
    Enumeration class that defines various parameters used in the code.

    Attributes:
        CENTROID_METHOD (int): Method for calculating the centroid of the ROI.
        ROI_HEIGHT_WIDTH (List[int]): Height and width of the ROI.
        UM_PER_PXL (float): Conversion factor from pixels to micrometers.
        ROI_FOV_MIX_FACTOR (float): Factor for mixing the FOV augmentation.
        USE_CLAHE (bool): Flag indicating whether to use CLAHE (Contrast Limited Adaptive Histogram Equalization).
        CLAHE_GRID_SIZE (int): Grid size for CLAHE.
        CLAHE_CLIPLIMIT (int): Clip limit for CLAHE.
        CLAHE_NORM (bool): Flag indicating whether to normalize the CLAHE output.
        TEMP_METHOD (int): Method for handling the template used for registration.
        MODE_TRANS_GEO (int): Method for geometric transformation/shifts.
        MODE_TRANS_NONRIGID (int): Method for non-rigid transformation/shifts.
        KWARGS_MODE_TRANS (None): Additional keyword arguments for the transformation/shift methods.
        MASK_BORDER (function): Function for determining the border of the mask.
        NUM_ITER (int): Number of iterations for the transformation/shift optimization.
        TERMINATION_EPS (float): Termination epsilon for the transformation/shift optimization.
        GAUSSFILT_SIZE (int): Size of the Gaussian filter.
        AUTOFIX_GAUSS (int): Gaussian filter parameter for autofixing.
        ROI_NORM (bool): Flag indicating whether to normalize the ROI.
        KERNEL_HALFWIDTH (int): Half width of the kernel for blurring.
        PLOT_KERNEL (bool): Flag indicating whether to plot the kernel.
        DL_METH (str): Method for downloading the ROInet model.
        DL_URL (str): URL for downloading the ROInet model.
        DL_HASH (str): Hash value for verifying the downloaded ROInet model.
        FP_VERS (str): Version of the fingerprinting algorithm.
        PREF_PLOT (bool): Flag indicating whether to plot the generated dataloader.
        JIT_SCRIPT_TRANS (bool): Flag indicating whether to use torch.jit.script for speeding up the process.
        BATCHSIZE (int): Batch size for the PyTorch dataloader.
        PINMEMORY (bool): Flag indicating whether to use pin_memory for the PyTorch dataloader.
        NUMWORKERS (float): Number of workers for the PyTorch dataloader.
        PERSISTENTWORKERS (bool): Flag indicating whether to use persistent workers for the PyTorch dataloader.
        PREFETCHFACTOR (int): Prefetch factor for the PyTorch dataloader.
        KWARGS_SCAT2D (function): Function for getting the keyword arguments for the scattering wavelet transform.
        SWT_BATCH_SIZE (int): Batch size for each iteration of the scattering wavelet transform.
        N_WORKERS (int): Number of workers for the similarity calculation.
        BLOCK_HEIGHT (int): Height of the blocks for the similarity calculation.
        BLOCK_WIDTH (int): Width of the blocks for the similarity calculation.
        ALGO_NN_SF (int): Algorithm for nearest neighbor search.
        ALGO_NN_NORM (int): Normalization method for nearest neighbor search.
        SF_MASKPOWER (float): Exponent for raising spatial footprints in the similarity calculation.
        KMAX (int): Maximum number of neighbors for the similarity calculation.
        KMIN (int): Minimum number of neighbors for the similarity calculation.
        PLOTS2SHOW (List[int]): List of plot indices to show.
        MAX_SAMPLES (int): Maximum number of samples for the similarity calculation.
        SCAT_SZ (int): Size of the scatter plot markers.
        SCAT_ALPHA (float): Alpha value for the scatter plot markers.
        D_CUTOFF (None): Cutoff distance for the similarity calculation.
        STRINGENCY (float): Stringency parameter for the similarity calculation.
        CONV2PROB (bool): Flag indicating whether to convert the similarity values to probabilities.
        N_BINS (None): Number of bins for clustering.
        SMOOTH_WINDOW (None): Window size for smoothing.
        KW_FP (function): Function for getting the keyword arguments for finding parameters.
        BOUND_FP (function): Function for getting the bounds for finding parameters.
        N_JOBS_FP (function): Function for getting the number of jobs for finding parameters.
        MIN_CLUSTER_SIZE (int): Minimum number of ROIs that can be considered a 'cluster'.
        N_ITER_VC (int): Number of times to redo clustering sweep after removing violations.
        SPLIT_IS_CLUSTER (bool): Flag indicating whether to split clusters with ROIs from the same session.
        CLUSTER_SEL_METH (str): Method of cluster selection for HDBSCAN.
        D_CLUSTMRG (None): Distance below which all ROIs are merged into a cluster.
        FIT_ALPHA (float): Scalar applied to distance matrix in HDBSCAN.
        DISCARD_FAIL (bool): Flag indicating whether to set all ROIs that could be separated from clusters with ROIs from the same sessions to label=-1.
        NSTEPS_CLSTER_SPLT (int): Number of steps to remove violations when splitting clusters.
        THRESH_COST (float): Threshold cost for sequential Hungarian fitting.
        SP_FP_EXP (float): Threshold for the spatial footprint expansion.
        CL_SILHOUETTE (int): Threshold for the cluster silhouette.
        CL_INTRA_MEANS (float): Threshold for the intra-cluster means.
    """

    # centroid
    CENTROID_METHOD = CENTROID_METH[0]  # 0 for median; 1 for centerOfMass
    # height and width of the ROI
    ROI_HEIGHT_WIDTH = [36, 36]
    # TODO - may change this to using a tif of avg TempEXPDS image instead of CNMF.estimates.b
    UM_PER_PXL = 1.3
    # SESS2USE4FOV = "numSess0"

    # FOV augmentation
    ROI_FOV_MIX_FACTOR = 0.3
    USE_CLAHE = True
    CLAHE_GRID_SIZE = 1
    CLAHE_CLIPLIMIT = 1
    CLAHE_NORM = True

    # Transformation/shifts
    # Method for how to handle template used for registration
    TEMP_METHOD = TEMPLATE_METHOD[0]  # 0 for image; 1 for sequential
    # Methods use for transformation
    MODE_TRANS_GEO = MDT_GEO[
        1  # 0 for translation; 1 for euclidean; 2 for affine; 3 for homography
    ]
    MODE_TRANS_NONRIGID = MDT_NR[
        0
    ]  # 0 for createOptFlow_DeepFlow; 1 for calcOptFlowFarneback
    KWARGS_MODE_TRANS = None
    # IMP!!! esp for discrete data
    # limits area of data used for applying transformation/shift
    MASK_BORDER_DG = _get_mask_borders(20)
    MASK_BORDER_CA3 = _get_mask_borders(80)
    MASK_BORDER_KET = _get_mask_borders(50)
    NUM_ITER = 50
    TERMINATION_EPS = 1e-9
    GAUSSFILT_SIZE = 11
    AUTOFIX_GAUSS = 10
    ROI_NORM = True

    # Blurrer
    KERNEL_HALFWIDTH = 2
    PLOT_KERNEL = False

    # ROInet
    DL_METH = "check_local_first"
    DL_URL = "https://osf.io/x3fd2/download"
    DL_HASH = "7a5fb8ad94b110037785a46b9463ea94"
    FP_VERS = "latent"

    # Generate Dataloader
    PREF_PLOT = False
    JIT_SCRIPT_TRANS = (
        False  # whether or not to use torch.jit.script to speed things up
    )
    BATCHSIZE = 8  # PyTorch dataloader batch_size
    PINMEMORY = True  # PyTorch dataloder pin_memory
    NUMWORKERS = 10.0  # PyTorch dataloader num_workers
    PERSISTENTWORKERS = True  # PyTorch dataloader persistent_workers
    PREFETCHFACTOR = 2  # PyTorch dataloader prefetch_factor

    # scattering wavelet transform
    # J is the number of convolutional layers
    # L is the number of wavelet angles
    KWARGS_SCAT2D = _get_kwargs_scat2d(J=2, L=12)
    SWT_BATCH_SIZE = 100  # batch size ofr each iteration

    # Similarity
    N_WORKERS = -1
    BLOCK_HEIGHT = 128
    BLOCK_WIDTH = 128
    ALGO_NN_SF = ALGO_NN[0]  # 0 for brute; 1 for kd_tree; 2 for balltree; 3 for auto
    ALGO_NN_NORM = ALGO_NN[1]
    SF_MASKPOWER = 1.0  # exponent to raise spatial footprints to care more or less about bright pixels
    KMAX = 100
    KMIN = 10

    # Sim Plotting
    PLOTS2SHOW = [1, 2, 3]
    MAX_SAMPLES = 100000
    SCAT_SZ = 3
    SCAT_ALPHA = 0.2
    D_CUTOFF = None
    STRINGENCY = 1.3
    CONV2PROB = False

    # Clustering
    N_BINS = None
    SMOOTH_WINDOW = None
    # see func above for variable descriptions
    KW_FP, BOUND_FP, N_JOBS_FP = _get_kwargsNbounds_find_params()
    # only need this for session num < 8 when fitting cluster

    # Params for fitting
    ## Minimum number of ROIs that can be considered a 'cluster'
    MIN_CLUSTER_SIZE = 2
    ## Number of times to redo clustering sweep after removing violations
    N_ITER_VC = 6
    ## Whether or not to split clusters with ROIs from the same session
    SPLIT_IS_CLUSTER = True
    ## (advanced) Method of cluster selection for HDBSCAN (see hdbscan documentation)
    CLUSTER_SEL_METH = "leaf"
    ## Distance below which all ROIs are merged into a cluster
    D_CLUSTMRG = None
    ## (advanced) Scalar applied to distance matrix in HDBSCAN (see hdbscan documentation)
    FIT_ALPHA = 0.999
    ## (advanced) Whether or not to set all ROIs that could be separated from clusters with ROIs from the same sessions to label=-1
    DISCARD_FAIL = True
    ## (advanced) How finely to step through distances to remove violations
    NSTEPS_CLSTER_SPLT = 100
    # for sequential Hungarian fitting
    THRESH_COST = 0.6

    # Quality Metric Thresholds
    SP_FP_EXP = 0.7
    CL_SILHOUETTE = 0
    CL_INTRA_MEANS = 0.4


class Txt(Enum):
    """
    Enumeration class that defines various text labels used in the code.

    Attributes:
        AUG_FOV (str): Label for FOV augmentation.
        SP_FOOT (str): Label for spatial footprint.
        DISCARD (str): Label for discarded ROIs.
        DISCARD_QC (str): Label for discarded ROIs post-QC.
        ROI (str): Label for ROI.
        ROI_ALIGN (str): Label for aligned ROIs.
        UCELL (str): Label for underlying cells.
        TR_AS (str): Label for tracked ROIs in all sessions.
        TR_D (str): Label for tracked ROIs daily (S1 vs S2).
        TR_W (str): Label for tracked ROIs weekly (S1 vs S3).
        TR_D_QC (str): Label for tracked ROIs daily post-QC (S1 vs S2).
        TR_W_QC (str): Label for tracked ROIs weekly post-QC (S1 vs S3).
        TR_AS_QC (str): Label for tracked ROIs in all sessions post-QC.
        CLUSTERS (str): Label for clusters.
        ACLUSTERS (str): Label for accepted clusters.
        ACLUSTERS_QC (str): Label for clusters passing QC.
        LABELS (str): Label for labels.
        ROIS (str): Label for ROIs.
        QM (str): Label for quality metrics.
        N_SESS (str): Label for number of sessions.
        CL_SILHOUETTE (str): Label for cluster silhouette threshold.
        CL_INTRA_MEANS (str): Label for cluster intra means threshold.
        PC_SX (str): Label for CueCells in session X. X is generated
        PC_SX_QC (str): Label for CueCells in session X post-QC.
        D2D (str): Label for day-to-day counts.
        D2D_QC (str): Label for day-to-day counts post-QC.
        W2W (str): Label for week-to-week counts.
        W2W_QC (str): Label for week-to-week counts post-QC.
        ALLSESS (str): Label for all sessions counts.
        ALLSESS_QC (str): Label for all sessions counts post-QC.
        CENT (str): Label for centroids.
        BOUND (str): Label for bounds.
        MIN_D (str): Label for minimum distance.
        CONTOUR (str): Label for contours.
        SHIFT_PLT (str): Format string for shift plot titles.
        SHIFT_FIG (str): Format string for shift figure titles.
        FIG_FOLDER (str): Folder name for saving figures.
        PAIRWISE (str): Label for pairwise similarities plot.
        SIM_REL (str): Label for similarity relationships plot.
        SIM (str): Label for similarity plot.
        SP_FP (str): Label for spatial footprints plot.
        NN (str): Label for neural network plot.
        SWT (str): Label for scattering wavelet transform plot.
        OP_KW (str): Label for optimal parameters keyword.
        OP_KW_SAVE (str): File name for saving optimal parameters.
        C_INFO (str): Label for cluster information.
        C_INFO_SAVE (str): File name for saving cluster information.
        RESULTS (str): Label for ROICaT results.
        RESULTS_SAVE (str): File name for saving ROICaT results.
        RUN_DATA (str): Label for ROICaT run data.
        RUN_DATA_SAVE (str): File name for saving ROICaT run data.
    """

    # keys for dicts
    AUG_FOV = "aFOV"
    SP_FOOT = "spFT"
    DISCARD = "discarded_ROIs"
    DISCARD_QC = "discarded_ROIs_post_QC"
    ROI = "ROI"
    ROI_ALIGN = "ROIs_aligned"
    UCELL = "underlying_cells"
    U_TC = "underlying_tuned_cells"
    U_CC = "underlying_cue_cells"
    T_TC = "tracked_tuned_cells"
    T_CC = "tracked_cue_cells"
    UNT_TC = "nontracked_tuned_cells"
    UNT_CC = "nontracked_cue_cells"
    TR_AS = "trackedROIs_in_allSess"
    TR_D = "trackedROIs_daily_S1vS2"
    TR_D_C = "trackedROIs_daily_S1vS2_CueCells"
    TR_D_T = "trackedROIs_daily_S1vS2_TunedCells"
    TR_W = "trackedROIs_weekly_S1vsS3"
    TR_W_C = "trackedROIs_weekly_S1vsS3_CueCells"
    TR_W_T = "trackedROIs_weekly_S1vsS3_TunedCells"
    TR_W2 = "trackedROIs_weekly_S2vsS3"
    TR_W2_C = "trackedROIs_weekly_S2vsS3_CueCells"
    TR_W2_T = "trackedROIs_weekly_S2vsS3_TunedCells"
    TR_D_QC = "trackedROIs_daily_S1vS2_postQC"
    TR_D_C_QC = "trackedROIs_daily_S1vS2_CueCells_postQC"
    TR_D_T_QC = "trackedROIs_daily_S1vS2_TunedCells_postQC"
    TR_W_QC = "trackedROIs_weekly_S1vsS3_postQC"
    TR_W_C_QC = "trackedROIs_weekly_S1vsS3_CueCells_postQC"
    TR_W_T_QC = "trackedROIs_weekly_S1vsS3_TunedCells_postQC"
    TR_W2_QC = "trackedROIs_weekly_S2vsS3_postQC"
    TR_W2_C_QC = "trackedROIs_weekly_S2vsS3_CueCells_postQC"
    TR_W2_T_QC = "trackedROIs_weekly_S2vsS3_TunedCells_postQC"
    TR_AS_QC = "trackedROIs_in_allSess_postQC"
    CLUSTERS = "clusters"
    ACLUSTERS = "accepted_clusters"
    ACLUSTERS_QC = "clusters_passing_qc"
    LABELS = "labels"
    ROIS = "ROIs"
    QM = "quality_metrics"
    N_SESS = "n_sessions"
    CL_SILHOUETTE = "cluster_silhouette_thresh"
    CL_INTRA_MEANS = "cluster_intra_means_thresh"

    # key dicts for counts
    D2D = "day2day"
    D2D_QC = "day2day_postQC"
    D2D_CC = "day2day_CueCells"
    D2D_CC_QC = "day2day_CueCells_postQC"
    D2D_TC = "day2day_TunedCells"
    D2D_TC_QC = "day2day_TunedCells_postQC"
    W2W = "week2week"
    W2W_QC = "week2week_postQC"
    W2W_CC = "week2week_CueCells"
    W2W_CC_QC = "week2week_CueCells_postQC"
    W2W_TC = "week2week_TunedCells"
    W2W_TC_QC = "week2week_TunedCells_postQC"
    W2W2 = "week2week2"
    W2W2_QC = "week2week2_postQC"
    W2W2_CC = "week2week2_CueCells"
    W2W2_CC_QC = "week2week2_CueCells_postQC"
    W2W2_TC = "week2week2_TunedCells"
    W2W2_TC_QC = "week2week2_TunedCells_postQC"
    ALLSESS = "allSess"
    ALLSESS_QC = "allSess_postQC"

    # keys for PC dict
    CENT = "centroids"
    BOUND = "bounds"
    MIN_D = "min_dist"
    CONTOUR = "contours"

    # FIG & PLOT titles
    SHIFT_PLT = "({}-Shift)"
    SHIFT_FIG = "{}Shift"

    # Figure save folder
    FIG_FOLDER = "_Figures_ROICaT"

    # Plotting
    PAIRWISE = "Pairwise similarities"
    SIM_REL = "Similarity relationships"
    SIM = "sim"
    SP_FP = "Spatial Footprints"
    NN = "Neural Network"
    SWT = "Scattering Wavelet Transform"

    # saving results dict
    OP_KW = "Optimal Parameters for best Conjunctive Distance Matrix"
    OP_KW_SAVE = "_kwargs_mCDM_best"
    C_INFO = "Cluster Information"
    C_INFO_SAVE = "_cluster_info_ROICaT"
    RESULTS = "ROICaT Results"
    RESULTS_SAVE = "_results_ROICaT"
    RUN_DATA = "ROICaT Run_Data"
    RUN_DATA_SAVE = "_rundata_ROICaT"

    TC_SX = "TunedCells_S{}"
    TC_SX_QC = "TunedCells_S{}_postQC"

    CC_SX = "CueCells_S{}"
    CC_SX_QC = "CueCells_S{}_postQC"


class TxtPrint(Enum):
    """
    Enumeration class that defines various text labels for printing results.

    Attributes:
        UCELL (str): Message for the number of underlying cells.
        ACLUSTER (str): Message for the number of clusters found/cells that appear in multiple sessions.
        ACLUSTER_QC (str): Message for the number of clusters found post-QC filter.
        DISCARD (str): Message for the number of discarded ROIs.
        DISCARD_QC (str): Message for the number of discarded ROIs post-QC filter.
        TR_D (str): Message for the number of cells tracked from S1 to S2 (day-to-day).
        TR_W (str): Message for the number of cells tracked from S1 to S3 (week-to-week).
        TR_AS (str): Message for the number of cells tracked present in all sessions.
        TR_D_QC (str): Message for the number of cells tracked from S1 to S2 (day-to-day) post-QC filter.
        TR_W_QC (str): Message for the number of cells tracked from S1 to S3 (week-to-week) post-QC filter.
        TR_AS_QC (str): Message for the number of cells tracked present in all sessions post-QC filter.
        PC_S1 (str): Message for the number of Cue Cells in S1.
        PC_S2 (str): Message for the number of Cue Cells in S2.
        PC_S3 (str): Message for the number of Cue Cells in S3.
        PC_S1_QC (str): Message for the number of Cue Cells in S1 post-QC filter.
        PC_S2_QC (str): Message for the number of Cue Cells in S2 post-QC filter.
        PC_S3_QC (str): Message for the number of Cue Cells in S3 post-QC filter.
    """

    UCELL = "Number of underlying cells:"
    U_TC = "Number of underlying tuned cells:"
    U_CC = "Number of underlying cue cells:"
    T_TC = "Number of tracked tuned cells:"
    T_CC = "Number of tracked cue cells:"
    UNT_TC = "Number of nontracked tuned cells:"
    UNT_CC = "Number of nontracked cue cells:"
    ACLUSTER = "Number of clusters found/cells that appear in multiple sessions:"
    ACLUSTER_QC = "Number of clusters found post-QC filter:"
    DISCARD = "Number of discarded ROIs:"
    DISCARD_QC = "Number of discarded ROIs post-QC filter:"
    TR_D = "Number of cells tracked from S1 to S2 (day2day):"
    TR_W = "Number of cells tracked from S1 to S3 (week2week):"
    TR_W2 = "Number of cells tracked from S2 to S3 (week2week2):"
    TR_AS = "Number of cells tracked present in all sessions:"
    TR_D_QC = "Number of cells tracked from S1 to S2 (day2day) post-QC filter:"
    TR_W_QC = "Number of cells tracked from S1 to S3 (week2week) post-QC filter:"
    TR_W2_QC = "Number of cells tracked from S2 to S3 (week2week2) post-QC filter:"
    TR_AS_QC = "Number of cells tracked present in all sessions post-QC filter:"
    TR_D_CC = "Number of Cue Cells tracked from S1 to S2 (day2day):"
    TR_D_CC_QC = "Number of Cue Cells tracked from S1 to S2 (day2day) post-QC filter:"
    TR_W_CC = "Number of Cue Cells tracked from S1 to S3 (week2week):"
    TR_W_CC_QC = "Number of Cue Cells tracked from S1 to S3 (week2week) post-QC filter:"
    TR_W2_CC = "Number of Cue Cells tracked from S2 to S3 (week2week2):"
    TR_W2_CC_QC = (
        "Number of Cue Cells tracked from S2 to S3 (week2week2) post-QC filter:"
    )
    TR_D_TC = "Number of Tuned Cells tracked from S1 to S2 (day2day):"
    TR_D_TC_QC = "Number of Tuned Cells tracked from S1 to S2 (day2day) post-QC filter:"
    TR_W_TC = "Number of Tuned Cells tracked from S1 to S3 (week2week):"
    TR_W_TC_QC = (
        "Number of Tuned Cells tracked from S1 to S3 (week2week) post-QC filter:"
    )

    CC_SX = "Number of Cue Cells in S{}:"
    CC_SX_QC = "Number of Cue Cells in S{} post-QC filter:"

    TC_SX = "Number of Tuned Cells in S{}:"
    TC_SX_QC = "Number of Tuned Cells in S{} post-QC filter:"


class Parser(Enum):
    """
    Enumeration class that defines various parameters for the parser used for cell registration analysis

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
    """

    PARSER = "Parser"
    HEADER = "Cell Registration Analysis"
    PARSER4 = "CRwROI"
    PARSER_FN = "CellRegistar with ROICaT"
    ARG_DICT = {
        ("sessFocus", "sf"): {
            "TYPE": "int",
            "DEFAULT": None,
            "HELP": "Set number of sessions to analyze. Default is None, which will analyze all sessions found within multSessSegStruc.",
        },
        ("useGPU", "G"): {
            "TYPE": "bool",
            "DEFAULT": True,
            "HELP": "Whether to use GPU for ROICaT functions. Default is True",
        },
        ("verbose", "v"): {
            "TYPE": "bool",
            "DEFAULT": True,
            "HELP": "Whether to print verbose output for ROICaT functions. Default is True",
        },
    }


class Parser4CI(Enum):
    """
    Enumeration class that defines various parameters for the parser used for cluster info collater

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
    """

    PARSER = "Parser4CI"
    HEADER = "Cluster Info Collater"
    PARSER4 = "CRwROI"
    PARSER_FN = "CellRegistrar Cluster Info Collater"
    ARG_DICT = {
        ("forPres", "4p"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": text_dict()["parser"]["forPres"],
        },
    }
