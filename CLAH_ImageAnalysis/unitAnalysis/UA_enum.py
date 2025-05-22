from enum import Enum

from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis.behavior import TDML2tBD_enum as TDMLE
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_enum

text_lib = utils.text_dict()
dict_name = text_lib["dict_name"]
PCLAkey = utils.enum_utils.enum2dict(PCLA_enum.TXT)
TDMLkey = utils.enum_utils.enum2dict(TDMLE.TXT)


#############################################################
# pks dict
#############################################################


class PKS(Enum):
    """
    Enum class representing constants for peak dictionaries.

    Attributes:
        PEAKS (str): Key for peaks.
        AMPS (str): Key for amplitudes.
        WAVEFORM (str): Key for waveform.
        SEG (str): Key for segment.
        LAP_TYPE (str): Key for lap type.
    """

    PEAKS = "pks"
    AMPS = "amps"
    WAVEFORM = "waveform"
    SEG = "seg"
    LAP_TYPE = "lapType"

    FRAME_WINDOW = 15
    SD_THRESH = 3
    TIMEOUT = 3

    PK_ALGO = ("iterative_diffs", "scipy")

    SP_HEIGHT = 0.1
    SP_DISTANCE = 1
    SP_PROMINENCE = 0.35

    BSL_ADJ2BSL = False
    BSL_SCALE_WINDOW = 500
    BSL_QUANTILE = 0.08
    BSL_SHIFT_MIN2ZERO = False
    BSL_LOG4BSL = True

    SMOOTH_BOOL = False
    WINDOW_SIZE = 11
    SMOOTHING_ORDER = 3
    LOG4SG = False


#############################################################
# cueShiftStruc
#############################################################


class CSS(Enum):
    """
    Enum class `CSS` defining string constants used as keys in the cue shift structure.

    Attributes:
        PKS_BY_LT: String for a key related to peaks by light.
        POS_LAP: String for a key representing position in a lap.
        PCLS: String for a key denoting 'PCLappedSess'.
        LAP_CUE: String for a key related to lap cues.
        LAP_KEY: String for a key representing a lap.
        LAP_TYPEARR: String for a key indicating an array of lap types.
        FNAME: String for a key for the filename.
        SEGDICT: String for a key related to segment dictionary, derived from a text library.
        PATH: String for a key representing a file or directory path.
        PKS: String for a key related to peaks.
        POSRATE: String for a key indicating position rate, derived from the PCLA constants.
        SHUFF: String for a key related to shuffling, derived from the PCLA constants.
        ISPC: String for a key representing interspike correlation, derived from the PCLA constants.
        LAP_DICT: String for a key related to lap information.
        CSS: String for a key denoting 'cueShiftStruc'.
    """

    PKS_BY_LT = "pksByLT"
    POS_LAP = "posLap"
    PCLS = "PCLappedSess"
    LAP_CUE = "lapCue"
    LAP_KEY = "lap"
    LAP_TYPEARR = "TypeArr"
    FNAME = "Filename"
    SD = dict_name["SD"]
    PATH = "path"
    PKS = "pks"
    POSRATE = PCLAkey["POSRATE"]
    SHUFF = PCLAkey["SHUFF"]
    ISPC = PCLAkey["ISPC"]
    LAP_DICT = "lapInfo"
    CSS = "cueShiftStruc"


#############################################################
# lapCueDict
#############################################################


class LCD(Enum):
    """
    Enum class `LCD` defining string constants used as keys in the lap cue dictionary.

    Attributes:
        BINNEDPOS: String for a key representing binned positions.
        LAP_TYPEARR: String for a key indicating an array of lap types.
        LAP_EPOCH: String for a key denoting epochs within a lap.
        LAP_FRIDX: String for a key related to frame indices in a lap.
        REFLAP: String for a key referring to the reference lap type.
        LAP_TANAME: String for a key representing the name of a lap type.
        NUM_LAP_TYPES: String for a key indicating the number of lap types.
        MAX_CUE: String for a key denoting the maximum cue value.
        POSITION_KEY: String for a key representing position, derived from the TDML constants.
        TYPE_KEY: String for a key indicating the type, derived from the TDML constants.
        START: String for a key marking the start, derived from the TDML constants.
        STOP: String for a key marking the stop, derived from the TDML constants.
        LAP_KEY: String for a key related to lap identification, derived from the TDML constants.
        LAP_NUM_KEY: String for a key indicating lap number, derived from the TDML constants.
        LAP_BIN_KEY: String for a key related to lap binning.
        LAP_LOC_KEY: String for a key indicating lap location.
        NAME_KEY: String for a key used for naming, derived from the TDML constants.
        TIME_KEY: String for a key related to time, derived from the TDML constants.
        OLF: String for a key associated with olfactory cues.
        CUE1, CUE2: Strings for keys related to the first and second cues, converted to uppercase from TDML constants.
        OPTO: String for a key related to optogenetics, derived from the TDML constants.
        TACT: String for a key related to tactile elements, derived from the TDML constants.
        TONE: String for a key related to tone, derived from the TDML constants.
        LED: String for a key related to LED, derived from the TDML constants.
        REW: String for a key related to reward, converted to uppercase from the TDML constant.
    """

    BINNEDPOS = "BinnedPos"
    LAP_TYPEARR = "TypeArr"
    LAP_EPOCH = "Epochs"
    LAP_FRIDX = "FrInds"
    REFLAP = "refLapType"
    LAP_TANAME = "lapTypeName"
    NUM_LAP_TYPES = "numLabTypes"
    MAX_CUE = "max_cue"
    POSITION_KEY = TDMLkey["POSITION_KEY"]
    TYPE_KEY = TDMLkey["TYPE"]
    START = TDMLkey["START"]
    STOP = TDMLkey["STOP"]
    LAP_KEY = TDMLkey["LAP_KEY"]
    LAP_NUM_KEY = TDMLkey["LAP_NUM_KEY"]
    LAP_BIN_KEY = "LapBin"
    LAP_LOC_KEY = "LabLoc"
    NAME_KEY = TDMLkey["NAME_KEY"]
    TIME_KEY = TDMLkey["TIME_KEY"]
    OLF = "olf"
    CUE1 = TDMLkey["CUE1"].upper()
    CUE2 = TDMLkey["CUE2"].upper()
    OPTO = TDMLkey["OPTO"].upper()
    TACT = TDMLkey["TACT"].upper()
    TONE = TDMLkey["TONE"].upper()
    LED = TDMLkey["LED"].upper()
    REW = TDMLkey["REWARD"].upper()
    NOCUE = TDMLkey["NOCUE"].upper()


def create_keys_lCD_cueTypes(const_dict: dict, cue_types: list) -> dict:
    """
    Create keys for lapCueDict based on cue types.

    Args:
        const_dict (dict): Dictionary of constants.
        cue_types (list): List of cue types.

    Returns:
        dict: Dictionary with keys for lapCueDict based on cue types.
    """

    cueTypeDict = {}
    for cue_type in cue_types:
        cueTypeDict[const_dict[cue_type]] = {
            const_dict["POSITION_KEY"]: {
                const_dict["START"]: [],
                const_dict["STOP"]: [],
            },
            const_dict["LAP_NUM_KEY"]: [],
            const_dict["LAP_BIN_KEY"]: [],
            const_dict["LAP_LOC_KEY"]: [],
            const_dict["TIME_KEY"]: [],
            const_dict["BINNEDPOS"]: [],
        }
    return cueTypeDict


def create_lapCueDict():
    """
    Create the lapCueDict dictionary.

    Returns:
        dict: The lapCueDict dictionary.
    """

    const_dict = utils.enum_utils.enum2dict(LCD)

    lapCueDict = {
        const_dict["LAP_KEY"]: {
            const_dict["LAP_TYPEARR"]: [],
            const_dict["LAP_EPOCH"]: [],
            const_dict["LAP_FRIDX"]: [],
        },
        const_dict["NUM_LAP_TYPES"]: [],
    }
    # List of cue types to be added
    cue_types = ["CUE1", "CUE2", "OPTO", "TACT", "TONE", "LED"]
    # Add the cue types
    lapCueDict.update(create_keys_lCD_cueTypes(const_dict, cue_types))

    return lapCueDict


######################################################
# CueCellFInder
######################################################


class CCF(Enum):
    """
    Enumeration class that defines various parameters for the CueCellFinder.

    Attributes:
        ATRIGSIG (str): Label for all trigger signals.
        CTRIGSIG (str): Label for cue trigger signals.
        RANKSUM (str): Label for RankSum dictionary.
        TTEST (str): Label for TTest dictionary.
        AMP (str): Label for cue amplitude.
        EVTIME (str): Label for event times.
        CELLIND (str): Label for CueCell indices.
        IND (List[int]): List of indices for CueCellFinder.

        OM (str): Label for OM parameter.
        SH (str): Label for SH parameter.
        C1 (str): Label for CUE1 parameter.
        C2 (str): Label for CUE2 parameter.
        B (str): Label for BOTH parameter.
        C1L1 (str): Label for CUE1_L1 parameter.
        C1L2 (str): Label for CUE1_L2 parameter.
        T (str): Label for TONE parameter.
        L (str): Label for LED parameter.
        A (str): Label for ALL parameter.
        SW (str): Label for SWITCH parameter.
    """

    ATRIGSIG = "allTrigSig"
    CTRIGSIG = "cueTrigSig"
    RANKSUM = "RankSumDict"
    TTEST = "TTestDict"
    AMP = "cueAmp"
    EVTIME = "evTimes"
    CELLIND = "CueCellInd"
    CELLTABLE = "CueCellTable"
    OPTODIFF = "OptoDiff"
    OPTOTRIGSIG = "OptoTrigSig"
    OPTOAMP = "OptoAmp"
    OPTOSTD = "OptoSTD"
    OPTOIDX = "OptoIdx"
    S_OPTOAMP = "selectOptoAmp"
    S_OPTOSTD = "selectOptoSTD"
    S_OPTOIDX = "selectOptoIdx"
    IND = [-30, 121]

    OM = "OMIT"
    SH = "SHIFT"
    C1 = "CUE1"
    C2 = "CUE2"
    B = "BOTH"
    C1L1 = "CUE1_L1"
    C1L2 = "CUE1_L2"
    T = "TONE"
    L = "LED"
    A = "ALL"
    OP = "OPTO"
    SW = "SWITCH"


class CCF_PLOT(Enum):
    """
    Enumeration class that defines various parameters for plotting in the CueCellFinder.

    Attributes:
        ORDER (List[str]): List of order labels for plotting.
        XTICKS (List[int]): List of x-tick positions.
        XPOS (str): Label for x position.
        CUELEDTONE (str): Label for CUE/LED/TONE.
        OMITCUE1_TTL (str): Label for OMITCUE1_TTL.
        OMITTONE_TTL (str): Label for OMITTONE_TTL.
        OMITLED_TTL (str): Label for OMITLED_TTL.
        CBAR_POS (str): Label for color bar position.
        MEAN_PR (str): Label for mean position rate.
        CUECELLS (str): Label for Cue Cells.
        SEM (str): Label for SEM.
        VIO (str): Label for VioPlot.
    """

    ORDER = [
        "CUE1",
        "CUE1_SWITCH",
        "CUE2",
        "CUE2_SWITCH",
        "CUEwOPTO",
        "LED",
        "OMITBOTH",
        "OMITCUE1",
        "OMITCUE1_SWITCH",
        "OMITCUE2",
        "OMITLED",
        "OMITOPTO",
        "OMITTONE",
        "OPTO",
        "TONE",
    ]
    XTICKS = [0, 25, 50, 75, 100]
    XPOS = "Position"
    CUELEDTONE = "CUE/LED/TONE"
    OMITCUE1_TTL = "LED/TONE"
    OMITTONE_TTL = "CUE/LED"
    OMITLED_TTL = "CUE/TONE"
    CBAR_POS = "left"
    MEAN_PR = "Mean posRate"
    CUECELLS = "Cue Cells"
    SEM = "_SEM"
    VIO = "VioPlot"


######################################################
#  PCR Cue Cell Finder
######################################################


class PCR_CFF(Enum):
    """
    Enumeration class that defines various parameters for the PCR Cue Cell Finder.

    Attributes:
        TS_BC (str): Label for trigger signals by cluster.
        CC_BTS_BC (str): Label for CueCell by trigger signal by cluster.
        MEAN (str): Label for mean by CueCell by CueType.
    """

    TS_BC = "TrigSig_byCluster"
    CC_BTS_BC = "CueCell_byTrigSig_byCluster"
    MEAN_TS = "mean_byCC_byCueType"
    MEAN_MV = "meanMaxVal_byCC_byCueType"
    OPTO = "OptoAmpArr"
    POS_RATE = "posRate_bySess"
    POS_RATE_TC = "posRate_bySess_TC"
    POS_RATE_CC = "posRate_bySess_CC"
    TC_IDX = "TC_IDX"
    CC_IDX = "CC_IDX"


######################################################
#  PARSER
######################################################


class Parser4QT(Enum):
    """
    Enumeration class that defines various parameters for the parser used for quickTuning

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
        TYPE_LIST (List[str]): List of types for the arguments.
    """

    PARSER = "Parser"
    HEADER = "Unit Analysis (Head Program: quickTuning)"
    PARSER4 = "UA"
    PARSER_FN = "quickTuning"
    ARG_DICT = {
        ("frameWindow", "fw"): {
            "TYPE": "int",
            "DEFAULT": PKS.FRAME_WINDOW.value,
            "HELP": "Window size (in frames) used for smoothing for the event/peak detection. Default is 15",
        },
        ("sdThresh", "sdt"): {
            "TYPE": "int",
            "DEFAULT": PKS.SD_THRESH.value,
            "HELP": "Threshold multiplier for event/peak detection based on the standard deviation of the signal's derivative. Default is 3",
        },
        ("timeout", "to"): {
            "TYPE": "int",
            "DEFAULT": PKS.TIMEOUT.value,
            "HELP": "Minimum distance between detected peaks/events in seconds. Default is 3",
        },
        ("overwrite", "ow"): {
            "TYPE": "bool",
            "DEFAULT": None,
            "HELP": "Overwrite existing files (e.g. pkl & mat for treadBehDict, lapDict, cueShiftStruc)",
        },
        ("toPlotPks", "pp"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Whether to plot results from pks_utils. Default is False",
        },
        ("forPres", "4p"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": text_lib["parser"]["forPres"],
        },
    }


class Parser4WMSS(Enum):
    """
    Enumeration class that defines various parameters for the parser used for wrapMultSessStruc

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
        TYPE_LIST (List[str]): List of types for the arguments.
    """

    PARSER = "Parser4WMSS"
    HEADER = "Unit Analysis (wrapMultSessStruc)"
    PARSER4 = "UA"
    PARSER_FN = "wrapMultSessStruc"
    ARG_DICT = {
        ("output_folder", "out"): {
            "TYPE": "out_path",
            "DEFAULT": [],
            "HELP": "Path for the output of pkl & mat files for multSessSegStruct. Default is None, which prompts user to input experiment keywords & brain region to create output folder name. All output paths will be prepended with '_MS_'.",
        },
    }


class Parser4PCRCFF(Enum):
    """
    Enumeration class that defines various parameters for the parser used for Post Cell Registration Cue Cell Finder

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
    """

    PARSER = "Parser4PCRCFF"
    HEADER = "Unit Analysis (Post Cell Registration Cue Cell Finder)"
    PARSER4 = "UA"
    PARSER_FN = "Post CellRegistrar Cue Cell Finder"
    ARG_DICT = {
        ("outlier_ts", "ots"): {
            "TYPE": "int",
            "DEFAULT": 10**2,
            "HELP": "Outlier threshold to filter out meanTrigSig by group where mean value exceeds threshold set here. Default is 10^2.",
        },
        ("sessFocus", "sf"): {
            "TYPE": "int",
            "DEFAULT": None,
            "HELP": "Select number of sessions to plot. Default is None, which plots all sessions.",
        },
        ("forPres", "4p"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": text_lib["parser"]["forPres"],
        },
        ("plotIndTrigSig", "pit"): {
            "TYPE": "bool",
            "DEFAULT": True,
            "HELP": "Whether to plot individual TrigSig by subjected ID. Default is True.",
        },
        ("concatCheck", "cat"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Set this to True if you are working with sessions that were motion corrected in a concatenated manner. THIS ONLY WORKS IF MS CONTAINS ONLY 2 SESSIONS AKA THE SPLIT OUTPUT OF 1 CONCATENATED SESSION. Default is False",
        },
    }
