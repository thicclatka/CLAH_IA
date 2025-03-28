import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sparse
import streamlit as st
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from CLAH_ImageAnalysis.utils import db_utils
from CLAH_ImageAnalysis.utils import Streamlit_utils
from CLAH_ImageAnalysis.utils import text_dict
from CLAH_ImageAnalysis.utils import findLatest
from CLAH_ImageAnalysis.utils import saveNloadUtils
from CLAH_ImageAnalysis.utils import image_utils
from CLAH_ImageAnalysis.utils import fig_tools
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import color_dict
from CLAH_ImageAnalysis.unitAnalysis import pks_utils
from CLAH_ImageAnalysis.unitAnalysis import UA_enum
from CLAH_ImageAnalysis.decoder import GeneralModelRunner
from CLAH_ImageAnalysis.decoder import NNModel4BinaryClassification
from CLAH_ImageAnalysis.decoder import FeatureExtractor4segDict
from CLAH_ImageAnalysis.decoder import dbHandler4CompEvaluator


ACC_REJ_JSON_FNAME = "accepted_rejected_components.json"
PKS_FNAME = "_pksDict.json"
DB_NAME = "paths_cache4SD"
DB_NAME4NN = "paths_cache4NN_COMP_EVAL"
EMOJIS = text_dict()["emojis"]
HASH = text_dict()["breaker"]["hash"]
CMAPS = [
    ListedColormap(["none", color_dict()["red"]]),
    ListedColormap(["none", color_dict()["blue"]]),
    ListedColormap(["none", color_dict()["green"]]),
    ListedColormap(["none", color_dict()["yellow"]]),
    ListedColormap(["none", color_dict()["violet"]]),
    ListedColormap(["none", color_dict()["orange"]]),
]


def init_session_state():
    if "paths" not in st.session_state:
        st.session_state.paths = []
    if "accepted" not in st.session_state:
        st.session_state.accepted = set()
    if "rejected" not in st.session_state:
        st.session_state.rejected = set()
    if "undecided" not in st.session_state:
        st.session_state.undecided = set()
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = 0
    if "prev_index" not in st.session_state:
        st.session_state.prev_index = 0
    if "frameWindow" not in st.session_state:
        st.session_state.frameWindow = UA_enum.PKS.FRAME_WINDOW.value
    if "sdThresh" not in st.session_state:
        st.session_state.sdThresh = UA_enum.PKS.SD_THRESH.value
    if "timeout" not in st.session_state:
        st.session_state.timeout = UA_enum.PKS.TIMEOUT.value
    if "allcmap_images" not in st.session_state:
        st.session_state.allcmap_images = []
    if "preview_json_bool" not in st.session_state:
        st.session_state.preview_json_bool = False
    if "prev_session" not in st.session_state:
        st.session_state.prev_session = None
    if "adj2bsl_bool" not in st.session_state:
        st.session_state.adj2bsl_bool = UA_enum.PKS.BSL_ADJ2BSL.value
    if "data2use" not in st.session_state:
        st.session_state.data2use = None
    if "temporal_data" not in st.session_state:
        st.session_state.temporal_data = None
    if "adj2bsl_data" not in st.session_state:
        st.session_state.adj2bsl_data = None
    if "peak_algorithm" not in st.session_state:
        st.session_state.peak_algorithm = UA_enum.PKS.PK_ALGO.value[0]
    if "height" not in st.session_state:
        st.session_state.height = UA_enum.PKS.SP_HEIGHT.value
    if "distance" not in st.session_state:
        st.session_state.distance = UA_enum.PKS.SP_DISTANCE.value
    if "prominence" not in st.session_state:
        st.session_state.prominence = UA_enum.PKS.SP_PROMINENCE.value
    if "scale_window" not in st.session_state:
        st.session_state.scale_window = UA_enum.PKS.BSL_SCALE_WINDOW.value
    if "quantile" not in st.session_state:
        st.session_state.quantile = UA_enum.PKS.BSL_QUANTILE.value
    if "shiftMin2Zero" not in st.session_state:
        st.session_state.shiftMin2Zero = UA_enum.PKS.BSL_SHIFT_MIN2ZERO.value
    if "log4bsl" not in st.session_state:
        st.session_state.log4bsl = UA_enum.PKS.BSL_LOG4BSL.value
    if "smooth_bool" not in st.session_state:
        st.session_state.smooth_bool = UA_enum.PKS.SMOOTH_BOOL.value
    if "window_size" not in st.session_state:
        st.session_state.window_size = UA_enum.PKS.WINDOW_SIZE.value
    if "smoothing_order" not in st.session_state:
        st.session_state.smoothing_order = UA_enum.PKS.SMOOTHING_ORDER.value
    if "log4sg" not in st.session_state:
        st.session_state.log4sg = UA_enum.PKS.LOG4SG.value
    if "preview_pks_dict_bool" not in st.session_state:
        st.session_state.preview_pks_dict_bool = False
    if "loaded_AL_fromSDISX" not in st.session_state:
        st.session_state.loaded_AL_fromSDISX = None
    if "CNMF_accepted" not in st.session_state:
        st.session_state.CNMF_accepted = None
    if "CNMF_rejected" not in st.session_state:
        st.session_state.CNMF_rejected = None
    if "export_menu_bool" not in st.session_state:
        st.session_state.export_menu_bool = False
    if "adjust_labels_bool" not in st.session_state:
        st.session_state.adjust_labels_bool = False
    if "pksDict" not in st.session_state:
        st.session_state.pksDict = {}
    if "CTemp" not in st.session_state:
        st.session_state.CTemp = None
    if "ASpat" not in st.session_state:
        st.session_state.ASpat = None
    if "DSimage_fname" not in st.session_state:
        st.session_state.DSimage_fname = None
    if "ASPatList" not in st.session_state:
        st.session_state.ASPatList = []
    if "CTempList" not in st.session_state:
        st.session_state.CTempList = []
    if "listExpNames" not in st.session_state:
        st.session_state.listExpNames = []
    if "labelList" not in st.session_state:
        st.session_state.labelList = []
    if "dim4spatial" not in st.session_state:
        st.session_state.dim4spatial = None
    if "dim4spatialList" not in st.session_state:
        st.session_state.dim4spatialList = []
    if "nn_success_bool" not in st.session_state:
        st.session_state.nn_success_bool = False
    if "model_runner" not in st.session_state:
        st.session_state.model_runner = None
    if "feature_extractor" not in st.session_state:
        st.session_state.feature_extractor = None
    if "features_vector" not in st.session_state:
        st.session_state.features_vector = None
    if "labels_vector" not in st.session_state:
        st.session_state.labels_vector = None
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "model_name2save" not in st.session_state:
        st.session_state.model_name2save = ""
    if "model_name2load" not in st.session_state:
        st.session_state.model_name2load = ""
    if "run_train_model_bool" not in st.session_state:
        st.session_state.run_train_model_bool = False
    if "NNCE_accepted" not in st.session_state:
        st.session_state.NNCE_accepted = set()
    if "NNCE_rejected" not in st.session_state:
        st.session_state.NNCE_rejected = set()
    if "NNCE_predictions" not in st.session_state:
        st.session_state.NNCE_predictions = None
    if "use_CNMF_CE_checkbox_result" not in st.session_state:
        st.session_state.use_CNMF_CE_checkbox_result = False
    if "use_AL_SDISX_checkbox_result" not in st.session_state:
        st.session_state.use_AL_SDISX_checkbox_result = False
    if "use_NNCE_checkbox_result" not in st.session_state:
        st.session_state.use_NNCE_checkbox_result = False
    if "CNMF_CE_bool" not in st.session_state:
        st.session_state.CNMF_CE_bool = False
    if "AL_SDISX_accepted" not in st.session_state:
        st.session_state.AL_SDISX_accepted = set()
    if "AL_SDISX_rejected" not in st.session_state:
        st.session_state.AL_SDISX_rejected = set()
    if "AL_SDISX_undecided" not in st.session_state:
        st.session_state.AL_SDISX_undecided = set()
    if "save_model_bool" not in st.session_state:
        st.session_state.save_model_bool = False
    if "load_model_bool" not in st.session_state:
        st.session_state.load_model_bool = False
    if "selected_model_dict" not in st.session_state:
        st.session_state.selected_model_dict = {}
    if "eval_completed_bool" not in st.session_state:
        st.session_state.eval_completed_bool = False
    if "PREV_JSON_accepted" not in st.session_state:
        st.session_state.PREV_JSON_accepted = set()
    if "PREV_JSON_rejected" not in st.session_state:
        st.session_state.PREV_JSON_rejected = set()
    if "PREV_JSON_undecided" not in st.session_state:
        st.session_state.PREV_JSON_undecided = set()
    if "use_PREV_JSON_checkbox_result" not in st.session_state:
        st.session_state.use_PREV_JSON_checkbox_result = False
    if "show_eval_results_bool" not in st.session_state:
        st.session_state.show_eval_results_bool = False


def file_check4dbCreation(filenames: list[str]):
    file_tag = text_dict()["file_tag"]
    sdcheck = any(
        f.endswith(file_tag["PKL"]) and file_tag["SD"] in f for f in filenames
    )
    return sdcheck


def file_check4SessionDict(session: Path):
    file_tag = text_dict()["file_tag"]
    sdcheck = any(
        p.is_file() and p.suffix == file_tag["PKL"] and file_tag["SD"] in p.name
        for p in session.iterdir()
    )
    return sdcheck


def create_paths_db() -> tuple[int, int]:
    """Create and populate SQLite database with paths and sessions that contain segDicts

    Returns:
        tuple[int, int]: Total paths and sessions in the database.
    """
    total_paths, total_sessions = db_utils.init_dbCreation(
        DB_NAME=DB_NAME,
        file_check_func4DB=file_check4dbCreation,
        file_check_func4SessionDict=file_check4SessionDict,
    )
    return total_paths, total_sessions


def get_segDict(selected_session: str):
    def _get_AL_from_SDISX(segDict_fname: str):
        CTempnew, ASpatnew, accepted_labels = saveNloadUtils.load_segDict(
            filename=segDict_fname,
            C_all=True,
            A_all=True,
            accepted_labels=True,
            print_prev_bool=False,
        )
        return CTempnew, ASpatnew, accepted_labels

    ss2print = "/".join(selected_session.split("/")[-2:])
    st.subheader(ss2print)
    file_tag = text_dict()["file_tag"]

    os.chdir(selected_session)
    st.session_state.segDict_fname = findLatest([file_tag["SD"], file_tag["PKL"]])
    st.session_state.DSimage_fname = findLatest(image_utils.get_DSImage_filename())
    if not st.session_state.DSimage_fname:
        st.session_state.DSimage_fname = findLatest([file_tag["PNG"], "cells-map"])
    accepted_rejected_json = findLatest([ACC_REJ_JSON_FNAME])

    st.write(f"Loaded segDict: ***{st.session_state.segDict_fname}***")
    print(f"Loaded segDict: {st.session_state.segDict_fname}")

    if st.session_state.CTemp is None or st.session_state.ASpat is None:
        st.session_state.CTemp, st.session_state.ASpat = saveNloadUtils.load_segDict(
            st.session_state.segDict_fname, C=True, A=True, print_prev_bool=False
        )
        st.session_state.ASpat = st.session_state.ASpat.toarray()

    print(f"Ctemp shape: {st.session_state.CTemp.shape}")
    print(f"ASpat shape: {st.session_state.ASpat.shape}")

    if st.session_state.CNMF_accepted is None or st.session_state.CNMF_rejected is None:
        try:
            CNMF_accepted, CNMF_rejected = saveNloadUtils.load_segDict(
                st.session_state.segDict_fname,
                idx_components=True,
                idx_components_bad=True,
                print_prev_bool=False,
            )
            st.session_state.CNMF_CE_bool = True
            print("Loaded component evaluation results from CNMF")
        except Exception as e:
            print(f"Error loading CNMF_accepted, CNMF_rejected: {e}")
            print("Culprit: no CNMF Component Evaluation found in segDict")
            CNMF_accepted = None
            CNMF_rejected = None
            st.session_state.CNMF_CE_bool = False

        st.session_state.CNMF_accepted = CNMF_accepted
        st.session_state.CNMF_rejected = CNMF_rejected
    elif (
        st.session_state.CNMF_accepted is not None
        and st.session_state.CNMF_rejected is not None
    ):
        st.session_state.CNMF_CE_bool = True

    if st.session_state.CNMF_CE_bool:
        st.write(
            "|-- Loaded accepted/rejected components from CNMF Component Evaluation"
        )

    if st.session_state.loaded_AL_fromSDISX is None:
        try:
            _, _, _ = _get_AL_from_SDISX(st.session_state.segDict_fname)
            st.session_state.loaded_AL_fromSDISX = True
        except Exception as e:
            print(f"No accepted labels found in segDict resetting to default: {e}")
            st.session_state.loaded_AL_fromSDISX = False

    if st.session_state.loaded_AL_fromSDISX:
        st.write("|-- Loaded accepted labels found within segDict (from ISX output)")
        CTempnew, ASpatnew, accepted_labels = _get_AL_from_SDISX(
            st.session_state.segDict_fname
        )
        st.session_state.CTemp = CTempnew
        st.session_state.ASpat = ASpatnew.toarray()
        st.session_state.AL_SDISX_accepted = set(
            np.where(accepted_labels == "accepted")[0].astype(int)
        )
        st.session_state.AL_SDISX_rejected = set(
            np.where(accepted_labels == "rejected")[0].astype(int)
        )
        st.session_state.AL_SDISX_undecided = set(
            np.where(accepted_labels == "undecided")[0].astype(int)
        )

    if isinstance(st.session_state.ASpat, sparse.csr_matrix):
        st.session_state.ASpat = st.session_state.ASpat.toarray()

    st.session_state.allcmap_images = create_allcmap_images(
        st.session_state.ASpat, st.session_state.DSimage_fname
    )

    if (
        not st.session_state.undecided
        or selected_session != st.session_state.prev_session
    ):
        st.session_state.undecided = set(list(range(st.session_state.CTemp.shape[0])))

    st.write(f"|-- Number of components: {st.session_state.CTemp.shape[0]}")
    st.write(f"|-- Number of timepoints: {st.session_state.CTemp.shape[1]}")

    if accepted_rejected_json and not st.session_state.adjust_labels_bool:
        with open(accepted_rejected_json, "r") as f:
            accepted_rejected_dict = json.load(f)
        st.session_state.PREV_JSON_accepted = set(accepted_rejected_dict["accepted"])
        st.session_state.PREV_JSON_rejected = set(accepted_rejected_dict["rejected"])
        st.session_state.PREV_JSON_undecided = set(accepted_rejected_dict["undecided"])

    if accepted_rejected_json:
        st.write("|-- Loaded previously done accepted/rejected components from json")

    temp_set = st.session_state.undecided.copy()
    for und in st.session_state.undecided:
        if und in st.session_state.accepted:
            temp_set.remove(und)
        if und in st.session_state.rejected:
            temp_set.remove(und)
    st.session_state.undecided = temp_set

    imported_accepted_mainBool = (
        st.session_state.CNMF_CE_bool
        or st.session_state.loaded_AL_fromSDISX
        or st.session_state.NNCE_accepted
        or st.session_state.PREV_JSON_accepted
    )

    options4radio = ["None"]
    if st.session_state.CNMF_CE_bool:
        options4radio.append("CNMF Component Evaluation")
    if st.session_state.AL_SDISX_accepted:
        options4radio.append("Accepted Labels from segDict (ISX)")
    if st.session_state.NNCE_accepted:
        options4radio.append("Neural Network Component Evaluation")
    if st.session_state.PREV_JSON_accepted:
        options4radio.append("Previously Done Accepted/Rejected Components")

    col1, col2 = st.columns(2)
    with col1:
        selected_option = st.radio(
            "Select Component Evaluation Method",
            options=options4radio,
            disabled=not imported_accepted_mainBool,
            key="component_eval_method",
            help="Select which accepted/rejected components to use. CNMF for the CNMF Component Evaluation, ISX for the Accepted Labels from segDict that was done by hand (ISX), and NNCE for the Neural Network Component Evaluation.",
        )

        if (
            selected_option == "CNMF Component Evaluation"
            and st.session_state.CNMF_CE_bool
        ):
            st.session_state.use_CNMF_CE_checkbox_result = True
            st.session_state.use_AL_SDISX_checkbox_result = False
            st.session_state.use_NNCE_checkbox_result = False
            st.session_state.use_PREV_JSON_checkbox_result = False
        elif (
            selected_option == "Accepted Labels from segDict (ISX)"
            and st.session_state.loaded_AL_fromSDISX
        ):
            st.session_state.use_AL_SDISX_checkbox_result = True
            st.session_state.use_CNMF_CE_checkbox_result = False
            st.session_state.use_NNCE_checkbox_result = False
            st.session_state.use_PREV_JSON_checkbox_result = False
        elif (
            selected_option == "Neural Network Component Evaluation"
            and st.session_state.NNCE_accepted
        ):
            st.session_state.use_NNCE_checkbox_result = True
            st.session_state.use_CNMF_CE_checkbox_result = False
            st.session_state.use_AL_SDISX_checkbox_result = False
            st.session_state.use_PREV_JSON_checkbox_result = False
        elif (
            selected_option == "Previously Done Accepted/Rejected Components"
            and st.session_state.PREV_JSON_accepted
        ):
            st.session_state.use_NNCE_checkbox_result = False
            st.session_state.use_CNMF_CE_checkbox_result = False
            st.session_state.use_AL_SDISX_checkbox_result = False
            st.session_state.use_PREV_JSON_checkbox_result = True
        else:  # "None" or disabled options
            st.session_state.use_CNMF_CE_checkbox_result = False
            st.session_state.use_AL_SDISX_checkbox_result = False
            st.session_state.use_NNCE_checkbox_result = False
            st.session_state.use_PREV_JSON_checkbox_result = False
    with col2:
        adjust_labels_checkbox = st.checkbox(
            "Adjust CNMF, ISX, or NNCE accepted/rejected components",
            disabled=not imported_accepted_mainBool,
            value=st.session_state.adjust_labels_bool,
            key="adjust_labels_checkbox",
            help="If you have already done accepted/rejected components but want to adjust them, check this box. Otherwise, if left unchecked, any changes made will not be registered.",
        )
        st.session_state.adjust_labels_bool = adjust_labels_checkbox

    if st.session_state.AL_SDISX_accepted and st.session_state.NNCE_accepted:
        st.write(
            f"- Accepted by ISX, but not by NNCE: {sorted(list(st.session_state.AL_SDISX_accepted - st.session_state.NNCE_accepted))}"
        )
        # st.write(
        #     f"- Accepted by NNCE, but not by ISX: {sorted(list(st.session_state.NNCE_accepted - st.session_state.AL_SDISX_accepted))}"
        # )
    if st.session_state.AL_SDISX_rejected and st.session_state.NNCE_rejected:
        st.write(
            f"- Rejected by ISX, but not by NNCE: {sorted(list(st.session_state.AL_SDISX_rejected - st.session_state.NNCE_rejected))}"
        )
        # st.write(
        #     f"- Rejected by NNCE, but not by ISX: {sorted(list(st.session_state.NNCE_rejected - st.session_state.AL_SDISX_rejected))}"
        # )

    if (
        st.session_state.use_CNMF_CE_checkbox_result
        and not st.session_state.adjust_labels_bool
    ):
        st.session_state.accepted = set(st.session_state.CNMF_accepted.astype(int))
        st.session_state.rejected = set(st.session_state.CNMF_rejected.astype(int))
        st.session_state.undecided = set()
    if (
        st.session_state.use_AL_SDISX_checkbox_result
        and not st.session_state.adjust_labels_bool
    ):
        st.session_state.accepted = st.session_state.AL_SDISX_accepted
        st.session_state.rejected = st.session_state.AL_SDISX_rejected
        st.session_state.undecided = st.session_state.AL_SDISX_undecided
    if (
        st.session_state.use_NNCE_checkbox_result
        and not st.session_state.adjust_labels_bool
    ):
        st.session_state.accepted = set(st.session_state.NNCE_accepted)
        st.session_state.rejected = set(st.session_state.NNCE_rejected)
        st.session_state.undecided = set()
    if (
        st.session_state.use_PREV_JSON_checkbox_result
        and not st.session_state.adjust_labels_bool
    ):
        st.session_state.accepted = st.session_state.PREV_JSON_accepted
        st.session_state.rejected = st.session_state.PREV_JSON_rejected
        st.session_state.undecided = st.session_state.PREV_JSON_undecided
    if (
        not st.session_state.use_CNMF_CE_checkbox_result
        and not st.session_state.use_AL_SDISX_checkbox_result
        and not st.session_state.use_NNCE_checkbox_result
        and not st.session_state.use_PREV_JSON_checkbox_result
    ):
        st.session_state.accepted = set()
        st.session_state.rejected = set()
        st.session_state.undecided = set(list(range(st.session_state.CTemp.shape[0])))

    print(
        f"Adjust labels checkbox clicked -- State: {st.session_state.adjust_labels_bool}"
    )
    st.divider()


def NNsectionHandler(selected_session: str, model_name: str = ""):
    def create_model_performance_string(metrics: dict, model_name: str = "") -> str:
        stats2print = ""
        stats2print += (
            f"Model Name: {model_name}\n" if model_name else "New Model (unsaved)\n"
        )
        stats2print += " | "
        numSessions = len(metrics["metadata"]["listExpNames"])
        for key, (mean, stdev) in metrics["mean"].items():
            stats2print += f"{key.upper()}: {mean:.3f} ± {stdev:.3f} | "
        st.markdown(
            f"**Model Performance (numSessions: {numSessions})**",
            help=stats2print,
        )

    with st.expander("Neural Network Settings"):
        cols = st.columns([6, 1, 1])
        with cols[0]:
            model2print = (
                st.session_state.model_name2load
                if st.session_state.model_name2load
                else "No model loaded"
            )
            st.write(f"Current Model: {model2print}")
        with cols[-1]:
            if st.button(
                "Reset",
                disabled=not st.session_state.model_runner,
                help="Reset NN settings to start fresh.",
            ):
                _refresh_NN_settings()
                st.rerun()

        col1, col2, col3 = st.columns((1, 2, 1))
        with col1:
            if st.button(
                "Add session to training set",
                help="Add the current session to the training set. All the cells need to be accepted/rejected before adding to the training set.",
            ):
                if selected_session not in st.session_state.listExpNames:
                    current_labels = np.full(st.session_state.CTemp.shape[0], np.nan)
                    for cell in sorted(st.session_state.accepted):
                        current_labels[cell] = 1
                    for cell in sorted(st.session_state.rejected):
                        current_labels[cell] = 0

                    if np.any(np.isnan(current_labels)):
                        st.warning(
                            "Undecided cells found. Please accept/reject all cells before adding to training set."
                        )
                    else:
                        st.session_state.listExpNames.append(selected_session)
                        st.session_state.ASPatList.append(st.session_state.ASpat)
                        st.session_state.CTempList.append(st.session_state.CTemp)
                        st.session_state.labelList.append(current_labels)
                        st.session_state.dim4spatialList.append(
                            st.session_state.dim4spatial
                        )
                elif st.session_state.undecided:
                    st.error(
                        "Undecided cells found. Please accept/reject all cells before adding to training set."
                    )
                elif selected_session in st.session_state.listExpNames:
                    st.error("Session already in training set")
        with col2:
            if st.session_state.listExpNames and st.session_state.CTempList:
                total_cells = sum(
                    st.session_state.CTempList[i].shape[0]
                    for i in range(len(st.session_state.listExpNames))
                )
                total_accepted = int(
                    sum(
                        np.sum(st.session_state.labelList[i])
                        for i in range(len(st.session_state.listExpNames))
                    )
                )
                total_rejected = total_cells - total_accepted
                st.write(f"Training set: {len(st.session_state.listExpNames)} sessions")
                st.write(
                    f"Total cells: {total_cells} | Accepted: {total_accepted} | Rejected: {total_rejected}"
                )
            else:
                st.write("No sessions in training set")
        with col3:
            if st.button("Delete last session from training set"):
                for ls in [
                    "listExpNames",
                    "ASPatList",
                    "CTempList",
                    "labelList",
                    "dim4spatialList",
                ]:
                    if ls in st.session_state:
                        if len(st.session_state[ls]) > 1:
                            st.session_state[ls] = st.session_state[ls][:-1]
                        else:
                            st.session_state[ls] = []
                    st.rerun()

        col1, col2, col3 = st.columns((1, 2, 1))
        with col1:
            if st.button(
                "Train Model",
                help="Train a model on the training set. This will create a new model and overwrite any existing model.",
                disabled=not st.session_state.listExpNames,
            ):
                st.session_state.run_train_model_bool = True

        with col2:
            if (
                st.session_state.run_train_model_bool
                and not st.session_state.model_name2load
            ):
                st.session_state.nn_success_bool = False
                st.session_state.model_runner = ""
                st.session_state.feature_extractor = None
                st.session_state.features_vector = None
                st.session_state.labels_vector = None
                st.session_state.metadata = None

                with st.spinner("Creating features vector..."):
                    feature_extractor = FeatureExtractor4segDict(
                        CTempList=st.session_state.CTempList,
                        ASpatList=st.session_state.ASPatList,
                        dim4spatialList=st.session_state.dim4spatialList,
                        listExpNames=st.session_state.listExpNames,
                        labelList=st.session_state.labelList,
                    )

                    features_vector, labels_vector, metadata = (
                        feature_extractor.create_Vectors(train=True)
                    )
                with st.spinner("Initializing model..."):
                    model_runner = GeneralModelRunner(
                        model=NNModel4BinaryClassification(
                            feature_size=int(len(feature_extractor.features_order))
                        ),
                        features_order=feature_extractor.features_order,
                    )
                with st.spinner("Cross-validating model..."):
                    model_runner.cross_validate(
                        features_vector=features_vector,
                        labels_vector=labels_vector,
                        metadata=metadata,
                    )
                    st.session_state.nn_success_bool = True
                    st.session_state.model_runner = model_runner
                    st.session_state.feature_extractor = feature_extractor
                    st.session_state.features_vector = features_vector
                    st.session_state.labels_vector = labels_vector
                    st.session_state.metadata = metadata
                    st.session_state.model_name2load = "New Model (unsaved)"
                    st.rerun()
            if st.session_state.nn_success_bool:
                create_model_performance_string(
                    st.session_state.model_runner.metrics,
                    st.session_state.model_name2load,
                )

        col1, col2, col3 = st.columns((1, 2, 1))
        with col1:
            if st.button(
                "Save Model",
                disabled=not st.session_state.nn_success_bool,
                help="Save model to file. Whatever is inputted will have number of sessions, model type, file ending, and date automatically appended.",
            ):
                st.session_state.save_model_bool = not st.session_state.save_model_bool

            if st.session_state.save_model_bool:
                st.session_state.model_name2save = st.text_input(
                    "Enter model name", value=st.session_state.model_name2save
                )
            if st.session_state.model_name2save:
                fname = st.session_state.model_runner.save_model(
                    fname=f"{st.session_state.model_name2save}",
                )
                print(f"Model saved as {fname}")
                st.success(f"Model saved as {fname}")
                st.session_state.model_name2save = ""
                st.session_state.model_name2load = fname.split("/")[-1].split(".")[0]
                st.session_state.save_model_bool = False
                st.rerun()
        with col2:
            CE_DB = dbHandler4CompEvaluator(DB_NAME4NN)
            num_models = CE_DB.get_num_of_models()
            model_check = num_models > 0
            if st.button(
                "Load Model",
                disabled=not model_check,
                help=f"Load a model. Disabled if no models are available. Loading models from: {paths.get_path2NNmodels()}. Current number of models: {num_models}",
            ):
                st.session_state.load_model_bool = not st.session_state.load_model_bool
        with col3:
            if st.session_state.model_runner:
                if st.button(
                    "Evaluate components",
                    help="Use current model to evaluate components and determine accepted/rejected components.",
                ):
                    # Get predictions for current session
                    st.session_state.NNCE_predictions = (
                        st.session_state.model_runner.predict_new_data(
                            CTemp=st.session_state.CTemp,
                            ASpat=st.session_state.ASpat,
                            dim4spatial=st.session_state.dim4spatial,
                        )
                    )

                    # Update accepted/rejected based on predictions
                    st.session_state.NNCE_accepted = set(
                        np.where(st.session_state.NNCE_predictions["predictions"] == 1)[
                            0
                        ]
                    )
                    st.session_state.NNCE_rejected = set(
                        np.where(st.session_state.NNCE_predictions["predictions"] == 0)[
                            0
                        ]
                    )

                    st.session_state.eval_completed_bool = True
        if st.session_state.eval_completed_bool:
            was_in_training = selected_session in st.session_state.listExpNames
            threshold_arbitrary = 0.5

            binary_predictions = (
                st.session_state.NNCE_predictions["predictions"].numpy()
                > threshold_arbitrary
            ).astype(int)

            # Show results
            st.success(
                f"Evaluation complete! Found {len(st.session_state.NNCE_accepted)} accepted and {len(st.session_state.NNCE_rejected)} rejected components."
            )
            if st.button("Show Evaluation Results"):
                st.session_state.show_eval_results_bool = (
                    not st.session_state.show_eval_results_bool
                )
            if st.session_state.show_eval_results_bool:
                st.markdown("#### Evaluation Results")
                st.write(f"- Total components evaluated: {len(binary_predictions)}")
                bools = sum(
                    [
                        bool(st.session_state.AL_SDISX_accepted),
                        bool(st.session_state.NNCE_accepted),
                    ],
                )
                cols = st.columns(bools)

                if st.session_state.NNCE_accepted:
                    cols[0].write(
                        f"- Accepted (NN): {len(st.session_state.NNCE_accepted)}"
                    )
                    cols[0].write(
                        f"- Rejected (NN): {len(st.session_state.NNCE_rejected)}"
                    )
                if st.session_state.AL_SDISX_accepted:
                    AL_labels = np.zeros(len(binary_predictions))
                    AL_labels[sorted(list(st.session_state.AL_SDISX_accepted))] = 1

                    st.session_state.optimal_threshold = (
                        st.session_state.model_runner.find_optimal_threshold(
                            predictions=st.session_state.NNCE_predictions[
                                "predictions"
                            ].numpy(),
                            labels2compare2=AL_labels,
                        )
                    )
                    binary_predictions = (
                        st.session_state.NNCE_predictions["predictions"].numpy()
                        > st.session_state.optimal_threshold
                    ).astype(int)
                    # Get confusion matrix values
                    tn, fp, fn, tp = confusion_matrix(
                        AL_labels, binary_predictions
                    ).ravel()

                    # Calculate metrics
                    total = len(binary_predictions)
                    accuracy = ((tp + tn) / total) * 100
                    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
                    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
                    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
                    f1 = (
                        2 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )
                    cols[1].write(
                        f"Optimal Threshold: {st.session_state.optimal_threshold:.2f}"
                    )
                    cols[1].write(
                        f"- Accepted (ISX): {len(st.session_state.AL_SDISX_accepted)}"
                    )
                    cols[1].write(
                        f"- Rejected (ISX): {len(st.session_state.AL_SDISX_rejected)}"
                    )
                    cols[0].write("In comparison to ISX:")
                    cols[0].write(f"- Accuracy: {accuracy:.2f}%")
                    cols[0].write(f"- Precision: {precision:.2f}%")
                    cols[0].write(f"- Recall: {recall:.2f}%")
                    cols[0].write(f"- Specificity: {specificity:.2f}%")
                    cols[0].write(f"- F1 Score: {f1:.2f}")

                    fig, ax = fig_tools.create_plt_subplots(figsize=(6, 6))
                    cm_display = np.array([[tp, fp], [fn, tn]])
                    ax.imshow(cm_display, cmap="Blues")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(["Accepted", "Rejected"])
                    ax.set_yticklabels(["Accepted", "Rejected"])
                    ax.text(-0.3, -0.1, f"TP: {tp}", ha="center")
                    ax.text(1.3, -0.1, f"FP: {fp}", ha="center")
                    ax.text(-0.3, 0.9, f"FN: {fn}", ha="center")
                    ax.text(1.3, 0.9, f"TN: {tn}", ha="center")
                    cols[1].write("Confusion Matrix:")
                    cols[1].pyplot(fig)
                if was_in_training:
                    st.write("- NOTE: This session was in the training set.")

                cols = st.columns(2)
                if st.session_state.optimal_threshold:
                    threshold2use = st.session_state.optimal_threshold
                else:
                    threshold2use = threshold_arbitrary

                # Show distribution of confidence scores
                scores = st.session_state.NNCE_predictions["scores"]
                fig, ax = fig_tools.create_plt_subplots(figsize=(8, 4))
                ax.hist(scores, bins=20, edgecolor="black")
                ax.axvline(
                    x=threshold2use,
                    color="r",
                    linestyle="--",
                    label="Threshold",
                )
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Model Confidence Scores")
                ax.legend()
                cols[0].pyplot(fig)

                # Show confidence scores for each component
                st.markdown("#### Component-wise Scores")
                score_df = pd.DataFrame(
                    {
                        "Component": range(len(scores)),
                        "Confidence": scores,
                        "Prediction": [
                            "Accepted" if p == 1 else "Rejected"
                            for p in binary_predictions
                        ],
                    }
                )
                cols[1].dataframe(
                    score_df.style.background_gradient(subset=["Confidence"])
                )

        if st.session_state.load_model_bool:
            st.session_state.eval_completed_bool = False
            metrics = [
                ("Accuracy", "accuracy", "accuracy_std"),
                ("Precision", "precision", "precision_std"),
                ("Recall", "recall", "recall_std"),
                ("F1 Score", "f1", "f1_std"),
                ("AUC", "auc", "auc_std"),
            ]

            selected_model_key = st.selectbox(
                "Select model",
                options=list(CE_DB.get_model_dict().keys()),
                key="select_model",
                help=f"Select a model to load. Models are stored in: {paths.get_path2NNmodels()}",
                format_func=lambda x: x.split("/")[-1],
            )
            st.session_state.selected_model_dict = CE_DB.select_model_from_db(
                selected_model_key
            )
            st.session_state.listExpNames = st.session_state.selected_model_dict[
                "listExpNames"
            ]

            model_cols = st.columns([1] * len(metrics))

            for i, col in enumerate(model_cols):
                col.write(metrics[i][0])
                col.write(
                    f"{st.session_state.selected_model_dict[metrics[i][1]]:.3f} ± {st.session_state.selected_model_dict[metrics[i][2]]:.3f}"
                )

        if st.session_state.selected_model_dict:
            if st.button("Confirm Choice"):
                model2load = st.session_state.selected_model_dict["model_path"]
                # only load model, don't need metadata
                st.session_state.model_runner, _ = GeneralModelRunner.load_model(
                    model2load
                )
                st.session_state.model_name2load = model2load.split("/")[-1].split(".")[
                    0
                ]
                st.session_state.nn_success_bool = True
                st.session_state.load_model_bool = False
                st.session_state.selected_model_dict = {}
                st.rerun()


def PlotSectionHandler():
    def _prime_list2print(list2print: list[int] | None) -> str:
        if list2print == "None":
            return "None"
        # if list2print is None:
        #     return "None"
        list2print = sorted(list2print)
        if len(list2print) > 100:
            list2print = f"{list2print[0:40]}...{list2print[-20:]}"
        return list2print

    acceptString = f"{EMOJIS['check']} Accept"
    rejectString = f"{EMOJIS['x']} Reject"
    undecidedString = f"{EMOJIS['question']} Undecided"
    col1, col2, col3, col4 = st.columns((7, 1, 1, 1))
    with col1:
        num_components = st.session_state.ASpat.shape[-1]
        selected_index_input = st.number_input(
            f"Select ROI to highlight (Range: 0-{num_components - 1})",
            min_value=0,
            max_value=num_components - 1,
            value=st.session_state.selected_index,
            key="input_index",
        )
        st.session_state.selected_index = selected_index_input
        print(f"Selected ROI: {selected_index_input}")
        print(f"Prev index: {st.session_state.prev_index}")
        print()

        if selected_index_input != st.session_state.prev_index:
            st.session_state.prev_index = st.session_state.selected_index

    with col2:
        if st.button(acceptString):
            print(
                f"Accept button clicked -- Adding to accepted: {st.session_state.selected_index}"
            )
            if st.session_state.selected_index in st.session_state.rejected:
                st.session_state.rejected.remove(st.session_state.selected_index)
            if st.session_state.selected_index in st.session_state.undecided:
                st.session_state.undecided.remove(st.session_state.selected_index)
            st.session_state.accepted.add(st.session_state.selected_index)
            st.session_state.accepted = set(sorted(st.session_state.accepted))
    with col3:
        if st.button(rejectString):
            print(
                f"Reject button clicked -- Adding to rejected: {st.session_state.selected_index}"
            )
            if st.session_state.selected_index in st.session_state.accepted:
                print(
                    f"-- Found in accepted -- Removing from accepted: {st.session_state.selected_index}"
                )
                st.session_state.accepted.remove(st.session_state.selected_index)
            if st.session_state.selected_index in st.session_state.undecided:
                print(
                    f"-- Found in undecided -- Removing from undecided: {st.session_state.selected_index}"
                )
                st.session_state.undecided.remove(st.session_state.selected_index)
            st.session_state.rejected.add(st.session_state.selected_index)
            st.session_state.rejected = set(sorted(st.session_state.rejected))
    with col4:
        if st.button(undecidedString):
            print(
                f"Undecided button clicked -- Adding to undecided: {st.session_state.selected_index}"
            )
            if st.session_state.selected_index in st.session_state.accepted:
                print(
                    f"-- Found in accepted -- Removing from accepted: {st.session_state.selected_index}"
                )
                st.session_state.accepted.remove(st.session_state.selected_index)
            if st.session_state.selected_index in st.session_state.rejected:
                print(
                    f"-- Found in rejected -- Removing from rejected: {st.session_state.selected_index}"
                )
                st.session_state.rejected.remove(st.session_state.selected_index)
            st.session_state.undecided.add(st.session_state.selected_index)
            st.session_state.undecided = set(sorted(st.session_state.undecided))

    st.sidebar.subheader("Component Evaluation")

    acc2print = list(st.session_state.accepted) if st.session_state.accepted else "None"
    rej2print = list(st.session_state.rejected) if st.session_state.rejected else "None"
    und2print = (
        list(st.session_state.undecided) if st.session_state.undecided else "None"
    )

    st.sidebar.write(f"{acceptString}ed Cells: {_prime_list2print(acc2print)}")
    st.sidebar.write(f"{rejectString}ed Cells: {_prime_list2print(rej2print)}")
    st.sidebar.write(f"{undecidedString} Cells: {_prime_list2print(und2print)}")

    col1, col2 = st.columns(2)
    with col1:
        print("Plotting ASpat with DSimage")
        print(HASH)
        plot_ASpat_wDSimage()
        print(HASH)
    with col2:
        print("Plotting CTemp")
        print(HASH)
        plot_CTemp()
        print(HASH)


def plot_ASpat_wDSimage():
    st.subheader("Spatial Profile")

    cmaps2select = ["gray", "turbo", "inferno", "viridis"]
    cmap = st.selectbox("Select color map", cmaps2select)

    CMAP2USE = CMAPS[1]
    C4FONT = color_dict()["red"]
    CMAP4ALL = CMAPS[2]
    CMAP4ACCEPTED = CMAPS[3]
    if cmap in ["turbo", "viridis"]:
        CMAP2USE = CMAPS[0]
        CMAP4ALL = CMAPS[4]
    elif cmap in ["inferno"]:
        C4FONT = color_dict()["white"]

    if st.session_state.DSimage_fname is not None:
        DSimage = image_utils.read_image(st.session_state.DSimage_fname)
        print(f"DSimage fname: {st.session_state.DSimage_fname}")
        print(f"DSimage shape: {DSimage.shape}")
        if "cells-map" in st.session_state.DSimage_fname:
            vmax4img = np.max(DSimage) / 4
            figsize = (6, 3)
            max_divisor4allcmap = 10
        else:
            vmax4img = np.max(DSimage)
            figsize = (6, 6)
            max_divisor4allcmap = 4
        print(f"vmax4img: {vmax4img}")
        print(f"figsize: {figsize}")
        print(f"max_divisor4allcmap: {max_divisor4allcmap}")

    fig, ax = fig_tools.create_plt_subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(DSimage, aspect="auto", cmap=cmap, vmax=vmax4img)

    cno_bool = False
    allcmap_bool = False
    accepted_bool = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("Show Cell Number Overlay"):
            cno_bool = not cno_bool
    with col2:
        if st.checkbox("Show All Components"):
            allcmap_bool = not allcmap_bool
    with col3:
        if st.checkbox(
            f"Show only accepted (n = {len(st.session_state.accepted)})",
            disabled=not st.session_state.accepted,
        ):
            accepted_bool = not accepted_bool

    for name, value in zip(
        [
            "Cell number overlay",
            "Show all components",
            "Show only accepted",
        ],
        [cno_bool, allcmap_bool, accepted_bool],
    ):
        print(f"{name} -- State: {value}")
    # plot selected ROI
    selected_spatial_data = st.session_state.ASpat[:, st.session_state.selected_index]
    nDim = int(np.sqrt(len(selected_spatial_data)))
    if len(selected_spatial_data) % nDim == 0:
        nDim = (nDim, nDim)
        selected_spatial_data = selected_spatial_data.reshape(nDim)
        square_bool = True
    else:
        nDim = DSimage.shape
        # st.write(f"Reshaping to {nDim}")
        selected_spatial_data = np.transpose(selected_spatial_data.reshape(nDim))
        square_bool = False

    st.session_state.dim4spatial = nDim
    ax.imshow(selected_spatial_data.T, cmap=CMAP2USE, alpha=0.95, aspect="auto")

    if cno_bool:
        print("Labeling cell numbers...")
        for cell in range(st.session_state.ASpat.shape[-1]):
            data = st.session_state.ASpat[:, cell]
            data_2d = data.reshape(nDim)
            data_2d = data_2d.T if not square_bool else data_2d
            fig_tools.label_cellNum_overDSImage(
                axis=ax,
                data=data_2d,
                cell_str=f"Cell_{cell}",
                color=C4FONT,
                fontsize=6,
            )

    if allcmap_bool:
        print("Plotting all components...")
        ax.imshow(
            st.session_state.allcmap_images,
            cmap=CMAP4ALL,
            alpha=0.5,
            vmax=np.max(st.session_state.allcmap_images) / max_divisor4allcmap,
            aspect="auto",
        )

    accepted_cmap_images = create_accepted_cmap_image()
    if accepted_bool:
        print("Plotting accepted components...")
        print(f"Accepted cmap images shape: {accepted_cmap_images.shape}")
        ax.imshow(
            accepted_cmap_images,
            cmap=CMAP4ACCEPTED,
            alpha=0.5,
            vmax=np.max(accepted_cmap_images) / max_divisor4allcmap,
            aspect="auto",
        )

    st.pyplot(fig)


@st.cache_data
def create_allcmap_images(ASpat: np.ndarray, DSimage_fname: str | None):
    """Create and cache images for all components."""
    allcmap_images = []
    for cell in range(ASpat.shape[-1]):
        data = ASpat[:, cell]
        nDim = int(np.sqrt(len(data)))  # Assuming square shape
        if len(data) % nDim == 0:
            nDim = (nDim, nDim)
            data_2d = data.reshape(nDim)
        else:
            nDim = image_utils.read_image(DSimage_fname).shape
            data_2d = data.reshape(nDim).T
        allcmap_images.append(data_2d.T)
    allcmap_images = np.sum(allcmap_images, axis=0)
    return allcmap_images


@st.cache_data
def create_accepted_cmap_image():
    accepted_cmap_images = []
    for cell in list(st.session_state.accepted):
        data = st.session_state.ASpat[:, cell]
        nDim = int(np.sqrt(len(data)))
        if len(data) % nDim == 0:
            nDim = (nDim, nDim)
            data_2d = data.reshape(nDim)
        else:
            nDim = image_utils.read_image(st.session_state.DSimage_fname).shape
            data_2d = data.reshape(nDim).T
        accepted_cmap_images.append(data_2d.T)
    accepted_cmap_images = np.sum(accepted_cmap_images, axis=0)
    return accepted_cmap_images


def _normalize_data(data: np.ndarray) -> np.ndarray:
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def plot_CTemp():
    def set_pk_params():
        st.sidebar.divider()
        st.sidebar.subheader("Peak Detection Parameters:")
        if st.session_state.peak_algorithm == "scipy":
            st.session_state.height = st.sidebar.number_input(
                "Height",
                value=st.session_state.height,
                key="height_slider",
                help="Height of the peak",
            )
            st.session_state.distance = st.sidebar.number_input(
                "Distance",
                value=st.session_state.distance,
                key="distance_slider",
                help="Distance between peaks",
            )
            st.session_state.prominence = st.sidebar.number_input(
                "Prominence",
                value=st.session_state.prominence,
                key="prominence_slider",
                help="Prominence of the peak",
            )
            names = ["height", "distance", "prominence"]
            values = [
                st.session_state.height,
                st.session_state.distance,
                st.session_state.prominence,
            ]
        elif st.session_state.peak_algorithm == "iterative_diffs":
            st.session_state.frameWindow = st.sidebar.number_input(
                "Frame Window",
                value=st.session_state.frameWindow,
                key="frameWindow_slider",
                help="Frame window for peak detection",
            )
            st.session_state.sdThresh = st.sidebar.number_input(
                "SD Threshold",
                value=st.session_state.sdThresh,
                key="sdThresh_slider",
                help="How many standard deviations above the mean to consider a peak",
            )
            st.session_state.timeout = st.sidebar.number_input(
                "Timeout",
                value=st.session_state.timeout,
                key="timeout_slider",
                help="Number of frames to consider before and after a peak",
            )
            names = ["frameWindow", "sdThresh", "timeout"]
            values = [
                st.session_state.frameWindow,
                st.session_state.sdThresh,
                st.session_state.timeout,
            ]
        print_namesNvalues(names, values)
        for name, value in zip(names, values):
            st.write(f"**{name.upper()}**: {value}")

    markdown_ca2 = "Ca$^{2+}$"
    status = (
        "Accepted"
        if st.session_state.selected_index in st.session_state.accepted
        else "Rejected"
        if st.session_state.selected_index in st.session_state.rejected
        else "Undecided"
    )
    st.subheader(
        f"{markdown_ca2} Signal for Cell {st.session_state.selected_index:03d} ({status})"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        help_msg4algo = (
            "Toggle between two peak detection algorithms:\n"
            "- **Scipy**: Utilizes the `find_peaks` function from the `scipy.signal` module. This method detects peaks by identifying local maxima in the calcium transient signal. The following parameters can be adjusted:\n"
            "  - **Height**: Specifies the minimum height of a peak relative to the baseline. Peaks must exceed this height to be considered valid, helping to filter out noise and small fluctuations.\n"
            "  - **Distance**: Defines the minimum number of samples between consecutive peaks. This prevents the detection of multiple peaks that are too close together, ensuring that only distinct peaks are identified.\n"
            "  - **Prominence**: Measures how much a peak stands out from its surrounding baseline. A peak's prominence is determined by the height of the peak relative to the lowest contour line that can be drawn between it and its nearest higher neighbor. This helps to ensure that only significant peaks are detected, filtering out minor fluctuations.\n"
            "- **Iterative Diffs**: This algorithm processes the calcium transient data by first applying a running mean to smooth the signal, then calculating the difference between consecutive frames. It identifies peaks by looking for local minima in the negative of the differentiated signal, which helps to highlight significant upward changes. The algorithm iteratively refines the detected peaks by re-baselining the data, allowing it to adapt to varying baseline levels and noise. The following parameters can be adjusted:\n"
            "  - **Frame Window**: Defines the number of frames to consider for the running mean.\n"
            "  - **SD Threshold**: Specifies the number of standard deviations above the mean to consider a peak.\n"
            "  - **Timeout**: Defines the timeout duration for peak detection in seconds. It determines how long to wait after detecting a peak before considering the next potential peak.\n"
            "Choose the algorithm that best fits your data characteristics for optimal peak detection."
        )
        if st.button("Switch Peak Algorithm"):
            if st.session_state.peak_algorithm == "scipy":
                st.session_state.peak_algorithm = "iterative_diffs"
            elif st.session_state.peak_algorithm == "iterative_diffs":
                st.session_state.peak_algorithm = "scipy"
            print(f"Peak algorithm set to: {st.session_state.peak_algorithm}")
        st.markdown(f"Current: {st.session_state.peak_algorithm}", help=help_msg4algo)
    with col2:
        # pass
        adjust2baseline_checkbox = st.checkbox(
            "Adjust to Baseline",
            value=st.session_state.adj2bsl_bool,
            key="adjust2baseline_checkbox",
            help="Toggle wheter to adjust the signal to the baseline. If adjusting, can also toggle between subtracting the baseline and dividing by the baseline, the size of the window used to find the baseline in a rolling manner, and the quantile used to determine the baseline.",
        )
        st.session_state.adj2bsl_bool = adjust2baseline_checkbox
        print(f"Adjust to baseline checkbox set to: {st.session_state.adj2bsl_bool}")
        if st.session_state.adj2bsl_bool:
            st.sidebar.divider()
            st.sidebar.subheader("Baseline Scaling Parameters:")
            st.session_state.scale_window = st.sidebar.number_input(
                "Scale Window",
                value=st.session_state.scale_window,
                key="scale_window_slider",
                help="Window size for to find baseline per bin. Units are in frames.",
            )
            st.session_state.quantile = st.sidebar.number_input(
                "Quantile",
                value=st.session_state.quantile,
                key="quantile_slider",
                help="Quantile to determine threshold for what is considered baseline.",
            )
            st.session_state.shiftMin2Zero = st.sidebar.checkbox(
                "Shift Min to Zero",
                value=st.session_state.shiftMin2Zero,
                key="shiftMin2Zero_checkbox",
                help="Shift the minimum value to zero. Helpful if dividing by baseline to avoid inversion of the signal due to negative values",
            )
            st.session_state.log4bsl = st.sidebar.checkbox(
                "Subtract Baseline (ON) | Divide by Baseline (OFF)",
                value=st.session_state.log4bsl,
                key="log4bsl_checkbox",
                help="If the checkbox is ON, the baseline will be subtracted from the data. If the checkbox is OFF, the data will be divided by the baseline",
            )
            print_namesNvalues(
                [
                    "Scale Window",
                    "Quantile",
                    "Shift Min to Zero",
                    "Subtract Baseline (ON) | Divide by Baseline (OFF)",
                ],
                [
                    st.session_state.scale_window,
                    st.session_state.quantile,
                    st.session_state.shiftMin2Zero,
                    st.session_state.log4bsl,
                ],
            )

    with col3:
        apply_smoothing_checkbox = st.checkbox(
            "Apply Smoothing",
            value=st.session_state.smooth_bool,
            key="apply_smoothing_checkbox",
            help="Toggle whether to apply smoothing via a [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) to the signal. If smoothing is applied, can also toggle whether to use log transform, the size of the window used for smoothing, and the order of the polynomial used for smoothing.",
        )
        st.session_state.smooth_bool = apply_smoothing_checkbox
        print(f"Apply smoothing set to: {st.session_state.smooth_bool}")
        if st.session_state.smooth_bool:
            st.sidebar.divider()
            st.sidebar.subheader("Smoothing Parameters:")
            st.session_state.window_size = st.sidebar.number_input(
                "Smoothing Window Size",
                value=st.session_state.window_size,
                key="window_size_slider",
                help="Window size for smoothing. Units are in frames.",
            )
            st.session_state.smoothing_order = st.sidebar.number_input(
                "Smoothing Order",
                value=st.session_state.smoothing_order,
                key="smoothing_order_slider",
                help="Order of the polynomial used to fit the smoothing filter.",
            )
            st.session_state.log4sg = st.sidebar.checkbox(
                "Log Transform",
                value=st.session_state.log4sg,
                key="log4sg_checkbox",
                help="Use natural log of signal as input for smoothing",
            )
            print_namesNvalues(
                ["Smoothing Window Size", "Smoothing Order", "Log Transform"],
                [
                    st.session_state.window_size,
                    st.session_state.smoothing_order,
                    st.session_state.log4sg,
                ],
            )

    PLOTDICT = {
        "color": CMAPS[1](0.5),
        "linewidth": 0.5,
        "markersize": 2,
        "fs": 8,
        "ymax": 1.1,
        "ymin": -0.4,
        "peak_color": color_dict()["violet"],
        "pre_peak": 50,
        "post_peak": 200,
        "figsize": (10, 3),
    }
    # selection_made = st.session_state.selected_index != st.session_state.prev_index

    st.session_state.temporal_data = st.session_state.CTemp[
        st.session_state.selected_index, :
    ]

    st.session_state.adj2bsl_data = pks_utils.normalizeByBaseline(
        st.session_state.temporal_data,
        scale_window=st.session_state.scale_window,
        shiftMin2Zero=st.session_state.shiftMin2Zero,
        log=st.session_state.log4bsl,
        quantile=st.session_state.quantile,
    )
    if st.session_state.smooth_bool:
        temp_data = pks_utils.applySGFilter(
            st.session_state.temporal_data,
            window_size=st.session_state.window_size,
            smoothing_order=st.session_state.smoothing_order,
            log=st.session_state.log4sg,
        )
        adj2bsl_data = pks_utils.applySGFilter(
            st.session_state.adj2bsl_data,
            window_size=st.session_state.window_size,
            smoothing_order=st.session_state.smoothing_order,
            log=st.session_state.log4sg,
        )
    else:
        temp_data = st.session_state.temporal_data
        adj2bsl_data = st.session_state.adj2bsl_data

    # st.write(f"Temp data: {temp_data}")
    # st.write(f"Adj2bsl data: {adj2bsl_data}")
    # st.write(f"Max of temp data: {np.max(temp_data)}")
    # st.write(f"Max of adj2bsl data: {np.max(adj2bsl_data)}")
    # st.write(f"Min of temp data: {np.min(temp_data)}")
    # st.write(f"Min of adj2bsl data: {np.min(adj2bsl_data)}")

    st.session_state.data2use = (
        _normalize_data(temp_data)
        if not st.session_state.adj2bsl_bool
        else _normalize_data(adj2bsl_data)
    )

    # st.write(f"Data used: {st.session_state.data2use}")

    figCTP = None
    figPkTrace = None

    figCTP, axCTP = fig_tools.create_plt_subplots(figsize=PLOTDICT["figsize"])

    axCTP.plot(
        st.session_state.data2use,
        color=PLOTDICT["color"],
        linewidth=PLOTDICT["linewidth"],
    )
    # axCTP.set_title(
    #     "No adjustment to Baseline"
    #     if not st.session_state.adj2bsl_bool
    #     else "Adjusted to Baseline"
    # )
    # axCTP.set_ylim(PLOTDICT["ymin"], PLOTDICT["ymax"])
    axCTP.set_ylabel("Normalized to Max", fontsize=PLOTDICT["fs"])
    axCTP.set_xlabel("Time (frames)", fontsize=PLOTDICT["fs"])

    figPkTrace, axPkTrace = fig_tools.create_plt_subplots(figsize=PLOTDICT["figsize"])
    if st.session_state.peak_algorithm == "scipy":
        selected_pks = get_pks_via_scipy(
            st.session_state.data2use,
            height=st.session_state.height,
            distance=st.session_state.distance,
            prominence=st.session_state.prominence,
        )
    elif st.session_state.peak_algorithm == "iterative_diffs":
        selected_pks = get_pks_via_iterative_diffs(
            st.session_state.data2use,
            frameWindow=st.session_state.frameWindow,
            sdThresh=st.session_state.sdThresh,
            timeout=st.session_state.timeout,
        )

    axCTP.plot(
        selected_pks,
        st.session_state.data2use[selected_pks],
        marker="o",
        color=PLOTDICT["peak_color"],
        markersize=PLOTDICT["markersize"],
        linestyle="none",
    )
    for pk in selected_pks:
        pk_slice = slice(pk - PLOTDICT["pre_peak"], pk + PLOTDICT["post_peak"] + 1)
        x_pk_idx = np.arange(-PLOTDICT["pre_peak"], PLOTDICT["post_peak"] + 1)
        y_pk_trace = st.session_state.data2use[pk_slice]
        if len(x_pk_idx) > len(y_pk_trace):
            x_pk_idx = x_pk_idx[: len(y_pk_trace)]
        axPkTrace.plot(
            x_pk_idx,
            y_pk_trace,
            linewidth=PLOTDICT["linewidth"],
        )
    axPkTrace.set_ylabel("Normalized to Max", fontsize=PLOTDICT["fs"])
    axPkTrace.set_xlabel("Onset from Peak (frames)", fontsize=PLOTDICT["fs"])
    axPkTrace.set_ylim(PLOTDICT["ymin"], PLOTDICT["ymax"])

    if figCTP is not None:
        st.pyplot(figCTP)

    if figPkTrace is not None:
        st.write("Pk Transient Profile")
        st.pyplot(figPkTrace)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Algorithm: ***{st.session_state.peak_algorithm.upper()}***")
            st.write(f"Peaks detected: **{int(len(selected_pks))}**")
        with col2:
            st.write("Parameters currently displayed:")
            print("Pk Parameters:")
            set_pk_params()


def get_pks_via_scipy(
    temporal_data: np.ndarray, height: float, distance: int, prominence: float
):
    pks, _ = pks_utils.find_pksViaScipy(
        Ca_arr=temporal_data,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    return pks.astype(int)


def get_pks_via_iterative_diffs(
    temporal_data: np.ndarray,
    frameWindow: int,
    sdThresh: float,
    timeout: int,
):
    pks, _, _ = pks_utils(
        frameWindow=frameWindow,
        sdThresh=sdThresh,
        timeout=timeout,
    ).find_CaTransients(temporal_data)
    return pks.astype(int)


def init_pksDict():
    st.session_state.pksDict = {}
    st.session_state.pksDict["baseline"] = {
        "adj2bsl_bool": st.session_state.adj2bsl_bool,
        "scale_window": st.session_state.scale_window
        if st.session_state.adj2bsl_bool
        else None,
        "quantile": st.session_state.quantile
        if st.session_state.adj2bsl_bool
        else None,
        "shiftMin2Zero": st.session_state.shiftMin2Zero
        if st.session_state.adj2bsl_bool
        else None,
        "log4bsl": st.session_state.log4bsl if st.session_state.adj2bsl_bool else None,
    }
    st.session_state.pksDict["smoothing"] = {
        "smoothing_bool": st.session_state.smooth_bool,
        "window_size": st.session_state.window_size
        if st.session_state.smooth_bool
        else None,
        "smoothing_order": st.session_state.smoothing_order
        if st.session_state.smooth_bool
        else None,
        "log4sg": st.session_state.log4sg if st.session_state.smooth_bool else None,
    }
    st.session_state.pksDict["algorithm"] = st.session_state.peak_algorithm
    if st.session_state.peak_algorithm == "scipy":
        st.session_state.pksDict["parameters"] = {
            "height": st.session_state.height,
            "distance": st.session_state.distance,
            "prominence": st.session_state.prominence,
        }
    elif st.session_state.peak_algorithm == "iterative_diffs":
        st.session_state.pksDict["parameters"] = {
            "frameWindow": st.session_state.frameWindow,
            "sdThresh": st.session_state.sdThresh,
            "timeout": st.session_state.timeout,
        }


def get_pks_for_all_cells():
    pksBYseg = []
    CTEMP2MOD = st.session_state.CTemp.copy()
    if st.session_state.adj2bsl_bool:
        for cell in range(CTEMP2MOD.shape[0]):
            CTEMP2MOD[cell, :] = pks_utils.normalizeByBaseline(
                CTEMP2MOD[cell, :],
                scale_window=st.session_state.scale_window,
                shiftMin2Zero=st.session_state.shiftMin2Zero,
                log=st.session_state.log4bsl,
                quantile=st.session_state.quantile,
            )
    if st.session_state.smooth_bool:
        for cell in range(CTEMP2MOD.shape[0]):
            CTEMP2MOD[cell, :] = pks_utils.applySGFilter(
                CTEMP2MOD[cell, :],
                window_size=st.session_state.window_size,
                smoothing_order=st.session_state.smoothing_order,
                log=st.session_state.log4sg,
            )
    for cell in range(CTEMP2MOD.shape[0]):
        # pksBYcell = []
        # seg_key = f"seg{cell}"
        # if seg_key not in st.session_state:
        #     st.session_state[seg_key] = None
        if st.session_state.peak_algorithm == "scipy":
            pks = get_pks_via_scipy(
                _normalize_data(CTEMP2MOD[cell, :]),
                height=st.session_state.height,
                distance=st.session_state.distance,
                prominence=st.session_state.prominence,
            )
        elif st.session_state.peak_algorithm == "iterative_diffs":
            pks = get_pks_via_iterative_diffs(
                CTEMP2MOD[cell, :],
                frameWindow=st.session_state.frameWindow,
                sdThresh=st.session_state.sdThresh,
                timeout=st.session_state.timeout,
            )
        pksBYseg.append(list(pks))
    return pksBYseg


def ExportMenuSetup(selected_session: str):
    def _write_json_export_statement():
        st.write("If export button is clicked, this JSON will be exported:")

    file_tag = text_dict()["file_tag"]
    dict2export = {
        "accepted": list(st.session_state.accepted),
        "rejected": list(st.session_state.rejected),
        "undecided": list(st.session_state.undecided),
    }

    if "pks" not in st.session_state.pksDict.keys():
        with st.spinner("Getting peaks for all cells..."):
            st.session_state.pksDict["pks"] = get_pks_for_all_cells()

    col1, col2 = st.columns(2)
    with col1:
        export_comp_selection_button = st.button(
            f"{EMOJIS['save']} Export Component Selection"
        )
        if export_comp_selection_button:
            with st.spinner("Exporting Component Selection..."):
                try:
                    json_file_name = os.path.join(selected_session, ACC_REJ_JSON_FNAME)
                    saveNloadUtils.savedict2file(
                        dict_to_save=dict2export,
                        dict_name="Component Selection",
                        filename=json_file_name,
                        filetype_to_save=file_tag["JSON"],
                    )
                    st.success(f"Exported to {os.path.abspath(json_file_name)}")
                except Exception as e:
                    st.error(f"Error exporting to {json_file_name}: {e}")
        with st.expander(f"{EMOJIS['preview']} Preview Component Selection"):
            _write_json_export_statement()
            for key, value in dict2export.items():
                st.write(f"{key}: {value}")
    with col2:
        export_pks_button = st.button(f"{EMOJIS['pk']} Export Peaks Dictionary")
        if export_pks_button:
            with st.spinner("Exporting Peaks Dictionary..."):
                try:
                    session_name = selected_session.split("/")[-1]
                    json_file_name = os.path.join(
                        selected_session, f"{session_name}{PKS_FNAME}"
                    )
                    saveNloadUtils.savedict2file(
                        dict_to_save=st.session_state.pksDict,
                        dict_name="Peaks Dictionary",
                        filename=json_file_name,
                        filetype_to_save=file_tag["JSON"],
                    )
                    st.success(f"Exported to {os.path.abspath(json_file_name)}")
                except Exception as e:
                    st.error(f"Error exporting to {json_file_name}: {e}")
        with st.expander(f"{EMOJIS['preview']} Preview Peaks Dictionary"):
            _write_json_export_statement()
            st.write(st.session_state.pksDict)


def _refresh_per_session_change():
    st.session_state.loaded_AL_fromSDISX = None
    st.session_state.accepted = set()
    st.session_state.rejected = set()
    st.session_state.undecided = set()
    st.session_state.prev_index = 0
    st.session_state.selected_index = 0
    st.session_state.export_menu_bool = False
    st.session_state.adjust_labels_bool = False
    st.session_state.adj2bsl_bool = False
    st.session_state.smooth_bool = False
    st.session_state.pksDict = {}
    st.session_state.CTemp = None
    st.session_state.ASpat = None
    st.session_state.DSimage_fname = None
    st.session_state.CNMF_accepted = None
    st.session_state.CNMF_rejected = None
    st.session_state.AL_SDISX_accepted = set()
    st.session_state.AL_SDISX_rejected = set()
    st.session_state.AL_SDISX_undecided = set()
    st.session_state.NNCE_accepted = set()
    st.session_state.NNCE_rejected = set()
    st.session_state.NNCE_predictions = None
    st.session_state.eval_completed_bool = False
    st.session_state.PREV_JSON_accepted = set()
    st.session_state.PREV_JSON_rejected = set()
    st.session_state.PREV_JSON_undecided = set()
    st.session_state.use_PREV_JSON_checkbox_result = False
    st.session_state.use_AL_SDISX_checkbox_result = False
    st.session_state.use_CNMF_CE_checkbox_result = False
    st.session_state.use_NNCE_checkbox_result = False
    st.session_state.show_eval_results_bool = False


def _refresh_NN_settings():
    st.session_state.model_runner = None
    st.session_state.model_name2load = None
    st.session_state.nn_success_bool = False
    st.session_state.feature_extractor = None
    st.session_state.features_vector = None
    st.session_state.save_model_bool = False
    st.session_state.load_model_bool = False
    st.session_state.selected_model_dict = {}
    st.session_state.run_train_model_bool = False
    st.session_state.listExpNames = []
    st.session_state.CTempList = []
    st.session_state.ASPatList = []
    st.session_state.dim4spatialList = []
    st.session_state.labelList = []
    st.session_state.eval_completed_bool = False
    st.session_state.NNCE_accepted = set()
    st.session_state.NNCE_rejected = set()
    st.session_state.NNCE_predictions = None
    st.session_state.show_eval_results_bool = False


def refresh_page():
    _refresh_per_session_change()
    _refresh_NN_settings()
    st.session_state.paths = []
    st.session_state["path_search"] = ""
    st.session_state.selected_session = None
    st.session_state.prev_session = None
    st.session_state.prev_index = 0
    st.session_state.selected_index = 0
    st.session_state.loaded_AL_fromSDISX = None
    st.rerun()


def print_namesNvalues(names: list[str], values: list[any]):
    for name, value in zip(names, values):
        print(f"--{name} set to: {value}")
    print()


def main():
    Streamlit_utils.setup_page_config(
        app_name="segDict Utility",
        app_abbrv="SD",
        init_session_state_func=init_session_state,
        refresh_func=refresh_page,
        create_path_db_func=create_paths_db,
        DB_NAME=DB_NAME,
    )
    # initiate, create, manage search section
    Streamlit_utils.init_SearchSection(DB_NAME=DB_NAME)

    # initiate, create, manage session selection
    selected_session = Streamlit_utils.init_SessionSelection(DB_NAME=DB_NAME)
    print(f"Selected session: {selected_session}")

    if st.session_state.prev_session is None:
        st.session_state.prev_session = selected_session

    if st.session_state.prev_session != selected_session:
        _refresh_per_session_change()

    if selected_session:
        col1, col2 = st.columns(2)
        with col1:
            print("--importing segDict")
            get_segDict(selected_session)
        with col2:
            NNsectionHandler(selected_session)

    if st.session_state.prev_session != selected_session:
        print(
            f"segDict loaded, setting previous session to: {selected_session}, and initializing pksDict"
        )
        st.session_state.prev_session = selected_session
        init_pksDict()

    if st.session_state.CTemp is not None and st.session_state.ASpat is not None:
        PlotSectionHandler()
        if st.button(f"{EMOJIS['tools']} Open Export Menu"):
            st.session_state.export_menu_bool = not st.session_state.export_menu_bool
            print(
                f"Export menu button clicked -- State: {st.session_state.export_menu_bool}"
            )

        if st.session_state.export_menu_bool:
            with st.spinner("Arranging output files..."):
                ExportMenuSetup(selected_session=selected_session)


if __name__ == "__main__":
    main()
