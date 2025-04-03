import streamlit as st
from pathlib import Path
import os
import getpass
import multiprocessing
import sqljobscheduler as sqljs
import re
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum
from CLAH_ImageAnalysis.utils import Streamlit_utils
from CLAH_ImageAnalysis.utils import text_dict
from CLAH_ImageAnalysis.utils import enum_utils
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import db_utils

DB_NAME = "paths_cache4M2SD"


def is_valid_email(email) -> bool:
    """
    Validate email address format

    Parameters:
        email (str): Email address to validate

    Returns:
        bool: True if email address is valid, False otherwise
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def file_check4dbCreation(filenames: list[str]) -> bool:
    """
    Check if any file in the list ends with H5 or ISXD

    Parameters:
        filenames (list[str]): List of filenames to check

    Returns:
        bool: True if any file ends with H5 or ISXD, False otherwise
    """
    file_tag = text_dict()["file_tag"]
    h5check = any(
        f.endswith(file_tag["H5"]) and file_tag["ELEMENT"] in f for f in filenames
    )
    isxdcheck = any(f.endswith(file_tag["ISXD"]) for f in filenames)
    return h5check or isxdcheck


def file_check4SessionDict(session: Path) -> bool:
    """
    Check if session has H5 or ISXD file

    Parameters:
        session (Path): Path to session directory

    Returns:
        bool: True if session has H5 or ISXD file, False otherwise
    """
    file_tag = text_dict()["file_tag"]
    h5check = any(
        p.is_file() and p.suffix == file_tag["H5"] and file_tag["ELEMENT"] in p.name
        for p in session.iterdir()
    )
    isxdcheck = any(
        p.is_file() and p.suffix == file_tag["ISXD"] for p in session.iterdir()
    )
    return h5check or isxdcheck


def create_paths_db() -> tuple[int, int]:
    """Create and populate SQLite database with paths

    Returns:
        tuple[int, int]: Total paths and sessions in the database.
    """
    total_paths, total_sessions = db_utils.init_dbCreation(
        DB_NAME=DB_NAME,
        file_check_func4DB=file_check4dbCreation,
        file_check_func4SessionDict=file_check4SessionDict,
    )

    return total_paths, total_sessions


def init_session_state():
    """Initialize session state variables"""
    if "paths" not in st.session_state:
        st.session_state.paths = []
    if "param_vars" not in st.session_state:
        st.session_state.param_vars = create_parameter_vars()
    if "is_onep" not in st.session_state:
        st.session_state.is_onep = False
    # if "sess2process_list" not in st.session_state:
    #     st.session_state.sess2process_list = []


def create_parameter_vars():
    """Create variables for all parameters using defaults from Parser4M2SD"""
    param_vars = {}
    for (param_name, _), param_info in TSF_enum.Parser4M2SD.ARG_DICT.value.items():
        if param_info["TYPE"] == "bool":
            param_vars[param_name] = param_info["DEFAULT"]
        elif param_info["TYPE"] == "int" and "n_proc" not in param_name:
            param_vars[param_name] = (
                param_info["DEFAULT"] if param_info["DEFAULT"] is not None else 1
            )
        elif param_info["TYPE"] == "int" and "n_proc" in param_name:
            param_vars[param_name] = (
                param_info["DEFAULT"]
                if param_info["DEFAULT"] is not None
                else multiprocessing.cpu_count()
            )
    return param_vars


def refresh_page():
    """Clear paths and path search before rerunning page"""
    st.session_state.paths = []
    st.session_state["path_search"] = ""
    st.rerun()


def get_email():
    """Get email address needed for analysis to receive notifications"""
    st.header("Email Address (REQUIRED)")
    st.write(
        "Please enter your email address to receive notifications regarding your scheduled analysis. Valid input will enable you to adjust settings and enable the 'Run Analysis' button."
    )
    # Email input
    email = st.text_input(
        "Email Address",
        label_visibility="collapsed",
    )

    if email and not is_valid_email(email):
        st.warning("Please enter a valid email address")
    elif email and is_valid_email(email):
        return email


def define_main_options():
    """Define main options for running analysis. If all left False, analysis will run but not doing anything."""
    # Main Options
    st.header("Main Options")
    motion_correct = st.checkbox(
        "Motion Correction (MC)",
        value=st.session_state.param_vars.get("motion_correct", False),
    )
    segment = st.checkbox(
        "Segment (SG)",
        help="Can be used without MC if MC was already performed in a previous run",
        value=st.session_state.param_vars.get("segment", False),
    )
    overwrite = st.checkbox(
        "Overwrite",
        help="Only use this if you want to start analysis from scratch & delete all existing M2SD output files",
        value=st.session_state.param_vars.get("overwrite", False),
    )

    st.divider()
    return motion_correct, segment, overwrite


def define_advanced_options():
    """Define advanced options for running analysis. These options are not required for analysis to run, but can significantly impact analysis results/downstream workflows"""
    with st.expander("Advanced Options"):
        st.warning(
            "Please be careful with these settings. They can significantly impact your analysis results/downstream workflows"
        )

        col1, col2 = st.columns(2)

        with col1:
            concatenate = st.checkbox(
                "Concatenate sessions on same day",
                help="If checked, will concatenate sessions. Best for 2 sessions of data from the same day.",
                value=st.session_state.param_vars.get("concatenate", False),
            )
            # prev_sd_varnames = st.checkbox(
            #     "Use previous SD variable names",
            #     help="Saves variable names in segDict using single letters",
            #     value=st.session_state.param_vars.get(
            #         "prev_sd_varnames", False
            #     ),
            # )
            compute_metrics = st.checkbox(
                "Compute Motion Correction Metrics",
                help="Export metrics related to quality of motion correction done. Useful for debugging, but will add extra time to analysis.",
                value=st.session_state.param_vars.get("compute_metrics", False),
            )

            export_postseg_residuals = st.checkbox(
                "Export Post-Segmentation Residuals",
                help="Export the results of segmentation as a video file. Useful to see which components in terms of spatial and temporal dynamics were accepted/rejected.",
                value=st.session_state.param_vars.get(
                    "export_postseg_residuals", False
                ),
            )

        with col2:
            # use_cropper = st.checkbox(
            #     "Use Cropper",
            #     help="Crop dimensions of recording (only for 1photon/.isxd files)",
            #     value=st.session_state.param_vars.get("use_cropper", False),
            # Session directory)
            separate_channels = st.checkbox(
                "Separate Channels",
                help="Motion correct channels separately (2photon data with 2 channels only)",
                value=st.session_state.param_vars.get("separate_channels", False),
            )

            mc_iter = st.number_input(
                "Number of Motion Correction iterations",
                min_value=1,
                value=st.session_state.param_vars.get("mc_iter", 1),
                help="Each additional iteration will add time to analysis. After the first run, MC parameters will be modified, going from coarse to fine correction.",
            )

    return (
        concatenate,
        compute_metrics,
        export_postseg_residuals,
        separate_channels,
        mc_iter,
    )


def modfify_params_section():
    """Modify MC/CNMF Parameters section"""
    with st.expander("Modify MC/CNMF Parameters"):  # Add 1P/2P toggle
        if "is_onep" not in st.session_state:
            st.session_state.is_onep = False

        title = f"### Set Specific Motion Correction/CNMF Parameters for {'1-Photon' if st.session_state.is_onep else '2-Photon'}"
        st.markdown(title)
        st.warning(
            "Default settings are decent, but you can modify them if you know what you're doing!"
        )

        is_onep = st.toggle(
            "Switch to 1-Photon Parameters", value=st.session_state.is_onep
        )
        if is_onep != st.session_state.is_onep:
            st.session_state.is_onep = is_onep
            st.rerun()

        tab1, tab2 = st.tabs(["Motion Correction", "CNMF"])
        PARAMNAMES = {
            "BOOLEAN": {
                "MC": [
                    "BORDER_NAN",
                    "NONNEG",
                    "PW_RIGID",
                    "SHIFTS_OPENCV",
                    "USE_CUDA",
                ],
                "CNMF": [
                    "CENTER_PSF",
                    "CHECK_NAN",
                    "ISDENDRITES",
                    "LOW_RANK_BACKGROUND",
                    "ONLY_INIT_PATCH",
                    "USE_CNN",
                ],
            },
            "CATEGORY": {
                "MC": [],
                "CNMF": ["METHOD_INIT", "METH_DECONV"],
            },
        }

        for k, v in PARAMNAMES.items():
            for type_key in v.keys():
                v[type_key].sort()

        with tab1:
            display_parameter_tab(
                param_type="MC", is_onep=is_onep, PARAMNAMES=PARAMNAMES
            )
        with tab2:
            display_parameter_tab(
                param_type="CNMF", is_onep=is_onep, PARAMNAMES=PARAMNAMES
            )


def display_parameter_tab(param_type: str, is_onep: bool, PARAMNAMES: dict):
    """
    Display parameter tab

    Parameters:
        param_type (str): Parameter type
        is_onep (bool): Whether to use 1-Photon parameters
        PARAMNAMES (dict): Parameter names
    """
    if param_type == "MC":
        params_enum = TSF_enum.MOCO_PARAMS
    elif param_type == "CNMF":
        params_enum = TSF_enum.CNMF_PARAMS

    params_all = enum_utils.enum2dict(params_enum)
    par_idx = 0 if not is_onep else 1
    params = {k: v[par_idx] for k, v in params_all.items()}
    help_text = {k: v[-1] for k, v in params_all.items()}

    continuous_params = [
        p
        for p in sorted(params.keys())
        if p
        not in PARAMNAMES["BOOLEAN"][param_type] + PARAMNAMES["CATEGORY"][param_type]
    ]

    st.subheader("Continuous Parameters")
    mc_cols = st.columns(4)
    for idx, param in enumerate(continuous_params):
        with mc_cols[idx % 4]:
            st.number_input(
                param,
                value=params[param],
                key=f"{param_type.lower()}_{param}",
                help=help_text[param],
            )

    if PARAMNAMES["BOOLEAN"][param_type]:
        st.subheader("Boolean Parameters")
        for param in PARAMNAMES["BOOLEAN"][param_type]:
            st.checkbox(
                param,
                value=params[param],
                key=f"{param_type.lower()}_{param}",
                help=help_text[param],
            )

    if PARAMNAMES["CATEGORY"][param_type]:
        for param in PARAMNAMES["CATEGORY"][param_type]:
            if param_type == "CNMF":
                options2use = TSF_enum.CNMF_OPTS[param]
            else:
                continue
            st.selectbox(
                param,
                options=options2use,
                key=f"{param_type.lower()}_{param}",
                help=help_text[param],
            )

    # Export button
    if st.button(f"Export {param_type} Parameters", key=f"export_{param_type.lower()}"):
        if not st.session_state.paths:
            st.error("Please select at least one directory first")
        else:
            for path in st.session_state.paths:
                # Get all session folders
                sess_folders = [
                    d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
                ]
                sess_folders.sort()

                # Collect parameters
                collected_params = {
                    param: st.session_state[f"{param_type.lower()}_{param}"]
                    for param in params.keys()
                }

                # Convert "None" strings to None
                collected_params = {
                    k: None if v == "None" else v for k, v in collected_params.items()
                }

                # Export to each session
                for sess in sess_folders:
                    sess_path = os.path.join(path, sess)
                    os.chdir(sess_path)
                    TSF_enum.generateNexport_user_set_params(
                        onePhotonCheck=is_onep,
                        param_type=param_type,
                        **collected_params,
                    )
            st.success(f"{param_type} Parameters exported successfully")


def run_analysis_button(email):
    """
    Run analysis button

    Parameters:
        email (str): Email address to send notifications to
    """
    if st.button(
        "⚡ Run Analysis",
        disabled=not (email and st.session_state.paths),
        help="Queue analysis for selected directories. Button only active if email and paths are provided.",
    ):
        try:
            run_analysis(email)
            st.success("Analysis jobs have been queued successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def main():
    # Setup page config, including session state, title, logo, images
    Streamlit_utils.setup_page_config(
        app_name="Motion Correction & Segmentation (M2SD)",
        app_abbrv="M2SD",
        init_session_state_func=init_session_state,
        refresh_func=refresh_page,
        create_path_db_func=create_paths_db,
        DB_NAME=DB_NAME,
    )

    Streamlit_utils.init_SearchSection(DB_NAME=DB_NAME)

    if st.session_state.paths:
        if "sess2process_list" not in st.session_state:
            st.session_state.sess2process_list = ["all"] * len(st.session_state.paths)

        Streamlit_utils.init_Session2process_selection(DB_NAME=DB_NAME)

        # Main Options
        motion_correct, segment, overwrite = define_main_options()

        # Advanced Options in expander
        (
            concatenate,
            compute_metrics,
            export_postseg_residuals,
            separate_channels,
            mc_iter,
        ) = define_advanced_options()

        # Expandable section for modifying MC/CNMF parameter
        modfify_params_section()

        # Email section
        email = get_email()
        if email:
            # load parameters into session state with email
            st.session_state.param_vars.update(
                {
                    "mc_iter": mc_iter,
                    "overwrite": overwrite,
                    "concatenate": concatenate,
                    "compute_metrics": compute_metrics,
                    "use_cropper": False,  # cropper will be disabled for sql jobs because won't make sense to request user input during a job
                    "separate_channels": separate_channels,
                    "export_postseg_residuals": export_postseg_residuals,
                    "motion_correct": motion_correct,
                    "segment": segment,
                    "email": email,
                    "from_sql": True,
                }
            )
        # setup run analysis button
        run_analysis_button(email)


def run_analysis(email):
    """Queue analysis jobs for all selected paths"""
    sqlUtils = sqljs.JobQueue()
    M2SD_path = os.path.join(paths.get_code_dir_path("tifStackFunc"), "MoCo2segDict.py")
    python_exec = str(paths.get_python_exec_path())

    for path, sess2process in zip(
        st.session_state.paths, st.session_state.sess2process_list
    ):
        parameters = {
            "path": path,
            "sess2process": sess2process,
            **st.session_state.param_vars,
        }
        sqlUtils.add_job(
            programPath=M2SD_path,
            path2python_exec=python_exec,
            parameters=parameters,
            email_address=email,
            user=getpass.getuser(),
            python_env=paths.get_python_env(),
        )


if __name__ == "__main__":
    main()
