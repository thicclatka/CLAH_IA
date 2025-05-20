import json

from rich import print

from CLAH_ImageAnalysis.utils.paths import get_config_wIN_pyDir


def color_dict() -> dict:
    """
    Returns a dictionary mapping color names to their corresponding hex values.
    Loads colors from the config/colors.json file.

    Returns:
        dict: A dictionary with color names as keys and hex values as values.
    """
    config_path = get_config_wIN_pyDir() / "colors.json"
    with open(config_path, "r") as f:
        color_data = json.load(f)
    return color_data["colors"]


def color_dict_4cues() -> dict:
    """
    Returns a dictionary mapping cue names to color codes for plotting.

    Returns:
        dict: A dictionary mapping cue names to color codes.
    """
    colors = color_dict()
    color_dict_4cues = {
        "CUE1": colors["green"],
        "OMITCUE1": colors["orange"],
        "CUE2": colors["red"],
        "OMITCUE2": colors["violet"],
        "CUE1_SWITCH": colors["cyan"],
        "CUE2_SWITCH": colors["yellow"],
        "OMITCUE1_SWITCH": colors["pink"],
        "TONE": colors["turquoise"],
        "OMITTONE": colors["darkturquoise"],
        "LED": colors["darkblue"],
        "OMITLED": colors["darkgreen"],
        "OMITBOTH": colors["gray"],
        "OPTO": colors["darkorange"],
        "CUEWOPTO": colors["darkyellow"],
        "BOTHCUES": colors["darkblue"],
        "NON": colors["darkgray"],
        "PLACE": colors["darkviolet"],
    }
    return color_dict_4cues


def color_bool(value: str) -> str:
    """
    Formats a boolean value with color tags.

    Parameters:
        value (str): The boolean value to be formatted.

    Returns:
        str: The formatted boolean value with color tags.

    Example:
        >>> color_bool("Yes")
        '[green]Yes[/green]'
        >>> color_bool("No")
        '[red]No[/red]'
    """
    colors = color_dict()
    yc = colors["green"]
    nc = colors["red"]
    if value == "Yes":
        return f"[{yc}]Yes[/{yc}]"
    elif value == "No":
        return f"[{nc}]No[/{nc}]"
    else:
        return value


def section_breaker(
    btype: str, mini: bool = False, alt_color_break: bool = False
) -> None:
    """
    Prints a section breaker with a specified type.

    Parameters:
        btype (str): The type of section breaker to print.
        mini (bool): Optional. If True, a mini section breaker will be printed.
    """
    text_to_use = text_dict()
    colors = color_dict()
    c_breaker = colors["darkturquoise"] if not alt_color_break else colors["green"]

    breaker = text_to_use["breaker"][btype]
    print(f"[{c_breaker}]{breaker}[/{c_breaker}]")
    if not mini:
        print()


def print_header(
    header: str,
    steps: list | None = None,
    subhead: bool = False,
    alt_color_break: bool = False,
    alt_color_txt: bool = False,
) -> None:
    """
    Prints a formatted header with optional subhead and steps.

    Parameters:
        header (str): The main header text.
        steps (list, optional): A list of steps to be printed below the header. Defaults to None.
        subhead (bool, optional): Indicates whether the header is a subhead. Defaults to False.
        alt_color_break (bool, optional): Indicates whether to use an alternative color for the section breaker. Defaults to False.
        alt_color_txt (bool, optional): Indicates whether to use an alternative color for the text. Defaults to False.
    """
    colors = color_dict()
    c_head = colors["darkcoral"] if not alt_color_txt else colors["orange"]

    btype = "hash"
    if subhead:
        btype = "dotted"
    else:
        header = header.upper()

    section_breaker(btype, alt_color_break)
    print(f"[bold {c_head}]{header}[/bold {c_head}]\n")
    if steps is not None and len(steps) > 0:
        for step in steps:
            print(step)
        print()
    section_breaker(btype, alt_color_break)


def text_dict(
    frame: str = "|-- ", empty_frame: str = "|   ", breaker_length: int = 100
) -> dict:
    """
    Returns a dictionary containing various text formatting elements used in the CLAH Image Analysis script.

    Parameters:
        frame (str, optional): The frame character used for text frames. Defaults to "|-- ".
        empty_frame (str, optional): The frame character used for empty frames. Defaults to "|   ".
        breaker_length (int, optional): The length of the breaker lines. Defaults to 100.

    Returns:
        dict: A dictionary containing various text formatting elements. Here is a list of the keys:
            frames
            cueType_abbrev
            headers
            steps
            parser
            breaker
            completion
            wrappers
            file_tag
            GPU
            selector
            dict_name
            brain_regions
            IMP_FILE_KW
            strings
            GUI_ELEMENTS
            GUICLASSUTILS
    """

    def _add_CLAH(text: str) -> str:
        """Add the CLAH prefix to the text.

        Parameters:
            text (str): The text to add the CLAH prefix to.

        Returns:
            str: The text with the CLAH prefix.
        """
        return f"{CLAH} -- {text}"

    CLAH = "CLAH IMAGE ANALYSIS"
    FOR_MISSING = "*|FOR MISSING"
    MOCO2SD = "M2SD"
    QUCKTUNING = "QT"
    CELLREGIST = "CRR"
    CLUSTERINFO = "CIC"
    POSTCR_CFF = "PCR_CFF"
    WRAPMSS = "WMSS"
    TWOODOR = "TOD"
    ISX_ANLZR = "ISX_ANLZR"

    main_titles = {
        MOCO2SD: "h5 image stack motion corrector & segmentation operator",
        QUCKTUNING: "Cue Shift Tuning",
        CELLREGIST: "cell register via ROICaT",
        CLUSTERINFO: "CR cluster info collater",
        POSTCR_CFF: "Post CR Cue Cell Finder",
        WRAPMSS: "wrapMultSessStruct",
        TWOODOR: "Two Odor Decoder",
        ISX_ANLZR: "ISX Analysis",
    }

    # load strings from config
    with open(get_config_wIN_pyDir() / "text.json", "r") as f:
        strings_from_config = json.load(f)

    text_dict = {
        "frames": strings_from_config["frames"],
        "cueType_abbrev": strings_from_config["cueType_abbrev"],
        "emojis": strings_from_config["emojis"],
        "headers": {
            **{
                f"main_title_{key}": _add_CLAH(title)
                for key, title in main_titles.items()
            },
            "path_fold_sel": "General Path & folder selector",
            "fold_anlz": "Analyzing/processing folder(s) {} of {}: {}",
            "sess_extrct": "Extracting data from session {} of {} for ID {}",
            f"start_msg_{WRAPMSS}": "Starting multSessSegStruc process for ID {}",
            "dup_note": "\n*NOTE: found previous processed file..."
            "creating file append w/date to avoid overwriting*",
        },
        "steps": {
            "path_fold_sel": {
                "s1": "01 - Determine path for analysis/processing",
                "s2": "02 - Select sessions/folders(s) for analysis/processing",
                "no_path1": "NOTE: No path was provided via command line. You will now be prompted to select a path.",
                "no_path2": " " * 6
                + "Path should be the parent folder containing the session folders.",
                "note": "NOTE: The selection provided is based on the available sessions/folders in the path, the contents of these folders, and the analysis.",
                f"note_{MOCO2SD}": "NOTE: For MoCo2segDict, the selection of folders is based on the presence of a H5 file named before any motion correcting processing,\n"
                + "" * 6
                + "with the file name format: [DATE]_[SUBJ_ID]_[TEST_TYPE]_Cycle00001_Element00001.h5.\n"
                + " " * 6
                + "If these H5 files are not present, autoClean.py needs to be run to conver the tifs stack into an analyzable H5.\n"
                + "" * 6
                + "If motion corrected H5s are present (eMC), user will be given option to either reprocess or skip.",
            },
            # TODO: NEED TO WRITE STEPS FOR QT, CRR, CIC
            f"{MOCO2SD}": {
                "main": [
                    "Steps:",
                    "| 01 -- Load h5 (2 photon)/isxd (1 photon) per session",
                    "| 02 -- Motion correct",
                    "| 03 -- Trim & filter motion corrected data",
                    "| 04 -- Create segDict derived from CNMF done on motion corrected data",
                ],
                "header": {
                    "s1": "01 - Loading h5 (2 photon)/isxd (1 photon)",
                    "s2": "02 - Motion correcting",
                    "s3": "03 - Trimming and filtering",
                    "s2-3_skip": "02-03 - Skipping motion correction & image stack manipulations",
                    "s4": "04 - Creating segDict via CNMF",
                    "s4_skip": "04 - Skipping segDict creation",
                },
            },
        },
        "parser": strings_from_config["parser"],
        "breaker": {
            "hash": "#" * breaker_length,
            "lean": "-" * breaker_length,
            "dotted": "." * breaker_length,
            "dotted_half": "." * int(breaker_length / 2),
            "hash_half": "#" * int(breaker_length / 2),
            "lean_half": "-" * int(breaker_length / 2),
            "line_break": "\n",
        },
        "completion": {
            "forLoop": "Processing complete for file {} of {}: {}",
            "forLoop_CSS": "Extraction complete for session {} of {} for ID {}",
            "whole_proc_MSS": "Processing multSessSegStruc complete for ID {}",
            "whole_proc": "Processing complete for all selected folder(s)",
            "whole_proc_CSS": "cueShiftTuning complete for all selected folder(s)",
            "small_proc": "---done",
            "GUI": "done!",
        },
        "wrappers": strings_from_config["wrappers"],
        "file_tag": strings_from_config["file_tag"],
        "Folders": strings_from_config["Folders"],
        "GPU": strings_from_config["GPU"],
        "selector": {
            "tags": {
                "EMC": "eMC",
                "H5": "h5",
                "SD": "sD",
                "CSS": "CSS",
                "TDML": "tdml",
                "MSS": "MSS",
                "CI": "CI",
                "ISXD": "isxd",
            },
            "choices": {
                "USER_CHOICE": "{}| User selection",
                "EMC_PRES": [
                    "1| All, NOT including those w/previous eMC processing: {}",
                    "2| All, including those w/previous eMC: {}",
                ],
                "EMC_ABS": [
                    "1| All [no current sessions w/in folder have previous eMC processing]: {}",
                ],
                "SD": [
                    "1| All session(s) with segDict present w/required .tdml: {}",
                ],
                "SD_NO_TDML": [
                    "1| All session(s) with segDict present: {}",
                ],
                "CSSbyID": [
                    "1| All subject(s) with multiple sessions w/respective cueShiftStrucs per session: {}",
                ],
                "CSS": [
                    "1| All session(s) with cueShiftStruc: {}",
                ],
                "MULTI": [
                    "1| All combined session folder(s): {}",
                ],
            },
            "strings": {
                "fold_h5": "\\Session(s) w/{} original H5 (2 photon):",
                "fold_need_h5": "FOR MISSING ORIGINAL H5s: Need to convert Tifs h5 before proceeding*",
                "fold_emc": "\\Session(s) w/{} motion corrected H5 processing:",
                "fold_need_emc": f"{FOR_MISSING} MOTION CORRECTED H5s: Run MoCo2segDict to create eMC H5s",
                "fold_isxd": "\\Session(s) w/{} isxd (1 photon):",
                "fold_need_isxd": f"{FOR_MISSING} ISXD: Need to locate these files & move to correct folder",
                "fold_sd": "\\Session(s) w/{} segDict present:",
                "fold_need_sd": f"{FOR_MISSING} SEGDICTS: Run MoCo2segDict to create segDict",
                "fold_tdml": "\\Session(s) w/{}tdml present:",
                "fold_need_tdml": f"{FOR_MISSING} TDML: Need to locate these files & move to correct folder",
                "fold_mss": "\\Combined Session folder(s) w/{} multSessSegStruc:",
                "fold_need_mss": f"{FOR_MISSING} MULTSESSSEGSTRUC: Run wrapMultSessStruct to create multSessSegStruc",
                "fold_ci": "\\Combined Session folder(s) w/{} cluster_info (JSON): ",
                "fold_need_ci": f"{FOR_MISSING} CLUSTER INFO: Run cellRegistrar_wROICaT create cluster_info",
                "fold_css": "\\Session(s) w/{} cueShiftStruc: ",
                "fold_need_css": f"{FOR_MISSING} CUESHIFTSTRUC: Run quickTuning to create cueShiftStruc",
                "subj_yes_CSS_all": "\\Mice w/multiple sessions containing their respective cueShiftStrucs:",
                "subj_no_CSS_all": "\\Mice w/multiple sessions BUT missing some or all of their respecitve cueShiftStrucs:",
                "subj_need_multCSS": "Found no subjects w/multiple sessions! Please run appropriate analyses before running again. Aborting script.",
                "subj_yes_SD_all": "\\Mice w/multiple sessions containing their respective segDicts:",
                "subj_no_SD_all": "\\Mice w/multiple sessions BUT missing some or all of their respecitve segDicts:",
                "subj_need_multSD": "Found no subjects w/multiple sessions! Please run appropriate analyses before running again. Aborting script.",
                "which_sess": "Which session(s) do you wish to process? ",
                "all_selected_noemc": "All available sessions with no previous eMC processing were selected to be processed",
                "all_selected_yesemc": "All available sessions including those with previous eMC processing were selected to be processed",
                "all_selected_sd": "All sessions with a present segDict AND .tdml were selected to be processed",
                "all_selected_sd_no_tdml": "All sessions with a present segDict were selected to be processed",
                "all_selected_css": "All subjects w/multiple sessions that contain appropriate cueShiftStrucs were selected to be processed",
                "all_selected_mss": "All combined session folders with multSessSegStruc were selected to be processed",
                "all_selected_ci": "All combined session folders with cluster_info were selected to be processed",
                "what_selected": "Session(s) to be processed:",
                "note_info_1": "NOTE: See numbers above to see which session(s)/subject(s) are available.",
                "note_dash_2": " " * 6
                + "If selecting more than one session/subject, separate values with commas",
                "note_dash_3": " " * 6 + "or use dashes (i.e. 1-3 = 1,2,3).",
                "note_all": " " * 6 + "Type 'all/ALL' to select all available sessions",
                "note_allnoemc": " " * 6
                + "Type 'allnoemc/ALLNOEMC' to select all available sessions with no previous eMC processing.",
                "note_exit": " " * 6
                + "Type 0, exit/EXIT, or quit/QUIT to exit script.",
                "input_str": "Type selection here: ",
                "wrong_sel": "***\nERROR: A session ({}) was selected that is out of range/cannot be processed at this time.\nRestarting selection process to ensure legitimate choice is entered.\nPlease read command line output to avoid this issue.\n***\n",
                "CSS_DETAIL": "***DETAIL ONLY FOR CSS SELECTION: PLEASE SELECT USING SUBJECT ID (SITS IN PARENTHESES), NOT SESSION***",
                "user_abort": "User selected to abort analysis... exiting program now.\n",
            },
        },
        "dict_name": strings_from_config["dict_name"],
        "brain_regions": strings_from_config["brain_regions"],
        "IMP_FILE_KW": strings_from_config["IMP_FILE_KW"],
        "YES_RESP": strings_from_config["YES_RESP"],
        "NO_RESP": strings_from_config["NO_RESP"],
        "REGEX": strings_from_config["REGEX"],
    }

    # adding other more complicated file tags given already set smaller ones

    ftags = text_dict["file_tag"]
    stags = text_dict["selector"]["tags"]
    comp_fname = (
        ftags["SQZ"]
        + ftags["EMC"]
        + ftags["CA"]
        + ftags["TEMPFILT"]
        + ftags["DOWNSAMPLE"]
    )

    comp_eMCfname = comp_fname
    comp_eMCh5fname = comp_fname + ftags["H5"]
    comp_sDfname_ending = comp_fname + ftags["SD"]

    pre_EMC_h5 = (
        ftags["CYCLE"] + ftags["CODE"] + ftags["ELEMENT"] + ftags["CODE"] + ftags["H5"]
    )

    text_dict["file_tag"]["COMP_EMCFNAME"] = comp_eMCfname
    text_dict["file_tag"]["COMP_EMCH5FNAME"] = comp_eMCh5fname
    text_dict["file_tag"]["COMP_SDFNAME"] = comp_sDfname_ending
    text_dict["file_tag"]["PRE_EMC_H5"] = pre_EMC_h5

    # creating more fleshed out selector tags

    text_dict["selector"]["file_types"] = {
        stags["EMC"]: [ftags["COMP_EMCFNAME"], ftags["H5"]],
        # stags["H5"]: ftags["PRE_EMC_H5"],
        stags["H5"]: [ftags["H5"], ftags["CYCLE"]],
        stags["SD"]: ftags["SD"],
        stags["CSS"]: ftags["CSS"],
        stags["TDML"]: ftags["TDML"],
        stags["MSS"]: ftags["MSS"],
        stags["CI"]: ftags["CLUSTER_INFO"],
        stags["ISXD"]: ftags["ISXD"],
    }

    text_dict["selector"]["singleSess"] = [
        stags["EMC"],
        stags["SD"],
        stags["H5"],
        stags["CSS"],
        stags["ISXD"],
    ]

    text_dict["selector"]["multiSess"] = [stags["MSS"], stags["CI"]]

    text_dict["QL_LNAMES"] = {
        "RAW_H5": "00_RAW_H5",
        "RAW_ISXD": "00_RAW_ISXD",
        "RAW_TIFF": "00_RAW_TIFF",
        "SQZ_H5": "01A_SQZ_H5",
        "SMP_SQZ_H5": "01B_SMP_SQZ_H5",
        "NORM_MMAP": "02_NORMCORRE_MMAP_OUTPUT",
        "NORM_TFDS_MMAP": "03A_NORMCORRE_TFDS_MMAP",
        "NORM_TFDS_H5": "03B_NORMCORRE_TFDS_H5",
        "NORM_TFDS_AVI": "03C_NORMCORRE_TFDS_CMAP_AVI",
        **{f"SD_{key}": f"04A_SEGDICT_{key}" for key in ["H5", "MAT", "PKL"]},
        "RES_MOV_CCAT_AVI": "04B_RESIDUAL_MOVIE_CDOTA_CONCAT_W_NONCAP_NOISE_AVI",
    }

    return text_dict


def convert_separate2conjoined_string(text: str, pre_text: str = "") -> str:
    """
    Converts a separate string into a conjoined string by capitalizing the first letter of each word and joining them together.

    Parameters:
        text (str): The separate string to be converted.
        pre_text (str, optional): A string to be added before each word in the converted string. Defaults to "".

    Returns:
        str: The converted conjoined string.
    """
    string_converted = pre_text.join(word.capitalize() for word in text.split(" "))
    return string_converted


def print_done_small_proc(new_line: bool = True) -> None:
    """
    Prints the completion message for small processing tasks.

    Parameters:
        new_line (bool): Whether to print a new line after the completion message. Default is True.
    """
    print(text_dict()["completion"]["small_proc"])
    if new_line:
        print()


def create_multiline_string(strings: list) -> str:
    if not strings:
        raise ValueError("Input list is empty in creating multiline string.")
    first_entry = f"{strings[0]}\n"
    remainder = "\n".join(strings[1:])
    return first_entry + remainder


def print_wFrame(
    text: str,
    frame_num: int = 0,
    end: str | None = None,
    flush: bool = False,
    new_line_before: bool = False,
    new_line_after: int = 0,
    color: str | None = None,
) -> None:
    """
    Prints the given text with a frame prepended to it.

    Parameters:
        text (str): The text to be printed.
        frame_num (int, optional): The frame number to be used. Defaults to 0.
        end (str, optional): The string to be appended at the end. Defaults to None.
        flush (bool, optional): Whether to flush the output buffer. Defaults to False.
        new_line_before (bool, optional): Whether to start the frame on a new line. Defaults to False.
        new_line_after (int, optional): The number of new lines to be added after the text. Defaults to 0.
        color (str, optional): The color to be used for the text. Defaults to None.
    """
    # initialize text library
    text_lib = text_dict()
    # get frame characters
    FR, EFR = text_lib["frames"]["FR"], text_lib["frames"]["EFR"]

    # create frames
    FRAME = f"{EFR * frame_num}{FR}"
    # add a new line before the frame
    if new_line_before:
        FRAME = f"\n{FRAME}"

    # add a new line after the text if specified
    blank_line_after = "\n" * new_line_after if new_line_after > 0 else ""

    # create the whole text
    whole_text = f"{FRAME}{text}{blank_line_after}"

    # colorize the text if a color is specified
    if color is not None:
        whole_text = f"[{color}]{whole_text}[/]"

    # print the whole text with the specified end & flush settings
    if end is None:
        print(whole_text, end="\n", flush=flush)
    else:
        print(whole_text, end=end, flush=flush)
