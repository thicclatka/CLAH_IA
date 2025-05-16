"""
This module provides functions to parse XML files and extract frame times for 2P imaging data.

Functions:
- get2pFRTimes(folder_path=[]): Retrieves the frame times from the XML file in the specified folder_path.
- importXML(folder_path=[]): Imports the XML file from the specified folder_path and returns the XML file, tree, and root.
- xml2FRtimes(root): Parses the XML root element and extracts the frame times.

Classes:
- xml2FRtimes_Txt(Enum): Enumerates the constants used in the XML parsing process.

"""

from enum import Enum

from lxml import etree

from CLAH_ImageAnalysis import utils


class xml2FRtimes_Txt(Enum):
    """
    Enum class for XML to FR times Constant string.

    Attributes:
        ABS_SEARCH (str): Constant string for absolute time search.
        REL_SEARCH (str): Constant string for relative time search.
        FRIDX_SEARCH (str): Constant string for frame index search.
        ABS_KEY (str): Constant string for absolute frame times key.
        REL_KEY (str): Constant string for relative frame times key.
        FRIDX_KEY (str): Constant string for frame indices key.
        FRAME (str): Constant string for frame.
        ADJ_FRAME (str): Constant string for adjusted frame times.
    """

    ABS_SEARCH = "absoluteTime"
    REL_SEARCH = "relativeTime"
    FRIDX_SEARCH = "index"
    ABS_KEY = "absFrTimes"
    REL_KEY = "relFrTimes"
    FRIDX_KEY = "frInds"
    FRAME = "Frame"
    ADJ_FRAME = "adjFrTimes"


def get2pFRTimes(folder_path: list | str = []) -> tuple:
    """
    Retrieves the frame times from the XML file in the specified folder_path.

    Args:
    - folder_path: The folder_path to the XML file (default=[]).

    Returns:
    - FRdict: A dictionary containing the frame times.
    - xml_file: The XML file.

    """
    folder_path = utils.path_selector(folder_path)
    xml_file, tree, root = importXML(folder_path)
    FRdict = xml2FRtimes(root)

    return FRdict, xml_file


def importXML(folder_path: list | str = []) -> tuple:
    """
    Imports the XML file from the specified folder_path and returns the XML file, tree, and root.

    Args:
    - folder_path: The folder_path to the XML file (default=[]).

    Returns:
    - xml_file: The XML file.
    - tree: The XML tree.
    - root: The root element of the XML tree.

    """
    file_tag = utils.text_dict()["file_tag"]
    utils.folder_tools.chdir_check_folder(folder_path, create=False)

    xml_file = utils.findLatest(file_tag["XML"])
    tree = etree.parse(xml_file)
    root = tree.getroot()

    return xml_file, tree, root


def xml2FRtimes(root) -> dict:
    """
    Parses the XML root element and extracts the frame times.

    Args:
    - root: The root element of the XML tree.

    Returns:
    - FRdict: A dictionary containing the frame times.

    """
    x2F_str = utils.enum_utils.enum2dict(xml2FRtimes_Txt)
    FRdict = {
        "TYPE": "XML",
        x2F_str["ABS_KEY"]: [],
        x2F_str["REL_KEY"]: [],
        x2F_str["FRIDX_KEY"]: [],
    }
    indices = []
    # Iterate over each 'Frame' element in the XML
    for frame in root.iter(x2F_str["FRAME"]):
        # Extract the attributes
        relTime = float(frame.get(x2F_str["REL_SEARCH"], 0))
        absTime = float(frame.get(x2F_str["ABS_SEARCH"], 0))
        idx = int(frame.get(x2F_str["FRIDX_SEARCH"], 0))

        # Store the extracted values in the dictionary
        FRdict[x2F_str["ABS_KEY"]].append(absTime)
        FRdict[x2F_str["REL_KEY"]].append(relTime)
        FRdict[x2F_str["FRIDX_KEY"]].append(idx)
        indices.append(idx)

    if max(indices) == 2:
        utils.print_wFrame("adjusing time arrays for 2 plane data")
        FRdict[x2F_str["ABS_KEY"]] = FRdict[x2F_str["ABS_KEY"]][::2]
        FRdict[x2F_str["REL_KEY"]] = FRdict[x2F_str["REL_KEY"]][::2]
        FRdict[x2F_str["FRIDX_KEY"]] = FRdict[x2F_str["FRIDX_KEY"]][::2]

    return FRdict
