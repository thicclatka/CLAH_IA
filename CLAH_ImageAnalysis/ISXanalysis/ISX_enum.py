from enum import Enum


class Parser4ISX_ANLZR(Enum):
    PARSER = "Parser4ISX_ANLZR"
    HEADER = "ISX analyzer"
    PARSER4 = "ISX_ANLZR"
    PARSER_FN = "ISX_analyzer"
    ARG_DICT = {
        ("fps", "f"): {
            "TYPE": "int",
            "DEFAULT": 10,
            "HELP": "Frames per second of recording. Default is 10.",
        },
        ("window_size", "ws"): {
            "TYPE": "int",
            "DEFAULT": 5,
            "HELP": "Window size (in frames) used for smoothing for the event/peak detection. Default is 5. Assuming 10 Hz, this is 500 ms (200ms before and 200ms after).",
        },
        ("epochCutoff", "ec"): {
            "TYPE": "int",
            "DEFAULT": 2,
            "HELP": "Epoch cutoff (in seconds) used for determining how long freezing/unfreezing must be to be considered an epoch of freezing/unfreezing. Default is 2 seconds.",
        },
        ("pseudoEpochSize", "pes"): {
            "TYPE": "int",
            "DEFAULT": 2,
            "HELP": "Pseudo epoch size (in seconds) used for determining the start and stop of unfreezing epochs. Default is 2 seconds. Using default, the pseudo epoch size is 5 seconds (2 seconds before and 2 seconds after onset of freezing).",
        },
    }
