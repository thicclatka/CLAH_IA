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
            "HELP": "Frames per second",
        },
        ("window_size", "ws"): {
            "TYPE": "int",
            "DEFAULT": 5,
            "HELP": "Window size (in frames) used for smoothing for the event/peak detection. Default is 5. Assuming 10 Hz, this is 500 ms (200ms before and 200ms after).",
        },
    }
