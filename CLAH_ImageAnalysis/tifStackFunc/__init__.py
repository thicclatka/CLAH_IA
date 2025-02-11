import os

# set order for modules to import
__all__ = [
    # "TSF_const",
    "TSF_enum",
    "MovieCropper",
    "MoCoPreprocessing",
    "H5_Utils",
    "Movie_Utils",
    "ImageStack_Utils",
    "CNMF_Utils",
    "MotionCorrectFPS10",
    "M2SD_manager",
    "MoCo2segDict",
]

# note which ones to import as is
modules_import_as_is = ["TSF_enum", "Movie_Utils"]

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    # from . import TSF_const
    from . import TSF_enum
    from . import MovieCropper
    from .MoCoPreprocessing import *
    from .H5_Utils import *
    from . import Movie_Utils
    from .ImageStack_Utils import *
    from .CNMF_Utils import *
    from .MotionCorrectFPS10 import *
    from .M2SD_manager import *
    from .MoCo2segDict import *
else:
    # import modules accordingly
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
