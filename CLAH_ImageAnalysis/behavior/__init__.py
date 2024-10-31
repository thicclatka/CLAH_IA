import os

# set order for modules to import
__all__ = [
    # "TDML2tBD_const",
    "TDML2tBD_enum",
    "behavior_utils",
    "XML2frTimes2P",
    "GPIOfrTimes",
    "TDML2tBD_utils",
    "TDML2treadBehDict",
    "tBD_lD_manager",
]

# note which ones to import as is
modules_import_as_is = [
    # "TDML2tBD_const",
    "TDML2tBD_enum",
    "behavior_utils",
]

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from . import TDML2tBD_enum
    from . import behavior_utils
    from .XML2frTimes2P import *
    from .GPIOfrTimes import *
    from .TDML2tBD_utils import *
    from .TDML2treadBehDict import *
    from .tBD_lD_manager import *
else:
    # import modules accordingly
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
