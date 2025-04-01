import os

# set order for modules to import
__all__ = [
    "runmean",
    "local_minima",
    "find_maxIndNsortmaxInd",
    "normalization_utils",
    "filter_utils",
    "geometric_tools",
    "CUDA_utils",
    # "GPU_SignalProc",
    # "GPU_Tools",
]

# note which ones to import as is
modules_import_as_is = []

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from .runmean import *
    from .local_minima import *
    from .find_maxIndNsortmaxInd import *
    from .filter_utils import *
    from .normalization_utils import *
    from .geometric_tools import *
    from .CUDA_utils import *
    # from .GPU_SignalProc import *
    # from .GPU_Tools import *
else:
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
