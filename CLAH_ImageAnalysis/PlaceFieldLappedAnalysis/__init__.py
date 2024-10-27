import os

# set order for modules to import
__all__ = [
    "PCLA_enum",
    "PCLA_dependencies",
    "PCLA_utils",
    "PlaceCellsLappedAnalysis",
]

# note which ones to import as is
modules_import_as_is = ["PCLA_enum"]

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from . import PCLA_enum
    from .PCLA_dependencies import *
    from .PCLA_utils import *
    from .PlaceCellsLappedAnalysis import *
else:
    # import modules accordingly
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
