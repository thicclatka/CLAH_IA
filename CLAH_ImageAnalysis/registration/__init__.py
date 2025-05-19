import os

# set order for modules to import
__all__ = [
    "CRwROI_enum",
    "CRwROI_utils",
    "CRwROI_plots",
    "CRwROI_manager",
    "cellRegistrar_wROICaT",
]

# note which ones to import as is
modules_import_as_is = ["CRwROI_enum"]

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from . import CRwROI_enum
    from .cellRegistrar_wROICaT import *
    from .CRwROI_manager import *
    from .CRwROI_plots import *
    from .CRwROI_utils import *
else:
    # import modules accordingly
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
