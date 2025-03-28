import os

# set order for modules to import
__all__ = ["ISX_enum", "ISX_folderUtils", "ISX_converter", "ISX_analyzer"]

# note which ones to import as is
modules_import_as_is = []

# Check for a specific environment variable or another condition
if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from .ISX_enum import *
    from .ISX_folderUtils import *
    from .ISX_converter import *
    from .ISX_analyzer import *
else:
    # Dynamic imports using exec for flexibility
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
