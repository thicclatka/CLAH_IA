import os

# set order for modules to import
__all__ = ["ISX_converter"]

# note which ones to import as is
modules_import_as_is = []

# Check for a specific environment variable or another condition
if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from .ISX_converter import *
else:
    # Dynamic imports using exec for flexibility
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
