import os

# set order for modules to import
__all__ = ["segDictWA", "segDictGUI"]

# note which ones to import as is
modules_import_as_is = []


if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from .GUI_Utils import *
    from .BasicGUIStruc import *
else:
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
