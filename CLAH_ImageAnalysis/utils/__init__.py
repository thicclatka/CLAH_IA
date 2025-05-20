import os

# set order for modules to import
__all__ = [
    "paths",
    "text_formatting",
    "time_utils",
    "debug_utils",
    "wrappers4output",
    "folder_tools",
    "enum_utils",
    "saveNloadUtils",
    "mdata_extractor",
    "parser_utils",
    "Streamlit_utils",
    "isx_utils",
    "db_utils",
    "selector",
    "iter_utils",
    "image_utils",
    "caiman_utils",
    "fig_tools",
]

# note which ones to import as is
modules_import_as_is = [
    "debug_utils",
    "enum_utils",
    "saveNloadUtils",
    "parser_utils",
    "iter_utils",
    "image_utils",
    "caiman_utils",
    "fig_tools",
]

# Check for a specific environment variable or another condition
if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    # Static imports for better LSP integration
    # from .array_toolkit import *
    from . import (
        Streamlit_utils,
        caiman_utils,
        db_utils,
        debug_utils,
        enum_utils,
        fig_tools,
        image_utils,
        isx_utils,
        iter_utils,
        parser_utils,
        saveNloadUtils,
    )
    from .folder_tools import *
    from .mdata_extractor import *
    from .paths import *
    from .selector import *
    from .text_formatting import *
    from .time_utils import *
    from .wrappers4output import *

else:
    # Dynamic imports using exec for flexibility
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
