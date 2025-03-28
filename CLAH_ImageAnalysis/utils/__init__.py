import os

# set order for modules to import
__all__ = [
    "text_formatting",
    "paths",
    "time_utils",
    "debug_utils",
    "wrappers4output",
    "folder_tools",
    "enum_utils",
    "saveNloadUtils",
    "mdata_extractor",
    "parser_utils",
    "Streamlit_utils",
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
    from .text_formatting import *
    from .paths import *
    from .time_utils import *
    from . import debug_utils
    from .wrappers4output import *
    from .folder_tools import *

    # from .array_toolkit import *
    from . import enum_utils
    from . import saveNloadUtils
    from .mdata_extractor import *
    from . import parser_utils
    from . import db_utils
    from . import Streamlit_utils
    from .selector import *
    from . import iter_utils
    from . import image_utils
    from . import caiman_utils
    from . import fig_tools

else:
    # Dynamic imports using exec for flexibility
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
