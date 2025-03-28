import os

# set order for modules to import
__all__ = [
    "decoder_enum",
    "GeneralDecoder",
    "TwoOdorDecoder",
    "ComponentEvaluator",
]

# note which ones to import as is
modules_import_as_is = []

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from .decoder_enum import *
    from .GeneralDecoder import *
    from .TwoOdorDecoder import *
    from .ComponentEvaluator import *
else:
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
