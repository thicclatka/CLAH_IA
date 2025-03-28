import os

__all__ = [
    "UA_enum",
    "RidgeWalker",
    "wrapMultSessStruc",
    "lapCueDict_utils",
    "sepCueShift_utils",
    "wrapCueShiftTuning",
    "CCF_Dep",
    "CCF_Plotter",
    "CCF_StatTesting",
    "CCF_Utils",
    "CueCellFinder",
    "PCR_CCF_Plotter",
    "PostCR_CueCellFinder",
    "QT_Plotters",
    "pks_utils",
    "QT_manager",
    "quickTuning",
]

# note which ones to import as is
modules_import_as_is = ["UA_enum"]

if os.getenv("STATIC_IMPORTS", "false").lower() == "true":
    from . import UA_enum
    from .RidgeWalker import *
    from .wrapMultSessStruc import *
    from .lapCueDict_utils import *
    from .sepCueShift_utils import *
    from .wrapCueShiftTuning import *
    from .CCF_Dep import *
    from .CCF_Plotter import *
    from .CCF_StatTesting import *
    from .CCF_Utils import *
    from .CueCellFinder import *
    from .PCR_CCF_Plotter import *
    from .PostCR_CueCellFinder import *
    from .QT_Plotters import *
    from .pks_utils import *
    from .QT_manager import *
    from .quickTuning import *
else:
    # import modules accordingly
    for module in __all__:
        if module in modules_import_as_is:
            exec(f"from . import {module}")
        else:
            exec(f"from .{module} import *")
