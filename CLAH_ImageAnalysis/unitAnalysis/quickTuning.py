"""
This script defines the `quickTuning` class for performing quick tuning analysis on
unit data, extending the `QT_manager` class. The main functionality includes
importing segmentation data, creating peaks dictionaries, exporting cue shift
structures, and plotting the results.

Functions:
    process_order():    Processes the order by importing segmentation data, initializing
                        and creating peaks dictionaries, exporting cue shift structures,
                        and plotting cue shift tuning.

Classes:
    quickTuning: Extends `QT_manager` to provide methods for quick tuning analysis.

Main Execution:
    If the script is run directly, it will execute the `run_CLAH_script` function to
    create an instance of `quickTuning`, run the parser, and execute the script.

Dependencies:
    - CLAH_ImageAnalysis.core.run_CLAH_script: Function to run the CLAH script.
    - CLAH_ImageAnalysis.unitAnalysis.QT_manager: Base class for quick tuning manager.
    - CLAH_ImageAnalysis.unitAnalysis.UA_enum: Enums for unit analysis.

Usage:
    This script is designed to be executed directly or imported as a module. When run
    directly, it uses the `run_CLAH_script` function to handle argument parsing and
    execution flow.

Example:
    To run the script directly:
    ```bash
    python quickTuning.py --path /path/to/data --sess2process '1,2,3' --frameWindow 15 --sdt 3 --to 3 --overwrite yes
    ```

    To import and use within another script:
    ```python
    from CLAH_ImageAnalysis.unitAnalysis import quickTuning

    qt = quickTuning(path='/path/to/data', sess2process=[1, 2, 3])
    qt.process_order()
    ```

Parser Arguments:
    The script uses the following parser arguments defined in `UA_enum.Parser`:
        --frameWindow, -fw: Window size (in frames) used for smoothing for the event/peak detection (default: 15)
        --overwrite, -ow: Overwrite existing files (default: None)
        --path, -p: Path to the data folder (default: [])
        --sdThresh, -sdt: Threshold multiplier for event/peak detection based on the standard deviation of the signal's derivative (default: 3)
        --sess2process, -s2p: Sessions to process (default: []), can be a list of session numbers or an empty list to prompt user selection.
        --timeout, -to: Minimum distance between detected peaks/events in seconds (default: 3)
"""

from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.unitAnalysis import QT_manager


class quickTuning(QT_manager):
    def __init__(self, **kwargs) -> None:
        self.program_name = "QT"
        QT_manager.__init__(self, program_name=self.program_name, **kwargs)

    def process_order(self) -> None:
        """
        Process the order by importing the segmentation dictionary,
        initializing and creating the peaks dictionary,
        exporting the cue shift structure,
        and plotting the cue shift tuning.

        Returns:
            None
        """
        self.import_segDict()
        self.init_N_create_pksDict()
        self.wrapCueShift2cueShiftStrucExporter()
        self.Plot_cueShiftTuning()


if __name__ == "__main__":
    from CLAH_ImageAnalysis.unitAnalysis import UA_enum

    # run parser, create instance of class, and run the script
    run_CLAH_script(
        quickTuning,
        parser_enum=UA_enum.Parser4QT,
    )
