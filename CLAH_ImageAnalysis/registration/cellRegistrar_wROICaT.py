"""
This script defines the `cellRegistrar_wROICaT` class for registering cells with ROI categorization using the ROICaT module. It extends the `CRwROI_manager` class and provides various methods for initializing, processing, and analyzing data, including loading multi-session structures, running neural networks, clustering, plotting results, and exporting data.

Functions:
    process_order(): Orchestrates the processing order for cell registration and ROI categorization.

Classes:
    cellRegistrar_wROICaT: Extends `CRwROI_manager` to provide methods for cell registration with ROI categorization using the ROICaT algorithm.

Main Execution:
    If the script is run directly, it will execute the `run_CLAH_script` function to create an instance of `cellRegistrar_wROICaT`, run the parser, and execute the script.

Dependencies:
    - CLAH_ImageAnalysis.registration: Provides the `CRwROI_manager` class and `CRwROI_enum` parser.
    - CLAH_ImageAnalysis.core: Provides the `run_CLAH_script` function.

Usage:
    This script is designed to be executed directly or imported as a module. When run directly, it uses the `run_CLAH_script` function to handle argument parsing and execution flow.

Example:
    To run the script directly:
    ```bash
    python cellRegistrar_wROICaT.py --path /path/to/data --sess2process '1,2,3' --useGPU --verbose
    ```

    To import and use within another script:
    ```python
    from CLAH_ImageAnalysis.registration import cellRegistrar_wROICaT

    cr = cellRegistrar_wROICaT(path='/path/to/data', sess2process=[1, 2, 3])
    cr.process_order()
    ```

Parser Arguments:
    The script uses the following parser arguments defined in `CRwROI_enum.Parser`:
        --useGPU, -G: Whether to use GPU (default: True)
        --verbose, -v: Whether to print verbose output (default: True)
        --path, -p: Path to the data folder (default: [])
        --sess2process, -s2p: Sessions to process (default: []), can be a list of session numbers.
"""

from CLAH_ImageAnalysis.registration import CRwROI_manager
from CLAH_ImageAnalysis.core import run_CLAH_script


class cellRegistrar_wROICaT(CRwROI_manager):
    """
    Class for registering cells with ROI categorization using the ROICaT algorithm.

    Parameters:g
        path (str): The path to the data.

    Attributes:
        sess_to_process (list): A list of session numbers to process.
        using_prev_data (bool): Flag indicating whether previous data is being used.

    Methods:
        forLoop_var_init: Initialize variables for the loop.
        load_multSessSegStruc: Load the multSessSegStruc.
        check4previous_data: Check for existence of previous runs.
        ROICaT_0_start: Start the ROICaT algorithm.
        ROICaT_1_runNN: Run the neural network for ROICaT.
        ROICaT_2_clustering: Perform clustering for ROICaT.
        ROICaT_3_plotting: Plot the results of ROICaT.
    """

    def __init__(self, **kwargs) -> None:
        self.program_name = "CRR"
        CRwROI_manager.__init__(
            self,
            program_name=self.program_name,
            **kwargs,
        )

    def process_order(self) -> None:
        """
        Orchestrates the processing order for cell registration and ROI categorization.
        """
        # load multSessSegStruc
        self.load_multSessSegStruc()

        # check for existence of previous runs
        # if previous kwargs exist, given option to use them
        self.check4prev_kwargs()
        # ROICaT funcs
        self.ROICaT_0_start()
        self.ROICaT_1_runNN()
        self.ROICaT_2_clustering()
        self.ROICaT_3_plotting()
        self.ROICaT_4_exportData()


######################################################
#  run script if called from command line
######################################################
if __name__ == "__main__":
    from CLAH_ImageAnalysis.registration import CRwROI_enum

    run_CLAH_script(cellRegistrar_wROICaT, parser_enum=CRwROI_enum.Parser)
