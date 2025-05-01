import numpy as np


class CCF_Dep:
    @staticmethod
    def findMax_fromTrigSig(
        TrigSig: np.ndarray, ind: list, baseline: bool = True
    ) -> float:
        """
        Finds the maximum value from a given trigger signal.

        Parameters:
            TrigSig (np.ndarray): The trigger signal.
            ind (list): The indices of the trigger signal.
            baseline (bool): If True, subtracts the first value from the maximum value. Default is True.

        Returns:
            maxVal (float): The maximum value from the trigger signal.

        Example:
        >>> TrigSig = [1, 2, 3, 4, 5]
        >>> ind = [0, 1, 2, 3, 4]
        >>> findMax_fromTrigSig(TrigSig, ind)
        4
        """
        start_idx = np.abs(ind[0])
        end_idx = start_idx + ind[-1] + 1
        maxVal = max(TrigSig[start_idx:end_idx])
        if baseline:
            maxVal = maxVal - TrigSig[start_idx]
        return maxVal

    @staticmethod
    def findAUC_fromTrigSig(TrigSig: np.ndarray, ind: list) -> float:
        """
        Finds the Area Under the Curve (AUC) - calculated as the sum of raw signal values -
        within the post-event window.

        Parameters:
            TrigSig (np.ndarray): The trigger signal snippet (pre and post event). NaNs indicate times outside the valid recording period.
            ind (list): The indices defining the window relative to the event (e.g., [-10, 20]).

        Returns:
            auc (float): The raw AUC (sum of values). Returns NaN if signal is all NaN in the window.
        """
        if np.all(np.isnan(TrigSig)):
            return np.nan

        pre_event_len = abs(ind[0])
        # Define indices for the response (post-event) period
        response_indices = slice(pre_event_len, pre_event_len + (ind[-1] - 29))

        # Calculate raw AUC (sum of values in the response window, ignoring NaNs)
        auc = np.nansum(TrigSig[response_indices])

        return auc

    @staticmethod
    def bootstrap2findSTD(
        cue1: np.ndarray,
        cue2: np.ndarray,
        n_samples: int,
        n_iterations: int = 1000,
    ) -> float:
        """
        Calculate the standard deviation of the bootstrap differences between two sets of cues.

        Parameters:
            cue1 (np.ndarray): The first set of cues.
            cue2 (np.ndarray): The second set of cues.
            n_samples (int): The number of samples to be resampled from each set of cues.
            n_iterations (int): The number of iterations to perform the bootstrap resampling. Default is 1000.

        Returns:
            stdDiff (float): The standard deviation of the bootstrap differences.
        """

        bootstrap_diffs = []
        for _ in range(n_iterations):
            resampled_cue1 = np.random.choice(cue1, size=n_samples, replace=True)
            resampled_cue2 = np.random.choice(cue2, size=n_samples, replace=True)

            resampled_meanCue = np.nanmean(resampled_cue1)
            resampled_meanCwO = np.nanmean(resampled_cue2)
            resampled_diff = resampled_meanCwO - resampled_meanCue

            bootstrap_diffs.append(resampled_diff)
        stdDiff = np.nanstd(bootstrap_diffs)
        return stdDiff

    @staticmethod
    def calcInd4evTrigSig(
        sigTime: np.ndarray, evTime: float, ind: list, len_Ca_arr: int
    ) -> tuple:
        """
        Calculate the start and end indices for a given event-triggered signal.

        Parameters:
            sigTime (np.ndarray): Array of signal timestamps.
            evTime (float): Event timestamp.
            ind (np.ndarray): Array of indices.
            len_Ca_arr (int): Length of the Ca array.

        Returns:
            start_idx (int): Start index for the slice.
            end_idx (int): End index for the slice.
            slice_length (int): Length of the slice.

        """
        nrstInd = np.argmin(np.abs(sigTime - evTime))
        start_idx = max(nrstInd + ind[0], 0)
        end_idx = min(nrstInd + ind[-1], len_Ca_arr)
        slice_length = end_idx - start_idx
        return start_idx, end_idx, slice_length

    @staticmethod
    def classify_cellType(cell: int, CueCellTable: dict) -> str:
        """
        Classify the cell type based on the cell number.

        Parameters:
            cell (str): The cell number.
            CueCellTable (dict): The table of cue and non-cue cells.

        Returns:
            cellType (str): The type of cell.
        """

        cell_num = int(cell.split("_")[-1])
        if cell_num in CueCellTable["MID_IDX"]:
            cellType = "MID"
        elif cell_num in CueCellTable["CUE_IDX"]:
            cellType = "CUE"
        else:
            cellType = "NON"
        return cellType
