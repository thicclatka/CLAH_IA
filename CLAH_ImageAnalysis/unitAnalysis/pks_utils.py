import numpy as np
import pandas as pd
from CLAH_ImageAnalysis.dependencies import local_minima
from CLAH_ImageAnalysis.dependencies import runmean
from CLAH_ImageAnalysis.unitAnalysis import QT_Plotters


class pks_utils:
    """
    A utility class for peak analysis of calcium imaging data.

    Parameters:
        frameWindow (int): Window size for running mean. Defaults to 15.
        sdThresh (int): Threshold value for peak detection. Defaults to 3.
        timeout (int): Timeout value for peak detection. Defaults to 3.
        sess_name (str): Name of the session. Defaults to None.

    Methods:
        find_CaTransients(Ca_arr, first_pt=30, toPlot=False, **kwargs): Find calcium transients in the given calcium array.
        _peak_processor(Ca_arr, peaks): Process the detected peaks.
        zScoreCa(Ca_arr): Calculate the z-score of the calcium array.
    """

    def __init__(
        self,
        frameWindow: int = 15,
        sdThresh: int = 3,
        timeout: int = 3,
        sess_name: str = None,
    ):
        """
        For defaults, see UA_enum.Parser.

        Parameters:
            sess_name (str, optional): The name of the session. Defaults to None.
            frameWindow (int, optional): Window size for running mean. Defaults to 15.
            sdThresh (int, optional): The threshold value for peak detection. Defaults to 3.
            timeout (int, optional): The timeout value for peak detection. Defaults to 3.
        """
        self.frameWindow = frameWindow
        self.sdThresh = sdThresh
        self.timeout = timeout
        if sess_name is not None:
            self.plotting = QT_Plotters(sess_name=sess_name)

    def find_CaTransients(
        self, Ca_arr: np.ndarray, first_pt: int = 30, toPlot: bool = False, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find calcium transients in the given calcium array.

        Parameters:
            Ca_arr (numpy.ndarray): Array of calcium values.
            first_pt (int, optional): First point to set for the array. Defaults to 30.
            toPlot (bool, optional): Whether to plot the results. Defaults to False.
            **kwargs: Additional keyword arguments.
                cell_num (int, optional): The cell number to plot. Defaults to None.

        Returns:
            tuple: A tuple containing the detected peaks, amplitudes, and waveforms.
        """

        # window = self.frameWindow * 2
        # beginning is weird, so set to first good point
        # first pt = double frameWindow
        Ca_arr[:first_pt] = [Ca_arr[first_pt]] * first_pt
        # averages over frames, reduce impact of short noisy activity
        running_mean = runmean(Ca_arr, self.frameWindow)
        dCa_arr = np.diff(running_mean)
        #  threshold in stdev
        threshold = self.sdThresh * np.nanstd(dCa_arr)
        peaks = local_minima(-dCa_arr, self.timeout * self.frameWindow, -threshold)
        peaks = [int(peak) for peak in peaks]

        for j in range(3):  # Iterative re-baselining
            for i in peaks:
                i = int(i)
                start = max(i - 3 * self.frameWindow, 0)
                end = min(i + 2 * self.frameWindow, len(dCa_arr))
                # Ensure start and end are integers
                start, end = int(start), int(end)
                dCa_arr[start : end + 1] = np.nan

            # Recompute threshold w/ updated dCa_arr
            threshold = self.sdThresh * np.nanstd(dCa_arr)
            pks2 = local_minima(-dCa_arr, self.timeout * self.frameWindow, -threshold)
            peaks = np.sort(
                np.unique(np.concatenate((peaks, pks2)))
            )  # Combine and sort peaks

        pkInds, amps, waveform = self._peak_processor(Ca_arr, peaks)

        if toPlot:
            cell_num = kwargs.get("cell_num", None)
            self.plotting.pks_amps_wavef(Ca_arr, peaks, pkInds, amps, cell_num)

        return peaks, amps, waveform

    def _peak_processor(
        self, Ca_arr: np.ndarray, peaks: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process the detected peaks.

        Parameters:
            Ca_arr (numpy.ndarray): Array of calcium values.
            peaks (list): List of detected peaks.

        Returns:
            tuple: A tuple containing the peak indices, amplitudes, and waveforms.

        """
        # Process peaks
        maxs = np.zeros(len(peaks))
        pkInds = np.zeros(len(peaks), dtype=int)
        mins = np.zeros(len(peaks))
        amps = np.zeros(len(peaks))
        waveform_length = 401  # Length based on -100 to +300 range
        waveforms = np.zeros((len(peaks), waveform_length))

        for i, pk in enumerate(peaks):
            pk = int(pk)
            try:
                timeoutXframeWindow = self.timeout * self.frameWindow
                timeout_frames = int(round(timeoutXframeWindow))
                peak_range = int(pk + timeout_frames)

                # Ensure range does not exceed the bounds of the array
                peak_range = min(peak_range, len(Ca_arr))

                # Find max & index in the specified range
                max_val = np.max(Ca_arr[pk:peak_range])
                maxs[i] = max_val
                inds = np.argmax(Ca_arr[pk:peak_range])
                pkInds[i] = int(inds + pk)

                # Find min in the specified range
                min_start = int(
                    max(pk - round(timeoutXframeWindow / 2), 0)
                )  # Ensure min_start is an integer
                mins[i] = np.min(Ca_arr[min_start:pk])

                # Calculate amplitude
                amps[i] = maxs[i] - mins[i]

                # Extract waveform
                waveform_start = int(
                    max(pk - 100, 0)
                )  # Ensure waveform_start is an integer
                waveform_end = int(
                    min(pk + 300, len(Ca_arr))
                )  # Ensure waveform_end is an integer

                waveforms[i, : waveform_end - waveform_start] = Ca_arr[
                    waveform_start:waveform_end
                ]

            except Exception as e:
                print(f"Error at index {i} ({pk}): {e}")

        return pkInds, amps, waveforms

    def zScoreCa(self, Ca_arr: np.ndarray) -> np.ndarray:
        """
        Calculate the z-score of the calcium array.

        Parameters:
            Ca_arr (numpy.ndarray): Array of calcium values.

        Returns:
            numpy.ndarray: Array of z-scored calcium values.
        """

        pks, _, _ = self.find_CaTransients(Ca_arr)
        means = np.full(len(pks), np.nan)
        std = np.full(len(pks), np.nan)
        for idx, pk in enumerate(pks):
            pk = int(pk)
            if pk < 299:
                start_idx = 0
                end_idx = pk
            else:
                start_idx = max(pk - 299, 0)
                end_idx = pk - 99
            epochCa = Ca_arr[start_idx:end_idx]
            means[idx] = np.mean(epochCa)
            std[idx] = np.std(epochCa)

        mask = std != 0
        std_nonzero = std[mask]
        means_nonzero = means[mask]

        N, edges = np.histogram(std_nonzero, bins=10)

        # Find indices of the elements in the lowest std bin
        bin_indices = np.digitize(std_nonzero, edges)
        lowest_std_bin = bin_indices == 1

        # Calculate mean and std of epochMean for the lowest std bin
        caMeanBaseline = np.mean(means_nonzero[lowest_std_bin])
        caStdBaseline = np.mean(std_nonzero[lowest_std_bin])

        Ca_arr_Z = (Ca_arr - caMeanBaseline) / caStdBaseline

        return Ca_arr_Z

    # @staticmethod
    # def normalizeByBaseline(
    #     Ca_arr: np.ndarray,
    #     scale_window: int = 300,
    #     quantile: float = 0.08,
    #     log: bool = False,
    # ) -> np.ndarray:
    #     baseline = (
    #         pd.DataFrame(Ca_arr)
    #         .T.rolling(scale_window, center=True, min_periods=0)
    #         .quantile(quantile)
    #         .values.T
    #     )
    #     return Ca_arr - baseline if log else Ca_arr / baseline

    @staticmethod
    def normalizeByBaseline(
        Ca_arr: np.ndarray,
        scale_window: int = 500,
        quantile: float = 0.05,
        log: bool = False,
        shiftMin2Zero: bool = False,
    ) -> np.ndarray:
        # Calculate the rolling quantile using NumPy
        def rolling_quantile(arr, window, quantile):
            """Calculate rolling quantile using a sliding window."""
            # Prepare an array to hold the quantile values
            quantiles = np.empty(len(arr))
            half_window = window // 2
            padded = np.pad(arr, (half_window, half_window), mode="edge")

            # Calculate the rolling quantile
            for i in range(len(arr)):
                start = i
                end = i + window
                quantiles[i] = np.quantile(padded[start:end], quantile)

            return quantiles

        if shiftMin2Zero:
            Ca_arr = Ca_arr - np.min(Ca_arr)
        # Calculate the baseline using the rolling quantile
        baseline = rolling_quantile(Ca_arr, scale_window, quantile)

        # Return the normalized array
        return Ca_arr - baseline if log else Ca_arr / baseline

    @staticmethod
    def applySGFilter(
        Ca_arr: np.ndarray,
        window_size: int = 11,
        smoothing_order: int = 3,
        log: bool = False,
    ) -> np.ndarray:
        from scipy.signal import savgol_filter

        if log:
            Ca_arr2use = np.log(Ca_arr - np.min(Ca_arr)) + 1
        else:
            Ca_arr2use = Ca_arr

        filtered_Ca = savgol_filter(
            x=Ca_arr2use, window_length=window_size, polyorder=smoothing_order
        )

        if log:
            max_val = np.nanmax(filtered_Ca[~np.isinf(filtered_Ca)])
            min_val = np.nanmin(filtered_Ca[~np.isinf(filtered_Ca)])
            filtered_Ca = np.clip(filtered_Ca, min_val, max_val)
        return filtered_Ca

    @staticmethod
    def find_pksViaScipy(
        Ca_arr: np.ndarray,
        height: float = 0.1,
        distance: int = 1,
        prominence: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find peaks in the given calcium array using scipy.signal.find_peaks. Recommended to adjust Ca_arr to be noramlized or adjusted to baseline.

        Parameters:
            Ca_arr (numpy.ndarray): Array of values of Ca2+ signal for a single cell.
            height (float, optional): The minimum height of a peak from baseline based on units of the signal. Defaults to 0.1.
            distance (int, optional): The minimum distance between peaks. Defaults to 1.
            prominence (float, optional): The minimum prominence of a peak given distance peak is from nearest valley, in units of the signal. Defaults to 0.15.

        Returns:
            tuple: A tuple containing the peak indices and a dictionary of peak properties.
        """
        from scipy.signal import find_peaks

        return find_peaks(
            Ca_arr, height=height, distance=distance, prominence=prominence
        )
