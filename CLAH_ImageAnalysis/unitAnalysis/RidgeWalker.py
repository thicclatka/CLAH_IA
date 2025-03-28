import numpy as np
from scipy.ndimage import maximum_filter
import pywt
from CLAH_ImageAnalysis.core import BaseClass as BC

# from CLAH_ImageAnalysis.dependencies import runmean


class RidgeWalker(BC):
    """
    RidgeWalker Class for Detecting and Analyzing Ridges in Calcium Imaging Data.

    This class implements a ridge detection algorithm using Continuous Wavelet Transform (CWT) to identify
    and analyze temporal features in calcium imaging time-series data. It utilizes a Morse wavelet with
    user-defined parameters for shape and oscillation properties (`beta` and `gamma`). The class supports
    identifying significant ridges, linking ridges across scales, and visualizing both the detected ridges
    and their respective peaks.

    Attributes:
        cell_num (int): Cell identifier for the current calcium trace.
        Ca_arr (numpy.ndarray): 1D array of calcium trace values for a single cell.
        total_experiment_time (int): Total number of time points in the calcium trace.
        wavelet_name (str): Morse wavelet name constructed using the `beta` and `gamma` parameters.
        scales (numpy.ndarray): Array of scales for the CWT.
        min_scale_length (int): Minimum number of scales a ridge must span to be considered significant.
        neighborhood_size (tuple): Size of the neighborhood window used to detect local maxima in the CWT.
        window_size (int): Temporal window size used to link ridges across scales.
        RWparams (dict): Dictionary storing the RidgeWalker parameters.
        output_folder (str): Directory path to store output plots and saved parameters.
        srMarker (str): Marker style for plotting significant ridges.
        srLineStyle (str): Line style for plotting significant ridges.
        srMarkerSize (int): Marker size for plotting significant ridges.
        pkColor (str): Color used to plot peak markers.
        pkMarker (str): Marker style for plotting peaks.
        pkMarkerSize (int): Marker size for plotting peaks.
        linked_ridges (list): List of ridges after linking across scales.
        sigRidges (list): List of significant ridges after filtering based on `min_scale_length`.
        peakTimes (list): List of time indices representing the peak times of each significant ridge.

    Methods:
        __init__(self, cell_num, Ca_arr, beta, gamma, window_size, min_scale_length, total_scales=100, neighborhood_size=(3, 3)):
            Initializes the RidgeWalker class with user-specified parameters and computes the CWT.

        _figure_var_setup(self):
            Sets up default plotting variables for significant ridge and peak visualization.

        _compute_CWT(self):
            Computes the Continuous Wavelet Transform (CWT) of the calcium array and detects local maxima.

        _find_ridges(self, local_max_mask):
            Identifies initial ridges from the local maxima at the largest scale of the CWT.

        _link_ridges(self, ridges, local_max_mask, cell_cwt):
            Links ridges across scales to form continuous paths, considering local maxima in a defined temporal window.

        _find_sigRidgesNpeakTimes(self, ridges, cell_cwt):
            Filters ridges based on minimum length and finds the time index of the maximum amplitude (peak) for each significant ridge.

        _plot_sigRidgesNpeaks(self, sigRidges, peakTimes, cell_cwt):
            Plots the significant ridges and peak points over the CWT matrix.

        showPeaksOverTrace(self):
            Creates an interactive Plotly plot to overlay detected peak points on the original calcium trace.

        export_params2JSON(self):
            Exports the current RidgeWalker parameters to a JSON file.

    Usage:
        >>> from CLAH_ImageAnalysis.core import BaseClass as BC
        >>> Ca_trace = np.random.rand(9000)  # Example calcium trace for one cell
        >>> rw = RidgeWalker(
                cell_num=0,
                Ca_arr=Ca_trace,
                beta=2,
                gamma=3,
                window_size=10,
                min_scale_length=10
            )
        >>> rw.showPeaksOverTrace()  # Visualize peaks over the calcium trace
        >>> rw.export_params2JSON()  # Export the parameters to a JSON file
    """

    def __init__(
        self,
        cell_num: int,
        Ca_arr: np.ndarray,
        beta: int,
        gamma: int,
        window_size: int,
        min_scale_length: int,
        total_scales: int,
        neighborhood_size: tuple,
        sigRidge_scaleMin: int = 10,
    ) -> None:
        """
        Initializes the RidgeWalker class with the given parameters.

        Parameters:
            cell_num (int): The index of the cell or segment being analyzed.
            Ca_arr (numpy.ndarray): 1D array representing the calcium trace (time series) for a single cell.
            beta (float): Morse wavelet parameter controlling the oscillatory nature of the wavelet. Higher values emphasize high-frequency components.
            gamma (float): Morse wavelet parameter controlling the symmetry and compactness of the wavelet. Higher values result in more symmetric wavelets with tighter temporal localization.
            window_size (int): Temporal window size for linking local maxima across scales during ridge linking.
            min_scale_length (int): Minimum number of scales a ridge must span to be considered significant.
            total_scales (int, optional): Total number of scales to use for the wavelet transform. Defaults to 100.
            neighborhood_size (tuple, optional): Size of the neighborhood window for detecting local maxima. Defaults to (3, 3).
            sigRidge_scaleMin (int, optional): Minimum scale index a ridge must span to be considered significant. Defaults to 10.
        """

        self.program_name = "RidgeWalker"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.cell_num = cell_num
        self.Ca_arr = Ca_arr
        # self.Ca_arr = runmean(Ca_arr, 15)
        self.total_experiment_time = self.Ca_arr.shape[-1]

        self.wavelet_name = f"cmor{beta}-{gamma}"
        self.scales = np.arange(1, total_scales)
        self.min_scale_length = min_scale_length
        self.neighborhood_size = neighborhood_size
        self.window_size = window_size
        self.min_scale_length = min_scale_length
        self.sigRidge_scaleMin = sigRidge_scaleMin

        self.RWparams = {
            "beta": beta,
            "gamma": gamma,
            "wavelet_name": self.wavelet_name,
            "window_size": window_size,
            "min_scale_length": min_scale_length,
            "total_scales": total_scales,
            "neighborhood_size": neighborhood_size,
        }

        self._figure_var_setup()

        curr_cell_cwt, curr_local_max_mask = self._compute_CWT()

        curr_ridges = self._find_ridges(curr_local_max_mask)
        self.linked_ridges = self._link_ridges(
            curr_ridges, curr_local_max_mask, curr_cell_cwt
        )
        self.sigRidges, self.peakTimes = self._find_sigRidgesNpeakTimes(
            self.linked_ridges, curr_cell_cwt
        )

        self._plot_sigRidgesNpeaks(self.sigRidges, self.peakTimes, curr_cell_cwt)

    def _figure_var_setup(self) -> None:
        """
        Sets up default plotting variables for significant ridge and peak visualization.
        """

        self.output_folder = "RIDGE"
        self.fig_save_path = f"Figures/{self.output_folder}"

        self.srMarker = "o"
        self.srLineStyle = "-"
        self.srMarkerSize = 2

        self.pkColor_CWT = self.color_dict["black"]
        self.pkColor_CA = self.color_dict["red"]
        self.pkMarker = "x"
        self.pkMarkeronIMSHOW = 50
        self.pkMarkerSize = 10

        self.rdgSize = 4

        self.colors2use = [color for color in self.color_dict.values()]

    def _compute_CWT(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the continuous wavelet transform (CWT) of the given calcium array.

        This method performs the CWT on the calcium array using specified scales and wavelet name.
        It then calculates the absolute values of the wavelet coefficients and identifies local maxima
        within a specified neighborhood size.

            tuple:
                numpy.ndarray: The absolute values of the wavelet coefficients.
                numpy.ndarray: A boolean mask indicating the local maxima in the wavelet coefficients.
        """

        coefficients, frequencies = pywt.cwt(
            self.Ca_arr, self.scales, self.wavelet_name
        )
        cell_cwt = np.abs(coefficients)
        local_max_mask = (
            maximum_filter(cell_cwt, size=self.neighborhood_size) == cell_cwt
        )
        # Set boundary regions to False to ignore them
        local_max_mask[:, : self.neighborhood_size[1] // 2] = False  # Ignore left edge
        local_max_mask[:, -(self.neighborhood_size[1] // 2) :] = (
            False  # Ignore right edge
        )

        return cell_cwt, local_max_mask

    def _find_ridges(self, mask2findRidges: np.ndarray) -> list:
        """
        Identify ridges in the local maxima mask.

        This method processes a mask of local maxima and identifies ridges,
        which are stored as a list of tuples containing the scale and time index.

        Parameters:
            local_max_mask (numpy.ndarray): A 2D array where the last row contains
                            boolean values indicating the presence
                            of local maxima.

        Returns:
            list:   A list of ridges, where each ridge is represented as
                    a list of tuples (scale, time index).
        """

        ridges = []  # Store ridges as a list of tuples (scale, time index)

        for scale in reversed(range(mask2findRidges.shape[0])):
            for time_idx in np.where(mask2findRidges[scale, :])[0]:
                ridges.append([(len(self.scales) - 1, time_idx)])

        return ridges

    def _link_ridges(
        self, ridges: list, local_max_mask: np.ndarray, cell_cwt: np.ndarray
    ) -> list:
        """
        Link ridges to form continuous paths.

        This method processes a list of ridges and links them to form continuous paths.
        It returns a list of paths, where each path is a list of tuples (scale, time index).

        Parameters:
            ridges (list):  A list of ridges, where each ridge is represented as a list of
                            tuples (scale, time index).
            local_max_mask (numpy.ndarray): A 2D array where the last row contains
                            boolean values indicating the presence
                            of local maxima.
            cell_cwt (numpy.ndarray): The absolute values of the wavelet coefficients.

        Returns:
            list: A list of paths, where each path is a list of tuples (scale, time index).
        """

        ridges2link = ridges.copy()
        for s_idx, scale in enumerate(
            reversed(range(len(self.scales) - 1))
        ):  # Start from second-largest scale
            # self.print_wFrm(f"Scales: {scale} of {len(self.scales)}")
            new_ridges = []  # Store updated ridges
            # Track indices already linked to prevent overwriting
            existing_time_indices = set()

            for ridge in ridges2link:
                current_scales = [scale_val for scale_val, _ in ridge]

                if s_idx > 1 and scale + 1 not in current_scales:
                    # If the previous scale is missing, the ridge has a gap -> skip
                    continue

                # Get the last point in the current ridge
                last_scale, last_time_idx = ridge[-1]

                # Determine the temporal window around last point
                min_time_idx = max(0, last_time_idx - self.window_size)
                max_time_idx = min(
                    self.total_experiment_time - 1, last_time_idx + self.window_size
                )

                # Find local maxima in current scale within the temporal window
                current_maxima = np.where(
                    local_max_mask[scale, min_time_idx : max_time_idx + 1]
                )[0]

                if len(current_maxima) > 0:
                    # Link the largest maximum to the current ridge
                    best_max_idx = current_maxima[
                        np.argmax(
                            cell_cwt[scale, min_time_idx : max_time_idx + 1][
                                current_maxima
                            ]
                        )
                    ]
                    new_time_idx = min_time_idx + best_max_idx

                    # Extend the current ridge if the time index is not already linked
                    if new_time_idx not in existing_time_indices:
                        new_ridges.append(ridge + [(scale, new_time_idx)])
                        existing_time_indices.add(new_time_idx)
                    else:
                        new_ridges.append(ridge)
                else:
                    new_ridges.append(ridge)

            # Update ridges for the next iteration
            ridges2link = new_ridges

        return ridges2link

    def _find_sigRidgesNpeakTimes(
        self, ridges: list, cell_cwt: np.ndarray
    ) -> tuple[list, dict]:
        sigRidges = [
            ridge
            for ridge in ridges
            if len(ridge) >= self.min_scale_length
            and max(ridge, key=lambda pt: cell_cwt[pt[0], pt[1]])[0]
            > self.sigRidge_scaleMin
            # and max(ridge, key=lambda point: self.Ca_arr[point[1]])[0]
            # > self.sigRidge_scaleMin
        ]

        peakTimes = {"CWT": [], "Ca": []}
        for ridge in sigRidges:
            # Find the time index with the maximum amplitude along the ridge
            peak_time_idx_cwt = max(
                ridge, key=lambda point: cell_cwt[point[0], point[1]]
            )[1]
            peakTimes["CWT"].append(peak_time_idx_cwt)

            peak_time_idx_CA = max(ridge, key=lambda point: self.Ca_arr[point[1]])[1]
            peakTimes["Ca"].append(peak_time_idx_CA)

        return sigRidges, peakTimes

    def _plot_sigRidgesNpeaks(
        self, sigRidges: list, peakTimes: dict, cell_cwt: np.ndarray
    ) -> None:
        """
        Plot significant ridges and peaks over the CWT matrix.

        Parameters:
            sigRidges (list): A list of significant ridges.
            peakTimes (dict): A dictionary containing the peak times for the CWT and Ca2+ trace.
            cell_cwt (numpy.ndarray): The absolute values of the wavelet coefficients.
        """

        fig, axis2plot = self.fig_tools.create_plt_subplots()
        extent = [0, self.total_experiment_time, self.scales[0], self.scales[-1]]

        self.fig_tools.plot_imshow(
            fig=fig,
            axis=axis2plot,
            data2plot=cell_cwt[::-1, :],
            aspect="auto",
            extent=extent,
        )

        for r_idx, (ridge, pkCWT, pkCA) in enumerate(
            zip(sigRidges, peakTimes["CWT"], peakTimes["Ca"])
        ):
            scl2plot, time_indices = zip(*ridge)
            axis2plot.plot(
                time_indices,
                scl2plot,
                marker=self.srMarker,
                linestyle=self.srLineStyle,
                markersize=self.srMarkerSize,
                color=self.colors2use[sigRidges.index(ridge) % len(self.colors2use)],
            )

            peak_scale_idx_cwt = scl2plot[time_indices.index(pkCWT)]
            axis2plot.scatter(
                [pkCWT],
                [peak_scale_idx_cwt],
                color=self.pkColor_CWT,
                marker=self.pkMarker,
                s=self.pkMarkeronIMSHOW,
            )

            peak_scale_idx_ca = scl2plot[time_indices.index(pkCA)]
            axis2plot.scatter(
                [pkCA],
                [peak_scale_idx_ca],
                color=self.pkColor_CA,
                marker=self.pkMarker,
                s=self.pkMarkeronIMSHOW,
            )

        axis2plot.set_xlabel("Time (frames)")
        axis2plot.set_ylabel("Scales")
        axis2plot.set_title(f"Significant Ridges for Seg {self.cell_num}")
        # fig.colorbar(label="CWT Coefficient Magnitude")

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name=f"sigRidges_seg{self.cell_num}",
            figure_save_path=f"{self.fig_save_path}/CWT",
        )

    def showPeaksOverTrace(self) -> None:
        """
        Create a Plotly figure to overlay detected peak points on the original calcium trace.
        """

        fig = self.fig_tools.create_plotly()

        self.fig_tools.add_plotly_trace(
            fig=fig,
            x=list(range(len(self.Ca_arr))),
            y=self.Ca_arr,
            mode="lines",
            line_color=self.color_dict["blue"],
            name="Ca2+ Trace",
        )

        # self.fig_tools.add_plotly_trace(
        #     fig=fig,
        #     x=self.peakTimes,
        #     y=self.Ca_arr[self.peakTimes],
        #     mode="markers",
        #     marker=dict(
        #         color=self.pkColor, symbol=self.pkMarker, size=self.pkMarkerSize
        #     ),
        #     name="Peaks",
        # )

        # Add traces for each significant ridge
        for i, ridge in enumerate(self.sigRidges):
            # Separate scale and time indices
            scl2plot, time_indices = zip(*ridge)
            # Get the amplitude values from `Ca_arr` for the ridge time indices
            ridge_amplitude = self.Ca_arr[list(time_indices)]

            # Add each ridge as a separate trace
            self.fig_tools.add_plotly_trace(
                fig=fig,
                x=list(time_indices),
                y=ridge_amplitude,
                # mode="lines+markers",
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color=self.colors2use[i % len(self.colors2use)],
                    size=self.rdgSize,
                ),
                # line=dict(dash="dash"),
                name=f"Ridge {i + 1}",
                line_color=self.colors2use[i % len(self.colors2use)],
            )

            # self.fig_tools.add_plotly_trace(
            #     fig=fig,
            #     x=time_indices,
            #     y=scl2plot,
            #     mode="lines",
            #     marker=dict(size=1, color=self.color_dict["red"]),
            #     yaxis="y2",  # Use the secondary y-axis
            #     # name=f"Ridge Points {i + 1}",
            # )

        # Set the axis labels and title
        fig.update_layout(
            xaxis_title="Time (frames)",
            yaxis=dict(
                title="Ca2+ Amp",
                showgrid=True,
            ),
            # yaxis2=dict(
            #     title="Scale",
            #     overlaying="y",  # Overlay onto the primary y-axis
            #     side="right",  # Place it on the right side
            #     showgrid=False,
            # ),
            title=f"Peaks and Ridges for Seg {self.cell_num} via RidgeWalker",
        )

        # Save the figure as an HTML file
        self.fig_tools.save_plotly(
            plotly_fig=fig,
            figure_save_path=f"{self.fig_save_path}/Peaks",
            fig_name=f"Peaks_seg{self.cell_num}",
        )

    def export_params2JSON(self) -> None:
        """
        Export the current RidgeWalker parameters to a JSON file.
        """

        self.savedict2file(
            dict_to_save=self.RWparams,
            dict_name="RidgeWalkerAlgoParams",
            filename=f"{self.fig_save_path}/RidgeWalkerAlgoParams",
            filetype_to_save=self.file_tag["JSON"],
        )
