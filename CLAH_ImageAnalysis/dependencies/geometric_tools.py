import numpy as np


class geometric_tools:
    def __init__(self):
        pass

    @staticmethod
    def find_center_of_mass(
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        arr_pts: np.ndarray | None = None,
        COM_type: str = "mean",
    ):
        """
        Calculate the center of mass for a set of points.

        Parameters:
        - rows (array-like, optional): The row coordinates of the points.
        - cols (array-like, optional): The column coordinates of the points.
        - arr_pts (array-like, optional): An array of points in the form [(row1, col1), (row2, col2), ...].
        - COM_type (str, optional): The type of center of mass calculation to perform. Can be "mean" or "median".

        Returns:
        - center_of_mass (ndarray): The calculated center of mass as a numpy array of shape (2,).

        Note:
        - If `rows` and `cols` are provided, `arr_pts` will be automatically generated from them.
        - If `COM_type` is "mean", the mean of the points will be calculated as the center of mass.
        - If `COM_type` is "median", the median of the points will be calculated as the center of mass.
        """
        if rows is not None and cols is not None and arr_pts is None:
            arr_pts = np.array(list(zip(rows, cols)))
        if COM_type == "mean":
            return np.mean(arr_pts, axis=0)
        elif COM_type == "median":
            return np.median(arr_pts, axis=0)

    @staticmethod
    def find_min_dist_from_COM(COM: tuple, arr_pts: np.ndarray) -> float:
        """
        Calculates the minimum distance from the center of mass (COM) to a set of points.

        Parameters:
        - COM (tuple): The coordinates of the center of mass in the format (row, column).
        - arr_pts (numpy.ndarray): An array of points in the format (row, column).

        Returns:
        - min_dist (float): The minimum distance from the COM to any of the points.
        """
        # extract needed rows & columns
        COM_row, COM_col = COM

        min_row, min_col = np.min(arr_pts, axis=0)
        max_row, max_col = np.max(arr_pts, axis=0)

        # find distances
        d2min_row = COM_row - min_row
        d2min_col = COM_col - min_col
        d2max_row = max_row - COM_row
        d2max_col = max_col - COM_col

        # find minimum distance
        min_dist = np.min([d2min_row, d2min_col, d2max_row, d2max_col])
        return min_dist

    @staticmethod
    def adjust_bounds_fromCOM_w_min_dist(
        COM: tuple, min_dist: float, max_val_allowed: int | None = None
    ) -> tuple:
        """
        Adjusts the bounds around a center of mass (COM) with a minimum distance.

        Args:
            COM (tuple): The center of mass coordinates (row, col).
            min_dist (float): The minimum distance from the center of mass.
            max_val_allowed (int, optional): The maximum value allowed for the bounds. Defaults to None.

        Returns:
            tuple: A tuple containing the adjusted bounds (min_row, min_col, max_row, max_col).
        """
        COM_row, COM_col = COM

        min_row = max(0, int(COM_row - min_dist))
        min_col = max(0, int(COM_col - min_dist))
        max_row = min(max_val_allowed - 1, int(COM_row + min_dist))
        max_col = min(max_val_allowed - 1, int(COM_col + min_dist))

        return (min_row, min_col, max_row, max_col)

    @staticmethod
    def apply_DBScan2find_CorePoints(
        arr_pts: np.ndarray, eps: float, min_samples: int
    ) -> np.ndarray:
        """
        Applies the DBSCAN algorithm to find the core points in the given array of points.

        Parameters:
        - arr_pts (numpy.ndarray): The array of points to be clustered.
        - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        - min_samples (int): The minimum number of samples in a neighborhood for a point to be considered as a core point.

        Returns:
        - core_points (numpy.ndarray): The array of core points found by the DBSCAN algorithm.
        """
        from sklearn.cluster import DBSCAN

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(arr_pts)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        core_points = arr_pts[core_samples_mask]
        return core_points

    @staticmethod
    def find_convel_hull(arr_pts: np.ndarray) -> np.ndarray:
        """
        Finds the convex hull of a set of points.

        Parameters:
        arr_pts (ndarray): An array of points in the form of a numpy ndarray.

        Returns:
        ndarray: An array of points representing the vertices of the convex hull.
        """
        from scipy.spatial import ConvexHull

        hull = ConvexHull(arr_pts)
        return arr_pts[hull.vertices]

    @staticmethod
    def find_contours(
        arr2use: np.ndarray, contour_level: float, normalize_before: bool = True
    ) -> list:
        """
        Find contours in an array.

        Parameters:
        - arr2use: numpy.ndarray
            The input array to find contours in.
        - contour_level: float
            The contour level to use for finding contours.
        - normalize_before: bool, optional
            Whether to normalize the input array before finding contours. Default is True.

        Returns:
        - contours: list of numpy.ndarray
            A list of arrays representing the contours found in the input array.
        """
        from skimage.measure import find_contours

        if normalize_before:
            arr2use /= np.max(arr2use)

        contours = find_contours(arr2use, contour_level)
        return contours
