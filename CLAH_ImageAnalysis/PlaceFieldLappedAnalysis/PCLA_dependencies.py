import numpy as np
from scipy.ndimage import convolve


class PCLA_dependencies:
    def __init__(self):
        pass

    def angleDiff(self, angle1: float, angle2: float) -> float:
        """
        Calculate the difference between two angles.

        Parameters:
            angle1 (float): The first angle in radians.
            angle2 (float): The second angle in radians.

        Returns:
            float: The difference between the two angles.

        """
        angle1 = self.normalizeAngle(angle1)
        angle2 = self.normalizeAngle(angle2)

        diff = self.normalizeAngle(angle2 - angle1, center=0)
        return diff

    @staticmethod
    def normalizeAngle(alpha: float, center: float = np.pi) -> float:
        """
        Normalize an angle to the range [-pi, pi].

        Parameters:
            alpha (float): The angle to be normalized.
            center (float, optional): The center of the range. Defaults to np.pi.

        Returns:
            float: The normalized angle.

        """
        alpha = (alpha - center + np.pi) % (2 * np.pi) + center - np.pi
        return alpha

    @staticmethod
    def create_linearly_spaced_vector(
        start: float, stop: float, num: int, LSV_end: float
    ) -> np.ndarray:
        """
        Create a linearly spaced vector with a specified start, stop, and number of elements.

        Parameters:
            start (float): The starting value of the vector.
            stop (float): The ending value of the vector.
            num (int): The number of elements in the vector.
            LSV_end (float): The value to replace the last element of the vector.

        Returns:
            numpy.ndarray: The linearly spaced vector.

        """
        LSV = np.linspace(start, stop, num)
        LSV[-1] = LSV_end
        return LSV

    def convolveWithTrim(self, sp: np.ndarray, win1: np.ndarray) -> np.ndarray:
        """
        Convolve a 2D array with a 1D window and trim the result.

        Parameters:
            sp (numpy.ndarray): The 2D array to be convolved.
            win1 (numpy.ndarray): The 1D window.

        Returns:
            numpy.ndarray: The convolved array with trimmed edges.

        """
        out = np.zeros_like(sp)
        for i in range(sp.shape[1]):  # Iterate over columns
            out[:, i] = self.convtrim(sp[:, i], win1, mode="same")

        out = out / np.sum(win1)
        return out

    @staticmethod
    def convtrim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Convolve two 1D arrays and trim the result.

        Parameters:
            a (numpy.ndarray): The first 1D array.
            b (numpy.ndarray): The second 1D array.

        Returns:
            numpy.ndarray: The trimmed convolution result.

        Raises:
            ValueError: If the length of vector a is smaller than vector b.

        """
        if len(a) <= len(b):
            raise ValueError("The length of vector a must be larger than vector b")

        tempC = np.convolve(a, b, mode="full")
        front_trim = len(b) // 2

        if len(b) % 2 != 0:
            back_trim = len(b) // 2
        else:
            back_trim = len(b) // 2 - 1

        trimmedConv = tempC[front_trim : len(tempC) - back_trim]
        return trimmedConv

    @staticmethod
    def suprathresh(Vector: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Find segments in a vector that are above a threshold.

        Parameters:
            Vector (numpy.ndarray): The input vector.
            thresh (float): The threshold value.

        Returns:
            tuple: A tuple containing the start and end indices of the segments, and the length of each segment.

        """
        Vector = np.array(Vector)
        if Vector.ndim == 1:
            Vector = Vector[:, np.newaxis]

        Vector = Vector >= thresh
        Vector = Vector.astype(int)
        Vector = np.insert(Vector, [0, Vector.shape[0]], 0)
        d = np.diff(Vector, axis=0)

        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1

        # Ensure starts and ends have the same length
        min_length = min(len(starts), len(ends))
        starts = starts[:min_length]
        ends = ends[:min_length]

        out = np.column_stack((starts, ends))
        segmentLength = np.diff(out, axis=1) + 1

        return out, segmentLength.squeeze()

    @staticmethod
    def process_intervalsNcounts(
        intervals: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process intervals and counts based on input arrays.

        Parameters:
            intervals (numpy.ndarray): The input intervals array.
            X (numpy.ndarray): The input X array.

        Returns:
            tuple: A tuple containing the processed intervals and counts arrays.

        Raises:
            ValueError: If the input intervals array is not in the correct format.

        """
        if intervals.shape[0] != 2 and min(intervals.shape) > 1:
            raise ValueError(
                "*Please format your inputs correctly. Signed Sincerely -The Management"
            )

        if intervals.size != 0 and X.size != 0:
            ind = np.argsort(np.mean(intervals, axis=0))
            intervals = intervals[:, ind]

            h1 = np.sort(intervals.T.reshape(-1))

            counts, bin_edges = np.histogram(X, bins=h1)
            whichInt = np.digitize(X, bin_edges, right=True) - 1

            counts[-2] += counts[-1]
            counts = counts[:-1]

            whichInt[whichInt == len(h1)] = len(h1) - 1

            whichInt = (whichInt + 1) / 2
            whichInt[whichInt != np.round(whichInt)] = 0

        else:
            whichInt = np.array([])
            counts = np.array([])

        return whichInt, counts

    @staticmethod
    def convolve2D(x: np.ndarray, m: np.ndarray, shape: str = "full") -> np.ndarray:
        """
        Convolve a 2D array with a 2D mask.

        Parameters:
            x (numpy.ndarray): The input 2D array.
            m (numpy.ndarray): The 2D mask.
            shape (str, optional): The shape of the output. Defaults to "full".

        Returns:
            numpy.ndarray: The convolved array.

        Raises:
            ValueError: If the shape parameter is unsupported.

        """
        if shape == "full":
            mode = "constant"  # Zero padding
        elif shape == "same":
            mode = "nearest"  # Reflects the vector at the boundary
        elif shape == "wrap":
            mode = "wrap"  # Circular wrap
        elif shape in ["reflect", "symmetric"]:
            mode = "reflect"  # Mirror reflection
        elif shape == "replicate":
            mode = "nearest"  # Replicates the edge value
        else:
            raise ValueError("Unsupported shape parameter")

        y = convolve(x, m, mode=mode)

        if shape == "valid":
            valid_size = np.subtract(x.shape, np.subtract(m.shape, 1))
            valid_start = np.floor_divide(m.shape, 2)
            valid_end = np.add(valid_start, valid_size)
            y = y[valid_start[0] : valid_end[0], valid_start[1] : valid_end[1]]

        return y
