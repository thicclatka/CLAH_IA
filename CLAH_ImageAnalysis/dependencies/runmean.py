import numpy as np


def runmean(X: np.ndarray, m: int, mode: str = "edge") -> np.ndarray:
    """
    Compute the running mean of a 1-D array.

    Parameters:
    X (ndarray): The input array.
    m (int): The window size for the running mean.
    mode (str, optional): The padding mode. Can be "edge", "zero", or "mean". Defaults to "edge".

    Returns:
    ndarray: The running mean of the input array.

    """
    if m == 0:
        return X

    mm = 2 * m + 1
    if mm >= len(X):
        return np.full_like(X, np.mean(X))

    if mode == "edge":
        X_padded = np.pad(X, (m, m), mode="edge")
    elif mode == "zero":
        X_padded = np.pad(X, (m, m), mode="constant", constant_values=0)
    elif mode == "mean":
        mean_val = np.mean(X)
        X_padded = np.pad(X, (m, m), mode="constant", constant_values=mean_val)

    X_padded = np.insert(X_padded, 0, 0)

    # Compute the cumulative sum and then the running mean
    Y = np.cumsum(X_padded, dtype=float)
    Y = (Y[mm:] - Y[:-mm]) / mm

    if len(Y) > len(X):
        Y = Y[: len(X)]
    elif len(Y) < len(X):
        #  just in case
        Y = np.append(Y, np.full(len(X) - len(Y), np.nan))

    return Y
