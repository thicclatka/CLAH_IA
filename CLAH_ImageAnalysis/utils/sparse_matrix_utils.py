import numpy as np
import scipy.sparse as sp


def threshold_sparse_matrix(
    sparse_matrix: sp.csr_matrix, threshold: float
) -> sp.csr_matrix:
    """
    Apply a threshold to a sparse matrix. Values below the threshold are set to zero.

    Args:
        sparse_matrix (scipy.sparse.csr_matrix): The sparse matrix to threshold.
        threshold (float): The threshold value below which values will be set to zero.

    Returns:
        scipy.sparse.csr_matrix: The thresholded sparse matrix, maintaining sparse format.
    """
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    # Create a mask for data above the threshold
    mask = sparse_matrix.data >= threshold
    new_data = sparse_matrix.data[mask]
    new_indices = sparse_matrix.indices[mask]

    # Compute the new indptr for the CSR matrix
    new_indptr = np.zeros_like(sparse_matrix.indptr)
    # Increment new_indptr at the positions where rows start
    for i in range(len(sparse_matrix.indptr) - 1):
        # Count non-zero entries for each row in the new data
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i + 1]
        # Apply mask within the range for this row and count the true values
        row_mask = mask[row_start:row_end]
        new_indptr[i + 1] = new_indptr[i] + np.sum(row_mask)

    # Create a new CSR matrix with the filtered data, indices, and indptr
    new_sparse_matrix = sp.csr_matrix(
        (new_data, new_indices, new_indptr), shape=sparse_matrix.shape
    )
    return new_sparse_matrix
