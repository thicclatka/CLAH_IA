from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
import umap


def calc_simMatrix(array: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine similarity matrix for a given array.

    Parameters:
        array (np.ndarray): Input array of shape (n_samples, n_features)

    Returns:
        np.ndarray: Cosine similarity matrix of shape (n_samples, n_samples)
    """
    similarity_matrix = cosine_similarity(array)

    # Normalize to [0, 1]
    if np.min(similarity_matrix) < 0:
        similarity_matrix = (similarity_matrix + 1) / 2

    # Set diagonal to 1.0
    np.fill_diagonal(similarity_matrix, 1.0)
    return similarity_matrix


def SpectralClustering_fit2simMatrix(
    similarity_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit a spectral clustering model to a similarity matrix.

    Parameters:
        similarity_matrix (np.ndarray): Input similarity matrix of shape (n_samples, n_samples)
        n_clusters (int): Number of clusters to fit. Should match the number of unique categories.

    Returns:
        np.ndarray: Cluster labels of shape (n_samples,)
    """
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
    )
    cluster_labels = spectral.fit_predict(similarity_matrix)
    return cluster_labels


def determine_projection_weights(
    feature_array: np.ndarray, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    Determine the projection weights for a given array and mapped values.

    Parameters:
        feature_array (np.ndarray): Input array of shape (n_samples, n_features)
        cluster_labels (np.ndarray): Input cluster labels of shape (n_samples,)

    Returns:
        np.ndarray: Projection weights of shape, after L2 normalization (n_samples, n_clusters)
    """
    mapped_values = find_mean_mapped_clusterValues(
        feature_array=feature_array, cluster_labels=cluster_labels
    )

    unique_labels = np.unique(cluster_labels)
    projection_weights = np.zeros((feature_array.shape[0], len(unique_labels)))
    for k in range(feature_array.shape[0]):
        X_k = feature_array[k]
        reg = LinearRegression().fit(np.column_stack(mapped_values), X_k)
        weights = reg.coef_
        # Normalize to L2 norm
        weights = weights / np.linalg.norm(weights)
        projection_weights[k] = weights
    return projection_weights


def create_distance_matrix_from_similarity_matrix(
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """
    Create a distance matrix from a similarity matrix.

    Parameters:
        similarity_matrix (np.ndarray): Input similarity matrix of shape (n_samples, n_samples)

    Returns:
        np.ndarray: Distance matrix of shape (n_samples, n_samples)
    """
    distance_matrix = 1 - similarity_matrix

    # Ensure diagonal is 0
    np.fill_diagonal(distance_matrix, 0)

    # Symmetrize (should already be symmetric, but enforce)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    return distance_matrix


def UMAP_fit2distMatrix(
    distance_matrix: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    Fit a UMAP model to a distance matrix.

    Parameters:
        distance_matrix (np.ndarray): Input distance matrix of shape (n_samples, n_samples)
        random_state (int): Random state for UMAP
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance for UMAP

    Returns:
        np.ndarray: UMAP embedding of shape (n_samples, 2)
    """
    reducer = umap.UMAP(
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=1,  # disable parallel processing since random_state is set
    )
    embedding = reducer.fit_transform(distance_matrix)
    return embedding


def find_optimal_assignment(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the optimal assignment of rows to columns using the Hungarian algorithm.

    Parameters:
        true_labels (np.ndarray): Input true labels of shape (n_samples,)
        cluster_labels (np.ndarray): Input cluster labels of shape (n_samples,)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Confusion matrix, row indices, column indices
    """
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm, row_ind, col_ind


def determineSpectralClustering_accuracy(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Determine the accuracy of a spectral clustering model.

    Parameters:
        true_labels (np.ndarray): Input true labels of shape (n_samples,)
        cluster_labels (np.ndarray): Input cluster labels of shape (n_samples,)

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Accuracy, row indices, column indices
    """

    # Find the confusion matrix & optimal assignment using the Hungarian algorithm
    # find the optimal assignment of rows to columns
    cm, row_ind, col_ind = find_optimal_assignment(true_labels, cluster_labels)

    # The maximum number of correctly matched samples is the sum along the optimal assignment
    optimal_matches = cm[row_ind, col_ind].sum()

    # Calculate accuracy
    accuracy = optimal_matches / len(true_labels)

    return accuracy


def map_cluster_labels_to_true_labels(
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    rowFROMassignment: np.ndarray | None = None,
    colFROMassignment: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map cluster labels to true labels.

    Parameters:
        true_labels (np.ndarray): Input true labels of shape (n_samples,)
        cluster_labels (np.ndarray): Input cluster labels of shape (n_samples,)
        rowFROMassignment (np.ndarray | None): Input row indices of shape (n_samples,) derived from linear_sum_assignment of confusion matrix
        colFROMassignment (np.ndarray | None): Input column indices of shape (n_samples,) derived from linear_sum_assignment of confusion matrix

    Returns:
        np.ndarray: Corrected cluster labels of shape (n_samples,) that are optimally matched to true labels
    """
    unique_true = np.unique(true_labels)
    unique_clust = np.unique(cluster_labels)

    # Find the confusion matrix & optimal assignment using the Hungarian algorithm
    if rowFROMassignment is None or colFROMassignment is None:
        _, row_ind, col_ind = find_optimal_assignment(true_labels, cluster_labels)
    else:
        row_ind = rowFROMassignment
        col_ind = colFROMassignment

    map_clust_to_true = {}
    for r, c in zip(row_ind, col_ind):
        original_cluster_label = unique_clust[c]
        assigned_true_label = unique_true[r]
        map_clust_to_true[original_cluster_label] = assigned_true_label

    # Apply the map to the original cluster labels
    corrected_cluster_labels = np.array(
        [map_clust_to_true[old_label] for old_label in cluster_labels]
    )

    return corrected_cluster_labels


def find_mean_mapped_clusterValues(
    feature_array: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Find the mean mapped cluster values for a given array and cluster labels.

    Parameters:
        feature_array (np.ndarray): Input array of shape (n_samples, n_features)
        cluster_labels (np.ndarray): Input cluster labels of shape (n_samples,)

    Returns:
        np.ndarray: Mean mapped cluster values of shape (n_clusters, n_features)
    """
    mean_mapped_clusterValues = []

    for label in np.unique(cluster_labels):
        mean_map = np.mean(feature_array[cluster_labels == label], axis=0)
        mean_mapped_clusterValues.append(mean_map)
    return np.array(mean_mapped_clusterValues)
