import caiman as cm
import h5py
import numpy as np
import cv2
from skimage.util import img_as_uint


######################################################
#  movie funcs
######################################################


def load_movie(
    source: str, is_memory_mapped: bool = True, normalize: bool = False
) -> np.ndarray | list:
    """
    Load a movie from a source file or memory map.

    Parameters:
        source (str): The path to the source file or memory map.
        is_memory_mapped (bool): Whether the source is a memory map. Default is True.
        normalize (bool): Whether to normalize the movie. Default is False.

    Returns:
        movie (ndarray or list): The loaded movie.

    If `is_memory_mapped` is True, the function loads the movie using `cm.load()`.
    If `is_memory_mapped` is False, the function loads the movie using `cm.load_movie_chain()`.

    If `normalize` is True, the function normalizes the movie using `normNconvert2uint()` and returns the normalized movie.
    If `normalize` is False, the function returns the loaded movie without normalization.
    """
    if is_memory_mapped:
        movie = cm.load(source)
    else:
        movie = cm.load_movie_chain([source])

    if normalize:
        # normalize and convert to uint
        norm_uint_movie = normNconvert2uint(movie)
        return norm_uint_movie
    else:
        return movie


def process_and_play_movie(
    movie: np.ndarray | list,
    downsample_ratio: float = 0.2,
    fr: int = 30,
    gain: int = 2,
    magnification: int = 2,
    offset: int = 0,
) -> None:
    """
    Process and play a movie.

    Parameters:
        movie (object): The movie object to be processed and played.
        downsample_ratio (float, optional): The ratio by which to downsample the movie. Defaults to 0.2.
        fr (int, optional): The frame rate of the played movie. Defaults to 30.
        gain (int, optional): The gain of the played movie. Defaults to 2.
        magnification (int, optional): The magnification of the played movie. Defaults to 2.
        offset (int, optional): The offset of the played movie. Defaults to 0.

    Returns:
        None
    """
    processed_movie = movie.resize(1, 1, downsample_ratio)
    if offset is None:
        offset = -np.min(processed_movie[:100])
    processed_movie.play(gain=gain, offset=offset, fr=fr, magnification=magnification)


def create_denoised_movie(
    cnm_estimates: object,
    dims: tuple,
    wBackground: bool = False,
    normalize: bool = False,
) -> object:
    """
    Create a denoised movie from CNMF-Estimates.

    Parameters:
        cnm_estimates (object): An object containing CNMF-Estimates.
        dims (tuple): The dimensions of the movie.
        wBackground (bool, optional): Whether to add background to the movie. Defaults to False.
        normalize (bool, optional): Whether to normalize and convert the movie to uint. Defaults to False.

    Returns:
        numpy.ndarray: The denoised movie.

    """
    movie_to_load = cnm_estimates.A.dot(cnm_estimates.C)
    # add background (b dot F)
    if wBackground:
        movie_to_load += cnm_estimates.b.dot(cnm_estimates.f)

    # turn into movie
    denoised = cm.movie(movie_to_load)

    # normalize and convert to uint
    if normalize:
        denoised = normNconvert2uint(denoised)

    return denoised.reshape(dims + (-1,), order="F").transpose([2, 0, 1])


def normNconvert2uint(stack_to_norm: np.ndarray) -> np.ndarray:
    """
    Normalize and convert a stack to unsigned integer format.

    Parameters:
        stack_to_norm (ndarray): The input stack to be normalized and converted.

    Returns:
        ndarray: The normalized and converted stack as a movie.
    """
    # normalize
    norm_stack = stack_to_norm / stack_to_norm.max()
    # clip values to be within -1 and 1
    norm_stack = np.clip(norm_stack, -1, 1)
    # convert to uint
    norm_uint_stack = img_as_uint(norm_stack)
    # convert to movie
    norm_uint_stack_movie = cm.movie(norm_uint_stack)
    return norm_uint_stack_movie


def concatenate_movies(
    movies: list, axis: int = 2, use_caiman: bool = True
) -> np.ndarray | object:
    """
    Concatenates a list of movies along a specified axis.

    Parameters:
        movies (list): A list of movies to be concatenated.
        axis (int, optional): The axis along which the movies should be concatenated. Defaults to 2.
        use_caiman (bool, optional): Whether to use Caiman's concatenate function. Defaults to True.
    Returns:
        ndarray: The concatenated movie.
    """
    if use_caiman:
        return cm.concatenate(movies, axis=axis)
    else:
        return np.concatenate(movies, axis=axis)


def save_movie(
    movie: object,
    fname: str,
    ftag: str,
    element_size_um: list[float] | None = None,
    use_caiman: bool = True,
) -> None:
    """
    Save a movie to a file.

    Parameters:
        movie (object): The movie object to be saved.
        fname (str): The base filename for the saved movie.
        ftag (str): The file tag indicating the file format (e.g., "AVI", "H5").
        element_size_um (list, optional): The element size in micrometers. Defaults to an empty list.
    """
    # strings for file names & process
    from CLAH_ImageAnalysis.utils import text_dict

    file_tag = text_dict()["file_tag"]

    if ftag == file_tag["AVI"]:
        if use_caiman:
            movie.save(f"{fname}{ftag}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                f"{fname}{ftag}", fourcc, 30.0, (movie.shape[2], movie.shape[1])
            )
            for frame in movie:
                out.write(frame)
            out.release()
    elif ftag == file_tag["H5"]:
        fname = f"{fname}{ftag}"
        with h5py.File(fname, "w") as hf:
            ds = hf.create_dataset("mov", data=movie)
            ds.attrs["DIMENSION_LABELS"] = ["t", "y", "x"]
            if element_size_um is not None:
                ds.attrs["element_size_um"] = np.array(element_size_um)
            else:
                ds.attrs["element_size_um"] = np.array([np.nan, np.nan, np.nan])


def add_caption_to_movie(
    movie: np.ndarray,
    text: str,
    num_frames: int = 100,
    bit_depth: type = np.uint8,
) -> object:
    """
    Add a caption with a semi-transparent background to the first N frames of a movie.

    Parameters:
        movie (np.ndarray): The input movie array with shape (frames, height, width).
        text (str): The text caption to add to the frames.
        num_frames (int, optional): Number of frames to add the caption to. Defaults to 30.
        bit_depth (type, optional): The bit depth of the movie. Defaults to np.uint8.

    Returns:
        np.ndarray: Movie array with caption added to specified number of frames.
            The caption includes a semi-transparent black background rectangle with
            red text overlaid.
    """
    movie_norm = cv2.normalize(
        movie, None, 0, np.iinfo(bit_depth).max, cv2.NORM_MINMAX
    ).astype(bit_depth)

    movie_bgr = []
    for frame in movie_norm:
        movie_bgr.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    movie_bgr = np.array(movie_bgr)

    # only process first num_frames (or all frames if num_frames > len(movie))
    num_frames = min(num_frames, len(movie))

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 0.5
    color = (0, 0, np.iinfo(bit_depth).max)  # Red
    thickness = 1

    # Add red text to first N frames
    for frame_idx in range(min(num_frames, len(movie))):
        cv2.putText(
            movie_bgr[frame_idx], text, position, font, font_scale, color, thickness
        )

    # # convert to movie
    # movie_bgr = cm.movie(movie_bgr)

    return movie_bgr
