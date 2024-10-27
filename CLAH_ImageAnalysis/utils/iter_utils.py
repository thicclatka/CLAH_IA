from itertools import repeat


def prepare_None_iter(*args) -> list:
    """
    Prepare None arguments for zip_longest.

    If an argument is None, replace it with an infinite iterator that yields None.
    Otherwise, leave it as it is.

    Parameters:
        *args: The iterables to prepare.

    Returns:
        A list of the prepared iterables.
    """
    return [repeat(None) if arg is None else arg for arg in args]
