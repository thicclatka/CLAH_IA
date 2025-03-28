import os


def get_image_path(image_name: str, context: str = "docs") -> str:
    """
    Get the path to an image file based on the context.

    Args:
        image_name: Name of the image file (e.g., 'M2SD_icon.png')
        context: Where the image is being used ('docs' or 'app')

    Returns:
        str: Full path to the image file
    """
    # Get the package root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(os.path.dirname(current_dir))

    if context == "docs":
        return os.path.join(package_root, "docs", "images", image_name)
    else:  # app context
        return os.path.join(package_root, "docs", "assets", image_name)
