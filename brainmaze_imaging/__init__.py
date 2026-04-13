"""brainmaze_imaging – DICOM exploration and conversion utilities."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("brainmaze_imaging")
except PackageNotFoundError:  # package not installed (e.g. during development)
    __version__ = "0.0.0.dev"

__all__ = ["__version__"]
