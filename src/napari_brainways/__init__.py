try:
    from napari_brainways._version import version as __version__
except ImportError:
    __version__ = "unknown"
from napari_brainways._sample_data import make_sample_data
from napari_brainways.widget import BrainwaysUI

__all__ = (
    "make_sample_data",
    "BrainwaysUI",
)
