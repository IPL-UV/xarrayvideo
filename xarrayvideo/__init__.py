import logging

# Set up library logger
logger = logging.getLogger(__name__)
# Default to WARNING level for library code
logger.setLevel(logging.WARNING)

# Only add handler if one doesn't exist to avoid duplicate output
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .utils import * #safe_eval, to_netcdf, normalize
from .ffmpeg_wrappers import * #_ffmpeg_read, _ffmpeg_write
from .plot import * #plot_simple, plot_image
from .metrics import * #SA, SNR
from .xarrayvideo import *