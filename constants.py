import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'assets'
RAW_IMAGES = DATA_ROOT / 'images'
RAW_MESHES = DATA_ROOT / 'meshes'
RAW_SILHOUETTES = DATA_ROOT / 'silhouettes'
CHECKPOINTS_ROOT = DATA_ROOT / 'checkpoints'