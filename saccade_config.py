from os import path

# Parent directory for the Brain Observatory cache
BRAIN_OBSERVATORY_CACHE_DRIVE_PATH = r"/Users/chase/Desktop/MindScope/visual-coding-2p"

# Manifest file for the Brain Observatory cache (do not edit)
BRAIN_OBSERVATORY_CACHE_MANIFEST_FILE = path.join(BRAIN_OBSERVATORY_CACHE_DRIVE_PATH, "manifest.json")

# Where the eye tracking data files are stored locally
EYE_DATA_PATH = r"/Users/chase/Desktop/MindScope/eye_data"
DATA_DIR = "/Users/chase/Desktop/MindScope/Documents/visual-coding-saccades/data"

# Where the experiment data (loaded from parallel_data_loading.py) will be saved
CELL_DATA_SAVE_DIR = f"{DATA_DIR}/cell_data"

# The directory of data already loaded from parallel_data_loading.py
LOADED_CELL_DATA_DIR = f"{DATA_DIR}/cell_data_v2"

# Where saccade-classified cell data will be saved
CLASSIFIED_CELLS_SAVE_DIR = f"{DATA_DIR}/classified_cells"

# CSV file of saccade-classified cells
LOADED_CLASSIFIED_CELLS_CSV = f"{DATA_DIR}/classified_cells_v2/cells.csv"

# CSV file of saccade-classified cells
LOADED_CLASSIFIED_CELLS_SPONTANEOUS_CSV = f"{DATA_DIR}/classified_cells_v1/cells-spontaneous.csv"

# Base directory for saved figures
# FIGURE_BASE_DIR = r"/Users/chase/Desktop/saccade_paper_figures"
FIGURE_BASE_DIR = r"/Users/chase/Library/CloudStorage/OneDrive-AllenInstitute/Manuscripts/Saccade Analysis Brain Observatory/saccade_paper_figures"

# Parameters for saccade loading
SACCADE_NAN_PADDING_SEC = 0.5 # Ignore saccades that occur near NaN eye tracking frames
SACCADE_NOISE_PADDING_SEC = 0.3 # Width sensitivity to noise
SACCADE_NOISE_STD_THRESH = 3 # Sensitivity of noise (higher value ==> less tolerant to noise)

FRAME_RATE = 30 # Frame rate for 2P imaging. Do not change.