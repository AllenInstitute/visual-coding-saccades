from os import path

# Parent directory for the Brain Observatory cache
BRAIN_OBSERVATORY_CACHE_DRIVE_PATH = r"/Users/chase/Desktop/MindScope/visual-coding-2p"

# Manifest file for the Brain Observatory cache (do not edit)
BRAIN_OBSERVATORY_CACHE_MANIFEST_FILE = path.join(BRAIN_OBSERVATORY_CACHE_DRIVE_PATH, "manifest.json")

# Where the experiment data (loaded from parallel_data_loading.py) will be saved
CELL_DATA_SAVE_DIR = r"/Users/chase/Desktop/MindScope/Documents/visual-coding-saccades/data/cell_data"

# The directory of data already loaded from parallel_data_loading.py
# LOADED_CELL_DATA_DIR = r"/Users/chase/Desktop/MindScope/Documents/visual-coding-saccades/data/cell_data_2022-05-24_08-40-28"
LOADED_CELL_DATA_DIR = r"/Users/chase/Desktop/MindScope/Documents/visual-coding-saccades/data/cell_data_2022-08-07_17-00-12"

# Where saccade-classified cell data will be saved
CLASSIFIED_CELLS_SAVE_DIR = r"/Users/chase/Desktop/MindScope/Documents/visual-coding-saccades/data/classified_cells_boot"

# CSV file of saccade-classified cells
LOADED_CLASSIFIED_CELLS_CSV = r"/Users/chase/Desktop/MindScope/Documents/visual-coding-saccades/data/classified_cells_boot_2022-05-30_17-41-16/cells.csv"

# Base directory for saved figures
FIGURE_BASE_DIR = r"/Users/chase/Desktop/saccade_paper_figures"

# Parameters for saccade loading
SACCADE_NAN_PADDING_SEC = 0.5 # Ignore saccades that occur near NaN eye tracking frames
SACCADE_NOISE_PADDING_SEC = 0.3 # Width sensitivity to noise
SACCADE_NOISE_STD_THRESH = 3 # Sensitivity of noise (higher value ==> less tolerant to noise)

FRAME_RATE = 30 # Frame rate for 2P imaging. Do not change.