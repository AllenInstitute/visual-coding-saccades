"""
This Python script packages the eye tracking data into `BASE_DIR`.
`BASE_DIR` can be supplied in the argumemts: e.g., python package_eye_data.py /path/to/target/dir

Specifically, it generates the following file structure (inside the target directory):
    `BASE_DIR`/data/[...].npy: Stores .npy files for eye tracking
    `BASE_DIR`/manifest.json: Manifest file stored with the following structure
        {
            "column_labels": [
                "frame",
                "eye_area",
                "pupil_area",
                "x_pos_deg",
                "y_pos_deg"
            ],
            "sessions": [
                {
                    "session_id": ...,
                    "has_eye_tracking": true,
                    "data_file": "data/xyz.npy"
                },
                ...
            ]
        }
"""

import sys
import os
from os import path
import glob
import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

# Local imports (hacky approach)
if ".." not in sys.path: sys.path.append("..")
from saccade_config import *
from parallel_data_loading import load_eye_tracking_data

# The columns to save in the data files
EYE_DATA_COLUMNS = ["frame", "eye_area", "pupil_area", "x_pos_deg", "y_pos_deg"]

def package_eye_tracking(save_dir: str):
    """Packages eye tracking data for each session into a given target directory.

    Args:
        base_dir (str): Base directory in which to save packaged eye tracking files
    """
    # Make directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(path.join(save_dir, "data"), exist_ok=True)

    manifest = {
        "column_labels": EYE_DATA_COLUMNS,
        "sessions": []
    }

    # Load DF that describes sessions and eye files
    eye_df = pd.read_csv(path.join(EYE_DATA_PATH, "data_info.csv"))

    # Load brain observatory cache
    boc = BrainObservatoryCache(manifest_file=BRAIN_OBSERVATORY_CACHE_MANIFEST_FILE)

    # Load session IDs with eye tracking
    all_eye_tracking_files = glob.glob(path.join(EYE_DATA_PATH, "*.h5"))
    eye_tracking_files_by_session = {} # session_id -> [eye_tracking_file_path, time_sync_file_path, allen_eye_tracking_file_path, allen_time_sync_file_path]

    print(f"There are {len(all_eye_tracking_files)} eye tracking files.")

    for full_file in all_eye_tracking_files:
        file_name = path.basename(full_file)
        
        # Locate the session name
        if "eyetracking_dlc_to_screen_mapping" in file_name:
            col = "eye_tracking_file"
            dict_idx = 0, 2
        elif "time_synchronization" in file_name:
            col = "time_sync_file"
            dict_idx = 1, 3
        else:
            print(f"WARN: Unknown eye tracking file {file_name}")
            continue
        
        df_idx = np.where(eye_df[col] == file_name)[0]
        if len(df_idx) == 0:
            print(f"WARN: Eye tracking file {file_name} has no known session")
            continue
        df_idx = df_idx[0]
        session_id = int(eye_df.session_id.iloc[df_idx])
        allen_full_file = eye_df[f"allen_{col}"].iloc[df_idx]
        
        # Clean up file name so it looks like "/allen/.../filename.h5"
        # (Unfortunately I had a small typo in this DataFrame... so the time sync file starts with Volumes instead of allen)
        allen_full_file = re.sub("^.*/programs/", "/allen/programs/", allen_full_file.replace("\\", "/"))
        
        if session_id not in eye_tracking_files_by_session:
            eye_tracking_files_by_session[session_id] = [None, None, None, None]
        eye_tracking_files_by_session[session_id][dict_idx[0]] = full_file
        eye_tracking_files_by_session[session_id][dict_idx[1]] = allen_full_file

    # Remove any bad sessions (with one eye file but not the other)
    n_bad = 0
    for session_id in list(eye_tracking_files_by_session.keys()):
        if None in eye_tracking_files_by_session[session_id]:
            del eye_tracking_files_by_session[session_id]
            n_bad += 1
    if n_bad > 0:
        print(f"Removed {n_bad} sessions from eye tracking because they didn't have both eye tracking files.")

    print(f"There are {len(eye_tracking_files_by_session)} sessions with eye tracking to process.")

    # Load the corresponding ophys experiments
    all_ophys_ids = [exp["id"] for exp in boc.get_ophys_experiments()]
    n_not_in_boc = 0

    for session_id, (eye_tracking_file, time_sync_file, allen_eye_tracking_file, allen_time_sync_file) in tqdm(eye_tracking_files_by_session.items(), desc="Packaging eye tracking data"):
        # Only load data if it is in the Brain Observatory
        if session_id not in all_ophys_ids:
            n_not_in_boc += 1
            continue

        # Load ophys session and eye tracking
        ophys_experiment_data = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
        flu = ophys_experiment_data.get_corrected_fluorescence_traces()[1]

        # Load the eye tracking files
        eye_tracking = load_eye_tracking_data(eye_tracking_file, time_sync_file, len_2p=flu.shape[1])

        # Package and save eye tracking
        data_array = eye_tracking[EYE_DATA_COLUMNS].values

        file_name = f"{session_id}.npy"
        data_file_relative = path.join("data", file_name)
        data_file = path.join(save_dir, data_file_relative)

        np.save(data_file, data_array)

        manifest["sessions"].append({
            "session_id": session_id,
            "has_eye_tracking": True,
            "data_file": data_file_relative,
            "allen_eye_tracking_file": allen_eye_tracking_file,
            "allen_time_sync_file": allen_time_sync_file
        })
    
    if n_not_in_boc > 0:
        print(f"There were {n_not_in_boc} eye tracking files saved locally that did not have corresponding sessions in Brain Observatory.")

    # Save the manifest file
    manifest_file_name = path.join(save_dir, "manifest.json")
    with open(manifest_file_name, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=4)

            
if __name__ == "__main__":
    if len(sys.argv) > 1:
        save_dir = sys.argv[1]
    else:
        # save_dir = r"/Users/chase/Desktop/MindScope/packaged_viscoding_eye_tracking"
        save_dir = r"/Users/chase/Library/CloudStorage/OneDrive-AllenInstitute/NEW_packaged_viscoding_eye_tracking"
    
    print(f"Packaging eye tracking files into {save_dir}")
    package_eye_tracking(save_dir)
    print("Done!")