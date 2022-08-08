import sys
sys.path.append("./")

import time
import traceback

import os
import gc
import time
import traceback

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.brain_observatory_exceptions import EpochSeparationException

from saccade_config import *
from experiment.cre_line import match_cre_line
from helpers.parallel_process import ParallelProcess
from eye_tracking.eye_tracking import get_saccades_from_gaze_traces

# =========================================================================
#                    TODO: INTEGRATE THIS INTO THE BOC

from eye_tracking.local_eye_data_repository import LocalEyeDataRepository
import h5py
import numpy as np
import pandas as pd

"""
Get a pd.DataFrame of eye tracking data indexed by frames that are synced with 2P imaging
(2P imaging length set in len_2p).
"""
def get_eye_tracking(session_id, len_2p):
    # Determine which files to load
    eye_data_path = r"/Users/chase/Desktop/MindScope/eye_data" # stores the time sync files
    eye_data_repo = LocalEyeDataRepository.from_dir(eye_data_path, boc)
    data = eye_data_repo.get_session_data(session_id)

    # Load and process the files
    eye_tracking_file = os.path.join(eye_data_path, data["eye_tracking_file"])
    time_sync_file = os.path.join(eye_data_path, data["time_sync_file"])
    eye_tracking = load_eye_tracking_data(eye_tracking_file, time_sync_file, len_2p)

    return eye_tracking


"""
Load the eye tracking data from eye tracking and time sync files, and synchronize the data points
to the 2P imaging (the length of the 2P imaging is set in len_2p). Return a pd.DataFrame of
synced eye tracking data.
"""
def load_eye_tracking_data(eye_tracking_file, time_sync_file, len_2p, debug=0):
    time_sync_key = "eye_tracking_alignment"
    pupil_area = pd.read_hdf(eye_tracking_file, "raw_pupil_areas").values
    eye_area = pd.read_hdf(eye_tracking_file, "raw_eye_areas").values
    pos = pd.read_hdf(eye_tracking_file, "new_screen_coordinates")
    pos_deg = pd.read_hdf(eye_tracking_file, "raw_screen_coordinates_spherical")

    if debug > 0:
        with h5py.File(eye_tracking_file, "r") as file:
            print("Eye tracking keys:", [key for key in file.keys()])
    
    # Temporal alignment
    with h5py.File(time_sync_file, "r") as file:
        if debug > 0:
            print("Temporal alignment keys:", [key for key in file.keys()])

        frames = file[time_sync_key][()]

        # Create mapping from eye frame --> 2P frame (or -1 if none exists)
        eye_frame_to_2p_frame = -np.ones(int(frames.max()+1), dtype=np.int)
        for i, frame in enumerate(frames):
            eye_frame_to_2p_frame[int(frame)] = i # The eye frame is frame, the 2P frame is its index in the list

    frame_sync = np.arange(len_2p) # We will have a frame for every 2P frame

    def sync(original):
        synced = np.empty(len_2p, dtype=original.dtype)
        synced[:] = np.nan
        for frame_eye, val in enumerate(original):
            frame_2p = eye_frame_to_2p_frame[frame_eye] if frame_eye < len(eye_frame_to_2p_frame) else -1
            if 0 <= frame_2p < len(synced):
                synced[frame_2p] = val
        return synced

    # Sync all traces to 2P frame times
    eye_area_sync = sync(eye_area)
    pupil_area_sync = sync(pupil_area)
    pupil_area_sync = sync(pupil_area)
    x_pos_sync = sync(pos_deg.x_pos_deg.values)
    y_pos_sync = sync(pos_deg.y_pos_deg.values)

    # Build the data frame
    columns = ["frame", "eye_area", "pupil_area", "x_pos_deg", "y_pos_deg"]
    eye_tracking = pd.DataFrame(data={
        "frame": frame_sync,
        "eye_area": eye_area_sync,
        "pupil_area": pupil_area_sync,
        "x_pos_deg": x_pos_sync,
        "y_pos_deg": y_pos_sync
    }, columns=columns)

    # Calculate rate of change of position using central difference approximation of f'(x)
    # f'(x) = [f(x+h) - f(x-h)] / (2h) = [f(x+h) - f(x-h)] / (2/frame_rate) = [f(x+h) - f(x-h)] * frame_rate/2
    eye_tracking["speed_x"] = np.nan
    eye_tracking["speed_y"] = np.nan
    eye_tracking["speed"] = np.nan

    for i in eye_tracking.index:
        if i-1 in eye_tracking.index and i+1 in eye_tracking.index:
            d_x_deg = (eye_tracking.at[i+1, "x_pos_deg"] - eye_tracking.at[i-1, "x_pos_deg"]) * FRAME_RATE/2
            d_y_deg = (eye_tracking.at[i+1, "y_pos_deg"] - eye_tracking.at[i-1, "y_pos_deg"]) * FRAME_RATE/2
            eye_tracking.at[i, "speed_x"] = d_x_deg
            eye_tracking.at[i, "speed_y"] = d_y_deg
            eye_tracking.at[i, "speed"] = np.sqrt(d_x_deg**2 + d_y_deg**2)

    # Set the index (but keep "frame" as a column)
    eye_tracking = eye_tracking.set_index("frame", drop=False)

    return eye_tracking

# =========================================================================


"""
Load information about a particular session from the Brain Observatory, along with saccades,
and cache it in a more quickly-loadable file.
"""
def job(boc, ophys_exp):
    # Example ophys_exp:
    # {'id': 649409874, 'imaging_depth': 175, 'targeted_structure': 'VISpm', 'cre_line': 'Vip-IRES-Cre', 'reporter_line': 'Ai148(TIT2L-GC6f-ICL-tTA2)', 'acquisition_age_days': 100, 'experiment_container_id': 646959440, 'session_type': 'three_session_A', 'donor_name': '350249', 'specimen_name': 'Vip-IRES-Cre;Ai148-350249', 'fail_eye_tracking': False}

    session_id = ophys_exp["id"]
    start_time = time.time()
    # print(f"Starting job for session {session_data['session_id']}.")

    try:
        # Load ophys experiment
        ophys_experiment_data = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
        flu = ophys_experiment_data.get_corrected_fluorescence_traces()[1]
        dff_traces = ophys_experiment_data.get_dff_traces()[1]
        running_speed = ophys_experiment_data.get_running_speed()[0] # dxcm, dxtime
        cre = match_cre_line(ophys_exp)

        # Eye tracking
        eye_tracking = get_eye_tracking(session_id, len_2p=flu.shape[1]) # TODO: Change to new API when ready
        
        # Experiment start and end times
        stim_epoch = ophys_experiment_data.get_stimulus_epoch_table()
        start = max(stim_epoch["start"].iloc[0], eye_tracking.index.min())
        end = min(stim_epoch["end"].iloc[-1], eye_tracking.index.max()) + 1

        # Load saccades
        saccades = get_saccades_from_gaze_traces(
            eye_tracking, start, end, x_deg_col="x_pos_deg", y_deg_col="y_pos_deg", speed_col="speed",
            nan_padding_sec=SACCADE_NAN_PADDING_SEC, frame_rate=FRAME_RATE, noise_padding_sec=SACCADE_NOISE_PADDING_SEC, noise_std_thresh=SACCADE_NOISE_STD_THRESH)
        
        # Build dictionary containing information about this session
        result = {
            "session_data": {
                "session_id": session_id,
                "container_id": ophys_exp["experiment_container_id"],
                "session_type": ophys_exp["session_type"],
                "cre_line": cre.get_full_cre_name(),
                "cre_abbrev": str(cre),
                "targeted_structure": ophys_exp["targeted_structure"],
            },
            "metadata": ophys_experiment_data.get_metadata(),
            "time_taken": time.time() - start_time,
            "start_end": (start, end),
            "n_neurons": len(flu),
            "cell_specimen_ids": ophys_experiment_data.get_cell_specimen_ids(),
            "corrected_fluorescence_traces": flu,
            "dff_traces": dff_traces.copy(),
            # "saccades_all_flagged": all_saccades,
            # "saccades_valid": valid_saccades,
            # "saccades_frac_valid": saccades_frac_valid,
            "saccades": saccades,
            "running_speed": running_speed,
            "eye_tracking": eye_tracking,
            "stim_epoch": stim_epoch,
            "stim_tables": {
                stim: ophys_experiment_data.get_stimulus_table(stim)
                for stim in stim_epoch["stimulus"].unique()
            },
            # "events": boc.get_ophys_experiment_events(session_id),
        }

        return result
    except Exception as e:
        if isinstance(e, EpochSeparationException):
            print(f"Failed job for session {session_id}. (EpochSeparationException: {str(e)})")
        else:
            print(f"Failed job for session {session_id}.")
            traceback.print_exc()
            print(f"Error for {session_id} above.")

        return None

"""
Runs the above data-loading process in parallel.
"""
if __name__ == "__main__":
    process = ParallelProcess(CELL_DATA_SAVE_DIR)

    def output_handler(result):
        if result is None:
            return

        session_id = result["session_data"]["session_id"]
        process.write_pickle_file_output(f"{session_id}.pickle", result)
        
        # print(psutil.Process(os.getpid()).memory_info().rss / 1e9, "GB memory used in main thread -- process", os.getpid())

    boc = BrainObservatoryCache(manifest_file=BRAIN_OBSERVATORY_CACHE_MANIFEST_FILE)
    args = []

    for ophys_exp in boc.get_ophys_experiments(require_eye_tracking=True):
        args.append((boc, ophys_exp))

    print(f"There are {len(args)} total sessions to process.")
    process.run(job, args, output_handler)