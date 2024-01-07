import sys
sys.path.append("./")

import time
import traceback

import os
import gc
import time
import traceback

import h5py
import numpy as np
import pandas as pd

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.brain_observatory_exceptions import EpochSeparationException

from saccade_config import *
from experiment.cre_line import match_cre_line
from helpers.parallel_process_old import ParallelProcess
from eye_tracking.eye_tracking import get_saccades_from_gaze_traces, get_eye_speed


def get_eye_tracking(session_id, boc: BrainObservatoryCache):
    """
    Load the eye tracking data from eye tracking and time sync files, and synchronize the data points
    to the 2P imaging (the length of the 2P imaging is set in len_2p). Return a pd.DataFrame of
    synced eye tracking data.
    """
    eye_tracking_arr = boc.get_eye_tracking(session_id)

    # Build the data frame
    columns = ["frame", "eye_area", "pupil_area", "x_pos_deg", "y_pos_deg"]
    eye_tracking = pd.DataFrame(data={
        "frame": eye_tracking_arr[:, 0].astype(int),
        "eye_area": eye_tracking_arr[:, 1],
        "pupil_area": eye_tracking_arr[:, 2],
        "x_pos_deg": eye_tracking_arr[:, 3],
        "y_pos_deg": eye_tracking_arr[:, 4]
    }, columns=columns)

    # Calculate rate of change of position using central difference approximation of f'(x)
    # f'(x) = [f(x+h) - f(x-h)] / (2h) = [f(x+h) - f(x-h)] / (2/frame_rate) = [f(x+h) - f(x-h)] * frame_rate/2
    eye_tracking["speed"] = get_eye_speed(eye_tracking, x_deg_col="x_pos_deg", y_deg_col="y_pos_deg", frame_rate=30)

    # Set the index (but keep "frame" as a column)
    eye_tracking = eye_tracking.set_index("frame", drop=False)

    return eye_tracking

# =========================================================================


def job(boc, ophys_exp):
    """
    Load information about a particular session from the Brain Observatory, along with saccades,
    and cache it in a more quickly-loadable file.
    """
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
        eye_tracking = get_eye_tracking(session_id, boc=boc)
        
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
        if ophys_exp["fail_eye_tracking"]:
            continue

        # NOTE: If you don't care about certain experiments (e.g., only certain cre line), then add more checks here to ignore
        # otherwise it will download all sessions, which can take a lot of space

        args.append((boc, ophys_exp))

    print(f"There are {len(args)} total sessions to process.")
    process.run(job, args, output_handler, parallel=True)