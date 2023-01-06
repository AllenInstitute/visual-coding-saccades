import numpy as np
import pandas as pd
from os import path
from scipy import stats
from tqdm import tqdm
import pickle
import glob
import random

from eye_tracking import get_direction_moved

from helpers.parallel_process import ParallelProcess
from experiment.cre_line import match_cre_line

from saccade_config import *


def get_all_cell_diffs(dff, cell_indices, frame, slice_after, slice_before):
    frames_after = slice(frame+slice_after.start, frame+slice_after.stop)
    frames_before = slice(frame+slice_before.start, frame+slice_before.stop)
    return dff[cell_indices, frames_after].mean(axis=1) - dff[cell_indices, frames_before].mean(axis=1)


"""
Compute the p values for each cell by bootstrapping random samples and comparing it to the saccade-triggered samples
"""
def get_cell_p_values(datas, cell_indices, saccades, saccade_responses, slice_after, slice_before, n_boot=1000, n_samples_per_boot=100):
    mean_saccade_response = saccade_responses.mean(axis=1)

    # mean_sample = np.empty(len(cell_indices), dtype=saccade_responses.dtype)

    # Used to sample sessions according to the # saccades in each session
    data_indices = [s[0] for s in saccades]
    pad = 30*3
    data_ranges = [(data["start_end"][0]+pad, data["start_end"][1]-pad) for data in datas]
    
    def sample():
        rand_data_i = random.choice(data_indices)
        data = datas[rand_data_i]
        random_frame = random.randrange(*data_ranges[rand_data_i])
        return get_all_cell_diffs(data["dff_traces"], cell_indices[:, rand_data_i], random_frame, slice_after, slice_before)

    # Bootstrap the distribution
    p_values = np.zeros(len(cell_indices))
    # boot_dist = np.zeros((len(cell_indices), n_boot))

    for b in range(n_boot):
        # 35 iterations/s
        mean_sample = np.mean(
            [sample() for s in range(n_samples_per_boot)],
            axis=0
        )

        # 34 iterations/s
        # mean_sample[:] = 0
        # for s in range(n_samples_per_boot):
        #     mean_sample += sample()
        # mean_sample /= n_samples_per_boot
        
        # p = 0 <==> all responses above boot
        p_values += (mean_saccade_response < mean_sample)

        # boot_dist[:, b] = mean_sample

    return p_values / n_boot
    
    # p_values = np.mean(mean_saccade_response[:, np.newaxis] < boot_dist, axis=1)
    # return p_values



def get_cell_saccade_responses(datas, cell_indices, saccades, slice_after, slice_before):
    sacc_responses_all = []
    sacc_responses_left = []
    sacc_responses_right = []

    for data_i, ss, se in saccades:
        data = datas[data_i]
        dff = data["dff_traces"]
        responses = get_all_cell_diffs(dff, cell_indices[:, data_i], ss, slice_after, slice_before)
        dir = get_direction_moved(data["eye_tracking"], ss, se)
        sacc_responses_all.append(responses)
        if dir == "L": sacc_responses_left.append(responses)
        elif dir == "R": sacc_responses_right.append(responses)

    return np.column_stack(sacc_responses_all), np.column_stack(sacc_responses_left), np.column_stack(sacc_responses_right)



def classify_response(p, thresh):
    # return np.logical_or(p < thresh, p > 1-thresh)
    # return p < thresh or p > 1-thresh
    if p < thresh:
        # Significant number of responses ABOVE bootstrap distribution (ENHANCD)
        return 1
    elif p > 1-thresh:
        # Significant number of responses BELOW bootstrap distribution (SUPPRESSED)
        return -1
    else:
        # No significant response
        return 0


def response_metric(a, b):
    return 0 if a+b == 0 else (a-b)/(a+b)


def job(session_filenames, cell_specimen_ids, slice_after, slice_before, corrected_p_thresh, n_boot):
    datas = []

    for filename in session_filenames:
        with open(filename, "rb") as file:
            data = pickle.load(file)
            datas.append(data)
    
    n_left, n_right = 0, 0
    saccade_padding = 2*30
    saccades = []
    sacc_dirs = []

    for data_i, data in enumerate(datas):
        start, end = data["start_end"]

        for ss, se in data["saccades"]:
            if ss-start < saccade_padding or end-se < saccade_padding:
                # Ignore saccades that happen too close to the end
                continue
            elif len(saccades) > 0 and saccades[-1][0] == data_i and ss - saccades[-1][2] <= saccade_padding:
                # Ignore saccades that happen too close to each other
                continue

            saccades.append((data_i, ss, se))
            dir = get_direction_moved(data["eye_tracking"], ss, se)
            sacc_dirs.append(dir)
            if dir == "L": n_left += 1
            elif dir == "R": n_right += 1

    # Ignore sessions with not enough saccades    
    if n_left < 15 or n_right < 15:
        print(f"Not enough saccades (lost {len(cell_specimen_ids)} cells; {n_left} L / {n_right} R; sessions {[data['session_data']['session_id'] for data in datas]})")
        return None
    
    sacc_dirs = np.array(sacc_dirs)
    cell_indices = np.empty((len(cell_specimen_ids), len(datas)), dtype=int) # the indices of cells in each session

    for i, specimen_id in enumerate(cell_specimen_ids):
        for j, data in enumerate(datas):
            cell_indices[i, j] = np.where(data["cell_specimen_ids"] == specimen_id)[0][0]

    sdata = datas[0]["session_data"]

    sacc_responses_all, sacc_responses_L, sacc_responses_R = get_cell_saccade_responses(datas, cell_indices, saccades, slice_after, slice_before)    
    p_values = get_cell_p_values(datas, cell_indices, saccades, sacc_responses_all, slice_after, slice_before, n_boot=n_boot, n_samples_per_boot=len(saccades))

    data_indices = [s[0] for s in saccades]
    single_bootstraps = np.zeros((len(cell_specimen_ids), n_boot), dtype=sacc_responses_all.dtype)
    for b in range(n_boot):
        rand_data_i = random.choice(data_indices)
        data = datas[rand_data_i]
        start, end = data["start_end"]
        random_frame = random.randrange(start+saccade_padding, end-saccade_padding)
        single_bootstraps[:, b] = get_all_cell_diffs(data["dff_traces"], cell_indices[:, rand_data_i], random_frame, slice_after, slice_before)

    # shape is (n_cells, n_saccades)
    
    mean_L_response = sacc_responses_L.mean(axis=1)
    mean_R_response = sacc_responses_R.mean(axis=1)
    p_thresh_dir = 0.05

    cell_info = []

    # this is used for alternative tests to mimic Miura
    dff_before = np.empty((len(cell_specimen_ids), len(saccades)))
    dff_after = np.empty((len(cell_specimen_ids), len(saccades)))
    for i, (data_i, ss, se) in enumerate(saccades):
        dff = datas[data_i]["dff_traces"]
        cell_idx = cell_indices[:, data_i]
        dff_before[:, i] = np.mean(dff[cell_idx, (ss-20):(ss-10)], axis=1)
        dff_after[:, i] = np.mean(dff[cell_idx, (ss+0):(ss+10)], axis=1)

    for cell_index, specimen_id in enumerate(cell_specimen_ids):
        p = p_values[cell_index]
        response_classification = classify_response(p, corrected_p_thresh)

        # if response_classification != 0: # The cell has a response
        n_L_enhanced, n_L_suppressed = 0, 0
        n_R_enhanced, n_R_suppressed = 0, 0

        for i in range(sacc_responses_L.shape[1]):
            pe = np.mean(sacc_responses_L[cell_index, i] < single_bootstraps[cell_index]) # = 0 <==> all responses above boot
            if pe < p_thresh_dir: n_L_enhanced += 1
            elif pe > 1-p_thresh_dir: n_L_suppressed += 1
        
        for i in range(sacc_responses_R.shape[1]):
            pe = np.mean(sacc_responses_R[cell_index, i] < single_bootstraps[cell_index]) # = 0 <==> all responses above boot
            if pe < p_thresh_dir: n_R_enhanced += 1
            elif pe > 1-p_thresh_dir: n_R_suppressed += 1

        frac_L_enhanced = n_L_enhanced / sacc_responses_L.shape[1]
        frac_L_suppressed = n_L_suppressed / sacc_responses_L.shape[1]
        frac_R_enhanced = n_R_enhanced / sacc_responses_R.shape[1]
        frac_R_suppressed = n_R_suppressed / sacc_responses_R.shape[1]
        metric_enhanced = response_metric(frac_R_enhanced, frac_L_enhanced)
        metric_suppressed = response_metric(frac_R_suppressed, frac_L_suppressed)
        metric_L = response_metric(frac_L_enhanced, frac_L_suppressed)
        metric_R = response_metric(frac_R_enhanced, frac_R_suppressed)

        cre = match_cre_line(datas[0])

        cell_info.append({
            "specimen_id": specimen_id,
            "cre_line": cre.get_full_cre_name(),
            "cre_abbrev": str(cre),
            "depth": data["metadata"]["imaging_depth_um"],
            "targeted_structure": sdata["targeted_structure"],
            "container_id": sdata["container_id"],
            "session_indices": str(tuple([ (datas[data_i]["session_data"]["session_id"], cell_indices[cell_index, data_i]) for data_i in range(len(datas)) ])),
            "response_classification": response_classification,
            "p_value": p,
            "n_left": n_left,
            "n_right": n_right,
            "mean_left_response": mean_L_response[cell_index],
            "mean_right_response": mean_R_response[cell_index],
            "median_left_response": np.median(sacc_responses_L[cell_index]),
            "median_right_response": np.median(sacc_responses_R[cell_index]),
            "direction_selectivity": response_metric(mean_R_response[cell_index], mean_L_response[cell_index]),
            # "left_saccade_response": metric_L,
            # "right_saccade_response": metric_R,
            "frac_left_significant": (n_L_enhanced + n_L_suppressed) / sacc_responses_L.shape[1],
            "frac_right_significant": (n_R_enhanced + n_R_suppressed) / sacc_responses_R.shape[1],
            "wilcoxon_signed_rank_p": stats.wilcoxon(dff_before[cell_index], dff_after[cell_index])[1],
            "ranksum_p_by_direction": stats.ranksums(dff_after[cell_index, sacc_dirs == "L"], dff_after[cell_index, sacc_dirs == "R"])[1],
        })

    return cell_info



def output_handler(process: ParallelProcess, df_cells: pd.DataFrame, result):
    if result is None:
        return
    
    cell_info = result

    for cell in cell_info:
        specimen_id = cell["specimen_id"]

        for k in cell.keys():
            if k != "specimen_id":
                df_cells.at[specimen_id, k] = cell[k]
    
    df_cells.to_csv(path.join(process.get_save_dir(), "cells.csv"), index_label="specimen_id")



if __name__ == "__main__":
    files_to_process = glob.glob(path.join(LOADED_CELL_DATA_DIR, f"*.pickle"))
    p_thresh = 0.05
    bonferroni_correct = 100
    corrected_p_thresh = p_thresh / bonferroni_correct
    n_boot = int(20 * (1/corrected_p_thresh))
    print(f"Using {n_boot:,} bootstraps; corrected p-value threshold: {corrected_p_thresh:.2e}")
    slice_after = slice(0, 10)
    slice_before = slice(-45, -15)

    # Load a list of sessions for every cell
    cell_to_sessions = {}

    for filename in tqdm(files_to_process):
        with open(filename, "rb") as file:
            data = pickle.load(file)
        
        session_id = data["session_data"]["session_id"]

        for cell in data["cell_specimen_ids"]:
            if cell in cell_to_sessions:
                sessions = cell_to_sessions[cell]
                if session_id not in sessions:
                    sessions.append(session_id)
            else:
                cell_to_sessions[cell] = [session_id]

    print(f"There are {len(cell_to_sessions):,} cells across {len(files_to_process)} sessions.")

    # Determine a list of cells for every combination of sessions
    sessions_to_cells = {}

    for cell, sessions in cell_to_sessions.items():
        key = tuple(sorted(sessions))

        if key in sessions_to_cells:
            sessions_to_cells[key].append(cell)
        else:
            sessions_to_cells[key] = [cell]

    print(f"There are {len(sessions_to_cells):,} different combinations of sessions that load all cells.")
    
    args = []

    for sessions, cell_specimen_ids in sessions_to_cells.items():
        session_filenames = [ path.join(LOADED_CELL_DATA_DIR, f"{session_id}.pickle") for session_id in sessions]
        args.append((session_filenames, cell_specimen_ids, slice_after, slice_before, corrected_p_thresh, n_boot))

    df_cells = pd.DataFrame()
    process = ParallelProcess(CLASSIFIED_CELLS_SAVE_DIR)

    def output_handler_wrapper(result):
        return output_handler(process, df_cells, result)

    process.run(job, args, output_handler_wrapper)