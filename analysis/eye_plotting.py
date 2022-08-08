import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from os import path
import scipy.stats as st
from tqdm.notebook import tqdm
from collections import defaultdict
import cv2
import imageio
import random

from eye_tracking.eye_tracking import get_eye_speed_outlier_threshold, get_degrees_moved

STIM_COLORS = {
    "drifting_gratings": "darkorange",
    # "locally_sparse_noise": "purple",
    "locally_sparse_noise": "fuchsia",
    "locally_sparse_noise_4deg": "fuchsia",
    "locally_sparse_noise_8deg": "mediumvioletred",
    "natural_movie_one": "darkgreen",
    "natural_movie_two": "green",
    "natural_movie_three": "limegreen",
    "natural_scenes": "blue",
    "spontaneous": "red",
    "static_gratings": "gold"
}


STIM_NAMES = {
    "drifting_gratings": "Drifting Gratings",
    # "locally_sparse_noise": "Locally Sparse Noise",
    "locally_sparse_noise": "Locally Sparse Noise ($4^\circ$)",
    "locally_sparse_noise_4deg": "Locally Sparse Noise ($4^\circ$)",
    "locally_sparse_noise_8deg": "Locally Sparse Noise ($4^\circ$)",
    "natural_movie_one": "Natural Movie 1",
    "natural_movie_two": "Natural Movie 2",
    "natural_movie_three": "Natural Movie 3",
    "natural_scenes": "Natural Scenes",
    "spontaneous": "Spontaneous",
    "static_gratings": "Static Gratings"
}


STIM_ABBREV = {
    "drifting_gratings": "DG",
    "locally_sparse_noise": "LSN-4",
    "locally_sparse_noise_4deg": "LSN-4",
    "locally_sparse_noise_8deg": "LSN-8",
    "natural_movie_one": "NM-1",
    "natural_movie_two": "NM-2",
    "natural_movie_three": "NM-3",
    "natural_scenes": "NS",
    "spontaneous": "S",
    "static_gratings": "SG"
}


STIM_ORDERING = ["S", "LSN-4", "LSN-8", "SG", "DG", "NM-1", "NM-2", "NM-3", "NS"]

        
# Add color abbreviations to stim colors
for stim in list(STIM_COLORS.keys()):
    STIM_COLORS[STIM_ABBREV[stim]] = STIM_COLORS[stim]
STIM_COLORS["LSN-All"] = STIM_COLORS["LSN-4"]
STIM_COLORS["NM-All"] = STIM_COLORS["NM-1"]



SESSION_TYPE_ABBREV = {
    "three_session_A": "A",
    "three_session_B": "B",
    "three_session_C": "C",
    "three_session_C2": "C2"
}


AREA_ORDERING = ["VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl"]
AREA_ABBREV = {
    "VISp": "V1",
    "VISl": "LM",
    "VISal": "AL",
    "VISpm": "PM",
    "VISam": "AM",
    "VISrl": "RL",
}

def plot_eye_data(exp_data, start, end, show_saccades=True, saccades=None, ax=None, ax_eye_pos=None, labels=True):
    if ax is None:
        fig = plt.figure(figsize=(9,6))
        ax = fig.gca()

    # Plot vertical frame marker lines
    # for x in range(start, end):
    #     if x % 250 == 0:
    #         lw = 0.5
    #         if x % 2000 == 0: lw = 1.5
    #         elif x % 1000 == 0: lw = 1
    #         plt.axvline(x, color="gray", lw=lw)

    # Get the relevant data rows for our desired frames
    eye_data_rows = exp_data.get_eye_tracking_rows(start, end)

    # Plot the stimuli regions
    x_axis = np.arange(start, end)
    # stim_in_window = natural_scene_table[((natural_scene_table["start"] >= start) | (natural_scene_table["end"] >= start)) & ((natural_scene_table["end"] <= end) | (natural_scene_table["start"] <= end))]
    # n_stim = len(stim_in_window)
    # if n_stim < 30:
    #     colors = ["red", "blue"]
    #     for i in range(n_stim):
    #         plt.axvspan(xmin=stim_in_window.start.iloc[i], xmax=stim_in_window.end.iloc[i], color=colors[i % len(colors)], alpha=0.15)

    if ax_eye_pos is None:
        ax.plot(x_axis, eye_data_rows["eye_area"].values*100 + 57, c="green", label="Eye area")
        ax.plot(x_axis, eye_data_rows["pupil_area"].values*1000 + 45, c="orange", label="Pupil area")
        ax.plot(x_axis, eye_data_rows["x_pos_deg"].values + 20, c="blue", label="x degree")
        ax.plot(x_axis, eye_data_rows["y_pos_deg"].values + 10, c="red", label="y degree")
        ax.set_ylim(-5, 65)
        
        # TO PLOT SAMPLE SACCADE
#         ax.plot(x_axis, eye_data_rows["x_pos_deg"].values + 15, c="blue", label="x degree")
#         ax.plot(x_axis, eye_data_rows["y_pos_deg"].values + 10, c="red", label="y degree")
#         ax.plot(x_axis, eye_data_rows["v_deg"].values / 10 + 44, label="Eye velocity (deg/sec)", color="green")
#         ax.axhline(44 + get_eye_speed_outlier_threshold(exp_data) / 10, color="lime", linestyle="dashed", linewidth=1)
#         ax.set_xlim(37400, 37800)
    else:
        ax.plot(x_axis, eye_data_rows["eye_area"].values*100 + 20, c="green", label="Eye area")
        ax.plot(x_axis, eye_data_rows["pupil_area"].values*1000 + 10, c="orange", label="Pupil area")
        ax.set_ylim(-2, 29)
        ax_eye_pos.plot(x_axis, eye_data_rows["x_pos_deg"].values, c="blue", label="x degree")
        ax_eye_pos.plot(x_axis, eye_data_rows["y_pos_deg"].values, c="red", label="y degree")
        ax_eye_pos.set_xlim(start, end-1)
#         ax_eye_pos.set_ylim(-10, 35)
        ax_eye_pos.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)
        ax_eye_pos.set_ylabel("Degrees", fontsize=12)
    
    ax.plot(x_axis, (0.2*exp_data.get_running_speed()[start:end]), c="purple", label="Running speed")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)

    # plt.plot(x_axis, eye_data_rows["v_deg_x"].values + 20, label="v_x (deg/s)")
    # plt.plot(x_axis, eye_data_rows["v_deg_y"].values + 10, label="v_y (deg/s)")

    title = f"Eye movements for frames {start}-{end}"

    if show_saccades:
        if saccades is None: saccades = exp_data.get_saccades()
        title = f"{title} ({len(saccades)} saccades)"
        for (ss, se, conf) in saccades:
            (ax if ax_eye_pos is None else ax_eye_pos).axvspan(ss, se, color="yellow", alpha=0.5)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(start, end)
    ax.get_yaxis().set_visible(False)

    if ax_eye_pos is None:
        ax.set_xlabel("Frame number", fontsize=12)
        plt.setp(ax.get_xticklabels(), fontsize=12)
    else:
        # Only show x tick labels on position plot
        plt.setp(ax.get_xticklabels(), visible=False)
        ax_eye_pos.set_xlabel("Frame number", fontsize=12)
        plt.setp(ax_eye_pos.get_xticklabels(), fontsize=12)
        
    return {
        "mean_eye_area": float(eye_data_rows["eye_area"].mean()), # convert to python data type
        "mean_pupil_area": float(eye_data_rows["pupil_area"].mean()),
        "mean_x_pos_deg": float(eye_data_rows["x_pos_deg"].mean()),
        "mean_y_pos_deg": float(eye_data_rows["y_pos_deg"].mean())
    }
    


def plot_eye_video(session_id, video_file, eye_data, running_speed, start, end, make_gif=False, gif_dir="gif_temp", gif_file_name=None):
    # video_file = exp_data.get_session_data()["eye_video_file"]
    
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file {video_file} for session {session_id} was not found")
    
    vid = cv2.VideoCapture(video_file)

    n_frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames in video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) # video width
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # video heights

    print(f"Video has {n_frames_total} total frames ({width} x {height})")

    r = 25
    eye_map_x_axis = ([-r, r], [0, 0])
    eye_map_y_axis = ([0, 0], [-r, r])

    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, aspect="equal", adjustable="box") # set aspect ratio
    ax3 = fig.add_subplot(212)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.set_title("Eye position/size")
    ax2.plot(*eye_map_x_axis, "k-", linewidth=2)
    ax2.plot(*eye_map_y_axis, "k-", linewidth=2)

    eye_data_rows = eye_data.loc[start:end-1]

    # Plot the stimuli regions
    # x_axis = np.arange(start, end)
    # stim_in_window = natural_scene_table[((natural_scene_table["start"] >= start) | (natural_scene_table["end"] >= start)) & ((natural_scene_table["end"] <= end) | (natural_scene_table["start"] <= end))]
    # n_stim = len(stim_in_window)
    # if n_stim < 30:
    #     for i in range(n_stim):
    #         ax3.axvspan(xmin=stim_in_window.start.iloc[i], xmax=stim_in_window.end.iloc[i], color="gray", alpha=(0.1 if i % 2 == 1 else 0.2))

    x_axis = np.arange(start, end)
    ax3.plot(x_axis, eye_data_rows["eye_area"].values*100 + 60, c="green", label="Eye area")
    ax3.plot(x_axis, eye_data_rows["pupil_area"].values*1000 + 50, c="orange", label="Pupil area")
    ax3.plot(x_axis, eye_data_rows["x_pos_deg"].values + 20, c="blue", label="x degree")
    ax3.plot(x_axis, eye_data_rows["y_pos_deg"].values + 10, c="red", label="y degree")

    # Plot running speed
    ax3.plot(x_axis, (0.2*running_speed[start:end]), c="purple", label="Running speed")
    ax3.set_title(f"Eye movements for frames {start}-{end}")
    ax3.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)
    fig.tight_layout()
    ax3.get_yaxis().set_visible(False)
    for item in ax3.get_xticklabels(): item.set_fontsize(12)
    ax3.set_xlim(start, end-1)
    curr_frame_line = ax3.axvline(start, color="k")

    # Skip to start
    vid.set(cv2.CAP_PROP_POS_FRAMES, start-1)
    gif_files = []

    if make_gif and not os.path.exists(gif_dir):
        os.mkdir(gif_dir)

    point = ax2.plot(0, 0, c="r")[0]

    for frame in range(start, end):
        flag, image = vid.read()

        ax1.clear()
        ax1.imshow(image)
        ax1.set_title(f"Frame {frame} {image.shape}")

        # Plot eye position
        xpos = eye_data["x_pos_deg"].loc[frame]
        ypos = eye_data["y_pos_deg"].loc[frame]
        if np.isnan(xpos) or np.isnan(ypos):
            point.set_data(0, 0)
            point.set_marker("X")
            point.set_markersize(20)
        else:
            point.set_data(xpos, ypos)
            point.set_marker("o")
            point.set_markersize(eye_data["pupil_area"].loc[frame] * 5000)

        # Update the current frame line in the third subplot
        curr_frame_line.set_xdata(frame)

        # Update the figure
        fig.canvas.draw()

        # Save gif file frame
        if make_gif:
            gif_file = os.path.join(gif_dir, f"{frame-start+1}.png")
            plt.savefig(gif_file)
            gif_files.append(gif_file)
    #     if cv2.waitKey(1) == 27:
    #         break

    # Build GIF
    if make_gif:
        if gif_file_name is None:
            gif_file_name = f"movie-{start}-{end}.gif"

        with imageio.get_writer(gif_file_name, mode="I") as writer:
            for file in gif_files:
                image = imageio.imread(file)
                writer.append_data(image)

        for file in gif_files:
            os.remove(file)


def plot_dff(exp_data, ax=None, show_saccades=False, saccades=None):
    # Plot the dF/F
    timestamps, dff = exp_data.get_ophys_exp_data().get_dff_traces()
    
    if ax is None:
        fig = plt.figure(figsize=(9,6))
        ax = fig.gca()
        
    n_neurons = min(len(dff), 20)
    ax.set_title(f"dF/F traces for {'all' if n_neurons == len(dff) else 'first'} {n_neurons} cells", fontsize=12)
    for i in range(min(n_neurons, 20)):
        ax.plot(dff[i,:]+(i*2), color='gray')

    # Plot running speed below
    ax.plot((0.2*exp_data.get_running_speed()) - 8)

    # Highlight the different stimulus types
    if not show_saccades:
        for i, row in exp_data.get_ophys_exp_data().get_stimulus_epoch_table().iterrows():
            color = STIM_COLORS[row["stimulus"]]
            ax.axvspan(xmin=row["start"], xmax=row["end"], color=color, alpha=0.25)
    
    # Highlight the saccades
    if show_saccades:
        if saccades is None: saccades=exp_data.get_saccades()
        color = "yellow"
        if len(saccades) < 20:
            color = "magenta"
            alpha = 1
        elif len(saccades) < 50:
            alpha = 0.3
        else:
            alpha = 0.2
        for (ss, se, conf) in saccades:
            ax.axvspan(ss, se, color=color, alpha=alpha)
        
    ax.set_xlabel("Frame number", fontsize=12)
    ax.set_ylim(-10, n_neurons*2 + 5)
    ax.set_xlim(0, dff.shape[1])
    
    return {} # no data



def plot_experiment_saccades(exp_data, saccades=None, start=-1, end=-1, ax=None, ax_hist=None, ax_bar_count=None, ax_bar_rate=None, labels=True, abbreviate=True):
    if start == -1:
        start = exp_data.get_start_and_end()[0]
    if end == -1:
        end = exp_data.get_start_and_end()[1]-1
        
    if ax is None:
        fig = plt.figure(figsize=(9,3))
        ax = fig.gca()
        
    if saccades is None: saccades = exp_data.get_saccades()
    eye_data_rows = exp_data.get_eye_tracking_rows(start, end)
    stim_epoch = exp_data.get_ophys_exp_data().get_stimulus_epoch_table()
    stim_saccades = {stim: {"n_saccades": 0, "total_frames": 0} for stim in STIM_COLORS.keys()}

    # print("Color key:")
    
    for stim_name in stim_epoch.stimulus.unique():
        stim = stim_epoch[stim_epoch.stimulus == stim_name]
        color = STIM_COLORS.get(stim_name, "pink")
    #     print(f" - {stim_name}: {color}")

        for j in range(len(stim)):
            stim_start = stim.start.iloc[j]
            stim_end = stim.end.iloc[j]
            ax.axvspan(stim_start, stim_end, color=color, alpha=0.6)
            stim_saccades[stim_name]["total_frames"] += stim_end - stim_start

            # Find saccades during this window
            for (saccade_start, saccade_end, conf) in saccades:
                if saccade_start > stim_start and saccade_end < stim_end:
                    stim_saccades[stim_name]["n_saccades"] += 1
                elif saccade_start > stim_end:
                    break # Since saccades are ordered by frame

    # print("Stim saccades:", stim_saccades)
    stim_saccade_density = {stim: stim_saccades[stim]["n_saccades"] / float(stim_saccades[stim]["total_frames"]) for stim in stim_saccades.keys() if stim_saccades[stim]["total_frames"] > 0}

#     print("Stimuli with most frequent saccades:")
#     toggle = 0
#     for stim in sorted(stim_saccade_density.keys(), key=lambda x: stim_saccade_density[x], reverse=True):
#         n_saccades = stim_saccades[stim]["n_saccades"]
#         frames = stim_saccades[stim]["total_frames"]
#         color = STIM_COLORS.get(stim, "pink")
#         print(f" - {stim} ({color}): {n_saccades:,} saccades in {frames:,} total frames (~{int(frames / n_saccades)} frames/saccade)")
        
#         row = stim_epoch[stim_epoch.stimulus == stim].iloc[0]
#         middle = (row["start"] + row["end"]) / 2.0
#         text = f"{stim}: {n_saccades:,} saccades\n{frames:,} total frames\n(~{int(frames / n_saccades)} frames/saccade)"
#         height = [0.8, 0.5, 0.2][toggle]
#         ax.text(middle, height, text, color="black", fontsize=10, ha="center",
#                 va="center", bbox={"facecolor": "white", "edgecolor": color, "alpha": 1, "pad": 6})
#         toggle = (toggle + 1) % 3


    toggle = 0
    for stim in sorted(stim_saccade_density.keys(), key=lambda x: stim_epoch[stim_epoch.stimulus == x]["start"].iloc[0]):
        n_saccades = stim_saccades[stim]["n_saccades"]
        frames = stim_saccades[stim]["total_frames"]
        color = STIM_COLORS.get(stim, "pink")
#         print(f" - {stim} ({color}): {n_saccades:,} saccades in {frames:,} total frames")
        
        row = stim_epoch[stim_epoch.stimulus == stim].iloc[0]
        middle = (row["start"] + row["end"]) / 2.0
        text = f"{STIM_ABBREV[stim]}: {n_saccades:,} saccades\n{frames:,} total frames"
        if n_saccades > 0:
            text = f"{text}\n(~{int(frames / n_saccades)} frames/saccade)"
        height = [0.8, 0.5, 0.2][toggle]
        ax.text(middle, height, text, color="black", fontsize=10, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": color, "linewidth": 3, "alpha": 1, "pad": 3})
        toggle = (toggle + 1) % 3
        
    # Plot the saccades
    for (ss, se, conf) in saccades:
        ax.axvspan(ss, se, color="black", alpha=0.3)

    if labels:
        ax.set_title(f"Saccades during different stimulus types ({len(saccades)} saccades total)", fontsize=12)
        # plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)
        ax.set_xlabel("Frame number", fontsize=12)
    ax.get_yaxis().set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    ax.set_xlim(start, end-1)
    ax.set_ylim(0, 1)
        
    if ax_hist is not None:
        x_bar = []
        y_bar = []
        window_width = exp_data.get_frame_rate() * 60
        for window_start in range(start, end, window_width):
            n_saccades_in_window = 0
            for (ss, se, conf) in saccades:
                if window_start <= ss:
                    if ss < window_start + window_width:
                        n_saccades_in_window += 1
                    else:
                        break

            x_bar.append(window_start)
            y_bar.append(n_saccades_in_window)

        for stim_name in stim_epoch.stimulus.unique():
            stim = stim_epoch[stim_epoch.stimulus == stim_name]
            color = STIM_COLORS.get(stim_name, "pink")
            for j in range(len(stim)):
                stim_start = stim.start.iloc[j]
                stim_end = stim.end.iloc[j]
                ax_hist.axvspan(stim_start, stim_end, color=color, alpha=0.3)

        ax_hist.bar(x_bar, y_bar, width=window_width, align="edge", color="black")
        ax_hist.set_xlim(start, end-1)

    stims = sorted(stim_saccade_density.keys(), key=lambda x: stim_epoch[stim_epoch.stimulus == stim]["start"].iloc[0])
    stim_labels = [STIM_ABBREV[stim] for stim in stims] #if abbreviate else [STIM_NAMES[stim] for stim in stims]
    colors = [STIM_COLORS[stim] for stim in stims]

    if ax_bar_count is not None:
        ax_bar_count.bar(stim_labels, [stim_saccades[stim]["n_saccades"] for stim in stims], color=colors)
        ax_bar_count.set_title("Saccade count by stimulus", fontsize=12)
        
        if abbreviate:
            plt.setp(ax_bar_count.xaxis.get_major_ticks()[1::2], pad=15) # stagger labels
        else:
            plt.setp(ax_bar_count.get_xticklabels(), rotation=45) # horizontalalignment="right"

    if ax_bar_rate is not None:
        # (sacc / frame) * (frame_rate frames / 1 sec) = sacc / sec
        saccade_rates = [float(stim_saccades[stim]["n_saccades"]) / stim_saccades[stim]["total_frames"] * exp_data.get_frame_rate() for stim in stims]
        ax_bar_rate.bar(stim_labels, saccade_rates, color=colors)
        ax_bar_rate.set_title("Saccade frequency (saccades/s)", fontsize=12)
        
        if abbreviate:
            plt.setp(ax_bar_rate.xaxis.get_major_ticks()[1::2], pad=15) # stagger labels
        else:
            plt.setp(ax_bar_rate.get_xticklabels(), rotation=0)
    
    # TODO maybe break down by each particular stimulus type within session (as most stimulus types are shown multiple times)
    return {
        "frames_of_each_stim": {
            stim: int(stim_saccades[stim]["total_frames"]) for stim in stims if stim_saccades[stim]["total_frames"] > 0
        },
        "saccade_count_by_stim": {
            stim: int(stim_saccades[stim]["n_saccades"]) for stim in stims if stim_saccades[stim]["total_frames"] > 0
        },
        "saccade_frequency_by_stim": {
            stim: float(stim_saccades[stim]["n_saccades"]) / stim_saccades[stim]["total_frames"] * exp_data.get_frame_rate() for stim in stims
        }
    }
    
    
def plot_saccade_motions(exp_data, saccades=None, num_to_plot=-1, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
    
    if saccades is None: saccades = exp_data.get_saccades()
    saccades_to_plot = saccades if num_to_plot == -1 or len(saccades) <= num_to_plot else random.sample(saccades, num_to_plot)
    x = []
    y = []
    total = []
    
    for (ss, se, conf) in saccades_to_plot:
        x_deg = exp_data.get_eye_tracking_data("x_pos_deg", frames=(ss, se+1))
        y_deg = exp_data.get_eye_tracking_data("y_pos_deg", frames=(ss, se+1))
        ax.plot(x_deg - x_deg.iloc[0], y_deg - y_deg.iloc[0], color="black", marker=".", alpha=0.2)
        
        # Save data
        dx = x_deg.values[-1] - x_deg.values[0]
        dy = y_deg.values[-1] - y_deg.values[0]
        euclidean = np.sqrt(dx**2 + dy**2)
        x.append(abs(dx))
        y.append(abs(dy))
        total.append(euclidean)
        
    ax.axhline(0, color="black", linestyle="dashed")
    ax.axvline(0, color="black", linestyle="dashed")
    ax.set_title("Saccade motions", fontsize=12)
    ax.set_xlabel("$x$-degree", fontsize=12)
    ax.set_ylabel("$y$-degree", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
#     ax.set_xlim(-12, 12)
#     ax.set_ylim(-12, 12)

    return {
        "saccade_x_movement_mean": float(np.nanmean(x)),
        "saccade_x_movement_std": float(np.nanstd(x)),
        "saccade_y_movement_mean": float(np.nanmean(y)),
        "saccade_y_movement_std": float(np.nanstd(y)),
        "saccade_movement_mean": float(np.nanmean(total)),
        "saccade_movement_std": float(np.nanstd(total))
    }

    
def plot_saccade_overlays(data, saccades=None, window_radius=30, num_to_plot=-1, speed_overlay=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
    
    if saccades is None: saccades = data["saccades"]
    saccades_to_plot = saccades if num_to_plot == -1 or len(saccades) <= num_to_plot else random.sample(saccades, num_to_plot)
    x_axis = np.arange(-window_radius, window_radius+1)
    alpha = 1 if len(saccades_to_plot) <= 5 else 0.25
    
    if speed_overlay:
        # Plot outlier threshold that the saccade detection threshold algorithm uses)
        ax.axhline(get_eye_speed_outlier_threshold(data), color="darkgreen", linestyle="dashed", linewidth=0.5)
    
    for ss, se in saccades_to_plot:
        ss += 1 # make it line up with zero since the saccade onset apparently starts one frame before the movement
        # x_deg = exp_data.get_eye_tracking_data("x_pos_deg", frames=(ss-window_radius, ss+window_radius+1))
        # y_deg = exp_data.get_eye_tracking_data("y_pos_deg", frames=(ss-window_radius, ss+window_radius+1))
        # x_0 = exp_data.get_eye_tracking_data("x_pos_deg", frames=ss)
        # y_0 = exp_data.get_eye_tracking_data("y_pos_deg", frames=ss)
        eye_data = data["eye_tracking"]
        x_deg = eye_data["x_pos_deg"].loc[ss-window_radius:ss+window_radius]
        y_deg = eye_data["y_pos_deg"].loc[ss-window_radius:ss+window_radius]
        x_0 = eye_data["x_pos_deg"].loc[ss]
        y_0 = eye_data["y_pos_deg"].loc[ss]
        ax.plot(x_axis, x_deg - x_0, color="blue", alpha=alpha)
        ax.plot(x_axis, y_deg - y_0, color="red", alpha=alpha)
        
        if speed_overlay:
            v_deg = eye_data["speed"].loc[ss-window_radius:ss+window_radius]
            ax.plot(x_axis, v_deg, color="green", alpha=alpha)
#         ax.axvline(se-ss, alpha=0.1, color="black", linestyle="dashed")

    ax.axvline(0, color="black", linestyle="dashed")
    ax.set_title(f"Saccade motions (blue = $x$; red = $y${'; green = speed (deg/s)' if speed_overlay else ''})", fontsize=12)
    ax.set_xlabel("Frames relative to saccade onset", fontsize=12)
    ax.set_ylabel("Degree", fontsize=12)
    ax.set_xlim(-window_radius, window_radius)
    ax.set_ylim(-15, 15)
    
    return {} # No data to return
    
    
def plot_saccade_speeds(exp_data, saccades=None, window_radius=30, num_to_plot=-1, ax=None, enlarged=False):
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        
    if saccades is None: saccades = exp_data.get_saccades()
    saccades_to_plot = saccades if num_to_plot == -1 or len(saccades) <= num_to_plot else random.sample(saccades, num_to_plot)
    x_axis = np.arange(-window_radius, window_radius+1)
    mean_trace = np.zeros_like(x_axis, dtype=np.float64)
    
    for (ss, se, se) in saccades_to_plot:
        v_deg = exp_data.get_eye_tracking_data("speed", frames=(ss-window_radius, ss+window_radius+1))
        mean_trace += v_deg.values / len(saccades_to_plot)
        ax.plot(x_axis, v_deg, color="green", alpha=0.1)
        
    ax.plot(x_axis, mean_trace, color="lime", linewidth=2, marker="o", label="Mean speed")
    ax.axhline(get_eye_speed_outlier_threshold(exp_data), color="lime", linestyle="dashed", linewidth=2, label="Threshold") # plot threshold
    ax.axvline(0, color="black", linestyle="dashed")
    plt.legend(loc=1, fontsize=(22 if enlarged else 12))
    fs = 24 if enlarged else 12
    ax.set_title("Saccade speeds", fontsize=fs)
    ax.set_xlabel("Frames relative to saccade onset", fontsize=fs)
    ax.set_ylabel("Eye speed (deg/s)", fontsize=fs)
#     ax.set_xlim(-window_radius, window_radius)
    
    if enlarged:
        ax.set_xlim(-2, 8)
        ax.set_ylim(0, 200)
        plt.setp((ax.get_xticklabels(), ax.get_yticklabels()), fontsize=20)
    else:
        ax.set_xlim(-5, 15)
        ax.set_ylim(0, 250)
    
    return {
        "mean_saccade_speed_trace": mean_trace.tolist()
    }

    
    
def plot_saccade_duration_histogram(exp_data, saccades=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.gca()
        
    if saccades is None: saccades = exp_data.get_saccades()
    saccade_durations = [se-ss for (ss, se, conf) in saccades]
    mean_saccade_dur = float(np.mean(saccade_durations))
    bins = list(range(1, 11))
    n, hist_bins, patches = ax.hist(saccade_durations, bins=bins)
    ax.set_title("Saccade duration histogram")
    ax.set_xlabel("Saccade duration (frames)")
    ax.set_ylabel("Number of matching saccades")
    ax.axvline(mean_saccade_dur, color="black", linestyle="dashed")
    ax.annotate(f"Mean duration:\n{mean_saccade_dur:.3f} frames", ha="center", xy=(mean_saccade_dur, ax.get_ylim()[1]*0.5),
                xytext=(mean_saccade_dur+3, ax.get_ylim()[1]*0.8),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1))
    
#     print("bins len", len(bins))
#     print("hist_bins len", len(hist_bins))
#     print("n len", len(n))
    return {
        "mean_saccade_duration": mean_saccade_dur,
        "saccade_duration_hist_counts": {f"{bins[i]}-{bins[i+1]}": int(n[i]) for i in range(len(bins)-1)}
    }

    
    
def plot_psth(exp_data, window_radius_frames, saccades=None, only_plot_standalone_saccades=False, ax_mean=None, ax_top=None, kernel="none"):
    timestamps, flu = exp_data.get_ophys_exp_data().get_corrected_fluorescence_traces()
    flu_demean_frames = int(0.5 * exp_data.get_frame_rate()) # Compute F_0 based on X frames from saccade onset (0.5 is seconds)
    
    if saccades is None: saccades = exp_data.get_saccades()
    if only_plot_standalone_saccades:
        saccades_to_plot = []
        for s in saccades:
            valid = True
            for (ss, se, conf) in saccades:
                if (se < s[0] and s[0] - se <= window_radius_frames) or (ss > s[1] and ss - s[1] <= window_radius_frames):
                    valid = False
                    break
            if valid: saccades_to_plot.append(s)
    else:
        saccades_to_plot = saccades
    
    n_sacc = len(saccades_to_plot)
    
    fig = None
    if ax_mean is None and ax_top is None:
        fig = plt.figure(figsize=(12, 7))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_top = fig.add_subplot(1, 2, 2)
        
        # Main title on plot
        sid = exp_data.get_session_data()['session_id']
        area = exp_data.get_session_data()['targeted_structure']
        cre = exp_data.get_session_data()['cre_line']
        session_type = exp_data.get_session_data()['session_type']
        meta = exp_data.get_ophys_exp_data().get_metadata()
        subtitle = f"({session_type} recorded on {meta['session_start_time'].strftime('%Y-%m-%d')}, {meta['sex'][0].upper()}, {meta['age_days']}D) [{sid}]"
        fig.suptitle(f"{area} {cre} activity from {len(flu)} neurons around {n_sacc} saccades\n{subtitle}", fontsize=14)
    
    x_axis = np.arange(-window_radius_frames, window_radius_frames+1) / exp_data.get_frame_rate()
    
    if kernel == "gaussian":
        sigma = window_radius_frames / 3.0
        kernel_vec = np.exp(-x_axis**2 / (2*(sigma**2))) / (np.sqrt(2*np.pi) * sigma)
    elif kernel == "step_right":
        kernel_vec = np.ones_like(x_axis)
        kernel_vec[:window_radius_frames] = 0
    else:
        kernel_vec = np.ones_like(x_axis)

    # Mean dF/F around saccades
    mean_cell_responses = [] # array of tuples (cell_index, total_response)
    for cell_index in range(len(flu)):
        mean_trace = np.zeros_like(x_axis)
        for (ss, se, conf) in saccades_to_plot:
            mean_trace += exp_data.get_dff_corrected(cell_index, flu_demean_frames, ss, frames=(ss-window_radius_frames, ss+window_radius_frames+1))
        if n_sacc > 0: mean_trace /= n_sacc

        mean_cell_responses.append((cell_index, mean_trace.dot(kernel_vec)))
        if ax_mean is not None:
            ax_mean.plot(x_axis, mean_trace, alpha=0.25)

    if ax_mean is not None:
        ax_mean.set_title(f"Mean dF/F values for all {len(flu)} cells", fontsize=12)
        ax_mean.set_ylabel("Mean dF/F", fontsize=12)
        ax_mean.axvline(0, color="black", linestyle="dashed", linewidth=0.2)
        ax_mean.set_xlabel("Seconds relative to saccade onset", fontsize=12)
        ax_mean.set_xlim(x_axis.min(), x_axis.max())

    # Identify the cells with the largest responses
    mean_cell_responses.sort(key=lambda x: x[1], reverse=True)
    cell_idx_to_plot = [x[0] for x in mean_cell_responses[:6]]

    # Plot 2
    if ax_top is not None:
        for i, cell_index in enumerate(cell_idx_to_plot):
            traces = np.zeros((n_sacc, window_radius_frames*2+1))
            for j, (ss, se, conf) in enumerate(saccades_to_plot):
                traces[j, :] = exp_data.get_dff_corrected(cell_index, flu_demean_frames, ss, frames=(ss-window_radius_frames, ss+window_radius_frames+1))

            # Plot the mean trace
            y_offset = (len(cell_idx_to_plot)-1-i)*5
            mean_trace = np.zeros_like(x_axis) if n_sacc == 0 else traces.mean(axis=0)
            lines = ax_top.plot(x_axis, y_offset + mean_trace, linestyle="dashed", linewidth=2, label=f"{cell_index}", zorder=10)
#             color = lines[-1].get_color()
        #     plt.setp(lines, color="black")

            # Plot the individual traces
            if n_sacc > 0: ax_top.plot(x_axis, y_offset + traces.T, color="gray", alpha=0.2)

        ax_top.axvline(0, color="black", linestyle="dashed", linewidth=0.2)
        ax_top.legend(fontsize=10, loc=1)
        ax_top.set_title(f"dF/F values for {len(cell_idx_to_plot)} cells with largest mean response", fontsize=12)
        ax_top.set_ylabel("dF/F", fontsize=12)
        ax_top.set_xlabel("Seconds relative to saccade onset", fontsize=12)
        ax_top.set_xlim(x_axis.min(), x_axis.max())
        ax_top.set_ylim(-2, len(cell_idx_to_plot)*5)
    
    return {}, fig
    

def plot_saccade_conf_hist(exp_data, saccades=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    if saccades is None: saccades = exp_data.get_saccades()
    confidence_scores = [s[2] for s in saccades]
    n_high = sum([1 for score in confidence_scores if score > 0.5])
    n_low = sum([1 for score in confidence_scores if score <= 0.5])
    ax.hist(confidence_scores, bins=100)
    ax.set_title(f"Confidence scores of {len(confidence_scores)} saccades")
    ax.set_xlabel("Confidence score", fontsize=12)
    ax.set_ylabel("Number of saccades", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="black", linestyle="dashed")
    ax.text(0.55, 0.5, f"{n_high} $>$ 50% score", ha="left", va="center", fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.5, f"{n_low} $\leq$ 50% score", ha="right", va="center", fontsize=12, transform=ax.transAxes)
    
    return {
        "n_high_conf_saccades": n_high,
        "n_low_conf_saccades": n_low
    }
    
    
def create_summary_plot(exp_data, start, end, sacc_conf_threshold=0.5, sacc_dist_threshold=1.5, window_radius_frames=30):
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(7, 7, figure=fig)
    sdata = exp_data.get_session_data()
    metadata = exp_data.get_ophys_exp_data().get_metadata()
    sex = metadata["sex"]
    data = {
        key: int(exp_data.get_session_data()[key])
        for key in [
            "session_id", "container_id"
        ]
    }
    data.update({
        key: exp_data.get_session_data()[key]
        for key in [
            "targeted_structure", "cre_line", "session_type", "published_at"
        ]
    })
    data["specimen_name"] = metadata["specimen_name"]
    
    # Add stim epochs
    stim_epoch = exp_data.get_ophys_exp_data().get_stimulus_epoch_table()
    data["stim_epoch_table"] = []
    for i, row in stim_epoch.iterrows():
        data["stim_epoch_table"].append({
            key: row[key] for key in ["stimulus", "start", "end"]
        })
    
    # Average running speed
    running_speed = exp_data.get_running_speed()
    data["running_speed_mean"] = float(np.nanmean(running_speed))
    data["running_speed_std"] = float(np.nanstd(running_speed))
    data["running_speed"] = running_speed.tolist()
    
    # Pupil trace
    data["pupil_area"] = exp_data.get_eye_tracking_data("pupil_area", frames=(start, end)).values.tolist()
    
    # Running speed histogram
    data["running_speed_hist"] = []
    window_width = exp_data.get_frame_rate() * 60
    for run_start in range(start, end, window_width):
        run_end = run_start + window_width
        trace = exp_data.get_running_speed()[run_start:run_end]
        data["running_speed_hist"].append({
            "start_frame": run_start,
            "end_frame": run_end,
            "running_speed_mean": float(np.nanmean(trace)),
            "running_speed_std": float(np.nanstd(trace))
        })
    
    saccades = [s for s in exp_data.get_saccades() if s[2] > sacc_conf_threshold]
    data["saccades"] = [
        {
            "start": ss,
            "end": se,
            "conf": conf,
            "dist": get_degrees_moved(exp_data, ss, se)
        } for (ss, se, conf) in exp_data.get_saccades()
    ]
    data["saccades_metric"] = "confidence"
    data["age"] = metadata['age_days']
    data["imaging_depth"] = metadata['imaging_depth_um']
    data["recording_date"] = metadata['session_start_time'].strftime('%Y-%m-%d')
    
    ax_desc = fig.add_subplot(gs[0, :2])
    ax_desc.set_axis_off()
    sexcolor = {"female": "pink", "male": "blue"}.get(sex, "gray")
    ax_desc.text(0.05, 0.85, f"Session {exp_data.get_session_id()} (sex {sex[0].upper()})", bbox={"facecolor": sexcolor, "alpha": 0.5, "pad": 8}, fontsize=16)
    height = 0.65
    ophys = exp_data.get_ophys_exp_data()
    mean_saccade_dur = np.mean([se-ss for (ss, se, conf) in saccades])
    for text in [
        f"{sdata['targeted_structure']}: {sdata['cre_line']} ({len(exp_data.get_ophys_exp_data().get_roi_mask_array())} neurons)",
        f"{sdata['session_type']} recorded on {metadata['session_start_time'].strftime('%Y-%m-%d')}",
    #     f"Mean saccade duration: {mean_saccade_dur:.3f} frames"
        fr"Age: {metadata['age_days']} days; Depth = {metadata['imaging_depth_um']} $\mu$m",
        f"Saccades: {len(saccades)} with >{sacc_conf_threshold*100:.0f}% confidence ({len(exp_data.get_saccades())} before)",
        f"Experiment container: {sdata['container_id']}"
    ]:
        ax_desc.text(0.05, height, text, fontsize=12, ha="left", va="center", transform=ax_desc.transAxes)
        height -= 0.15
        
    ax_rois = fig.add_subplot(gs[0, 2:4], aspect="auto")
    ax_rois.imshow(exp_data.get_ophys_exp_data().get_roi_mask_array().sum(axis=0), cmap="gray")
    ax_rois.set_title("Neuron ROIs")
    ax_rois.set_axis_off()

    ax_sacc_conf_hist = fig.add_subplot(gs[0, -2:])
    data.update(plot_saccade_conf_hist(exp_data, ax=ax_sacc_conf_hist))
    
    ax_general_data = fig.add_subplot(gs[1, :-2])
    ax_eye_pos = fig.add_subplot(gs[2, :-2], sharex=ax_general_data)
    data.update(plot_eye_data(exp_data, start, end, saccades=saccades, ax=ax_general_data, ax_eye_pos=ax_eye_pos))

    ax_saccades = fig.add_subplot(gs[3, :-2])
    ax_hist = fig.add_subplot(gs[4, :-2])
    ax_bar_count = fig.add_subplot(gs[5, 4])
    ax_bar_rate = fig.add_subplot(gs[6, 4])
    data.update(plot_experiment_saccades(exp_data, saccades=saccades, start=start, end=end, ax=ax_saccades, ax_hist=ax_hist, ax_bar_count=ax_bar_count, ax_bar_rate=ax_bar_rate))

    ax_motions = fig.add_subplot(gs[5, :2])
    data.update(plot_saccade_motions(exp_data, saccades=saccades, ax=ax_motions))

    ax_overlays = fig.add_subplot(gs[5, 2:4])
    # plot_saccade_overlays(exp_data, saccades, ax=ax_overlays, num_to_plot=1, velocity_overlay=True)
    data.update(plot_saccade_overlays(exp_data, saccades=saccades, ax=ax_overlays))

    ax_duration_hist = fig.add_subplot(gs[6, :2])
    data.update(plot_saccade_duration_histogram(exp_data, saccades=saccades, ax=ax_duration_hist))

    ax_speeds = fig.add_subplot(gs[6, 2:4])
    data.update(plot_saccade_speeds(exp_data, saccades=saccades, ax=ax_speeds))
    # plot_saccade_overlays(exp_data, saccades, ax=ax_speeds, num_to_plot=1, velocity_overlay=True)
    # ax_speeds.set_xlabel("Sample saccade profile")

    # Neural activity
    ax_dff = fig.add_subplot(gs[1:3, -2:])
    ax_psth_mean = fig.add_subplot(gs[3:5, -2:])
    ax_psth_top = fig.add_subplot(gs[5:7, -2:])
    data.update(plot_dff(exp_data, ax=ax_dff, saccades=saccades))
    d, _ = plot_psth(exp_data, window_radius_frames=window_radius_frames, saccades=saccades, only_plot_standalone_saccades=False, ax_mean=ax_psth_mean, ax_top=ax_psth_top)
    data.update(d)
    
    return fig, data
    