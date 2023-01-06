import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pickle
import glob
import os
from os import path
import re
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from saccade_config import *

def load_data_by_session_id(session_id):
    """
    Load a cached experiment data by session ID.
    """
    filename = path.join(LOADED_CELL_DATA_DIR, f"{session_id}.pickle")
    data = None

    if path.exists(filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
    return data



def load_data(tqdm_desc=None):
    """
    Load all cached experiment data; to be used in a loop.
    """
    files_to_process = glob.glob(path.join(LOADED_CELL_DATA_DIR, f"*.pickle"))
    iter = files_to_process if tqdm_desc is None else tqdm(files_to_process, desc=tqdm_desc)

    for filename in iter:
        with open(filename, "rb") as file:
            data = pickle.load(file)
            yield data
            


def load_additional_data(filename, in_data_dir=True, **kwargs):
    """
    Load additional data.
    """
    if in_data_dir: filename = path.join("..", "data", filename)
    return pd.read_csv(filename, **kwargs)



def savefig(fig, filename, base_dir=FIGURE_BASE_DIR, sub_dir=None, dpi=250):
    """
    Shorthand to save an analysis figure.
    sub_dir can be used to set the sub directory to save to a sub-directory.
    """
    if sub_dir is not None:
        base_dir = path.join(base_dir, sub_dir)

    os.makedirs(base_dir, exist_ok=True)
    fig.savefig(path.join(base_dir, filename), dpi=dpi, transparent=False, bbox_inches="tight")


def set_major_minor_ticks(ax, window_radius, minor_tick_step=100, major_tick_step=500, center=0):
    """Set major and minor axis tick labels on a response plot.

    Args:
        ax (matplotlib axis): Axis
        window_radius (int): Radius of traces
        minor_tick_step (int, optional): Minor tick step. Defaults to 100.
        major_tick_step (int, optional): Major tick step; also includes label text. Defaults to 500.
        center (int, optional): _description_. Defaults to 0.
    """
    window_radius_max = window_radius/30*1000
    minor_ticks = [0]
    while minor_ticks[-1] + minor_tick_step <= window_radius_max:
        tick = minor_ticks[-1] + minor_tick_step
        minor_ticks.insert(0, -tick)
        minor_ticks.append(tick)
    ax.set_xticks([ms/1000*30+center for ms in minor_ticks], minor=True)

    # MAJOR ticks every 500 ms
    major_ticks = [tick for tick in minor_ticks if tick % 500 == 0]
    ax.set_xticks([ms/1000*30+center for ms in major_ticks])
    ax.set_xticklabels(major_ticks, fontsize=14)

    ax.tick_params(axis="x", which="major", direction="out", length=6, width=1, color="black")
    ax.tick_params(axis="x", which="minor", direction="out", length=2, width=1, color="black")


def heatmap_log_proba_plot(p_matrix, heatmap_labels=None, ticklabels=None, title="Probabilities", titlefontsize=18, xticklabelrotation=90, cbar_label="", ax=None, ticklabelfontsize=16, figsize=(10, 8), significance_thresh=0.05, log=True, correct_comparisons=True, cbartickfontsize=16, is_inset=False):
    """
    Shorthand to create a probability heatmap matrix.
    heatmap_labels can be used to annotate cells in the matrix.
    tick_labels can be used to set axis tick labels.
    """
    n_comparisons = np.isfinite(p_matrix[0]).sum()
    centervalue = significance_thresh / n_comparisons if correct_comparisons else significance_thresh
    if log:
        centervalue = np.log10(centervalue)
    cbar_ticks = [centervalue-2, centervalue-1, centervalue, centervalue+1, centervalue+2]
    # cbar_ticklabels = [f"{x:.0e}" for x in np.power(10, cbar_ticks)]
    cbar_ticklabels = []

    for x in cbar_ticks:
        if x == centervalue:
            x = np.power(10, x)
            label = f"{x:.2f}" if x == 0.05 else f"{x:.3f}"
        else:
            x = np.power(10, x)
            x = re.split("e(\+|\-)", f"{x:.0e}")
            a = int(x[0])
            b = (1 if x[1] == "+" else -1) * int(x[2])
            if b == 0:
                label = f"{a}"
            else:
                label = f"{a} $\\times 10^{'{'}{b}{'}'}$"
        cbar_ticklabels.append(label)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig, ax = ax.get_figure(), ax

    logmatrix = np.log10(p_matrix)
    cbar = dict(cbar=False) if is_inset else dict(cbar=True, cbar_kws=dict(ticks=cbar_ticks))
    sns.heatmap(
        logmatrix, annot=heatmap_labels, fmt="", annot_kws=dict(fontsize=14),
        ax=ax, linewidths=0.5, square=True, cmap="seismic_r",
        center=centervalue, vmin=centervalue-3, vmax=centervalue+3, **cbar
    )

    if p_matrix.shape[1] == len(ticklabels):
        ax.set_xticklabels(ticklabels, fontsize=ticklabelfontsize, rotation=xticklabelrotation)
    else:
        ax.set_xticklabels([])

    if p_matrix.shape[0] == len(ticklabels):
        ax.set_yticklabels(ticklabels, fontsize=ticklabelfontsize, rotation=0)
    else:
        ax.set_yticklabels([])

    if is_inset:
        cbar_ax = inset_axes(ax, width="5%", height="100%", loc="lower left", bbox_to_anchor=(1.05, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(ax.collections[0], cax=cbar_ax)
    else:
        cbar = ax.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=cbartickfontsize)
    cbar.set_label(cbar_label, fontsize=16) # , rotation=270, va="bottom"
    ax.set_title(title, fontsize=titlefontsize)
    return fig, ax



"""
Test to save a blank figure.
"""
if __name__ == "__main__":
    fig = plt.figure()
    savefig(fig, "test.png")