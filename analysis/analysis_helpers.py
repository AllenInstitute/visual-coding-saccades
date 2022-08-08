import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pickle
import glob
import os
import seaborn as sns

from saccade_config import *

"""
Load a cached experiment data by session ID.
"""
def load_data_by_session_id(session_id):
    with open(os.path.join(LOADED_CELL_DATA_DIR, f"{session_id}.pickle"), "rb") as file:
        data = pickle.load(file)
    return data


"""
Load all cached experiment data; to be used in a loop.
"""
def load_data(tqdm_desc=None):
    files_to_process = glob.glob(os.path.join(LOADED_CELL_DATA_DIR, f"*.pickle"))
    iter = files_to_process if tqdm_desc is None else tqdm(files_to_process, desc=tqdm_desc)

    for filename in iter:
        with open(filename, "rb") as file:
            data = pickle.load(file)
            yield data
            

"""
Load additional data.
"""
def load_additional_data(filename, in_data_dir=True, **kwargs):
    if in_data_dir: filename = os.path.join("..", "data", filename)
    return pd.read_csv(filename, **kwargs)


"""
Shorthand to save an analysis figure.
sub_dir can be used to set the sub directory to save to a sub-directory.
"""
def savefig(fig, filename, base_dir=FIGURE_BASE_DIR, sub_dir=None, dpi=250):
    if sub_dir is not None:
        base_dir = os.path.join(base_dir, sub_dir)

    os.makedirs(base_dir, exist_ok=True)
    fig.savefig(os.path.join(base_dir, filename), dpi=dpi, transparent=False, bbox_inches="tight")


"""
Shorthand to create a probability heatmap matrix.
heatmap_labels can be used to annotate cells in the matrix.
tick_labels can be used to set axis tick labels.
"""
def heatmap_log_proba_plot(p_matrix, heatmap_labels=None, ticklabels=None, title="Probabilities", cbar_label="p", ax=None, figsize=(10, 8), significance_thresh=0.05, log=True, correct_comparisons=True):
    n_comparisons = np.isfinite(p_matrix[0]).sum()
    centervalue = significance_thresh / n_comparisons if correct_comparisons else significance_thresh
    if log:
        centervalue = np.log10(centervalue)
    cbar_ticks = [centervalue-2, centervalue-1, centervalue, centervalue+1, centervalue+2]
    cbar_ticklabels = [f"{x:.0e}" for x in np.power(10, cbar_ticks)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig, ax = ax.get_figure(), ax


    sns.heatmap(
        np.log10(p_matrix), annot=heatmap_labels, fmt="", annot_kws=dict(fontsize=14),
        ax=ax, linewidths=0.5, square=True, cmap="seismic_r",
        center=centervalue, vmin=centervalue-3, vmax=centervalue+3, cbar_kws=dict(ticks=cbar_ticks)
    )

    if p_matrix.shape[1] == len(ticklabels):
        ax.set_xticklabels(ticklabels, fontsize=14, rotation=90)
    else:
        ax.set_xticklabels([])

    if p_matrix.shape[0] == len(ticklabels):
        ax.set_yticklabels(ticklabels, fontsize=14, rotation=0)
    else:
        ax.set_yticklabels([])

    cbar = ax.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(cbar_label, fontsize=16) # , rotation=270, va="bottom"
    ax.set_title(title, fontsize=18)
    return fig, ax



"""
Test to save a blank figure.
"""
if __name__ == "__main__":
    fig = plt.figure()
    savefig(fig, "test.png")