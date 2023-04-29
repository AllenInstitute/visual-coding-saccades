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


LABEL_NONE = "No response"
LABEL_ENHANCED = "Enhanced: No DS"
LABEL_SUPPRESSED = "Suppressed"
LABEL_NASAL = "Enhanced: Nasal"
LABEL_TEMPORAL = "Enhanced: Temporal"

SR_LABELS = [
    (LABEL_NONE, "gray"),
    (LABEL_SUPPRESSED, "purple"),
    (LABEL_ENHANCED, "green"),
    (LABEL_NASAL, "b"),
    (LABEL_TEMPORAL, "r"),
]

SR_LABELS_ABBREV = {
    "No response": "Not SR",
    "Enhanced: No DS": "No DS",
    "Enhanced: Nasal": "N",
    "Enhanced: Temporal": "T",
}

def add_cell_df_columns(cells):
    from statsmodels.sandbox.stats.multicomp import multipletests

    cells["depth_trunc"] = 100 * (cells["depth"]/100).astype(int)
    cells.at[cells["cre_line"] == "Scnn1a-Tg3-Cre; Camk2a-tTA; Ai93(TITL-GCaMP6f)", "depth_trunc"] = 200
    cells.at[cells["cre_line"] == "Nr5a1-Cre; Camk2a-tTA; Ai93(TITL-GCaMP6f)", "depth_trunc"] = 200
    cells.at[cells["cre_line"] == "Fezf2-CreER; Ai148(TIT2L-GC6f-ICL-tTA2)", "depth_trunc"] = 300
    cells.at[cells["depth_trunc"] == 400, "depth_trunc"] = 300

    depth_to_layer = {
        100: "2/3",
        200: "4",
        300: "5",
        500: "6",
    }

    cells["cortical_layer"] = cells["depth_trunc"].apply(lambda depth_trunc: depth_to_layer[depth_trunc])



    miura_thresh = 5e-2 # 0.05
    dir_thresh = 0.05

    response_classification = cells["response_classification"]
    cells["is_sr"] = (response_classification.abs() == 1)
    is_ds = (cells["ranksum_p_by_direction"] < dir_thresh)
    is_sr_ds = (cells["is_sr"]) & is_ds
    cells["is_sr_ds"] = is_sr_ds
    method = "fdr_bh"
    alpha = 0.1
    cells["is_sr_miura"] = (
        (multipletests(cells["wilcoxon_signed_rank_p"], alpha=alpha, method=method)[1] < 0.05)
        | (multipletests(cells["ranksum_p_by_direction"], alpha=alpha, method=method)[1] < 0.05)
    )
    cells["is_sr_ds_miura"] = cells["is_sr_miura"] & is_ds


    # Compute preferred direction
    larger_R_response = cells["mean_right_response"] > cells["mean_left_response"]
    cells["preferred_direction"] = 0

    # pref_dir_temporal = (
    #     ((cells["response_classification"] == 1) & larger_R_response)
    #      | ((cells["response_classification"] == -1) & ~larger_R_response)
    # )

    # pref_dir_nasal = (
    #     ((cells["response_classification"] == 1) & ~larger_R_response)
    #      | ((cells["response_classification"] == -1) & larger_R_response)
    # )

    # ONLY CONSIDER ENHANCED RESPONSES
    pref_dir_temporal = (cells["response_classification"] == 1) & larger_R_response
    pref_dir_nasal = (cells["response_classification"] == 1) & ~larger_R_response

    cells.loc[is_sr_ds & pref_dir_temporal, "preferred_direction"] = 1
    cells.loc[is_sr_ds & pref_dir_nasal, "preferred_direction"] = -1

    cells.loc[cells["is_sr_miura"] & is_ds & pref_dir_temporal, "preferred_direction_miura"] = 1
    cells.loc[cells["is_sr_miura"] & is_ds & pref_dir_nasal, "preferred_direction_miura"] = -1

    cells["sr_label"] = LABEL_NONE
    cells.loc[(cells["response_classification"] == 1) & ~(cells["is_sr_ds"]), "sr_label"] = LABEL_ENHANCED
    cells.loc[(cells["response_classification"] == -1), "sr_label"] = LABEL_SUPPRESSED
    cells.loc[(cells["is_sr"]) & (cells["preferred_direction"] == -1), "sr_label"] = LABEL_NASAL
    cells.loc[(cells["is_sr"]) & (cells["preferred_direction"] == 1), "sr_label"] = LABEL_TEMPORAL

    cells["sr_label_miura"] = LABEL_NONE
    cells.loc[(cells["is_sr_miura"]) & ~(cells["is_sr_ds_miura"]), "sr_label_miura"] = "SR"
    cells.loc[(cells["is_sr_miura"]) & (cells["preferred_direction_miura"] == -1), "sr_label_miura"] = LABEL_NASAL
    cells.loc[(cells["is_sr_miura"]) & (cells["preferred_direction_miura"] == 1), "sr_label_miura"] = LABEL_TEMPORAL

    # print(cells["sr_label"].value_counts())
    # print()
    # print(cells["sr_label_miura"].value_counts())



def load_metrics(cells=None):
    metrics = load_additional_data("metrics.csv")
    metrics.drop([c for c in metrics.columns if "Unnamed" in c], axis=1, inplace=True)

    dsi_df = load_additional_data("dsi.csv")
    metrics = metrics.merge(dsi_df[["cell_specimen_id", "DSI_pref_tf"]], on="cell_specimen_id")
    metrics.set_index("cell_specimen_id", inplace=True)
    # metrics.head()

    if cells is None:
        return metrics
    else:
        cells_and_metrics = cells.join(metrics, how="inner")
        print(f"{len(cells_and_metrics):,}/{len(cells):,} cells have associated metrics")
        return metrics, cells_and_metrics


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


def heatmap_log_proba_plot(p_matrix, heatmap_labels=None, ticklabels=None, yticklabels=None, xticklabels=None, title="Probabilities", titlefontsize=18, xticklabelrotation=90, show_cbar=True, cbar_label="", cbar_label_fontsize=16, ax=None, ticklabelfontsize=16, figsize=(10, 8), significance_thresh=0.05, log=True, correct_comparisons=True, cbartickfontsize=16, is_inset=False, cbar_ax=None):
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
                label = f"{a}$\\times 10^{'{'}{b}{'}'}$"
        cbar_ticklabels.append(label)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        fig, ax = ax.get_figure(), ax

    logmatrix = np.log10(p_matrix)
    cbar = dict(cbar=False) if is_inset or not show_cbar else dict(cbar=True, cbar_kws=dict(ticks=cbar_ticks), cbar_ax=cbar_ax)
    sns.heatmap(
        logmatrix, annot=heatmap_labels, fmt="", annot_kws=dict(fontsize=14, color="black"),
        ax=ax, linewidths=0.5, square=True, cmap="seismic_r",
        center=centervalue, vmin=centervalue-2.5, vmax=centervalue+2.5, **cbar
    )

    ax.set_title(title, fontsize=titlefontsize)

    if xticklabels is None:
        xticklabels = ticklabels
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=ticklabelfontsize, rotation=xticklabelrotation)

    if yticklabels is None:
        yticklabels = ticklabels
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=ticklabelfontsize, rotation=0)

    if show_cbar:
        if is_inset:
            if cbar_ax is None:
                cbar_ax = inset_axes(ax, width="5%", height="90%", loc="lower left", bbox_to_anchor=(1.05, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            
            if cbar_ax is False: # if cbar_ax is False, no cbar
                cbar = None
            else:
                cbar = plt.colorbar(ax.collections[0], cax=cbar_ax)
        else:
            cbar = ax.collections[0].colorbar
        
        if cbar is not None:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)
            cbar.ax.tick_params(labelsize=cbartickfontsize)
            cbar.set_label(cbar_label, fontsize=cbar_label_fontsize) # , rotation=270, va="bottom"

    return fig, ax



"""
Test to save a blank figure.
"""
if __name__ == "__main__":
    fig = plt.figure()
    savefig(fig, "test.png")