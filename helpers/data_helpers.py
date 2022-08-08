import numpy as np

def get_dff(flu, cell_index, demean_frames, frames=None):
    if type(demean_frames) is slice:
        f0 = flu[:, demean_frames].mean(axis=1) if cell_index is None else flu[cell_index, demean_frames].mean()
    else:
        f0 = demean_frames
    frames_slice = None

    if type(frames) is slice or type(frames) is int:
        frames_slice = frames
    elif type(frames) is tuple or type(frames) is list:
        frames_slice = slice(frames[0], frames[1])

    if cell_index is None:
        trace = flu[:, :] if frames_slice is None else flu[:, frames_slice]
        # trace.shape = (n_cells, trace_len), f0.shape = (n_cells,)
        # have to transpose trace so it can be broadcasted with f0
        return (trace - f0[:, np.newaxis]) / f0[:, np.newaxis]
    else:
        trace = flu[cell_index, :] if frames_slice is None else flu[cell_index, frames_slice]
        return (trace - f0) / f0
        