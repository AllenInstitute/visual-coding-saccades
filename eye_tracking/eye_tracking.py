import numpy as np
import pandas as pd
import math


def get_eye_speed_outlier_threshold(eye_speed):
    """
    Get the eye speed outlier threshold, defined as the maximum of 10 and three
    standard deviations above the mean eye speed in the entire experiment. All times
    where the eye exceeds this threshold will be flagged as the potential start of
    a saccade.
    """
    return max(10, eye_speed.mean() + 3*eye_speed.std())


def get_eye_speed_outlier_threshold_end(eye_speed):
    """
    Get the ending eye speed outlier threshold, defined as one standard deviation
    above the mean eye speed in an entire experiment. This is used to define the
    ending frame of saccades.
    """
    return eye_speed.mean() + 1*eye_speed.std()

def get_saccade_magnitude_direction(eye_data, start, end):
    """
    Find the absolute degrees moved between two (inclusive) frames and the direction
    of movement.
    """
    xs, xe = eye_data["x_pos_deg"].loc[start], eye_data["x_pos_deg"].loc[end]
    ys, ye = eye_data["y_pos_deg"].loc[start], eye_data["y_pos_deg"].loc[end]
    dx = xe - xs
    dy = ye - ys
    magnitude = np.sqrt(dx**2 + dy**2)
    theta = math.atan2(dy, dx) # [-pi, pi]
    pi_4 = math.pi / 4
    if -pi_4 <= theta < pi_4:
        angle = "R"
    elif pi_4 <= theta < 3*pi_4:
        angle = "U"
    elif 3*pi_4 <= theta or theta < -3*pi_4:
        angle = "L"
    else: # -3*pi_4 <= theta < -pi_4:
        angle = "D"
    
    return magnitude, angle


def get_degrees_moved(eye_data, start, end):
    """
    Get the absolute degrees moved between two (inclusive) frames.
    Preferable to instead use get_saccade_magnitude_direction.
    """
    return get_saccade_magnitude_direction(eye_data, start, end)[0]

def get_direction_moved(eye_data, start, end):
    """
    Get the direction the eye moved between two (inclusive) frames.
    Preferable to instead use get_saccade_magnitude_direction.
    """
    return get_saccade_magnitude_direction(eye_data, start, end)[1]

def get_stim_at_frame(stim_epoch, frame):
    """
    Find which stimulus was being shown at a particular frame.
    """
    rows = stim_epoch[(stim_epoch["start"] <= frame) & (frame <= stim_epoch["end"])]

    if len(rows) > 0:
        return rows.iloc[0]["stimulus"]
    else:
        return "spontaneous"





def get_valid_saccades(eye_data, saccades, n_frames=8, std_thresh=3, debug=False):
    valid_saccades = []
    dims = ("x", "y")

    for i, s in enumerate(saccades):
        ss, se = s[0], s[1]
        has_valid_dim = False
    
        for dim in dims:
            before = eye_data[f"{dim}_pos_deg"].loc[ss-n_frames:ss-1]
            after = eye_data[f"{dim}_pos_deg"].loc[se+1:se+n_frames]
            before_mean, before_std = np.mean(before), np.std(before)
            after_mean, after_std = np.mean(after), np.std(after)
            mean_diff = abs(after_mean - before_mean)
            thresh = std_thresh*max(before_std, after_std)
            valid = mean_diff > thresh

            if valid:
                has_valid_dim = True
                if debug:
                    print(f"saccade {i} valid in {dim} -- {mean_diff:.4f} > {thresh:.4f}")
                break

        if has_valid_dim:
            valid_saccades.append((ss, se))


    return valid_saccades


# TODO: Change this to include frame_rate
def __get_valid_saccades_from_gaze(gaze_data, saccades, x_deg_col="x_deg", y_deg_col="y_deg", padding_sec=0.3, std_thresh=3, frame_rate=None, debug=False):
    valid_saccades = []
    cols = (x_deg_col, y_deg_col)
    n_frames = None if frame_rate is None else int(padding_sec * frame_rate)

    for i, (ss, se) in enumerate(saccades):
        has_valid_dim = False
    
        for col in cols:
            if frame_rate is None:
                before = gaze_data[col].loc[((ss - padding_sec) <= gaze_data.index) & (gaze_data.index < ss)]
                after = gaze_data[col].loc[(se < gaze_data.index) & (gaze_data.index < (se + padding_sec))]
            else:
                before = gaze_data[col].loc[ss-n_frames:ss-1]
                after = gaze_data[col].loc[se+1:se+n_frames]


            before_mean, before_std = np.mean(before), np.std(before)
            after_mean, after_std = np.mean(after), np.std(after)
            mean_diff = abs(after_mean - before_mean)
            thresh = std_thresh*max(before_std, after_std)
            valid = mean_diff > thresh

            if valid:
                has_valid_dim = True
                if debug:
                    print(f"saccade {i} valid {col} -- {mean_diff:.4f} > {thresh:.4f}")
                break

        if has_valid_dim:
            valid_saccades.append((ss, se))


    return valid_saccades


def arclength(azi1, alt1, azi2, alt2, degrees=True, method="gs"):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    from numpy import sin, cos
    if degrees:
        azi1, alt1, azi2, alt2 = np.radians((azi1, alt1, azi2, alt2))

    delta_alt = np.abs(alt2 - alt1)

    if method == "gs": # Great-circle distance
        angle = np.arccos(sin(azi1)*sin(azi2) + cos(azi1)*cos(azi2)*cos(delta_alt))
    elif method == "hs":
        pass
    
    return np.degrees(angle) if degrees else angle


def get_eye_speed(gaze_data, x_deg_col="x_deg", y_deg_col="y_deg", frame_rate=None):
    # Calculate rate of change of position using central difference approximation of f'(x)
    speed = gaze_data.apply(lambda row: np.nan, axis=1)

    # [f(x+h) - f(x-h)] / (2h)
    #   = [f(x+h) - f(x-h)] / (2/frame_rate)
    #   = [f(x+h) - f(x-h)] * frame_rate/2

    indices = gaze_data.index.values

    for i in range(len(indices)):
        if 0 < i < len(indices) - 1:
            # There is another data point before and after this index
            i1 = indices[i+1]
            i0 = indices[i-1]
            x0, x1 = gaze_data.at[i0, x_deg_col], gaze_data.at[i1, x_deg_col]
            y0, y1 = gaze_data.at[i0, y_deg_col], gaze_data.at[i1, y_deg_col]
            dt = 1/frame_rate if frame_rate is not None else (i1 - i0)
            # x_speed = dx / (2*dt)
            # y_speed = dy / (2*dt)
            # speed_x.iat[i] = x_speed
            # speed_y.iat[i] = y_speed
            # speed.iat[i] = np.sqrt(x_speed**2 + y_speed**2)
            speed.iat[i] = arclength(x0, y0, x1, y1, degrees=True) / (2*dt)

    return speed


def get_saccades_from_gaze_traces(gaze_data, start, end, x_deg_col="x_deg", y_deg_col="y_deg", speed_col=None, nan_padding_sec=0.5, frame_rate=None, noise_padding_sec=0.3, noise_std_thresh=3, include_peak_frame=False):
    if speed_col is None:
        eye_speed = get_eye_speed(gaze_data, x_deg_col=x_deg_col, y_deg_col=y_deg_col, frame_rate=frame_rate)
    else:
        eye_speed = gaze_data[speed_col]

    speed_threshold = get_eye_speed_outlier_threshold(eye_speed.loc[start:end-1])
    offset_threshold = get_eye_speed_outlier_threshold_end(eye_speed.loc[start:end-1])
    # print("SPEED THRESHOLD:", speed_threshold, offset_threshold)
    i = 0
    last_nan_index = 0
    indices = gaze_data.index
    saccades = []
    cols = (x_deg_col, y_deg_col)
    noise_n_frames = None if frame_rate is None else int(noise_padding_sec * frame_rate)
    
    # debug_mean_backtrack = []

    while i < len(indices):
        speed_i = eye_speed.iloc[i]
        is_valid = True

        if np.isnan(eye_speed.iloc[i]):
            last_nan_index = i

        # Check if there is a nearby NaN frame
        if frame_rate is None:
            if i - last_nan_index < nan_padding_sec:
                is_valid = False
        else:
            if i - last_nan_index < nan_padding_sec * frame_rate:
                is_valid = False

        # if frame_rate is None:
        #     nearby_speed_traces = eye_speed[np.abs(eye_speed.index - i) < nan_padding_sec]
        # else:
        #     nearby_speed_traces = eye_speed[np.abs(eye_speed.index - i) < nan_padding_sec * frame_rate]

        # if np.any(np.isnan(nearby_speed_traces)):
        #     # Ignore if there is a nearby NaN frame
        #     is_valid = False

        # If the velocity is an outlier
        if is_valid and speed_i >= speed_threshold:
            # Work backward to find saccade onset time
            onset = i-1
                            
            while is_valid:
                if onset < 0:
                    is_valid = False
                elif np.isnan(eye_speed.iloc[onset]):
                    is_valid = False
#                 elif v.loc[onset-1] > v.loc[onset] and v.loc[onset] < v.loc[onset+1]: # A local min
                elif eye_speed.iloc[onset] < offset_threshold:
                    # We have reached the onset
                    break
                else:
                    onset -= 1
                    
            # Find saccade offset (next local min)
            offset = i+1
            
            while is_valid:
                if offset >= len(indices):
                    is_valid = False
                elif np.isnan(eye_speed.iloc[offset]):
                    is_valid = False
                    last_nan_index = offset
#                 elif v.loc[offset-1] > v.loc[offset] and v.loc[offset] < v.loc[offset+1]: # Local min
                elif eye_speed.iloc[offset] < offset_threshold:
                    # We have reached the offset
                    break
                else:
                    offset += 1
            
            # Check if the eye area abruptly changes during saccade
#             if is_valid and offset - onset <= 3:
#                 eye_area = exp_data.get_eye_tracking_data("eye_area", frames=(onset-10, offset+10))
#                 mean, std = eye_area.mean(), eye_area.std()
# #                     plt.axhline(60+100*(mean), c="k")
# #                     plt.axhline(60+100*(mean-3*std), c="k")
                
#                 # Check if any point in the saccade is an eye area outlier
#                 for f in range(onset, offset+1):
#                     if abs(eye_area.loc[f] - mean) >= 3*std:
#                         is_valid = False
#                         break
            
            # Make sure there is a nonnegligible change in absolute degree
            # if is_valid:
            #     x0 = exp_data.get_eye_tracking_data("x_pos_deg", frames=onset)
            #     y0 = exp_data.get_eye_tracking_data("y_pos_deg", frames=onset)
            #     xn = exp_data.get_eye_tracking_data("x_pos_deg", frames=offset)
            #     yn = exp_data.get_eye_tracking_data("y_pos_deg", frames=offset)
            #     degree_change = np.sqrt((xn-x0)**2 + (yn-y0)**2)
            
            ss, se = indices[onset], indices[offset]

            if is_valid:
                # Ignore saccades out of bounds
                if ss < start or se >= end:
                    is_valid = False

            # Final validity check to ignore "noisy" saccades
            if is_valid:
                has_valid_dim = False
            
                for col in cols:
                    if frame_rate is None:
                        before = gaze_data[col].loc[((ss - noise_padding_sec) <= gaze_data.index) & (gaze_data.index < ss)]
                        after = gaze_data[col].loc[(se < gaze_data.index) & (gaze_data.index < (se + noise_padding_sec))]
                    else:
                        before = gaze_data[col].loc[ss-noise_n_frames:ss-1]
                        after = gaze_data[col].loc[se+1:se+noise_n_frames]

                    before_mean, before_std = np.mean(before), np.std(before)
                    after_mean, after_std = np.mean(after), np.std(after)
                    mean_diff = abs(after_mean - before_mean)
                    thresh = noise_std_thresh*max(before_std, after_std)
                    valid = mean_diff > thresh

                    if valid:
                        has_valid_dim = True
                        # if debug: print(f"saccade {i} valid {col} -- {mean_diff:.4f} > {thresh:.4f}")
                        break
                
                is_valid = has_valid_dim

            if is_valid:
                # Finally, add it as a saccade
                if include_peak_frame:
                    saccades.append((ss, se, indices[i]))
                else:
                    saccades.append((ss, se))
                # debug_mean_backtrack.append(i-ss)
                        
            i = offset + 1
        else:
            i += 1
    
    # print("MEAN BACKTRACK", np.mean(debug_mean_backtrack))

    return saccades
