import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, interpolate
from preprocessing.preprocessing_helper import decode_filename, BAD_FILES, get_bar_positions, \
    calc_offset_time, normalize_time, add_ms, calc_offset_index, calc_windowsize
import utils.paths as dirs


def calculate_bar(x, support_points=20):
    ixs = list(range(support_points)) + \
        list(range(x.shape[-1] - support_points, x.shape[-1]))
    return interpolate.interp1d(ixs, x[ixs])


def get_indices_over_threshold(x, threshold=None, merge_distance=None):
    if np.isnan(x).all():
        return np.array([np.nan, np.nan])
    if not threshold:
        threshold = np.nanmean(x)
    if not merge_distance:
        merge_distance = int(np.nanstd(x) * 165)
    if merge_distance > 200:
        merge_distance = 29
    # print(threshold, merge_distance)
    # index of all values greater threshold
    idx = x > threshold
    if not np.any(idx):
        return np.array([0, 0])
    # array of start and end points of areas containing values greater threshold
    idx = np.argwhere(idx != np.roll(idx, -1))
    # merge near areas
    even_idx = idx[0::2][np.where(
        np.abs(idx[0::2] - np.roll(idx, 1)[0::2]) > merge_distance)]
    odd_idx = idx[1::2][np.where(
        np.abs(np.roll(idx, -1)[1::2] - idx[1::2]) > merge_distance)]
    idx = np.array(list(zip(even_idx, odd_idx))).flatten()
    # remove very small areas (peaks)
    even_idx = idx[1::2][np.where(idx[1::2] - np.roll(idx, 1)[1::2] > 50)]
    odd_idx = idx[0::2][np.where(np.roll(idx, -1)[0::2] - idx[0::2] > 50)]
    if not len(even_idx) or not len(odd_idx):
        return np.array([0, 0])
    # return start and end point of the tape hopefully
    return np.array([odd_idx[0], even_idx[-1]])


def preprocess_series(data):
    sensor_dim = 800
    params_dim = 17
    data.columns = range(data.shape[1])
    timestamp = add_ms(normalize_time(data.loc[:, 0].to_numpy()[:, np.newaxis]))
    num = data._get_numeric_data()
    num[num < -100] = np.nan
    xmean_val = np.nanmean(num.loc[:, params_dim:sensor_dim + params_dim - 1])
    ymean_val = np.nanmean(num.loc[:, sensor_dim + params_dim:])
    x = num.values
    params_dim = x.shape[-1] - 2 * sensor_dim
    print(x.shape)

    y = x[:, (sensor_dim + params_dim):] - ymean_val
    x = x[:, params_dim:(sensor_dim + params_dim)] - xmean_val

    x[x < -0.3] = np.nan
    x[x > 0.3] = np.nan
    y[y < -0.2] = np.nan
    y[y > 0.2] = np.nan

    x = df_fillna(pd.DataFrame(x))
    y = df_fillna(pd.DataFrame(y))
    x, y = smooth_profile(x, y)

    x_sav = signal.savgol_filter(x, 15, 3)
    y_sav = signal.savgol_filter(y, 15, 3)
    return timestamp, x_sav, y_sav


def replace_peaks(profile):
    peaks = signal.find_peaks(profile, threshold=0.01)[0]
    widths = signal.peak_widths(profile, peaks, rel_height=0.4)
    widths = zip(widths[2].astype(int), widths[3].astype(int) + 1)
    widths = [index for peak_start, peak_end in widths
              for index in range(peak_start, peak_end) if peak_end - peak_start < 150]
    profile[widths] = np.nan
    return profile


def smooth_profile(x, y):
    for row in x:
        replace_peaks(row)
    for row in y:
        replace_peaks(row)
    x = df_fillna(pd.DataFrame(x))
    y = df_fillna(pd.DataFrame(y))
    return x, y


def df_fillna(data):
    data = data.interpolate(limit_direction='both', method='linear', axis=1)
    return data.values


def correct_tape_idx(tape_idx, normalized, width_bounds, height_thresholds=(0.045, 0.075),
                     min_distances=(29, 9)):
    tape_width = tape_idx[:, -1] - tape_idx[:, 0]
    tape_width_in_range = np.where(tape_width < width_bounds[0])
    if tape_width_in_range[0].size:
        tape_idx[tape_width_in_range] = np.apply_along_axis(get_indices_over_threshold, axis=1,
                                                            arr=normalized[tape_width_in_range],
                                                            threshold=height_thresholds[0],
                                                            merge_distance=min_distances[0])
    tape_width = tape_idx[:, -1] - tape_idx[:, 0]
    tape_width_in_range = np.where(tape_width > width_bounds[1])
    if tape_width_in_range[0].size:
        tape_idx[tape_width_in_range] = np.apply_along_axis(get_indices_over_threshold, axis=1,
                                                            arr=normalized[tape_width_in_range],
                                                            threshold=height_thresholds[1],
                                                            merge_distance=min_distances[1])


def subproc(f):
    sensor_dim = 800
    x_values = np.arange(sensor_dim)
    print(f.name)
    setup, velocity = decode_filename(f.name)
    # remove major flaws
    time, x, y = preprocess_series(pd.read_csv(f))
    # rotate data so that the underlying bar is horizontal
    x_bars = np.array([calculate_bar(sample)(x_values) for sample in x])
    y_bars = np.array([calculate_bar(sample)(x_values) for sample in y])
    normalized_x = x - x_bars
    normalized_y = y - y_bars
    print('normalized shapes', normalized_x.shape, normalized_y.shape)
    # remove time offset of x and y data
    offset_time = calc_offset_time(setup, velocity)
    offset_index = calc_offset_index(offset_time, time)
    normalized_x = normalized_x[:-offset_index]
    normalized_y = normalized_y[offset_index:]
    print('after offset', normalized_x.shape, normalized_y.shape)

    # average data
    normalized_x, normalized_y = average_data(normalized_x, normalized_y, velocity)
    normalized_x, normalized_y = remove_nans(normalized_x, normalized_y)
    print('after averaging and removing nans', normalized_x.shape, normalized_y.shape)
    # get start and end point of each tape measurement
    x_tape_idx = np.apply_along_axis(
        get_indices_over_threshold, axis=1, arr=normalized_x)
    y_tape_idx = np.apply_along_axis(
        get_indices_over_threshold, axis=1, arr=normalized_y)
    # try to fix start and end points where tape width is out of bounds
    correct_tape_idx(x_tape_idx, normalized_x, (180, 400))
    correct_tape_idx(y_tape_idx, normalized_y, (270, 450))

    # center data
    center_data(x_tape_idx, y_tape_idx, normalized_x, normalized_y)
    bar_positions = np.repeat(np.expand_dims(
        get_bar_positions(setup), axis=0), normalized_x.shape[0], axis=0)
    data = np.concatenate((bar_positions, normalized_x, normalized_y), axis=1)
    # write to file
    # pd.DataFrame(data).to_csv(output_dir / f.name, index=False,
    #                           header=False, float_format='%.3f')
    return data


def remove_nans(normalized_x, normalized_y):
    x_nans = ~np.isnan(normalized_x).all(axis=1)
    normalized_x = normalized_x[x_nans]
    normalized_y = normalized_y[x_nans]
    y_nans = ~np.isnan(normalized_y).all(axis=1)
    normalized_x = normalized_x[y_nans]
    normalized_y = normalized_y[y_nans]
    return normalized_x, normalized_y


def average_data(normalized_x, normalized_y, velocity):
    win_size = calc_windowsize(velocity)

    def grouped_avg(a, n=2.):
        result = np.cumsum(a, 0)[n - 1::n] / n
        result[1:] = result[1:] - result[:-1]
        return result

    normalized_x = grouped_avg(normalized_x, win_size)
    normalized_y = grouped_avg(normalized_y, win_size)
    return normalized_x, normalized_y


def center_data(x_tape_idx, y_tape_idx, x_data, y_data):
    tape_width = x_tape_idx[:, -1] - x_tape_idx[:, 0]
    centered_x_idx = np.empty_like(x_tape_idx)
    centered_x_idx[:, 0] = x_data.shape[-1] // 2 - (tape_width // 2)
    centered_x_idx[:, 1] = centered_x_idx[:, 0] + tape_width
    tape_width = y_tape_idx[:, -1] - y_tape_idx[:, 0]
    centered_y_idx = np.empty_like(y_tape_idx)
    centered_y_idx[:, 0] = y_data.shape[-1] // 2 - (tape_width // 2)
    centered_y_idx[:, 1] = centered_y_idx[:, 0] + tape_width
    x_data[((centered_x_idx[:, 0][:, np.newaxis] - 1 < np.arange(x_data.shape[-1])) &
            (centered_x_idx[:, -1][:, np.newaxis] + 1 > np.arange(x_data.shape[-1])))] = x_data[
                ((x_tape_idx[:, 0][:, np.newaxis] - 1 < np.arange(x_data.shape[-1])) &
                 (x_tape_idx[:, -1][:, np.newaxis] + 1 > np.arange(x_data.shape[-1])))]

    y_data[((centered_y_idx[:, 0][:, np.newaxis] - 1 < np.arange(y_data.shape[-1])) &
            (centered_y_idx[:, -1][:, np.newaxis] + 1 > np.arange(y_data.shape[-1])))] = y_data[
                ((y_tape_idx[:, 0][:, np.newaxis] - 1 < np.arange(y_data.shape[-1])) &
                 (y_tape_idx[:, -1][:, np.newaxis] + 1 > np.arange(y_data.shape[-1])))]

    x_data[~((centered_x_idx[:, 0][:, np.newaxis] - 1 < np.arange(x_data.shape[-1])) &
             (centered_x_idx[:, -1][:, np.newaxis] + 1 > np.arange(x_data.shape[-1])))] = 0
    y_data[~((centered_y_idx[:, 0][:, np.newaxis] - 1 < np.arange(y_data.shape[-1])) &
             (centered_y_idx[:, -1][:, np.newaxis] + 1 > np.arange(y_data.shape[-1])))] = 0


if __name__ == '__main__':
    data_path = Path(dirs.DATA)
    output_dir = data_path / 'preprocessed'
    if not output_dir.exists():
        output_dir.mkdir()
    files = [f for f in data_path.iterdir() if f.suffix ==
             '.csv' and f.name not in BAD_FILES]
    print(len(files))
    files = [f for f in files if not (output_dir / f.name).exists()]
    print(len(files))
    subproc(Path('/media/julia/Data/datasets/lufpro/real/Razieh27.08_1Tow_Setup5_6,0m_min_3.csv'))
    # preprocess one file after another (single core usage)
    # for f in files:
    #     subproc(f)
    # or use multiprocessing
    # pool = multiprocessing.Pool(6)
    # pool.map(subproc, files)
