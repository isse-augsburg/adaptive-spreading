import math
import re
from datetime import datetime
import numpy as np
import torch


MAX_ACTION = 60.

ACTION_RANGES = torch.tensor([[17., 27.], [11., 19.], [12., 37.], [11., 16.], [17., 27.]])

# bar positions according to setup files (see bar_setups.txt oder Setup.pdf)

BAR_CONFIG = {
    'Setup1': [17.2, 16.1, 17.4, 16.1, 17.2],
    'Setup2': [17.2, 16.1, 22.4, 16.1, 17.2],
    'Setup3': [17.2, 11.2, 22.4, 11.1, 17.2],
    'Setup4': [22.2, 11.2, 22.4, 11.1, 22.4],
    'Setup5': [17.1, 11.2, 27.5, 11.1, 17.3],
    'Setup6': [17.2, 16.1, 17.2, 16.1, 17.3],
    'Setup7': [17.2, 16.1, 27.3, 16.1, 17.3],
    'Setup8': [17.2, 16.1, 32.2, 16.1, 17.3],
    'Setup9': [17.2, 16.1, 37.2, 16.1, 17.3],
    'Setup10': [22.1, 16.1, 37.2, 16.1, 22.1],
    'Setup11': [22.1, 16.1, 32.2, 16.1, 22.1],
    'Setup12': [27.3, 16.1, 17.6, 16.1, 27.3],
    'Setup13': [27.3, 11.0, 12.5, 11.0, 27.3],
    'Setup14': [27.3, 11.0, 31.2, 16.2, 22.2],
    'Setup15': [27.3, 19.5, 31.2, 7.8, 22.2],
    'Setup16': [27.3, 11.0, 12.5, 11.0, 27.3],
    'Setup17': [22.1, 16.1, 37.2, 16.1, 22.1],
    'Setup18': [17.2, 16.1, 32.2, 16.1, 17.3]}


FILENAME_DELIMITER = '_'
FILENAME_SETUP_POS = 0
FILENAME_VELO_POS = 1

DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'


BAD_FILES = ['Setup10_2,7,0m_min_1.csv',
             'Setup10_2,7,0m_min_2.csv',
             'Setup10_2,7,0m_min_3.csv',
             'Setup7_2,7m_min_1.csv']

BAR_GAP = 8.0
BASE = 5.0  # measure tape for 5mm
FREQUENCY = 500


def decode_filename(name):
    comp = name.split(FILENAME_DELIMITER)
    setup = comp[FILENAME_SETUP_POS]
    velo = comp[FILENAME_VELO_POS]
    velo = float(velo[:-1].replace(',', '.'))
    return setup, velo


def calc_offset_index(offset, time_entries):
    end = time_entries[0] + offset
    for i, time in enumerate(time_entries):
        if time >= end:
            return i
    return len(time_entries) - 1


def calc_offset_time(setup, velocity):
    bar_pos = BAR_CONFIG[setup]
    # meter/minute to cm/second
    velocity *= 100.0
    velocity /= 60.0
    c = 0
    for i in range(len(bar_pos) - 1):
        # pythagoras
        a = BAR_GAP
        b = abs(bar_pos[i] - bar_pos[i + 1])
        c += math.sqrt(a ** 2 + b ** 2)
    return c / velocity


def normalize_time(timestamp):
    time_normalized = []
    first_day = timestamp[0][0]
    first_day = datetime.strptime(first_day, DATE_FORMAT)
    for time in timestamp:
        current_day = time[0]
        current_day = datetime.strptime(current_day, DATE_FORMAT)
        difference = (current_day - first_day).total_seconds()
        time_normalized.append(difference)
    return time_normalized


def add_ms(time):
    time_precise = []
    frequencies = check_frequencies(time)
    base_ms = 1 / FREQUENCY
    current_entry = 0
    for i in range(frequencies[0]):
        precise = time[current_entry] + \
            ((i + (FREQUENCY - frequencies[0])) * base_ms)
        time_precise.append(precise)
        current_entry += 1
    for i in frequencies[1:]:
        for j in range(i):
            precise = time[current_entry] + (j * base_ms)
            time_precise.append(precise)
            current_entry += 1
    return time_precise


def check_frequencies(time):
    frequency_count = np.bincount(time)
    return frequency_count


def calc_windowsize(velocity):
    velo_precise = (velocity / 60) * 1000
    p = BASE / velo_precise
    window_size = int(p * FREQUENCY)
    return window_size


def get_bar_positions(setup):
    return np.array(BAR_CONFIG[setup])
