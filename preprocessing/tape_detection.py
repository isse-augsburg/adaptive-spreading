import torch


def get_tape_edges(profiles, axis=1):
    nonzeros = (profiles > torch.unsqueeze(torch.mean(profiles, dim=1), -1))
    tape_counter = nonzeros.cumsum(axis)
    tape_counter[~nonzeros] = 0
    temp_tape_profile, temp_tape_start_idx = (tape_counter == 1).max(axis)
    temp_tape_start_idx[temp_tape_profile == 0] = 0
    temp_tape_profile, temp_tape_end_idx = tape_counter.max(axis)
    temp_tape_end_idx[temp_tape_profile == 0] = 0
    return temp_tape_start_idx, temp_tape_end_idx


def get_tape_width(profiles, axis=1):
    temp_tape_start_idx, temp_tape_end_idx = get_tape_edges(profiles, axis)
    return temp_tape_end_idx - temp_tape_start_idx
