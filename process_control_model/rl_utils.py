import numpy as np
import torch
from torch.utils.data import Dataset
from process_model.feedforward import Feedforward
from preprocessing.tape_detection import get_tape_edges
from preprocessing.preprocessing_helper import MAX_ACTION, ACTION_RANGES


def create_target(target_width, target_height, dim=1, sensor_dim=800):
    target = torch.zeros((dim, sensor_dim))
    begin_target_tape = sensor_dim // 2 - target_width // 2
    target[:, begin_target_tape: begin_target_tape + target_width] = target_height
    return target


def load_pretrained_ff_model(path, device, hidden_dims=(300, 300, 300)):
    ff_model = Feedforward(805, hidden_dims, 800)
    ff_model.load_state_dict(torch.load(path, map_location=device))
    ff_model.to(device)
    for param in ff_model.parameters():
        param.requires_grad = False
    ff_model.eval()
    return ff_model


class RLDataSet(Dataset):
    def __init__(self, observations):
        self._observations = observations

    def __len__(self):
        return self._observations.shape[0]

    def __getitem__(self, item):
        return self._observations[item]


def clear_setup(data):
    data[:, :5] = np.array([17., 11., 22., 11., 17.]) / MAX_ACTION


def _get_height_difference(rows, edges, target_height, e_min=0, e_max=799, min_len=180):
    edges[0][edges[0] < e_min] = e_min
    edges[1][edges[0] == edges[1]] += min_len
    edges[1][edges[1] > e_max] = e_max
    device = 'cuda:0' if rows.is_cuda else 'cpu'
    avg_heights = torch.empty(rows.shape[0]).to(device)
    for i, row in enumerate(rows):
        avg_heights[i] = torch.mean(row[edges[0][i]: edges[1][i]])
    return -torch.abs(avg_heights - target_height)


def _get_width_difference(edges, target_width, e_min=0, e_max=799, max_width=800):
    edges[0][edges[0] < e_min] = e_min
    edges[1][edges[1] > e_max] = e_max
    return -(abs((edges[1] - edges[0]) - target_width).float() / max_width)


def _get_bar_movement(actions, previous_actions, max_movement=250):
    total_movement = torch.sum(torch.abs(previous_actions - actions), dim=1)
    return -total_movement / max_movement


def get_reward(resulting_tapes, actions, previous_actions, reward_factors, tape_targets):
    edges = get_tape_edges(resulting_tapes)
    y_height = _get_height_difference(resulting_tapes, edges, tape_targets.target_height)
    y_width = _get_width_difference(edges, tape_targets.target_width)
    y_bar_movement = _get_bar_movement(actions, previous_actions)
    y = reward_factors.height_factor * y_height +\
        reward_factors.width_factor * y_width +\
        reward_factors.movement_factor * y_bar_movement
    if torch.isnan(y).any():
        print('NAN')
    return y


def transform_actions_prob(actions):
    device = 'cuda:0' if actions.is_cuda else 'cpu'
    actions *= (ACTION_RANGES[:, 1] - ACTION_RANGES[:, 0]).to(device)
    actions += ACTION_RANGES[:, 0].to(device)
