import sys
from collections import OrderedDict
from pathlib import Path
import logging
import torch
import numpy as np
from preprocessing.split_datasets import get_test_set_respecting_files
from preprocessing.preprocessing_helper import BAR_CONFIG
from utils.tape_width_evaluation import evaluate_tape_width
from utils import logging_utils
from utils.logging_utils import set_multi_file_logging_parameters
from process_control_model.model_adapter import init_world_model
from process_control_model.ne_utils import MAX_ACTION, ACTION_RANGES, create_target
from utils.paths import NN_BACKEND, RF_BACKEND
from process_control_model.ne_parameters import TapeTargets
import process_control_model.ne_parameters as rn
import utils.paths as dirs


def test_fixed_setup(setup='Setup9'):
    if setup is 'MAX':
        actions = ACTION_RANGES[:, -1].float().to(device)
    elif setup is 'MIN':
        actions = ACTION_RANGES[:, 0].float().to(device)
    else:
        actions = torch.tensor(BAR_CONFIG[setup]).float().to(device)
    test_x[:, :RL_PARAM_DICT[rn.setup_dim]] = actions / MAX_ACTION
    resulting_tapes = world_model.predict(test_x.to(device))
    target = create_target(RL_PARAM_DICT[rn.target_width], RL_PARAM_DICT[rn.target_height],
                           dim=test_x.shape[0])
    evaluate_tape_width(target, resulting_tapes.cpu(), logger=LOGGER)
    #choice = torch.randint(0, test_x.shape[0], (200,))
    #render(test_x[choice, RL_PARAM_DICT[rn.setup_dim]:].cpu(), resulting_tapes[choice].cpu(),
    #       actions[choice].cpu(), RL_PARAM_DICT[rn.target_width], RL_PARAM_DICT[rn.target_height])


if __name__ == '__main__':
    RL_PARAM_DICT = OrderedDict()
    # default values
    RL_PARAM_DICT[rn.random_seed] = 42
    RL_PARAM_DICT[rn.data_seed] = 1111
    RL_PARAM_DICT[rn.batch_size] = 64
    RL_PARAM_DICT[rn.sensor_dim] = 800
    RL_PARAM_DICT[rn.setup_dim] = 5
    # define target
    if len(sys.argv) > 1:
        tape_targets = TapeTargets(int(sys.argv[1]), 0.15)
    else:
        tape_targets = TapeTargets(240, 0.15)
    RL_PARAM_DICT[rn.target_width] = tape_targets.target_width
    RL_PARAM_DICT[rn.target_height] = tape_targets.target_height
    # world model
    RL_PARAM_DICT[rn.world_model_type] = 'nn'
    RL_PARAM_DICT[rn.world_model_path] = NN_BACKEND if RL_PARAM_DICT[rn.world_model_type] is 'nn'\
                                              else RF_BACKEND

    torch.manual_seed(RL_PARAM_DICT[rn.random_seed])
    np.random.seed(RL_PARAM_DICT[rn.random_seed])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('load world model')
    world_model = init_world_model(RL_PARAM_DICT[rn.world_model_type],
                                   RL_PARAM_DICT[rn.world_model_path], device)
    print('load data')
    test_x, _ = get_test_set_respecting_files(Path(dirs.DATA),
        RL_PARAM_DICT[rn.sensor_dim], RL_PARAM_DICT[rn.setup_dim], random_seed=1111)
    test_x = torch.from_numpy(test_x).float().to(device)
    setups = range(1, 17)
    for i in setups:
        setup = f'Setup{i}'
        RL_PARAM_DICT[rn.rl_model_path] = f'fixed_actions={setup}_on_' \
                                          f'{RL_PARAM_DICT[rn.world_model_type]}_target_w=' \
                                          f'{RL_PARAM_DICT[rn.target_width]}_' \
                                          f'target_h={RL_PARAM_DICT[rn.target_height]}' \
                                          f'_new_eval_simpleprep3'
        RL_PARAM_DICT[rn.logging_path] = dirs.NE_LOGS
        set_multi_file_logging_parameters(setup, RL_PARAM_DICT[rn.rl_model_path])
        LOGGER = logging.getLogger(setup)
        print(RL_PARAM_DICT[rn.rl_model_path])
        LOGGER.info(f'Create {RL_PARAM_DICT[rn.rl_model_path]}')
        rn.save_parameters(RL_PARAM_DICT, RL_PARAM_DICT[rn.rl_model_path])

        test_fixed_setup(setup=setup)
