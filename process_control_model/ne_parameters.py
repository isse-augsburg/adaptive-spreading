import json
from collections import namedtuple
from pathlib import Path
import utils.paths as dirs

random_seed = 'random_seed'
data_seed = 'data_seed'
sensor_dim = 'sensor_dim'
setup_dim = 'setup_dim'
target_width = 'target_width'
target_height = 'target_height'
k_h = 'k_h'
k_w = 'k_w'
k_m = 'k_m'
world_model_type = 'world_model_type'
world_model_path = 'world_model_path'
hidden_dims = 'hidden_dims'
learning_rate = 'learning_rate'
batch_size = 'batch_size'
rl_model_path = 'ne_model_path'
logging_path = 'logging_path'
critic_architecture = 'critic_architecture'
actor_architecture = 'actor_architecture'
population_size = 'population_size'
ga_noise = 'ne_noise'
reproducing_parents = 'reproducing_parents'


def create_rl_model_path(prefix, param_dict):
    return '_'.join([prefix] + [str(i) for i in param_dict[hidden_dims]] +
                    [param_dict[world_model_type], f'lr={param_dict[learning_rate]}',
                     f'bs={param_dict[batch_size]}',
                     'rescaled_actions', f'target_w={param_dict[target_width]}',
                     f'target_h={param_dict[target_height]}', f'k_h={param_dict[k_h]}',
                     f'k_w={param_dict[k_w]}',
                     f'k_m={param_dict[k_m]}'])


def create_ga_model_path(prefix, param_dict):
    return '_'.join([prefix] + [str(i) for i in param_dict[hidden_dims]] +
                    [param_dict[world_model_type],
                     f'population_size={param_dict[population_size]}',
                     f'noise={param_dict[ga_noise]}',
                     f'rep_parents={param_dict[reproducing_parents]}',
                     f'target_w={param_dict[target_width]}',
                     f'target_h={param_dict[target_height]}',
                     f'k_h={param_dict[k_h]}', f'k_w={param_dict[k_w]}',
                     f'k_m={param_dict[k_m]}'])


RewardFactors = namedtuple('RewardFactors', ['height_factor', 'width_factor', 'movement_factor'])
TapeTargets = namedtuple('TapeTargets', ['target_width', 'target_height'])


PARAMETERS_DIR = Path(dirs.NE_PARAMS)


def save_parameters(param_dict, filename):
    if not PARAMETERS_DIR.exists():
        PARAMETERS_DIR.mkdir()
    with (PARAMETERS_DIR / (filename + '.json')).open(mode='w') as output_file:
        json.dump(param_dict, output_file)


def load_parameters(filename):
    rl_param_dict = json.load((PARAMETERS_DIR / Path(filename)).open())
    tape_targets = TapeTargets(rl_param_dict[target_width], rl_param_dict[target_height])
    reward_factors = RewardFactors(rl_param_dict[k_h], rl_param_dict[k_w], rl_param_dict[k_m])
    return rl_param_dict, reward_factors, tape_targets
