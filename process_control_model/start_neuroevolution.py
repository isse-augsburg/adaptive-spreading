import sys
from pathlib import Path
import copy
from collections import OrderedDict
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from preprocessing.split_datasets import split_data_respecting_files, get_test_set_respecting_files
from utils.tape_width_evaluation import evaluate_tape_width
import utils.logging_utils as logging_config
from utils.nn_utils import count_trainable_parameters
from process_control_model.model_adapter import init_world_model
from process_control_model.ne_utils import MAX_ACTION, create_target, clear_setup, \
    transform_actions_prob, get_reward
from process_control_model.ne_parameters import RewardFactors, TapeTargets
import process_control_model.ne_parameters as rp
from utils.paths import NN_BACKEND, RF_BACKEND
from plotter.ne_plotter import render
import utils.paths as dirs


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(Net, self).__init__()
        act = nn.ReLU()
        layers = OrderedDict()
        layers['layer_0'] = nn.Linear(state_dim, hidden_dims[0])
        layers['act_0'] = act
        for i, h_d in enumerate(hidden_dims[1:]):
            layers[f'layer_{i + 1}'] = nn.Linear(hidden_dims[i], h_d)
            layers[f'act_{i + 1}'] = act
        layers[f'layer_{len(hidden_dims)}'] = nn.Linear(hidden_dims[-1], action_dim)
        layers[f'act_{len(hidden_dims)}'] = nn.Sigmoid()
        self.base = nn.Sequential(layers)

    def forward(self, x):
        return self.base(x)


def create_new_population():
    networks = [Net(RL_PARAM_DICT[rp.sensor_dim] + RL_PARAM_DICT[rp.setup_dim],
                    RL_PARAM_DICT[rp.setup_dim],
                    RL_PARAM_DICT[rp.hidden_dims]).to(device)
                for _ in range(RL_PARAM_DICT[rp.population_size])]
    print('validate')
    select_data = np.random.choice(train_x.shape[0], 512, replace=False)
    population = [(determine_fitness(net, train_x[select_data]), net)
                  for net in networks]
    return population


def rank_genotypes(population):
    population.sort(key=lambda p: p[0], reverse=True)


def reduce_mutation_rate(factor=0.6, min_rate=0.01):
    RL_PARAM_DICT[rp.ga_noise] *= factor
    if RL_PARAM_DICT[rp.ga_noise] < min_rate:
        RL_PARAM_DICT[rp.ga_noise] = min_rate


def save_genotype(genotype):
    if not dirs.NE_WEIGHTS.exists():
        dirs.NE_WEIGHTS.mkdir()
    torch.save(genotype.state_dict(),
               dirs.NE_WEIGHTS / (RL_PARAM_DICT[rp.rl_model_path] + '.pth'))
    if not dirs.NE_MODELS.exists():
        dirs.NE_MODELS.mkdir()
    torch.save(genotype, dirs.NE_MODELS / (RL_PARAM_DICT[rp.rl_model_path] + '.pth'))


def set_up_first_generation():
    population = create_new_population()
    rank_genotypes(population)
    best_fitness = determine_fitness(population[0][1], val_x, 0).item()
    logger.info(str(population[0][1]))
    logger.info(f'Trainable Parameters: {count_trainable_parameters(population[0][1])}')
    return population, best_fitness


def start_evolution():
    population, best_fitness = set_up_first_generation()
    patience = 10
    stagnating_epochs = 0
    for i in range(1000):
        best_parent = population[0]
        select_parents = np.random.choice(RL_PARAM_DICT[rp.reproducing_parents], len(population))
        mutants = [mutate(population[j][1]) for j in select_parents]
        select_data = np.random.choice(train_x.shape[0], 512, replace=False)
        population = [(determine_fitness(net, train_x[select_data]),
                       net) for net in mutants]
        population.append(best_parent)
        rank_genotypes(population)
        print(f'generation {i + 1}')
        print(f'best on train: {population[0][0].item()}')
        logger.info(f'generation {i + 1}')
        logger.info(f'best on train: {population[0][0].item()}')
        writer.add_scalar('train reward', population[0][0].item(), i)
        population_fitness = determine_fitness(population[0][1], val_x, i, True).item()
        print(f'best on validation: {population_fitness}')
        logger.info(f'best on validation: {population_fitness}')
        writer.add_scalar('validation reward', population_fitness, i)
        if population_fitness > best_fitness:
            save_genotype(population[0][1])
            print('save states')
            stagnating_epochs = 0
            best_fitness = population_fitness
        else:
            stagnating_epochs += 1
        if stagnating_epochs > patience:
            print(f'early stopping kicked in after {patience} epochs without improvement')
            logger.info(f'early stopping kicked in after {patience} epochs without improvement')
            test_fitness = determine_fitness(population[0][1], test_x, -1, True, False).item()
            print(f'best on test: {test_fitness}')
            logger.info(f'best on test: {test_fitness}')
            break
        if i % 5 == 0:
            reduce_mutation_rate()
            logger.info('reduced mutation rate')
            print('reduced mutation rate')


def determine_fitness(net, data, generation=1, evaluate_resulting_tapes=False, plot_results=False):
    with torch.no_grad():
        actions = net(data)
        scaled_actions = actions.clone().detach()
        transform_actions_prob(scaled_actions)
        normed_actions = scaled_actions.clone() / MAX_ACTION
        world_model_input = data.clone().detach()
        world_model_input[:, :RL_PARAM_DICT[rp.setup_dim]] = normed_actions
        resulting_tapes = world_model.predict(world_model_input)
        rewards = get_reward(resulting_tapes.detach(), scaled_actions,
                             data[:, :RL_PARAM_DICT[rp.setup_dim]].detach() * MAX_ACTION,
                             REWARD_FACTORS, TAPE_TARGETS)
        if evaluate_resulting_tapes:
            target = create_target(RL_PARAM_DICT[rp.target_width],
                                   RL_PARAM_DICT[rp.target_height],
                                   dim=resulting_tapes.shape[0])
            avg_tape_width_dev = evaluate_tape_width(target, resulting_tapes.cpu())
            if generation >= 0:
                writer.add_scalar('validation width ratio', avg_tape_width_dev, generation)
        if plot_results:
            choice = torch.randint(0, data.shape[0], (200,))
            render(data[choice, RL_PARAM_DICT[rp.setup_dim]:].cpu(),
                   resulting_tapes[choice].cpu(),
                   scaled_actions[choice].cpu(),
                   RL_PARAM_DICT[rp.target_width],
                   RL_PARAM_DICT[rp.target_height])

    return torch.mean(rewards)


def mutate(parent_net):
    net = copy.deepcopy(parent_net)
    for p in net.parameters():
        p.data += RL_PARAM_DICT[rp.ga_noise] * torch.randn_like(p.data)
    return net


def test_trained_model():
    if (dirs.NE_WEIGHTS / (RL_PARAM_DICT[rp.rl_model_path] + '.pth')).exists():
        net = torch.load(dirs.NE_MODELS / (RL_PARAM_DICT[rp.rl_model_path] + '.pth'))
    else:
        net = Net(RL_PARAM_DICT[rp.sensor_dim] + RL_PARAM_DICT[rp.setup_dim], RL_PARAM_DICT[rp.setup_dim],
                  RL_PARAM_DICT[rp.hidden_dims])
        net.load_state_dict(torch.load(dirs.NE_WEIGHTS / (RL_PARAM_DICT[rp.rl_model_path] + '.pth')))
    net.to(device)
    test_fitness = determine_fitness(net, test_x, -1, True, True).item()
    print(f'reward on test set: {test_fitness}')
    logger.info(f'reward on test set: {test_fitness}')


def set_parameters():
    rl_param_dict = OrderedDict()
    # default values
    rl_param_dict[rp.random_seed] = 42
    rl_param_dict[rp.data_seed] = 1111
    rl_param_dict[rp.sensor_dim] = 800
    rl_param_dict[rp.setup_dim] = 5
    # define target
    tape_targets = TapeTargets(240, 0.15)
    rl_param_dict[rp.target_width] = tape_targets.target_width
    rl_param_dict[rp.target_height] = tape_targets.target_height
    # specify reward function
    reward_factors = RewardFactors(1.0, 4.0, 0.5)
    rl_param_dict[rp.k_h] = reward_factors.height_factor
    rl_param_dict[rp.k_w] = reward_factors.width_factor
    rl_param_dict[rp.k_m] = reward_factors.movement_factor
    # world model parameters
    rl_param_dict[rp.world_model_type] = 'nn'
    rl_param_dict[rp.world_model_path] = NN_BACKEND if rl_param_dict[rp.world_model_type]\
                                                       == 'nn' else RF_BACKEND
    # ga parameters
    rl_param_dict[rp.hidden_dims] = [128, 16]
    # population size  >  reproducing parents
    rl_param_dict[rp.population_size] = 300
    rl_param_dict[rp.reproducing_parents] = 15
    rl_param_dict[rp.ga_noise] = 0.2
    rl_param_dict[rp.rl_model_path] = rp.create_ga_model_path(
        'ga_wo_env_first_step_noise_reduction_new_eval_simpleprep3_small_backend', rl_param_dict)
    rl_param_dict[rp.logging_path] = dirs.NE_LOGS
    print(rl_param_dict[rp.rl_model_path])
    rp.save_parameters(rl_param_dict, rl_param_dict[rp.rl_model_path])
    return rl_param_dict, reward_factors, tape_targets


if __name__ == '__main__':
    TRAINING = True
    if len(sys.argv) > 1:
        RL_PARAM_DICT, REWARD_FACTORS, TAPE_TARGETS = rp.load_parameters(sys.argv[1])
    else:
        RL_PARAM_DICT, REWARD_FACTORS, TAPE_TARGETS = set_parameters()
    torch.manual_seed(RL_PARAM_DICT[rp.random_seed])
    np.random.seed(RL_PARAM_DICT[rp.random_seed])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('load world model')
    world_model = init_world_model(RL_PARAM_DICT[rp.world_model_type],
                                   RL_PARAM_DICT[rp.world_model_path], device)
    print('load data')
    if TRAINING:
        train_x, test_x, val_x = split_data_respecting_files(Path(dirs.DATA),
            RL_PARAM_DICT[rp.sensor_dim], RL_PARAM_DICT[rp.setup_dim],
            random_seed=RL_PARAM_DICT[rp.data_seed])[:3]
        clear_setup(train_x)
        clear_setup(test_x)
        clear_setup(val_x)
        train_x = torch.from_numpy(train_x).float().to(device)
        test_x = torch.from_numpy(test_x).float().to(device)
        val_x = torch.from_numpy(val_x).float().to(device)
    else:
        test_x = get_test_set_respecting_files(Path(dirs.DATA),
            RL_PARAM_DICT[rp.sensor_dim], RL_PARAM_DICT[rp.setup_dim],
            random_seed=RL_PARAM_DICT[rp.data_seed])[0]
        clear_setup(test_x)
        test_x = torch.from_numpy(test_x).float().to(device)

    logging_config.set_rl_logging_parameters(RL_PARAM_DICT[rp.rl_model_path])
    logger = logging.getLogger(__name__)

    logger.info(f'target_width={RL_PARAM_DICT[rp.target_width]}')
    logger.info(f'target_height={RL_PARAM_DICT[rp.target_height]}')

    logger.info(f'reward factors: height_diff * {RL_PARAM_DICT[rp.k_h]} + '
                f'width_diff * {RL_PARAM_DICT[rp.k_w]}'
                f' + movement * {RL_PARAM_DICT[rp.k_m]}')
    logger.info(f'Create {RL_PARAM_DICT[rp.rl_model_path]} on {device}')
    if TRAINING:
        writer = SummaryWriter(Path(dirs.NE_TENSORBOARD, RL_PARAM_DICT[rp.rl_model_path]))
        start_evolution()
    else:
        test_trained_model()
