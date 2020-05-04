from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from preprocessing.preprocessing_helper import ACTION_RANGES
from preprocessing.tape_detection import get_tape_edges
from preprocessing.split_datasets import get_test_set_respecting_files
from utils.logging_utils import get_best_results, get_results_from_logs
import utils.paths as dirs
from process_control_model.model_adapter import NNAdapter, RFAdapter
from process_control_model.ne_utils import transform_actions_prob


def _get_best_models():
    nn_best_mean, nn_best_mean_std = get_best_results(get_results_from_logs(
        ['relu', 'fixed_seed'], path=dirs.NN_LOGS))
    nn_model = NNAdapter(dirs.NN_MODELS / (nn_best_mean_std.filename + '.pth'), 'cpu')
    rf_best_mean, rf_best_mean_std = get_best_results(get_results_from_logs(
        ['random_forest'], path=dirs.RF_LOGS))
    rf_model = RFAdapter(dirs.RF_MODELS / (rf_best_mean_std.filename + '.joblib'), 'cpu')
    return nn_model, rf_model


def _get_test_tensors(num_samples=10):
    test_x, test_y = get_test_set_respecting_files(Path(dirs.DATA), 800, 5, 1111)
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    select_data = np.random.choice(test_x.shape[0], num_samples, replace=False)
    return test_x[select_data], test_y[select_data]


def plot_samples(samples_list, legend, title, plot_edges=False):
    if plot_edges:
        edges = [get_tape_edges(samples_sublist) for samples_sublist in samples_list]
    num_samples = samples_list[0].shape[0]

    font_size = 22
    # rc('text', usetex=True)
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size-4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size-4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
    fig, axs = plt.subplots(num_samples, sharex=True, sharey=True)
    for i, samples_sublist in enumerate(samples_list):
        plot_i = 1
        for j in range(num_samples):
            # plt.subplot(num_samples, 1, plot_i)
            if plot_edges:
                axs[j].plot(samples_sublist[j], '-o', linewidth=1.4,
                            markevery=[edges[i][0][j], edges[i][1][j]])
                # plt.plot(samples_sublist[j], '-o', linewidth=0.8,
                #          markevery=[edges[i][0][j], edges[i][1][j]])
            else:
                # plt.plot(samples_sublist[j], linewidth=0.8)
                axs[j].plot(samples_sublist[j], linewidth=1.4)
            plot_i += 1
    fig.text(0.5, 0.045, 'Sensor pixels', ha='center')
    fig.text(0.08, 0.5, 'Tow height', va='center', rotation='vertical')
    plt.legend(legend, loc='right')
    ttl = fig.suptitle(title)
    ttl.set_position([.5, 0.92])
    plt.show()


def compare_random_actions(num_samples=10):
    nn_model, rf_model = _get_best_models()
    test_x, test_y = _get_test_tensors(num_samples=num_samples)
    # random actions
    # actions = torch.rand_like(test_x[:, :5])
    # transform_actions_prob(actions)
    # or max actions to see the differences
    actions = torch.empty_like(test_x[:, :5])
    actions[:, 0::2] = ACTION_RANGES[0::2, 1]
    actions[:, 1::2] = ACTION_RANGES[1::2, 0]
    # or min actions to see the differences
    # actions = torch.empty_like(test_x[:, :5])
    # actions[:, 0::2] = ACTION_RANGES[0::2, 0]
    # actions[:, 1::2] = ACTION_RANGES[1::2, 1]
    test_x[:, :5] = actions / 60.
    nn_preds = nn_model.predict(test_x)
    rf_preds = rf_model.predict(test_x)
    plot_samples([nn_preds, rf_preds], legend=['ffnn_predictions', 'rf_predictions'],
                 title='Process models - maximum spreading', plot_edges=True)


def compare_on_real_data(num_samples=10):
    nn_model, rf_model = _get_best_models()
    test_x, test_y = _get_test_tensors(num_samples=num_samples)
    nn_preds = nn_model.predict(test_x)
    rf_preds = rf_model.predict(test_x)
    plot_samples([nn_preds, rf_preds, test_y], legend=['ffnn_predictions', 'rf_predictions', 'target'],
                 title='Process models', plot_edges=False)


if __name__ == '__main__':
    # compare_on_real_data()
    compare_random_actions()
