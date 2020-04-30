import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from preprocessing.tape_detection import get_tape_width


def geo_mean(stats):
    # use log to avoid overflow
    # drop zero values (to avoid zero output) TODO other solution?
    log_stats = torch.log(stats[stats > 0])
    return torch.exp(torch.sum(log_stats) / log_stats.shape[0])


def geo_standard_dev(stats):
    return torch.exp(torch.sqrt(
        torch.sum(torch.log(stats[stats > 0] / geo_mean(stats[stats > 0])) ** 2)
        / stats[stats > 0].shape[0]))


def get_fraction_above_threshold(widths, threshold):
    return (torch.sum(widths > threshold).to(dtype=torch.float) / widths.shape[0]).item()


def get_fraction_in_area(widths, lower_bounds, step=0.05, title=''):
    widths = widths.detach().numpy()
    bins = np.empty(len(lower_bounds))
    lower_bounds -= step / 2
    for i, lower_bound in enumerate(lower_bounds):
        bins[i] = np.sum((widths >= lower_bound) & (widths < lower_bound + step)) / widths.shape[0]
    lower_bounds += step / 2
    hist, bin_edges = np.histogram(widths, bins=12, range=(lower_bounds[0], lower_bounds[-1]))
    hist = hist / widths.shape[0]
    plt.subplot(1, 2, 1)
    plt.bar(["{:.3f}".format(l_b) for l_b in lower_bounds], bins)
    plt.subplot(1, 2, 2)
    plt.hist(bin_edges[:-1], bin_edges, weights=hist)
    plt.suptitle(title)
    plt.show()


def evaluate_tape_width(targets, preds, plot_stats=False, logger=None):
    if not logger:
        logger = logging.getLogger(__name__)
    targets_widths = get_tape_width(
        targets.view(targets.shape[0], targets.shape[-1]))
    preds_widths = get_tape_width(preds.view(preds.shape[0], preds.shape[-1]))
    logger.info(f'r2 score: {r2_score(targets_widths, preds_widths)}')
    print(f'r2 score: {r2_score(targets_widths, preds_widths)}')
    dev = torch.abs(targets_widths - preds_widths).to(dtype=torch.float) / targets_widths
    width_std, width_mean = torch.std_mean(dev)
    logger.info('stats for (target_w - pred_w)/target_w')
    logger.info(f'std, mean {width_std.item()} {width_mean.item()}')
    logger.info(f'geo_mean dev (> 0) {geo_mean(dev).item()}')
    logger.info(f'geo_std dev (> 0) {geo_standard_dev(dev).item()}')
    ratio = 1. + torch.abs(1 - preds_widths.to(dtype=torch.float) /
                           targets_widths.to(dtype=torch.float))
    logger.info('stats for  pred_w/target_w')
    logger.info(f'ratio mean {torch.mean(ratio).item()}')
    logger.info(f'ratio geo_mean {geo_mean(ratio).item()}')
    logger.info(f'ratio geo_std {geo_standard_dev(ratio).item()}')
    print('stats for  pred_w/target_w')
    print(f'ratio geo_mean {geo_mean(ratio).item()}')
    print(f'ratio geo_std {geo_standard_dev(ratio).item()}')
    print(f'average target width={targets_widths.float().mean().item()} '
          f'prediction width={preds_widths.float().mean().item()}')
    print(f'std target width={targets_widths.float().std().item()} '
          f'prediction width={preds_widths.float().std().item()}')
    logger.info(f'average target width={targets_widths.float().mean().item()} '
                f'prediction width={preds_widths.float().mean().item()}')
    logger.info(f'std target width={targets_widths.float().std().item()} '
                f'prediction width={preds_widths.float().std().item()}')
    if plot_stats:
        thresholds = np.arange(0., 0.3, 0.025)
        get_fraction_in_area(dev, thresholds, step=0.025,
                             title='Distribution (pred_w - target_w) / target_w')
        thresholds = np.arange(0.7, 1.3, 0.025)
        get_fraction_in_area(ratio, thresholds, step=0.025, title='Distribution pred_w / target_w')

    return geo_mean(ratio).item()


def evaluate_tape_width_np_wrapper(targets, preds, plot_stats=False):
    return evaluate_tape_width(torch.tensor(targets), torch.tensor(preds), plot_stats)
