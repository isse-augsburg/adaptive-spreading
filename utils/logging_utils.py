import logging
import datetime
from collections import namedtuple
from pathlib import Path
import utils.paths as dirs


def _get_logger_path(filename, path=None):
    if not path:
        path = Path(dirs.NN_LOGS)
    path = Path(path)
    if not path.exists():
        path.mkdir()
    if not filename:
        filename = 'output_' + str(datetime.datetime.now().strftime('%Y-%m-%d'))
    return path / (filename + '.log')


def set_logging_parameters(filename, path=None):
    logging.basicConfig(level=logging.INFO,
                        filename=_get_logger_path(filename, path),
                        format='%(levelname)s:%(name)s: %(message)s')


def set_rl_logging_parameters(filename):
    set_logging_parameters(filename, dirs.RL_LOGS)


def set_multi_file_logging_parameters(logger_name, filename):
    log_setup = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(_get_logger_path(filename, dirs.RL_LOGS), mode='a')
    stream_handler = logging.StreamHandler()
    log_setup.setLevel(logging.INFO)
    log_setup.addHandler(file_handler)
    log_setup.addHandler(stream_handler)


def log_to_file(msg, logfile):
    log = logging.getLogger(logfile)
    log.info(msg)


def get_results_from_logs(substrings, path=dirs.RF_LOGS):
    files = [f for f in Path(path).iterdir() if f.suffix == '.log' and
             all(s in f.name for s in substrings)]
    BestValueFiles = namedtuple('BestValueFiles', ['mean', 'std', 'lower_mean_std',
                                                   'upper_mean_std', 'filename'])
    results = []
    for f in files:
        std = 1.5
        for line in reversed(list(f.open())):
            if 'ratio geo_mean' in line:
                mean = float(line.split()[-1])
                results.append(BestValueFiles(mean, std, mean / std, mean * std, f.stem))
                break
            if 'ratio geo_std' in line:
                std = float(line.split()[-1])
    return results


def get_best_results(results):
    results.sort(key=lambda p: p[0])
    best_mean = results[0]
    results.sort(key=lambda p: p[3])
    return best_mean, results[0]


def _compare_best_models(shared_ids, unique_ids, path=dirs.RL_LOGS):
    print('find results with', shared_ids)
    for identifier in unique_ids:
        results = get_results_from_logs(shared_ids + identifier, path)
        if len(results) == 0:
            print('no reults for', identifier)
        else:
            best_mean, best_mean_std = get_best_results(results)
            print('best mean for', identifier, best_mean)
            print('best mean*std for', identifier, best_mean_std)


if __name__ == '__main__':
    # best world models (ff nn and rf)
    # _compare_best_models([], [['relu'], ['random_forest']], path=dirs.NN_LOGS)
    # best action model (fixed actions, neuroevolution)
    _compare_best_models(['new_eval', 'nn', 'target_w=340'], [['fixed_actions'],
                         ['ga_wo_env']])
