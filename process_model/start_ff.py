import sys
import logging
from pathlib import Path
import torch
import numpy as np
from preprocessing.split_datasets import split_data_respecting_files
from process_model.ff_trainer import FFTrainer
from preprocessing.datasetloader import load_data_and_meta, load_test_data_and_meta
from plotter.profile_plotter import plot_samples
from utils.nn_utils import count_trainable_parameters
from utils.logging_utils import set_logging_parameters
import utils.paths as dirs

# execute in adaptive-spreading directory: python3 -m process_model.start_ff


def _start_ff():
    set_logging_parameters(MODEL_NAME)
    logger = logging.getLogger(__name__)
    logger.info(f'Create {MODEL_NAME} on {DEVICE}')
    trainer = FFTrainer(sensor_dim=SENSOR_DIM, param_dim=PARAMS_DIM, hidden_dimensions=HIDDEN_DIMS,
                        learning_rate=LEARNING_RATE,
                        profile_only_input=PROFILE_ONLY_INPUT, model_path=MODEL_PATH)
    logger.info(str(trainer.model))
    trainer.model.to(DEVICE)
    logger.info(f'Trainable Parameters: {count_trainable_parameters(trainer.model)}')
    logger.info(f'Splitting strategy: {SPLIT_FUNCTION.__name__} {SPLIT_PARAM}')
    args = [DATA_PATH, SENSOR_DIM, PARAMS_DIM, PROFILE_ONLY_INPUT]
    if SPLIT_PARAM:
        args.append(SPLIT_PARAM)
    if IS_TRAINING:
        train_loader, validation_loader, test_loader = load_data_and_meta(
            DEVICE, BATCH_SIZE, *args)
        trainer.train_network(train_loader, validation_loader)
    else:
        test_loader = load_test_data_and_meta(DEVICE, DATA_PATH, SENSOR_DIM, PARAMS_DIM,
                                              SPLIT_PARAM)
    _, all_targets, all_preds, all_filenames, all_indices = trainer.test_network(test_loader)
    # plot_samples(MODEL_NAME, SENSOR_DIM, (all_preds, all_targets),
    #              (all_filenames, all_indices))


if __name__ == '__main__':
    IS_TRAINING = True
    PROFILE_ONLY_INPUT = False
    if len(sys.argv) > 1:
        HIDDEN_DIMS = [int(i) for i in sys.argv[1].strip('[]').split(',')]
    else:
        HIDDEN_DIMS = [100, 100, 100]
    DATA_PATH = Path(dirs.DATA)
    SPLIT_FUNCTION = split_data_respecting_files
    SPLIT_PARAM = 1111
    SENSOR_DIM = 800
    PARAMS_DIM = 5
    LEARNING_RATE = 5e-04
    BATCH_SIZE = 64
    torch.manual_seed(42)
    np.random.seed(SPLIT_PARAM)
    MODEL_NAME = '_'.join([str(i) for i in HIDDEN_DIMS] + ['relu', f'lr={LEARNING_RATE}',
                                                           f'batch_size={BATCH_SIZE}',
                                                           SPLIT_FUNCTION.__name__,
                                                           'new_eval_fixed_seed'])
    MODEL_NAME = '100_100_100_relu_lr=0.0001_batch_size=64_split_data_respecting_files_new_eval'
    if SPLIT_PARAM:
        MODEL_NAME += f'_param={SPLIT_PARAM}'
    if PROFILE_ONLY_INPUT:
        MODEL_NAME += '_profile_only'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = dirs.NN_WEIGHTS / f'{MODEL_NAME}.pth'
    _start_ff()
