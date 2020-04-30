import logging
from pathlib import Path
from preprocessing.split_datasets import split_data_respecting_files_including_meta, \
    get_test_set_including_meta
from process_model.random_forest_regression import train_rf, validate_rf, save_model, load_model
from plotter.profile_plotter import plot_samples_np_wrapper
from utils.logging_utils import set_logging_parameters
import utils.paths as dirs


def _start_rf():
    set_logging_parameters(MODEL_NAME)
    logger = logging.getLogger(__name__)
    logger.info(MODEL_NAME)
    logger.info('Load data...')
    if IS_TRAINING:
        train_data, test_data, val_data = split_data_respecting_files_including_meta(
            DATA_PATH, SENSOR_DIM, PARAMS_DIM, PROFILE_ONLY_INPUT, RANDOM_SEED)
        logger.info(
            f'Loaded! Trainset size: {train_data[0].shape}, Test set size: {test_data[0].shape}')
        logger.info('Build random forest...')
        reg = train_rf(train_data[0], train_data[1], random_state=RANDOM_SEED,
                       estimators=ESTIMATORS, min_samples_split=MIN_SAMPLES_SPLIT,
                       max_features=MAX_FEAT, max_depth=MAX_DEPTH, jobs=10)
        save_model(reg, MODEL_NAME)
        mse, _, _ = validate_rf(reg, train_data[0], train_data[1])
        logger.info(f'Train loss: {mse}')
        mse, _, preds = validate_rf(reg, val_data[0], val_data[1])
        logger.info(f'Val loss: {mse}')
    else:
        test_data = get_test_set_including_meta(
            DATA_PATH, SENSOR_DIM, PARAMS_DIM, RANDOM_SEED)
        logger.info(f'Loaded! Test set size: {test_data[0].shape}')
        logger.info('Load pretrained model...')
        reg = load_model(MODEL_NAME)
    mse, _, preds = validate_rf(reg, test_data[0], test_data[1], plot_stats=False)
    logger.info(f'Test loss: {mse}')
    plot_samples_np_wrapper(preds, test_data[1], SENSOR_DIM, MODEL_NAME,
                            meta=(test_data[2], test_data[3]))


if __name__ == '__main__':
    IS_TRAINING = False
    DATA_PATH = Path(dirs.DATA)
    SENSOR_DIM = 800
    PARAMS_DIM = 5
    PROFILE_ONLY_INPUT = False
    RANDOM_SEED = 1111
    ESTIMATORS = 50
    MAX_DEPTH = 12
    MIN_SAMPLES_SPLIT = 150
    MAX_FEAT = 300
    MODEL_NAME = f'est_{ESTIMATORS}_maxdepth_{MAX_DEPTH}_min_split_{MIN_SAMPLES_SPLIT}_' \
                 f'max_feats_{MAX_FEAT}_random_forest_regressor_new_eval'
    MODEL_NAME = 'est_50_maxdepth_15_min_split_150_max_feats_200_random_forest_regressor_new_eval'
    _start_rf()
