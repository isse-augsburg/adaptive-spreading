import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error as mse
from joblib import dump, load
from utils.tape_width_evaluation import evaluate_tape_width_np_wrapper
import utils.paths as dirs


LOGGER = logging.getLogger(__name__)


def load_model(name):
    return load(dirs.RF_MODELS / (name + '.joblib'))


def save_model(reg, name):
    if not dirs.RF_MODELS.exists():
        dirs.RF_MODELS.mkdir()
    dump(reg, dirs.RF_MODELS / (name + '.joblib'))


def train_rf(train_x, train_y, estimators=10, min_samples_split=2, min_samples_leaf=1,
             max_depth=None, random_state=1111, max_features='auto', jobs=8):
    reg = RandomForestRegressor(n_estimators=estimators, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                                n_jobs=jobs, verbose=20, random_state=random_state,
                                max_features=max_features, bootstrap=True)
    LOGGER.info(str(reg))
    reg.fit(train_x, train_y)
    return reg


def train_extra_trees(train_x, train_y, estimators=10, min_samples_split=2, min_samples_leaf=1,
                      max_depth=None, random_state=1111, max_features='auto', jobs=8):
    reg = ExtraTreesRegressor(n_estimators=estimators, min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf, max_depth=max_depth, n_jobs=jobs,
                              verbose=20, random_state=random_state, max_features=max_features,
                              bootstrap=True)
    LOGGER.info(str(reg))
    reg.fit(train_x, train_y)
    return reg


def validate_rf(reg, val_x, val_y, plot_stats=False):
    preds = reg.predict(val_x)
    tape_width_dev = evaluate_tape_width_np_wrapper(val_y, preds, plot_stats)
    LOGGER.info(f'tape width dev: {tape_width_dev:.4f}')
    return mse(val_y, preds), tape_width_dev, preds
