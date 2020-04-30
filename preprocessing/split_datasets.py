from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from preprocessing.preprocessing_helper import MAX_ACTION
import utils.paths as dirs


NUM_PROCESSES = 6


def scale_setup_values(data, setup_dim):
    data[:, :setup_dim] /= MAX_ACTION


def shuffled_x_y_split(data, sensor_dim, setup_dim, profile_only):
    if profile_only:
        train_x, test_x, train_y, test_y = train_test_split(
            data[:, setup_dim: sensor_dim + setup_dim],
            data[:, sensor_dim + setup_dim:], test_size=0.2, random_state=42)
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=0.2, random_state=42)
    else:
        scale_setup_values(data, setup_dim)
        train_x, test_x, train_y, test_y = train_test_split(
            data[:, : sensor_dim + setup_dim], data[:, sensor_dim + setup_dim:],
            test_size=0.2, random_state=42)
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=0.2, random_state=42)
    return train_x, test_x, val_x, train_y, test_y, val_y


def x_y_split(data, sensor_dim, setup_dim, profile_only):
    if profile_only:
        data_x, data_y = data[:, setup_dim: sensor_dim + setup_dim],\
                         data[:, sensor_dim + setup_dim:]
    else:
        scale_setup_values(data, setup_dim)
        data_x, data_y = data[:, : sensor_dim + setup_dim], data[:, sensor_dim + setup_dim:]
    return data_x, data_y


def get_meta_data(filenames, data, sensor_dim, setup_dim, profile_only):
    index = [np.arange(len(df)) for df in data]
    filenames = [f.name for i, f in enumerate(filenames) for _ in range(len(index[i]))]
    index = np.concatenate(index)
    data_x, data_y = x_y_split(np.concatenate(data), sensor_dim, setup_dim, profile_only)
    return data_x, data_y, filenames, index


def ordered_split_respecting_files(training_files, test_files, sensor_dim, setup_dim, profile_only):
    training_files, validation_files = train_test_split(training_files, test_size=0.2,
                                                        random_state=42)
    training_data = mp_read_file_lists(training_files)
    testing_data = mp_read_file_lists(test_files)
    validation_data = mp_read_file_lists(validation_files)
    train_x, train_y = x_y_split(np.concatenate(training_data), sensor_dim, setup_dim, profile_only)
    test_x, test_y = x_y_split(np.concatenate(testing_data), sensor_dim, setup_dim, profile_only)
    val_x, val_y = x_y_split(np.concatenate(validation_data), sensor_dim, setup_dim, profile_only)
    return train_x, test_x, val_x, train_y, test_y, val_y


def ordered_split_and_meta(training_files, test_files, sensor_dim, setup_dim, profile_only):
    training_files, validation_files = train_test_split(training_files, test_size=0.2,
                                                        random_state=42)
    training_data = mp_read_file_lists(training_files)
    testing_data = mp_read_file_lists(test_files)
    validation_data = mp_read_file_lists(validation_files)
    return get_meta_data(training_files, training_data, sensor_dim, setup_dim, profile_only),\
        get_meta_data(test_files, testing_data, sensor_dim, setup_dim, profile_only),\
        get_meta_data(validation_files, validation_data, sensor_dim, setup_dim, profile_only)


def mp_read_file_lists(files):
    pool = Pool(NUM_PROCESSES)
    data = pool.map(_mp_read_csv, files)
    pool.close()
    pool.join()
    return data


def _mp_read_csv(filename):
    return pd.read_csv(filename, header=None).to_numpy()


def read_csvs_from_list(filenames):
    data = None
    for filename in filenames:
        temp_df = pd.read_csv(filename, header=None)
        data = temp_df if data is None else data.append(temp_df)
    return data


def load_one_file_and_split(filepath, sensor_dim, setup_dim, profile_only=False):
    data = pd.read_csv(filepath, header=None).to_numpy()
    return shuffled_x_y_split(data, sensor_dim, setup_dim, profile_only)


def shuffle_all_data_and_split(data_path, sensor_dim, setup_dim, profile_only=False):
    files = [f for f in data_path.iterdir() if f.suffix == '.csv']
    data = read_csvs_from_list(files).to_numpy()
    return shuffled_x_y_split(data, sensor_dim, setup_dim, profile_only)


def split_data_excluding_multiple_setups(data_path, sensor_dim, setup_dim,
                                         profile_only=False, setups=(8, 18)):
    substrings = [f'Setup{setup}' for setup in setups]
    training_files = shuffle([f for f in data_path.iterdir()
                              if f.suffix == '.csv'
                              and not any(s in f.name for s in substrings)])
    test_files = [f for f in data_path.iterdir() if f.suffix == '.csv'
                  and any(s in f.name for s in substrings)]
    return ordered_split_respecting_files(training_files, test_files, sensor_dim,
                                          setup_dim, profile_only)


def split_data_excluding_one_setup(data_path, sensor_dim, setup_dim, profile_only=False, setup=8):
    substring = f'Setup{setup}_'
    training_files = shuffle([f for f in data_path.iterdir()
                              if f.suffix == '.csv'
                              and substring not in f.name])
    test_files = [f for f in data_path.iterdir() if f.suffix == '.csv'
                  and substring in f.name]
    return ordered_split_respecting_files(training_files, test_files, sensor_dim,
                                          setup_dim, profile_only)


def split_data_respecting_files_including_meta(data_path, sensor_dim, setup_dim,
                                               profile_only=False, random_seed=42):
    files = shuffle([f for f in data_path.iterdir() if f.suffix == '.csv'],
                    random_state=random_seed)
    training_files, test_files = train_test_split(files, test_size=0.2,
                                                  random_state=random_seed)
    return ordered_split_and_meta(training_files, test_files, sensor_dim,
                                  setup_dim, profile_only)


def get_test_set_including_meta(data_path, sensor_dim, setup_dim, random_seed=42):
    files = shuffle([f for f in data_path.iterdir() if f.suffix == '.csv'],
                    random_state=random_seed)
    _, test_files = train_test_split(files, test_size=0.2, random_state=random_seed)
    return get_meta_data(test_files, mp_read_file_lists(test_files), sensor_dim,
                         setup_dim, profile_only=False)


def split_data_respecting_files(data_path, sensor_dim, setup_dim,
                                profile_only=False, random_seed=42):
    files = shuffle([f for f in data_path.iterdir() if f.suffix == '.csv'],
                    random_state=random_seed)
    training_files, test_files = train_test_split(files, test_size=0.2,
                                                  random_state=random_seed)
    return ordered_split_respecting_files(training_files, test_files, sensor_dim,
                                          setup_dim, profile_only)


def get_test_set_respecting_files(data_path, sensor_dim, setup_dim, random_seed=42):
    files = shuffle([f for f in data_path.iterdir() if f.suffix == '.csv'],
                    random_state=random_seed)
    _, test_files = train_test_split(files, test_size=0.2, random_state=random_seed)
    test_data = np.concatenate(mp_read_file_lists(test_files))
    test_x, test_y = x_y_split(test_data, sensor_dim, setup_dim, profile_only=False)
    return test_x, test_y


if __name__ == '__main__':
    FUNC = split_data_respecting_files
    TRAIN_X, TEST_X, VAL_X, TRAIN_Y, TEST_Y, VAL_Y = FUNC(
        Path(dirs.DATA),
        800, 5)
    print(FUNC.__name__, TRAIN_X.shape, TRAIN_Y.shape, TEST_X.shape,
          TEST_Y.shape, VAL_X.shape, VAL_Y.shape)
