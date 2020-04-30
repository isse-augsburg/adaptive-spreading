from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import utils.paths as dirs


SOURCE_DIR = Path(dirs.DATA)
RL_START_FILE = Path('../process_control_model/start_values.csv')
NON_TOUCHING_DIR = Path(dirs.DATA)

SETUP_DIM = 5
SENSOR_DIM = 800

PROCESS_COUNT = 6


def _get_random_start_values_from_file(filename, num_lines=2):
    df = pd.read_csv(filename, header=None)
    random_idx = [np.random.randint(low=0, high=df.shape[0]) for _ in range(num_lines)]
    return df.loc[random_idx, SETUP_DIM:(SETUP_DIM + SENSOR_DIM - 1)].to_numpy()


def _get_random_lines_from_file(filename, num_lines=2):
    df = pd.read_csv(filename, header=None)
    random_idx = [np.random.randint(low=0, high=df.shape[0]) for _ in range(num_lines)]
    return df.loc[random_idx, :].to_numpy()


def _mp_get_subset(func, lines_per_file=2):
    files = [(file, lines_per_file) for file in SOURCE_DIR.iterdir() if file.suffix == '.csv']
    pool = Pool(PROCESS_COUNT)
    data = pool.starmap(func, files)
    pool.close()
    pool.join()
    return data


def generate_rl_start_values():
    data = np.concatenate(_mp_get_subset(_get_random_start_values_from_file, lines_per_file=15))
    print(data.shape)
    pd.DataFrame(data).to_csv(RL_START_FILE, header=False, index=False)


def generate_non_touching_subset():
    data = np.concatenate(_mp_get_subset(_get_random_lines_from_file, lines_per_file=25))
    print(data.shape)
    bars_below = (0, 2, 4)
    bars_above = (1, 3)
    # set output values = input values
    data[:, SETUP_DIM + SENSOR_DIM:] = data[:, SETUP_DIM: SENSOR_DIM + SETUP_DIM]
    # random bar setups without touching the tape
    data[:, bars_below] = np.random.randint(0, 11, size=data[:, bars_below].shape)
    data[:, bars_above] = np.random.randint(18, 45, size=data[:, bars_above].shape)
    np.random.shuffle(data)
    num_output_files = 12
    split_idx = [i * len(data) // num_output_files for i in range(num_output_files)]
    for i in range(len(split_idx) - 1):
        pd.DataFrame(data[split_idx[i]: split_idx[i+1]]).to_csv(NON_TOUCHING_DIR / f'Setup0_{i}.csv',
                                                                header=False, index=False)
    pd.DataFrame(data[split_idx[-1]:]).to_csv(NON_TOUCHING_DIR / f'Setup0_{num_output_files - 1}.csv',
                                              header=False, index=False)


if __name__ == '__main__':
    generate_rl_start_values()
