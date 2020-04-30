import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import utils.paths as dirs

SOURCE_PATH = Path(dirs.DATA)
TARGET_PATH = Path(dirs.DATA, 'shady')
SENSOR_DIM = 800
PARAMS_DIM = 5


def _create_target_dir():
    if not TARGET_PATH.exists():
        TARGET_PATH.mkdir()


def plot_line(filename, line):
    if not (SOURCE_PATH / filename).exists():
        print("file doesn't exist")
        return
    data = pd.read_csv(SOURCE_PATH / filename, header=None).to_numpy()
    print('file size:', len(data))
    plt.subplot(2, 1, 1)
    plt.plot(data[line, PARAMS_DIM:SENSOR_DIM+PARAMS_DIM])
    plt.subplot(2, 1, 2)
    plt.plot(data[line, SENSOR_DIM+PARAMS_DIM:])
    plt.show()


def remove_lines(filename, lines):
    if not (SOURCE_PATH / filename).exists():
        print("file doesn't exist")
        return
    data = pd.read_csv(SOURCE_PATH / filename, header=None)
    len_before = len(data)
    dropped_lines = data.loc[lines, :]
    print(len(dropped_lines))
    idx = [i for i in range(len(data)) if i not in lines]
    data = data.loc[idx, :]
    print(f'dropped {len_before - len(data)} lines')
    data.to_csv(SOURCE_PATH / filename, index=False, header=False)
    _create_target_dir()
    if (TARGET_PATH / filename).exists():
        print('Appended lines to', TARGET_PATH / filename)
        part_two = pd.read_csv(TARGET_PATH / filename, header=None).append(dropped_lines)
        part_two.to_csv(TARGET_PATH / filename, header=False, index=False)
    else:
        print('Moved lines to', TARGET_PATH / filename)
        dropped_lines.to_csv(TARGET_PATH / filename, header=False, index=False)


def move_file(filename):
    if not (SOURCE_PATH / filename).exists():
        print("file doesn't exist")
        return
    _create_target_dir()
    if (TARGET_PATH / filename).exists():
        print('Append file to', TARGET_PATH / filename)
        part_one = pd.read_csv(SOURCE_PATH / filename, header=None)
        part_two = pd.read_csv(TARGET_PATH / filename, header=None).append(part_one)
        part_two.to_csv(TARGET_PATH / filename, header=False, index=False)
        (SOURCE_PATH / filename).unlink()
    else:
        print('Moved file to', TARGET_PATH / filename)
        shutil.move(SOURCE_PATH / filename, TARGET_PATH / filename)
