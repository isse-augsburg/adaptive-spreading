import logging
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader

from preprocessing.datasets import DefaultDataset, ExtendedMetaDataset
from preprocessing.split_datasets import get_test_set_respecting_files,\
    split_data_respecting_files_including_meta, get_test_set_including_meta
import utils.paths as dirs


LOGGER = logging.getLogger(__name__)


def load_data_and_meta(device, batch_size, *args):
    train_data, test_data, val_data = split_data_respecting_files_including_meta(*args)

    def get_dataloader(data_x, data_y, data_filenames, data_index, shuffle=False, batchsize=1):
        return DataLoader(ExtendedMetaDataset(torch.from_numpy(data_x).float().to(device),
                                              torch.from_numpy(data_y).float().to(device),
                                              data_filenames, data_index),
                          batch_size=batchsize, shuffle=shuffle)
    train_loader = get_dataloader(*train_data, shuffle=True, batchsize=batch_size)
    validation_loader = get_dataloader(*val_data)
    test_loader = get_dataloader(*test_data)
    return train_loader, validation_loader, test_loader


def load_test_data_and_meta(device, datapath, sensor_dim, setup_dim, random_seed=42):
    test_x, test_y, filenames, indices = get_test_set_including_meta(datapath, sensor_dim,
                                                                     setup_dim, random_seed)
    return DataLoader(ExtendedMetaDataset(torch.from_numpy(test_x).float().to(device),
                                          torch.from_numpy(test_y).float().to(device),
                                          filenames, indices))


def load_data(device, split_function, batch_size, *args):
    train_x, test_x, val_x, train_y, test_y, val_y = split_function(*args)
    LOGGER.debug(f'size training set: {train_x.shape}, test set: {test_y.shape}')
    train_loader = DataLoader(DefaultDataset(torch.from_numpy(train_x).float().to(device),
                                             torch.from_numpy(train_y).float().to(device)),
                              batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(DefaultDataset(torch.from_numpy(val_x).float().to(device),
                                                  torch.from_numpy(val_y).float().to(device)))
    test_loader = DataLoader(DefaultDataset(torch.from_numpy(test_x).float().to(device),
                                            torch.from_numpy(test_y).float().to(device)))
    return train_loader, validation_loader, test_loader


def load_test_data(device, datapath, sensor_dim, setup_dim, random_seed=42):
    test_x, test_y = get_test_set_respecting_files(
        datapath, sensor_dim, setup_dim, random_seed)
    return DataLoader(DefaultDataset(torch.from_numpy(test_x).float().to(device),
                                     torch.from_numpy(test_y).float().to(device)))


if __name__ == '__main__':
    DEVICE = torch.device("cpu")
    SPLIT_ARGS = [Path(dirs.DATA), 800, 5, False, 1111]
    _, _, TEST_LOADER = load_data_and_meta(DEVICE, 20, *SPLIT_ARGS)
    _, _, FILENAME, INDEX = TEST_LOADER.__iter__().__next__()
    print(FILENAME[0], INDEX[0].item())
