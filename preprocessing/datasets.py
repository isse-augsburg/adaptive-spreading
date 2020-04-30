from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(self, input_profiles, target_profiles):
        super(DefaultDataset, self).__init__()
        self.input_profiles = input_profiles
        self.target_profiles = target_profiles
        self.datasetsize = len(self.input_profiles)

    def __getitem__(self, index):
        return self.input_profiles[index], self.target_profiles[index]

    def __len__(self):
        return self.datasetsize


class ExtendedMetaDataset(Dataset):
    def __init__(self, input_profiles, target_profiles, filenames, lines):
        super(ExtendedMetaDataset, self).__init__()
        self.input_profiles = input_profiles
        self.target_profiles = target_profiles
        self.filenames = filenames
        self.lines = lines
        self.dataset_size = len(self.input_profiles)

    def __getitem__(self, index):
        return self.input_profiles[index], self.target_profiles[index], \
               self.filenames[index], self.lines[index]

    def __len__(self):
        return self.dataset_size
