from joblib import load
import torch

from process_control_model.ne_utils import load_pretrained_ff_model


def init_world_model(identifier, path, device):
    if identifier == 'nn':
        adapter = NNAdapter(path, device)
    else:
        adapter = RFAdapter(path, device)
    return adapter


class RFAdapter:
    def __init__(self, path, device):
        self.model = load(path)
        self.model.verbose = 0
        self.device = device

    def predict(self, data):
        resulting_tapes = self.model.predict(data.detach().cpu().numpy())
        return torch.from_numpy(resulting_tapes).float().to(self.device)


class NNAdapter:
    def __init__(self, path, device):
        self.device = device
        self._model = torch.load(path).to(device)

    def predict(self, data):
        with torch.no_grad():
            return self._model(data)
