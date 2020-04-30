from pathlib import Path
import torch
from torch.optim import lr_scheduler
from process_model.feedforward import Feedforward
from process_model.trainer import Trainer


class FFTrainer(Trainer):
    def __init__(self, sensor_dim=800, param_dim=5,
                 hidden_dimensions=(300, 300, 300), learning_rate=0.001, profile_only_input=False,
                 model_path=Path('model.pth')):
        # network
        if profile_only_input:
            model = Feedforward(sensor_dim, hidden_dimensions, sensor_dim)
        else:
            model = Feedforward(sensor_dim + param_dim,
                                hidden_dimensions, sensor_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=5,
            verbose=True,
            threshold=1e-5,
            min_lr=1e-6)
        super(FFTrainer, self).__init__(model, sensor_dim, model_path, optimizer, scheduler)
