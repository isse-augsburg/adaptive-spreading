import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.tape_width_evaluation import evaluate_tape_width
import utils.paths as dirs


class Trainer:
    def __init__(self, model, output_dim, model_path, optimizer, lr_scheduler=None):
        self.output_dim = output_dim
        self.model = model
        self.model_path = model_path
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.patience = 10
        self.num_stagnating_epochs = 0
        self.lowest_error = 10000
        # tensorboard
        self.writer = SummaryWriter(Path(dirs.NN_TENSORBOARD, self.model_path.name))
        self.logger = logging.getLogger(__name__)

    def save_model_weights(self):
        if not self.model_path.parents[0].exists():
            self.model_path.parents[0].mkdir()
        torch.save(self.model.state_dict(), self.model_path)
        if not dirs.NN_MODELS.exists():
            dirs.NN_MODELS.mkdir()
        torch.save(self.model, dirs.NN_MODELS / self.model_path.name)

    def early_stopper(self, val_loss):
        """
        Returns flag to stop training when validation loss did not decrease over the last
        epochs (patience). Saves the network parameters after the best epoch (lowest
        validation loss)
        :param val_loss: Loss on validation set during current epoch
        :return: True to stop, False to continue training
        """
        if val_loss < self.lowest_error:
            self.lowest_error = val_loss
            self.save_model_weights()
            self.logger.info('save states')
            self.num_stagnating_epochs = 0
        else:
            self.num_stagnating_epochs += 1
        if self.num_stagnating_epochs >= self.patience:
            return True
        return False

    def train_network(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        loss_history, val_loss_history = [], []
        for i in range(2000):
            losses = 0
            self.model.train()
            for sample, target, _, _ in train_loader:
                self.optimizer.zero_grad()
                output = self.model(sample)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                losses += loss.item()
            losses /= len(train_loader)
            self.writer.add_scalar("training_loss", losses, i)
            self.logger.info(f'Training loss after epoch {i + 1}: {losses}')
            loss_history.append(losses)
            val_loss, _, _, _, _ = self.test_network(val_loader, True, i)
            val_loss_history.append(val_loss)
            self.writer.add_scalar("validation_loss", val_loss, i)

            if self.early_stopper(val_loss):
                break
            if self.scheduler:
                self.scheduler.step(val_loss)

        return loss_history, val_loss_history

    def test_network(self, loader, validating=False, epoch=0):
        """
        Evaluates network using the given data.
        :param loader: Dataloader containing validation or testing data
        :param validating: Indicates if the function is called for validating or testing.
        :param epoch: epoch counter when validating
        :return: Accumulated loss on the given data, target and prediction tensor
        """
        if not validating:
            # for testing: restore parameters with best performance on
            # validation set
            self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        loader_size = len(loader)

        all_preds = torch.empty(
            (loader_size, loader.batch_size, self.output_dim))
        all_targets = torch.empty_like(all_preds)
        i = 0
        # evaluate network without gradient computation
        with torch.no_grad():
            criterion = nn.MSELoss()
            for sample, target, _, _ in loader:
                pred = self.model(sample)
                all_preds[i] = pred
                all_targets[i] = target
                i += 1

            loss = criterion(all_preds, all_targets).item()
            tape_width_dev = evaluate_tape_width(all_targets, all_preds, plot_stats=False)
            self.logger.info(f'tape width dev: {tape_width_dev:.4f}')
            _, _, all_filenames, all_indices = loader.dataset[:]
            if validating:
                self.logger.info(f'Val Loss: {loss}')
                self.writer.add_scalar("avg_width_dev", tape_width_dev, epoch)
            else:
                self.logger.info(f'Test Loss: {loss}')

            return loss, all_targets, all_preds, all_filenames, all_indices
