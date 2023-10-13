import abc
import json
import logging
from typing import Dict, List, Literal, Optional, Tuple
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils import data
from .pinn_4dof_lr import PINNNet, RecurrentPINNDataset

from ...networks import loss, rnn
from ...tracker.base import BaseEventTracker
from .. import base, utils
from ..base import DynamicIdentificationModelConfig

logger = logging.getLogger(__name__)


class FrigatePINNModel_4dof_lrConfig(DynamicIdentificationModelConfig):
    inputNode: int
    hiddenNode: int
    outputNode: int
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs: int
    alpha: int


class FrigatePINNModel_4dof_lr(base.DynamicIdentificationModel, abc.ABC):
    CONFIG = FrigatePINNModel_4dof_lrConfig

    def __init__(self, config: FrigatePINNModel_4dof_lrConfig):
        super().__init__(config)
        self.device_name = config.device_name
        self.device = torch.device(self.device_name)
        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.epochs = config.epochs
        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.model = (
            PINNNet(
                inputNode=self.control_dim,
                hiddenNode=config.hiddenNode,
                outputNode=self.state_dim
            ).to(self.device)
        )
        self.optimizer = optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )
        self.loss_function = nn.MSELoss()
        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_std: Optional[NDArray[np.float64]] = None
        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None

    def train(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
            initial_seqs: Optional[List[NDArray[np.float64]]] = None,
            tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
        epoch_losses = []
        self.model.train()
        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]
        dataset = RecurrentPINNDataset(control_seqs, state_seqs, self.sequence_length)
        for i in range(self.epochs):
            data_loader = data.DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            labels, state_pred = None, None
            data_len = len(data_loader)
            for batch_idx, batch in enumerate(data_loader):
                self.model.zero_grad()
                signals = batch['x'].float().to(self.device)
                labels = batch['y'].float().to(self.device)
                state_pred = self.model.forward(signals)
                MSE_r = self.loss_function(
                    state_pred, labels
                )
                phi = signals[:, :, -1].unsqueeze(dim=2)
                # print(labels.shape, state_pred.shape, phi.shape, state_pred.shape)
                MSE_R = self.model.pinn_loss_4dof(labels[:, :, 0], labels[:, :, 1], labels[:, :, 2], labels[:, :, 3], phi[:,:,0], state_pred[:, :, 0], state_pred[:, :, 1], state_pred[:, :, 2], state_pred[:, :, 3])
                batch_loss = MSE_r + self.alpha * MSE_R
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()
            gt = labels.cpu().detach().numpy().reshape((-1, 1))
            pred = state_pred.cpu().detach().numpy().reshape((-1, 1))
            gt = utils.denormalize(gt, self.state_mean, self.state_std)
            pred = utils.denormalize(pred, self.state_mean, self.state_std)
            mse = mean_squared_error(pred, gt)
            rmse = np.sqrt(mse)
            mse_Xudot = mean_squared_error(-17400, self.model.Xudot)
            logger.info(
                f'Epoch {i + 1}/{self.epochs} - Epoch Loss: {total_loss} - Avg loss: {total_loss / data_len} - MSE: {mse} - RMSE {rmse}')
            epoch_losses.append([i, total_loss])

            print(f'K_delta:{self.model.Xudot},MSE of K_delta is :{mse_Xudot}')

        return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64))

    def simulate(
            self,
            initial_control: NDArray[np.float64],
            initial_state: NDArray[np.float64],
            control: NDArray[np.float64],
            x0: Optional[NDArray[np.float64]] = None,
            initial_x0: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        self.model.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        states = []
        with torch.no_grad():
            state_window = (
                torch.from_numpy(initial_control.flatten())
                    .float()
                    .to(self.device)
                    .unsqueeze(0)
            )
            control_window = (
                torch.from_numpy(initial_control)
                    .float()
                    .to(self.device)
                    .unsqueeze(0)
            )
            # control_in = torch.from_numpy(control).float().to(self.device)
            state = self.model.forward(control_window)
            y_np: NDArray[np.float64] = (
                state.cpu().detach().squeeze().unsqueeze(1).numpy().astype(np.float64)
            )
        y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
        return y_np

    def save(
            self,
            file_path: Tuple[str, ...],
            tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        if (
                self.state_mean is None
                or self.state_std is None
                or self.control_mean is None
                or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')
        torch.save(self.model.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
            json.dump(
                {
                    'state_mean': self.state_mean.tolist(),
                    'state_std': self.state_std.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self.model.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self.state_std = np.array(norm['state_std'], dtype=np.float64)
        self.control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self.control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'pth', 'json'

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
