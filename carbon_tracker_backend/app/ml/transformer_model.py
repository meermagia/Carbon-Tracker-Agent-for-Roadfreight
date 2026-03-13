"""
Transformer-based models for forecasting lane-level carbon emissions.

This module provides:
- dataset preparation utilities for time-series emission data
- a Transformer-based forecasting model
- training and inference helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


@dataclass
class TimeSeriesConfig:
    """
    Configuration for time-series emission forecasting.

    Attributes:
        input_length: Number of historical timesteps used as input.
        forecast_horizon: Number of future timesteps to predict.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Dimension of feedforward network.
        dropout: Dropout probability.
    """

    input_length: int = 24
    forecast_horizon: int = 6
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


def _build_sequences_for_lane(
    series: List[float],
    input_length: int,
    forecast_horizon: int,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Internal helper to build input/target sequences for a single lane.
    """
    inputs: List[List[float]] = []
    targets: List[List[float]] = []

    total_length = len(series)
    window = input_length + forecast_horizon
    if total_length < window:
        return inputs, targets

    for start in range(0, total_length - window + 1):
        end = start + window
        window_series = series[start:end]
        x = window_series[:input_length]
        y = window_series[input_length:]
        inputs.append(x)
        targets.append(y)

    return inputs, targets


def prepare_emission_timeseries_dataset(
    lane_emissions: Dict[str, List[float]],
    input_length: int,
    forecast_horizon: int,
    device: torch.device | None = None,
) -> Tuple[Dataset[Tuple[Tensor, Tensor]], Dict[str, int]]:
    """
    Prepare a PyTorch Dataset from lane-level emission time-series.

    Each lane is identified by a string key, e.g. "origin|destination" or any
    stable lane identifier. For each lane, rolling windows of size
    (input_length + forecast_horizon) are extracted.

    Args:
        lane_emissions: Mapping lane_id -> list of historical emissions (kg CO2e).
        input_length: Number of historical timesteps used as model input.
        forecast_horizon: Number of future timesteps to predict.
        device: Optional device to move tensors to.

    Returns:
        A tuple of:
        - Dataset yielding (input_seq, target_seq) tensors of shapes
          [input_length, 1] and [forecast_horizon, 1].
        - Mapping from lane_id to integer index (for future use).
    """
    all_inputs: List[List[float]] = []
    all_targets: List[List[float]] = []
    lane_to_idx: Dict[str, int] = {}

    for idx, (lane_id, series) in enumerate(lane_emissions.items()):
        lane_to_idx[lane_id] = idx
        x_seqs, y_seqs = _build_sequences_for_lane(series, input_length, forecast_horizon)
        all_inputs.extend(x_seqs)
        all_targets.extend(y_seqs)

    if not all_inputs:
        # Empty dataset
        x_tensor = torch.empty((0, input_length, 1), dtype=torch.float32)
        y_tensor = torch.empty((0, forecast_horizon, 1), dtype=torch.float32)
    else:
        x_tensor = torch.tensor(all_inputs, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(all_targets, dtype=torch.float32).unsqueeze(-1)

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    dataset: Dataset[Tuple[Tensor, Tensor]] = TensorDataset(x_tensor, y_tensor)
    return dataset, lane_to_idx


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Adapted for univariate time-series forecasting with Transformer encoders.
    """

    def __init__(self, d_model: int, max_len: int = 10_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10_000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class EmissionTransformer(nn.Module):
    """
    Transformer-based model for lane-level emission forecasting.

    The model maps a univariate input sequence of emissions
    to a forecast horizon of future emissions.
    """

    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(1, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output a sequence of length forecast_horizon of scalar emissions
        self.fc_out = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.forecast_horizon),
        )

    def forward(self, src: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            src: Input tensor of shape [batch_size, input_length, 1].

        Returns:
            Tensor of shape [batch_size, forecast_horizon, 1].
        """
        x = self.input_proj(src)  # [B, L, d_model]
        x = self.pos_encoder(x)
        encoded = self.encoder(x)  # [B, L, d_model]

        # Use representation of last timestep to forecast future horizon
        last_hidden = encoded[:, -1, :]  # [B, d_model]
        forecast = self.fc_out(last_hidden)  # [B, H]
        return forecast.unsqueeze(-1)  # [B, H, 1]


def train_emission_transformer(
    model: EmissionTransformer,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device | None = None,
) -> EmissionTransformer:
    """
    Train the emission Transformer model.

    Args:
        model: EmissionTransformer instance.
        dataloader: DataLoader yielding (input_seq, target_seq) tensors.
        num_epochs: Number of training epochs.
        optimizer: Optimizer instance (e.g., Adam).
        device: Optional torch.device to move model and data to.

    Returns:
        The trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(num_epochs):
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

    return model


def predict_emissions(
    model: EmissionTransformer,
    history: Tensor,
    device: torch.device | None = None,
) -> Tensor:
    """
    Run inference to predict future emissions for a lane.

    Args:
        model: Trained EmissionTransformer.
        history: Tensor of historical emissions with shape [batch_size, input_length, 1].
        device: Optional torch.device to perform inference on.

    Returns:
        Predicted emissions with shape [batch_size, forecast_horizon, 1].
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        history = history.to(device)
        forecast = model(history)
    return forecast

