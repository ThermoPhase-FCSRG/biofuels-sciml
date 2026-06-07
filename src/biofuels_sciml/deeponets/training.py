"""Training and evaluation helpers for DeepONet notebooks."""

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
)

from biofuels_sciml.deeponets.models import DeepONet, FNN, set_torch_seed


def _scaled_column_indices(features: np.ndarray, x1_column_index: int) -> list[int]:
    """Return the feature columns that should be scaled."""
    return [
        column_idx
        for column_idx in range(features.shape[1])
        if column_idx != x1_column_index
    ]


def fit_transform_except_x1(
    scaler: StandardScaler, features: np.ndarray, x1_column_index: int = 0
) -> np.ndarray:
    """Fit a scaler to non-x_1 columns and return x_1 unchanged."""
    transformed_features = features.copy()
    scaled_columns = _scaled_column_indices(features, x1_column_index)

    if not scaled_columns:
        return transformed_features

    transformed_features[:, scaled_columns] = scaler.fit_transform(
        features[:, scaled_columns]
    )
    return transformed_features


def transform_except_x1(
    scaler: StandardScaler, features: np.ndarray, x1_column_index: int = 0
) -> np.ndarray:
    """Apply a fitted scaler to non-x_1 columns and return x_1 unchanged."""
    transformed_features = features.copy()
    scaled_columns = _scaled_column_indices(features, x1_column_index)

    if not scaled_columns:
        return transformed_features

    transformed_features[:, scaled_columns] = scaler.transform(
        features[:, scaled_columns]
    )
    return transformed_features


def compute_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Compute regression metrics for model predictions."""
    device = predictions.device
    metric_objects = {
        "mae (K)": MeanAbsoluteError().to(device),
        "rmse (K)": MeanSquaredError(squared=False).to(device),
        "mape": MeanAbsolutePercentageError().to(device),
        "r2": R2Score().to(device),
    }

    return {
        name: metric(predictions, targets).detach().cpu().item()
        for name, metric in metric_objects.items()
    }


def pure_component_weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    x1_values: torch.Tensor,
    model: nn.Module,
    pure_component_loss_weight: float = 1.0,
    pure_component_atol: float = 1e-6,
    l1_loss_weight: float = 1e-5,
) -> torch.Tensor:
    """
    Compute MSE plus an extra MSE term for pure-component points,
    with L1 regularization on model parameters.
    """
    base_loss = nn.functional.mse_loss(predictions, targets)
    l1_loss = sum(param.abs().sum() for param in model.parameters())

    flattened_x1 = x1_values.reshape(-1)
    pure_component_mask = torch.isclose(
        flattened_x1,
        torch.zeros_like(flattened_x1),
        atol=pure_component_atol,
    ) | torch.isclose(
        flattened_x1,
        torch.ones_like(flattened_x1),
        atol=pure_component_atol,
    )

    loss = base_loss + l1_loss_weight * l1_loss

    if pure_component_mask.any():
        squared_errors = (predictions.reshape(-1) - targets.reshape(-1)) ** 2
        pure_component_loss = squared_errors[pure_component_mask].mean()
        loss = loss + pure_component_loss_weight * pure_component_loss

    return loss


def plot_fold_results(
    model_name: str,
    fold_idx: int,
    train_losses: list[float],
    val_losses: list[float],
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Plot loss evolution and experimental versus predicted FP values for a fold."""
    y_true = targets.detach().cpu().numpy().ravel()
    y_pred = predictions.detach().cpu().numpy().ravel()

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, label="Train loss")
    axes[0].plot(val_losses, label="Validation loss")
    axes[0].set_title(f"{model_name} - Fold {fold_idx} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    axes[1].scatter(y_true, y_pred, alpha=0.8)
    axes[1].plot([min_value, max_value], [min_value, max_value], "k--")
    axes[1].set_title(f"{model_name} - Fold {fold_idx}")
    axes[1].set_xlabel("Experimental FP")
    axes[1].set_ylabel("Predicted FP")

    plt.tight_layout()
    plt.show()


def train_fnn_fold(
    model: FNN,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 10000,
    patience: int = 1000,
    min_delta: float = 1e-4,
    pure_component_loss_weight: float = 1.0,
    pure_component_atol: float = 1e-6,
    l1_loss_weight: float = 1e-5,
    random_state: int | None = 42,
) -> tuple[list[float], list[float], torch.Tensor]:
    """Train the FNN for one fold and return loss histories and predictions."""
    set_torch_seed(random_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x1_train = x_train[:, 0]
    x1_val = x_val[:, 0]

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        train_predictions = model(x_train)
        train_loss = pure_component_weighted_mse_loss(
            train_predictions,
            y_train,
            x1_train,
            model,
            pure_component_loss_weight=pure_component_loss_weight,
            pure_component_atol=pure_component_atol,
            l1_loss_weight=l1_loss_weight,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val)
            val_loss = pure_component_weighted_mse_loss(
                val_predictions,
                y_val,
                x1_val,
                model,
                pure_component_loss_weight=pure_component_loss_weight,
                pure_component_atol=pure_component_atol,
                l1_loss_weight=l1_loss_weight,
            )

        train_losses.append(train_loss.detach().cpu().item())
        current_val_loss = val_loss.detach().cpu().item()
        val_losses.append(current_val_loss)

        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        val_predictions = model(x_val)

    return train_losses, val_losses, val_predictions


def train_deeponet_fold(
    model: DeepONet,
    branch_train: torch.Tensor,
    trunk_train: torch.Tensor,
    y_train: torch.Tensor,
    branch_val: torch.Tensor,
    trunk_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 10000,
    patience: int = 1000,
    min_delta: float = 1e-4,
    pure_component_loss_weight: float = 1.0,
    pure_component_atol: float = 1e-6,
    l1_loss_weight: float = 1e-5,
    random_state: int | None = 42,
) -> tuple[list[float], list[float], torch.Tensor]:
    """Train the DeepONet for one fold and return loss histories and predictions."""
    set_torch_seed(random_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x1_train = trunk_train[:, 0]
    x1_val = trunk_val[:, 0]

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        train_predictions = model(branch_train, trunk_train)
        train_loss = pure_component_weighted_mse_loss(
            train_predictions,
            y_train,
            x1_train,
            model,
            pure_component_loss_weight=pure_component_loss_weight,
            pure_component_atol=pure_component_atol,
            l1_loss_weight=l1_loss_weight,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(branch_val, trunk_val)
            val_loss = pure_component_weighted_mse_loss(
                val_predictions,
                y_val,
                x1_val,
                model,
                pure_component_loss_weight=pure_component_loss_weight,
                pure_component_atol=pure_component_atol,
                l1_loss_weight=l1_loss_weight,
            )

        train_losses.append(train_loss.detach().cpu().item())
        current_val_loss = val_loss.detach().cpu().item()
        val_losses.append(current_val_loss)

        if current_val_loss < best_val_loss - min_delta:
            best_val_loss = current_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        val_predictions = model(branch_val, trunk_val)

    return train_losses, val_losses, val_predictions
