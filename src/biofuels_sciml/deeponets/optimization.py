"""Optuna objective functions for DeepONet flash point modeling."""

import numpy as np
import torch

from collections.abc import Sequence
from optuna.trial import Trial
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from biofuels_sciml.deeponets.models import (
    ACTIVATION_FUNCTIONS,
    DeepONet,
    FNN,
)
from biofuels_sciml.deeponets.training import (
    compute_metrics,
    fit_transform_except_x1,
    train_deeponet_fold,
    train_fnn_fold,
    transform_except_x1,
)

DEFAULT_ACTIVATION_CHOICES = tuple(ACTIVATION_FUNCTIONS)


def _suggest_model_params(
    trial: Trial,
    hidden_width_range: tuple[int, int],
    n_layers_range: tuple[int, int],
    activation_choices: Sequence[str],
) -> dict[str, int | str]:
    """Suggest shared neural network architecture hyperparameters."""
    return {
        "hidden_width": trial.suggest_int(
            "hidden_width", hidden_width_range[0], hidden_width_range[1]
        ),
        "n_layers": trial.suggest_int("n_layers", n_layers_range[0], n_layers_range[1]),
        "activation": trial.suggest_categorical("activation", list(activation_choices)),
    }


def objective_fnn(
    trial: Trial,
    fnn_features: np.ndarray,
    fp_values: np.ndarray,
    device: torch.device,
    folds: int = 5,
    random_state: int = 42,
    epochs: int = 10000,
    patience: int = 1000,
    min_delta: float = 1e-4,
    hidden_width_range: tuple[int, int] = (4, 64),
    n_layers_range: tuple[int, int] = (2, 5),
    activation_choices: Sequence[str] = DEFAULT_ACTIVATION_CHOICES,
    pure_component_atol: float = 1e-6,
) -> float:
    """Evaluate one Optuna trial for FNN learning rate and architecture."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model_params = _suggest_model_params(
        trial, hidden_width_range, n_layers_range, activation_choices
    )
    pure_component_loss_weight = trial.suggest_float(
        "pure_component_loss_weight", 1, 1e4, log=True
    )
    l1_loss_weight = trial.suggest_float("l1_loss_weight", 1e-5, 1e-2, log=True)
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(fnn_features))
    fold_rmses: list[float] = []

    for train_idx, val_idx in splits:
        fnn_scaler = StandardScaler()

        fnn_train = fit_transform_except_x1(fnn_scaler, fnn_features[train_idx])
        fnn_val = transform_except_x1(fnn_scaler, fnn_features[val_idx])

        fnn_train_tensor = torch.tensor(fnn_train, dtype=torch.float32).to(device)
        fnn_val_tensor = torch.tensor(fnn_val, dtype=torch.float32).to(device)
        y_train_tensor = (
            torch.tensor(fp_values[train_idx], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        y_val_tensor = (
            torch.tensor(fp_values[val_idx], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        model = FNN(
            input_dim=fnn_train_tensor.shape[1],
            output_dim=1,
            random_state=random_state,
            **model_params,
        ).to(device)
        _, _, val_predictions = train_fnn_fold(
            model,
            fnn_train_tensor,
            y_train_tensor,
            fnn_val_tensor,
            y_val_tensor,
            lr=lr,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            pure_component_loss_weight=pure_component_loss_weight,
            pure_component_atol=pure_component_atol,
            l1_loss_weight=l1_loss_weight,
            random_state=random_state,
        )

        metrics = compute_metrics(val_predictions, y_val_tensor)
        fold_rmses.append(metrics["rmse (K)"])

    return float(np.mean(fold_rmses))


def objective_deeponet(
    trial: Trial,
    branch_features: np.ndarray,
    trunk_features: np.ndarray,
    fp_values: np.ndarray,
    device: torch.device,
    folds: int = 5,
    random_state: int = 42,
    epochs: int = 10000,
    patience: int = 1000,
    min_delta: float = 1e-4,
    hidden_width_range: tuple[int, int] = (2, 32),
    n_layers_range: tuple[int, int] = (2, 5),
    activation_choices: Sequence[str] = DEFAULT_ACTIVATION_CHOICES,
    pure_component_atol: float = 1e-6,
) -> float:
    """Evaluate one Optuna trial for DeepONet learning rate and architecture."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model_params = _suggest_model_params(
        trial, hidden_width_range, n_layers_range, activation_choices
    )
    pure_component_loss_weight = trial.suggest_float(
        "pure_component_loss_weight", 1, 1e4, log=True
    )
    l1_loss_weight = trial.suggest_float("l1_loss_weight", 1e-5, 1e-2, log=True)
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(branch_features))
    fold_rmses: list[float] = []

    for train_idx, val_idx in splits:
        branch_scaler = StandardScaler()
        trunk_scaler = StandardScaler()

        branch_train = fit_transform_except_x1(
            branch_scaler, branch_features[train_idx]
        )
        branch_val = transform_except_x1(branch_scaler, branch_features[val_idx])
        trunk_train = fit_transform_except_x1(trunk_scaler, trunk_features[train_idx])
        trunk_val = transform_except_x1(trunk_scaler, trunk_features[val_idx])

        branch_train_tensor = torch.tensor(branch_train, dtype=torch.float32).to(device)
        branch_val_tensor = torch.tensor(branch_val, dtype=torch.float32).to(device)
        trunk_train_tensor = torch.tensor(trunk_train, dtype=torch.float32).to(device)
        trunk_val_tensor = torch.tensor(trunk_val, dtype=torch.float32).to(device)
        y_train_tensor = (
            torch.tensor(fp_values[train_idx], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        y_val_tensor = (
            torch.tensor(fp_values[val_idx], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )

        model = DeepONet(
            branch_dim=branch_train_tensor.shape[1],
            trunk_dim=trunk_train_tensor.shape[1],
            output_dim=1,
            random_state=random_state,
            **model_params,
        ).to(device)
        _, _, val_predictions = train_deeponet_fold(
            model,
            branch_train_tensor,
            trunk_train_tensor,
            y_train_tensor,
            branch_val_tensor,
            trunk_val_tensor,
            y_val_tensor,
            lr=lr,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            pure_component_loss_weight=pure_component_loss_weight,
            pure_component_atol=pure_component_atol,
            l1_loss_weight=l1_loss_weight,
            random_state=random_state,
        )

        metrics = compute_metrics(val_predictions, y_val_tensor)
        fold_rmses.append(metrics["rmse (K)"])

    return float(np.mean(fold_rmses))
