"""Optuna objective functions for DeepONet flash point modeling."""

import numpy as np
import torch

from collections.abc import Sequence
from optuna.exceptions import TrialPruned
from optuna.trial import Trial
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from biofuels_sciml.deeponets.models import (
    ACTIVATION_FUNCTIONS,
    DeepONet,
    FNN,
    count_trainable_parameters,
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


def _suggest_loss_weight(
    trial: Trial, pure_component_loss_weight_range: tuple[float, float]
) -> float:
    """Suggest the pure-component loss weight."""
    return trial.suggest_float(
        "pure_component_loss_weight",
        pure_component_loss_weight_range[0],
        pure_component_loss_weight_range[1],
    )


def _prune_if_too_many_parameters(
    trial: Trial,
    model: torch.nn.Module,
    min_train_samples: int,
) -> None:
    """Prune a trial when trainable parameters exceed available train samples."""
    n_trainable_params = count_trainable_parameters(model)
    trial.set_user_attr("n_trainable_params", n_trainable_params)
    trial.set_user_attr("min_train_samples", min_train_samples)

    if n_trainable_params > min_train_samples:
        raise TrialPruned(
            "Trial pruned because the model has "
            f"{n_trainable_params} trainable parameters and only "
            f"{min_train_samples} training samples in the smallest fold."
        )


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
    hidden_width_range: tuple[int, int] = (2, 32),
    n_layers_range: tuple[int, int] = (2, 5),
    activation_choices: Sequence[str] = DEFAULT_ACTIVATION_CHOICES,
    pure_component_loss_weight_range: tuple[float, float] = (0.01, 100),
    pure_component_atol: float = 1e-6,
) -> float:
    """Evaluate one Optuna trial for FNN learning rate and architecture."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model_params = _suggest_model_params(
        trial, hidden_width_range, n_layers_range, activation_choices
    )
    pure_component_loss_weight = _suggest_loss_weight(
        trial, pure_component_loss_weight_range
    )
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(fnn_features))
    min_train_samples = min(len(train_idx) for train_idx, _ in splits)
    reference_model = FNN(
        input_dim=fnn_features.shape[1],
        output_dim=1,
        **model_params,
    )
    _prune_if_too_many_parameters(trial, reference_model, min_train_samples)

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
    pure_component_loss_weight_range: tuple[float, float] = (0.01, 100),
    pure_component_atol: float = 1e-6,
) -> float:
    """Evaluate one Optuna trial for DeepONet learning rate and architecture."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model_params = _suggest_model_params(
        trial, hidden_width_range, n_layers_range, activation_choices
    )
    pure_component_loss_weight = _suggest_loss_weight(
        trial, pure_component_loss_weight_range
    )
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(branch_features))
    min_train_samples = min(len(train_idx) for train_idx, _ in splits)
    reference_model = DeepONet(
        branch_dim=branch_features.shape[1],
        trunk_dim=trunk_features.shape[1],
        output_dim=1,
        **model_params,
    )
    _prune_if_too_many_parameters(trial, reference_model, min_train_samples)

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
        )

        metrics = compute_metrics(val_predictions, y_val_tensor)
        fold_rmses.append(metrics["rmse (K)"])

    return float(np.mean(fold_rmses))
