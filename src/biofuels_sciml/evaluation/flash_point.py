"""Flash point prediction helpers for repeated evaluation scenarios."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from attrs import define
from collections.abc import Callable, Mapping, Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from typing import Any

from biofuels_sciml.deeponets import DeepONet, FNN, transform_except_x1


@define
class BinarySystemEvaluationResult:
    """Container for predictions, plot data, and optional model artifacts."""

    predictions: pd.DataFrame
    plot_data: pd.DataFrame
    figure: Figure | None = None
    axes: Axes | None = None
    artifact: Mapping[str, Any] | None = None
    model: torch.nn.Module | None = None


def _as_prediction_array(
    predictions: np.ndarray | pd.Series | torch.Tensor,
) -> np.ndarray:
    """Convert prediction outputs to a one-dimensional NumPy array."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    return np.asarray(predictions).ravel()


def restore_deeponet_from_artifact(
    artifact: Mapping[str, Any],
    branch_dim: int,
    trunk_dim: int,
    device: torch.device,
    output_dim: int = 1,
    state_dict_key: str = "model_state_dict",
    model_params_key: str = "model_params",
    model_params: Mapping[str, Any] | None = None,
) -> DeepONet:
    """Restore a DeepONet model from a saved fold artifact."""
    resolved_model_params = dict(model_params or artifact.get(model_params_key, {}))
    model = DeepONet(
        branch_dim=branch_dim,
        trunk_dim=trunk_dim,
        output_dim=output_dim,
        **resolved_model_params,
    ).to(device)
    model.load_state_dict(artifact[state_dict_key])
    model.eval()

    return model


def restore_fnn_from_artifact(
    artifact: Mapping[str, Any],
    input_dim: int,
    device: torch.device,
    output_dim: int = 1,
    state_dict_key: str = "model_state_dict",
    model_params_key: str = "model_params",
    model_params: Mapping[str, Any] | None = None,
) -> FNN:
    """Restore an FNN model from a saved fold artifact."""
    resolved_model_params = dict(model_params or artifact.get(model_params_key, {}))
    model = FNN(
        input_dim=input_dim,
        output_dim=output_dim,
        **resolved_model_params,
    ).to(device)
    model.load_state_dict(artifact[state_dict_key])
    model.eval()

    return model


def predict_deeponet_flash_point(
    data: pd.DataFrame,
    model: DeepONet,
    branch_scaler: StandardScaler,
    branch_columns: Sequence[str],
    trunk_columns: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    """Predict flash point values with a DeepONet from DataFrame columns."""
    branch_features = data.loc[:, list(branch_columns)].to_numpy()
    trunk_features = data.loc[:, list(trunk_columns)].to_numpy()
    branch_scaled = transform_except_x1(branch_scaler, branch_features)

    branch_tensor = torch.tensor(branch_scaled, dtype=torch.float32).to(device)
    trunk_tensor = torch.tensor(trunk_features, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(branch_tensor, trunk_tensor)

    return _as_prediction_array(predictions)


def predict_fnn_flash_point(
    data: pd.DataFrame,
    model: FNN,
    scaler: StandardScaler,
    feature_columns: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    """Predict flash point values with an FNN from DataFrame columns."""
    features = data.loc[:, list(feature_columns)].to_numpy()
    scaled_features = transform_except_x1(scaler, features)

    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(feature_tensor)

    return _as_prediction_array(predictions)


def build_binary_system_plot_data(
    predictions: pd.DataFrame,
    reference_substance: str,
    prediction_column: str = "predicted_FP",
    x_column: str = "x_1",
    primary_substance_column: str = "substance_1",
    secondary_substance_column: str = "substance_2",
) -> pd.DataFrame:
    """Build plotting data with pure-component endpoints for binary systems."""
    pure_component_mask = predictions[secondary_substance_column].isna() | (
        predictions[secondary_substance_column] == ""
    )
    binary_system_predictions = predictions.loc[~pure_component_mask].copy()
    pure_component_predictions = predictions.loc[pure_component_mask].copy()

    reference_predictions = pure_component_predictions.loc[
        pure_component_predictions[primary_substance_column] == reference_substance
    ]
    if reference_predictions.empty:
        raise ValueError(
            f"Missing pure-component prediction for {reference_substance}."
        )

    reference_prediction = reference_predictions.iloc[0]
    plot_frames: list[pd.DataFrame] = []

    for secondary_substance, system_data in binary_system_predictions.groupby(
        secondary_substance_column
    ):
        secondary_predictions = pure_component_predictions.loc[
            pure_component_predictions[primary_substance_column] == secondary_substance
        ]
        if secondary_predictions.empty:
            raise ValueError(
                f"Missing pure-component prediction for {secondary_substance}."
            )

        secondary_prediction = secondary_predictions.iloc[0]
        system_plot_data = system_data.copy()
        system_plot_data["plot_x_1"] = system_plot_data[x_column]
        system_plot_data["system"] = f"{reference_substance} + {secondary_substance}"

        endpoints = pd.DataFrame(
            {
                "plot_x_1": [0.0, 1.0],
                prediction_column: [
                    secondary_prediction[prediction_column],
                    reference_prediction[prediction_column],
                ],
                "system": [
                    f"{reference_substance} + {secondary_substance}",
                    f"{reference_substance} + {secondary_substance}",
                ],
                secondary_substance_column: [secondary_substance, secondary_substance],
            }
        )
        plot_frames.append(
            pd.concat(
                [
                    endpoints.iloc[[0]],
                    system_plot_data[
                        [
                            "plot_x_1",
                            prediction_column,
                            "system",
                            secondary_substance_column,
                        ]
                    ],
                    endpoints.iloc[[1]],
                ],
                ignore_index=True,
            ).sort_values("plot_x_1")
        )

    if not plot_frames:
        return pd.DataFrame(
            columns=[
                "plot_x_1",
                prediction_column,
                "system",
                secondary_substance_column,
            ]
        )

    return pd.concat(plot_frames, ignore_index=True)


def plot_binary_system_predictions(
    plot_data: pd.DataFrame,
    reference_substance: str,
    prediction_column: str = "predicted_FP",
    model_name: str = "Model",
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot predicted flash points by composition for each binary system."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 6))
    else:
        figure = ax.figure

    for system_name, system_data in plot_data.groupby("system"):
        ax.plot(
            system_data["plot_x_1"],
            system_data[prediction_column],
            marker="o",
            label=system_name,
        )

    ax.set_xlabel("x_1")
    ax.set_ylabel("Predicted FP")
    ax.set_title(title or f"{model_name} Predicted FP by {reference_substance} System")
    ax.legend(title="System")
    ax.grid(alpha=0.3)
    figure.tight_layout()
    plt.show()

    return figure, ax


def evaluate_binary_system_predictions(
    data: pd.DataFrame,
    predict: Callable[[pd.DataFrame], np.ndarray | pd.Series | torch.Tensor],
    reference_substance: str,
    prediction_column: str = "predicted_FP",
    x_column: str = "x_1",
    primary_substance_column: str = "substance_1",
    secondary_substance_column: str = "substance_2",
    model_name: str = "Model",
    title: str | None = None,
    plot: bool = True,
    ax: Axes | None = None,
) -> BinarySystemEvaluationResult:
    """Predict and evaluate one binary-system flash point scenario."""
    predictions = data.copy()
    predictions[prediction_column] = _as_prediction_array(predict(data))
    plot_data = build_binary_system_plot_data(
        predictions,
        reference_substance=reference_substance,
        prediction_column=prediction_column,
        x_column=x_column,
        primary_substance_column=primary_substance_column,
        secondary_substance_column=secondary_substance_column,
    )

    figure = None
    axes = None
    if plot:
        figure, axes = plot_binary_system_predictions(
            plot_data,
            reference_substance=reference_substance,
            prediction_column=prediction_column,
            model_name=model_name,
            title=title,
            ax=ax,
        )

    return BinarySystemEvaluationResult(
        predictions=predictions,
        plot_data=plot_data,
        figure=figure,
        axes=axes,
    )


def evaluate_deeponet_binary_system_predictions(
    data: pd.DataFrame,
    fold_artifacts: Sequence[Mapping[str, Any]],
    branch_dim: int,
    trunk_dim: int,
    device: torch.device,
    branch_columns: Sequence[str],
    trunk_columns: Sequence[str],
    reference_substance: str,
    prediction_column: str = "predicted_FP",
    model_name: str = "DeepONet",
    title: str | None = None,
    metric_key: str = "rmse (K)",
    scaler_key: str = "branch_scaler",
    state_dict_key: str = "model_state_dict",
    model_params_key: str = "model_params",
    model_params: Mapping[str, Any] | None = None,
    output_dim: int = 1,
    plot: bool = True,
    ax: Axes | None = None,
) -> BinarySystemEvaluationResult:
    """Evaluate a binary-system scenario using the best DeepONet fold artifact."""
    if not fold_artifacts:
        raise ValueError("fold_artifacts must contain at least one artifact.")

    best_artifact = min(fold_artifacts, key=lambda artifact: artifact[metric_key])
    model = restore_deeponet_from_artifact(
        best_artifact,
        branch_dim=branch_dim,
        trunk_dim=trunk_dim,
        device=device,
        output_dim=output_dim,
        state_dict_key=state_dict_key,
        model_params_key=model_params_key,
        model_params=model_params,
    )
    branch_scaler = best_artifact[scaler_key]

    result = evaluate_binary_system_predictions(
        data=data,
        predict=lambda frame: predict_deeponet_flash_point(
            frame,
            model=model,
            branch_scaler=branch_scaler,
            branch_columns=branch_columns,
            trunk_columns=trunk_columns,
            device=device,
        ),
        reference_substance=reference_substance,
        prediction_column=prediction_column,
        model_name=model_name,
        title=title,
        plot=plot,
        ax=ax,
    )
    result.artifact = best_artifact
    result.model = model

    return result


def evaluate_fnn_binary_system_predictions(
    data: pd.DataFrame,
    fold_artifacts: Sequence[Mapping[str, Any]],
    input_dim: int,
    device: torch.device,
    feature_columns: Sequence[str],
    reference_substance: str,
    prediction_column: str = "predicted_FP",
    model_name: str = "FNN",
    title: str | None = None,
    metric_key: str = "rmse (K)",
    scaler_key: str = "fnn_scaler",
    state_dict_key: str = "model_state_dict",
    model_params_key: str = "model_params",
    model_params: Mapping[str, Any] | None = None,
    output_dim: int = 1,
    plot: bool = True,
    ax: Axes | None = None,
) -> BinarySystemEvaluationResult:
    """Evaluate a binary-system scenario using the best FNN fold artifact."""
    if not fold_artifacts:
        raise ValueError("fold_artifacts must contain at least one artifact.")

    best_artifact = min(fold_artifacts, key=lambda artifact: artifact[metric_key])
    model = restore_fnn_from_artifact(
        best_artifact,
        input_dim=input_dim,
        device=device,
        output_dim=output_dim,
        state_dict_key=state_dict_key,
        model_params_key=model_params_key,
        model_params=model_params,
    )
    scaler = best_artifact[scaler_key]

    result = evaluate_binary_system_predictions(
        data=data,
        predict=lambda frame: predict_fnn_flash_point(
            frame,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            device=device,
        ),
        reference_substance=reference_substance,
        prediction_column=prediction_column,
        model_name=model_name,
        title=title,
        plot=plot,
        ax=ax,
    )
    result.artifact = best_artifact
    result.model = model

    return result
