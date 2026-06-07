"""Evaluation helpers for model scenarios."""

from biofuels_sciml.evaluation.flash_point import (
    BinarySystemEvaluationResult,
    build_binary_system_plot_data,
    evaluate_binary_system_predictions,
    evaluate_deeponet_binary_system_predictions,
    evaluate_fnn_binary_system_predictions,
    plot_binary_system_predictions,
    predict_deeponet_flash_point,
    predict_fnn_flash_point,
    restore_deeponet_from_artifact,
    restore_fnn_from_artifact,
)

__all__ = [
    "BinarySystemEvaluationResult",
    "build_binary_system_plot_data",
    "evaluate_binary_system_predictions",
    "evaluate_deeponet_binary_system_predictions",
    "evaluate_fnn_binary_system_predictions",
    "plot_binary_system_predictions",
    "predict_deeponet_flash_point",
    "predict_fnn_flash_point",
    "restore_deeponet_from_artifact",
    "restore_fnn_from_artifact",
]
