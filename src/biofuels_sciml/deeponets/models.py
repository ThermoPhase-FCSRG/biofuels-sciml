"""Neural network architectures used in DeepONet flash point modeling."""

import torch
import torch.nn as nn

ACTIVATION_FUNCTIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}

ActivationLike = str | type[nn.Module]


def get_activation_function(activation: ActivationLike) -> type[nn.Module]:
    """Return an activation module class from a name or module class."""
    if isinstance(activation, str):
        try:
            return ACTIVATION_FUNCTIONS[activation]
        except KeyError as exc:
            valid_names = ", ".join(ACTIVATION_FUNCTIONS)
            raise ValueError(
                f"Unknown activation '{activation}'. Valid options: {valid_names}."
            ) from exc

    return activation


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class FNN(nn.Module):
    """Standard feedforward neural network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int = 14,
        n_layers: int = 3,
        activation: ActivationLike = nn.ReLU,
    ) -> None:
        """Initialize the feedforward network layers."""
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        activation_function = get_activation_function(activation)

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_width))
            layers.append(activation_function())
            current_dim = hidden_width

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input tensor through the network."""
        return self.network(x)


class DeepONet(nn.Module):
    """Standard Deep Operator Network with branch and trunk subnetworks."""

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        output_dim: int = 1,
        hidden_width: int = 14,
        n_layers: int = 3,
        activation: ActivationLike = nn.ReLU,
    ) -> None:
        """Initialize branch, trunk, and bias parameters."""
        super().__init__()
        self.branch = FNN(branch_dim, output_dim, hidden_width, n_layers, activation)
        self.trunk = FNN(trunk_dim, output_dim, hidden_width, n_layers, activation)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self, branch_input: torch.Tensor, trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """Combine branch and trunk outputs into DeepONet predictions."""
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)

        return torch.sum(branch_output * trunk_output, dim=-1, keepdim=True) + self.bias
