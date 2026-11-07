"""Arquitetura e carregamento do `MultiLayerPerceptron` de `esemble/MLP.py` (`[42, 256, 128, 64, 2]`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Configuração exata definida em esemble/multimodal.py (linhas 153-165) e esemble/MLP.py (linhas 49-74)
MLP_DEFAULT_ARCHITECTURE: list[int] = [42, 256, 128, 64, 2]
MLP_DEFAULT_DROPOUT: float = 0.0


def build_mlp_model(
    mlp_shape: list[int] | None = None,
    dropout_hidden_layers: float = MLP_DEFAULT_DROPOUT,
) -> Any:
    """
    Constrói o `MultiLayerPerceptron` exatamente como em `esemble/MLP.py` (linhas 49-74):
    - Cada camada oculta: `Linear(in_f, out_f) -> ReLU() -> BatchNorm1d(out_f) -> Dropout(dropout_rate)`
    - Camada de saída: `Linear(64, 2)`
    """
    import torch
    import torch.nn as nn

    shape = mlp_shape or MLP_DEFAULT_ARCHITECTURE

    def make_layer(input_feat: int, output_feat: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_feat, output_feat),
            nn.ReLU(),
            nn.BatchNorm1d(output_feat),
            nn.Dropout(dropout_hidden_layers),
        )

    class MultiLayerPerceptron(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for i in range(len(shape) - 2):
                layers.append(make_layer(shape[i], shape[i + 1]))
            layers.append(nn.Linear(shape[-2], shape[-1]))
            self.layers = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    return MultiLayerPerceptron()


def load_mlp_checkpoint(tab_path: Path, device: str = "cpu") -> Any:
    """
    Carrega os pesos `best_mlp.pt` no `MultiLayerPerceptron` conforme `esemble/multimodal.py`
    (linhas 153-177).
    """
    import torch

    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = build_mlp_model().to(torch_device)
    checkpoint = torch.load(str(tab_path), map_location=torch_device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model
