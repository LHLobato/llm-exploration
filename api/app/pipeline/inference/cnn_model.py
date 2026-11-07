"""Arquitetura e carregamento da `ConvolutionalNeuralNetwork` de `esemble/CNN.py` (GADF 42×42 RGB)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.pipeline.constants import GADF_IMAGE_SIZE, GADF_IN_CHANNELS

# Arquitetura exata definida em esemble/multimodal.py (linhas 110-112) e esemble/CNN.py (linhas 68-78)
CNN_DEFAULT_ARCH: dict[str, Any] = {
    "name": "mid_3b_b",
    "blocks": [(32, 3, True), (64, 3, True), (128, 3, True)],
}
CNN_DEFAULT_FC_DIMS: list[int] = [512]
CNN_DEFAULT_DROPOUT: float = 0.5
CNN_NUM_CLASSES: int = 2


def build_cnn_model(
    arch: dict[str, Any] | None = None,
    fc_dims: list[int] | None = None,
    dropout: float = CNN_DEFAULT_DROPOUT,
    use_bn: bool = True,
) -> Any:
    """
    Constrói a `ConvolutionalNeuralNetwork` exatamente como `build_model` em `esemble/CNN.py`:
    - 3 blocos convolucionais `(32, 3, True) -> (64, 3, True) -> (128, 3, True)` com `BatchNorm2d`, `ReLU`, `MaxPool2d(2, 2)`.
    - Dimensão achatada para imagem `3×42×42`: `128 * 5 * 5 = 3200`.
    - Cabeça FC: `Linear(3200, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 2)`.
    """
    import torch
    import torch.nn as nn

    effective_arch = arch or CNN_DEFAULT_ARCH
    effective_fc_dims = fc_dims if fc_dims is not None else CNN_DEFAULT_FC_DIMS

    class ConvolutionalNeuralNetwork(nn.Module):
        def __init__(self, conv_layers: list[nn.Module], fc_layers: list[nn.Module]) -> None:
            super().__init__()
            self.conv_block = nn.Sequential(*conv_layers)
            self.fc_block = nn.Sequential(*fc_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv_block(x)
            x = x.flatten(start_dim=1)
            x = self.fc_block(x)
            return x

    conv_layers: list[nn.Module] = []
    ch_in = GADF_IN_CHANNELS
    for ch_out, k, pool in effective_arch["blocks"]:
        conv_layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=k, padding=k // 2))
        if use_bn:
            conv_layers.append(nn.BatchNorm2d(ch_out))
        conv_layers.append(nn.ReLU(inplace=True))
        if pool:
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        ch_in = ch_out

    conv_seq = nn.Sequential(*conv_layers)
    with torch.no_grad():
        dummy = torch.zeros(1, GADF_IN_CHANNELS, GADF_IMAGE_SIZE, GADF_IMAGE_SIZE)
        flat_size = int(conv_seq(dummy).flatten(start_dim=1).shape[1])

    fc_layers: list[nn.Module] = []
    in_dim = flat_size
    for h in effective_fc_dims:
        fc_layers.append(nn.Linear(in_dim, h))
        fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.Dropout(dropout))
        in_dim = h
    fc_layers.append(nn.Linear(in_dim, CNN_NUM_CLASSES))

    return ConvolutionalNeuralNetwork(list(conv_seq), fc_layers)


def load_cnn_checkpoint(cnn_path: Path, device: str = "cpu") -> Any:
    """
    Carrega os pesos `CustomCNN0-GADF-42_30.pth` na `ConvolutionalNeuralNetwork`
    conforme `esemble/multimodal.py` (linhas 110-127).
    """
    import torch

    torch_device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = build_cnn_model()
    checkpoint = torch.load(str(cnn_path), map_location=torch_device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()
    return model
