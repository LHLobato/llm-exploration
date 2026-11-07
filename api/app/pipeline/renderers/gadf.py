"""Geração em memória da imagem GADF 42×42 RGB (`esemble/CNN.py` e `esemble/multimodal.py`)."""

from __future__ import annotations

import base64
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Sequence

from app.pipeline.constants import GADF_IMAGE_SIZE, GADF_IN_CHANNELS


@dataclass(frozen=True)
class GadfArtifact:
    """Artefato visual 42×42 contendo a matriz GADF, tensor RGB `(3, 42, 42)` e PNG em Base64."""

    matrix_42x42: list[list[float]]  # Valores em [-1.0, 1.0]
    rgb_tensor_chw: list[list[list[float]]]  # Shape (3, 42, 42) em [0.0, 1.0]
    png_base64_data_uri: str


def compute_gadf_matrix(values_42: Sequence[float]) -> list[list[float]]:
    """
    Calcula a matriz Gramian Angular Difference Field (42×42) sobre os 42 atributos:
    1. Re-escala o vetor 1D para o intervalo `[-1.0, 1.0]` (`sample_range=(-1, 1)` do `pyts.image.GramianAngularField`).
    2. Com $\\phi_i = \\arccos(x_i)$, calcula $GADF_{i, j} = \\sin(\\phi_i - \\phi_j) = x_j \\sqrt{1 - x_i^2} - x_i \\sqrt{1 - x_j^2}$.
    Usa `pyts.image.GramianAngularField(image_size=42, method='difference')` se instalado, ou a identidade trigonométrica exata.
    """
    if len(values_42) != GADF_IMAGE_SIZE:
        raise ValueError(
            f"O GADF exige exatamente {GADF_IMAGE_SIZE} atributos, recebidos {len(values_42)}."
        )

    try:
        import numpy as np  # type: ignore
        from pyts.image import GramianAngularField  # type: ignore

        arr = np.asarray([values_42], dtype=np.float64)
        gadf = GramianAngularField(image_size=GADF_IMAGE_SIZE, method="difference")
        mat = gadf.fit_transform(arr)[0]
        return [[float(mat[i, j]) for j in range(GADF_IMAGE_SIZE)] for i in range(GADF_IMAGE_SIZE)]
    except Exception:
        pass

    v_min = min(values_42)
    v_max = max(values_42)
    span = v_max - v_min
    if span <= 1e-12:
        scaled = [0.0 for _ in values_42]
    else:
        scaled = [max(-1.0, min(1.0, 2.0 * (float(v) - v_min) / span - 1.0)) for v in values_42]

    sin_phi = [math.sqrt(max(0.0, 1.0 - x * x)) for x in scaled]
    matrix: list[list[float]] = []
    for i in range(GADF_IMAGE_SIZE):
        row: list[float] = []
        xi = scaled[i]
        si = sin_phi[i]
        for j in range(GADF_IMAGE_SIZE):
            xj = scaled[j]
            sj = sin_phi[j]
            # sin(arccos(xi) - arccos(xj)) = sin(phi_i)cos(phi_j) - cos(phi_i)sin(phi_j)
            val = si * xj - xi * sj
            row.append(max(-1.0, min(1.0, val)))
        matrix.append(row)
    return matrix


def _colormap_rgb_unit(value_minus1_to_1: float) -> tuple[float, float, float]:
    """
    Mapeia um valor $v \\in [-1, 1]$ para 3 canais RGB normalizados em `[0.0, 1.0]`
    (interpolação contínua Viridis-like utilizada para imagens GADF 3 canais).
    """
    t = max(0.0, min(1.0, (value_minus1_to_1 + 1.0) * 0.5))
    # Polinômio suave aproximando colormap Viridis (R, G, B em [0, 1])
    r = max(0.0, min(1.0, 0.267 + t * (-0.004 + t * (1.280 - 0.550 * t))))
    g = max(0.0, min(1.0, 0.005 + t * (1.015 - 0.115 * t)))
    b = max(0.0, min(1.0, 0.329 + t * (1.140 - t * (1.790 - 0.465 * t))))
    return (r, g, b)


def _encode_rgb_png_bytes(rgb_chw: list[list[list[float]]]) -> bytes:
    """Codifica um tensor `(3, H, W)` em `[0, 1]` como imagem PNG RGB válida em memória."""
    height = len(rgb_chw[0])
    width = len(rgb_chw[0][0])

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)  # Filtro PNG = 0 (None)
        for x in range(width):
            r_byte = int(round(max(0.0, min(1.0, rgb_chw[0][y][x])) * 255))
            g_byte = int(round(max(0.0, min(1.0, rgb_chw[1][y][x])) * 255))
            b_byte = int(round(max(0.0, min(1.0, rgb_chw[2][y][x])) * 255))
            raw_rows.extend((r_byte, g_byte, b_byte))

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", crc)

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB (color_type=2)
    compressed = zlib.compress(bytes(raw_rows), level=6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


def render_gadf_42x42(values_42: Sequence[float]) -> GadfArtifact:
    """
    Gera em memória a matriz GADF `42×42`, o tensor RGB `(3, 42, 42)` float32 em `[0.0, 1.0]`
    para entrada na `CustomCNN` (`esemble/CNN.py`) e a string `data:image/png;base64,...`.
    """
    matrix = compute_gadf_matrix(values_42)
    rgb_chw: list[list[list[float]]] = [
        [[0.0 for _ in range(GADF_IMAGE_SIZE)] for _ in range(GADF_IMAGE_SIZE)]
        for _ in range(GADF_IN_CHANNELS)
    ]

    for i in range(GADF_IMAGE_SIZE):
        for j in range(GADF_IMAGE_SIZE):
            r, g, b = _colormap_rgb_unit(matrix[i][j])
            rgb_chw[0][i][j] = r
            rgb_chw[1][i][j] = g
            rgb_chw[2][i][j] = b

    png_bytes = _encode_rgb_png_bytes(rgb_chw)
    b64_str = base64.b64encode(png_bytes).decode("ascii")
    return GadfArtifact(
        matrix_42x42=matrix,
        rgb_tensor_chw=rgb_chw,
        png_base64_data_uri=f"data:image/png;base64,{b64_str}",
    )
