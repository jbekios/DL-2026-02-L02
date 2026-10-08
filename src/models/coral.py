"""
Experimento 2: Conv1D + MLP con cabeza ordinal CORAL.

Este archivo contiene esqueletos que el alumno debe completar.
El tronco Conv1D + MLP (Conv1DBackbone) ya esta implementado y es el mismo
del experimento Softmax: solo cambia la cabeza de salida.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import Conv1DBackbone


class CoralLayer(nn.Module):
    """
    TODO(alumno):
    Capa de salida CORAL.

    Debe producir K-1 logits acumulativos a partir de un vector de
    caracteristicas de tamano input_size.

    Pistas:
    - un peso lineal compartido hacia un unico puntaje latente s = w^T h
      (sin sesgo),
    - K-1 sesgos ordenados b_0 >= b_1 >= ... >= b_{K-2},
    - las diferencias entre sesgos pueden construirse con softplus y cumsum
      para forzar el orden,
    - logits z_k = s + b_k.

    Formas esperadas:
    - x: (batch_size, input_size)
    - salida: (batch_size, num_classes - 1)
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        # TODO(alumno): definir la proyeccion compartida y los parametros
        # de los sesgos ordenados.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TODO(alumno): implementar CoralLayer.forward() en src/models/coral.py."
        )


class Conv1DCoralNet(nn.Module):
    """
    TODO(alumno):
    Conv1D + MLP con cabeza CORAL.

    Arquitectura sugerida:
        (B, 15) -> Conv1DBackbone (ya entregado) -> (B, hidden_dim)
                -> CoralLayer(hidden_dim, K)     -> (B, K-1)

    El forward debe devolver logits de forma (batch_size, K-1).
    No aplicar sigmoide en el forward: la perdida usa BCE con logits.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 3,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Conv1DBackbone(
            num_features=num_features,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # TODO(alumno): crear la cabeza CORAL sobre self.backbone.output_dim.

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TODO(alumno): implementar Conv1DCoralNet.forward() en src/models/coral.py."
        )
