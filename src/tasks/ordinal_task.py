"""
Tarea ordinal CORAL.

La clase CoralTask ya conecta la perdida y la prediccion con el Trainer.
Las cuatro funciones matematicas son TODO(alumno):

1. labels_to_levels
2. coral_loss
3. effective_number_weights
4. logits_to_ordinal_predictions
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import TrainingConfig
from src.tasks.base import ClassificationTask, LossFunction


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    TODO(alumno):
    Convierte clases enteras a umbrales binarios acumulativos.

    Ejemplo:
    Si num_classes = 5 y la etiqueta es 2, el vector debe ser [1, 1, 0, 0].

    Formas:
    - labels: (batch_size,)
    - salida: (batch_size, num_classes - 1) float32
    """

    raise NotImplementedError(
        "TODO(alumno): implementar labels_to_levels() en src/tasks/ordinal_task.py."
    )


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    TODO(alumno):
    BCE con logits sobre los K-1 umbrales ordinales.

    Pistas:
    - convertir labels con labels_to_levels,
    - usar binary_cross_entropy_with_logits sin reduccion,
    - sumar sobre los K-1 umbrales y promediar sobre el batch,
    - si class_weights no es None, ponderar cada muestra por el peso
      de su clase real.

    Formas:
    - logits: (batch_size, num_classes - 1)
    - labels: (batch_size,)
    - class_weights: (num_classes,) o None
    - salida: escalar
    """

    raise NotImplementedError(
        "TODO(alumno): implementar coral_loss() en src/tasks/ordinal_task.py."
    )


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    """
    TODO(alumno):
    Pesos por numero efectivo de muestras (Cui et al., CVPR 2019):

        w_c = (1 - beta) / (1 - beta ** n_c)

    Normalizar los pesos para que su media sea 1.
    Cuidado con clases sin muestras en el fold (n_c = 0).

    Formas:
    - labels: (N,)
    - salida: (num_classes,) float32
    """

    raise NotImplementedError(
        "TODO(alumno): implementar effective_number_weights() en src/tasks/ordinal_task.py."
    )


@torch.no_grad()
def logits_to_ordinal_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    TODO(alumno):
    Convierte logits CORAL en una clase entera.

    Pista:
    aplicar sigmoide, contar cuantos umbrales superan threshold
    y devolver ese conteo como y_hat.

    Formas:
    - logits: (batch_size, K-1)
    - salida: (batch_size,) int64
    """

    raise NotImplementedError(
        "TODO(alumno): implementar logits_to_ordinal_predictions() "
        "en src/tasks/ordinal_task.py."
    )


class CoralLoss:
    """Envoltorio invocable de coral_loss con K y pesos fijos por fold."""

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return coral_loss(logits, labels, self.num_classes, self.class_weights)


class CoralTask(ClassificationTask):
    """
    Clasificacion ordinal con K-1 umbrales acumulativos.

    Ya entregado: la conexion con el Trainer y el calculo de pesos por fold.
    TODO(alumno): las funciones de este modulo que lanzan NotImplementedError.

    Formas:
        logits: (B, K-1)
        labels: (B,) int64 en 0 .. K-1
    """

    name = "coral"

    def build_criterion(
        self,
        y_train: np.ndarray,
        config: TrainingConfig,
        device: torch.device,
    ) -> LossFunction:
        class_weights = None
        if config.use_class_weights:
            class_weights = effective_number_weights(
                y_train, self.num_classes, beta=config.beta
            ).to(device)
        return CoralLoss(self.num_classes, class_weights)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits_to_ordinal_predictions(logits)
