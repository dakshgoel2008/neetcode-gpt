import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions

        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)


        # forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2   # y_pred

        # backward pass
        loss = np.mean((z2 - y_true)**2).reshape(-1)

        n = y_true.shape[0]
        dL_dz2 = (2 / n) * (z2 - y_true)

        # inner layer L - 1:
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2
        dL_da1 = W2.T @ dL_dz2

        # ReLU:
        dL_dz1 = dL_da1 * (z1 > 0)

        # inner Layer L - 2:
        dL_dW1 = np.outer(dL_dz1, x) + 0.0
        dL_db1 = dL_dz1

        return {
            'loss': round(loss.item(), 4),
            'dW1': np.round(dL_dW1, 4).tolist(),
            'db1': np.round(dL_db1, 4).tolist(),
            'dW2': np.round(dL_dW2, 4).tolist(),
            'db2': np.round(dL_db2, 4).tolist()
        }