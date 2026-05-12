"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; uses variance-based PCA reduction (98%), 
    standard scaling, and a smooth SiLU MLP architecture optimized against overfitting.
    """

    def __init__(self, n_components: float = 0.98) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # Строится лениво в fit()
        self._scaler = StandardScaler()
        # Используем float (0.98) вместо фиксированного числа. 
        # Это заставляет PCA автоматически сохранять 98% всей важной информации,
        # полностью исключая падения кода из-за нехватки сэмплов или признаков.
        self._n_components = n_components
        self._pca: PCA | None = PCA(n_components=n_components, svd_solver='full', random_state=42)
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with AdamW + ``BCEWithLogitsLoss``.
        """
        X_scaled = self._scaler.fit_transform(X)
        X_reduced = self._pca.fit_transform(X_scaled)

        self._build_network(X_reduced.shape[1])

        X_t = torch.from_numpy(X_reduced).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.05)

        self.train()
        for epoch in range(120):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1."""
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)

        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)

        X_t = torch.from_numpy(X_scaled).float()

        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
