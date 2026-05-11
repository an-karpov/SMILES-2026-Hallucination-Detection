from __future__ import annotations
import torch
import torch.nn.functional as F

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    selected_layer_indices = [19, 20, 21, 22, 23, 24]
    
    real_positions = attention_mask.nonzero(as_tuple=False)
    if len(real_positions) == 0:
        last_pos = 0
    else:
        last_pos = int(real_positions[-1].item())

    layer_features = []
    for idx in selected_layer_indices:
        layer = hidden_states[idx]
        token_feat = layer[last_pos]
        layer_features.append(token_feat)

    feature = torch.cat(layer_features, dim=0)

    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # Снова ориентируемся на последний реальный токен
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item()) if len(real_positions) > 0 else -1

    relevant_h = [hidden_states[i][last_pos] for i in range(10, 25)]

    geo_features = []
    
    for i in range(1, len(relevant_h)):
        h_prev = relevant_h[i-1]
        h_curr = relevant_h[i]
        
        # 1. Относительное изменение (Residual Norm)
        diff = h_curr - h_prev
        rel_change = torch.norm(diff) / (torch.norm(h_prev) + 1e-6)
        geo_features.append(rel_change.unsqueeze(0))
        
        # 2. Куртoзис (насколько "острые" активации)
        # Упрощенно: среднее от 4-й степени / (квадрат среднего квадратов)
        kurt = torch.mean(h_curr**4) / (torch.mean(h_curr**2)**2 + 1e-6)
        geo_features.append(kurt.unsqueeze(0))

    # 3. Глобальная статистика траектории
    all_h_tensor = torch.stack(relevant_h) # (layers, 896)
    trajectory_std = torch.std(all_h_tensor, dim=0).mean() # Насколько слои "шумят"
    geo_features.append(trajectory_std.unsqueeze(0))

    for i in range(12, 25):
        h = hidden_states[i][last_pos]
        norm = torch.norm(h, p=2).unsqueeze(0)
        geo_features.append(norm)

    for i in range(18, 24):
        h_current = hidden_states[i][last_pos]
        h_next = hidden_states[i+1][last_pos]
        sim = F.cosine_similarity(h_current.unsqueeze(0), h_next.unsqueeze(0)).view(1)
        geo_features.append(sim)

    return torch.cat(geo_features, dim=0)

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features