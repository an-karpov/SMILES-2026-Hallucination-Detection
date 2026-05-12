from __future__ import annotations
import torch
import torch.nn.functional as F

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector (Safe & Robust).

    Combines last-token states from the upper layers, mean-pooled states over 
    the last tokens, and differences between the final state and earlier states.
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape

    num_real_tokens = int(torch.sum(attention_mask).item())
    if num_real_tokens == 0:
        last_pos = 0
    else:
        last_pos = max(0, num_real_tokens - 1)

    layer_features = []

    # --- Идея 1: Last-token states из финального и нескольких верхних слоев ---
    # Берём последние слои динамически (например, последние 4 слоя)
    for i in range(n_layers - 4, n_layers):
        token_feat = hidden_states[i, last_pos].detach().float()
        layer_features.append(token_feat)

    # --- Идея 2: Mean-pooled states по хвосту (последние 16/64 токенов) ---
    # Защита: если текст короткий, берем сколько есть реальных токенов
    tail_size = min(64, num_real_tokens if num_real_tokens > 0 else 1)
    start_tail = max(0, last_pos - tail_size + 1)
    
    # Срез хвоста последовательности для финального слоя
    final_layer_tail = hidden_states[-1, start_tail:last_pos + 1].detach().float()
    mean_tail_feat = torch.mean(final_layer_tail, dim=0)
    layer_features.append(mean_tail_feat)

    # --- Идея 3: Разности между финальным состоянием последнего токена и ранними слоями ---
    final_state = hidden_states[-1, last_pos].detach().float()
    # Берем слои из середины сети (семантическое ядро, ~50% и ~75% глубины)
    mid_layer_1 = int(n_layers * 0.5)
    mid_layer_2 = int(n_layers * 0.75)
    
    diff_1 = final_state - hidden_states[mid_layer_1, last_pos].detach().float()
    diff_2 = final_state - hidden_states[mid_layer_2, last_pos].detach().float()
    
    layer_features.append(diff_1)
    layer_features.append(diff_2)

    return torch.cat(layer_features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted advanced geometric / statistical features from hidden states.

    Features include: adjacent-layer drift norms, last-16-token mean for layers,
    normalized input length, per-layer L2 norms (final token, full-sequence mean, 
    max pool, tail variation), and cosine similarities.
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    num_real_tokens = int(torch.sum(attention_mask).item())
    if num_real_tokens == 0:
        last_pos = 0
    else:
        last_pos = max(0, num_real_tokens - 1)

    geo_features = []

    # --- 1. Мета-фичи: Нормализованная длина последовательности ---
    # Делим на условный максимум (например, 2048), чтобы фича была в адекватном масштабе
    norm_length = torch.tensor([num_real_tokens / 2048.0], dtype=torch.float32, device=hidden_states.device)
    geo_features.append(norm_length)

    # --- 2. Статистика по хвосту последовательности (последние 16 токенов) ---
    tail_size_16 = min(16, num_real_tokens if num_real_tokens > 0 else 1)
    start_tail_16 = max(0, last_pos - tail_size_16 + 1)
    
    # Для экономии размерности считаем среднее/std по слоям из второй половины сети
    start_layer = int(n_layers * 0.5)
    
    # Вырезаем блок: (выбранные_слои, токены_хвоста, hidden_dim)
    tail_slice = hidden_states[start_layer:, start_tail_16:last_pos + 1].detach().float()
    
    # Среднее по последним 16 токенам для каждого верхнего слоя
    layer_tail_mean = torch.mean(tail_slice, dim=1).mean(dim=1) # Сжимаем до (layers,)
    geo_features.append(layer_tail_mean)
    
    # Вариативность хвоста (tail standard deviation)
    layer_tail_std = torch.std(tail_slice, dim=1).mean(dim=1) # (layers,)
    geo_features.append(layer_tail_std)

    # --- 3. Комплексные L2 нормы по слоям (финальная треть модели) ---
    # Замеряем нормы для: final token, full-sequence mean, max pool
    upper_slice = hidden_states[start_layer:].detach().float() # (upper_layers, seq_len, hidden_dim)
    
    # L2 норма финального токена
    norm_final_token = torch.norm(upper_slice[:, last_pos, :], p=2, dim=1) # (upper_layers,)
    geo_features.append(norm_final_token)
    
    # L2 норма среднего по всей реальной последовательности
    real_sequence = upper_slice[:, :last_pos + 1, :]
    norm_seq_mean = torch.norm(torch.mean(real_sequence, dim=1), p=2, dim=1)
    geo_features.append(norm_seq_mean)
    
    # L2 норма Max Pool по последовательности
    max_pooled, _ = torch.max(real_sequence, dim=1)
    norm_max_pool = torch.norm(max_pooled, p=2, dim=1)
    geo_features.append(norm_max_pool)

    # --- 4. Дрейф между соседними слоями (Adjacent-layer drift norms & CosSim) ---
    # Считаем для векторов последнего токена на средних и верхних слоях
    h_prev = upper_slice[:-1, last_pos, :]
    h_curr = upper_slice[1:, last_pos, :]

    # Норма дрейфа (Adjacent-layer drift norm)
    drift_norms = torch.norm(h_curr - h_prev, p=2, dim=1) / (torch.norm(h_prev, p=2, dim=1) + 1e-6)
    geo_features.append(drift_norms)

    # Косинусное сходство между соседними слоями
    cos_sims = F.cosine_similarity(h_prev, h_curr, dim=1)
    geo_features.append(cos_sims)

    # --- 5. Куртозис (Острота распределения активаций последнего токена) ---
    mean_4 = torch.mean(upper_slice ** 4, dim=2).mean(dim=1)
    mean_2 = (torch.mean(upper_slice ** 2, dim=2) ** 2).mean(dim=1)
    kurtosis = mean_4 / (mean_2 + 1e-6)
    geo_features.append(kurtosis)

    return torch.cat(geo_features, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features