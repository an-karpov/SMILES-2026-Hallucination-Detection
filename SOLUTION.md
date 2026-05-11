
# SOLUTION.md — Hallucination Detection in Qwen2.5-0.5B

**Final test AUROC: 68.56 %**

**`predictions.csv` (publicly hosted):** <https://drive.google.com/file/d/1e7SPhGHYFnX7pg1hdVgMAP9G5Ow7wmTu/view?usp=sharing>

**`results.json`:** <https://drive.google.com/file/d/1EfmHTOED9cac0Q3LNg-DS-AozJwv3F43/view?usp=sharing>

## Descrition

This approach tackles LLM hallucination detection by analyzing the hidden states across layers during the inference step of the last generated token. The framework consists of three pipeline stages:
- Multi-Layer Activation Concatenation: I extract raw activation vectors from the late-middle layers ($19$ through $24$) specifically focusing on the last non-padded generated token, capturing the model's localized belief state just before producing the next word.Geometrical and Trajectory 

- Probing: In addition to raw activations, I extract handcrafted statistical metrics of the hidden state dynamics across the layers. These include:
    - Representation Drift: Relative $L_2$ norm changes (residual stream increments) and cross-layer Cosine Similarities to track state updates.

    - Distributional Characteristics: Layer-wise Kurtosis to capture activation sparsity and concentration, alongside raw $L_2$ norms to measure signal energy.
    
    - Trajectory Variance: Standard deviation across layers to quantify state representation noise.
    
- Dimensionality Reduction & Classifier Probe: Features are standardized and compressed using Principal Component Analysis (PCA) to prevent overfitting. A Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout is trained using class-weighted binary classification. Finally, a validation-based search determines the optimal decision threshold to maximize the F1-score.

## By files

### `aggregation.py`:

Semantic Features (aggregate): Extracts hidden state vectors for the last non-padding token across a predefined list of late-middle layers ([19, 20, 21, 22, 23, 24]) and concatenates them into a single high-dimensional feature vector.

Geometric Features (extract_geometric_features): Calculates statistical properties of the hidden state trajectories across layers, including residual norm differences (information drift), kurtosis (activation sharpness), trajectory standard deviation across a mid-layer slice, $L_2$ norms, and layer-to-layer cosine similarity.

### `probe.py` (The Classifier):

Uses StandardScaler followed by PCA (reducing to 256 components) to manage the high dimensionality of concatenated layers.

Employs an MLP with BatchNorm, ReLU, Dropout(0.2), and a final Sigmoid.

Optimizes class imbalance via pos_weight.

Automatically tunes the decision threshold on validation data to maximize the F1-score.

### `splitting.py` (Data Partitioning):

Uses simple stratified splits to partition data into Train, Validation, and Test sets, preserving the ratio of truthful to hallucinated labels.

Representation Probe with Geometrical and Cross-Layer Activation FeaturesOur approach tackles LLM hallucination detection by analyzing the internal representation spaces (hidden states) across layers during the inference step of the last generated token.
The framework consists of three pipeline stages:

- Multi-Layer Activation Concatenation: We extract raw activation vectors from the late-middle layers ($19$ through $24$) specifically focusing on the last non-padded generated token, capturing the model's localized belief state just before producing the next word.
- Geometrical and Trajectory Probing: In addition to raw activations, we extract handcrafted statistical metrics of the hidden state dynamics across the layers. These include:
  - Representation Drift: Relative $L_2$ norm changes (residual stream increments) and cross-layer Cosine Similarities to track state updates.
  - Distributional Characteristics: Layer-wise Kurtosis to capture activation sparsity and concentration, alongside raw $L_2$ norms to measure signal energy.
  - Trajectory Variance: Standard deviation across layers to quantify state representation noise.Dimensionality Reduction & Classifier Probe: Features are standardized and compressed using Principal Component Analysis (PCA) to prevent overfitting.
  - A Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout is trained using class-weighted binary classification. Finally, a validation-based search determines the optimal decision threshold to maximize the F1-score.

## Sources

1) https://arxiv.org/html/2510.11529v1

2) https://huggingface.co/blog/krogoldAI/llm-hallucination-detection

3) https://aclanthology.org/2025.acl-long.880.pdf