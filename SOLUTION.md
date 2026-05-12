
# SOLUTION.md — Hallucination Detection in Qwen2.5-0.5B

**Final test AUROC: 68.8 % (Best result across folds)**

**`predictions.csv` (publicly hosted):** <https://drive.google.com/file/d/1WVSzLbSPWOi7wLsUGdscaf7BmoU3Zu_i/view?usp=sharing>

**`results.json`:** <https://drive.google.com/file/d/1Q1gm7xYKPw6-lSmmh1_8MeRz48LM-g7d/view?usp=sharing>

## Descrition

This approach tackles LLM hallucination detection by analyzing the hidden states across layers during the inference step of the last generated token. The framework consists of three pipeline stages:
- Multi-Layer Activation Concatenation: I extract raw activation vectors from the late-middle layers ($19$ through $24$) specifically focusing on the last non-padded generated token, capturing the model's localized belief state just before producing the next word.

Geometrical and Trajectory Probing: In addition to raw activations, I extract handcrafted statistical metrics of the hidden state dynamics across the layers. These include:
    - Representation Drift: Relative $L_2$ norm changes (residual stream increments) and cross-layer Cosine Similarities to track state updates.

    - Distributional Characteristics: Layer-wise Kurtosis to capture activation sparsity and concentration, alongside raw $L_2$ norms to measure signal energy.
    
    - Trajectory Variance: Standard deviation across layers to quantify state representation noise.
    
- Dimensionality Reduction & Classifier Probe: Features are standardized and compressed using Principal Component Analysis (PCA) to prevent overfitting. A Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout is trained using class-weighted binary classification. Finally, a validation-based search determines the optimal decision threshold to maximize the F1-score.

## Features

After much experimentation, I have identified the following features:

- States of the last token from the upper layers (Last-token states): Embeddings of the final (real, non-padding) token for the last 4 layers of the model are extracted.

- Semantic drift / Layer differences: The difference between the vector of the last token on the final layer and its vectors on the middle layers of the network ($\sim50\%$ and $\sim75\%$ depths) is calculated. This shows how the meaning of the "final thought" changed and became more precise as it passed through the layers.

**Geometric features:**

- Tail statistics by layers: For the upper layers (starting from the middle of the network), the last 16 tokens are cut out and counted:

  - The average value of activations (layer_tail_mean) by layers.

  - Standard deviation (layer_tail_std) — a measure of the variability and "excitement" of the model at the end of the text.

- L2 vector norms (activation intensity): The "length" of vectors in Euclidean space is measured for the upper layers using three different text compression strategies:

  - The norm of the final token.
  
  - The norm of the average vector over the entire real sequence.

  - The norm of the vector after Max Pooling (maximum activations for each dimension).
  
- Cosine similarity (cos_sims) between layers

- Kurtosis of activations: A statistical indicator of the "sharpness" of the distribution of values in the vector of the last token. A high kurtosis indicates that there are rare but very strong bursts in the vector (activation of specific rare features), and a low one indicates a more even distribution.

## Project structure

### `aggregation.py`:

Semantic Features (aggregate): Extracts hidden state vectors for the last non-padding token across a predefined list of late-middle layer and concatenates them into a single high-dimensional feature vector.

Geometric Features (extract_geometric_features): Calculates statistical properties of the hidden state trajectories across layers, including residual norm differences (information drift), kurtosis (activation sharpness), tail statistics by layers, trajectory standard deviation across a mid-layer slice, $L_2$ norms, and layer-to-layer cosine similarity.

### `probe.py` (The Classifier):

The architecture of Neural Network looks like this:
 - The input layer is $\to$ Hidden layer (128 neurons) $\to$SiLU$\to$Dropout(0.4)
 
 - Hidden layer (128) $\to$ Hidden layer (32 neurons) $\to$SiLU$\to$Dropout(0.2)
 
 - Hidden layer (32) $\to$ Output layer (1 logit)
 
Features: The SiLU activation function (Swish) is used, which provides a smoother gradient compared to ReLU. The high Dropout coefficients (0.4 and 0.2) and Weight Decay regularization (0.05) in the AdamW optimizer are applied specifically to severely suppress overfitting (overfitting) on small samples.

Automatically tunes the decision threshold on validation data to maximize the F1-score.

### `splitting.py` (Data Partitioning):

Uses simple stratified splits to partition data into Train, Validation, and Test sets, preserving the ratio of truthful to hallucinated labels.

## Sources

1) https://arxiv.org/html/2510.11529v1

2) https://huggingface.co/blog/krogoldAI/llm-hallucination-detection

3) https://aclanthology.org/2025.acl-long.880.pdf
