
1 - aggregation.py (Feature Extraction & Aggregation):

Semantic Features (aggregate): Extracts hidden state vectors for the last non-padding token across a predefined list of late-middle layers ([19, 20, 21, 22, 23, 24]) and concatenates them into a single high-dimensional feature vector.

Geometric Features (extract_geometric_features): Calculates statistical properties of the hidden state trajectories across layers, including residual norm differences (information drift), kurtosis (activation sharpness), trajectory standard deviation across a mid-layer slice, $L_2$ norms, and layer-to-layer cosine similarity.

2 - probe.py (The Classifier):

Uses StandardScaler followed by PCA (reducing to 256 components) to manage the high dimensionality of concatenated layers.

Employs an MLP with BatchNorm, ReLU, Dropout(0.2), and a final Sigmoid.

Optimizes class imbalance via pos_weight.

Automatically tunes the decision threshold on validation data to maximize the F1-score.

3 - splitting.py (Data Partitioning):

Uses simple stratified splits to partition data into Train, Validation, and Test sets, preserving the ratio of truthful to hallucinated labels.

Representation Probe with Geometrical and Cross-Layer Activation FeaturesOur approach tackles LLM hallucination detection by analyzing the internal representation spaces (hidden states) across layers during the inference step of the last generated token.
The framework consists of three pipeline stages:

- Multi-Layer Activation Concatenation: We extract raw activation vectors from the late-middle layers ($19$ through $24$) specifically focusing on the last non-padded generated token, capturing the model's localized belief state just before producing the next word.
- Geometrical and Trajectory Probing: In addition to raw activations, we extract handcrafted statistical metrics of the hidden state dynamics across the layers. These include:
  - Representation Drift: Relative $L_2$ norm changes (residual stream increments) and cross-layer Cosine Similarities to track state updates.
  - Distributional Characteristics: Layer-wise Kurtosis to capture activation sparsity and concentration, alongside raw $L_2$ norms to measure signal energy.
  - Trajectory Variance: Standard deviation across layers to quantify state representation noise.Dimensionality Reduction & Classifier Probe: Features are standardized and compressed using Principal Component Analysis (PCA) to prevent overfitting.
  - A Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout is trained using class-weighted binary classification. Finally, a validation-based search determines the optimal decision threshold to maximize the F1-score.

# Sources
https://huggingface.co/blog/krogoldAI/llm-hallucination-detection