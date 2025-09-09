# A Step-by-Step Guide to a Transformer with Mixture of Experts (MoE)

This document breaks down the inner workings of a Transformer model, similar to the one implemented in `llm.py`. We'll trace the journey of a token from a simple ID to a final prediction, explaining how vectors and matrices are transformed at each step. We will use small, simplified examples with easy-to-follow numbers.

Let's assume our model has the following tiny dimensions for simplicity:
- **Vocabulary size**: 10 (we have 10 unique words)
- **Embedding dimension (`d_model`)**: 4
- **Sequence length (`T`)**: 3
- **Number of attention heads (`n_heads`)**: 2
- **Head dimension (`d_k`)**: `d_model / n_heads` = 2
- **Number of experts (`num_experts`)**: 4
- **Top-k routing (`k`)**: 2

---

## 1. Vector Embeddings: From Words to Numbers

The model can't work with words directly. The first step is to convert input tokens (words or sub-words) into numerical vectors. This is done using an **embedding matrix**.

Imagine our input text is "hello world ai".

**Step 1: Tokenization**
First, we map each word to a unique integer ID from our vocabulary.

Vocabulary is a list of 50,000 to 300,000 tokens. Tokens are words, letters, subwords, other character and character groups.

Let's presume the following IDs for these tokens:
- "hello" -> 1
- "world" -> 5
- "ai" -> 9

Our input sequence of IDs is `[1, 5, 9]`.

**Step 2: Embedding Lookup**
Next, each token has it's own vector embedding (vector that represents its meaning).
Each number (position, dimernsion) in vector is a learned feature.

Let's say our embedding dimension is 4, so each token is represented by a vector with 4 numbers. Each number in the vector can (very loosely) represent a different feature or property of the token. For example, maybe:
- The first dimension could represent "positivity"
- The second could represent "formality"
- The third could represent "abstractness"
- The fourth could represent "concreteness"

Suppose we have two tokens:
- "happy" (token ID 1)
- "boss" (token ID 2)

Their embeddings might look like:
- embedding("happy") = [0.9, 0.2, 0.7, 0.1]  # high positivity, low formality, high abstractness, low concreteness
- embedding("boss")  = [0.1, 0.8, 0.3, 0.9]  # low positivity, high formality, low abstractness, high concreteness

If a number in the embedding vector is high, it means the token strongly expresses that feature. If it's low, the token doesn't express that feature much. For example, "happy" has a high value in the first dimension (positivity), while "boss" has a high value in the second dimension (formality) and fourth dimension (concreteness).

So, the embedding matrix (for vocab size 4) could look like:
[
  [0.0, 0.0, 0.0, 0.0],  # ID 0 (maybe padding)
  [0.9, 0.2, 0.7, 0.1],  # ID 1 ("happy")
  [0.1, 0.8, 0.3, 0.9],  # ID 2 ("boss")
  [0.3, 0.3, 0.2, 0.5]   # ID 3 (another token)
]

Let's say our embedding matrix (size `10 x 4`) looks like this:

```
# Embedding Matrix (Vocab Size x d_model)
[
  [0.1, 0.2, 0.3, 0.4],  # ID 0
  [1.1, 1.2, 1.3, 1.4],  # ID 1 ("hello")
  [2.1, 2.2, 2.3, 2.4],  # ID 2
  ...
  [5.1, 5.2, 5.3, 5.4],  # ID 5 ("world")
  ...
  [9.1, 9.2, 9.3, 9.4]   # ID 9 ("ai")
  ...
]
```

We look up the vector for each token ID in our input sequence.

- `embedding(1)` -> `[1.1, 1.2, 1.3, 1.4]`
- `embedding(5)` -> `[5.1, 5.2, 5.3, 5.4]`
- `embedding(9)` -> `[9.1, 9.2, 9.3, 9.4]`

This gives us our initial input matrix `X` of shape `(T, d_model)` or `(3, 4)`:

```
X = [
  [1.1, 1.2, 1.3, 1.4],  # "hello"
  [5.1, 5.2, 5.3, 5.4],  # "world"
  [9.1, 9.2, 9.3, 9.4]   # "ai"
]
```
In `llm.py`, this is done by `self.token_embedding`. The code also multiplies by `math.sqrt(self.config.d_model)`.

---

## 2. Positional Information: Rotary Positional Embeddings (RoPE)

Self-attention is permutation-invariant, meaning it doesn't know the order of tokens. Without positional information, the sequence "hello world ai" would be treated the same as "ai hello world" or "world ai hello". We need to inject positional information so the model understands that "hello" comes first, "world" comes second, and "ai" comes third.

RoPE (Rotary Positional Embeddings) solves this by rotating our query and key vectors based on their position in the sequence. Think of it like adding a unique "signature" to each token that encodes where it sits in the sentence.

### How RoPE Works: Step by Step

**Step 1: Understanding the Core Idea**

RoPE treats pairs of features in our vectors as if they were complex numbers on a 2D plane. Instead of adding positional encodings (like older methods), RoPE rotates the vectors themselves. The amount of rotation depends on the token's position.

Imagine you have a vector `v = [v1, v2, v3, v4]` at position `m`. RoPE groups the features into pairs: `(v1, v2)` and `(v3, v4)`. Each pair gets rotated by an angle that's unique to that position.

**Step 2: The Rotation Mathematics**

For a pair of features `(x, y)` from a vector at position `m`, RoPE applies a rotation. The formulas for this are:

- `x' = x * cos(θ_m) - y * sin(θ_m)`
- `y' = x * sin(θ_m) + y * cos(θ_m)`

Where:
- `m` is the position of the token in the sequence (e.g., 0, 1, 2...).
- `θ_m` is the angle of rotation, which is calculated as `m * θ_i`.
- `θ_i = 1.0 / (10000^(2i / d))`, where `i` is the index of the feature pair (0 for the first pair, 1 for the second, etc.) and `d` is the embedding dimension.

This formula means that the angle of rotation is different for each pair of features and for each position, creating a unique positional signal.

**Step 3: A Concrete Example**

Let's apply RoPE to the query vector of "world" which is at position `m=1`.
Assume our `d_model` is 4, and the query vector for "world" is `q_world = [0.5, 0.8, 0.2, 0.7]`.

1.  **Group into pairs:**
    - Pair 1: `(0.5, 0.8)` (`i=0`)
    - Pair 2: `(0.2, 0.7)` (`i=1`)

2.  **Calculate `θ_i` for each pair:** (`d=4`)
    - For `i=0`: `θ_0 = 1 / (10000^(2*0 / 4)) = 1 / (10000^0) = 1.0`
    - For `i=1`: `θ_1 = 1 / (10000^(2*1 / 4)) = 1 / (10000^0.5) = 1 / 100 = 0.01`

3.  **Calculate the rotation angle `θ_m` for position `m=1`:**
    - Angle for Pair 1: `m * θ_0 = 1 * 1.0 = 1.0` radian
    - Angle for Pair 2: `m * θ_1 = 1 * 0.01 = 0.01` radians

4.  **Rotate Pair 1 (angle = 1.0 rad):**
    - `cos(1.0) ≈ 0.54`, `sin(1.0) ≈ 0.84`
    - `x' = 0.5 * 0.54 - 0.8 * 0.84 = 0.27 - 0.672 = -0.402`
    - `y' = 0.5 * 0.84 + 0.8 * 0.54 = 0.42 + 0.432 = 0.852`
    - Rotated Pair 1: `[-0.402, 0.852]`

5.  **Rotate Pair 2 (angle = 0.01 rad):**
    - `cos(0.01) ≈ 0.99995`, `sin(0.01) ≈ 0.01`
    - `x' = 0.2 * 0.99995 - 0.7 * 0.01 = 0.19999 - 0.007 = 0.193`
    - `y' = 0.2 * 0.01 + 0.7 * 0.99995 = 0.002 + 0.699965 = 0.702`
    - Rotated Pair 2: `[0.193, 0.702]`

6.  **Combine the rotated pairs:**
    The final rotated query vector for "world" is `q'_world = [-0.402, 0.852, 0.193, 0.702]`. This new vector now contains information about its content *and* its position.

### Why is RoPE so effective?

The magic of RoPE is that the dot product between two rotated vectors `q'_m` and `k'_n` (query at position `m` and key at position `n`) depends only on their *relative position* `m-n`, not their absolute positions. This property makes the attention scores sensitive to the distance between tokens, which is exactly what we want.

### Python Code Example

Here is a simplified Python implementation of RoPE, similar to what you might find in `llm.py`.

```python
import torch

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_complex: torch.Tensor
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    """
    # Reshape x for complex number representation
    # (B, T, H, D) -> (B, T, H, D/2)
    x_complex = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_complex)

    # Apply rotation in the complex domain
    # (B, T, H, D/2) * (T, D/2) -> (B, T, H, D/2)
    x_rotated = x_complex * freqs_complex.unsqueeze(0).unsqueeze(2)

    # Convert back to real numbers and reshape
    x_out = torch.view_as_real(x_rotated).flatten(3)

    return x_out.type_as(x)

# Precompute the frequencies
def precompute_theta_pos_frequencies(head_dim: int, max_seq_len: int, device: str) -> torch.Tensor:
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(positions, theta).float()
    return torch.polar(torch.ones_like(freqs), freqs) # complex64
```
---

## 3. Multi-Head Attention: How Tokens Talk to Each Other

Attention allows tokens to "look" at other tokens in the sequence and decide which ones are important.

**Step 1: Create Query, Key, and Value (Q, K, V) Vectors**
We start with our input matrix `X`. For each token, we create three vectors: a Query, a Key, and a Value. This is done by multiplying `X` with three weight matrices (`W_q`, `W_k`, `W_v`) that are learned during training.

Let's focus on a single head first. The dimensions of `W_q`, `W_k`, `W_v` for one head are `(d_model, d_k)`, i.e., `(4, 2)`.

```
# Simplified weight matrices for one head
W_q = [[...], [...], [...], [...]] (4x2)
W_k = [[...], [...], [...], [...]] (4x2)
W_v = [[...], [...], [...], [...]] (4x2)
```

`Q = X @ W_q`
`K = X @ W_k`
`V = X @ W_v`

This results in Q, K, and V matrices of shape `(T, d_k)` or `(3, 2)`.

```
Q = [
  [q11, q12], # "hello" query
  [q21, q22], # "world" query
  [q31, q32]  # "ai" query
] (3x2)
```
(Same shapes for K and V)

**Step 2: Apply RoPE**
Now, we apply RoPE to our Q and K matrices. Each row (token vector) in Q and K is rotated according to its position.
- Row 0 ("hello") is rotated by position 0 (no rotation).
- Row 1 ("world") is rotated by position 1.
- Row 2 ("ai") is rotated by position 2.

After RoPE, we have `Q_rot` and `K_rot`.

**Step 3: Calculate Attention Scores**
The score determines how much attention a token should pay to other tokens. We calculate scores by taking the dot product of a token's Query vector with the Key vectors of all other tokens.

`AttentionScores = Q_rot @ K_rot.T`

```
# AttentionScores (3x3)
[
  [q1.k1, q1.k2, q1.k3],  # Scores for "hello" w.r.t all tokens
  [q2.k1, q2.k2, q2.k3],  # Scores for "world"
  [q3.k1, q3.k2, q3.k3]   # Scores for "ai"
]
```

**Step 4: Scale and Mask**
We scale the scores by dividing by the square root of `d_k` to stabilize gradients.

`ScaledScores = AttentionScores / sqrt(d_k)`

For a decoder-style model, we apply a mask to prevent tokens from looking at future tokens. We set the scores for future positions to `-infinity`.

```
# Masked Scores
[
  [s11, -inf, -inf],
  [s21, s22, -inf],
  [s31, s32, s33]
]
```

**Step 5: Softmax**
We apply a softmax function along each row to convert scores into probabilities (attention weights).

`AttentionWeights = softmax(ScaledScores)`

```
# AttentionWeights (3x3)
[
  [w11, 0,   0  ],  # "hello" attends mostly to itself
  [w21, w22, 0  ],  # "world" attends to "hello" and itself
  [w31, w32, w33]   # "ai" attends to all three
]
```
The sum of each row is 1.

**Step 6: Compute Output**
The output for each token is a weighted sum of all Value vectors.

`Output = AttentionWeights @ V`

This gives an output matrix of shape `(T, d_k)` or `(3, 2)`.

**Multi-Head:**
We do this for `n_heads` (2 in our case) in parallel, each with its own `W_q`, `W_k`, `W_v` matrices. This gives us two output matrices of size `(3, 2)`. We then concatenate them to get a `(3, 4)` matrix. Finally, we multiply this by a final output weight matrix `W_o` (size `4x4`) to produce the final attention output. This is implemented in `MultiHeadAttention`.

---

## 4. Add & Norm (RMSNorm)

The output of the attention mechanism is added to the original input `X` (a residual connection), and then normalized.

`X_after_attn = X + AttentionOutput`

**RMSNorm (Root Mean Square Normalization)**
RMSNorm is a simpler and faster alternative to LayerNorm.

The formula is: `RMSNorm(x) = (x / sqrt(mean(x*x) + epsilon)) * g`

where `x` is a token's vector, `g` is a learnable gain parameter, and `epsilon` is a small value for stability.

Let's apply it to a single vector `v = [2, 3, -1, 4]` from our `X_after_attn` matrix.

1.  **Square the elements**: `[4, 9, 1, 16]`
2.  **Calculate the mean**: `(4 + 9 + 1 + 16) / 4 = 7.5`
3.  **Take the square root**: `sqrt(7.5) = 2.74`
4.  **Normalize**: `v / 2.74 = [0.73, 1.10, -0.37, 1.46]`
5.  **Apply gain**: Multiply by a learned vector `g`.

This normalization is applied to each token vector independently. In `llm.py`, this is `self.norm1`.

---

## 5. Mixture of Experts (MoE)

Instead of a single large Feed-Forward Network (FFN), MoE uses multiple smaller FFNs called "experts" and a "router" network that decides which experts to send each token to. This allows the model to have many more parameters but keep the computation for each token constant.

We take the output of the Add & Norm step, let's call it `X_normed`, as input to the MoE layer.

**Step 1: The Router**
The router is a simple linear layer that takes a token's vector and outputs a logit for each expert.

`RouterLogits = X_normed @ W_router` where `W_router` has shape `(d_model, num_experts)` or `(4, 4)`.

```
# X_normed (3x4)
[
  [...], # "hello" vector
  [...], # "world" vector
  [...], # "ai" vector
]

# RouterLogits (3x4)
[
  [2.1, 0.5, 1.3, 3.5], # "hello" logits for experts 1-4
  [4.2, 3.1, 1.1, 0.9], # "world" logits
  [0.8, 4.5, 2.5, 3.3]  # "ai" logits
]
```

**Step 2: Top-K Routing**
For each token, we select the `k=2` experts with the highest logits. We then apply softmax to just these `k` logits to get the weights for combining the experts' outputs.

- **"hello"**: Top-2 logits are `3.5` (Expert 4) and `2.1` (Expert 1).
  - `softmax([3.5, 2.1]) = [0.80, 0.20]`
  - So, "hello" will be sent to Expert 4 with weight 0.80 and Expert 1 with weight 0.20.

- **"world"**: Top-2 logits are `4.2` (Expert 1) and `3.1` (Expert 2).
  - `softmax([4.2, 3.1]) = [0.75, 0.25]`
  - Sent to Expert 1 (weight 0.75) and Expert 2 (weight 0.25).

- **"ai"**: Top-2 logits are `4.5` (Expert 2) and `3.3` (Expert 4).
  - `softmax([4.5, 3.3]) = [0.77, 0.23]`
  - Sent to Expert 2 (weight 0.77) and Expert 4 (weight 0.23).

**Step 3: Process with Experts**
Each expert is a standard FFN (e.g., two linear layers with a non-linearity). We pass the token vectors to their assigned experts.

- `Output_hello = 0.80 * Expert4(hello_vec) + 0.20 * Expert1(hello_vec)`
- `Output_world = 0.75 * Expert1(world_vec) + 0.25 * Expert2(world_vec)`
- `Output_ai = 0.77 * Expert2(ai_vec) + 0.23 * Expert4(ai_vec)`

Note that each token is still only processed by `k` experts, which makes this computationally efficient.

The final output of the MoE layer is a matrix of the same shape as the input `(3, 4)`. This is implemented in the `MixtureOfExperts` class in `llm.py`.

---

## Final Steps: Another Add & Norm and Prediction

Just like after the attention layer, we have a final residual connection and normalization:

`X_final = X_normed + MoE_Output`
`X_final_normed = RMSNorm(X_final)` (this is `self.norm2` in the code)

**Prediction Head**
Finally, to predict the next token, we apply a linear layer (the "LM Head") to our final output vectors. This layer maps the `d_model`-sized vector back to the vocabulary size.

`Logits = X_final_normed @ W_lm_head` where `W_lm_head` has shape `(d_model, vocab_size)` or `(4, 10)`. Often, `W_lm_head` is the same as the embedding matrix (`self.token_embedding.weight`).

This gives us a `(3, 10)` matrix of logits. We can then apply a softmax to the last vector (for the "ai" token) to get a probability distribution over the entire vocabulary for the next token. The token with the highest probability is our prediction.

---

## 6. Stacking Multiple Layers

The real power of transformers comes from stacking many of these blocks (or layers) on top of each other. The process described above—from Multi-Head Attention to the MoE network—constitutes a single transformer block.

A deep transformer model might have dozens of these blocks. Here’s how they connect:

1.  The initial token embeddings (`X`) are fed into the **first transformer block**.
2.  The output of this first block (`X_final_normed` in our example) becomes the **input to the second transformer block**.
3.  This continues for all subsequent blocks. The output of block `N` is the input to block `N+1`.

Each layer refines the token representations. Early layers might capture basic syntax and local word relationships, while deeper layers can build more complex, abstract semantic meanings that span the entire sequence. The residual connections (`Add & Norm`) at each step are crucial, as they allow gradients to flow more easily through this deep network during training and prevent the model from losing information from earlier layers.

In `llm.py`, the `Transformer` class holds a list of these `TransformerBlock`s in `self.layers`.

---

## 7. Training the Model: A High-Level Overview

This document has focused on the **forward pass**: how an input sequence produces a prediction. But how does the model learn to make *good* predictions? This happens during training, which involves a **backward pass**.

**Step 1: Calculate the Loss**

For a given input sequence like "hello world", the model predicts the next token, "ai". We compare the model's predicted probability distribution (from the final softmax) with the actual correct token. The difference between the prediction and the ground truth is quantified by a **loss function**. A common choice is **Cross-Entropy Loss**.

- If the model assigns a high probability to the correct token ("ai"), the loss is low.
- If it assigns a low probability to the correct token, the loss is high.

The goal of training is to adjust the model's weights to minimize this loss across a huge dataset of text.

**Step 2: Backpropagation**

This is the core of the learning process. Backpropagation is an algorithm that calculates the **gradient** of the loss function with respect to every single learnable weight in the model (e.g., the matrices `W_q`, `W_k`, `W_v`, the expert FFN weights, the gain parameters in RMSNorm, etc.).

A gradient tells us two things:
- **Direction**: Which way to adjust a weight to decrease the loss.
- **Magnitude**: How much that weight contributed to the final error.

Think of it as the model figuring out the "blame" for its mistake and assigning it to each weight.

**Step 3: The Optimizer**

Once we have the gradients, an **optimizer** (like Adam or SGD) updates all the weights. It takes a small step in the direction opposite to the gradient, effectively nudging the millions of parameters in the model so that the next time it sees a similar input, its prediction will be slightly closer to the correct answer.

This cycle of `forward pass -> calculate loss -> backpropagation -> update weights` is repeated millions or billions of times on a massive text corpus, allowing the model to gradually learn the patterns of language.

---

## 8. A Deep Dive into the Optimizers

In `llm.py`, a hybrid optimization strategy is used. Instead of a single optimizer for all model parameters, the parameters are split into two groups, each handled by a different optimizer. This is a sophisticated approach that applies the best tool for the job.

- **AdamW**: Handles the "non-matrix" parameters like embeddings, normalization gains, and any biases. These parameters are often 1D vectors or have unique structures.
- **Muon**: A custom optimizer designed specifically for the core 2D weight matrices in the attention and feed-forward layers.

### AdamW: The Dependable Workhorse

AdamW is an evolution of the popular Adam optimizer. It combines two key ideas: **momentum** and **adaptive learning rates**, with an improved handling of **weight decay**.

**1. Core Idea: Adaptive Learning Rates**
Instead of using a single, fixed learning rate, AdamW adapts it for each individual parameter. It maintains two moving averages for each parameter:
- **First Moment (the mean of the gradients)**: This is the **momentum**. It keeps track of the general direction of the gradients, helping to accelerate movement and smooth out oscillations.
- **Second Moment (the variance of the gradients)**: This tracks how much the gradients for a parameter vary. If a parameter's gradients are all over the place (high variance), the optimizer will take smaller steps. If the gradients are consistent (low variance), it will take larger, more confident steps.

**2. Key Feature: Decoupled Weight Decay**
Weight decay is a regularization technique that prevents weights from growing too large by adding a penalty to the loss function. The original Adam optimizer mixed this decay with the adaptive learning rate, which could lead to suboptimal results.

AdamW "decouples" the weight decay. Instead of modifying the gradients, it applies the decay directly to the weights *after* the main optimization step. This allows for more effective regularization.

**The Math (Simplified):**
For a weight `w` with gradient `g`:
1. `m = beta1 * m + (1 - beta1) * g`  (Update momentum)
2. `v = beta2 * v + (1 - beta2) * g*g`  (Update variance)
3. `w = w - lr * m / (sqrt(v) + epsilon)` (Main update step)
4. `w = w - lr * weight_decay * w` (Decoupled weight decay)

### Muon: Orthogonalized Momentum

Muon (`MomentUm Orthogonalized by Newton-schulz`) is the custom optimizer from your code. It's designed to improve upon standard momentum by ensuring the updates it makes are "well-behaved," particularly for large matrix multiplications.

**1. Core Idea: Standard Momentum**
First, Muon calculates the momentum update, just like a standard SGD with Momentum or Nesterov momentum optimizer. It maintains a `momentum_buffer` that accumulates a moving average of the past gradients.

`buf = momentum * buf + (1 - momentum) * g`

This `buf` (buffer) represents the smoothed, historical direction of the gradients. With Nesterov momentum, it calculates the gradient "a little ahead" in this direction, which can help the optimizer anticipate changes and converge faster.

**2. Key Feature: Orthogonalization**
This is the novel part of Muon. After calculating the momentum-infused gradient `g`, it passes it through an **orthogonalization** step:

`g = zeropower_via_newtonschulz5(g)`

What does this mean? Think of a matrix. If its rows (or columns) are orthogonal, they are all perpendicular to each other. They represent independent, non-redundant directions in space. In deep learning, when weight matrices have this property, training can be much more stable and efficient. The updates don't interfere with each other, and information flows more cleanly.

The `zeropower_via_newtonschulz5` function is an algorithm that takes any matrix and iteratively transforms it into a matrix that is **orthogonal**, without changing its fundamental "shape" or "span."

By orthogonalizing the gradient update matrix `g`, we ensure that the update applied to the model's weights is as clean and non-redundant as possible. It pushes the weights in independent directions, which can prevent the "exploding" or "vanishing" gradient problems and lead to more stable training.

**3. The Final Update Step**

Finally, the orthogonalized gradient `g` is used to update the parameter `p`:

`p.add_(g, alpha=-lr * scale_factor)`

The `scale_factor` (`max(1, rows / cols)**0.5`) is a small adjustment to account for the shape of the matrix, ensuring that wider or taller matrices are updated appropriately.

**Illustrative Example: Why Orthogonalize?**

Imagine you're navigating a robot with two joysticks.
- **Standard Momentum**: You push the "forward" joystick. But due to bad wiring, it also makes the robot turn slightly right. You want to turn left, but that joystick also makes the robot slow down a bit. The controls are *coupled* and inefficient. This is like a non-orthogonal update, where changing one feature inadvertently affects another.
- **Orthogonalized Momentum (Muon)**: You push the "forward" joystick, and the robot goes perfectly straight. You use the "turn" joystick, and it turns perfectly on the spot. The controls are *independent* and clean. This is what Muon aims for: an update where each component of the gradient influences a unique, independent aspect of the weight matrix.
