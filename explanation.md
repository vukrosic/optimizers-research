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

Self-attention is permutation-invariant, meaning it doesn't know the order of tokens. We need to inject positional information. RoPE does this by rotating our query and key vectors (which we'll create in the next step) based on their position.

RoPE works by viewing pairs of features in our vectors as complex numbers and rotating them. For a vector `v = [v1, v2, v3, v4]` at position `m`, we would group features into pairs `(v1, v2)` and `(v3, v4)`. Each pair is rotated by an angle that depends on the position `m`.

The rotation formula for a pair `(x_i, x_{i+1})` at position `m` is:

\[
\begin{pmatrix} x'_i \\ x'_{i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix}
\]

Where `\theta_j` is a frequency that depends on the dimension `j`.

This happens inside the attention mechanism, after we've created Query (Q) and Key (K) vectors. We'll see this in the next section. In `llm.py`, this is handled by the `Rotary` class.

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

The formula is:
\[
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot g
\]
where `x` is a token's vector, `g` is a learnable gain parameter, and `\epsilon` is a small value for stability.

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
