import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a directory to save the images
os.makedirs("rope_tutorial/images", exist_ok=True)

# --- RoPE Implementation ---
def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precomputes the rotational frequencies for RoPE."""
    # Calculate theta_i for each pair of dimensions
    theta_i = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # Create position indices
    m = torch.arange(max_seq_len)
    
    # Calculate the angles m * theta_i
    freqs = torch.outer(m, theta_i)
    
    # Return as complex numbers (for easy rotation)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor):
    """Applies RoPE to a tensor of vectors."""
    # Reshape x into pairs of features to treat as complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Apply rotation by multiplying with the precomputed complex frequencies
    # The shape of freqs_complex is broadcasted to match x_complex
    x_rotated = x_complex * freqs_complex
    
    # Reshape back to the original tensor shape
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)

# --- Visualization 1: The Rotation Effect ---
dim = 2
max_seq_len = 16
freqs = precompute_rope_freqs(dim, max_seq_len)

# A single vector to be rotated
v = torch.tensor([[1.0, 0.0]]) # Start with a vector pointing along the x-axis

# Apply RoPE at each position
rotated_vectors = []
for i in range(max_seq_len):
    # We need to reshape v to (1, 1, 2) to match expected input dim
    # and select the frequency for the current position i
    rotated_v = apply_rope(v.unsqueeze(1), freqs[i:i+1, :])
    rotated_vectors.append(rotated_v.squeeze().numpy())

rotated_vectors = np.array(rotated_vectors)

plt.figure(figsize=(8, 8))
origin = np.zeros_like(rotated_vectors)
plt.quiver(origin[:, 0], origin[:, 1], rotated_vectors[:, 0], rotated_vectors[:, 1], 
           angles='xy', scale_units='xy', scale=1, 
           color=plt.cm.viridis(np.linspace(0, 1, max_seq_len)))

for i in range(max_seq_len):
    plt.text(rotated_vectors[i, 0] * 1.1, rotated_vectors[i, 1] * 1.1, f'm={i}', fontsize=9)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title("RoPE: Rotation of a 2D Vector by Position (m)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("rope_tutorial/images/rotation_effect.png")
plt.close()
print("✅ Saved plot: rope_tutorial/images/rotation_effect.png")

# --- Visualization 2: Relative Position Property ---
dim = 16 # Use a slightly larger dimension
max_seq_len = 50
freqs = precompute_rope_freqs(dim, max_seq_len)

# Create two random vectors for query (q) and key (k)
torch.manual_seed(42)
q = torch.randn(1, 1, dim)
k = torch.randn(1, 1, dim)

relative_distances = range(-10, 11)
attention_scores = {}

# Calculate attention scores for different absolute positions but same relative distance
for dist in relative_distances:
    scores = []
    # We test 10 different pairs of (m, n) for each distance
    for m in range(10, 20):
        n = m - dist
        if 0 <= n < max_seq_len:
            # Apply RoPE to q at position m and k at position n
            q_rot = apply_rope(q, freqs[m:m+1, :])
            k_rot = apply_rope(k, freqs[n:n+1, :])
            
            # Calculate dot product (attention score)
            # We need to ensure they are 1D vectors for the dot product
            score = torch.dot(q_rot.view(-1), k_rot.view(-1)).item()
            scores.append(score)
    
    # The scores should be (almost) identical for the same relative distance
    if scores:
        attention_scores[dist] = np.mean(scores)

distances = list(attention_scores.keys())
scores = list(attention_scores.values())

plt.figure(figsize=(10, 6))
plt.plot(distances, scores, 'o-')
plt.title("RoPE's Key Property: Attention Depends on Relative Position")
plt.xlabel("Relative Distance (m - n)")
plt.ylabel("Attention Score (Dot Product)")
plt.xticks(np.arange(min(distances), max(distances)+1, 2))
plt.grid(True)
plt.savefig("rope_tutorial/images/relative_attention.png")
plt.close()
print("✅ Saved plot: rope_tutorial/images/relative_attention.png")
