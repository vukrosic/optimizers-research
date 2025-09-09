import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a directory to save the images
os.makedirs("adamw_tutorial/images", exist_ok=True)

# Define a simple quadratic objective function to optimize
# This function is easy to visualize and has a clear minimum at (0, 0)
def objective_function(w):
    return w[0]**2 + w[1]**2

# Define the gradient of the objective function
def gradient(w):
    return torch.tensor([2 * w[0], 2 * w[1]])

def run_optimizer(optimizer_class, **kwargs):
    """
    Runs an optimizer on the objective function and records its path.
    """
    # Initialize weight tensor
    w = torch.tensor([-4.0, 3.0], requires_grad=True)
    
    # Instantiate the optimizer
    optimizer = optimizer_class([w], **kwargs)
    
    path = [w.detach().clone().numpy()]
    
    # Run optimization for a number of steps
    for i in range(50):
        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Calculate loss and gradients
        loss = objective_function(w)
        loss.backward()
        
        # Perform optimization step
        optimizer.step()
        
        path.append(w.detach().clone().numpy())
        
    return np.array(path)

# --- Run Optimizers ---
# 1. Standard SGD with momentum for baseline
path_sgd = run_optimizer(torch.optim.SGD, lr=0.1, momentum=0.9)

# 2. Adam optimizer
path_adam = run_optimizer(torch.optim.Adam, lr=0.3, betas=(0.9, 0.999), weight_decay=0.0)

# 3. AdamW optimizer
path_adamw = run_optimizer(torch.optim.AdamW, lr=0.3, betas=(0.9, 0.999), weight_decay=0.3)


# --- Plotting ---
def plot_paths(paths, labels, filename):
    """
    Generates and saves a contour plot showing the optimization paths.
    """
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')
    plt.colorbar(contour, label='Loss Value')

    for path, label in zip(paths, labels):
        plt.plot(path[:, 0], path[:, 1], 'o-', label=label, markersize=3)

    plt.title("Optimizer Paths on a Simple Quadratic Function")
    plt.xlabel("Weight 1")
    plt.ylabel("Weight 2")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.savefig(f"adamw_tutorial/images/{filename}")
    plt.close()
    print(f"✅ Saved plot: adamw_tutorial/images/{filename}")

# Plot all three paths together
plot_paths(
    [path_sgd, path_adam, path_adamw],
    ["SGD with Momentum", "Adam", "AdamW"],
    "optimizer_paths_comparison.png"
)

# Plot Adam vs AdamW to highlight the effect of weight decay
plot_paths(
    [path_adam, path_adamw],
    ["Adam (Weight Decay = 0.3, ineffective)", "AdamW (Weight Decay = 0.3, effective)"],
    "adam_vs_adamw_weight_decay.png"
)

print("\nVisualizations for the AdamW tutorial have been successfully generated.")
