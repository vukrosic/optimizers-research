import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_1d(start_x, learning_rate=0.1, max_iters=5):
    """
    Simple gradient descent for a 1D quadratic f(x) = x^2
    """
    def f(x):
        return x**2

    def grad_f(x):
        return 2*x

    x = start_x
    points = [x]

    for _ in range(max_iters):
        grad = grad_f(x)
        x = x - learning_rate * grad
        points.append(x)

    return points

def plot_descent(points):
    """
    Plot the function and gradient descent steps with clear individual markers
    """
    x_range = np.linspace(-2, 2, 200)
    y_range = x_range**2

    plt.figure(figsize=(12, 8))

    # Plot the function
    plt.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x²')

    # Plot each step individually with different colors and step numbers
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

    for i, (x, color) in enumerate(zip(points, colors)):
        y = x**2
        plt.scatter(x, y, color=color, s=60, alpha=0.8, edgecolors='black', linewidth=1)
        plt.text(x+0.02, y+0.05, str(i), fontsize=8, ha='center')

        # Connect consecutive points with a line
        if i > 0:
            prev_x, prev_y = points[i-1], points[i-1]**2
            plt.plot([prev_x, x], [prev_y, y], color=color, alpha=0.6, linewidth=2)

    # Mark start and end points
    start_y = points[0]**2
    end_y = points[-1]**2
    plt.scatter(points[0], start_y, color='red', s=100, marker='*', label='Start')
    plt.scatter(points[-1], end_y, color='green', s=100, marker='*', label='End')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent on f(x) = x² (5 steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Simple example
if __name__ == "__main__":
    points = gradient_descent_1d(start_x=2.0, learning_rate=0.1)

    print(f"Started at: {points[0]:.3f}")
    print(f"Ended at: {points[-1]:.3f}")
    print(f"Steps: {len(points)-1}")

    plot_descent(points)
