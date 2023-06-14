import numpy as np
import matplotlib.pyplot as plt


def visualize_som(weights, iteration):
    plt.figure(figsize=(6, 6))
    plt.imshow(weights.T, cmap='viridis', origin='lower')
    plt.title(f'SOM Weights at Iteration {iteration}')
    plt.colorbar()
    plt.show()


# initialize weights
def initialize_weights(input_dim, output_dim):
    return np.random.rand(input_dim, output_dim[0], output_dim[1])


# set topological neighborhood parameters
def topological_neighborhood(radius, center, grid):
    distance = np.sqrt((grid[0] - center[0]) ** 2 + (grid[1] - center[1]) ** 2)
    return np.exp(-(distance ** 2) / (2 * (radius ** 2)))


# set learning rate parameters
def learning_rate(initial_lr, iteration, max_iteration):
    return initial_lr * (1 - iteration / max_iteration)


# Kohonen SOM algorithm
def kohonen_som(inputs, output_dim, max_iteration, initial_lr, initial_radius, visualize_interval=10):
    input_dim = inputs.shape[1]
    weights = initialize_weights(input_dim, output_dim)
    grid = np.array(list(np.ndindex(output_dim)))
    total_output_neurons = output_dim[0] * output_dim[1]
    for iteration in range(max_iteration):
        # update learning rate and radius
        lr = learning_rate(initial_lr, iteration, max_iteration)
        radius = initial_radius * np.exp(-iteration / np.log(max_iteration))
        for input_vector in inputs:
            # compute distances
            distances = np.sum((weights - input_vector.reshape(input_dim, 1, 1)) ** 2, axis=0)
            # find the winner neuron
            winner = np.unravel_index(np.argmin(distances), output_dim)
            # update weights of winner and its neighbors
            for j in range(total_output_neurons):
                idx = np.unravel_index(j, output_dim)
                neighborhood = topological_neighborhood(radius, winner, idx)
                weights[:, idx[0], idx[1]] += lr * neighborhood * (input_vector - weights[:, idx[0], idx[1]])

        # Visualize the SOM at specific intervals
        if iteration % visualize_interval == 0:
            visualize_som(weights, iteration)

    return weights



