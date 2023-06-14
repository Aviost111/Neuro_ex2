import numpy as np


# initialize weights
def initialize_weights(input_dim, output_dim):
    return np.random.rand(input_dim, output_dim)


# set topological neighborhood parameters
def topological_neighborhood(radius, center, grid):
    distance = np.sqrt((grid[0] - center[0]) ** 2 + (grid[1] - center[1]) ** 2)
    return np.exp(-(distance ** 2) / (2 * (radius ** 2)))


# set learning rate parameters
def learning_rate(initial_lr, iteration, max_iteration):
    return initial_lr * (1 - iteration / max_iteration)


# Kohonen SOM algorithm
def kohonen_som(inputs, output_dim, max_iteration, initial_lr, initial_radius):
    input_dim = inputs.shape[1]
    weights = initialize_weights(input_dim, output_dim)
    grid = np.array(list(np.ndindex(output_dim)))
    for iteration in range(max_iteration):
        # update learning rate and radius
        lr = learning_rate(initial_lr, iteration, max_iteration)
        radius = initial_radius * np.exp(-iteration / np.log(max_iteration))
        for input_vector in inputs:
            # compute distances
            distances = np.sum((weights - input_vector) ** 2, axis=1)
            # find the winner neuron
            winner = np.argmin(distances)
            # update weights of winner and its neighbors
            for j in range(output_dim):
                neighborhood = topological_neighborhood(radius, grid[j], grid)
                weights[:, j] += lr * neighborhood * (input_vector - weights[:, j])
    return weights
