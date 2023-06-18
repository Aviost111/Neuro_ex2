# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def visualize_som(weights, iteration):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(weights.T, cmap='viridis', origin='lower')
#     plt.title(f'SOM Weights at Iteration {iteration}')
#     plt.colorbar()
#     plt.show()
#
#
# # initialize weights
# def initialize_weights(input_dim, output_dim):
#     return np.random.rand(input_dim, output_dim[0], output_dim[1])
#
#
# # set topological neighborhood parameters
# def topological_neighborhood(radius, center, grid):
#     distance = np.sqrt((grid[0] - center[0]) ** 2 + (grid[1] - center[1]) ** 2)
#     return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
#
#
# # set learning rate parameters
# def learning_rate(initial_lr, iteration, max_iteration):
#     return initial_lr * (1 - iteration / max_iteration)
#
#
# # Kohonen SOM algorithm
# def kohonen_som(inputs, output_dim, max_iteration, initial_lr, initial_radius, visualize_interval=10):
#     input_dim = inputs.shape[1]
#     weights = initialize_weights(input_dim, output_dim)
#     grid = np.array(list(np.ndindex(output_dim)))
#     total_output_neurons = output_dim[0] * output_dim[1]
#     for iteration in range(max_iteration):
#         # update learning rate and radius
#         lr = learning_rate(initial_lr, iteration, max_iteration)
#         radius = initial_radius * np.exp(-iteration / np.log(max_iteration))
#         for input_vector in inputs:
#             # compute distances
#             distances = np.sum((weights - input_vector.reshape(input_dim, 1, 1)) ** 2, axis=0)
#             # find the winner neuron
#             winner = np.unravel_index(np.argmin(distances), output_dim)
#             # update weights of winner and its neighbors
#             for j in range(total_output_neurons):
#                 idx = np.unravel_index(j, output_dim)
#                 neighborhood = topological_neighborhood(radius, winner, idx)
#                 weights[:, idx[0], idx[1]] += lr * neighborhood * (input_vector - weights[:, idx[0], idx[1]])
#
#         # Visualize the SOM at specific intervals
#         if iteration % visualize_interval == 0:
#             visualize_som(weights, iteration)
#
#     return weights
#
#
#
import numpy as np
import matplotlib.pyplot as plt

# def initialize_weights(input_dim, output_dim):
#     return np.random.rand(input_dim, output_dim)
#
#
# def topological_neighborhood(radius, center, grid):
#     if radius == 0:
#         return 0
#     distance = np.abs(grid - center)
#     return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
#
#
# def learning_rate_or_radius(lr, iteration):
#     return lr * 0.9 * (1 - iteration / 1000)
#
#
# def visualize_som(weights, iteration):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(weights[0], weights[1])
#
#     # Draw lines connecting each neuron and its neighbors
#     for i in range(weights.shape[1] - 1):
#         plt.plot(weights[0, i:i + 2], weights[1, i:i + 2], 'r-')
#
#     plt.title(f'SOM Weights at Iteration {iteration}')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.show()
#
#
# def kohonen_som(inputs, output_dim, max_iteration, initial_lr, initial_radius, visualize_interval=10):
#     input_dim = inputs.shape[1]
#     weights = initialize_weights(input_dim, output_dim)
#     radius = initial_radius
#     # grid = np.arange(output_dim)
#     lr = initial_lr
#     for iteration in range(max_iteration):
#         # update learning rate and radius
#         lr = learning_rate_or_radius(lr, iteration)
#         radius = learning_rate_or_radius(radius, iteration)
#         for input_vector in inputs:
#             # compute distances
#             distances = np.sum((weights - input_vector.reshape(input_dim, -1)) ** 2, axis=0)
#             # find the winner neuron
#             winner = np.argmin(distances)
#             # update weights of winner and its neighbors
#             for idx in range(output_dim):
#                 neighborhood = topological_neighborhood(radius, winner, idx)
#                 weights[:, idx] += lr * neighborhood * (input_vector - weights[:, idx])
#
#         # Visualize the SOM at specific intervals
#         if iteration % visualize_interval == 0:
#             visualize_som(weights, iteration)
#
#     return weights
#
# import numpy as np
# import matplotlib.pyplot as plt


# class KohonenSOM:
#     def __init__(self, input_dim, output_dim, max_iteration, initial_lr, initial_radius, visualize_interval=10):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.max_iteration = max_iteration
#         self.initial_lr = initial_lr
#         self.initial_radius = initial_radius
#         self.visualize_interval = visualize_interval
#
#     def initialize_weights(self):
#         return np.random.rand(self.input_dim, self.output_dim)
#
#     def topological_neighborhood(self, radius, center, grid):
#         if radius == 0:
#             return 0
#         distance = np.abs(grid - center)
#         return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
#
#     def learning_rate_or_radius(self, lr, iteration):
#         return lr * 0.9 * (1 - iteration / 1000)
#
#     def visualize_som(self, weights, iteration):
#         plt.figure(figsize=(6, 6))
#         plt.scatter(weights[0], weights[1])
#
#         # Draw lines connecting each neuron and its neighbors
#         for i in range(weights.shape[1] - 1):
#             plt.plot(weights[0, i:i + 2], weights[1, i:i + 2], 'r-')
#
#         plt.title(f'SOM Weights at Iteration {iteration}')
#         plt.xlim(0, 1)
#         plt.ylim(0, 1)
#         plt.show()
#
#     def fit(self, inputs):
#         self.weights = self.initialize_weights()
#         radius = self.initial_radius
#         lr = self.initial_lr
#         for iteration in range(self.max_iteration):
#             # update learning rate and radius
#             lr = self.learning_rate_or_radius(lr, iteration)
#             radius = self.learning_rate_or_radius(radius, iteration)
#             for input_vector in inputs:
#                 # compute distances
#                 distances = np.sum((self.weights - input_vector.reshape(self.input_dim, -1)) ** 2, axis=0)
#                 # find the winner neuron
#                 winner = np.argmin(distances)
#                 # update weights of winner and its neighbors
#                 for idx in range(self.output_dim):
#                     neighborhood = self.topological_neighborhood(radius, winner, idx)
#                     self.weights[:, idx] += lr * neighborhood * (input_vector - self.weights[:, idx])
#
#             # Visualize the SOM at specific intervals
#             if iteration % self.visualize_interval == 0:
#                 self.visualize_som(self.weights, iteration)
#
#         return self.weights
import numpy as np
import matplotlib.pyplot as plt


class KohonenSOM:
    def __init__(self, learning_rate=0.1, neurons_amount=[100], max_iteration=10000, visualize_interval=1000):
        self.learning_rate = learning_rate
        self.neurons_amount = neurons_amount
        self.max_iteration = max_iteration
        self.visualize_interval = visualize_interval
        self.data = None
        self.neurons = None
        self.radius = max(neurons_amount[0], len(neurons_amount)) / 2
        self.lamda = None

    def initialize_neurons(self, input_dim, output_dim):
        neurons = []
        for _ in range(output_dim):
            weights = np.random.uniform(0, 1, size=input_dim)
            neurons.append(weights)
        return np.array(neurons)

    def topological_neighborhood(self, radius, center, grid):
        if radius == 0:
            return 0
        distance = np.abs(grid - center)
        return np.exp(-(distance ** 2) / (2 * (radius ** 2)))

    def learning_rate_or_radius(self, value, iteration):
        return value * np.exp(-iteration / self.lamda)

    def visualize_som(self, iteration):
        plt.figure(figsize=(6, 6))
        plt.scatter(self.neurons[:, 0], self.neurons[:, 1])

        # Draw lines connecting each neuron and its neighbors
        for i in range(self.neurons.shape[0] - 1):
            plt.plot(self.neurons[i:i + 2, 0], self.neurons[i:i + 2, 1], 'r-')

        # Plot input data points
        plt.scatter(self.data[:, 0], self.data[:, 1], color='lightblue', alpha=1.0)

        plt.title(f'SOM Weights at Iteration {iteration}')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    def fit(self, data_set):
        if len(data_set) == 0:
            raise ValueError("Data set cannot be empty.")
        self.data = np.array(data_set)
        self.lamda = self.max_iteration / np.log(self.radius)
        self.neurons = self.initialize_neurons(len(data_set[0]), self.neurons_amount[0])

        for i in range(self.max_iteration):
            vec = self.data[np.random.randint(0, len(self.data))]
            nn = self.nearest_neuron(vec)
            curr_learning_rate = self.learning_rate_or_radius(self.learning_rate, i)
            curr_radius = self.learning_rate_or_radius(self.radius, i)

            for neuron_idx, neuron_weights in enumerate(self.neurons):
                curr_neuron = neuron_weights
                d = np.linalg.norm(nn - neuron_idx)
                neighborhood = self.topological_neighborhood(curr_radius, 0, d)
                self.neurons[neuron_idx] += curr_learning_rate * neighborhood * (vec - curr_neuron)

            if (i % self.visualize_interval == 0) or (i == self.max_iteration - 1):
                self.visualize_som(i)

    def nearest_neuron(self, vec):
        min_dist = np.inf
        loc = None
        for neuron_idx, neuron_weights in enumerate(self.neurons):
            curr_dist = np.linalg.norm(vec - neuron_weights)
            if min_dist > curr_dist:
                loc = neuron_idx
                min_dist = curr_dist
        return loc
