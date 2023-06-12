import numpy as np


class KohonenSOM:
    def __init__(self, input_dim, map_dim, iterations):
        self.input_dim = input_dim
        self.map_dim = map_dim
        self.iterations = iterations
        self.map = np.random.rand(map_dim[0], map_dim[1], input_dim)

    def train(self, data):
        for t in range(self.iterations):
            # Select a random input vector
            x = data[np.random.randint(data.shape[0])]

            # Compute Euclidean distance between input vector and all nodes in the map
            distances = np.linalg.norm(self.map - x, axis=2)

            # Find the node with the smallest distance (BMU)
            bmu = np.unravel_index(np.argmin(distances), distances.shape)

            # Determine topological neighbourhood and its radius
            sigma = self.map_dim[0] / 2.0 * np.exp(-t / self.iterations)
            neighbourhood = np.exp(
                -np.linalg.norm(np.indices(self.map_dim).T - np.array(bmu), axis=2) ** 2 / (2 * sigma ** 2))

            # Update weights of nodes in the BMU neighbourhood
            learning_rate = 0.1 * np.exp(-t / self.iterations)
            delta = learning_rate * neighbourhood[:, :, np.newaxis] * (x - self.map)
            self.map += delta

    def predict(self, data):
        distances = np.linalg.norm(self.map - data[:, np.newaxis, np.newaxis], axis=3)
        return np.argmin(distances, axis=2)
