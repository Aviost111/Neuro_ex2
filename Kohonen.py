import random
import numpy as np
import matplotlib.pyplot as plt


class Kohonen:
    """
    Kohonen self-organizing map (SOM) implementation for clustering and visualization of high-dimensional data.

    Parameters:
    -----------
    learning_rate : float, default=0.1
        The learning rate for updating the weights of the neurons.
    neurons_amount : list of int, default=[100]
        The number of neurons in each layer of the SOM.
    """

    def __init__(self, learning_rate=0.1, neurons_amount=[100]):
        self.learning_rate = learning_rate
        self.neurons_amount = neurons_amount
        self.data = None
        self.neurons = []
        self.radius = max(self.neurons_amount[0], len(self.neurons_amount)) / 2
        self.lamda = None

    def fit(self, data_set, iteration=10000):
        """
        Train the SOM on the given data set.

        Parameters:
        -----------
        data_set : array-like, shape (n_samples, n_features)
            The input data to be clustered and visualized.
        iteration : int, default=10000
            The number of iterations to train the SOM for.
        """
        if len(data_set) == 0:
            raise ValueError("Data set cannot be empty.")
        self.data = np.array(data_set)
        self.lamda = iteration / np.log(self.radius)

        # initializing the neurons with random weights.
        for layer_size in self.neurons_amount:
            layer_neurons = []
            for _ in range(layer_size):
                weights = np.random.uniform(0, 1, size=len(data_set[0]))
                layer_neurons.append(weights)
            self.neurons.append(layer_neurons)
        self.neurons = np.array(self.neurons)

        # starting the fitting process.
        for i in range(iteration):
            # selecting random vector from the given data.
            vec = self.data[random.randint(0, len(self.data) - 1)]
            # find which neuron is the closest to the vector.
            nn = self.nearest_neuron(vec)
            # updating the learning rate and the radius.
            curr_learning_rate = self.learning_rate * np.exp(-i / self.lamda)
            curr_radius = self.radius * np.exp(-i / self.lamda)
            # going over the neurons in each layer and compute how to change each neuron.
            for layer_idx, layer_neurons in enumerate(self.neurons):
                for neuron_idx, neuron_weights in enumerate(layer_neurons):
                    curr_neuron = neuron_weights
                    d = np.linalg.norm(np.array(nn) - np.array([layer_idx, neuron_idx]))
                    neighbourhood = np.exp(- (d ** 2) / (2 * (curr_radius ** 2)))
                    self.neurons[layer_idx][neuron_idx] += curr_learning_rate * neighbourhood * (vec - curr_neuron)

            # plotting the network to track progress.
            if (i % 1000 == 0) or i == iteration - 1:
                if len(self.neurons_amount) == 1:
                    self.plot1D(i)
                else:
                    self.plot2D(i)

    def refit(self, data, iteration=1000):
        """
        Refit the SOM on a new data set.

        Parameters:
        -----------
        data : array-like, shape (n_samples, n_features)
            The new input data to be clustered and visualized.
        iteration : int, default=1000
            The number of iterations to train the SOM for.
        """
        if len(data) == 0:
            raise ValueError("Data set cannot be empty.")
        self.data = np.array(data)
        self.lamda = iteration / np.log(self.radius)
        for i in range(iteration):
            vec = self.data[random.randint(0, len(self.data) - 1)]
            nn = self.nearest_neuron(vec)
            curr_learning_rate = self.learning_rate * np.exp(-i / self.lamda)
            curr_radius = self.radius * np.exp(-i / self.lamda)

            for layer_idx, layer_neurons in enumerate(self.neurons):
                for neuron_idx, neuron_weights in enumerate(layer_neurons):
                    curr_neuron = neuron_weights
                    d = np.linalg.norm(np.array(nn) - np.array([layer_idx, neuron_idx]))
                    neighbourhood = np.exp(- (d ** 2) / (2 * (curr_radius ** 2)))
                    self.neurons[layer_idx][neuron_idx] += curr_learning_rate * neighbourhood * (
                            vec - curr_neuron)
            if (i % 1000 == 0) or i == iteration - 1:
                if len(self.neurons_amount) == 1:
                    self.plot1D(i)
                else:
                    self.plot2D(i)

    def nearest_neuron(self, vec):
        """
        Find the index of the neuron closest to the given input vector.

        Parameters:
        -----------
        vec : array-like, shape (n_features,)
            The input vector to find the closest neuron to.

        Returns:
        --------
        tuple of int
            The indices of the closest neuron in the format (layer_idx, neuron_idx).
        """
        min_dist = np.inf
        loc = None
        for layer_idx, layer_neurons in enumerate(self.neurons):
            for neuron_idx, neuron_weights in enumerate(layer_neurons):
                curr_dist = np.linalg.norm(vec - neuron_weights)
                if min_dist > curr_dist:
                    loc = (layer_idx, neuron_idx)
                    min_dist = curr_dist
        return loc

    def plot1D(self, t):
        """
        Plot the 1D SOM and the input data at a given iteration.

        Parameters:
        -----------
        t : int
            The iteration number to plot the SOM at.
        """
        xs = []
        ys = []
        for layer_neurons in self.neurons:
            for neuron_weights in layer_neurons:
                xs.append(neuron_weights[0])
                ys.append(neuron_weights[1])

        fig, ax = plt.subplots()
        ax.scatter([xs], [ys], c='r')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.plot(xs, ys, 'b-')
        ax.scatter(self.data[:, 0], self.data[:, 1], alpha=0.3)
        plt.title("Plot1D Iteration No. " + str(t))
        plt.show()

    def plot2D(self, t):
        """
        Plot the 2D SOM and the input data at a given iteration.

        Parameters:
        -----------
        t : int
            The iteration number to plot the SOM at.
        """
        neurons_x = self.neurons[:, :, 0]
        neurons_y = self.neurons[:, :, 1]
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i in range(neurons_x.shape[0]):
            xh = []
            yh = []
            xs = []
            ys = []
            for j in range(neurons_x.shape[1]):
                xs.append(neurons_x[i, j])
                ys.append(neurons_y[i, j])
                xh.append(neurons_x[j, i])
                yh.append(neurons_y[j, i])
            ax.plot(xs, ys, 'r-', markersize=0, linewidth=1)
            ax.plot(xh, yh, 'r-', markersize=0, linewidth=1)
        ax.plot(neurons_x, neurons_y, color='b', marker='o', linewidth=0, markersize=3)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="g", alpha=0.05, s=5)
        plt.title("Plot2D Iteration No. " + str(t))
        plt.show()
