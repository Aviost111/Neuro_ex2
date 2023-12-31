# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# #
# # class KohonenSOM:
# #     def __init__(self, learning_rate=0.1, neurons_amount=[100], max_iteration=10000, visualize_interval=1000):
# #         self.learning_rate = learning_rate
# #         self.neurons_amount = neurons_amount
# #         self.max_iteration = max_iteration
# #         self.visualize_interval = visualize_interval
# #         self.data = None
# #         self.neurons = None
# #         self.radius = max(neurons_amount[0], len(neurons_amount)) / 2
# #         self.lamda = None
# #
# #     def initialize_neurons(self, input_dim, output_dim):
# #         neurons = []
# #         for _ in range(output_dim):
# #             weights = np.random.uniform(0, 1, size=input_dim)
# #             neurons.append(weights)
# #         return np.array(neurons)
# #
# #     def topological_neighborhood(self, radius, center, grid):
# #         if radius == 0:
# #             return 0
# #         distance = np.abs(grid - center)
# #         return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
# #
# #     def learning_rate_or_radius(self, value, iteration):
# #         return value * np.exp(-iteration / self.lamda)
# #
# #     def visualize_som(self, iteration, flag):
# #         plt.figure(figsize=(6, 6))
# #         plt.scatter(self.neurons[:, 0], self.neurons[:, 1])
# #
# #         # Draw lines connecting each neuron and its neighbors
# #         for i in range(self.neurons.shape[0] - 1):
# #             plt.plot(self.neurons[i:i + 2, 0], self.neurons[i:i + 2, 1], 'r-')
# #
# #         # Plot input data points
# #         plt.scatter(self.data[:, 0], self.data[:, 1], color='lightblue', alpha=1.0)
# #
# #         plt.title(f'SOM Weights at Iteration {iteration}')
# #         if flag:
# #             plt.xlim(0, 1)
# #             plt.ylim(0, 1)
# #         else:
# #             plt.xlim(-4, 4)
# #             plt.ylim(-4, 4)
# #         plt.show()
# #
# #     def fit(self, data_set, flag):
# #         if len(data_set) == 0:
# #             raise ValueError("Data set cannot be empty.")
# #         self.data = np.array(data_set)
# #         self.lamda = self.max_iteration / np.log(self.radius)
# #         self.neurons = self.initialize_neurons(len(data_set[0]), self.neurons_amount[0])
# #
# #         for i in range(self.max_iteration):
# #             vec = self.data[np.random.randint(0, len(self.data))]
# #             nn = self.nearest_neuron(vec)
# #             curr_learning_rate = self.learning_rate_or_radius(self.learning_rate, i)
# #             curr_radius = self.learning_rate_or_radius(self.radius, i)
# #
# #             for neuron_idx, neuron_weights in enumerate(self.neurons):
# #                 curr_neuron = neuron_weights
# #                 d = np.linalg.norm(nn - neuron_idx)
# #                 neighborhood = self.topological_neighborhood(curr_radius, 0, d)
# #                 self.neurons[neuron_idx] += curr_learning_rate * neighborhood * (vec - curr_neuron)
# #
# #             if (i % self.visualize_interval == 0) or (i == self.max_iteration - 1):
# #                 self.visualize_som(i, flag)
# #
# #     def nearest_neuron(self, vec):
# #         min_dist = np.inf
# #         loc = None
# #         for neuron_idx, neuron_weights in enumerate(self.neurons):
# #             curr_dist = np.linalg.norm(vec - neuron_weights)
# #             if min_dist > curr_dist:
# #                 loc = neuron_idx
# #                 min_dist = curr_dist
# #         return loc
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class KohonenSOM:
#     def __init__(self, learning_rate=0.1, neurons_amount=[100], max_iteration=10000, visualize_interval=1000):
#         self.learning_rate = learning_rate
#         self.neurons_amount = neurons_amount
#         self.max_iteration = max_iteration
#         self.visualize_interval = visualize_interval
#         self.data = None
#         self.neurons = None
#         self.radius = max(neurons_amount[0], len(neurons_amount)) / 2
#         self.lamda = None
#
#     def initialize_neurons(self, input_dim, output_dim):
#         neurons = []
#         for _ in range(output_dim):
#             weights = np.random.uniform(0, 1, size=input_dim)
#             neurons.append(weights)
#         return np.array(neurons)
#
#     def topological_neighborhood(self, radius, center, grid):
#         if radius == 0:
#             return 0
#         distance = np.abs(grid - center)
#         return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
#
#     def learning_rate_or_radius(self, value, iteration):
#         return value * np.exp(-iteration / self.lamda)
#
#     def visualize_som(self, iteration, flag):
#         plt.figure(figsize=(6, 6))
#         plt.scatter(self.neurons[:, 0], self.neurons[:, 1])
#
#         # Draw lines connecting each neuron and its neighbors
#         for i in range(self.neurons.shape[0] - 1):
#             plt.plot(self.neurons[i:i + 2, 0], self.neurons[i:i + 2, 1], 'r-')
#
#         # Plot input data points
#         plt.scatter(self.data[:, 0], self.data[:, 1], color='lightblue', alpha=1.0)
#
#         plt.title(f'SOM Weights at Iteration {iteration}')
#         if flag:
#             plt.xlim(0, 1)
#             plt.ylim(0, 1)
#         else:
#             plt.xlim(-4, 4)
#             plt.ylim(-4, 4)
#         plt.show()
#
#     def fit(self, data_set, flag=True, plot2D=False):
#         if len(data_set) == 0:
#             raise ValueError("Data set cannot be empty.")
#         self.data = np.array(data_set)
#         self.lamda = self.max_iteration / np.log(self.radius)
#         self.neurons = self.initialize_neurons(len(data_set[0]), self.neurons_amount[0])
#
#         for i in range(self.max_iteration):
#             vec = self.data[np.random.randint(0, len(self.data))]
#             nn = self.nearest_neuron(vec)
#             curr_learning_rate = self.learning_rate_or_radius(self.learning_rate, i)
#             curr_radius = self.learning_rate_or_radius(self.radius, i)
#
#             for neuron_idx, neuron_weights in enumerate(self.neurons):
#                 curr_neuron = neuron_weights
#                 d = np.linalg.norm(nn - neuron_idx)
#                 neighborhood = self.topological_neighborhood(curr_radius, 0, d)
#                 self.neurons[neuron_idx] += curr_learning_rate * neighborhood * (vec - curr_neuron)
#
#             if (i % 1000 == 0) or i == self.max_iteration - 1:
#                 if len(self.neurons_amount) == 1:
#                     self.visualize_som(i, flag)
#                 else:
#                     self.plot2D(i)
#         #     if (i % self.visualize_interval == 0) or (i == self.max_iteration - 1):
#         #         self.visualize_som(i, flag)
#         #
#         # if plot2D:
#         #     self.plot2D()
#
#     def nearest_neuron(self, vec):
#         min_dist = np.inf
#         loc = None
#         for neuron_idx, neuron_weights in enumerate(self.neurons):
#             curr_dist = np.linalg.norm(vec - neuron_weights)
#             if min_dist > curr_dist:
#                 loc = neuron_idx
#                 min_dist = curr_dist
#         return loc
#
#     def plot2D(self, t):
#         """
#         Plot the 2D SOM and the input data at a given iteration.
#
#         Parameters:
#         -----------
#         t : int
#             The iteration number to plot the SOM at.
#         """
#         neurons_x = self.neurons[:, :, 0]
#         neurons_y = self.neurons[:, :, 1]
#         fig, ax = plt.subplots()
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         for i in range(neurons_x.shape[0]):
#             xh = []
#             yh = []
#             xs = []
#             ys = []
#             for j in range(neurons_x.shape[1]):
#                 xs.append(neurons_x[i, j])
#                 ys.append(neurons_y[i, j])
#                 xh.append(neurons_x[j, i])
#                 yh.append(neurons_y[j, i])
#             ax.plot(xs, ys, 'r-', markersize=0, linewidth=1)
#             ax.plot(xh, yh, 'r-', markersize=0, linewidth=1)
#         ax.plot(neurons_x, neurons_y, color='b', marker='o', linewidth=0, markersize=3)
#         ax.scatter(self.data[:, 0], self.data[:, 1], c="g", alpha=0.05, s=5)
#         plt.title("Plot2D Iteration No. " + str(t))
#         plt.show()
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class KohonenSOM:
#     def __init__(self, learning_rate=0.1, neurons_amount=[100], max_iteration=10000, visualize_interval=1000):
#         self.learning_rate = learning_rate
#         self.neurons_amount = neurons_amount
#         self.max_iteration = max_iteration
#         self.visualize_interval = visualize_interval
#         self.data = None
#         self.neurons = None
#         self.radius = max(neurons_amount[0], len(neurons_amount)) / 2
#         self.lamda = None
#
# def initialize_neurons(self, input_dim, output_dim):
#     neurons = []
#     for _ in range(output_dim):
#         weights = np.random.uniform(0, 1, size=input_dim)
#         neurons.append(weights)
#     return np.array(neurons)
#
#     def topological_neighborhood(self, radius, center, grid):
#         if radius == 0:
#             return 0
#         distance = np.abs(grid - center)
#         return np.exp(-(distance ** 2) / (2 * (radius ** 2)))
#
#     def learning_rate_or_radius(self, value, iteration):
#         return value * np.exp(-iteration / self.lamda)
#
#     def visualize_som(self, iteration, flag):
#         plt.figure(figsize=(6, 6))
#         plt.scatter(self.neurons[:, 0], self.neurons[:, 1])
#
#         # Draw lines connecting each neuron and its neighbors
#         for i in range(self.neurons.shape[0] - 1):
#             plt.plot(self.neurons[i:i + 2, 0], self.neurons[i:i + 2, 1], 'r-')
#
#         # Plot input data points
#         plt.scatter(self.data[:, 0], self.data[:, 1], color='lightblue', alpha=1.0)
#
#         plt.title(f'SOM Weights at Iteration {iteration}')
#         if flag:
#             plt.xlim(0, 1)
#             plt.ylim(0, 1)
#         else:
#             plt.xlim(-4, 4)
#             plt.ylim(-4, 4)
#         plt.show()
#
#     def fit(self, data_set, flag):
#         if len(data_set) == 0:
#             raise ValueError("Data set cannot be empty.")
#         self.data = np.array(data_set)
#         self.lamda = self.max_iteration / np.log(self.radius)
#         self.neurons = self.initialize_neurons(len(data_set[0]), self.neurons_amount[0])
#
#         for i in range(self.max_iteration):
#             vec = self.data[np.random.randint(0, len(self.data))]
#             nn = self.nearest_neuron(vec)
#             curr_learning_rate = self.learning_rate_or_radius(self.learning_rate, i)
#             curr_radius = self.learning_rate_or_radius(self.radius, i)
#
# for neuron_idx, neuron_weights in enumerate(self.neurons):
#     curr_neuron = neuron_weights
#     d = np.linalg.norm(nn - neuron_idx)
#     neighborhood = self.topological_neighborhood(curr_radius, 0, d)
#     self.neurons[neuron_idx] += curr_learning_rate * neighborhood * (vec - curr_neuron)
#
#             if (i % self.visualize_interval == 0) or (i == self.max_iteration - 1):
#                 self.visualize_som(i, flag)
#
#     def nearest_neuron(self, vec):
#         min_dist = np.inf
#         loc = None
#         for neuron_idx, neuron_weights in enumerate(self.neurons):
#             curr_dist = np.linalg.norm(vec - neuron_weights)
#             if min_dist > curr_dist:
#                 loc = neuron_idx
#                 min_dist = curr_dist
#         return loc
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

    def initialize_neurons2d(self, input_dim, output_dim):
        self.neurons = []
        for layer_size in output_dim:
            neurons = []
            for _ in range(layer_size):
                weights = np.random.uniform(0, 1, size=input_dim)
                neurons.append(weights)
            self.neurons.append(neurons)
        self.neurons = np.array(self.neurons)

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

    def visualize_som(self, iteration, flag):
        plt.figure(figsize=(6, 6))
        plt.scatter(self.neurons[:, 0], self.neurons[:, 1])

        # Draw lines connecting each neuron and its neighbors
        for i in range(self.neurons.shape[0] - 1):
            plt.plot(self.neurons[i:i + 2, 0], self.neurons[i:i + 2, 1], 'r-')

        # Plot input data points
        plt.scatter(self.data[:, 0], self.data[:, 1], color='lightblue', alpha=1.0)

        plt.title(f'SOM Weights at Iteration {iteration}')
        if flag:
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        else:
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
        plt.show()

    def fit(self, data_set, flag=True, plot2D=False):
        if len(data_set) == 0:
            raise ValueError("Data set cannot be empty.")
        self.data = np.array(data_set)
        self.lamda = self.max_iteration / np.log(self.radius)
        if len(self.neurons_amount) != 1:
            self.initialize_neurons2d(len(data_set[0]), self.neurons_amount)
        else:
            self.neurons = self.initialize_neurons(len(data_set[0]), self.neurons_amount[0])

        for i in range(self.max_iteration):
            vec = self.data[np.random.randint(0, len(self.data))]
            nn = self.nearest_neuron(vec)
            curr_learning_rate = self.learning_rate_or_radius(self.learning_rate, i)
            curr_radius = self.learning_rate_or_radius(self.radius, i)

            if len(self.neurons_amount) != 1:
                for layer_idx, layer_neurons in enumerate(self.neurons):
                    for neuron_idx, neuron_weights in enumerate(layer_neurons):
                        curr_neuron = neuron_weights
                        d = np.linalg.norm(np.array(nn) - np.array([layer_idx, neuron_idx]))
                        neighbourhood = np.exp(- (d ** 2) / (2 * (curr_radius ** 2)))
                        self.neurons[layer_idx][neuron_idx] += curr_learning_rate * neighbourhood * (vec - curr_neuron)

                if (i % 1000 == 0) or i == self.max_iteration - 1:
                    self.plot2D(i)
            else:
                for neuron_idx, neuron_weights in enumerate(self.neurons):
                    curr_neuron = neuron_weights
                    d = np.linalg.norm(nn - neuron_idx)
                    neighborhood = self.topological_neighborhood(curr_radius, 0, d)
                    self.neurons[neuron_idx] += curr_learning_rate * neighborhood * (vec - curr_neuron)
                if (i % 1000 == 0) or i == self.max_iteration - 1:
                    self.visualize_som(i, flag)

        #     if (i % self.visualize_interval == 0) or (i == self.max_iteration - 1):
        #         self.visualize_som(i, flag)
        #
        # if plot2D:
        #     self.plot2D()

    def nearest_neuron(self, vec):
        min_dist = np.inf
        loc = None
        for neuron_idx, neuron_weights in enumerate(self.neurons):
            curr_dist = np.linalg.norm(vec - neuron_weights)
            if min_dist > curr_dist:
                loc = neuron_idx
                min_dist = curr_dist
        return loc

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
