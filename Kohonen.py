# import numpy as np
#
#
# class KohonenSOM:
#     def __init__(self, input_dim, map_dim, iterations):
#         self.input_dim = input_dim
#         self.map_dim = map_dim
#         self.iterations = iterations
#         self.map = np.random.rand(map_dim[0], map_dim[1], input_dim)
#
#     def train(self, data):
#         for t in range(self.iterations):
#             # Select a random input vector
#             x = data[np.random.randint(data.shape[0])]
#
#             # Compute Euclidean distance between input vector and all nodes in the map
#             distances = np.linalg.norm(self.map - x, axis=2)
#
#             # Find the node with the smallest distance (BMU)
#             bmu = np.unravel_index(np.argmin(distances), distances.shape)
#
#             # Determine topological neighbourhood and its radius
#             sigma = self.map_dim[0] / 2.0 * np.exp(-t / self.iterations)
#             neighbourhood = np.exp(
#                 -np.linalg.norm(np.indices(self.map_dim).T - np.array(bmu), axis=2) ** 2 / (2 * sigma ** 2))
#
#             # Update weights of nodes in the BMU neighbourhood
#             learning_rate = 0.1 * np.exp(-t / self.iterations)
#             delta = learning_rate * neighbourhood[:, :, np.newaxis] * (x - self.map)
#             self.map += delta
#
#     def predict(self, data):
#         distances = np.linalg.norm(self.map - data[:, np.newaxis, np.newaxis], axis=3)
#         return np.argmin(distances, axis=2)
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Parameters
number_of_inputs = 1000
N = 10  # N^2 points

upper_bound_x = 1
lower_bound_x = -1
upper_bound_m = 0.1
lower_bound_m = -0.1

alpha_initial = 0.1
sigma_initial = 0.05
neighbour_radius_initial = N / 2

T = 300  # number of iterations
t = 1

# Initiate input and neural field
x1 = np.random.uniform(lower_bound_x, upper_bound_x, number_of_inputs)
x2 = np.random.uniform(lower_bound_x, upper_bound_x, number_of_inputs)
w1 = np.random.uniform(lower_bound_m, upper_bound_m, (N, N))
w2 = np.random.uniform(lower_bound_m, upper_bound_m, (N, N))

# Initial figures
plt.figure(1)
plt.plot(x1, x2, 'ob')
plt.gca().set_prop_cycle(None)
plt.plot(w1, w2, 'or')
plt.plot(w1, w2, 'r', linewidth=1)
plt.plot(w1.T, w2.T, 'r', linewidth=1)
plt.title('t=0')
plt.pause(0.001)

# Start training
while t <= T:
    # Update parameters
    alpha = alpha_initial * (1 - t / T)
    sigma = sigma_initial * (1 - t / T)
    max_neighbour_radius = int(neighbour_radius_initial * (1 - t / T))

    # Loop over all input values
    for i in range(number_of_inputs):
        # Find minimum distance neural unit (winner)
        e_norm = (x1[i] - w1) ** 2 + (x2[i] - w2) ** 2
        min_index = np.argmin(e_norm)
        minj1 = min_index // N
        minj2 = min_index % N
        j1_c, j2_c = minj1, minj2  # Winner coordinates

        # Update the winning neuron
        e_factor = np.exp(-((j1_c - j1_c) ** 2 + (j2_c - j1_c) ** 2) / (2 * sigma))
        w1[j1_c, j2_c] += alpha * (x1[i] - w1[j1_c, j2_c])
        w2[j1_c, j2_c] += alpha * (x2[i] - w2[j1_c, j2_c])

        # Update the neighbour neurons
        for neighbour_radius in range(1, max_neighbour_radius + 1):
            jj1, jj2 = j1_c - neighbour_radius, j2_c
            if jj1 >= 0:
                e_factor = np.exp(-((j1_c - jj1) ** 2 + (j2_c - jj2) ** 2) / (2 * sigma))
                w1[jj1, jj2] += alpha * e_factor * (x1[i] - w1[jj1, jj2])
                w2[jj1, jj2] += alpha * e_factor * (x2[i] - w2[jj1, jj2])

            jj1, jj2 = j1_c + neighbour_radius, j2_c
            if jj1 < N:
                e_factor = np.exp(-((j1_c - jj1) ** 2 + (j2_c - jj2) ** 2) / (2 * sigma))
                w1[jj1, jj2] += alpha * e_factor * (x1[i] - w1[jj1, jj2])
                w2[jj1, jj2] += alpha * e_factor * (x2[i] - w2[jj1, jj2])

            jj1, jj2 = j1_c, j2_c - neighbour_radius
            if jj2 >= 0:
                e_factor = np.exp(-((j1_c - jj1) ** 2 + (j2_c - jj2) ** 2) / (2 * sigma))
                w1[jj1, jj2] += alpha * e_factor * (x1[i] - w1[jj1, jj2])
                w2[jj1, jj2] += alpha * e_factor * (x2[i] - w2[jj1, jj2])

            jj1, jj2 = j1_c, j2_c + neighbour_radius
            if jj2 < N:
                e_factor = np.exp(-((j1_c - jj1) ** 2 + (j2_c - jj2) ** 2) / (2 * sigma))
                w1[jj1, jj2] += alpha * e_factor * (x1[i] - w1[jj1, jj2])
                w2[jj1, jj2] += alpha * e_factor * (x2[i] - w2[jj1, jj2])

    t += 1
    plt.figure(1)
    plt.plot(x1, x2, 'ob')
    plt.gca().set_prop_cycle(None)
    plt.plot(w1, w2, 'or')
    plt.plot(w1, w2, 'r', linewidth=1)
    plt.plot(w1.T, w2.T, 'r', linewidth=1)
    plt.title(f't={t}')
    plt.pause(0.001)

plt.show()
