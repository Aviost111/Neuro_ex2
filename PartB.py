import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
import cv2

# # Load and preprocess the hand image
# hand_img = cv2.imread("hand.jpg", cv2.IMREAD_GRAYSCALE)
# hand_img = cv2.resize(hand_img, (0, 0), fx=0.5, fy=0.5)
# hand_data = np.argwhere(hand_img < 255)
#
# # Define the dimensions of the SOM grid
# grid_width = 20
# grid_height = 20
#
# # Initialize the SOM
# som = MiniSom(grid_width, grid_height, 2, sigma=1.0, learning_rate=0.5)
#
# # Start the training process
# iterations = 1000
# som.random_weights_init(hand_data)
# som.train_random(hand_data, iterations)
#
# # Get the final positions of the SOM neurons
# neuron_positions = som.get_weights()
#
# # Plot the hand data and the SOM grid
# plt.scatter(hand_data[:, 1], hand_data[:, 0], color='b', label='hand Data')
# plt.scatter(neuron_positions[:, :, 1], neuron_positions[:, :, 0], color='r', label='SOM Neurons')
# plt.legend()
# plt.title('SOM Superimposed on Hand Data')
# plt.show()


# Load and preprocess the hand image
hand_img = cv2.imread("hand.jpg", cv2.IMREAD_GRAYSCALE)
hand_img = cv2.resize(hand_img, (0, 0), fx=0.5, fy=0.5)
hand_data = np.argwhere(hand_img < 255)

# Define the dimensions of the SOM grid
grid_width = 20
grid_height = 20

# Initialize the SOM
som = MiniSom(grid_width, grid_height, 2, sigma=1.0, learning_rate=0.5)

# Start the training process
iterations = 10000
som.random_weights_init(hand_data)

# Plot the initial state
plt.scatter(hand_data[:, 1], hand_data[:, 0], color='b', label='Hand Data')
plt.scatter(som._weights[:, :, 1].flatten(), som._weights[:, :, 0].flatten(), color='r', label='SOM Neurons')
plt.legend()
plt.title('SOM Superimposed on Hand Data - Iteration 0')
plt.show()

# Perform iterations and plot the results
for i in range(iterations):
    som.train_random(hand_data, 1)

    if (i % 1000 == 0) or i == iterations - 1:
        # Plot the current state
        plt.scatter(hand_data[:, 1], hand_data[:, 0], color='b', label='Hand Data')
        plt.scatter(som._weights[:, :, 1].flatten(), som._weights[:, :, 0].flatten(), color='r', label='SOM Neurons')
        plt.legend()
        plt.title(f'SOM Superimposed on Hand Data - Iteration {i}')
        plt.show()
# def visualize_som(som, iteration):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(som.weights[:, :, 1].flatten(), som.weights[:, :, 0].flatten(), color='red', label='SOM Neurons')
#     plt.scatter(som.data[:, 1], som.data[:, 0], color='lightblue', label='hand Data')
#     plt.legend()
#     plt.title(f'SOM Weights at Iteration {iteration}')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.show()
#
# # Load and preprocess the hand image
# hand_img = cv2.imread("hand.jpg", cv2.IMREAD_GRAYSCALE)
# hand_img = cv2.resize(hand_img, (0, 0), fx=0.5, fy=0.5)
# hand_data = np.argwhere(hand_img < 255)
#
# # Define the dimensions of the SOM grid
# grid_width = 20
# grid_height = 20
#
# # Initialize the SOM
# som = MiniSom(grid_width, grid_height, 2, sigma=1.0, learning_rate=0.5)
#
# # Start the training process
# iterations = 1000
# som.random_weights_init(hand_data)
# for i in range(iterations):
#     t = i + 1  # Current iteration
#     x = hand_data[i % len(hand_data)]
#     winning_neuron = som.winner(x)
#     som.update(x, winning_neuron, t)
#     if t % 100 == 0 or t == iterations:
#         visualize_som(som, t)
#
# # Get the final positions of the SOM neurons
# neuron_positions = som.get_weights()
#
# # Plot the final SOM and hand data
# visualize_som(som, 'Final')
