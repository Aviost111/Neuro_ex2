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
iterations = 1000
som.random_weights_init(hand_data)
for i in range(iterations):
    som.train_random(hand_data, 1)  # Perform one iteration of training
    if (i % 100 == 0) or i == iterations - 1:
        plot2D(som, i)

