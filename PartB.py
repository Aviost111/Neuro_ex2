import cv2
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('hand.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary so we only have the hand
_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# Get the x, y coordinates of all the black points
y, x = np.nonzero(img)
data = np.array([x, y]).T / np.array([img.shape[1], img.shape[0]])  # normalize to [0, 1]

# Initialize SOM
som = MiniSom(20, 20, 2, sigma=1.0, learning_rate=0.5)

# Train the SOM
som.train_random(data, 10000)

# Visualize the SOM grid superimposed on the image
plt.imshow(img, cmap='gray')
plt.plot(som.get_weights()[:, :, 0].flatten() * img.shape[1], som.get_weights()[:, :, 1].flatten() * img.shape[0], 'r.')
plt.show()

# Define the range of x and y coordinates for the finger to be removed
x_min, x_max = 0.4, 0.5  # replace with actual values
y_min, y_max = 0.2, 0.4  # replace with actual values

# Remove one finger
data = data[~((data[:, 0] > x_min) & (data[:, 0] < x_max) & (data[:, 1] > y_min) & (data[:, 1] < y_max))]

# Continue training the SOM
som = MiniSom(20, 20, 2, sigma=1.0, learning_rate=0.5)  # re-initialize SOM
som.train_random(data, 10000)

# Visualize the updated SOM grid
plt.imshow(img, cmap='gray')
plt.plot(som.get_weights()[:, :, 0].flatten() * img.shape[1], som.get_weights()[:, :, 1].flatten() * img.shape[0], 'r.')
plt.show()
