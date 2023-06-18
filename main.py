import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Kohonen import Kohonen
from Kohonen2 import KohonenSOM
import cv2 as cv


def Part_A1A():
    points = []
    for i in range(1000):
        points.append([random.uniform(0, 1), random.uniform(0, 1)])
    ko = KohonenSOM(neurons_amount=[20])
    ko.fit(points, flag=True)
    ko = KohonenSOM(neurons_amount=[200])
    ko.fit(points, flag=True)
    return


def Part_A1B():
    points = []
    for i in range(1000):
        points.append([random.gauss(0.5, 0.15), random.uniform(0, 1)])
    ko = KohonenSOM(neurons_amount=[20])
    ko.fit(points, flag=True)
    ko = KohonenSOM(neurons_amount=[200])
    ko.fit(points, flag=True)

    points = []
    for i in range(1000):
        points.append([random.gauss(0.5, 0.15), random.gauss(0.5, 0.15)])
    ko = KohonenSOM(neurons_amount=[20])
    ko.fit(points, flag=True)
    ko = KohonenSOM(neurons_amount=[200])
    ko.fit(points, flag=True)
    return


def Part_A2():
    points = []
    for i in range(7000):
        x = random.uniform(-4, 4)
        y = random.uniform(-4, 4)
        distance = x ** 2 + y ** 2
        if 4 <= distance <= 16:
            points.append([x, y])
    circle = points
    ko = KohonenSOM(neurons_amount=[300], learning_rate=0.05)
    ko.fit(circle, flag=False)
    pass


def Part_B():
    hand = cv.imread("hand.jpg")
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    hand = cv2.resize(hand, (0, 0), fx=0.5, fy=0.5)

    points = np.argwhere(hand != 255).astype(np.float32)
    plt.show()
    max = points.max(axis=0)
    max = max * 1.0
    points[:, 0] = points[:, 0] / max[0]
    points[:, 1] = points[:, 1] / max[1]
    print(len(points))
    shape = hand.shape
    print(points.max(axis=0), shape)
    layers = (np.ones(15) * 15).astype(int)
    ko = Kohonen(neurons_amount=layers, learning_rate=0.4)
    ko.fit(points, iteration=10000)

    hand2 = cv.imread("80%_hand.jpg")
    hand2 = cv2.cvtColor(hand2, cv2.COLOR_BGR2GRAY)
    hand2 = cv2.resize(hand2, (0, 0), fx=0.5, fy=0.5)
    points2 = np.argwhere(hand2 != 255).astype(np.float32)
    max = points2.max(axis=0)
    max = max * 1.0
    points2[:, 0] = points2[:, 0] / max[0]
    points2[:, 1] = points2[:, 1] / max[1]
    ko.refit(points2)

    plt.imshow(hand2)
    pass


if __name__ == '__main__':
    # Part_A1A()
    # Part_A1B()
    # Part_A2()
    Part_B()
