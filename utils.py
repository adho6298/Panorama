
# LIBRARIES
# importing required libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import cv2
import os
import glob

# FUNCTIONS AND UTILITIES
# Display Function for plotting an image using Matplotlib:
def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# Function to scale an image by a given factor
def scale_img(img, scale):
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized

# Function to import an image from a given path
def import_img(path):
    img = cv2.imread(path)
    return img

def test(num):
    sum = num + num
    return sum