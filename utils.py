
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
# Display Function for plotting an image using cv2.imshow
def show_img(img, window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = img.shape[:2]
    cv2.resizeWindow(window_name, w, h)
    cv2.imshow(window_name, img)


def show_all_images(image_list, window_name):
    """
    Displays all images in the list in separate resizable windows.
    """
    for idx, img in enumerate(image_list):
        show_img(img, window_name=f"{window_name} - Image {idx+1}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to scale an image by a given factor
def scale_img(image_list, scale):
    """
    Scales every image in the provided list by the given scaling factor.
    Returns a new list of scaled images.
    """
    scaled_images = []
    for img in image_list:
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        scaled_images.append(resized)
    return scaled_images

# Function to import an image from a given path
def import_img(path):
    img = cv2.imread(path)
    return img

def load_images_from_folder(folder_path):
    image_list = []
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    for ext in extensions:
        search_pattern = os.path.join(folder_path, ext)
        print(f"Searching for: {search_pattern}")  # Debug print
        for filename in glob.glob(search_pattern):
            print(f"Found file: {filename}")  # Debug print
            img = cv2.imread(filename)
            if img is not None:
                image_list.append(img)
    return image_list

def convert_to_gray(image_list):
    gray_images = []
    for img in image_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_images.append(gray)
    return gray_images