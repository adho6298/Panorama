# LIBRARIES
# importing required libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import cv2
import os
import glob

# # # # # # 1. Reduce the image size (optional) for faster processing

print("Step 1")                           # Printing a title
print("Scaled Images")

scale = 0.6        # setting scaling factor to resize image by

# writing all the image paths to a vector to load later
# not really necessary now but good practice and can help later with large data sets
image_paths = [f'../Images/VictoriaLibrary/{i}.jpg' for i in range(1, 4)]   # only works because we know how many images are in our folder
# image_paths = ['VictoriaLibrary/1.jpg', 'VictoriaLibrary/2.jpg', 'VictoriaLibrary/3.jpg']

images = []        # creating list to store images in
dimensions = []    # creating list to store dimension data in

# Loop to load, get dimensions, resize, and print
for path in image_paths:                  # Python magic - assigning each item to path and then iterating through all the objects in the image_paths list
    img = cv2.imread(path)                # reading the image with cv2 and loading it into a variable called img
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # Resizing the image
    images.append(resized)                # adding the resized image to the list in images[]
    h,w = resized.shape[:2]               # getting the resized resolution of the image
    dimensions.append((h,w))              # adding the resolution data to dimensions[]

# Displaying the scaled down images
for i in range(len(images)):
    print(f"Scaling factor: {scale*100}%  {dimensions[i]}")  # printing description for images
    show_img(images[i])

# at this point we have imported all images, resized them, and displayed them with some useful information





# # # # # # 2. Convert Image to gray scale image (Try 'cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)')





print("")                            # Printing a space and a title
print("Step 2")
print("Gray scaled images")

# We could do this in the same for loop we used above but I will make a new one for simplicitys sake

grays = []                           # Creating a container to store all the gray images inside of

# Loop to convert all the images to grayscale
for image in images:                                 # for every image object in images[]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # convert the image to grayscale
    grays.append(gray)                               # and add it to the grays[] container list thingy

# Displaying the gray images
for i in range(len(grays)):
    plt.imshow(grays[i], cmap='gray')                # the cmap='gray' just changes the way the matlab ploting function displays the image.
    plt.axis('off')                                  # it would be blue-yellow instead if we didn't do that.
    plt.show()





# # # # # # 3. Detect Corners (Use 'cv2.cornerHarris' or 'cv2.goodFeaturesToTrack' or equivalent functions)





print("")                            # Printing a space and a title
print("Step 3")
print("Detect Corners")

# Using cornerHarris() to detect corners. (grayscale image, block size, ksize, k)
# cornerHarris() returns a 2D array the same size of the grayscale image that is basically a heat map of how likely
# it is that any 1 pixel is a corner

# block_size - Size of the neighborhood (window) considered for corner detection.
# Typical values: 2–5
# Effect: Larger values consider more context but may blur fine details. Smaller values are more sensitive but noisier.
block_size = 2

# ksize - Aperture parameter for the Sobel operator used to compute image gradients.
# Typical values: 3 or 5
# Effect: Controls how gradients are calculated. Larger ksize smooths more but may miss sharp corners.
ksize = 3

# k- Typical range: 0.04 to 0.06
# Effect: Controls sensitivity to corners vs edges. Smaller k favors corners more aggressively.
k = 0.04

# Again I could do this all in one loop but we are going to make a new loop just to keep things simple

corners = []                         # Creating object to hold corner data in

# Loop to generate corner heat map for all grayscale images
for gray in grays:                   # for every object in grays[]
    corner = cv2.cornerHarris(gray, block_size, ksize, k)  # perform cornerHarris() using the above set parameters
    corners.append(corner)           # adding the corner heat map to the list of corner heat maps

print("Nothing to show for step 3")





# # # # # # 4. Plot the corners on the image





print("")                             # Printing a space and a section title
print("Step 4")
print("Plotting corners onto image")

masks = []                            # creating a list to store the corner masks in
imgs_with_corners = []                # creating a list to store the edited images with corners marked

# loop to take a corner mask, set each location to 1 or 0 (instead of a float32) and then turn that pixel on the image red
for i in range(len(images)):          # for loop using indexing this time because we need to access indices in multiple lists

    # from https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
    corner_mask = corners[i] > 0.01 * corners[i].max()   # creates a true/false mask of whether or not the algo thinks a pixel is a corner

    masks.append(corner_mask)         # adding the mask to the masks list so we can play with it later if needed
    img = images[i].copy()            # Creating a copy of the images[] list so we can overwrite values without messing up the original data
    img[corner_mask] = [255, 0, 0]    # this line takes the image, and at every pixel where the mask says "true" it changes that color to red
    imgs_with_corners.append(img)     # Adding the edited image to our list of edited images