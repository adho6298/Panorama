
# LIBRARIES
# importing required libraries
import matplotlib.pyplot as plt             # for plotting images
import matplotlib.image as mpimg            # for reading in images   
from scipy import ndimage                   # for image processing
import numpy as np                          # for numerical processing   
import cv2                                  # for computer vision tasks
import os                                   # for file path operations
import glob                                 # for file pattern matching
from scipy.ndimage import maximum_filter    # importing this library so we can call scipy.ndimage.filters.maximum_filter as simply: maximum_filter()

''' 
FUNCTIONS AND UTILITIES 
'''

''' Display Function for plotting an image using cv2.imshow '''
def show_img(img, window_name):               # img = image to be displayed, window_name = name of the window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create a resizable window
    h, w = img.shape[:2]                            # Get image dimensions
    cv2.resizeWindow(window_name, w, h)             # Resize window to image dimensions
    cv2.imshow(window_name, img)                    # Display the image in the window


''' Displays all images in the list in separate resizable windows '''
def show_all_images(image_list, window_name):                     # image_list = list of images to be displayed, window_name = name of the window
    for idx, img in enumerate(image_list):                              # Looping through the list of images with index
        show_img(img, window_name=f"{window_name} - Image {idx+1}")     # Displaying each image with its index in the window name
    cv2.waitKey(0)                                                      # Wait for a key press to close the windows
    cv2.destroyAllWindows()                                             # Close all OpenCV windows


''' Function to scale an image by a given factor '''
def scale_img(image_list, scale):             # image_list = list of images to be scaled, scale = scaling factor (0 < scale <= 1)
    scaled_images = []                              # Creating an empty list to store the scaled images
    for img in image_list:                          # Looping through the list of images
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)       # Scaling the image using cv2.resize()
        scaled_images.append(resized)               # Adding the scaled image to the list
    return scaled_images                            # Returning the list of scaled images


''' Function to import images from a given folder '''
def load_images_from_folder(folder_path):                     # folder_path = path to the folder containing images
    image_list = []                                                 # Creating an empty list to store the images
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')    # Supported image file extensions (only grabbing images)
    for ext in extensions:                                          # Looping through the list of extensions
        search_pattern = os.path.join(folder_path, ext)             # Creating the search pattern to find images in "folder_path" with extension "ext"
        print(f"Searching for: {search_pattern}")                   # Debug print to show what pattern we are searching for
        for filename in glob.glob(search_pattern):                  # Looping through the list of files matching the search pattern
            print(f"Found file: {filename}")                        # Debug print to show each file found
            img = cv2.imread(filename)                              # Reading the image using cv2.imread()
            if img is not None:                                     # Checking if the image was read successfully
                image_list.append(img)                              # Adding the image to the list
    return image_list                                               # Returning the list of images

''' Function to convert a list of images from color to grayscale '''
def convert_to_gray(image_list):                  # image_list = list of color images to be converted
    gray_images = []                                    # Creating an empty list to store the grayscale images
    for img in image_list:                              # Looping through the list of images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Converting the image to grayscale using cv2.cvtColor()
        gray_images.append(gray)                        # Adding the grayscale image to the list
    return gray_images                                  # Returning the list of grayscale images

''' Function to detect corners in a list of grayscale images '''
def detect_corners(image_list):             # image_list = list of grayscale images to detect corners in
    # Using cornerHarris() to detect corners. (grayscale image, block size, ksize, k)
    # cornerHarris() returns a 2D array the same size of the grayscale image that is basically a heat map of how likely
    # it is that any 1 pixel is a corner

    # block_size - Size of the neighborhood (window) considered for corner detection.
    # Typical values: 2–5
    # Effect: Larger values consider more context but may blur fine details. Smaller values are more sensitive but noisier.v
    block_size = 2

    # ksize - Aperture parameter for the Sobel operator used to compute image gradients.
    # Typical values: 3 or 5
    # Effect: Controls how gradients are calculated. Larger ksize smooths more but may miss sharp corners.
    ksize = 3

    # k- Typical range: 0.04 to 0.06
    # Effect: Controls sensitivity to corners vs edges. Smaller k favors corners more aggressively.
    k = 0.04

    corners = []                         # Creating object to hold corner data in

    for img in image_list:                                      # Looping through the list of images
        corner = cv2.cornerHarris(img, block_size, ksize, k)    # Perform cornerHarris() using the above set parameters
        corners.append(corner)                                  # Adding the corner heat map to the list of corner heat maps
    
    masks = []                            # creating a list to store the corner masks in
    imgs_with_corners = []                # creating a list to store the edited images with corners marked

    # loop to take a corner mask, set each location to 1 or 0 (instead of a float32) and then turn that pixel on the image red
    for i in range(len(image_list)):          # for loop using indexing this time because we need to access indices in multiple lists

        # from https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
        corner_mask = corners[i] > 0.01 * corners[i].max()   # creates a true/false mask of whether or not the algo thinks a pixel is a corner

        masks.append(corner_mask)         # Adding the mask to the masks list so we can play with it later if needed

        img_gray = image_list[i]            # Getting the grayscale image to draw on
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        img_bgr[corner_mask] = [0, 0, 255]  # Set corner pixels to red in BGR format
        imgs_with_corners.append(img_bgr)           # Adding the edited image to the list
        
    return corners, masks, imgs_with_corners        # Returning the list of corner heat maps, masks, and images with corners marked



'''
ANMS Algorithm:
Input: Corner Response Map, N_Strong Corner Set, nBest Number of Best Corners
Output: N_Best Corners Set
'''


# cmap = corners[]
# nBest = pick something between 200–500

def anms(cmap, nBest):

  # Find all local maxima
  local_max = maximum_filter(cmap, size=3)   # compares a pixel to its surrounding pixels and changes its value to the highest of the group
  local_max_mask = (cmap == local_max)       # creates a 2D array of true/false values telling us if a pixel was changed or not ==> is the local max or not

  # Find (x, y) coordinates of all local maxima
  max_coords = np.argwhere(local_max_mask)   # returns the indices of all True values i.e., the positions of local maxima. (x, y) values
  max_coords = max_coords[:, [1, 0]]         # Convert (row y, col x) to (x, y) - idk how standard this syntax is in python
  # could do it in one line with
  # max_coords = np.argwhere(local_max_mask)[:, [1, 0]]

  # Filtering just the top n number of scores to avoid making 97200*97200 comparisons and taking a million years
  n = 1000                                                # maximum number of strongest corner candidates to consider
  scores = np.array([cmap[y, x] for x, y in max_coords])  # extract corner strength values from cmap at each (x, y) local maximum coordinate
  top_k = min(n, len(max_coords))                         # ensure we don't exceed available candidates if fewer than n maxima exist
  top_indices = np.argsort(-scores)[:top_k]               # get indices of top_k strongest corners by sorting scores in descending order
  max_coords = max_coords[top_indices]                    # keep only the top_k strongest corner coordinates for ANMS

  # initalize r_i = inf for i = [1:N_Strong]
  radii = np.full(len(max_coords), np.inf)           # radii = an array of [inf inf inf ...] for every coord pair

  for i in range(len(max_coords)):                   # Loop over each candidate corner i
    xi, yi = max_coords[i]                           # Get coordinates (x_i, y_i) of corner i
    for j in range(len(max_coords)):                 # Loop over each other candidate corner j
        xj, yj = max_coords[j]                       # Get coordinates (x_j, y_j) of corner j
        if cmap[yj, xj] > cmap[yi, xi]:              # If corner j has a stronger score than corner i
            dist_sq = (xi - xj)**2 + (yi - yj)**2    # Compute squared Euclidean distance between i and j
            if dist_sq < radii[i]:                   # If this distance is smaller than current r_i
                radii[i] = dist_sq                   # Update r_i to the new minimum distance

  # Sort radii in decending order and pick top nBest points
  # Sort suppression radii in descending order to prioritize corners that are both strong and spatially isolated
  sorted_indices = np.argsort(-radii)                # get indices that would sort radii from largest to smallest (negative flips to descending)

  # Select the coordinates of the top nBest corners based on their suppression radii
  N_Best = max_coords[sorted_indices[:nBest]]        # keep only the top nBest corners with the largest radii

  return N_Best, max_coords


def plot_anms_results(images, corners, nBest=500):
    """
    Returns a list of images with ANMS-selected corners (green) and N_Strong corners (red) drawn.
    Each image will have both sets of corners marked for visual comparison.
    """
    images_with_corners = []

    for idx in range(len(images)):
        print(f"\nProcessing image {idx + 1}/{len(images)}")
        # Run ANMS to get N_Best corners
        coordsTemp, max_coords = anms(corners[idx], nBest)

        # Create a copy of the image to draw on
        img_out = images[idx].copy()

        # Draw N_Strong candidates (top scores before ANMS) in red
        for x, y in max_coords:
            cv2.circle(img_out, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

        # Draw N_Best corners (after ANMS) in green
        for x, y in coordsTemp:
            cv2.circle(img_out, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

        images_with_corners.append(img_out)

    return images_with_corners


'''
Feature Descriptor:
Input: Corner Coordinates, Image
Output: Feature Descriptors
'''
def feature_desc(gray_img, nBest):

  desc = [] # list to store feature descriptors
  des_coords = [] # list to store coordinates of valid descriptors

  h, w = gray_img.shape # getting image dimensions

  # Generate Feature Descriptor For Corners Here
  # 1. Define Patch Size, Blur Kernels and Subsample Size
  patch_size = 40
  blur_kernel = (5,5)  # kernal for gaussian blur. try (3,3) or (7,7)
  subsample_size = 8

  # 2. Extract the Patch Centered Around Corners (Ensure the Patch is within the image boundaries)
  for x, y in nBest:                          # At every coord in nBest
    half_patch = patch_size // 2              # calculate half of patch size
    x1, x2 = x - half_patch, x + half_patch   # set min x and max x values on either side of x
    y1, y2 = y - half_patch, y + half_patch   # set min y and max y values above and below y
    # basically taking a coord and building a square of size patch_size around it

    # checking to make sure the bounding box of the patch size is within range of the image
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:  # if any of the edge pixels are out of range
      continue                                # skip the rest of this for loop and move on to the next coord in nBest

    # 3. Apply Gaussian Blur and Subsample
    patch = gray_img[y1:y2, x1:x2]                      # extracting the image data from only within the bounds of the patch
    blurred = cv2.GaussianBlur(patch, blur_kernel, 0)   # applying the gaussian blur (0 = autocalculate sigmaX)

    subsampled = cv2.resize(blurred, (subsample_size, subsample_size), interpolation=cv2.INTER_AREA)
    # Subsampling buy resizing the blurred image from 40x40 to 8x8

    # 4. Reshape Subsampled patch to a 64x1 vector and standardize the descriptor
    vector = subsampled.flatten().astype(np.float32)   # Reshaping the [8x8] into a [64x1]
    mean, std = np.mean(vector), np.std(vector)        # Calculating mean and standard deviation of the vector
    if std == 0:                                       # skipping subsample if std = 0 cause we can't divide by 0 (and it means the whole subsample is flat)
      continue

    standardized = (vector - mean) / std               # making a new vector of normalized values

    # Return Descriptors and their Coordinates
    desc.append(standardized)    # storing results
    des_coords.append((x,y))

  return desc, des_coords


# def plot_feature_desc(grays, selected_coords):
#     """
#     Returns a list of images with feature descriptor points drawn (in blue),
#     compatible with show_all_images (expects a list of np.ndarray images).
#     """
#     for i in range(len(grays)):
#         feat_desc, feat_des_coords = feature_desc(grays[i], selected_coords[i])
#         plt.imshow(feat_desc, cmap = "gray")
#         plt.axis("off")
#         plt.show()
#     return  

def plot_feature_desc(grays, selected_coords):
    """
    Plots each feature descriptor visualization in a separate figure with a title.
    All figures are shown at once (non-blocking).
    """
    for i in range(len(grays)):
        feat_desc, _ = feature_desc(grays[i], selected_coords[i])
        plt.figure(figsize=(5, 5))
        plt.imshow(feat_desc, cmap="gray")
        plt.suptitle("Feature Descriptors")
        plt.title(f"Image: {i+1}")
        plt.axis("off")
        plt.show(block=False)
    plt.show()  # Keeps all figures open until you close them manually
    return

