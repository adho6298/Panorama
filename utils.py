
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
    # Effect: Larger values consider more context but may blur fine details. Smaller values are more sensitive but noisier.
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


''' ANMS algorithm to select the best corners from the corner heat map '''
def anms(cmap, nBest):                 # cmap = corner response map (corners[]), nBest = number of best corners to select (pick between 200-500)

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

  return N_Best, max_coords               # returning the coordinates of the N_Best corners and the original max_coords for plotting

''' Function to plot ANMS results on images'''
def plot_anms_results(images, corners, nBest=500):
    images_with_corners = []    # list to store images with corners drawn
    for idx in range(len(images)):                                  # Looping through each image
        print(f"\nProcessing image {idx + 1}/{len(images)}")        # Debug print to show progress
        coordsTemp, max_coords = anms(corners[idx], nBest)          # Applying ANMS to get the best corner coordinates
        img_out = images[idx].copy()                                # Creating a copy of the original image to draw on
        for x, y in max_coords:                                     # Draw all detected corners in red
            cv2.circle(img_out, (x, y), radius=1, color=(0, 0, 255), thickness=-1)  # red circles
        for x, y in coordsTemp:                                     # Draw ANMS selected corners in green
            cv2.circle(img_out, (x, y), radius=2, color=(0, 255, 0), thickness=-1)  # green circles
        images_with_corners.append(img_out)                         # Adding the edited image to the list
    return images_with_corners                                      # returning the list of images with corners drawn


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



def plot_feature_desc(grays, selected_coords):
    """
    Plots each feature descriptor visualization in a separate figure with a title.
    All figures are shown at once (non-blocking).
    """
    feature_desc_list = []
    feature_coords_list = []
    for i in range(len(grays)):
        feat_desc, coords = feature_desc(grays[i], selected_coords[i])
        plt.figure(figsize=(5, 5))
        plt.imshow(feat_desc, cmap="gray")
        plt.suptitle("Feature Descriptors")
        plt.title(f"Image: {i+1}")
        plt.axis("off")
        plt.show(block=False)
        feature_desc_list.append(feat_desc)
        feature_coords_list.append(coords)
    plt.waitforbuttonpress()  # Wait for a key or mouse button press in any figure
    plt.close('all')          # Close all open figure windows
    return feature_desc_list, feature_coords_list

           

def match_features(feature_desc_list, coords_list, img1_idx=0, img2_idx=1, ratio_thresh=0.8):
    """
    Matches feature descriptors between two images using SSD and Lowe's ratio test.

    Args:
        feature_desc_list: List of lists of 64-d feature descriptors for each image.
        coords_list: List of lists of (x, y) coordinates for each descriptor in each image.
        img1_idx: Index of the first image in the list.
        img2_idx: Index of the second image in the list.
        ratio_thresh: Threshold for Lowe's ratio test (default 0.8).

    Returns:
        matches: List of tuples ((x1, y1), (x2, y2)) of matched keypoint coordinates.
    """

    desc1 = feature_desc_list[img1_idx]      # Get the list of descriptors for image 1
    desc2 = feature_desc_list[img2_idx]      # Get the list of descriptors for image 2
    coords1 = coords_list[img1_idx]          # Get the list of coordinates for image 1
    coords2 = coords_list[img2_idx]          # Get the list of coordinates for image 2
    matches = []                             # Create an empty list to store the matches

    for i, d1 in enumerate(desc1):           # Loop through each descriptor in image 1 (with its index)
        # Compute SSD (Sum of Squared Differences) between d1 and every descriptor in image 2
        ssds = [np.sum((d1 - d2) ** 2) for d2 in desc2]  # List comprehension to calculate SSD for all descriptors in image 2

        if len(ssds) < 2:                    # If there are fewer than 2 descriptors in image 2
            continue                         # Skip this descriptor (cannot apply ratio test)

        sorted_idx = np.argsort(ssds)        # Get the indices that would sort the SSDs in ascending order (best match first)
        best_idx = sorted_idx[0]             # Index of the best match (lowest SSD)
        second_idx = sorted_idx[1]           # Index of the second-best match (second lowest SSD)
        best_ssd = ssds[best_idx]            # Value of the best SSD
        second_ssd = ssds[second_idx]        # Value of the second-best SSD

        # Apply Lowe's ratio test: accept the match if the best SSD is much smaller than the second-best
        if best_ssd / (second_ssd + 1e-10) < ratio_thresh:  # Add small value to denominator to avoid division by zero
            matches.append((coords1[i], coords2[best_idx])) # Store the coordinates of the matched keypoints

    return matches  # Return the list of matched keypoint coordinate pairs



def draw_feature_matches(img1, img2, matches, coords1, coords2):
    """
    Returns an image with two images side by side, matching keypoints highlighted in the same color, and lines connecting them.
    Args:
        img1: First image (numpy array, BGR or grayscale).
        img2: Second image (numpy array, BGR or grayscale).
        matches: List of tuples ((x1, y1), (x2, y2)) of matched keypoint coordinates.
        coords1: List of (x, y) coordinates for keypoints in img1.
        coords2: List of (x, y) coordinates for keypoints in img2.
    Returns:
        out_img: The output image with matches drawn.
    """
    # Convert grayscale to BGR for color drawing
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()

    # Create a new image by concatenating img1 and img2 horizontally
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    out_height = max(h1, h2)
    out_width = w1 + w2
    out_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1_color
    out_img[:h2, w1:w1 + w2] = img2_color

    # Draw matches
    for idx, (pt1, pt2) in enumerate(matches):
        # Generate a unique color for each match
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        # Draw circles at keypoints
        cv2.circle(out_img, (x1, y1), 5, color, -1)
        cv2.circle(out_img, (x2 + w1, y2), 5, color, -1)
        # Draw line connecting the keypoints
        cv2.line(out_img, (x1, y1), (x2 + w1, y2), color, 2)

    return out_img


def show_all_feature_matches(scale_list, feature_desc_list, feature_coords_list, ratio_thresh=0.8, top_n=40):
    """
    For all consecutive image pairs, match features and return a list of output images with matches drawn,
    as well as the list of matched keypoints for each pair.
    Args:
        scale_list: List of scaled images.
        feature_desc_list: List of feature descriptors for each image.
        feature_coords_list: List of feature coordinates for each image.
        ratio_thresh: Ratio test threshold for feature matching.
        top_n: Number of top matches to display per pair.
    Returns:
        match_images: List of images with matches drawn for each image pair.
        matches_list: List of matched keypoints for each image pair.
    """
    match_images = []  # List to store output images with matches drawn
    matches_list = []  # List to store matched keypoints for each image pair

    for i in range(len(scale_list) - 1):
        # Match features between image i and image i+1
        matches = match_features(feature_desc_list, feature_coords_list, img1_idx=i, img2_idx=i+1, ratio_thresh=ratio_thresh)
        matches_list.append(matches)
        print(f"Image pair {i}-{i+1}: {len(matches)} matches found")
        # Only draw the top N matches
        top_matches = matches[:top_n]
        match_img = draw_feature_matches(
            scale_list[i], scale_list[i+1],
            top_matches,
            feature_coords_list[i], feature_coords_list[i+1]
        )
        match_images.append(match_img)
    return match_images, matches_list



def ransac_homography(matches, max_iter=2000, inlier_thresh=5.0):
    """
    Refine feature matches using RANSAC and compute a robust homography matrix.

    Args:
        matches: List of tuples ((x1, y1), (x2, y2)) of matched keypoint coordinates.
        max_iter: Number of RANSAC iterations.
        inlier_thresh: Distance threshold (in pixels) to count as an inlier.

    Returns:
        best_H: Best homography matrix (3x3 numpy array).
        inliers_img1: List of inlier keypoints from image 1.
        inliers_img2: List of inlier keypoints from image 2.
    """
    if len(matches) < 4:
        raise ValueError("At least 4 matches are required to compute homography.")

    pts1 = np.float32([m[0] for m in matches])
    pts2 = np.float32([m[1] for m in matches])

    best_H = None
    max_inliers = 0
    best_inliers = []

    for _ in range(max_iter):
        # 1. Randomly select 4 pairs
        idx = np.random.choice(len(matches), 4, replace=False)
        src = pts2[idx]  # points from image 2
        dst = pts1[idx]  # corresponding points from image 1

        # 2. Estimate homography
        H, status = cv2.findHomography(src, dst, method=0)  # 0 = regular (not RANSAC)
        if H is None:
            continue

        # 3. Apply homography to all keypoints from image 2
        pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N, 3)
        pts2_proj = (H @ pts2_hom.T).T  # (N, 3)
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2, np.newaxis]  # Normalize

        # 4. Compute SSD (actually Euclidean distance) between projected and actual keypoints in image 1
        dists = np.linalg.norm(pts1 - pts2_proj, axis=1)
        inlier_mask = dists < inlier_thresh
        num_inliers = np.sum(inlier_mask)

        # 5. Keep homography with most inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inlier_mask

    # 6. Recompute homography using all inliers
    if best_H is not None and max_inliers >= 4:
        inliers_img1 = pts1[best_inliers]
        inliers_img2 = pts2[best_inliers]
        best_H, _ = cv2.findHomography(inliers_img2, inliers_img1, method=0)
        return best_H, inliers_img1, inliers_img2
    else:
        raise RuntimeError("RANSAC failed to find a valid homography.")
    

def inliers(match_images, matches_list, scale_list, feature_coords_list, top_n=40):
    # Refine matches and compute homographies between all consecutive image pairs using RANSAC
    inlier_images = []  # To store images with only inlier matches drawn
    homographies = []   # To store the computed homography matrices

    for i, matches in enumerate(matches_list):
        if len(matches) < 4:
            print(f"Not enough matches for image pair {i}-{i+1} to compute homography.")
            inlier_images.append(match_images[i])  # Just show all matches if not enough for RANSAC
            homographies.append(None)
            continue
        try:
            H, inliers_img1, inliers_img2 = ransac_homography(matches, max_iter=2000, inlier_thresh=5.0)
            homographies.append(H)
            # Prepare inlier match pairs for drawing
            inlier_pairs = list(zip(inliers_img1, inliers_img2))
            # Only draw the top_n inlier matches
            top_inlier_pairs = inlier_pairs[:top_n]
            # Draw only inlier matches
            inlier_img = draw_feature_matches(
                scale_list[i], scale_list[i+1],
                top_inlier_pairs,
                feature_coords_list[i], feature_coords_list[i+1]
            )
            inlier_images.append(inlier_img)
            print(f"Image pair {i}-{i+1}: {len(inlier_pairs)} inliers after RANSAC (showing top {len(top_inlier_pairs)})")
        except Exception as e:
            print(f"RANSAC failed for image pair {i}-{i+1}: {e}")
            inlier_images.append(match_images[i])
            homographies.append(None)
    return inlier_images, homographies


def cylindrical_warp(img, K):
    """
    Warps an image into cylindrical coordinates using its intrinsic matrix.

    Args:
        img: Input image (numpy array, HxWxC or HxW).
        K: Camera intrinsic matrix (3x3 numpy array), where K[0,0] = fx, K[1,1] = fy, K[0,2] = cx, K[1,2] = cy.

    Returns:
        cyl_img: Cylindrically warped image (same size as input).
    """
    h, w = img.shape[:2]
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    cyl_img = np.zeros_like(img)
    for y_cyl in range(h):
        for x_cyl in range(w):
            # Convert cylindrical pixel to normalized coordinates
            theta = (x_cyl - cx) / fx
            h_ = (y_cyl - cy) / fy

            # Project to 3D cylinder surface
            X = np.sin(theta)
            Y = h_
            Z = np.cos(theta)

            # Project back to image plane
            x_img = fx * X / Z + cx
            y_img = fy * Y / Z + cy

            # Bilinear interpolation
            if 0 <= x_img < w - 1 and 0 <= y_img < h - 1:
                x0, y0 = int(np.floor(x_img)), int(np.floor(y_img))
                x1, y1 = x0 + 1, y0 + 1
                dx, dy = x_img - x0, y_img - y0

                if img.ndim == 2:
                    val = (
                        img[y0, x0] * (1 - dx) * (1 - dy) +
                        img[y0, x1] * dx * (1 - dy) +
                        img[y1, x0] * (1 - dx) * dy +
                        img[y1, x1] * dx * dy
                    )
                    cyl_img[y_cyl, x_cyl] = val
                else:
                    for c in range(img.shape[2]):
                        val = (
                            img[y0, x0, c] * (1 - dx) * (1 - dy) +
                            img[y0, x1, c] * dx * (1 - dy) +
                            img[y1, x0, c] * (1 - dx) * dy +
                            img[y1, x1, c] * dx * dy
                        )
                        cyl_img[y_cyl, x_cyl, c] = val
    return cyl_img

def warp_image_with_homography(img, H, output_shape=None):
    """
    Warps an image using a homography matrix, expanding the canvas to fit both the reference and warped image content.

    Args:
        img: Input image (numpy array).
        H: Homography matrix (3x3 numpy array).
        ref_shape: (height, width) tuple for the reference image (the panorama so far).
        ref_offset: (x_offset, y_offset) tuple for the reference image's top-left corner in the panorama.
    Returns:
        warped_img: The warped image, placed in the panorama canvas.
        offset: (x_offset, y_offset) translation applied to the output image.
        out_shape: (width, height) of the panorama canvas.
    """
    def get_corners(shape, offset=(0,0)):
        h, w = shape[:2]
        x_off, y_off = offset
        return np.array([
            [x_off, y_off],
            [x_off + w, y_off],
            [x_off + w, y_off + h],
            [x_off, y_off + h]
        ], dtype=np.float32).reshape(-1, 1, 2)

    h, w = img.shape[:2]
    # We'll require ref_shape and ref_offset as arguments
    # For backward compatibility, if not provided, use (h, w) and (0,0)
    import inspect
    frame = inspect.currentframe().f_back
    ref_shape = frame.f_locals.get('ref_shape', (h, w))
    ref_offset = frame.f_locals.get('ref_offset', (0, 0))

    # Corners of the reference image (already in panorama coordinates)
    ref_corners = get_corners(ref_shape, ref_offset)
    # Corners of the new image, warped by H
    img_corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(img_corners, H)

    # Combine all corners to find the bounding box
    all_corners = np.vstack((ref_corners, warped_corners)).reshape(-1, 2)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

    out_w = x_max - x_min
    out_h = y_max - y_min

    # Compute translation to keep all pixels in view
    x_offset = -x_min
    y_offset = -y_min

    # Adjust homography to include translation
    translation = np.array([[1, 0, x_offset],
                            [0, 1, y_offset],
                            [0, 0, 1]], dtype=np.float32)
    H_translated = translation @ H

    warped_img = cv2.warpPerspective(img, H_translated, (out_w, out_h))
    return warped_img, (x_offset, y_offset), (out_w, out_h)


def blend_warped_images(img1, img2, offset=(0, 0)):
    """
    Blends two warped images into a single panorama using weighted averaging in overlapping regions.

    Args:
        img1: First warped image (numpy array).
        img2: Second warped image (numpy array).
        offset: (x_offset, y_offset) tuple indicating where to place img2 relative to img1.

    Returns:
        panorama: The blended panorama image.
    """
    # Calculate canvas size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    x_offset, y_offset = offset

    # Determine the size of the output canvas
    pano_w = max(w1, x_offset + w2)
    pano_h = max(h1, y_offset + h2)
    panorama = np.zeros((pano_h, pano_w, 3), dtype=np.float32)

    # Place img1 on the canvas
    panorama[:h1, :w1] = img1.astype(np.float32)

    # Feather width in pixels
    feather_width = 10

    # Create mask for img2 (nonzero pixels)
    mask2 = np.any(img2 > 0, axis=2).astype(np.uint8)
    # Distance transform: distance to nearest zero (edge)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)
    # Normalize to [0, 1] in feather region
    feather_mask = np.clip(dist2 / feather_width, 0, 1)
    feather_mask = feather_mask[..., np.newaxis]  # shape (h2, w2, 1)

    for y in range(h2):
        for x in range(w2):
            pano_x = x + x_offset
            pano_y = y + y_offset
            if 0 <= pano_x < pano_w and 0 <= pano_y < pano_h:
                if np.any(img2[y, x] > 0):
                    if np.any(panorama[pano_y, pano_x] > 0):
                        # Overlap: use feathered blend only in feather region, else use img2 (top-most)
                        alpha = feather_mask[y, x, 0]
                        if alpha < 1.0:
                            panorama[pano_y, pano_x] = (
                                (1 - alpha) * panorama[pano_y, pano_x] + alpha * img2[y, x]
                            )
                        else:
                            panorama[pano_y, pano_x] = img2[y, x]
                    else:
                        # Only img2 has content
                        panorama[pano_y, pano_x] = img2[y, x]

    # Convert back to uint8
    panorama = np.clip(panorama, 0, 255).astype(np.uint8)
    return panorama