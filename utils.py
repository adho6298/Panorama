
''' LIBRARIES '''
# importing required libraries
import matplotlib.pyplot as plt             # for plotting images
import matplotlib.image as mpimg            # for reading in images   
from scipy import ndimage                   # for image processing
import numpy as np                          # for numerical processing   
import cv2                                  # for computer vision tasks
import os                                   # for file path operations
import glob                                 # for file pattern matching
import inspect                              # Import inspect to access the calling frame
from scipy.ndimage import maximum_filter    # importing this library so we can call scipy.ndimage.filters.maximum_filter as simply: maximum_filter()



''' 
FUNCTIONS AND UTILITIES 
'''



''' Display Function for plotting an image using cv2.imshow '''
def show_img(img, window_name):               # img = image to be displayed, window_name = name of the window

    h, w = img.shape[:2]                            # Get image dimensions
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create a resizable window
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
  local_max = maximum_filter(cmap, size=3)                  # compares a pixel to its surrounding pixels and changes its value to the highest of the group
  local_max_mask = (cmap == local_max)                      # creates a 2D array of true/false values telling us if a pixel was changed or not ==> is the local max or not

  # Find (x, y) coordinates of all local maxima
  max_coords = np.argwhere(local_max_mask)                  # returns the indices of all True values i.e., the positions of local maxima. (x, y) values
  max_coords = max_coords[:, [1, 0]]                        # Convert (row y, col x) to (x, y) - idk how standard this syntax is in python
  # could do it in one line with
  # max_coords = np.argwhere(local_max_mask)[:, [1, 0]]

  # Filtering just the top n number of scores to avoid making 97200*97200 comparisons and taking a million years
  n = 1000                                                  # maximum number of strongest corner candidates to consider
  
  scores = np.array([cmap[y, x] for x, y in max_coords])    # extract corner strength values from cmap at each (x, y) local maximum coordinate
  top_k = min(n, len(max_coords))                           # ensure we don't exceed available candidates if fewer than n maxima exist
  top_indices = np.argsort(-scores)[:top_k]                 # get indices of top_k strongest corners by sorting scores in descending order
  max_coords = max_coords[top_indices]                      # keep only the top_k strongest corner coordinates for ANMS
  
  # initalize r_i = inf for i = [1:N_Strong]
  radii = np.full(len(max_coords), np.inf)                  # radii = an array of [inf inf inf ...] for every coord pair
  
  for i in range(len(max_coords)):                          # Loop over each candidate corner i
    xi, yi = max_coords[i]                                  # Get coordinates (x_i, y_i) of corner i
  
    for j in range(len(max_coords)):                        # Loop over each other candidate corner j
        xj, yj = max_coords[j]                              # Get coordinates (x_j, y_j) of corner j
  
        if cmap[yj, xj] > cmap[yi, xi]:                     # If corner j has a stronger score than corner i
            dist_sq = (xi - xj)**2 + (yi - yj)**2           # Compute squared Euclidean distance between i and j
  
            if dist_sq < radii[i]:                          # If this distance is smaller than current r_i
                radii[i] = dist_sq                          # Update r_i to the new minimum distance
  
  # Sort radii in decending order and pick top nBest points
  # Sort suppression radii in descending order to prioritize corners that are both strong and spatially isolated
  sorted_indices = np.argsort(-radii)                       # get indices that would sort radii from largest to smallest (negative flips to descending)
  
  # Select the coordinates of the top nBest corners based on their suppression radii
  N_Best = max_coords[sorted_indices[:nBest]]               # keep only the top nBest corners with the largest radii
  
  return N_Best, max_coords                                 # returning the coordinates of the N_Best corners and the original max_coords for plotting



''' Function to plot ANMS results on images'''
def plot_anms_results(images, corners, nBest=500): # images = list of original images, corners = list of corner heat maps, nBest = number of best corners to select (default 500)
    
    images_with_corners = []                                        # list to store images with corners drawn
    
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



''' Function to compute feature descriptors for corners in a grayscale image '''
def feature_desc(gray_img, nBest):       # gray_img = grayscale image, nBest = list of (x, y) coordinates of corners to compute descriptors for

  desc = []                                             # list to store feature descriptors
  des_coords = []                                       # list to store coordinates of valid descriptors
  h, w = gray_img.shape                                 # getting image dimensions

  # Generate Feature Descriptor For Corners Here
  # 1. Define Patch Size, Blur Kernels and Subsample Size
  patch_size = 40
  blur_kernel = (5,5)                                   # kernal for gaussian blur. try (3,3) or (7,7)
  subsample_size = 8

  # 2. Extract the Patch Centered Around Corners (Ensure the Patch is within the image boundaries)
  for x, y in nBest:                                    # At every coord in nBest
    half_patch = patch_size // 2                        # calculate half of patch size
    x1, x2 = x - half_patch, x + half_patch             # set min x and max x values on either side of x
    y1, y2 = y - half_patch, y + half_patch             # set min y and max y values above and below y
    # basically taking a coord and building a square of size patch_size around it

    # checking to make sure the bounding box of the patch size is within range of the image
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:            # if any of the edge pixels are out of range
      continue                                          # skip the rest of this for loop and move on to the next coord in nBest

    # 3. Apply Gaussian Blur and Subsample
    patch = gray_img[y1:y2, x1:x2]                      # extracting the image data from only within the bounds of the patch
    blurred = cv2.GaussianBlur(patch, blur_kernel, 0)   # applying the gaussian blur (0 = autocalculate sigmaX)

    subsampled = cv2.resize(blurred, (subsample_size, subsample_size), interpolation=cv2.INTER_AREA)
    # Subsampling buy resizing the blurred image from 40x40 to 8x8

    # 4. Reshape Subsampled patch to a 64x1 vector and standardize the descriptor
    vector = subsampled.flatten().astype(np.float32)    # Reshaping the [8x8] into a [64x1]
    mean, std = np.mean(vector), np.std(vector)         # Calculating mean and standard deviation of the vector

    if std == 0:                                        # skipping subsample if std = 0 cause we can't divide by 0 (and it means the whole subsample is flat)
      continue

    standardized = (vector - mean) / std                # making a new vector of normalized values

    # Return Descriptors and their Coordinates
    desc.append(standardized)                           # storing results
    des_coords.append((x,y))

  return desc, des_coords                        # returning the list of feature descriptors and their coordinates


''' Function to plot feature descriptors for a list of grayscale images and their selected corner coordinates '''
def plot_feature_desc(grays, selected_coords):  # grays = list of grayscale images, selected_coords = list of lists of (x, y) coordinates of corners for each image

    feature_desc_list = []                                                  # Creating a list to store feature descriptors
    feature_coords_list = []                                                # Creating a list to store feature coordinates

    for i in range(len(grays)):                                             # Looping through each grayscale image
        feat_desc, coords = feature_desc(grays[i], selected_coords[i])      # Getting the feature descriptors and their coordinates

        plt.figure(figsize=(5, 5))                                          # Creating a new figure for each image
        plt.imshow(feat_desc, cmap="gray")                                  # Displaying the feature descriptors as a grayscale image
        plt.suptitle("Feature Descriptors")                                 # Adding a title to the figure
        plt.title(f"Image: {i+1}")                                          # Adding a subtitle with the image index
        plt.axis("off")                                                     # Turning off the axis
        plt.show(block=False)                                               # Displaying the figure without blocking the code execution

        feature_desc_list.append(feat_desc)                                 # Adding the feature descriptors to the list
        feature_coords_list.append(coords)                                  # Adding the feature coordinates to the list

    plt.waitforbuttonpress()                                                # Wait for a key or mouse button press in any figure
    plt.close('all')                                                        # Close all open figure windows

    return feature_desc_list, feature_coords_list              # Returning the list of feature descriptors and their coordinates

           
''' Function to match feature descriptors between two images using SSD and Lowe's ratio test '''
def match_features(feature_desc_list, coords_list, img1_idx=0, img2_idx=1, ratio_thresh=0.8): # feature_desc_list = list of lists of 64-d feature descriptors for each image, coords_list = list of lists of (x, y) coordinates for each descriptor in each image, img1_idx = index of the first image in the list, img2_idx = index of the second image in the list, ratio_thresh = threshold for Lowe's ratio test (default 0.8)
 
    desc1 = feature_desc_list[img1_idx]                         # Get the list of descriptors for image 1
    desc2 = feature_desc_list[img2_idx]                         # Get the list of descriptors for image 2
    coords1 = coords_list[img1_idx]                             # Get the list of coordinates for image 1
    coords2 = coords_list[img2_idx]                             # Get the list of coordinates for image 2
    matches = []                                                # Create an empty list to store the matches

    for i, d1 in enumerate(desc1):                              # Loop through each descriptor in image 1 (with its index)
        # Compute SSD (Sum of Squared Differences) between d1 and every descriptor in image 2
        ssds = [np.sum((d1 - d2) ** 2) for d2 in desc2]         # List comprehension to calculate SSD for all descriptors in image 2

        if len(ssds) < 2:                                       # If there are fewer than 2 descriptors in image 2
            continue                                            # Skip this descriptor (cannot apply ratio test)

        sorted_idx = np.argsort(ssds)                           # Get the indices that would sort the SSDs in ascending order (best match first)
        best_idx = sorted_idx[0]                                # Index of the best match (lowest SSD)
        second_idx = sorted_idx[1]                              # Index of the second-best match (second lowest SSD)
        best_ssd = ssds[best_idx]                               # Value of the best SSD
        second_ssd = ssds[second_idx]                           # Value of the second-best SSD

        # Apply Lowe's ratio test: accept the match if the best SSD is much smaller than the second-best
        if best_ssd / (second_ssd + 1e-10) < ratio_thresh:      # Add small value to denominator to avoid division by zero
            matches.append((coords1[i], coords2[best_idx]))     # Store the coordinates of the matched keypoints

    return matches  # Return the list of matched keypoint coordinate pairs


''' Function to draw feature matches between two images '''
def draw_feature_matches(img1, img2, matches, coords1, coords2):    # img1 = first image, img2 = second image, matches = list of matched keypoint coordinate pairs, coords1 = list of (x, y) coordinates for image 1, coords2 = list of (x, y) coordinates for image 2
  
    img1_color = img1.copy()                                            # Create a copy of img1 to draw on
    img2_color = img2.copy()                                            # Create a copy of img2 to draw on

    # Create a new image by concatenating img1 and img2 horizontally
    h1, w1 = img1_color.shape[:2]                                       # Get dimensions of img1
    h2, w2 = img2_color.shape[:2]                                       # Get dimensions of img2

    out_height = max(h1, h2)                                            # Height of the output image is the max height of the two images
    out_width = w1 + w2                                                 # Width of the output image is the sum of the widths of the two images

    out_img = np.zeros((out_height, out_width, 3), dtype=np.uint8)      # Create a blank output image with 3 color channels (RGB)
    
    out_img[:h1, :w1] = img1_color                                      # Place img1 on the left side of the output image
    out_img[:h2, w1:w1 + w2] = img2_color                               # Place img2 on the right side of the output image

    # Draw matches
    for idx, (pt1, pt2) in enumerate(matches):                          # Loop through each match with its index

        color = tuple(np.random.randint(0, 255, 3).tolist())            # Generate a random color for each match line and circles
        x1, y1 = int(pt1[0]), int(pt1[1])                               # Get coordinates of the keypoint in img1
        x2, y2 = int(pt2[0]), int(pt2[1])                               # Get coordinates of the keypoint in img2

        # Draw circles at keypoints
        cv2.circle(out_img, (x1, y1), 2, color, -1)                     # Circle on img1 keypoint (radius 2)
        cv2.circle(out_img, (x2 + w1, y2), 2, color, -1)                # Circle on img2 keypoint (radius 2, offset x by width of img1)

        # Draw line connecting the keypoints
        cv2.line(out_img, (x1, y1), (x2 + w1, y2), color, 1)            # Drawing line from img1 keypoint to img2 keypoint (thickness 1)

    return out_img                                      # Return the output image with matches drawn



''' Function to match and display features for all consecutive image pairs '''
def show_all_feature_matches(scale_list, feature_desc_list, feature_coords_list, ratio_thresh=0.8, top_n=160):  # scale_list = list of scaled images, feature_desc_list = list of feature descriptors for each image, feature_coords_list = list of feature coordinates for each image, ratio_thresh = ratio test threshold for feature matching (default 0.8), top_n = number of top matches to display per pair (default 160)

    match_images = []                                                   # List to store output images with matches drawn
    matches_list = []                                                   # List to store matched keypoints for each image pair

    for i in range(len(scale_list) - 1):                                # Loop through each consecutive image pair
        # Match features between image i and image i+1
        matches = match_features(feature_desc_list, feature_coords_list, img1_idx=i, img2_idx=i+1, ratio_thresh=ratio_thresh)   
        matches_list.append(matches)                                    # Store the matches for this image pair

        print(f"Image pair {i}-{i+1}: {len(matches)} matches found")
        
        # Only draw the top N matches
        top_matches = matches[:top_n]                                   # Select the top N matches to display
        match_img = draw_feature_matches(                               # draw matches between image i and image i+1
            scale_list[i], scale_list[i+1],                             # Input images
            top_matches,                                                # Matches to draw
            feature_coords_list[i], feature_coords_list[i+1]            # Feature coordinates for both images
        )

        match_images.append(match_img)                                  # Store the output image with matches drawn

    return match_images, matches_list                    # Return the list of match images and the list of matches for each pair



''' RANSAC algorithm to refine matches and compute homography '''
def ransac_homography(matches, max_iter=2000, inlier_thresh=5.0):       # matches = list of matched keypoint coordinate pairs, max_iter = maximum number of RANSAC iterations (default 2000), inlier_thresh = threshold to consider a point an inlier (default 5.0 pixels)

    if len(matches) < 4:                                                                # Need at least 4 matches to compute homography
        raise ValueError("At least 4 matches are required to compute homography.")      # Raise error if not enough matches

    pts1 = np.float32([m[0] for m in matches])                                          # points from image 1
    pts2 = np.float32([m[1] for m in matches])                                          # points from image 2

    best_H = None                                                                       # Best homography matrix
    max_inliers = 0                                                                     # Maximum number of inliers found
    best_inliers = []                                                                   # Best inlier mask

    for _ in range(max_iter):                                                           # RANSAC iterations
        # 1. Randomly select 4 pairs
        idx = np.random.choice(len(matches), 4, replace=False)                          # Randomly select 4 unique indices
        src = pts2[idx]                                                                 # source points from image 2
        dst = pts1[idx]                                                                 # destination points from image 1

        # 2. Estimate homography
        H, status = cv2.findHomography(src, dst, method=0)                              # 0 = regular (not RANSAC)  # Compute homography from src to dst

        if H is None:                                                                   # If homography could not be computed
            continue                                                                    # Skip to the next iteration

        # 3. Apply homography to all keypoints from image 2
        pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))])                       # Convert to homogeneous coordinates (x, y) -> (x, y, 1)
        pts2_proj = (H @ pts2_hom.T).T                                                  # Project points using homography (N, 3) (@ is matrix multiplication)
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2, np.newaxis]                      # Convert back to Cartesian coordinates (N, 2)

        # 4. Compute SSD (actually Euclidean distance) between projected and actual keypoints in image 1
        dists = np.linalg.norm(pts1 - pts2_proj, axis=1)                                # Euclidean distances (N,)
        inlier_mask = dists < inlier_thresh                                             # Boolean mask of inliers
        num_inliers = np.sum(inlier_mask)                                               # Count number of inliers

        # 5. Keep homography with most inliers
        if num_inliers > max_inliers:                                                   # If this model has more inliers than previous best
            max_inliers = num_inliers                                                   # Update max inliers
            best_H = H                                                                  # Update best homography
            best_inliers = inlier_mask                                                  # Update best inlier mask

    # 6. Recompute homography using all inliers
    if best_H is not None and max_inliers >= 4:                                         # If a valid homography was found with enough inliers
        inliers_img1 = pts1[best_inliers]                                               # Inlier points from image 1
        inliers_img2 = pts2[best_inliers]                                               # Inlier points from image 2    
        best_H, _ = cv2.findHomography(inliers_img2, inliers_img1, method=0)            # Recompute homography using all inliers
        
        return best_H, inliers_img1, inliers_img2                                       # Return the best homography and inlier points
    
    else:                                                                               # If no valid homography was found
        raise RuntimeError("RANSAC failed to find a valid homography.")                 # Raise error
    


''' Function to refine matches and compute homographies for all consecutive image pairs using RANSAC '''
def inliers(match_images, matches_list, scale_list, feature_coords_list, top_n=160): # match_images = list of images with all matches drawn, matches_list = list of matched keypoints for each image pair, scale_list = list of scaled images, feature_coords_list = list of feature coordinates for each image, top_n = number of top inlier matches to display per pair (default 160)

    # Refine matches and compute homographies between all consecutive image pairs using RANSAC
    inlier_images = []  # To store images with only inlier matches drawn
    homographies = []   # To store the computed homography matrices

    for i, matches in enumerate(matches_list):                                              # Loop through each image pair's matches
        
        if len(matches) < 4:                                                                # Need at least 4 matches to compute homography
            print(f"Not enough matches for image pair {i}-{i+1} to compute homography.")    # Debug print
            inlier_images.append(match_images[i])                                           # Just show all matches if not enough for RANSAC
            homographies.append(None)                                                       # No homography computed
            continue
        
        try:                                                                                # Try-except block to catch errors in RANSAC
            H, inliers_img1, inliers_img2 = ransac_homography(matches,                      # Run RANSAC to get homography and inliers
                                                              max_iter=2000,                # max iterations
                                                              inlier_thresh=5.0)            # inlier threshold
            homographies.append(H)                                                          # Store the computed homography

            # Prepare inlier match pairs for drawing
            inlier_pairs = list(zip(inliers_img1, inliers_img2))                            # Create list of inlier match pairs
            
            # Only draw the top_n inlier matches
            top_inlier_pairs = inlier_pairs[:top_n]                                         # Select the top N inlier matches to display

            # Draw only inlier matches
            inlier_img = draw_feature_matches(                                              # draw matches between image i and image i+1
                scale_list[i], scale_list[i+1],                                             # Input images
                top_inlier_pairs,                                                           # Matches to draw
                feature_coords_list[i], feature_coords_list[i+1])                           # Feature coordinates for both images
            
            inlier_images.append(inlier_img)                                                # Store the output image with inlier matches drawn

            print(f"Image pair {i}-{i+1}: {len(inlier_pairs)} inliers after RANSAC (showing top {len(top_inlier_pairs)})")  # Debug print
        
        except Exception as e:                                                              # Catch any exceptions raised during RANSAC
            print(f"RANSAC failed for image pair {i}-{i+1}: {e}")                           # Debug print the error message
            inlier_images.append(match_images[i])                                           # Just show all matches if RANSAC fails
            homographies.append(None)                                                       # No homography computed
    
    return inlier_images, homographies                                          # Return the list of inlier images and homographies



''' Function to warp an image into cylindrical coordinates using its intrinsic matrix '''
def cylindrical_warp(img, K):           # img = input image, K = intrinsic matrix of the camera (3x3 numpy array)

    h, w = img.shape[:2]                # Get image dimensions
    fx = K[0, 0]                        # Focal length in x direction
    fy = K[1, 1]                        # Focal length in y direction
    cx = K[0, 2]                        # Principal point x-coordinate
    cy = K[1, 2]                        # Principal point y-coordinate

    cyl_img = np.zeros_like(img)        # Initialize output cylindrical image

    for y_cyl in range(h):               # Loop over each pixel in the output cylindrical image
        for x_cyl in range(w):           # Loop over each pixel in the output cylindrical image
            
            # Convert cylindrical pixel to normalized coordinates
            theta = (x_cyl - cx) / fx       # Angle around the cylinder
            h_ = (y_cyl - cy) / fy          # Height on the cylinder

            # Project to 3D cylinder surface
            X = np.sin(theta)               # X coordinate on cylinder
            Y = h_                          # Y coordinate on cylinder
            Z = np.cos(theta)               # Z coordinate on cylinder

            # Project back to image plane
            x_img = fx * X / Z + cx         # Corresponding x in original image
            y_img = fy * Y / Z + cy         # Corresponding y in original image

            # Bilinear interpolation
            if 0 <= x_img < w - 1 and 0 <= y_img < h - 1:               # Check if the projected pixel is within image bounds
                x0, y0 = int(np.floor(x_img)), int(np.floor(y_img))     # Finding top-left pixel
                x1, y1 = x0 + 1, y0 + 1                                 # Finding bottom-right pixel
                dx, dy = x_img - x0, y_img - y0                         # computing fractional parts (the locations between the pixels where the projected pixel lies. (not really though: "weights"))

                for c in range(img.shape[2]):                           # Loop over each color channel
                    val = (                                             # Bilinear interpolation formula for color images
                        img[y0, x0, c] * (1 - dx) * (1 - dy) +          # Top-left pixel
                        img[y0, x1, c] * dx * (1 - dy) +                # Top-right pixel
                        img[y1, x0, c] * (1 - dx) * dy +                # Bottom-left pixel
                        img[y1, x1, c] * dx * dy)                       # Bottom-right pixel
                    
                    cyl_img[y_cyl, x_cyl, c] = val                      # Assign interpolated value to the cylindrical image

    return cyl_img                     # Return the cylindrical warped image



''' Function to blend two warped images into a panorama '''
def blend_warped_images(img1, img2, offset=(0, 0)):     # img1 = first warped image, img2 = second warped image, offset = (x_offset, y_offset) to place img2 relative to img1

    # Calculate canvas size
    h1, w1 = img1.shape[:2]                                     # Dimensions of first image
    h2, w2 = img2.shape[:2]                                     # Dimensions of second image
    x_offset, y_offset = offset                                 # Unpack offset

    # Determine the size of the output canvas
    pano_w = max(w1, x_offset + w2)                             # Width of panorama
    pano_h = max(h1, y_offset + h2)                             # Height of panorama
    panorama = np.zeros((pano_h, pano_w, 3), dtype=np.float32)  # Initialize panorama canvas

    # Place img1 on the canvas
    panorama[:h1, :w1] = img1.astype(np.float32)                # Place img1 at the top-left corner

    # Feather width in pixels
    feather_width = 10                                          # Width of the feathering region

    # Create mask for img2 (nonzero pixels)
    mask2 = np.any(img2 > 0, axis=2).astype(np.uint8)           # Finding all non-black pixels in img2 and making a mask of them

    # Distance transform: distance to nearest zero (edge)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)        # dist2 now contains the distance of each pixel in img2 to the nearest black pixel (edge of content)
    
    # Normalize to [0, 1] in feather region
    feather_mask = np.clip(dist2 / feather_width, 0, 1)         # Normalize distances to [0, 1] within feather width
    feather_mask = feather_mask[..., np.newaxis]                # Expand dimensions for broadcasting (Numpy trick to make mishapen arrays compatible for element-wise operations)

    for y in range(h2):                                         # Loop over each pixel in img2
        for x in range(w2):                                     # Loop over each pixel in img2
            pano_x = x + x_offset                               # Corresponding x in panorama
            pano_y = y + y_offset                               # Corresponding y in panorama
            
            if 0 <= pano_x < pano_w and 0 <= pano_y < pano_h:   # If pixel is within panorama bounds
                if np.any(img2[y, x] > 0):                      # If pixel is not black
                    if np.any(panorama[pano_y, pano_x] > 0):    # If panorama pixel already has content (overlap region)
                        
                        alpha = feather_mask[y, x, 0]           # Feathering weight from mask
                        
                        if alpha < 1.0:                         # If within feather region
                            panorama[pano_y, pano_x] = ((1 - alpha) * panorama[pano_y, pano_x] + alpha * img2[y, x])     # Blend the two images using weighted average of pixel values and feathering weight
                        
                        else:                                       # Else If outside feather region, use img2 pixel directly  
                            panorama[pano_y, pano_x] = img2[y, x]   # Use img2 pixel directly
                    
                    else:                                           # Else If panorama pixel is empty
                        panorama[pano_y, pano_x] = img2[y, x]       # Use img2 pixel directly

    panorama = np.clip(panorama, 0, 255).astype(np.uint8)       # Clip values to valid range and convert to uint8

    return panorama             # Return the blended panorama image