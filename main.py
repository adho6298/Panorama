from utils import *         # Loading in all of our libraries and utility functions


''' 
Part 1 - Inputing Images
'''


''' Load images from the specified folder. ONLY UNCOMMENT ONE OF THE FOLLOWING LINES '''
image_list = load_images_from_folder('Images/VictoriaLibrary')
# image_list = load_images_from_folder('Images/Indoors')
# image_list = load_images_from_folder('Images/Flatirons')
# image_list = load_images_from_folder('Images/CULogo')
# image_list = load_images_from_folder('Images/CUBoulderSatView')
# image_list = load_images_from_folder('Images/Checkerboard')
#show_all_images(image_list, "Source Images")        # Displaying all of the source images

'''Scaling down the images to make processing faster '''
scale_images = [img.copy() for img in image_list]   # Creating a copy of the original list for processing
scale = 0.6                                         # Scaling factor (0 < scale <= 1)
scale_list = scale_img(image_list, scale)           # Scaling all of the images
#show_all_images(scale_list, "Scaled Images")        # Displaying all of the scaled images

''' Converting to grayscale to make corner detection easier '''
gray_images = [img.copy() for img in scale_list]    # Creating a copy of the scaled list for processing
gray_images = convert_to_gray(gray_images)          # Converting all of the images to grayscale
#show_all_images(gray_images, "Gray Images")         # Displaying all of the gray images


'''
Part 2 - Detect Corners
'''


''' Detecting corners in all of the grayscale images '''
corners, masks, imgs_with_corners = detect_corners(gray_images)  # Detecting corners in all of the grayscale images
#show_all_images(imgs_with_corners, "Corners Detected")           # Displaying all of the images with corners marked in red


'''
Part 3 - ANMS
'''


''' Applying ANMS to the detected corners in all of the grayscale images '''
anms_coords_list = []  # Store ANMS coordinates for each image
for idx in range(len(corners)):
    anms_coords, _ = anms(corners[idx], nBest=500)
    anms_coords_list.append(anms_coords)

images_with_anms = plot_anms_results(scale_list, corners)
#show_all_images(images_with_anms, "ANMS Selected Corners")


'''
Part 4 - Feature Descriptors
'''


''' Computing feature descriptors for all of the grayscale images '''
feature_desc_list, feature_coords_list = plot_feature_desc(gray_images, anms_coords_list)  # Computing feature descriptors for all of the images


'''
Part 5 Feature Matching
'''


''' Matching features between all consecutive image pairs and displaying the matches '''
match_images, matches_list = show_all_feature_matches(scale_list, feature_desc_list, feature_coords_list)
#show_all_images(match_images, "Feature Matches")


'''
Part 6 - Refine Matches
'''

inlier_images, homographies = inliers(match_images, matches_list, scale_list, feature_coords_list)  # Refine matches and compute homographies between all consecutive image pairs using RANSAC
# Show all inlier match images
#show_all_images(inlier_images, "Inlier Feature Matches (RANSAC)")

'''
Part 7 - Cylindrical Warping
'''

# Example intrinsic matrix (you should set fx, fy, cx, cy based on your camera or image size)
h, w = scale_list[0].shape[:2]
fx = fy = 700  # Focal length in pixels (adjust as needed)
cx = w // 2
cy = h // 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float32)

# Apply cylindrical warping to all images
cylindrical_images = [cylindrical_warp(img, K) for img in scale_list]
#show_all_images(cylindrical_images, "Cylindrical Warped Images")

'''
Part 8 - Warping Images with Homographies
'''


# --- New: Compute global homographies for all images ---
global_homographies = [np.eye(3, dtype=np.float32)]
for i in range(1, len(cylindrical_images)):
    if homographies[i-1] is not None:
        global_H = global_homographies[i-1] @ homographies[i-1]
        global_homographies.append(global_H)
    else:
        global_homographies.append(global_homographies[i-1])

# --- Warp all images into the global panorama coordinate frame ---
warped_images = []
offsets = []
all_corners = []
for img, H in zip(cylindrical_images, global_homographies):
    h, w = img.shape[:2]
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    all_corners.append(warped_corners)

# Compute global bounding box
all_corners_concat = np.vstack(all_corners)
min_x, min_y = np.floor(all_corners_concat.min(axis=0)).astype(int)
max_x, max_y = np.ceil(all_corners_concat.max(axis=0)).astype(int)
pano_w = max_x - min_x
pano_h = max_y - min_y

# Warp each image and record its offset
for img, H in zip(cylindrical_images, global_homographies):
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
    H_translated = translation @ H
    warped_img = cv2.warpPerspective(img, H_translated, (pano_w, pano_h))
    warped_images.append(warped_img)
    offsets.append((0, 0))  # All images are now in the same canvas

#show_all_images(warped_images, "Warped Images (Homography)")

'''
Part 9 - Blending to Create Panorama
'''

# Initialize the panorama canvas
panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

# Blend each warped image into the panorama (all are already aligned)
for img in warped_images:
    panorama = blend_warped_images(panorama, img, offset=(0, 0))

# Show the final panorama
show_all_images([panorama], "Final Panorama")