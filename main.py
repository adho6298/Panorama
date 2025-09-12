from utils import *         # Loading in all of our libraries and utility functions

''' 
Part A - Inputing Images
'''

''' Load images from the specified folder. ONLY UNCOMMENT ONE OF THE FOLLOWING LINES '''
image_list = load_images_from_folder('Panorama/Images/VictoriaLibrary')
# image_list = load_images_from_folder('Panorama/Images/Indoors')
# image_list = load_images_from_folder('Panorama/Images/Flatirons')
# image_list = load_images_from_folder('Panorama/Images/CULogo')
# image_list = load_images_from_folder('Panorama/Images/CUBoulderSatView')
# image_list = load_images_from_folder('Panorama/Images/Checkerboard')
show_all_images(image_list, "Source Images")        # Displaying all of the source images

'''Scaling down the images to make processing faster '''
scale_images = [img.copy() for img in image_list]   # Creating a copy of the original list for processing
scale = 0.6                                         # Scaling factor (0 < scale <= 1)
scale_list = scale_img(image_list, scale)           # Scaling all of the images
show_all_images(scale_list, "Scaled Images")        # Displaying all of the scaled images

''' Converting to grayscale to make corner detection easier '''
gray_images = [img.copy() for img in scale_list]    # Creating a copy of the scaled list for processing
gray_images = convert_to_gray(gray_images)          # Converting all of the images to grayscale
show_all_images(gray_images, "Gray Images")         # Displaying all of the gray images

'''
Part B - Detect Corners
'''

''' Detecting corners in all of the grayscale images '''
corners, masks, imgs_with_corners = detect_corners(gray_images)  # Detecting corners in all of the grayscale images
show_all_images(imgs_with_corners, "Corners Detected")           # Displaying all of the images with corners marked in red

'''
Part C - ANMS
'''

''' Applying ANMS to the detected corners in all of the grayscale images '''
anms_coords_list = []  # Store ANMS coordinates for each image
for idx in range(len(corners)):
    anms_coords, _ = anms(corners[idx], nBest=500)
    anms_coords_list.append(anms_coords)

images_with_anms = plot_anms_results(scale_list, corners)
show_all_images(images_with_anms, "ANMS Selected Corners")

'''
Part D - Feature Descriptors
'''

plot_feature_desc(gray_images, anms_coords_list)  # Computing feature descriptors for all of the images