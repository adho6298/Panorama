'''
Part 9 - Blending to Create Panorama
'''

# Step 1: Compute global offsets for each warped image (relative to panorama origin)
global_offsets = [(0, 0)]
current_offset = np.array([0, 0], dtype=int)
for i in range(1, len(offsets)):
    current_offset = current_offset + np.array(offsets[i])
    global_offsets.append(tuple(current_offset))

# Step 2: Find minimum x and y offsets (could be negative)
xs, ys = zip(*global_offsets)
min_x = min(xs)
min_y = min(ys)

# Step 3: Shift all offsets so that the panorama starts at (0, 0)
shifted_offsets = [(x - min_x, y - min_y) for (x, y) in global_offsets]

# Step 4: Compute the required canvas size for the panorama
pano_w = 0
pano_h = 0
for img, (x_off, y_off) in zip(warped_images, shifted_offsets):
    h, w = img.shape[:2]
    pano_w = max(pano_w, x_off + w)
    pano_h = max(pano_h, y_off + h)

# Step 5: Initialize the panorama canvas
panorama = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

# Step 6: Blend each warped image into the panorama at its shifted offset
for img, (x_off, y_off) in zip(warped_images, shifted_offsets):
    panorama = blend_warped_images(panorama, img, offset=(x_off, y_off))

# Step 7: Show the final panorama
show_all_images([panorama], "Final Panorama")