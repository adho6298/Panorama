# Panorama
This is the panorama generation code used for my first project in MCEN 5228 Computer Vision. Original writen in Google Colab, I am bringing it to a local instance and using github to practice industry standards with regards to programming and version control.  

# Project 1: Panorama! (Total 100 points)

The primary objective of this project is to develop a comprehensive end-to-end pipeline for panorama stitching, similar to the panorama mode available on modern smartphones. Panorama stitching is a technique used to seamlessly combine multiple photographs to create a single, wide-angle seamless image that captures a broader view than a single shot could achieve.

Each image should have few repeated local features (∼30−50% or more, emperically chosen). The following method of stitching images should work for most image sets but you’ll need to be creative for working on harder image sets.
From a set of narrow field of view unordered pair of images:

<img src="https://drive.google.com/uc?export=view&id=1I2vSX5lCQiMZ1H1YffiyGKKR0VzFbwkD">

To a Seamless Panorama:

<img src="https://drive.google.com/uc?export=view&id=1nYI-tK_TFXwe0Qvdxw7QVmFCCb5hUBHp">


---


# Part 1: Input Images
Overview:
<img src="https://drive.google.com/uc?export=view&id=17CubCdTxpJMTPMxAxBbqAuDN7pZXpTPG">

# Part 2: Detect Corners [5 pts]
In this part, you will:

  1) Detect corners
  2) Display corners on top of the images

Sample Corner Detection:

The objective of this step is to detect corners such that they are equally distributed across the image in order to avoid weird artifacts in warping. Corners in the image can be detected using cv2.cornerHarris function with the appropriate parameters. The output is a matrix of corner scores: the higher the score, the higher the probability of that pixel being a corner.

# Part 3: ANMS [10 pts]


In this part, you will:
1.   Select $N$ best corners that are equally spaced through out the image using <b>ANMS</b> (Adaptive Non-Maximal Suppression)
2.   Create Feature Descriptors from Corners

Sample Corner Detection after ANMS:<br>
<img src="https://drive.google.com/uc?export=view&id=1Jihg0GOuppYSXOhuQXt6jc5zfbwqCu0d">

---



To find particular strong corners that are spread across the image, first we need to find $N_{strong}$ corners. Because, when you take a real image, the corner is never perfectly sharp, each corner might get a lot of hits out of the $N_{strong}$ corners - we want to choose only the $N_{best}$ corners after ANMS. In essence, you will get a lot more corners than you should! ANMS will try to find corners which are local maxima.

## ANMS Algorithm:

<img src="https://drive.google.com/uc?export=view&id=1_t7wiw4cFeIHji8AWwcvias-7AI210i0">

Note: Python's `scipy.ndimage.filters.maximum_filter` is similar to `imregionalmax` from MATLAB.

Intuitively, we are trying to find $N_{best}$ corners, say `200` from `500` $N_{strong}$ corners such that they are equally distant from one another. To do sp, we define an adaptive radius $r_i$ as the distance between two feature points `i` and `j`.

<img src="https://drive.google.com/uc?export=view&id=1lx1VfylVBxejl-m0baRe8Kklkb-gHeKn">


# Part 4: Feature Descriptors [15 pts]

In the previous step, you found the feature points (locations of the N best best corners after ANMS are called the feature points). You need to describe each feature point by a feature vector, this is like encoding the information at each feature points by a vector. One of the easiest feature descriptor is described next. <br>
<b>Note:</b> <u>The following kernel/patch/vector sizes are user chosen. Feel free to modify them according to your data</u>

---

Sample Feature Desciptors:<br>
<img src="https://drive.google.com/uc?export=view&id=1f5yY-P6SCTdL68rm3Hr4ofD0vgIY_wWe">

---
<br>

1. Take a patch of size `40×40` centered (this is very important and depends on the input image resolution) around the keypoint.
2. Now apply `gaussian blur` (feel free to play around with the parameters and use `cv.GaussianBlur()` function).
3. Now, sub-sample the blurred output (this reduces the dimension) to `8×8`.
4. `reshape` it to obtain a `64×1` vector.
5. `Standardize` the vector to have `mean` of `0` and `variance` of `1` (This can be done by subtracting all values by mean and then dividing by the standard deviation). Note: Standardization is used to remove bias and some illumination effect.


---

# Part 5: Feature Matching [15 pts]

In the previous step, you encoded each keypoint by `64×1`
 feature vector. Now, you want to match the feature points among the two images you want to stitch together. In computer vision terms, this step is called as finding <u>feature correspondences</u> or <u>feature matching</u> between the 2 images. <br>

---

Sample Feature Matching: (Note some errors and bad matches)<br>
<img src="https://drive.google.com/uc?export=view&id=1S9Stx2qj_yk4VFoB28EuR_GQj2SOlYlR">

---
<br>


 1. Pick a point in image 1, compute sum of square difference between all points in image 2.
 2. Take the ratio of best match (lowest distance) to the second best match (second lowest distance) and if this is below some ratio keep the matched pair or reject it. Repeat this for all points in image 1.
 $$SSD=\sum(v_1−v_2)^2$$
 3. You will be left with only the confident feature correspondences and these points will be used later to estimate the transformation between the 2 images also called as Homography.

---
<br>
Brute-force matching compares every descriptor in one image to every descriptor in another image. While this method guarantees finding matches, it's computationally expensive and may not produce the best results due to its non-selective nature.

# Part 6: Refine Matches [15 pts]

We now have matched all the features correspondences but not all matches will be right. To remove incorrect matches, we will use a robust method called Random Sampling Concensus or RANSAC to compute homography.

RANSAC, or Random Sample Consensus, is an iterative algorithm used to estimate the parameters of a mathematical model. In the context of image stitching, we can use RANSAC to estimate the transformation that maps points from one image to another. (Refer: [Wiki](en.wikipedia.org/wiki/Random_sample_consensus))

---

Sample Refined Feature Matching: (Note the <u>bad</u> feature matches disappeared)<br>
<img src="https://drive.google.com/uc?export=view&id=14BDjsDNMS3YCQ6ZvbNQBLtHEGBQv2zK9">

---
<br>

1. Select four feature pairs (at random), $p_i$ from `image 1`, $p^1_i$ from `image 2`.
2. Compute homography $H$ (exact). Write a function `est_homography`.
3. Compute inliers where $SSD(p^1_i,Hp_i)<\texttt{thresh}$. Here, $Hp_i$
 computed using another function that you will be writing `apply_homography`.
4. Repeat the last three steps until you have exhausted $N_\texttt{max}$ number of iterations (specified by user) or you found more than percentage of inliers (Say 90% for example).
5. Keep the largest set of inliers
6. Re-compute the least-square $\hat{H}$ estimate on all of the inliers.

# Part 7: Warp and Blend [15 pts]
<b>Note: 15 pts extra credit for implementing everything from scratch for this section. </b>

Panorama can be produced by overlaying the pairwise aligned images to create the final output image. Understanding `cv2.warpPerspective` and `cv2.perspectiveTransform` functions will be helpful. For such implementation, apply bilinear tranpolation when you copy pixel values. Feel free to use any third party code for warping and transforming images.<br><br><hr>


Sample Final Warping and Blending Results: (Note the <u>bad</u> feature matches disappeared)<br>
<img src="https://drive.google.com/uc?export=view&id=18mSQS7rRYrbotlfKAAUfAkWWLOy5BApS" width="80%">

---
<br>

For a few datasets, you might need to perform cylindrical projection before warping and blending images together. This is because a simple projective transform such as homography will produce substandard results and the images will be stretched/shrunken to a large extent over the edges.<br>
<u><b>Note:</u> Feel free to use any open source code for this part.</b>

To overcome such distortion problem at the edges, we will be using cylinderical projection on the images first before performing other operations. Essentially, this is a pre-processing step. The following equations transform between normal image co-ordinates and cylinderical co-ordinates:

$$ x' = f \cdot tan \left(\cfrac{x-x_c}{f}\right) + x_c$$
$$ y' = \left(\cfrac{y-y_c}{cos \left(\cfrac{x-x_c}{f}\right)}\right) + y_c$$

These equations describe a cylindrical projection, where $f$ represents the focal length of the camera in pixels (typical values range from `100` to `500`). The original image coordinates $(x, y)$ are transformed into cylindrical coordinates $(x', y')$. The center of the image is denoted by $(x_c, y_c)$.

Note: The pipeline talks about how to stitch a pair of images, you need to extend this to work for multiple images. You can re-run your images pairwise or do something smarter.

Your end goal is to be able to stitch any number of given images - maybe 2 or 3 or 4 or 100, your algorithm should work. If a random image with no matches are given, your algorithm needs to report an error.

Note: When blending these images, there are inconsistency between pixels from different input images due to different exposure/white balance settings or photometric distortions or vignetting. This can be resolved by algorithms such as Poisson blending. You can use third party code for the seamless panorama stitching.

# Final Stage: Putting Everything Together! [5 pts in total]
