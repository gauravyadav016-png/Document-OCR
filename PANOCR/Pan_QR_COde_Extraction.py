import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
import numpy as np

# Parameters to remove small and large size connected pixels outliers
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100
constant_parameter_4 = 18

# Read the input image
img = cv2.imread('1.jpeg', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

# Connected component analysis
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)

# Variables for the largest component
the_biggest_component = 0
total_area = 0
counter = 0
for region in regionprops(blobs_labels):
    if region.area > 10:
        total_area += region.area
        counter += 1
    if region.area >= 250 and region.area > the_biggest_component:
        the_biggest_component = region.area

average = total_area / counter if counter > 0 else 0
a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

# Remove small and large objects
pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
component_sizes = np.bincount(pre_version.ravel())
too_big = component_sizes > a4_big_size_outliar_constant
too_big_mask = too_big[pre_version]
pre_version[too_big_mask] = 0

# Convert pre_version to binary and label it for contour finding
pre_version_binary = pre_version > 0
labeled_image = measure.label(pre_version_binary)

# Find the largest connected component
largest_area = 0
largest_bbox = None
for region in regionprops(labeled_image, intensity_image=img):
    if region.area >= largest_area:
        largest_area = region.area
        largest_bbox = region.bbox

# Crop and save the signature
if largest_bbox:
    minr, minc, maxr, maxc = largest_bbox
    cropped_signature = img[minr:maxr, minc:maxc]
    cv2.imwrite("cropped_signature.png", cropped_signature)
else:
    print("No significant signature component found.")

# Display the extracted signature
extracted_signature = cv2.imread('cropped_signature.png')
cv2.imshow('Cropped Signature', extracted_signature)
cv2.waitKey(0)
cv2.destroyAllWindows()
