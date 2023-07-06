"""
A simple to demonstrate the top-hat method.

author: Fabrizio musacchio
date: June 22, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.ndimage import white_tophat, black_tophat
from skimage import morphology, segmentation, color
# %% LOAD IMAGE
# Load sample image from scikit-image
image = data.camera()
# %% APPLY FILTERS
# Perform white top-hat background subtraction
white_tophat_image = white_tophat(image, size=20)

# Perform black top-hat background subtraction
black_tophat_image = black_tophat(image, size=20)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')

# White top-hat image
axes[1].imshow(white_tophat_image, cmap='gray')
axes[1].set_title('White Top-Hat Image')

# Black top-hat image
axes[2].imshow(black_tophat_image, cmap='gray')
axes[2].set_title('Black Top-Hat Image')

# Adjusting subplot spacing
plt.tight_layout()

# Display the plots
plt.show()
# %% OTHER BG SUBTRACTION METHODS
# Load the camera image


# Load the camera image
image = data.camera()

# Perform morphological opening to estimate the background
background = morphology.opening(image, morphology.disk(20))

# Subtract the estimated background from the image
image_subtracted = image - background



# Plot the original image and the background-subtracted image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(image_subtracted, cmap='gray')
ax[1].set_title('Background-Subtracted Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()
