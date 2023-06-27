"""
A simple script to test different spatial image filters on 2D images.

author: Fabrizio musacchio
date: June 22, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter
# %% LOAD IMAGE
# Load sample image from scikit-image
image = data.camera()

# Add Gaussian noise to the image
noisy_image = random_noise(image, mode='gaussian', var=0.05)
# %% APPLY FILTERS
# Apply mean filter
mean_filtered = uniform_filter(noisy_image, size=3, mode='constant')

# Apply median filter
median_filtered = median_filter(noisy_image, size=3)

# Apply Gaussian filter
sigma = 1
gaussian_filtered = gaussian_filter(noisy_image, sigma=sigma)
# %%
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# original and noisy image:
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 12))
# denoised image using mean filter:
axes[0].imshow(mean_filtered, cmap='gray')
axes[0].set_title('Mean Filtered Image')

# denoised image using median filter:
axes[1].imshow(median_filtered, cmap='gray')
axes[1].set_title('Median Filtered Image')

# denoised image using Gaussian filter:
axes[2].imshow(gaussian_filtered, cmap='gray')
axes[2].set_title('Gaussian Filtered Image')

plt.tight_layout()
plt.show()
# %% END
