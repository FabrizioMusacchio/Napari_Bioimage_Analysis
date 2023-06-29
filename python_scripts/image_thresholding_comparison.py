"""
A simple script to compare different thresholding methods on 2D images.

author: Fabrizio musacchio
date: June 22, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import try_all_threshold, threshold_otsu

# %% LOAD IMAGE
cells_image = data.cells3d()
nuclei_2D_projection = np.max(cells_image[:, 1, :, :], axis=0).astype(np.float32)
# Add noise to the blob image
noise = np.random.normal(0, 5000.0, size=nuclei_2D_projection.shape)
nuclei_2D_projection_noisy = (nuclei_2D_projection + noise).astype(np.float32)
# %% THRESHOLDING
# display the original 2D projection
plt.imshow(nuclei_2D_projection, cmap='gray')
plt.title('Original 2D Projection')
plt.show()
plt.imshow(nuclei_2D_projection_noisy, cmap='gray')
plt.title('Noisy image')
plt.show()

fig, ax = try_all_threshold(nuclei_2D_projection, figsize=(5, 10), verbose=True)
plt.tight_layout()
plt.show()

fig, ax = try_all_threshold(nuclei_2D_projection_noisy, figsize=(5, 10), verbose=True)
plt.tight_layout()
plt.show()
# %% 


# Create the noise array
noise = np.random.random((50, 50))

# Create the square in the center with values 100 times larger
center = 10
size = 30
square = np.zeros((50, 50))
square[center:center+size, center:center+size] = 2 * noise[center:center+size, center:center+size]

# Combine the noise and square arrays
image = noise + square

# Flatten the image to 1D
image_flat = image.flatten()

# Calculate Otsu threshold
thresh = threshold_otsu(image_flat)

# Plot the image
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.axis('off')
plt.show()

# Plot the histogram of the image
plt.figure(figsize=(5, 5))
plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
plt.axvline(x=thresh, color='r', linestyle='--', linewidth=2, label='Otsu Threshold')
plt.legend()
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()


# Plot the 1D image with the Otsu threshold
plt.figure(figsize=(5, 5))
plt.plot(image_flat, color='gray', label='1D-flattened image')
plt.axhline(y=thresh, color='r', linestyle='--', linewidth=2, label='Otsu Threshold')
plt.legend()
plt.title('1D Image with Otsu Threshold')
plt.xlabel('Pixel Index')
plt.ylabel('Pixel Value')
plt.show()
