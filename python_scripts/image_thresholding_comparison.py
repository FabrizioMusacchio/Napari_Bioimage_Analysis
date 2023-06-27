"""
A simple script to compare different thresholding methods on 2D images.

author: Fabrizio musacchio
date: June 22, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import try_all_threshold

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