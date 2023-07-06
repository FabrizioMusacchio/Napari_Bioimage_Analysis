# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, img_as_ubyte
from skimage.util import img_as_float
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.exposure import rescale_intensity, equalize_adapthist, match_histograms
from skimage.filters import threshold_local
# %% LOAD IMAGE
# Load the "Cells 3D" image from scikit-image data
cells_3d = data.cells3d()

# Extract the nuclei channel (channel index 2)
nuclei_channel = cells_3d[:, 1, :, :]

# Maximum project the nuclei to 2D
nuclei_2d = np.max(nuclei_channel, axis=0)

# Create a new stack with 5 times the 2D projections
stack = np.repeat(nuclei_2d[np.newaxis, :, :], 5, axis=0)

# Apply simulated photo-bleaching to layers 1 to 4
""" for i in range(1, 5):
    decay_factor = 1*i  # Adjust the decay factor to control the intensity drop
    stack[i] = stack[i]/decay_factor """

# Apply simulated photo-bleaching to layers 1 to 4
for i in range(1, 5):
    # Calculate adaptive threshold for each frame
    block_size = 51  # Adjust the block size as needed
    adaptive_thresh = threshold_local(stack[i], block_size, offset=0)

    # Apply the adaptive threshold to bright pixels
    stack[i][stack[i] > adaptive_thresh] = stack[i][stack[i] > adaptive_thresh]* (0.8 ** i)

# Create figure with 5 columns and 2 rows for visualization
fig, axs = plt.subplots(2, 5, figsize=(16, 7))

vmin = np.min(stack[0])
vmax = np.max(stack[0])
for i in range(5):
    # Plot bleached image
    axs[0, i].imshow(stack[i], cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, i].set_title(f'Layer {i}')
    axs[0, i].axis('off')

    # Compute histogram of the bleached image
    bins = 256
    #hist, bins = np.histogram(stack[i].ravel(), bins=256, range=(0, 1))

    # Plot histogram
    #axs[1, i].plot(bins[:-1], hist, lw=2)
    axs[1, i].hist(stack[i].ravel(), bins=bins, histtype='step', color='black')
    axs[1, i].set_title(f'Layer {i} Histogram')
    axs[1, i].set_ylim(0, 3800)

plt.tight_layout()
plt.show()

# %% CORRECT THE BLEACHING WITH HISTOGRAM MATCHING
corrected_image = stack.copy().astype("float32")
for i in range(1,5):
    # Apply photo-bleaching correction using histogram matching:
    corrected_image[i,:,:] = match_histograms(stack[i], stack[0])

# Create figure with 5 columns and 2 rows for visualization
fig, axs = plt.subplots(2, 5, figsize=(16, 7))
for i in range(5):
    # Plot bleached image
    axs[0, i].imshow(corrected_image[i], cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, i].set_title(f'Layer {i}')
    axs[0, i].axis('off')

    # Plot histogram:
    bins = 256
    axs[1, i].hist(corrected_image[i].ravel(), bins=bins, histtype='step', color='black')
    axs[1, i].set_title(f'Layer {i} Histogram')
    axs[1, i].set_ylim(0, 3800)
    
plt.tight_layout()
plt.show()
# %% CORRECT THE BLEACHING WITH HISTOGRAM MATCHING
corrected_image = stack.copy().astype("float32")
for i in range(1,5):
    # Apply photo-bleaching correction using histogram matching:
    corrected_image[i,:,:] = rescale_intensity(stack[i], in_range='image', out_range=(0, 1))


# Create figure with 5 columns and 2 rows for visualization
fig, axs = plt.subplots(2, 5, figsize=(16, 7))
for i in range(5):
    # Plot bleached image
    axs[0, i].imshow(corrected_image[i], cmap='gray')
    axs[0, i].set_title(f'Layer {i}')
    axs[0, i].axis('off')

    # Plot histogram:
    bins = 256
    axs[1, i].hist(corrected_image[i].ravel(), bins=bins, histtype='step', color='black')
    axs[1, i].set_title(f'Layer {i} Histogram')
    axs[1, i].set_ylim(0, 3800)
    
plt.tight_layout()
plt.show()

# %% CORRECT THE BLEACHING WITH ADAPTIVE HISTOGRAM EQUALIZATION

corrected_image = stack.copy().astype("float32")
for i in range(1,5):
    # Apply photo-bleaching correction using adaptive histogram equalization:
    corrected_image[i,:,:] = equalize_adapthist(stack[i])
    
# Create figure with 5 columns and 2 rows for visualization
fig, axs = plt.subplots(3, 5, figsize=(16, 12))
for i in range(5):
    # Plot bleached image
    axs[0, i].imshow(corrected_image[i], cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, i].set_title(f'Layer {i}')
    axs[0, i].axis('off')
    
    axs[1, i].imshow(corrected_image[i], cmap='gray')
    axs[1, i].set_title(f'Layer {i}')
    axs[1, i].axis('off')

    # Compute histogram of the bleached image
    bins = 256

    # Plot histogram
    #axs[1, i].plot(bins[:-1], hist, lw=2)
    axs[2, i].hist(corrected_image[i].ravel(), bins=bins, histtype='step', color='black')
    axs[2, i].set_title(f'Layer {i} Histogram')
    axs[1, i].set_ylim(0, 3700)

plt.tight_layout()
plt.show()
