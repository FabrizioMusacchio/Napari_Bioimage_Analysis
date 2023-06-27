"""
A simple script to create some noisy sample images.

author: Fabrizio musacchio
date: June 22, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pingouin as pg
from skimage import data
from skimage.util import random_noise
# %% TEST

# Print the versions of the packages imported:
print('numpy version: ', np.__version__)
print('matplotlib version: ', mpl.__version__)
print('pandas version: ', pd.__version__)
print('pingouin version: ', pg.__version__)
# %% CREATE SOME SAMPLE IMAGES
cells_image = data.cells3d()
nuclei_2D_projection = np.max(cells_image[:, 1, :, :], axis=0)

# display the original 2D projection
plt.imshow(nuclei_2D_projection, cmap='gray')
plt.title('Original 2D Projection')
plt.show()

# add gaussian noise to the nuclei 2D projection:
noise = np.random.normal(0, 10000.0, size=nuclei_2D_projection.shape)
nuclei_2D_projection_noisy = nuclei_2D_projection + noise

# add salt and pepper noise to the nuclei 2D projection:
nuclei_2D_projection_snp_noisy = random_noise(nuclei_2D_projection, mode='s&p', amount=0.05)


# display the noisy 2D projection
plt.imshow(nuclei_2D_projection_noisy, cmap='gray')
plt.title('Noisy 2D Projection')
plt.show()

plt.imshow(nuclei_2D_projection_snp_noisy, cmap='gray')
plt.title('Noisy 2D Projection')
plt.show()

# save nuclei_2D_projection_noisy as tiff:
import tifffile
tifffile.imsave('nuclei_2D_projection_gaussian_noisy.tif', nuclei_2D_projection_noisy)
tifffile.imsave('nuclei_2D_projection_snp_noisy.tif', nuclei_2D_projection_snp_noisy)

# %% END
