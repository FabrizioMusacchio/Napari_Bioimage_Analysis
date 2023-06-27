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