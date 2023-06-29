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
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter, convolve
from math import exp, pi
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
# %% DEMONSTRATE FILTER KERNELS
# Create filter kernels:
def create_identity_kernel():
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

# define a 5x5 mean filter kernel:
def create_mean_kernel_5x5():
    return np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]]) / 25
    

def create_ridge_detection_kernel():
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

def create_edge_detection_kernel():
    return np.array([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1, -1]])

def create_sharpen_kernel():
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])
    
def create_sharpen_kernel_5x5():
    return np.array([[-1, -1, -1, -1, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, 2, 8, 2, -1],
                        [-1, 2, 2, 2, -1],
                        [-1, -1, -1, -1, -1]])

def create_mean_kernel():
    return np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]) / 9

def create_gaussian_blur_kernel(size=3, sigma=1):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2 * sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def create_gaussian_blur_kernel_formula(size=3, sigma=1):
    kernel = np.zeros((size, size))
    center = size // 2
    for x in range(size):
        for y in range(size):
            exponent = -((x - center)**2 + (y - center)**2) / (2 * sigma**2)
            kernel[x, y] = (1 / (2 * pi * sigma**2)) * exp(exponent)
    return kernel / np.sum(kernel)

def create_unsharp_masking_kernel(size=5, sigma=1, amount=1):
    kernel = create_gaussian_blur_kernel(size, sigma)
    sharpen_kernel = create_sharpen_kernel_5x5()
    mask = kernel - (amount * sharpen_kernel)
    return mask

def create_unsharp_masking_kernel_2():
    return np.array([
                    [-1, -4, -6, -4, -1],
                    [-4, -16, -24, -16, -4],
                    [-6, -24, 476, -24, -6],
                    [-4, -16, -24, -16, -4],
                    [-1, -4, -6, -4, -1]
                ]) / -256


def apply_filter(image, kernel):
    # Pad the image to handle filter border effects
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Apply the filter
    filtered_image = convolve(padded_image, kernel)

    # Clip values to ensure they are within the valid range
    filtered_image = np.clip(filtered_image, 0, 255)

    return filtered_image

def apply_median_filter(image, size=3):
    return median_filter(image, size=size)

# Load the camera image
image = data.camera()
image_array = np.array(image)

# Apply filters
identity_kernel = create_identity_kernel()
ridge_detection_kernel = create_ridge_detection_kernel()
edge_detection_kernel = create_edge_detection_kernel()
sharpen_kernel = create_sharpen_kernel()
mean_kernel_3x3 = create_mean_kernel()
mean_kernel_5x5 = create_mean_kernel_5x5()
gaussian_blur_kernel_3x3 = create_gaussian_blur_kernel(size=3)
gaussian_blur_kernel_3x3_2 = create_gaussian_blur_kernel_formula(size=3)
gaussian_blur_kernel_5x5 = create_gaussian_blur_kernel(size=5)
unsharp_masking_kernel = create_unsharp_masking_kernel(size=5, sigma=1, amount=1)
unsharp_masking_kernel_2 = create_unsharp_masking_kernel_2()
median_filtered_image_3x3 = apply_median_filter(image_array, size=3)
median_filtered_image_5x5 = apply_median_filter(image_array, size=5)

filtered_images = [
    ("Identity", apply_filter(image_array, identity_kernel)),
    ("Ridge Detection", apply_filter(image_array, ridge_detection_kernel)),
    ("Edge Detection", apply_filter(image_array, edge_detection_kernel)),
    ("Sharpen", apply_filter(image_array, sharpen_kernel)),
    ("Mean (3x3)", apply_filter(image_array, mean_kernel_3x3)),
    ("Mean (5x5)", apply_filter(image_array, mean_kernel_5x5)),
    ("Gaussian Blur (3x3)", apply_filter(image_array, gaussian_blur_kernel_3x3)),
    ("Gaussian Blur (5x5)", apply_filter(image_array, gaussian_blur_kernel_5x5)),
    ("Unsharp Masking (variant)", apply_filter(image_array, unsharp_masking_kernel)),
    ("Unsharp Masking", apply_filter(image_array, unsharp_masking_kernel_2)),
    ("Median Filter (3x3)", median_filtered_image_3x3),
    ("Median Filter (5x5)", median_filtered_image_5x5)
]

# Plot and save filtered images
for name, filtered_image in filtered_images:
    plt.figure(figsize=(8, 8))
    plt.imshow(filtered_image, cmap='gray')
    plt.title(name, fontsize=28)
    plt.axis('off')
    plt.savefig(f"{name.lower().replace(' ', '_')}_filtered.png", bbox_inches='tight', pad_inches=0)
    plt.show()



# %% END
