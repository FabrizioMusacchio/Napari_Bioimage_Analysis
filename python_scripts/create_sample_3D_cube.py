"""
A simple script to create a 3D cube filled with spheres
author: Fabrizio musacchio
date: June 26, 2023
"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tifffile import imsave
# %% FUNCTIONS
def check_sphere_overlap(center, radius, existing_centers, existing_radii):
    for existing_center, existing_radius in zip(existing_centers, existing_radii):
        distance = np.sqrt(np.sum((center - existing_center) ** 2))
        if distance <= radius + existing_radius:
            return True  # Overlap detected
    return False  # No overlap
# %% CREATE THE CUBE

# Set cube size
cube_size = 500

# Create cube filled with noise
cube = np.random.normal(0, 10, (cube_size, cube_size, cube_size))

# Set sphere parameters
num_spheres = 10
min_radius = 25
max_radius = 70
sphere_value = 100

# Place spheres inside the cube
sphere_centers = []
sphere_radii = []
placed_spheres = 0

while placed_spheres < num_spheres:
    center = np.random.randint(max_radius, cube_size - max_radius, size=3)
    radius = np.random.randint(min_radius, max_radius)
    
    if not check_sphere_overlap(center, radius, sphere_centers, sphere_radii):
        sphere_centers.append(center)
        sphere_radii.append(radius)
        
        indices = np.ogrid[:cube_size, :cube_size, :cube_size]
        mask = np.sqrt(np.sum((indices - center) ** 2, axis=0)) <= radius
        cube[mask] = sphere_value
        
        placed_spheres += 1
    
# Save the cube as a TIF file
imsave('cube.tif', cube)
# %% PLOTS
# Plotting the cube in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(cube, facecolors='b', edgecolor='k')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


