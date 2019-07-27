import random
import numpy as np
import matplotlib.pyplot as plt


# from mpl_toolkits.mplot3d import axes3d


# A fade function for fixing the transitions between grids
def fade(val):
    return 6 * val ** 5 - 15 * val ** 4 + 10 * val ** 3


# Amount of grid squares
vector_grid_x = 16
vector_grid_y = 16

# Total pixel count
pixels_x = 256
pixels_y = 256

if pixels_x % vector_grid_x != 0 or pixels_y % vector_grid_y != 0:
    raise ValueError('Pixel count should be evenly divisible by vector grid count in each dimension')

# Pixels per square
pps_x = pixels_x / (vector_grid_x - 1)
pps_y = pixels_y / (vector_grid_y - 1)

# Step sizes in a grid square
step_x = 1 / pps_x
step_y = 1 / pps_y

gradients = [[(random.uniform(-1, 1), random.uniform(-1, 1)) for x in range(vector_grid_x)]
             for y in range(vector_grid_y)]

# An alternative gradient implementation:
# possible_grad_values = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# gradients = [[random.choice(possible_grad_values) for x in range(vector_grid_x)] for y in range(vector_grid_y)]

# Main loop for calculating each pixel value
values = []
for y in range(pixels_y):
    values.append([])
    for x in range(pixels_x):
        # Dot product of each gradient and distance to chosen pixel in between the gradient edges
        upper_left_product = np.dot(gradients[int(y / pps_y)][int(x / pps_x)],
                                    ((x % pps_x) * step_x, (y % pps_y) * step_y))

        upper_right_product = np.dot(gradients[int(y / pps_y)][int(x / pps_x) + 1],
                                     ((x % pps_x) * step_x - 1, (y % pps_y) * step_y))

        lower_left_product = np.dot(gradients[int(y / pps_y) + 1][int(x / pps_x)],
                                    ((x % pps_x) * step_x, (y % pps_y) * step_y - 1))

        lower_right_product = np.dot(gradients[int(y / pps_y) + 1][int(x / pps_x) + 1],
                                     ((x % pps_x) * step_x - 1, (y % pps_y) * step_y - 1))

        # Linear interpolations for upper and lower dot products + fade function.
        upper_interpolation = upper_left_product + fade((x % pps_x) * step_x) * (
                upper_right_product - upper_left_product)

        lower_interpolation = lower_left_product + fade((x % pps_x) * step_x) * (
                lower_right_product - lower_left_product)

        # Linear interpolation for y axis
        values[y].append(upper_interpolation + fade((y % pps_y) * step_y) * (lower_interpolation - upper_interpolation))

plt.imshow(values, cmap=plt.get_cmap('Greys'))
plt.show()

''' Some 3d data visualization (bad with many grid squares)
fig = plt.figure()
ax = plt.axes(projection='3d')
xv, yv = np.meshgrid(np.linspace(0, 1, pixels_x), np.linspace(0, 1, pixels_y))
ax.plot_surface(xv, yv, np.array(values))
plt.show()
'''
