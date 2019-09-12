import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.model_selection import KFold

from assignment_one.src.ordinary_least_squares import perform_regression

terrain = np.array(Image.open('../data/SRTM_data_Norway_2.tif'))
aspect_ratio = terrain.shape[0] / terrain.shape[1]
print(f'Aspect ratio of terrain data = {aspect_ratio : 2.3f}')
# terrain = terrain / np.linalg.norm(terrain)
terrain = (terrain - np.min(terrain)) / np.max(terrain)

plt.imshow(terrain)
plt.show()

N = terrain.shape[0]
M = terrain.shape[1]

x = np.linspace(0, 1, N)
y = np.linspace(0, aspect_ratio, M)

X, Y = np.meshgrid(x, y)

params = np.dstack((X, Y)).reshape(-1, 2)
beta, z_hat = perform_regression(X, Y, terrain.ravel(), polynomial_degree=2, ridge=False, l=0.1)

fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')

ax1.plot_surface(X, Y, terrain.T)
ax2.plot_surface(X, Y, z_hat)
plt.show()
