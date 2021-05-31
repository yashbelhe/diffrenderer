import numpy as np
import matplotlib.pyplot as plt

from main import *
from settings import *



# NOTE: Make sure that there are no repeats of vertices here, else there could be duplication of gradients
vertices = [
    Vertex([0.0, 1.0], 0),
    Vertex([0.2, 0.0], 1),
    Vertex([0.0, 0.0], 2),
    Vertex([0.0, -1.0], 3),
    Vertex([1.0, 1.0], 4),
    Vertex([-1.0, -1.0], 5),
]

vertices_N = len(vertices)

tri1 = Triangle(vertices[0], vertices[1], vertices[2], np.array([1.0,0.0,0.0]))
# tri1 = Triangle(vertices[0], vertices[1], vertices[2], np.array([1.0,1.0,0.0]))
tri2 = Triangle(vertices[3], vertices[4], vertices[5], np.array([0.5,0.0,0.0]))
# the order of the triangles decides the z orientation of the triangles. i.e index 0 is at the very top and N - 1 is at the very end
# scene = [tri1]
scene = [tri1, tri2]


# BUG WITH half pixel size for corner pixels possibly

# TODO use finite differences to see whether the gradients are correct with and without the edge length term
nr, nc = 2, 2

img = derivative_edge(triangle_list=scene)
vmin = np.min(img)
vmax = np.max(img)

plt.figure()
plt.subplot(nr,nc,1)
plt.imshow(np.sum(img[:,:,0:3,0],axis=2), vmin=vmin, vmax=vmax)
plt.title('gradient of triangle 1 if we move y coodinates upwards')
# plt.imshow(img[:,:,3,0], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(nr,nc,2)
plt.imshow(img[:,:,0,0], vmin=vmin, vmax=vmax)
# plt.imshow(np.sum(img[:,:,3:6,1],axis=2), vmin=vmin, vmax=vmax)
# plt.title('gradient of triangle 2 if we move x coodinates rightwards')
plt.colorbar()

plt.subplot(nr,nc,3)
# plt.imshow(img[:,:,4,0])
plt.imshow(img[:,:,1,0], vmin=vmin, vmax=vmax)
plt.colorbar()


plt.subplot(nr,nc,4)
img = forward(triangle_list=scene)
plt.imshow(img)
plt.show()