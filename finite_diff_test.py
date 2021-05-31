import numpy as np
import matplotlib.pyplot as plt

from main import *
from settings import *

# NOTE: Make sure that there are no repeats of vertices here, else there could be duplication of gradients
vertices = [
    Vertex([-1.0, 0.0], 0),
    Vertex([1.2, 0.0], 1),
    Vertex([1.0, 1.0], 2),
]

tri1 = Triangle(vertices[0], vertices[1], vertices[2], np.array([1.0,0.0,0.0]))
scene = [tri1]



# derivative of rasterized image w.r.t vertex coords
dI_dv_chain = derivative_edge(triangle_list=scene)[:,:,0,0]

I_v = forward(triangle_list=scene)
h = 1e-2
tri1.vertices[0].p[0] += h

I_v_h = forward(triangle_list=scene)
dI_dv_finite_diff = (I_v_h - I_v)/h

plt.subplot(1,3,1)
plt.imshow(dI_dv_chain)
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(dI_dv_finite_diff)
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(dI_dv_chain)
plt.colorbar()
plt.show()

