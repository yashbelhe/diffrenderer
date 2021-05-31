import numpy as np
import matplotlib.pyplot as plt

from main import *
from settings import *


# NOTE: Make sure that there are no repeats of vertices here, else there could be duplication of gradients
vertices_ref = [
    Vertex([0.5, 1.0], 0),
    Vertex([1.7, 0.0], 1),
    Vertex([0.5, 0.0], 2),
    # Vertex([0.0, -1.0], 3),
    # Vertex([1.0, 1.0], 4),
    # Vertex([-1.0, -1.0], 5),
]

vertices_N = len(vertices_ref)

tri1_ref = Triangle(vertices_ref[0], vertices_ref[1], vertices_ref[2], np.array([1.0,0.0,0.0]))
# tri2_ref = Triangle(vertices_ref[3], vertices_ref[4], vertices_ref[5], np.array([0.5,0.0,0.0]))
# the order of the triangles decides the z orientation of the triangles. i.e index 0 is at the very top and N - 1 is at the very end
# scene = [tri1]
scene_ref = [tri1_ref]
# scene_ref = [tri1_ref, tri2_ref]

# NOTE: Make sure that there are no repeats of vertices here, else there could be duplication of gradients
vertices = [
    Vertex([-1.0, 0.0], 0),
    Vertex([1.2, 0.0], 1),
    Vertex([1.0, 1.0], 2),
    # Vertex([0.0, -1.0], 3),
    # Vertex([1.0, 1.0], 4),
    # Vertex([-1.0, -1.0], 5),
]

vertices_N = len(vertices)

tri1 = Triangle(vertices[0], vertices[1], vertices[2], np.array([1.0,0.0,0.0]))
# tri2 = Triangle(vertices[3], vertices[4], vertices[5], np.array([0.5,0.0,0.0]))
# the order of the triangles decides the z orientation of the triangles. i.e index 0 is at the very top and N - 1 is at the very end
# scene = [tri1]
scene = [tri1]
# scene = [tri1, tri2]

aa_N *= 100
ref_img = forward(triangle_list=scene_ref)
aa_N = aa_N // 100

eta = 1e4
iter_N = 100
img_hist = []

for i in range(iter_N):

    eta *= 0.97

    if i % 5 == 0:
        aa_N *= 10
        curr_img = forward(triangle_list=scene)
        img_hist.append(curr_img)
        aa_N = int(aa_N // 10)
    else:
        curr_img = forward(triangle_list=scene)



    L = np.mean(np.square(ref_img - curr_img))
    # derivative of loss w.r.t rasterized image
    dL_dI = -2*(ref_img - curr_img)/ref_img.size
    # derivative of rasterized image w.r.t vertex coords
    dI_dv = derivative_edge(triangle_list=scene)

    dL_dv = np.sum(dL_dI[:,:,None,None]*dI_dv, axis=(0,1))
    
    for v_idx in range(3):
        print("iter: {}, vidx: {}, grad: {}".format(i, v_idx, eta*dL_dv[v_idx]))
        tri1.vertices[v_idx].p -= eta*dL_dv[v_idx]

    print(L)

# N_img = 25
# for i in range(N_img):
#     plt.subplot(5,5,i+1)
#     if i == 0:
#         plt.imshow(ref_img)
#         plt.title('ref')
#     else:
#         if i - 1 < len(img_hist):
#             plt.imshow(img_hist[i-1])
#             plt.title(str(i-1))
for i in range(len(img_hist)):
    img = np.zeros((S,S,3))
    img[:,:,0] = img_hist[i]
    img[:,:,1] = ref_img
    plt.imsave("output/optimize/{}.png".format(i),  img)