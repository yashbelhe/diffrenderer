import numpy as np
import matplotlib.pyplot as plt

from settings import *

class Vertex:
    p = None
    idx = None

    def __init__(self, coords, idx):
        self.p = np.array(coords)
        self.idx = idx


class Triangle:
    vertices = []
    color = None

    def __init__(self, v0, v1, v2, c):
        self.vertices = [v0, v1, v2]
        self.color = c

    def inside(self, h_loc, w_loc):
        p0x, p0y = self.vertices[0].p
        p1x, p1y = self.vertices[1].p
        p2x, p2y = self.vertices[2].p

        px, py = h_loc, w_loc

        Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)

        s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
        t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)

        return (s >= 0.0) and (s <= 1.0) and (t >= 0.0) and (t <= 1.0) and (s + t <= 1.0)
    
    def get_color(self, h_loc, w_loc):
        return self.color



# 0,0 is the origin, up and right are positive

def get_loc_for_idx(h_idx, w_idx):
    return bounds_min + px_sz*h_idx, bounds_min + px_sz*w_idx

def get_idx_for_loc(h_loc, w_loc):
    return int((h_loc + px_sz/2 - bounds_min)//px_sz), int((w_loc + px_sz/2  - bounds_min)//px_sz)


def get_anti_aliased_locs_for_loc(h_loc, w_loc):
    # size of each pixel we have 199 + 0.5 + 0.5 = 201 - 1 pixels
    locs = []
    for aa_idx in range(aa_N):
        locs.append((h_loc + px_sz*np.random.uniform(-0.5, 0.5), w_loc + px_sz*np.random.uniform(-0.5, 0.5)))
    return locs

def get_color_at_loc(triangle_list, h_loc, w_loc):
    for tri in triangle_list:
        if tri.inside(h_loc, w_loc):
            return tri.get_color(h_loc, w_loc)
    return np.zeros(3)

def forward(triangle_list):
    # start at bottom left, go to top right
    out_img = np.zeros((S,S,3))
    for h_idx in range(S):
        for w_idx in range(S):
            h_loc, w_loc = get_loc_for_idx(h_idx, w_idx)
            aa_locs = get_anti_aliased_locs_for_loc(h_loc, w_loc)
            c = np.zeros(3)
            for h_loc_aa, w_loc_aa in aa_locs:
                c += get_color_at_loc(triangle_list, h_loc_aa, w_loc_aa)
            out_img[h_idx, w_idx] = c/aa_N
        # if h_idx % 10 == 0:
        #     print(h_idx)
    
    # TODO: return color image
    return np.flipud(out_img)[:,:,0]
    # return np.flipud(out_img)




class EdgeDerivative:
    loc_2_grad = {}
    a = None
    b = None
    ab_len = None
    n = None
    grads = None
    grads_count = None

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.ab_len = np.sqrt(np.sum(np.square((a-b))))
        self.n = np.array([(a[1] - b[1]), (b[0] - a[0])])/self.ab_len
        # 4 gradients at each location, ax, ay, bx, by
        self.grads = np.zeros((S,S,4))
        self.grads_count = np.zeros((S,S))
    
    def solve_for_w(self, h_val):
        a = self.a
        b = self.b
        return (-(a[1] - b[1])*h_val - (a[0]*b[1] - b[0]*a[1]))/(b[0] - a[0])
    
    def solve_for_h(self, w_val):
        a = self.a
        b = self.b
        return (-(b[0] - a[0])*w_val - (a[0]*b[1] - b[0]*a[1]))/(a[1] - b[1])
    
    def get_edge_len_inside_pixel(self, h_loc, w_loc):
        # print(h_loc, w_loc)
        # find 4 intersections

        # NOTE: solve_for_w and solve_for_h will return infinity if the slope is 0 or 180 respectively (or the other way round)
        # In those cases, the edge length is just the size of the pixel

        # TODO: there will be another edge case where there is only a single intersection point due to one of the end points being inside the pixel

        if self.b[0] - self.a[0] == 0 or self.a[1] - self.b[1] == 0:
            return px_sz

        sol = [
            np.array([h_loc - px_sz*0.5, self.solve_for_w(h_loc - px_sz*0.5)]),
            np.array([h_loc + px_sz*0.5, self.solve_for_w(h_loc + px_sz*0.5)]),
            np.array([self.solve_for_h(w_loc - px_sz*0.5), w_loc - px_sz*0.5]), 
            np.array([self.solve_for_h(w_loc + px_sz*0.5), w_loc + px_sz*0.5]), 
        ]

        # print(sol)

        final_sol = []
        for s in sol[:2]:
            if w_loc - px_sz*0.5 <= s[1] and s[1] <= w_loc + px_sz*0.5:
                final_sol.append(s)
        
        for s in sol[2:]:
            if h_loc - px_sz*0.5 <= s[0] and s[0] <= h_loc + px_sz*0.5:
                final_sol.append(s)
        
        final_sol = np.unique(final_sol, axis=0)

        # print(final_sol)
        
        if len(final_sol) != 2:
            print(final_sol)
            print(np.unique(final_sol, axis=0))
        assert len(final_sol) == 2
        return np.sqrt(np.sum(np.square(final_sol[0] - final_sol[1])))

    def uniformly_sample_grads(self, N, triangle_list):
        assert N > 1
        delta = (self.b - self.a)/(N - 1)
        # TODO change this to 0,N, 1,N is only for debugging
        for i in range(0,N):
        # for i in range(1,N):
            p = self.a + delta*i
            p_plus = p + eps_es*self.n
            f_plus = get_color_at_loc(triangle_list, p_plus[0] , p_plus[1])
            p_minus = p - eps_es*self.n
            f_minus = get_color_at_loc(triangle_list, p_minus[0] , p_minus[1])
            grad_alpha = np.array([
                self.b[1] - p[1],
                p[0] - self.b[0],
                p[1] - self.a[1],
                self.a[0] - p[0],
            ])

            idx_for_p = get_idx_for_loc(p[0], p[1])
            
            loc_px_center = get_loc_for_idx(idx_for_p[0], idx_for_p[1])

            # only do it for red channel
            f_plus = f_plus[0]
            f_minus = f_minus[0]

            # only compute grad w.r.t ax
            # grad_alpha = grad_alpha[0]

            edge_len_inside_pixel =  self.get_edge_len_inside_pixel(loc_px_center[0], loc_px_center[1])
            grad = grad_alpha*(f_plus - f_minus)/self.ab_len*edge_len_inside_pixel

            # print(grad.shape)
            self.grads[idx_for_p[0], idx_for_p[1]] += grad
            self.grads_count[idx_for_p[0], idx_for_p[1]] += 1





def derivative_edge(triangle_list):
    # two gradients per vertex, one for its vertical position and the other for the horizontal position
    # grad_img = np.zeros((S,S,vertices_N,2))
    grad_img = np.zeros((S,S,len(triangle_list)*3,2))

    for triangle in triangle_list:
        for vertex_idx_1 in range(0,2):
            for vertex_idx_2 in range(vertex_idx_1+1,3):
                v1 = triangle.vertices[vertex_idx_1]
                v2 = triangle.vertices[vertex_idx_2]

                edge_derivative = EdgeDerivative(v1.p, v2.p)
                edge_derivative.uniformly_sample_grads(edge_samples_N, triangle_list)

                avg_grad = np.zeros_like(edge_derivative.grads)
                avg_grad[edge_derivative.grads_count != 0] = edge_derivative.grads[edge_derivative.grads_count != 0]/edge_derivative.grads_count[edge_derivative.grads_count != 0][:,None]
                # print(v1.idx, v2.idx)
                grad_img[:,:,v1.idx,0] += avg_grad[:,:,0]
                grad_img[:,:,v1.idx,1] += avg_grad[:,:,1]
                grad_img[:,:,v2.idx,0] += avg_grad[:,:,2]
                grad_img[:,:,v2.idx,1] += avg_grad[:,:,3]
    return np.flipud(grad_img)