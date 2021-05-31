import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

ax, ay = -2, -3
bx, by = 5, 6

x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)


X, Y = np.meshgrid(x, y)

def z_func(x,y):
 return (ax-bx)*y + (by - ay)*x + (bx*ay - ax*by)


def plot(Z):
    im = plt.contourf(X, Y, Z, levels=100,cmap=cm.RdBu, vmin=-20,vmax=20) # drawing the function
    plt.gca().set_aspect('equal')
    plt.colorbar(im) # adding the colobar on the right

def grad_ax(x,y):
    return by - y


plt.subplot(2,2,1)
y_line = (-(by - ay)*x - (bx*ay - ax*by))/(ax-bx)
plt.plot(x,y_line)
plt.scatter([ax,bx],[ay,by])
plt.gca().set_aspect('equal')
plt.xlim([-10, 10])
plt.ylim([-10, 10])

plt.subplot(2,2,2)
Z = grad_ax(X, Y) # evaluation of the function on the grid
plot(Z)
plt.show()
