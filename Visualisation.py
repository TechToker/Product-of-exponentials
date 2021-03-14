import matplotlib.pyplot as plt
import numpy as np
from MatrixHelper import *

q = [0, np.pi / 4, 3]
link_length = [10, 10]
base_center = [0, 0, 0]


def Visualisation(q, links):
    ax = plt.axes(projection='3d')
    pos0 = [0, 0, 0]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tz(links[0])])
    pos1 = T[0:3, 3]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tz(links[0]),
                             Rx(q[1]),
                             Ty(links[1])])
    pos2 = T[0:3, 3]

    T = np.linalg.multi_dot([Rz(q[0]),
                             Tz(links[0]),
                             Rx(q[1]),
                             Ty(links[1]),
                             Ty(q[2])])
    pos3 = T[0:3, 3]

    x = [pos0[0], pos1[0], pos2[0], pos3[0]]
    y = [pos0[1], pos1[1], pos2[1], pos3[1]]
    z = [pos0[2], pos1[2], pos2[2], pos3[2]]

    ax.plot3D(x, y, z, linewidth=1, c='b')
    ax.scatter3D(x, y, z, s=25, c='r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(0, 15)

    plt.show()


Visualisation(q, link_length)
