from MatrixHelper import *
from numpy.linalg import norm
import numpy as np

I = np.eye(3)

def Skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def NearZero(z):
    return abs(z) < 1e-6


def SkewToVec(skew):
    return np.array([skew[2][1], skew[0][2], skew[1][0]])


def VecTose3(V):
    return np.r_[np.c_[Skew([V[0], V[1], V[2]]), [V[3], V[4], V[5]]], np.zeros((1, 4))]


def Normalize(V):
    return V / np.linalg.norm(V)


def AxisAng3(expc3):
    return (Normalize(expc3), np.linalg.norm(expc3))


def MatrixExp3(so3mat):
    omgtheta = SkewToVec(so3mat)

    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)


def MatrixExp6(se3mat):

    omgtheta = SkewToVec(se3mat[0: 3, 0: 3])

    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat, omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]


thetalist = np.array([-np.pi/2, 0, 10])
links_length = [10, 10]

M = np.array([[1, 0, 0, 0],
              [0, 1, 0, links_length[1]],
              [0, 0, 1, links_length[0]],
              [0, 0, 0, 1]])

Slist = np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0]]).T


def FKinSpace(M, Slist, thetalist):
    T = np.array(M)

    # Reverse loop
    for i in range(len(thetalist) - 1, -1, -1):

        se3 = VecTose3(np.array(Slist)[:, i] * thetalist[i])

        # exp ^ (S * theta)
        exp6 = MatrixExp6(se3)

        T = np.dot(exp6, T)

    return T


anw = FKinSpace(M, Slist, thetalist)
print(anw)

T = np.linalg.multi_dot([Rz(thetalist[0]),
                             Tz(10),
                             Rx(thetalist[1]),
                             Ty(10),
                             Ty(thetalist[2])])

# print(FKinSpace(M, Slist, thetalist))
# print(T)

def Adjoint(T):
    R = T[0: 3, 0: 3]
    p = T[0: 3, 3]
    return np.r_[np.c_[np.transpose(R), np.dot(np.transpose(R), Skew(p))],
                 np.c_[np.zeros((3, 3)), np.transpose(R)]]

def JacobianSpace(Slist, thetalist):
    Js = np.array(Slist).copy().astype(float)
    T_J = np.eye(4)
    for i in range(1, len(thetalist)):
        T_J = np.dot(T_J, MatrixExp6(VecTose3(np.array(Slist)[:, i - 1] * thetalist[i - 1])))
        Js[:, i] = np.dot(Adjoint(T_J), np.array(Slist)[:, i])
    return Js

print("Jacobian spatial:")
print(JacobianSpace(Slist, thetalist))
