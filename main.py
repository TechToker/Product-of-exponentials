import numpy as np
from numpy.linalg import norm
from MatrixHelper import *
import Visualisation as visual

I = np.eye(3)


def Skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def SkewToVec(skew):
    return np.array([skew[2][1], skew[0][2], skew[1][0]])


def CloseToZero(z):
    return abs(z) < 0.000001


def AxisAng3(expc3):
    return expc3 / np.linalg.norm(expc3), np.linalg.norm(expc3)


def GetRodriguesFormula(theta, skew_w):
    rodr = I + np.sin(theta) * skew_w + (1 - np.cos(theta)) * np.dot(skew_w, skew_w)
    return rodr


def GetRodriguesDash(omgmat, v_theta, theta):
    rodr_dash = np.eye(3) * theta + (1 - np.cos(theta)) * omgmat + (theta - np.sin(theta)) * np.dot(omgmat, omgmat)
    rodr_dash = np.dot(rodr_dash, v_theta)

    return rodr_dash / theta


def MatrixExp3(so3mat):
    omega_theta = SkewToVec(so3mat)

    if CloseToZero(np.linalg.norm(omega_theta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omega_theta)[1]
        omgmat = so3mat / theta

        res = GetRodriguesFormula(theta, omgmat)
        return res


def MatrixExp6(se3mat):
    #print(f"Input: \n{se3mat}\n")
    omega_theta = SkewToVec(se3mat[0: 3, 0: 3])
    v_theta = se3mat[0: 3, 3]

    last_row = [0, 0, 0, 1]

    if CloseToZero(np.linalg.norm(omega_theta)):
        res = se3mat
        res[0: 3, 0: 3] = I
        res[3] = last_row

        return res
    else:
        theta = AxisAng3(omega_theta)[1]

        omgmat = se3mat[0: 3, 0: 3] / theta
        last_column_3x1 = GetRodriguesDash(omgmat, v_theta, theta).reshape(3, 1)

        # print(f"Input: \n{se3mat[0: 3, 0: 3]}\n")
        # print(f"Output: \n{MatrixExp3(se3mat[0: 3, 0: 3])}\n")

        # Matrix 3x3
        anw = MatrixExp3(se3mat[0: 3, 0: 3])

        # Add column at the end
        anw = np.hstack([anw, last_column_3x1])
        # Add last row
        anw = np.vstack([anw, last_row])

        return anw


q = np.array([-np.pi/2, 0, 10])
links_length = [10, 10]

M = np.array([[1, 0, 0, 0],
              [0, 1, 0, links_length[1]],
              [0, 0, 1, links_length[0]],
              [0, 0, 0, 1]])


omega_list = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 0, 0]])

v_list = np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0]])


def GetCombinedOmegaAndV(omega_arr, v_arr):
    return np.hstack([omega_arr, v_arr])


S_list = GetCombinedOmegaAndV(omega_list, v_list)


def Get_s_mul_theta(w, v, theta):
    skew_w = Skew(w)

    # Left part of matrix S
    s_matrix = np.vstack([skew_w, [0, 0, 0]])
    # Last Column of matrix S
    v_4x1 = np.hstack([v, 0]).reshape(4, 1)

    # Combined matrix
    s_matrix = np.hstack([s_matrix, v_4x1])

    s_multiply_theta = s_matrix * theta
    return s_multiply_theta


def FKinSpace(M, omega_arr, v_arr, thetas, lastJointID):
    T = np.array(M)

    for i in range(len(thetas) - 1 - lastJointID, -1, -1):
        se3 = Get_s_mul_theta(omega_arr[i], v_arr[i], thetas[i])

        # exp ^ (S * theta)
        exp6 = MatrixExp6(se3)

        #print(f"arr: {exp6} \n  pos:{exp6[0:3, -1]}\n")

        T = np.dot(exp6, T)

    return T


anw = FKinSpace(M, omega_list, v_list, q, 0)
print(f"New FK: \n{anw}\n")

# print(f"Pos0 {FKinSpace(M, omega_list, v_list, q, 3)[0:3, -1]}")
# print(f"Pos1 {FKinSpace(M, omega_list, v_list, q, 2)[0:3, -1]}")
# print(f"Pos2 {FKinSpace(M, omega_list, v_list, q, 1)[0:3, -1]}")
# print(f"Pos3 {FKinSpace(M, omega_list, v_list, q, 0)[0:3, -1]}")



T = np.linalg.multi_dot([Rz(q[0]),
                         Tz(links_length[0]),
                         Rx(q[1]),
                         Ty(links_length[1]),
                         Ty(q[2])])

print(f"Classical FK: \n{T}\n")


### Jacobian ###

def AdJoint(T):
    R = T[0: 3, 0: 3]
    p = T[0: 3, 3]
    return np.r_[np.c_[np.transpose(R), np.dot(np.transpose(R), Skew(p))],
                 np.c_[np.zeros((3, 3)), np.transpose(R)]]


def JacobianSpace(s_list, thetas):
    Js = np.array(s_list.T).copy().astype(float)
    T_J = np.eye(4)

    for i in range(1, len(thetas)):
        se3 = Get_s_mul_theta(omega_list[i - 1], v_list[i - 1], thetas[i - 1])

        T_J = np.dot(T_J, MatrixExp6(se3))
        Js[:, i] = np.dot(AdJoint(T_J), np.array(s_list.T)[:, i])

    return Js


print(f"Jacobian spatial: \n{JacobianSpace(S_list, q)}\n")

print()
visual.Visualisation(q, links_length)
