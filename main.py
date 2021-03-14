import numpy as np


def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


I = np.eye(3)


# Theta - rotation distance, W - axis of rotation
def RodriguesFormula(theta, w):
    anw = I + np.sin(theta) * skew(w) + (1 - np.cos(theta)) * np.dot(skew(w), skew(w))
    return anw


res = RodriguesFormula(np.pi, [0, 0, 1])
print(f"Res: \n {res} \n")


def TranslationPoE(theta, s_v):
    anw = np.vstack([I, [0, 0, 0]])

    last_column = [[s_v * theta], [s_v * theta], [s_v * theta], [1]]
    anw = np.hstack([anw, last_column])

    return anw


def RotationPoE(theta, s_v):
    rodr = RodriguesFormula(theta, s_v)

    anw = np.vstack([rodr, [0, 0, 0]])

    rodr_dash = I * theta + (1 - np.cos(theta)) * skew()
    #last_column =
    anw = np.hstack([anw, [[s_v * theta], [s_v * theta], [s_v * theta], [1]]])

    return anw


from numpy import array,mat,sin,cos,dot,eye
from numpy.linalg import norm

def rodrigues(r):
    def S(n):
        Sn = array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
        return Sn
    theta = norm(r)
    if theta > 1e-30:
        n = r/theta
        Sn = S(n)
        R = eye(3) + sin(theta)*Sn + (1-cos(theta))*dot(Sn,Sn)
    else:
        Sr = S(r)
        theta2 = theta**2
        R = eye(3) + (1-theta2/6.)*Sr + (.5-theta2/24.)*dot(Sr,Sr)
    return mat(R)

def so3ToVec(so3mat):
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def NearZero(z):
    return abs(z) < 1e-6
def Normalize(V):
    return V / np.linalg.norm(V)

def AxisAng3(expc3):
    return (Normalize(expc3), np.linalg.norm(expc3))

def MatrixExp3(so3mat):
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatrixExp6(se3mat):
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def VecToso3(omg):
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def VecTose3(V):
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]

def FKinBody(M, Blist, thetalist):
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Blist)[:, i] \
                                          * thetalist[i])))
    return T

L2 = 10
L1 = 10

M = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, L1],
              [0, 0, 0, 1]])
Blist = np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, L1, 0],
                  [0, 0, 0, 0, 1, 0]]).T
thetalist = np.array([-np.pi/2, 0, 3])

print(FKinBody(M, Blist, thetalist))
