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