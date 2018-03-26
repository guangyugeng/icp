import numpy as np


def rotation_matrix_by_SVD(A, B):
    #用 SVD
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[A.shape[1]-1,:] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def rough_fit(A, B):

    assert A.shape == B.shape

    c = A.shape[1]

    #trans_to_centroid
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    nocent_A = A - centroid_A
    nocent_B = B - centroid_B

    #rotation Matrix
    R = rotation_matrix_by_SVD(nocent_A, nocent_B)

    #translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    #transform Matrix
    T = np.identity(c+1)
    T[:c, :c] = R
    T[:c, c] = t

    return T, R, t


def icp():
    pass

