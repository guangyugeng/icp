import numpy as np
from utils import rotation_matrix_by_SVD, nearest_neighbor



def register_by_SVD(A, B):

    assert A.shape == B.shape

    n = A.shape[1]

    # trans_to_centroid
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    nocent_A = A - centroid_A
    nocent_B = B - centroid_B

    # rotation Matrix
    R = rotation_matrix_by_SVD(nocent_A, nocent_B)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # transform Matrix
    T = np.identity(n+1)
    T[:n, :n] = R
    T[:n, n] = t

    return T, R, t


def icp(A, B, max_iterations=10000, tolerance=0.001):

    assert A.shape == B.shape

    n = A.shape[1]

    # 构造齐次坐标
    src = np.ones((n+1,A.shape[0]))
    dst = np.ones((n+1,B.shape[0]))
    src[:n,:] = np.copy(A.T)
    dst[:n,:] = np.copy(B.T)

    error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:n,:].T, dst[:n,:].T)


        # print('distances', distances)
        T,_,_ = register_by_SVD(src[:n,:].T, dst[:n,indices].T)

        # 变换矩阵作用于齐次坐标
        src = np.dot(T, src)

        # check error
        new_error = np.mean(distances)
        if np.abs(error - new_error) < tolerance:
            break
        error = new_error

    # calculate final transformation
    T,_,_ = register_by_SVD(A, src[:n,:].T)

    return T, distances, i


