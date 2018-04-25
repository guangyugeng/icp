# distance_threshold_optimization_icp
import numpy as np
from utils import rotation_matrix_by_SVD, nearest_neighbor
from collections import Counter


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


def dis_icp(A, B, max_iterations=10000, tolerance=0.001):

    assert A.shape == B.shape

    n = A.shape[1]
    # print(A)
    # print(n)

    # 构造齐次坐标
    src = np.ones((n+1,A.shape[0]))
    dst = np.ones((n+1,B.shape[0]))
    src[:n,:] = np.copy(A.T)
    dst[:n,:] = np.copy(B.T)

    s = src[:n,:]
    d = dst[:n,:]
    error = 0
    match_l = []

    for i in range(max_iterations):
        # distances, indices = nearest_neighbor(src[:n,:].T, dst[:n,:].T)
        distances, indices = nearest_neighbor(s.T, d.T)
        print('indices', indices)
        c = Counter(indices)
        # print(c)
        # print('d', distances.shape[0])
        # d = d[:, indices]
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        diff = max_distance - min_distance
        threshold = min_distance + diff/2

        # match_l = []
        # 剔除过大值
        while True:
            # print(distances, threshold)
            nomatch_indices = np.where(distances>threshold)[0]
            print('nomatch_indices', nomatch_indices)
            # break
            # available_indices = []
            # for m in match_indices:
            #     print(m)
            #     if Counter(m) == 1:
            #         available_indices.append(m)
            # print('available_indices', available_indices)
            # available_indices = np.array(available_indices)
            # match_indices = np.array([i[0] for i in np.argwhere(distances<threshold)])
            # print('match_indices', match_indices)
            # match_l.append(match_indices)
            # break
            # print(match_indices)
            # match_distances = np.minimum(distances, threshold)
            if distances.shape[0]*0.8 < nomatch_indices.shape[0]:
                # print(1)
                break
            else:
                threshold += (max_distance - threshold)/2
                # distances = distances[match_indices]
            # else:
            #     threshold = (max_distance+threshold)/2

        # match_l.append(match_indices)


        # for d in match_distances:
        #     np.where()
        # print('min_distance', min_distance)
        # print('max_distance', max_distance)
        # print('match_indices', match_indices)
        # print('src[:n,:]', src[:n,:].shape[1], src[:n,:])
        # print('threshold', threshold)
        # s = src[:n, :]
        # d = dst[:n, indices]
        # print('match_l', len(match_l))
        # for m in match_l:
        #     print(m.shape, m)
        #     distances = distances[m]

        # for m in match_l:
        #    print(s.shape, m.shape, )
        #
        #    s = s[:, m]
        #    d = d[:, m]
        # s = s[:, match_indices]
        # d = d[:, match_indices]
        # distances = distances[match_indices]
        # print(s.shape[1])
        # print('src[:n,:][match_indices]', s.shape[1], s)
        # print('dst[:n,indices]', dst[:n,indices])
        # print('distances', distances)
        T,_,_ = register_by_SVD(s[:,match_indices].T, d[:,indices][:,match_indices].T)
        # T,_,_ = register_by_SVD(src[:n,:].T, dst[:n,indices].T)
        # print('T', T)
        # 变换矩阵作用于齐次坐标
        src = np.dot(T, src)
        # print('before', distances)
        print('distances', distances.shape[0])
        # print('after', distances)


        # check error

        new_error = np.mean(distances)
        if np.abs(error - new_error) < tolerance:
            break
        error = new_error

    # calculate final transformation
    T,_,_ = register_by_SVD(A, src[:n,:].T)

    return T, distances, i


