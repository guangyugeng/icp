import numpy as np
import re
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def file_to_np(file_path):
    a = []
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])

            b = np.array([x,y,z])
            a.append(b)

    data = np.array(a)
    return data


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def nearest_neighbor(src, dst):
    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    # print('indices.ravel()', indices.ravel())
    return distances.ravel(), indices.ravel()


def view_numpy_data(source_data, target_data):
    x1 = [-k[0] for k in source_data]
    y1 = [k[1] for k in source_data]
    z1 = [k[2] for k in source_data]
    x2 = [-k[0] for k in target_data]
    y2 = [k[1] for k in target_data]
    z2 = [k[2] for k in target_data]

    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(z2, x2, y2, c='r', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')
    ax.scatter(z1, x1, y1, c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')

    #ax.set_facecolor((0,0,0))
    ax.axis('scaled')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.rotate(45)
    plt.title('point cloud')
    plt.show()
