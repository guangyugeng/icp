import os
import unittest
import numpy as np
from utils import file_to_np, rotation_matrix, view_numpy_data
import time
# from base_icp import register_by_SVD, icp
from dis_icp import dis_icp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 匹配前
# Constants
TESTS_NUM = 100                            # number of test iterations
DIM = 3                                    # number of dimensions of the points
NOISE_SIGMA = .01                           # standard deviation error to be added
TRANSLATION = .1                            # max translation of the test set
ROTATION = .1                               # max rotation (radians) of the test set

A = file_to_np(os.path.join(os.getcwd(), 'data/Stanford_bunny.txt'))
B = np.copy(A)
# Translate
t = np.random.rand(DIM)*TRANSLATION
print('before', B)
B += t
# print('Translate', self.t)
# print('after', self.B
# Rotate
R = rotation_matrix(np.random.rand(DIM), np.random.rand()*ROTATION)
B = np.dot(R, B.T).T
# print('Rotate', self.R
# Add noise
# print('noise', np.random.randn(self.A.shape[0], DIM))
B += np.random.randn(A.shape[0], DIM) * NOISE_SIGMA

view_numpy_data(A, B)

