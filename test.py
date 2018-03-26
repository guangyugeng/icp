__author__ = 'guangyugeng'

import os
import unittest
import numpy as np
from utils import file_to_np
import time
from base_icp import rough_fit


# Constants
# N = 10                                    # number of random points in the dataset
num_tests = 100                             # number of test iterations
DIM = 3                                    # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


class TestModel(unittest.TestCase):
    def setUp(self):
        self.A = file_to_np(os.path.join(os.getcwd(), 'data/Stanford_bunny.txt'))
        self.B = np.copy(self.A)

        # Translate
        self.t = np.random.rand(DIM)*translation
        self.B += self.t

        # Rotate
        self.R = rotation_matrix(np.random.rand(DIM), np.random.rand()*rotation)
        self.B = np.dot(self.R, self.B.T).T

        # Add noise
        print(np.random.randn(self.A.shape[0], DIM))
        self.B += np.random.randn(self.A.shape[0], DIM) * noise_sigma


    # def tearDown(self):

    def test_rough_fit(self):
        total_time = 0

        for i in range(num_tests):

            start = time.time()
            T, R1, t1 = rough_fit(self.B, self.A)
            total_time += time.time() - start

            # Make C a homogeneous representation of B
            C = np.ones((self.A.shape[0], 4))
            C[:,0:3] = self.B

            # Transform C
            C = np.dot(T, C.T).T

            assert np.allclose(C[:,0:3], self.A, atol=6*noise_sigma) # T should transform B (or C) to A
            assert np.allclose(-t1, self.t, atol=6*noise_sigma)      # t and t1 should be inverses
            assert np.allclose(R1.T, self.R, atol=6*noise_sigma)     # R and R1 should be inverses

        print('best fit time: {:.3}'.format(total_time/num_tests))

        return


if __name__ == '__main__':
    unittest.main()