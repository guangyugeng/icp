import os
import unittest
import numpy as np
from utils import file_to_np, rotation_matrix
import time
from base_icp import register_by_SVD, icp
from dis_icp import dis_icp


# Constants
TESTS_NUM = 10                             # number of test iterations
DIM = 3                                    # number of dimensions of the points
NOISE_SIGMA = .01                           # standard deviation error to be added
TRANSLATION = .1                            # max translation of the test set
ROTATION = .1                               # max rotation (radians) of the test set


class TestModel(unittest.TestCase):
    def setUp(self):
        self.A = file_to_np(os.path.join(os.getcwd(), 'data/Stanford_bunny.txt'))
        self.B = np.copy(self.A)

        # Translate
        self.t = np.random.rand(DIM)*TRANSLATION
        # print('before', self.B)
        self.B += self.t
        # print('Translate', self.t)
        # print('after', self.B)

        # Rotate
        self.R = rotation_matrix(np.random.rand(DIM), np.random.rand()*ROTATION)
        self.B = np.dot(self.R, self.B.T).T
        # print('Rotate', self.R)

        # Add noise
        # print('noise', np.random.randn(self.A.shape[0], DIM))
        self.B += np.random.randn(self.A.shape[0], DIM) * NOISE_SIGMA


    # def tearDown(self):

    def test_rough_fit(self):
        total_time = 0

        for i in range(TESTS_NUM):

            start = time.time()
            T, R1, t1 = register_by_SVD(self.B, self.A)
            total_time += time.time() - start

            # Make C a homogeneous representation of B
            C = np.ones((self.A.shape[0], 4))
            C[:,0:3] = self.B

            # Transform C
            C = np.dot(T, C.T).T

            assert np.allclose(C[:,0:3], self.A, atol=6*NOISE_SIGMA) # T should transform B (or C) to A
            assert np.allclose(-t1, self.t, atol=6*NOISE_SIGMA)      # t and t1 should be inverses
            assert np.allclose(R1.T, self.R, atol=6*NOISE_SIGMA)     # R and R1 should be inverses

        print('rough fit time: {:.3}'.format(total_time/TESTS_NUM))

        return


    def test_icp(self):
        total_time = 0

        for i in range(TESTS_NUM):

            start = time.time()
            T, R1, t1 = register_by_SVD(self.B, self.A)
            total_time += time.time() - start

            # Make C a homogeneous representation of B
            C = np.ones((self.A.shape[0], 4))
            C[:,0:3] = self.B

            # Transform C
            C = np.dot(T, C.T).T

            assert np.allclose(C[:,0:3], self.A, atol=6*NOISE_SIGMA) # T should transform B (or C) to A
            assert np.allclose(-t1, self.t, atol=6*NOISE_SIGMA)      # t and t1 should be inverses
            assert np.allclose(R1.T, self.R, atol=6*NOISE_SIGMA)     # R and R1 should be inverses
            # Run ICP
            start = time.time()
            T, distances, iterations = icp(self.B, self.A, max_iterations=100, tolerance=0.000001)
            total_time += time.time() - start

            # # Make C a homogeneous representation of B
            # C = np.ones((N, 4))
            # C[:,0:3] = np.copy(B)
            #
            # # Transform C
            # C = np.dot(T, C.T).T

            assert np.mean(distances) < 6*NOISE_SIGMA                   # mean error should be small
            assert np.allclose(T[0:3,0:3].T, self.R, atol=6*NOISE_SIGMA)     # T and R should be inverses
            assert np.allclose(-T[0:3,3], self.t, atol=6*NOISE_SIGMA)        # T and t should be inverses

        print('icp time: {:.3}'.format(total_time/TESTS_NUM))

        return


    def test_dis_icp(self):
        total_time = 0

        for i in range(TESTS_NUM):

            start = time.time()
            T, R1, t1 = register_by_SVD(self.B, self.A)
            total_time += time.time() - start

            # Make C a homogeneous representation of B
            C = np.ones((self.A.shape[0], 4))
            C[:,0:3] = self.B

            # Transform C
            C = np.dot(T, C.T).T

            assert np.allclose(C[:,0:3], self.A, atol=6*NOISE_SIGMA) # T should transform B (or C) to A
            assert np.allclose(-t1, self.t, atol=6*NOISE_SIGMA)      # t and t1 should be inverses
            assert np.allclose(R1.T, self.R, atol=6*NOISE_SIGMA)     # R and R1 should be inverses
            # Run ICP
            start = time.time()
            T, distances, iterations = dis_icp(self.B, self.A, max_iterations=100, tolerance=0.000001)
            total_time += time.time() - start

            # # Make C a homogeneous representation of B
            # C = np.ones((N, 4))
            # C[:,0:3] = np.copy(B)
            #
            # # Transform C
            # C = np.dot(T, C.T).T

            assert np.mean(distances) < 6*NOISE_SIGMA                   # mean error should be small
            assert np.allclose(T[0:3,0:3].T, self.R, atol=6*NOISE_SIGMA)     # T and R should be inverses
            assert np.allclose(-T[0:3,3], self.t, atol=6*NOISE_SIGMA)        # T and t should be inverses

        print('distance_threshold_optimization_icp time: {:.3}'.format(total_time/TESTS_NUM))

        return



if __name__ == '__main__':
    unittest.main()