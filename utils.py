import numpy as np
import re
import os


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