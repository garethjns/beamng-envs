import numpy as np
from beamngpy.types import Float3


def euclidean_distance_3d(pos_1: Float3, pos_2: Float3) -> float:
    return np.sqrt(np.sum((np.array(pos_1) - np.array(pos_2)) ** 2))
