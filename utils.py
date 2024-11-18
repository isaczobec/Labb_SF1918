import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def covariance(
        x : np.array,
        y : np.array,
        ) -> float:
    """returns the covariance between two arrays x and y"""

    if len(x) != len(y):
        raise ValueError('x and y must have the same length')

    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    return np.sum((x - mean_x)*(y - mean_y))/(n-1)


def correlation(
        x : np.array,
        y : np.array,
        ) -> float:
    """returns the correlation between two arrays x and y"""

    n = len(x)
    m = len(y)
    if n != m:
        raise ValueError('x and y must have the same length')
    

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    return np.sum((mean_x-x)*(mean_y-y)) / (std_x*std_y*(n-1))