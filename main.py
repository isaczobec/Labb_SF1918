import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    b = stats.binom.rvs(n=10, p=0.5, size=1000)
    norm = np.linalg.norm(b)
    print(norm)


if __name__ == '__main__':
    main()