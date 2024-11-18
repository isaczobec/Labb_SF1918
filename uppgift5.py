import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp

def skewedness(
        arr : np.array
        ) -> float:
    n = len(arr)
    u = np.mean(arr)
    s = np.sqrt(np.sum((arr - u)**2)/(n-1))
    return np.sum((arr - u)**3)/(n*s**3)

def kurtosis(
        arr : np.array
        ) -> float:
    n = len(arr)
    u = np.mean(arr)
    s = np.sqrt(np.sum((arr - u)**2)/(n-1))
    return np.sum((arr - u)**4)/(n*s**4) - 3

def jarque_bera_test_p_value(
        x : np.array,
)-> float: 
    n = len(x)
    skew = skewedness(x)
    kurt = kurtosis(x)
    jb = n/6*(skew**2 + (kurt-3)**2/4)
    p = 1 - stats.chi2.cdf(jb,2)
    return p


def main():

    data = np.loadtxt('data/birth.dat')
    vikt = data[:,2]
    m_ålder = data[:,3]
    m_längd = data[:,15]
    m_vikt = data[:,14] 

    print('med inbyggda funktioner')
    print('p-värde av att barnets vikt är normalfördelad',stats.jarque_bera(vikt).pvalue)
    print('p-värde av att mammans ålder är normalfördelad',stats.jarque_bera(m_ålder).pvalue)
    print('p-värde av att mammans längd är normalfördelad',stats.jarque_bera(m_längd).pvalue)
    print('p-värde av att mammans vikt är normalfördelad',stats.jarque_bera(m_vikt).pvalue)

    print('med egna funktioner')
    print('p-värde av att barnets vikt är normalfördelad',jarque_bera_test_p_value(vikt))
    print('p-värde av att mammans ålder är normalfördelad',jarque_bera_test_p_value(m_ålder))
    print('p-värde av att mammans längd är normalfördelad',jarque_bera_test_p_value(m_längd))
    print('p-värde av att mammans vikt är normalfördelad',jarque_bera_test_p_value(m_vikt))





main()