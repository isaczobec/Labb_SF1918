import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp

def remove_nan(x):
    return x[~np.isnan(x)]

def main():

    data = np.loadtxt('data/birth.dat')
    vikt = remove_nan(data[:,2])
    m_ålder = remove_nan(data[:,3])
    m_längd = remove_nan(data[:,15])
    m_vikt = remove_nan(data[:,14])

    
    # plotta probplot
    plt.figure(figsize=(8,9))
    plt.subplot(2,2,1)
    stats.probplot(vikt, dist="norm", plot=plt)
    plt.title('Barnets vikt')
    plt.subplot(2,2,2)
    stats.probplot(m_ålder, dist="norm", plot=plt)
    plt.title('Mammans ålder')
    plt.subplot(2,2,3)
    stats.probplot(m_längd, dist="norm", plot=plt)
    plt.title('Mammans längd') # denna verkar mest normalfördelad, passar linjen bäst
    plt.subplot(2,2,4)
    stats.probplot(m_vikt, dist="norm", plot=plt)
    plt.title('Mammans vikt')
    plt.show()


    print('med inbyggda funktioner')
    print('p-värde av att barnets vikt är normalfördelad',stats.jarque_bera(vikt).pvalue)
    print('p-värde av att mammans ålder är normalfördelad',stats.jarque_bera(m_ålder).pvalue)
    print('p-värde av att mammans längd är normalfördelad',stats.jarque_bera(m_längd).pvalue) # detta är den ända där nollhypotesen (att fördelningen är normalfördelad) inte förkastas på signifikansnivån 0.05. Denna är också den som passar linjen bäst, vilket är förväntat
    print('p-värde av att mammans vikt är normalfördelad',stats.jarque_bera(m_vikt).pvalue)
    print('p-värde av att test',stats.jarque_bera(stats.norm.rvs(size=10000)).pvalue)

main()