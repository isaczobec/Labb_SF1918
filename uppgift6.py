import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp
import tools

def main():
    data = np.loadtxt('data/moore.dat')
    
    # logaritmera data
    n = data.shape[0]
    X = np.column_stack((np.ones(n),data[:,0]))
    w = np.log(data[:,1])

    # gör linjär regression
    B = tools.regress(X,w)[0]
    skattade_punkter = X @ B # skattade värden, våra y_hat

    # plotta data och linjär regression
    plt.figure(figsize=(8,8))
    plt.scatter(data[:,0],w)
    plt.plot(data[:,0],skattade_punkter,'r')
    plt.legend(['Datapunkter','Regressionslinje'])

    plt.show()

    # plotta residualer
    residuals = w - skattade_punkter
    plt.figure(figsize=(8,8)) 
    plt.subplot(2,1,1)
    _ = stats.probplot(residuals, plot=plt)

    # skatta sigma för residualerna
    x = X[:,1]
    x_mean = np.mean(x)
    y_mean = np.mean(w)
    Syy = (y_mean - w).T @ (y_mean - w)
    Sxy = np.sum((y_mean - w) * (x_mean - x))
    Sxx = (x_mean - x).T @ (x_mean - x)
    sigma = np.sqrt((Syy - Sxy**2/Sxx)/(n-2))
    
    xs = np.linspace(np.min(residuals),np.max(residuals),100)
    ys = stats.norm.pdf(xs,0,sigma)

    plt.subplot(2,1,2)
    plt.hist(residuals, bins=40, density=True)
    plt.plot(xs,ys,'r')
    plt.legend(['Residualer',f'Normalfördelning med skattad sigma={sigma}'])

    plt.show()


    # beräkna transistorer per ytenhet 2025
    transistors_per_area_2025_skattning = np.exp(B[0] + B[1]*2025)
    print('Skattning av antal transistorer per år 2025:',transistors_per_area_2025_skattning)
    



main()