import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def test_konfidensintervall_rayleigh(
        alpha = 0.01
):
    y = np.loadtxt('data/wave_data.dat')

    plt.figure(figsize=(4,8))
    plt.subplot(2,1,1)
    plt.plot(y[:100])

    plt.subplot(2,1,2)
    plt.hist(y, bins=100, density=True)


    # skatta parametern med MK eller ML
    M = len(y)
    est = np.sum(y)/M * np.sqrt(2/np.pi) # MK
    est = np.sqrt(np.sum(y**2) * 1/(2*M)) # ML

    # parametern (variabeln est) kommer vara asymptotiskt normalfördelad eftersom 
    # aritmetiska medelvärdet av Rayleigh-fördelade värden är normalfördelat eftersom 
    # antalet värden är stort (enligt centrala gränsvärdessatsen)
    # och vara N(sigma, sqrt((4-pi)/(M*pi)) * sigma)
    # Alltså kan ett approximativt konfidensintervall hittas genom den fördelningen
    # HITTA KONFIDENSINTERVALL
    lambda_alpha_2 = stats.norm.ppf(1-alpha/2)
    std = np.sqrt((4-np.pi)/(M*np.pi)) * est
    upper_bound = est + lambda_alpha_2 * std
    lower_bound = est - lambda_alpha_2 * std

    print('lower bound:', lower_bound,', skattning:',est, 'upper bound:', upper_bound)


    # plotta konfidensintervall och skattning
    plt.plot([est, est], [0, 0.2], 'r')
    plt.plot([lower_bound,upper_bound], [0.6,0.6], 'r*', markersize=10)

    # plotta täthetsfunktion med det skattade värdet
    xs = np.linspace(np.min(y),np.max(y), 100)
    ys = stats.rayleigh.pdf(xs, scale=est)
    plt.plot(xs, ys, 'b')

    plt.legend(['Estimat av b', 'Konfidensintervall','Täthetsfunktion med skattat parametervärde'])
    plt.show()

    test_parameter_disribution(est) # testa att den hittade normalfördeingen är rimlig



def test_parameter_disribution(
        parameter_value,
        M = 1000, # amount of simulations per test
        N = 10000, # amount of tests
):
    """Function to test that the distribution of the parameter value actually can be approximated with a normal distribution"""
    
    
    # testa skatta parametern med MK många gånger
    M = 100
    x = stats.rayleigh.rvs(loc=0, scale=parameter_value, size=(M,N))
    est = np.sum(x, axis=0)/M * np.sqrt(2/np.pi) # MK
    print(est)

    plt.figure(figsize=(4,8))
    plt.hist(est, bins=100, density=True)
    
    # approximate with normal distribution
    xs = np.linspace(np.min(est), np.max(est), 100)
    ys = stats.norm.pdf(xs, loc=parameter_value, scale=np.sqrt((4-np.pi)/(M*np.pi)) * parameter_value)
    plt.plot(xs, ys, 'r')

    plt.show()


test_konfidensintervall_rayleigh(alpha=0.001)