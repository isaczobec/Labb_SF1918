import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    test_konfidensintervall_normalfördelning(            
            n = 25,
            sigma = 1,
            mu = 2,
            alpha = 0.05,
            m = 100,
            )
    test_konfidensintervall_normalfördelning(            
            n = 25,
            sigma = 2, # prova att ändra sigma till 2
            mu = 4, # prova att ändra mu till 4
            alpha = 0.05,
            m = 100, 
            )
    test_konfidensintervall_normalfördelning(            
            n = 100, # prova att ändra n till 100, intervallen blir hälften så breda som förväntat (sqrt av n)
            sigma = 2,
            mu = 4,
            alpha = 0.05,
            m = 100, 
            )
    test_konfidensintervall_normalfördelning(            
        n = 100,
        sigma = 2,
        mu = 4,
        alpha = 0.01, # ändra alpha till 0.01, intervallen blir bredare
        m = 100, 
        )



def test_konfidensintervall_normalfördelning(
            n = 25,
            sigma = 1,
            mu = 2,
            alpha = 0.05,
            m = 1000,
            ):
    # skapa en array med 1000 normalfördelade värden
    x = np.random.normal(loc=mu, scale=sigma, size=(m,n))
    mean = np.mean(x, axis=1)
    D = sigma/np.sqrt(n)
    lambda_alpha_2 = stats.norm.ppf(1-alpha/2)

    # beräkna gränser
    upper = mean + lambda_alpha_2*D
    lower = mean - lambda_alpha_2*D

    # skapa figur och plotta gränser
    fig, ax = plt.subplots(figsize=(4,8))
    contained = 0
    for i in range(m):
        if mu < lower[i] or mu > upper[i]:
            color = 'red'
        else:
            color = 'blue'
            contained += 1
        plt.plot([lower[i],upper[i]],[i,i],color=color)
    plt.plot([mu,mu],[0,m],color='black') # plotta mu, rakt streck i mitten

    # lägg till subplot av alla parametervärden
    parametertext = "\n".join([
        f"n = {n}",
        f"sigma = {sigma}",
        f"mu = {mu}",
        f"alpha = {alpha}",
        f"m = {m}",
        ])
    
    ax.text(0.05, 0.95, parametertext, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # skriv ut antal intervall som innehåller mu och förväntade antalet (innan plotten)
    percentage = contained/m
    print('expected percentage of intervals containing mu: ' + str(1-alpha))
    print('actual percentage of intervals containing mu: ' + str(percentage))
    plt.show()

    


main()