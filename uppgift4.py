import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp
import utils

PLOT_KDE = 1
PLOT_HISTOGRAM = 2

def compare_factors(
        column1,
        col1_comparer_func,
        column2,
        data,
        removenan = True,
        x_points = 100,
        titles = ['värde 1','värde 2'],
        plot_mode = PLOT_KDE
):
    indexed_data = data[:,(column1,column2)]
    if removenan: 
        valid_indicies = ~np.isnan(indexed_data).any(axis=1)
        indexed_data = indexed_data[valid_indicies]

    bool_true = col1_comparer_func(indexed_data[:,0])

    
    # skapa kdes och funktionsvärden för 1 respektive 2
    col1 = indexed_data[bool_true,1]
    col2 = indexed_data[~bool_true,1]
    if plot_mode == PLOT_KDE:
        kde_1 = stats.gaussian_kde(col1)
        kde_2 = stats.gaussian_kde(col2)
        xs1 = np.linspace(np.min(col1),np.max(col1),x_points)
        xs2 = np.linspace(np.min(col2),np.max(col2),x_points)
        dist_1 = kde_1(xs1)
        dist_2 = kde_2(xs2)

    mean_1 = np.mean(col1)
    mean_2 = np.mean(col2)

    plt.figure(figsize=(8,8))

    plt.subplot(2,2,1)
    plt.title(f'Boxplot {titles[0]}')
    plt.boxplot(col1)
    plt.subplot(2,2,2)
    plt.title(f'Boxplot {titles[1]}')
    plt.boxplot(col2)

    plt.subplot(2,2,(3,4))
    plt.title(f'{titles[0]} respektive {titles[1]}')

    if plot_mode == PLOT_KDE:
        plt.plot(xs1,dist_1,'g')
        plt.plot(xs2,dist_2,'r')
        plt.plot([mean_1,mean_1],[0,np.max(dist_1)/2],'g')
        plt.plot([mean_2,mean_2],[0,np.max(dist_2)/2],'r')
    if plot_mode == PLOT_HISTOGRAM:
        plt.hist(col1,bins=100,alpha=0.5, weights=np.ones_like(col1)/len(col1), color='g')
        plt.hist(col2,bins=100,alpha=0.5, weights=np.ones_like(col2)/len(col2), color='r')
        plt.plot([mean_1,mean_1],[0,0.5],'g')
        plt.plot([mean_2,mean_2],[0,0.5],'r')

    plt.legend(titles)


    # beräkna korrelationen
    
    valid_indicies = ~np.isnan(data[:,(column1,column2)]).any(axis=1)
    valid_entries = data[valid_indicies][:,(column1,column2)]
    correlation = utils.correlation(valid_entries[:,0], valid_entries[:,1])
    print(f'Korrelation mellan {titles[0]} och {titles[1]}: ', correlation)

    plt.show()


# ---------


data = np.loadtxt('data/birth.dat')

# skapa histogram
vikt = data[:,2]
m_ålder = data[:,3]
m_längd = data[:,15]
m_vikt = data[:,14]
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.hist(vikt,bins=100)
plt.title('Histogram vikt')
plt.subplot(2,2,2)
plt.hist(m_ålder,bins=100)
plt.title('Histogram mammans ålder')
plt.subplot(2,2,3)
plt.hist(m_längd,bins=100)
plt.title('Histogram mammans längd')
plt.subplot(2,2,4)
plt.hist(m_vikt,bins=100)
plt.title('Histogram mammans vikt')
plt.show()


# testa rökning
rök = lambda a : a < 3
compare_factors(19,rök,2,data,titles=['Ickerökare vikt (g)','Rökare vikt (g)'])

# testa kön
kön = lambda a : a <= 1
compare_factors(1,kön,2,data,titles=['Pojkars vikt (g)','Flickors vikt (g)'])

# testa alkohol
alkohol = lambda a : a <= 1
compare_factors(25,alkohol,2,data,titles=['Bebisar som inte drickers vikt (g)','Bebisar som drickers vikt (g)'])

# testa alkohol vs utbildning
utbildning = lambda a : a <= 1
compare_factors(16,utbildning,25,data,titles=['Högutbildade mammors alkoholkonsumption','Lågutbildade mammors alkoholkonsumption'],plot_mode=PLOT_HISTOGRAM)
