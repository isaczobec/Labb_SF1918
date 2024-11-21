import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp
import tools


def enkel_linjär_regression(col1,col2,titel):
    data = np.loadtxt('data/birth.dat')

    col1_col2 = data[:,(col1,col2)]
    # remove nans

    valid_indicies = ~np.isnan(col1_col2).any(axis=1)
    col1_col2 = col1_col2[valid_indicies]

    X_mat = np.column_stack((np.ones(col1_col2.shape[0]),col1_col2[:,1]))
    B = tools.regress(X_mat,col1_col2[:,0])[0]

    print('Skattning av Beta: ',B[1])

    # gör regressionslinje
    xs = np.linspace(np.min(col1_col2[:,1]),np.max(col1_col2[:,1]),2)
    ys = B[0] + B[1]*xs

    plt.plot(col1_col2[:,1],col1_col2[:,0],'o') # plotta data
    plt.plot(xs,ys,'r') # plotta regressionslinje
    plt.title(titel)
    plt.legend(['Data','Regressionslinje'])

    plt.show()

def multipel_linjär_regression(
        y_index: int,
        column_indexes: np.array,
        titel: str,
        variable_names: list[str],
        significance_level: float = 0.05,
        func_dict : dict[int,callable] = None
        ):
    data = np.loadtxt('data/birth.dat')

    columns = data[:,np.hstack(([y_index],column_indexes))]
    # remove nans
    valid_indicies = ~np.isnan(columns).any(axis=1)
    columns = columns[valid_indicies,:]

    # applicera funktioner på valda kolumner
    if func_dict is not None:
        for key in func_dict:
            columns[:,key] = func_dict[key](columns[:,key])

    X_mat = np.column_stack((np.ones(columns.shape[0]),columns[:,1:]))
    regress = tools.regress(X_mat,columns[:,0],alpha=significance_level)
    B = regress[0]
    intervals = regress[1]

    # print confidence intervals
    for i in range(1,len(B)):
        print(f'Confidence interval for Beta for {variable_names[i]}: lower:{intervals[i,0]} beta:{B[i]} upper:{intervals[i,1]}')

    

enkel_linjär_regression(2,15,'Barnets vikt som funktion av mammans längd')

multipel_linjär_regression(
        2,
        np.array([14,19,25]),
        'Barnets vikt',
        ['barnets vikt (g)','mammans vikt (kg)','rökning?','alkohol?'],
        significance_level=0.05,

        # funktioner för att konvertera kategoriska data till {0,1}
        func_dict= {
            2: lambda x: x == 3,
            3: lambda x: x == 2,
        }

        )

