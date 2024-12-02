import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.special as sp

def test_rayleigh_parameter(
        M = 1000,
        b = 4,
):
    
    expected_value = b*np.sqrt(np.pi/2)

    # simulera M stycken Rayleigh-fördelade värden med parameter b
    x = stats.rayleigh.rvs(loc=0, scale=b, size=M)
    
    fig, ax = plt.subplots(figsize=(8,4))

    # ---- ML-skattning ----
    ML_parameter = np.sqrt(np.sum(x**2) * 1/(2*M)) # denna är biased (men spelar inte så stor roll för stora M)
    ML_expected_value = ML_parameter*np.sqrt(np.pi/2) # beräkna förväntat värde för fördelningen för ML-skattning
    # ---- MK-skattning ----
    MK_parameter = np.sum(x)/M * np.sqrt(2/np.pi) # denna är unbiased
    MK_expected_value = MK_parameter*np.sqrt(np.pi/2) # beräkna förväntat värde för fördelningen MK-skattning

    # MK-skattning

    # plotta histogram
    ax.hist(x, bins=100, density=True, alpha=0.75)
    ax.set_title(f'Histogram för Rayleigh-fördelning med b={b}')

    # VÄLJ EN AV PARAMETERVÄRDE ELLER SKATTNING AV VÄNTEVÄRDE ATT PLOTTA!
    # plotta parametervärde 
    ax.plot([b], [0.2], 'ro')
    ax.plot([ML_parameter], [0.2], 'go')
    ax.plot([MK_parameter], [0.2], 'bo') 
    # plotta förväntat värde för fördelningen
    ax.plot([expected_value, expected_value], [0, 0.2], 'r') # plotta faktiskt värde
    ax.plot([ML_expected_value, ML_expected_value], [0, 0.2], 'g')
    ax.plot([MK_expected_value, MK_expected_value], [0, 0.2], 'b')

    # legend
    ax.legend(['Faktiskt b-värde', 'ML-skattning av b (biased)', 'MK-skattning (unbiased)','medelvärde','ML-skattning av väntevärde','MK-skattning av väntevärde'])

    # beräkna och plotta pdf för ML och MK
    pdf_x = np.linspace(np.min(x),np.max(x), 60)
    pdf_y_ML = stats.rayleigh.pdf(pdf_x, scale=ML_parameter)
    pdf_y_MK = stats.rayleigh.pdf(pdf_x, scale=MK_parameter)
    ax.plot(pdf_x, pdf_y_ML, 'g', label='ML-skattning biased')
    ax.plot(pdf_x, pdf_y_MK, 'b', label='MK-skattning biased')



    plt.show()
    




test_rayleigh_parameter(1000)

    

