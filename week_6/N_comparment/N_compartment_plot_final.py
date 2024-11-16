import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probability_analysis import N_prob

"""
The code to check if the data of outbreak frequency in complete network model fits the model imported from probability_analysis.py
"""

r_0 = 2
epsilon_list = [0.0001,0.003,0.006]
N = 16
volume = 1000

directory_to_import_data_from = "dissertation_codes/week_6/data/N_16_all_to_all_data"

file_path_1 = directory_to_import_data_from + "/num_outbreak16_comp_eps0.0001R2_test.csv"
file_path_2 = directory_to_import_data_from + "/num_outbreak16_comp_eps0.003R2_test.csv"
file_path_3 = directory_to_import_data_from + "/num_outbreak16_comp_eps0.006R2_test.csv"

data1 = pd.read_csv(file_path_1)
data2 = pd.read_csv(file_path_2)
data3 = pd.read_csv(file_path_3)

# get the data with columns outbreak_num,frequency
outbreak_num1 = data1['outbreak_num']
frequency1 = data1['frequency']
frequency2 = data2['frequency']
frequency3 = data3['frequency']
frequency = [frequency1,frequency2,frequency3]

fig, ax = plt.subplots(1,3,figsize=(15,5))
for i,epsilon in enumerate(epsilon_list):
    ax[i].bar(outbreak_num1,frequency[i], color="blue", width=1)
    ax[i].set_xlabel("Number of compartments",fontsize=15)
    ax[i].set_ylabel("Frequency",fontsize=15)
    ax[i].set_ylim(0,1000)
    ax[i].set_title(rf"$\epsilon = ${epsilon}",fontsize=15)
    #if i == 0:
        #ax[i].set_title(r"$\epsilon = 10^{-5}$",fontsize=15)

    prob = N_prob(r_0,epsilon,N,volume)
    print(prob)
    ax[i].scatter(outbreak_num1[1:],prob * (1 - 1/r_0)*volume,color="red",marker = "+",label="model")
    ax[i].scatter(0,1/r_0 * volume,color="red",marker = "+")
    ax[i].legend()
    # show ticks only at integer value of x-axis
    ax[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("dissertation_codes/week_6/data/final_plot/final_plot_16_3_all_to_all.png")
plt.show()