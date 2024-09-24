from SIR_gillespie import SIR_Gillespie
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw
import pandas as pd
import matplotlib as mpl
from scipy.integrate import odeint


"""
Code to plot the outbreak size of the SIR model for different basic reproduction number specified by 'R_list' based on the simulation with the initial state 'init_state' and the parameters 'gamma' and 'beta'. The plot is accompanied by the outbreak size calcualted from deterministic model (calculated by Lambert W function).
"""

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']) 
# Obtain the colors from the current color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

infected_rate = 0.001
recovery_rate = 0.0
total_num = 1000
init_state = (
    np.array([1 - infected_rate - recovery_rate, infected_rate, recovery_rate]) * total_num
)
init_state = init_state.astype(int)
time_end = 1000
sample_num = 1000
R_list = [1.2,2]
#R_list = [1.22]
volume = np.sum(init_state)
gamma = 1

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
for j,r_0 in enumerate(R_list):
    # calcualte the outbreak size of the deterministic model
    beta = gamma * r_0
    r_infty = 1 + lambertw(
            -(1 - infected_rate - recovery_rate) * r_0 * np.exp(-r_0 * (1 - recovery_rate))
        ) / (r_0)
    peak_size = r_infty.real * total_num
    if r_0 > 1:
        threshold_minor = peak_size * 0.2
    else:
        threshold_minor = 0

    # run the simulation
    sir = SIR_Gillespie(init_state, time_end, sample_num, gamma, beta, volume,threshold_minor)
    outbreak_sizes,interpolant_time,mean_chemical_state = sir.sampling()
    survival_time = sir.inf_end_list
    survival_time = np.array(survival_time)

    print(outbreak_sizes)
    outbreak_sizes = np.array(outbreak_sizes)
    # Plotting the histogram in the first subplot
    # count the frequency of each outbreak size
    count = np.zeros(total_num)
    for i in range(0,total_num):
        #print(outbreak_sizes == i)
        count[i] = np.count_nonzero(outbreak_sizes == i)
    print(count[0])
    
    counts, bin_edges = np.histogram(outbreak_sizes, bins=100)
    # calculate the average of outbreak sizes of major outbreak
    major_average = np.mean(outbreak_sizes[outbreak_sizes > threshold_minor])
    average = np.mean(outbreak_sizes)
    # generate bins and counts for histogram
    if r_0 < 1:
        ax[j].hist(outbreak_sizes, bins=100, color="blue", alpha=0.7,width=1)
    else:
        ax[j].hist(outbreak_sizes, bins=100, color="blue", alpha=0.7)
    #ax1.plot(np.arange(1,total_num+1),count)
    if r_0 > 1:
        ax[j].vlines(average, 0, np.max(counts), label=f"Stochastic mean: {average:.0f}", color='red')
        ax[j].vlines(peak_size, 0, np.max(counts), label=f"Deterministic: {peak_size:.0f}",linestyle="--", color='black')
        ax[j].vlines(major_average, 0, np.max(counts), label=f"Stochastic mean of \n   major outbreak: {major_average:.0f}", color='red',linestyle=":")
        ax[j].vlines(threshold_minor, 0, np.max(counts), label="Threshold of the major outbreak", color='green',linestyle="dashdot")
        ax[j].legend(fontsize=15)
    #if r_0 > 1:
        # put the cross mark to indicate the extinction probability
        #ax[j].scatter(1,1/r_0 * sample_num,marker="+",color="red",label="Extinction")
    ax[j].set_title(fr"Final number of recovered ($R_0={r_0}$)",fontsize=15)
    ax[j].set_xlabel("Recovered",fontsize=15)
    ax[j].set_ylabel("Frequency",fontsize=15)
    ax[j].set_ylim(0,100)
    

#plt.legend(fontsize=15)
plt.savefig("dissertation_codes/one_compartment/data/SIR_outbreak_size3.png",bbox_inches = "tight",dpi=300)
plt.show()

