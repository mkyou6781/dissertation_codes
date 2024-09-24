from SIR_gillespie import SIR_Gillespie
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw
import pandas as pd
import matplotlib as mpl
from scipy.integrate import odeint

"""
Code to plot the time dynamics of the recovered population R of the SIR model for different basic reproduction number specified by 'R_list' based on the simulation with the initial state 'init_state' and the parameters 'gamma' and 'beta'. The plot is made for both stochastic and deterministic models.
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
#init_state = np.array([299,1,0])
init_state = init_state.astype(int)
time_end = 100
sample_num = 100
R_list = [0.5,1,2,3,9]
#R_list = [1]
volume = 1000
gamma = 1

# SIR model differential equations
def sir_model(y, t, volume, beta, gamma):
    S, I, R = y # given in terms of concentration
    dSdt = -beta * S * I / volume
    dIdt = beta *S * I / volume - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

for j,r_0 in enumerate(R_list):
    beta = gamma * r_0
    threshold_minor = 0
    sir = SIR_Gillespie(init_state, time_end, sample_num, gamma, beta, volume,threshold_minor)

    cluster_sizes,interpolant_time,mean_chemical_state = sir.sampling()
    survival_time = sir.inf_end_list
    survival_time = np.array(survival_time)
    #print(mean_chemical_state)
    plt.plot(interpolant_time,mean_chemical_state[:,2],color=colors[j],alpha=0.7,linestyle="-")

    time = np.linspace(0,time_end,time_end*50)
    # compare against the deterministic solution
    result = odeint(sir_model, init_state, time, args=(volume, beta, gamma))
    S, I, R = result.T
    print("outbreak size",R[-1])
    plt.plot(time,R,linestyle="--",color = colors[j])
    # show R value above each curve


    plt.text(time_end, R[-1], fr"$R_0=${r_0}", fontsize=12, ha='right')

    # Plotting the histogram in the first subplot
    count = np.zeros(total_num)
    for i in range(1,total_num):
        #print(cluster_sizes == i)
        count[i] = np.count_nonzero(cluster_sizes == i)
    #print(count)

plt.plot([],[],linestyle="-",color="k",label = "stochastic")
plt.plot([],[],linestyle="--",color="k",label = "deterministic")

plt.legend(fontsize=12)
plt.xlabel('Time',fontsize=12)
plt.ylabel('Recovered R',fontsize=12)
plt.savefig("dissertation/one_compartment/data/SIR_plot3.png",bbox_inches = "tight")
plt.show()

