import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from N_compartment_gillespie import N_Comp_Gillespie
from calc_R import calc_R
from probability_analysis import N_prob
import time 

"""
code for testing the N_compartment_gillespie.py
The code is iterated over different values of epsilon in epsilon_list. For each value of epsilon, the code is iterated over sample_num times. For each epsilon, the frequency of outbreak (how many local outbreak are there), the time until the infection ends, and the delay time list are saved in the csv file. The frequency is plotted as a bar plot, and the trajectory of the stochastic simulation is plotted. The parameters are displayed in the third subplot. The frequency is fitted against the model imported from probability_analysis.py. The fitting is plotted as a scatter plot. The plot is saved in the directory_to_save.
"""

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']) 
# Obtain the colors from the current color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

start = time.process_time()

N = 6
volume = 1000
init_state = np.zeros(3 * N)
init_state[0:N] = volume
init_state[0] -= 1
init_state[N] += 1
time_end = 50 
sample_num = 1000
beta = 2
gamma = 1
epsilon_list = [0.1,0.003,0.001]

threshold_minor = 300
r_0 = beta/gamma

sample_time_criteria = [0,1]

sample_traj = 10

directrory_to_save = "dissertation_codes/week_6/data/N_6_all_to_all_data/"

epsilon_list = epsilon_list[::-1]
for epsilon in epsilon_list:

    n_comp = N_Comp_Gillespie(init_state,time_end,sample_num, beta,gamma,volume, epsilon, threshold_minor,N,sample_time_criteria)
    epidemic_size, mean_chemical_states = n_comp.sampling()
    fitting = N_prob(r_0,epsilon,N,volume) * (1 - 1/r_0) *sample_num
    fitting = np.append(1/r_0 * sample_num, fitting)
    inf_end = n_comp.inf_end
    delay_list = n_comp.delay_time_list
    #print(delay_list)

    num_outbreak = np.zeros(N+1) # number of outbreak given there is at least one outbreak
    for i in range(sample_num):
        cluster = epidemic_size[:,i]
        outbreak_num = np.count_nonzero(np.array([cluster > threshold_minor]))
        num_outbreak[outbreak_num] += 1

    # uncomment the following code to save the data
    """results_df = pd.DataFrame({
        'outbreak_num' : np.arange(N+1),
        'frequency' : num_outbreak
    })

    # Save DataFrame to CSV
    csv_file_path = directory_to_save + "num_outbreak{}_comp_eps{}R{}_test.csv".format(N,epsilon,int(beta/gamma))
    #results_df.to_csv(csv_file_path, index=False)

    results_df = pd.DataFrame({
        'time till the infection ends' : n_comp.inf_end_list
    })

    # Save DataFrame to CSV
    csv_file_path = directory_to_save + "end_time_dist{}_comp_eps{}R{}_given_two_outbreak.csv".format(N,epsilon,int(beta/gamma))
    #results_df.to_csv(csv_file_path, index=False)

    # save the delay time list, which is 2D array with np.savetxt
    delay_list_csv_path = directory_to_save + "delay_time_list{}_comp_eps{}R{}_test.csv".format(N,epsilon,int(beta/gamma))
    #np.savetxt(delay_list_csv_path, delay_list, delimiter=",")
"""

    

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].bar(np.arange(1,N+1),num_outbreak[1:], color="blue", width=1)
    ax[0].scatter(np.arange(1,N+1),fitting[1:],color="red",marker = "+")
    ax[0].set_xlabel("Number of compartments")
    ax[0].set_ylabel("Frequency")
    ax[0].set_ylim(0,sample_num+5)

    def generate_labels(N):
        labels = []
        for prefix in ['S', 'I', 'R']:
            for i in range(1, N+1):
                labels.append(rf"${prefix}_{i}$")
        return labels

    # plotting the trajectory of stochastic simultion given there is a major outbreak at C1
    labels = generate_labels(N)
    print(labels)
    print(n_comp.samples.shape)
    for i in range(2*N,3*N):
        for j in range(sample_traj):
            print(j)
            ax[1].plot(n_comp.interpolant_time, n_comp.samples[j,:,i], color=colors[(i-N) % len(colors)],alpha=0.2)
        ax[1].plot(n_comp.interpolant_time, mean_chemical_states[:, i], label=f"{labels[i]} stochastic", color=colors[(i-N) % len(colors)])
        
    #ax[1].legend()
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Number")
    ax[1].set_xlim(0,inf_end)
    
    # Displaying parameters in the second subplot
    params_text = f"""Parameters:
    beta: {beta}
    gamma: {gamma}
    epsilon: {epsilon}
    Time End: {time_end}
    Sample Number: {sample_num}
    """

    ax[2].axis('off')
    ax[2].text(0.5, 0.5, params_text, fontsize=9, ha='center', va='center')
    # uncomment the following code to save the plot
    #plt.savefig(directory_to_save + "N{}_eps_{}_R_{}plot_test.png".format(N,epsilon,int(beta/gamma)))
    end = time.process_time()
    print("time",end-start)
    plt.show()