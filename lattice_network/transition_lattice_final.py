import numpy as np
import matplotlib.pyplot as plt
from lattice_monte_carlo import Lattice_Monte_Carlo
from scipy.special import lambertw
import pandas as pd
import matplotlib as mpl

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']) 
# Obtain the colors from the current color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

r_0 = 2
#epsilon_list = np.array([0.03,0.01,0.006])
SS_epsilon_list = np.array([0.06,0.03,0.01,0.006,0.003,0.001])
#epsilon_list = np.array([0.01])
volume = 300
N_list = [25,64,81,100]

sample_num = 1000
network_type = "lattice"
starting_compartment = 41


for j,N in enumerate(N_list):
    SS_outbreak_fraction_list = []
    SS_one_minus_alpha_list = []

    # Load the CSV file
    #csv_file_path_MC = f"/Users/bouningen0909/dissertation/week_7/data/monte_carlo/monte_carlo_outbreak_count{N}_comp_eps{epsilon}R{r_0}.csv"
    csv_file_path_MC = f"/Users/bouningen0909/dissertation/week_7/data/lattice_N_{N}/monte_carlo/N_{N}epsilon_prob_mass.csv"
    MC_df = pd.read_csv(csv_file_path_MC)
    # extract the data epsilon,probability_mass,one_minus_alpha
    MC_epsilon_list = MC_df["epsilon"]
    prob_mass_list = MC_df["probability_mass"]
    MC_one_minus_alpha_list = MC_df["one_minus_alpha"]
    MC_half_alpha_list = []

    MC_outbreak_fraction_list = prob_mass_list/N
    if N != 100:
        for k, epsilon in enumerate(SS_epsilon_list):
            lattice_monte_carlo = Lattice_Monte_Carlo(N, r_0,  epsilon,volume, sample_num, network_type,starting_compartment)

            alpha,peak_size = lattice_monte_carlo.calc_alpha()
            SS_one_minus_alpha_list.append(1 - alpha)
            #csv_file_path_SS = f"/Users/bouningen0909/dissertation/week_7/data/lattice_N_{N}/outbreak_count{N}_comp_eps{epsilon}R{r_0}.csv"
            csv_file_path_SS = f"/Users/bouningen0909/dissertation/week_7/data/lattice_N_{N}/outbreak_count{N}_comp_eps{epsilon}R2.csv"
            SS_df = pd.read_csv(csv_file_path_SS)

            # Drop the first column
            SS_df = SS_df.drop(SS_df.columns[0], axis=1)

            # Convert the DataFrame to a numpy array
            SS_data_array = SS_df.to_numpy()

            #print("SS shape",SS_data_array.shape)
            SS_outbreak_prob_by_comp = np.zeros(N)
            SS_outbreak_count = np.sum(SS_data_array,axis = 1)
            SS_center_outbreak_num = np.max(SS_outbreak_count)
            SS_outbreak_prob_by_comp = SS_outbreak_count / SS_center_outbreak_num

            SS_outbreak_fraction_list.append(np.sum(SS_outbreak_prob_by_comp)/N)
            # obtain the half alpha value
    else:
        pass
    half = (1 - 1/N) / 2 + 1/N
    quart_1 = (1 - 1/N) / 4 + 1/N
    quart_3 = 3 * (1 - 1/N) / 4 + 1/N
    print("half",half)

    half_diff = np.abs(MC_outbreak_fraction_list - half)
    half_alpha = MC_one_minus_alpha_list[np.argmin(half_diff)]
    #print(half_alpha)
    MC_half_alpha_list.append(half_alpha)

    quart_1_diff = np.abs(MC_outbreak_fraction_list - quart_1)
    quart_1_alpha = MC_one_minus_alpha_list[np.argmin(quart_1_diff)]
    #print(quart_1_alpha)

    quart_3_diff = np.abs(MC_outbreak_fraction_list - quart_3)
    quart_3_alpha = MC_one_minus_alpha_list[np.argmin(quart_3_diff)]
    #print(quart_3_alpha)


    #plt.plot(epsilon_list, prob_mass_list, color=colors[N_list.index(N)],label=f"N = {N}")
    plt.vlines(half_alpha,0,half,linestyle="--",color=colors[N_list.index(N) %len(colors)],label=rf"$N = {N}$"+r"$, 1 - \alpha_{1/2}$"+rf"$ = {half_alpha:.3f}$"+"\n"+ f"   width = {quart_3_alpha - quart_1_alpha:.3f}",alpha=0.5)
    # sort the value of epsilon_list, alpha_list and MC_outbreak_fraction_list in ascending order of epsilon
    plt.plot(MC_one_minus_alpha_list, MC_outbreak_fraction_list, color=colors[j],alpha=0.5,zorder=1)
    if N != 100:
        plt.scatter(SS_one_minus_alpha_list[1:], SS_outbreak_fraction_list[1:], color=colors[j], marker='+',zorder=2)
    else:
        pass
plt.scatter([],[], color='k', marker='+',label='Stochastic simulation')
plt.xlabel(r"$1 - \alpha$",fontsize=12)
plt.ylabel("Outbreak fraction",fontsize=12)
plt.title("Transition in two-dimensional lattice",fontsize=12)
plt.legend(fontsize=12,loc = 'best',bbox_to_anchor=(0.6, 0.11, 0.5, 0.5))
plt.savefig("/Users/bouningen0909/dissertation/week_7/data/lattice_transition/transition_plot.png",dpi=300,bbox_inches='tight')
plt.show()



