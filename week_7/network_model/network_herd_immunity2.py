import numpy as np
import matplotlib.pyplot as plt
from network_gillespie import Network_Gillespie
import pandas as pd

"""
This code is to simulate the spread of infection on a immunised network. The objective of this analysis is to check at what condition the population is protected against the infectious disease. Or, rather, is there any such conditions which ensures the protection in SIR model in compartment based network model simulated stochastically. The simulation is done for each networks given and then the average of the infected number is calculated. The average is then plotted against the fraction of immunised individuals. The full analysis using box-plot is done with herd_immunity_box_plot.py.
### check above

This code accepts the input of immunised networks from HI_generate_networks.py. Then, the simulation of infection is done for different immunised networks with the starting compartment chosen randomly. Each result of the simulation is characterised by three numbers: fraction of the immunised population, network number, and the starting compartment. The first number is the fraction of the total immunised individuals in the population (= total immunised population / total population). Since the immunisation by HI_generate_networks.py are probabilistic (both natural immunity and random immunity), different networks are produced by simulating the infection stochastically. ```total_network_num''' gives the number of such different networks generated for each value of the fraction of the total immunised individuals. The last number, starting compartment is there because often the networks we deal with have a lot of compartments. So, the compartment we introduce the first infection matters. We choose ``` trial_num''' compartments randomly as such starting compartments and start the simulation by introducing initial one infection. 

As usual, such simulation is iterated on same network and the same starting compartment ```sample_num''' times. 

Then, the total number of newly formed infection (the number of infections spread on such immunised networks by the introduction of the initial one infection, not taking into the account of the already immunised population) are counted and stored in a csv file. 
"""

N = 16
N_sqrt = int(np.sqrt(N))
volume = 300
time_end = 100 # take a large number but not too large in order not to make the data size too large
sample_num = 300 # number of iterations 
beta = 2
gamma = 1
epsilon_to_read = 0.1 # the value of epsilon HI_generate_networks.py used to generate the immunised networks
epsilon_to_simulate = 0.01 # the value of epsilons we use to simulate the infection on the immunised networks
initial_infected_num = 1 # if you want to speed up, or just want to ignore the branching process type of extinction of the infection, put 10 or something.
threshold_minor = 0 # just dummy variable. We do not need this but the class requires this


network_type = "lattice" # "lattice" or "all-to-all"
total_network_num = 30 # refer to the description
random = False # if True, then random immunity is chosen. if False, then the natural immunity is chosen
trial_num = 8 # number of starting compartments we choose

#fraction_list = [0.3]
fraction_list = [0.3,0.4,0.45,0.5,0.55,0.6,0.7] # match this with HI_generate_networks.py

r_0 = beta/gamma 
threshold_HIT = 1 - (1/r_0)

infected_num_for_each_frac = []

path_header = "dissertation_codes/week_7/data/herd_immunity/"

to_save = np.zeros((total_network_num*len(fraction_list),2 * trial_num + 1)) # the variable to store the data
# it is converted into a DataFrame later
# with the structure
#            fraction   trial1_comp   trial2_comp    ...   trial1_infected_num  trial2_infected_num ...
# network1      0.4        5              3          ...     20                  30                 ...
# network2      0.4        7              71         ...     20                  30
# ...
# networkK       0.4       1              27          ...     20                   30                 ...

# import the generated immunised networks from other csv files
# iterate over the fraction of the immunised population
for fraction in fraction_list:
    to_save[total_network_num * fraction_list.index(fraction):total_network_num * (fraction_list.index(fraction) + 1),0] = fraction
    if network_type == "all-to-all":
        if random:
            csv_file_path = path_header + f"complete_network/N_16/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        else:
            csv_file_path = path_header + f"complete_network/N_16/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
    elif network_type == "lattice":
        if random:
            csv_file_path = path_header + f"lattice_network/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        else:
            csv_file_path = path_header + f"lattice_network/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        
    df = pd.read_csv(csv_file_path)
    
    # Convert the DataFrame to a numpy array
    data_array = df.to_numpy()
    total_network_num = data_array.shape[0]
    if fraction == fraction_list[0]:
        pass
        # uncomment to visualise the first immunised network from the file
        """plt.imshow(data_array[0].reshape(N_sqrt,N_sqrt), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.show()"""
    
    # Convert the DataFrame to a numpy array
    print("fraction",fraction)
    data_array = df.to_numpy()

    # iterate over each network in the csv file imported (with the fixed fraction of the immunised population)
    # variable ```network_index''' is the index of the network
    for network_index in range(data_array.shape[0]):
        print("network num",network_index)

        immunised_network = data_array[network_index]
        print("check if immunised population agree",np.sum(immunised_network),volume*N*fraction)

        outbreak_num_original = np.count_nonzero(np.array([immunised_network > threshold_minor])) # the number of major outbreaks in the immunised network
        outbreak_num_original = int(outbreak_num_original)

        total_infected_num_for_each_trial = np.zeros(trial_num) # given an immunised network and fraction, the variable stores the number of infected population for each trial (infection simulation for different starting compartment)
        picked_starting_compartments = np.random.randint(1,N+1,trial_num) # starting compartment for this trial

        for trial in range(trial_num):
            infected_num_array = np.zeros(N)

            starting_compartment = int(picked_starting_compartments[trial])

            to_save[total_network_num * fraction_list.index(fraction) + network_index,trial+1] = starting_compartment # check the beginning of the code to check the indexing

            init_state = np.zeros(3 * N)
            init_state[:N] = volume
            init_state[2*N:] += immunised_network # recovered population is set by the immunised network
            init_state[:N] -= immunised_network # susceptible population is subtrated by the immunised population
            init_state[starting_compartment-1] -= initial_infected_num
            init_state[starting_compartment-1 + N] += initial_infected_num # adding the initial infection to the network

            n_comp = Network_Gillespie(init_state,time_end,sample_num, beta,gamma,volume, epsilon_to_simulate, threshold_minor,N,network_type,collect_outbreak = True,starting_compartment=starting_compartment)
            epidemic_size, mean_chemical_states = n_comp.sampling()
            infected_num_total = n_comp.infected_num_total
            if infected_num_total > (volume * N * (1 - fraction)):
                print(infected_num_total)
                raise ValueError("The total infected number is greater than the expected value")
            to_save[total_network_num * fraction_list.index(fraction) + network_index,trial + trial_num+1] = infected_num_total

# reminder of the structure of to_save
#            fraction   trial1_comp   trial2_comp    ...   trial1_infected_num  trial2_infected_num ...
# network1      0.4        5              3          ...     20                  30                 ...
# network2      0.4        7              71         ...     20                  30
# ...
# networkK       0.4       1              27          ...     20                   30                 ...

# save fraction_list and infected_num_for_each_frac_and_each_network
# merge the two arrays
# Assuming infected_num_for_each_frac_and_each_network, fraction_list, and total_network_num are already defined

print(to_save)

# Generate the network index labels
network_name_all = [f"network{i}" for i in range(total_network_num)] * len(fraction_list)

# Create the column names for the DataFrame
columns = ["fraction"] + [f"trial{i}_comp" for i in range(trial_num)] + [f"trial{i}_infected_num" for i in range(trial_num)]

# Create the DataFrame
df = pd.DataFrame(to_save, columns=columns, index = network_name_all)
print(df)

# take mean over all trials
infected_num_averaged_over_trials = np.mean(to_save[:,trial_num:],axis = 1)
print("infected_num_averaged_over_trials",infected_num_averaged_over_trials)

infected_num_for_each_frac = np.zeros(len(fraction_list)) # to store the infected population for each fraction of immunised population

# make a plot of secondary infection on the network (labeled as "infected number") versus the fraction of immunised population of the network
for k in range(len(fraction_list)):
    infected_num_for_each_frac[k] = np.mean(infected_num_averaged_over_trials[k * total_network_num:(k+1) * total_network_num])
plt.plot(fraction_list,infected_num_for_each_frac)
plt.xlabel("Fraction of immunised individuals")
plt.ylabel("Infected number")
plt.title(f"Infected number (averaged over trial and networks)\n vs fraction of immunised individuals for {N} compartments")

# store the data as csv files
if network_type == "all-to-all":
    if random:
        df.to_csv(path_header + f"complete_network/infected_num/complete_(random)infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.csv")
        plt.savefig(path_header + f"complete_network/infected_num/complete_(random)infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.png")
    else:
        df.to_csv(path_header + f"complete_network/infected_num/complete_infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.csv")
        plt.savefig(path_header + f"complete_network/infected_num/complete_infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.png")

elif network_type == "lattice":
    if random:
        df.to_csv(path_header + f"lattice_network/infected_num/lattice_(random)infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.csv")
        plt.savefig(path_header + f"lattice_network/infected_num/lattice_(random)infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.png")
    else:
        df.to_csv(path_header + f"lattice_network/infected_num/lattice_infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.csv")
        plt.savefig(path_header + f"lattice_network/infected_num/lattice_infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.png")

plt.show()