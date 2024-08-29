import numpy as np
import matplotlib.pyplot as plt
from network_gillespie import Network_Gillespie
import pandas as pd


N = 16
N_sqrt = int(np.sqrt(N))
volume = 300
time_end = 100
sample_num = 300
beta = 2
gamma = 1
epsilon_to_read = 0.1
epsilon_to_simulate = 0.01
threshold_minor = 0
r_0 = beta/gamma
starting_compartment = 1
initial_infected_num = 1

threshold_HIT = 1 - (1/r_0)

network_type = "lattice"
network_num = 30
random = False
trial_num = 8

#fraction_list = [0.3]
fraction_list = [0.3,0.4,0.45,0.5,0.55,0.6,0.7]

infected_num_for_each_frac = []

to_save = np.zeros((network_num*len(fraction_list),2 * trial_num + 1)) # it is converted into a DataFrame later
# with the structure
#            fraction   trial1_comp   trial2_comp    ...   trial1_infected_num  trial2_infected_num ...
# network1      0.4        5              3          ...     20                  30                 ...
# network2      0.4        7              71         ...     20                  30
# ...
# networkK       0.4       1              27          ...     20                   30                 ...

for fraction in fraction_list:
    to_save[network_num * fraction_list.index(fraction):network_num * (fraction_list.index(fraction) + 1),0] = fraction
    if network_type == "all-to-all":
        if random:
            csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/complete_network/N_16/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        else:
            csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/complete_network/N_16/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
    elif network_type == "lattice":
        if random:
            csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        else:
            csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        
    df = pd.read_csv(csv_file_path)
    
    # Convert the DataFrame to a numpy array
    data_array = df.to_numpy()
    network_num = data_array.shape[0]
    if fraction == fraction_list[0]:
        pass
        """plt.imshow(data_array[0].reshape(N_sqrt,N_sqrt), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.show()"""
    
    infected_num_for_each_network = []
    for network in range(data_array.shape[0]):

        # Convert the DataFrame to a numpy array
        print(fraction)
        data_array = df.to_numpy()
        immunised_network = data_array[network]
        print("sanity check",np.sum(immunised_network),volume*N*fraction)

        outbreak_num_original = np.count_nonzero(np.array([immunised_network > threshold_minor]))
        outbreak_num_original = int(outbreak_num_original)

        total_infected_num_for_each_trial = np.zeros(trial_num)
        picked_starting_compartments = np.random.randint(1,N+1,trial_num)

        for trial in range(trial_num):
            infected_num_array = np.zeros(N)

            starting_compartment = int(picked_starting_compartments[trial])

            to_save[network_num * fraction_list.index(fraction) + network,trial+1] = starting_compartment

            init_state = np.zeros(3 * N)
            init_state[:N] = volume
            init_state[2*N:] += immunised_network
            init_state[:N] -= immunised_network
            init_state[starting_compartment-1] -= initial_infected_num
            init_state[starting_compartment-1 + N] += initial_infected_num

            n_comp = Network_Gillespie(init_state,time_end,sample_num, beta,gamma,volume, epsilon_to_simulate, threshold_minor,N,network_type,collect_outbreak = True,starting_compartment=starting_compartment)
            epidemic_size, mean_chemical_states = n_comp.sampling()
            outbreak_list = n_comp.outbreak_list
            infected_num_total = n_comp.infected_num_total
            if infected_num_total > (volume * N * (1 - fraction)):
                print(infected_num_total)
                raise ValueError("The total infected number is greater than the expected value")
            to_save[network_num * fraction_list.index(fraction) + network,trial + trial_num+1] = infected_num_total

# save fraction_list and infected_num_for_each_frac_and_each_network
# merge the two arrays
# Assuming infected_num_for_each_frac_and_each_network, fraction_list, and network_num are already defined

print(to_save)

# Generate the network index labels
network_index = [f"network{i}" for i in range(network_num)] * len(fraction_list)

# Create the column names for the DataFrame
columns = ["fraction"] + [f"trial{i}_comp" for i in range(trial_num)] + [f"trial{i}" for i in range(trial_num)]

# Create the DataFrame
df = pd.DataFrame(to_save, columns=columns, index = network_index)
print(df)

# take mean over all trials and all the networks
infected_num_averaged_over_network = np.mean(to_save[:,trial_num:],axis = 1)

infected_num_for_each_frac = np.zeros(len(fraction_list))
for k in range(len(fraction_list)):
    infected_num_for_each_frac[k] = np.mean(infected_num_averaged_over_network[k * network_num:(k+1) * network_num])
plt.plot(fraction_list,infected_num_for_each_frac)
plt.xlabel("Fraction of immunised individuals")
plt.ylabel("Infected number")
plt.title(f"Infected number (averaged over trial and networks)\n vs fraction of immunised individuals for {N} compartments")

if network_type == "all-to-all":
    if random:
        df.to_csv(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/complete_network/infected_num/complete_(random)infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.csv")
        plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/complete_network/infected_num/complete_(random)infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.png")
    else:
        df.to_csv(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/complete_network/infected_num/complete_infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.csv")
        plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/complete_network/infected_num/complete_infected_num_vs_fraction_{N}_I_{initial_infected_num}_eps_{epsilon_to_simulate}_addition_final.png")

elif network_type == "lattice":
    if random:
        df.to_csv(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/infected_num/lattice_(random)infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.csv")
        plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/infected_num/lattice_(random)infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.png")
    else:
        df.to_csv(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/infected_num/lattice_infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.csv")
        plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/infected_num/lattice_infected_num_vs_fraction_{N}_eps_{epsilon_to_simulate}_I_{initial_infected_num}_final.png")