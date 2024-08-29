import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network_gillespie import Network_Gillespie

starting_compartment = 1
N = 36
volume = 300
init_state = np.zeros(3 * N)
init_state[0:N] = volume
init_state[starting_compartment-1] -= 10
init_state[starting_compartment-1 + N] += 10
time_end = 100
sample_num = 300 # dummy I dont need this
beta = 2
gamma = 1
#epsilon_list = [0.06,0.03,0.01,0.006,0.003,0.001]
epsilon = 0.1
threshold_minor = 100
network_type = "lattice"
r_0 = beta/gamma
networks_to_produce = 30
herd_immunity_fraction_stop_list = [0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7]

resulted_network = np.zeros((networks_to_produce,N))

for herd_immunity_fraction_stop in herd_immunity_fraction_stop_list:
    #epsilon_list = epsilon_list[::-1]
    print(herd_immunity_fraction_stop)
    network_gillespie = Network_Gillespie(init_state,time_end,sample_num, beta,gamma,volume, epsilon, threshold_minor,N,network_type,starting_compartment=starting_compartment,collect_outbreak=False,herd_mmunity=True,herd_immunity_fraction_stop = herd_immunity_fraction_stop)

    num_network_acquired = 0
    while num_network_acquired < networks_to_produce:
        print("new trial")
        network_gillespie.generate_adj_matrix()
        network_gillespie.gillespie()
        if network_gillespie.herd_immunity_flag:
            num_network_acquired += 1
            total_immunity_by_compartment = [network_gillespie.state[network_gillespie.N + i] + network_gillespie.state[2 * network_gillespie.N + i] for i in range(network_gillespie.N)]
            resulted_network[num_network_acquired - 1] = np.array(total_immunity_by_compartment)
            # sanity check
            print(total_immunity_by_compartment)
            print(np.sum(total_immunity_by_compartment),volume*N*herd_immunity_fraction_stop)


    """# plot the resulted network with two subplots
    fig,ax = plt.subplots(1,2,figsize=(10,5))

    # plot the resulted network with the number of infected more than the threshold
    more_than_threshold = resulted_network[0] > threshold_minor
    im = ax[0].imshow(more_than_threshold.reshape(int(np.sqrt(N)),int(np.sqrt(N))), cmap='viridis', aspect='auto',vmin=0, vmax=1)
    ax[0].set_title("Major outbreak")
    cbar = fig.colorbar(im, ax=ax[0],shrink=0.75)

    # plot the network with the number of infected more than the herd immunity threshold
    herd_immunity_threshold = 1 - (1/r_0)
    herd_immunity = resulted_network[0] > volume * herd_immunity_threshold
    im = ax[1].imshow(herd_immunity.reshape(int(np.sqrt(N)),int(np.sqrt(N))), cmap='viridis', aspect='auto',vmin=0, vmax=1)
    ax[1].set_title("Herd immunity")
    cbar = fig.colorbar(im, ax=ax[1],shrink=0.75)

    plt.tight_layout()"""
    #plt.show()


    # save the resulted network to a csv file
    df = pd.DataFrame(resulted_network, columns=[f"compartment{i}" for i in range(N)])
    csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/herd_immunity_network_N_{N}_comp_eps{epsilon}R{r_0}_fraction{herd_immunity_fraction_stop}.csv"
    df.to_csv(csv_file_path, index=False)

    # generate random networks with the same number of compartments and with the same number of infected, volume*N*herd_immunity_fraction_stop
    random_networks = np.zeros((networks_to_produce,N))

    def generate_integer_array(length, total_sum):
        if length > total_sum:
            raise ValueError("It's not possible to create an array with the given length and total sum.")
        
        # Start with an array of ones
        result = np.zeros(length, dtype=int)
        
        # Randomly distribute the remaining (total_sum - length) into the array
        remaining_sum = total_sum
        for _ in range(remaining_sum):
            index = np.random.randint(length)
            result[index] += 1
        
        return result

    for i in range(networks_to_produce):
        random_networks[i,:] = generate_integer_array(N, int(volume * N * herd_immunity_fraction_stop))
        print(random_networks[i,:])

    # save the resulted network to a csv file
    df = pd.DataFrame(random_networks, columns=[f"compartment{i}" for i in range(N)])
    csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/random_network_N_{N}_comp_eps{epsilon}R{r_0}_fraction{herd_immunity_fraction_stop}.csv"
    df.to_csv(csv_file_path, index=False)