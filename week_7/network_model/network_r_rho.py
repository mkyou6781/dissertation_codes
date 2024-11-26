import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network_gillespie import Network_Gillespie
import scipy as sp
from scipy.optimize import fsolve

# function to calculate R_eff
def R_eff(adj_matrix, beta, gamma, epsilon, volume, N, immunised_array):
    # note the adj_matrix needs to have 1 - epsilon on the diagonal
    next_generation_matrix = np.zeros((N,N))
    immunised_conc = (volume - immunised_array)/volume
    # make a diagonal matrix with its diagonal elements being immunised_conc
    immunised_matrix = np.diag(immunised_conc)
    next_generation_matrix = np.matmul(adj_matrix,immunised_matrix)
    # obtain the spectral radius of the next generation matrix
    r_eff = beta / gamma * np.max(np.linalg.eigvals(next_generation_matrix))
    return r_eff

# function to get the coefficient matrix for the branching process approximation
def get_coeff_matrix(adj_matrix, beta, gamma, epsilon, volume, N, immunised_array):
    # note the adj_matrix needs to have 1 - epsilon on the diagonal
    immunised_conc = (volume - immunised_array)/volume
    coeff_matrix = np.zeros((N,N+1))
    coeff_matrix[:,0] = gamma
    coeff_matrix[:,1:] = beta * np.matmul(adj_matrix,np.diag(immunised_conc))
    # normalise the rows with the sum of the rows
    coeff_matrix = coeff_matrix / np.sum(coeff_matrix,axis=1)[:,np.newaxis]
    return coeff_matrix

def rho(adj_matrix, beta, gamma, epsilon, volume, N, immunised_array):
    # note the adj_matrix needs to have 1 - epsilon on the diagonal
    m_matrix = np.zeros((N,N))
    coeff_matrix = get_coeff_matrix(adj_matrix, beta, gamma, epsilon, volume, N, immunised_array)

    m_matrix += 2 * np.diag(coeff_matrix[np.arange(N),1+np.arange(N)])
    mask = np.ones((N,N+1))
    mask[np.arange(N),1+np.arange(N)] = 0
    m_matrix += np.multiply(coeff_matrix[:,1:],mask[:,1:]) # keep the diagonal unaffected by the addition
    # add diagonal elements
    diag_element = np.sum(np.multiply(coeff_matrix[:,1:],mask[:,1:]),axis=1)
    m_matrix += np.diag(diag_element)

    # obtain the spectral radius of m matrix
    rho = np.max(np.linalg.eigvals(m_matrix))
    return rho

# make a polynomial equation the branching process approximation need to solve
def equations(s, A, N):
    eqs = np.zeros(N)
    for i in range(N):
        eqs[i] = s[i] - (A[i,0] + np.sum(A[i, 1:] * s[i] * s))
    return eqs

# function to calculate the extinction probability q of the branching process approximation
def calc_q(adj_matrix, beta, gamma, epsilon, volume, N, immunised_array):
    coeff_matrix = get_coeff_matrix(adj_matrix, beta, gamma, epsilon, volume, N, immunised_array)
    s_initial_guess = np.ones(N) * 0.5
    solution = fsolve(equations, s_initial_guess, args=(coeff_matrix, N))
    return solution


# the following is for a test
"""n_comp = Network_Gillespie(init_state,time_end,sample_num, beta,gamma,volume, epsilon, threshold_minor,N,network_type,starting_compartment=starting_compartment,collect_outbreak=True)
n_comp.generate_adj_matrix()

adj_matrix = n_comp.adj_matrix
##print(adj_matrix)
# convert adj_matrix to dense matrix
adj_matrix = sp.sparse.csr_matrix.todense(adj_matrix)
np.fill_diagonal(adj_matrix, (1-epsilon))

 


# import the data from herd immunity networks
# make a plot of R_eff against rho
R_eff_list = []
rho_list = []
for fraction in fraction_list:
    
    if random:
        csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
    else:
        csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/herd_immunity/lattice_network/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
        
    df = pd.read_csv(csv_file_path)
    
    # Convert the DataFrame to a numpy array
    data_array = df.to_numpy()
    network_num = data_array.shape[0]

    if fraction == fraction_list[0]:
        infected_num_for_each_frac_and_each_network = np.zeros((len(fraction_list),network_num))
    
    infected_num_for_each_network = []
    for j in range(data_array.shape[0]):
    #for j in range(1):
        ##print("network",j)
        # Convert the DataFrame to a numpy array
        data_array = df.to_numpy()
        immunised_network = data_array[j]
        #immunised_network = np.zeros(N)

        r_eff = R_eff(adj_matrix, beta, gamma, epsilon, volume, N, immunised_network)
        #print("r_eff",r_eff)
        rho_val = rho(adj_matrix, beta, gamma, epsilon, volume, N, immunised_network)
        #print("rho",rho_val)
        R_eff_list.append(r_eff)
        rho_list.append(rho_val)

plt.scatter(rho_list,R_eff_list)
plt.xlabel("rho")
plt.ylabel("R_eff")
plt.show()"""