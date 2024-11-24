import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import exponential
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

"""
class for calculating the timedevelopment of chemicals in the multi-compartment SIR model inscribed in the complete network with size N with simple Gillespie algorithm 
Note: the state is expressed with numpy array with the following format [S_1, ...S_N, I_1, ..., I_N, R_1, ..., R_N]
"""


class Network_Gillespie:
    """
    class for the N-compartment stochastic SIR model
    
    parameters:
        init_state: numpy array (1 by 3N) 
            initial state
        time_end: float
            the time point which the simulation ends
        sample_num: int
            number of samples
        beta: float
            beta in SIR model
        gamma: float
            gamma in SIR model
        volume: float
            volume of each compartment
        epsilon: float
            epsilon
        threshold_minor: float
            threshold for minor outbreak
        N: int
            number of compartments
        network_type: str
            type of network ("lattice","3D-lattice","all-to-all","random","two-community"). "all-to-all" corresponds to the complete network, "random" corresponds to the random network with fixed degree given by variable ```degree'''. If "two-community" is chosen, ```degree1''', ```degree2''', ```eps12''' should be given. "lattice" corresponds to the 2D lattice network, "3D-lattice" corresponds to the 3D lattice network.
        starting_compartment: int (default: 0)
            the compartment where one infected individual is initially placed 
        collect_outbreak: bool (default: False)
            If True, the information about which compartment has a major outbreak is collected over all the iterations. The data is stored in the variable ```outbreak_array'''.
        herd_mmunity: bool (default: False)
            If True, the simulation is done to generate the configuration of immunised population. The simulation stops when the fraction of the immunised population (infected + recovered) reaches the fraction given by the variable ```herd_immunity_fraction_stop'''. 
        herd_immunity_fraction_stop=1: float (default: 1)
            The fraction of the immunised population at which the simulation stops (only valid when ```herd_immunity''' is True).
        degree: int (default: None)
            the degree of the random network (only valid when ```network_type''' is "random")
        adj_matrix: scipy.sparse.coo_matrix (default: None)
            the adjacent matrix of the network, which is input externally. Because only random network and "two-community" networks are probabilistic, the variable is only valid when ```network_type''' is "two-community" or "random". Note that the network needs to be unweighted (without epsilon distributed)
        degree1: int (default: None)
            the internal degree of the first community (only valid when ```network_type''' is "two-community")
        degree2: int (default: None)
            the internal degree of the second community (only valid when ```network_type''' is "two-community")
        eps12: float (default: None)
            the epsilon between the two communities (only valid when ```network_type''' is "two-community")
    Returns:
        outbreak_size: numpy array (N by sample_num)
            the size of the outbreak
        mean_chemical_states: numpy array (time_end*10 by 3N)
            the mean of the chemical states given there is a major outbreak in the compartment 1 (R_1[-1] > threshold_minor)
        interpolant_time: numpy array
            the time points for interpolation
        inf_end_list: list
            the list of the time at which the infection dies out
        inf_end: float
            the time at which the infection dies out. This variable is just for sanity check. If this is the same as the ```time_end''', the simulation is terminated by the specified end time and indicates that the infection does not die out by that time point. So, we need to extend the simulation time.
        samples: numpy array (N by length of interpolant_time)
            the sample trajectories given there is a major outbreak
        outbreak_array: numpy array (sample_num by N)
            the array recording which compartment has a major outbreak at each iteration. 1 for the major outbreak, 0 for the minor outbreak
        infected_num_total: float
            the total number of infected individuals over all the compartments averaged over the iterations
    """
    def __init__(self,init_state,time_end,sample_num, beta,gamma,volume, epsilon, threshold_minor,N,network_type,starting_compartment = 0, collect_outbreak = False,herd_mmunity = False,herd_immunity_fraction_stop=1,degree = None,adj_matrix = None,degree1=None,degree2=None,eps12 = None):
        self.init_state = init_state # two dimensional array
        self.init_sum = np.sum(init_state)
        self.time_end = time_end
        self.data = []  # To store data
        self.sample_num = sample_num
        self.gamma = gamma
        self.beta = beta
        self.volume = volume
        self.epsilon = epsilon
        self.threshold_minor = threshold_minor
        self.N = N
        self.network_type = network_type
        self.starting_compartment = starting_compartment -1
        self.collect_outbreak = collect_outbreak
        self.herd_immunity = herd_mmunity
        self.herd_immunity_fraction_stop = herd_immunity_fraction_stop
        self.herd_immunity_flag = False
        self.degree = degree
        self.adj_matrix = adj_matrix # in case there is adjacent matrix beforehand (Note it should not be weighted)
        self.degree1 = degree1
        self.degree2 = degree2
        self.eps12 = eps12

    def generate_adj_matrix(self):
        # the function to generate the adjacent matrix with the epsilon distributed.
        # note that diagonal components are 0 in the adjacent matrix because loops are not allowed in the network
        # however, later in ```gen_rate_vector''' function, the diagonal components are added to the rate vector
        # maybe I should fix this for clarity
        if self.network_type == "lattice":
            N_sqrt = int(np.sqrt(self.N))
            
            # Create the 2-D lattice graph
            G = nx.grid_2d_graph(N_sqrt, N_sqrt,periodic = True)
            # Get the adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)

            # note the distribution of epsilon is done here
            adj_matrix = adj_matrix * (self.epsilon / 4)
        elif self.network_type == "3D-lattice":
            N_cbrt = int(np.cbrt(self.N))
            
            # Create the 2-D lattice graph
            G = nx.grid_graph(dim=[N_cbrt, N_cbrt, N_cbrt], periodic=True)
            # Get the adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)

            # note the distribution of epsilon is done here
            adj_matrix = adj_matrix * (self.epsilon / 6)
        if self.network_type == "all-to-all":
            G = nx.complete_graph(self.N)
            # Get the adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)

            # note the distribution of epsilon is done here
            adj_matrix = adj_matrix * (self.epsilon / (self.N-1))
        
        elif self.network_type == "random":
            if self.adj_matrix is None:
                if self.degree >= self.N:
                    raise ValueError("Degree must be less than the number of nodes")

                # Generate the random regular graph
                G = nx.random_regular_graph(self.degree, self.N)
                adj_matrix = nx.adjacency_matrix(G)
                # note the distribution of epsilon is done here
                adj_matrix = adj_matrix * (self.epsilon / self.degree)
            else:
                adj_matrix = self.adj_matrix
                adj_matrix = adj_matrix * (self.epsilon / self.degree)
        elif self.network_type == "two-community":
            # two-community network. Used stochastic block model with two distinct blocks
            half_N = int(self.N / 2)
            if self.adj_matrix is None:
                if (self.degree1 >= half_N) or (self.degree2 >= half_N):
                    raise ValueError("Degree must be less than the number of nodes")
                G1 = nx.random_regular_graph(self.degree1, half_N)
                G2 = nx.random_regular_graph(self.degree2, half_N)
                
                # Relabel nodes of G2 to avoid overlap with G1 nodes
                G2 = nx.relabel_nodes(G2, {i: i + half_N for i in range(half_N)})
                
                # Create a new graph to combine G1 and G2
                G_combined = nx.Graph()
                G_combined.add_nodes_from(G1.nodes())
                G_combined.add_nodes_from(G2.nodes())
                G_combined.add_edges_from(G1.edges())
                G_combined.add_edges_from(G2.edges())
                
                # Connect nodes between G1 and G2
                edges_between = [(i, i + half_N) for i in range(half_N)]
                G_combined.add_edges_from(edges_between)
                
                adj_matrix = nx.adjacency_matrix(G)
                # distribute epsilon
                eps1 = (self.epsilon - self.eps12)  / self.degree1
                eps2 = eps1 * self.degree1 / self.degree2
                adj_matrix = adj_matrix[0:half_N,0:half_N] * eps1
                adj_matrix = adj_matrix[half_N:,half_N:] * eps2
                adj_matrix = adj_matrix[0:half_N,half_N:] * self.eps12
                adj_matrix = adj_matrix[half_N:,0:half_N] * self.eps12

            else:
                adj_matrix = self.adj_matrix
                adj_matrix = adj_matrix.todense()
                eps1 = (self.epsilon - self.eps12)  / self.degree1
                scaled_adj_matrix = np.copy(adj_matrix)
                scaled_adj_matrix = adj_matrix.astype(float)
                eps2 = eps1 * self.degree1 / self.degree2
                scaled_adj_matrix[0:half_N, 0:half_N] *= eps1  # Top-left quadrant
                scaled_adj_matrix[half_N:, half_N:] *= eps2    # Bottom-right quadrant
                scaled_adj_matrix[0:half_N, half_N:] *= self.eps12  # Top-right quadrant
                scaled_adj_matrix[half_N:, 0:half_N] *= self.eps12  # Bottom-left quadrant
                # turn back to sparse form
                adj_matrix = nx.adjacency_matrix(nx.from_numpy_array(scaled_adj_matrix))
        self.adj_matrix = adj_matrix


    def gen_rate_vector(self):
        # generate the rate vector which reflects the propensity function for each reaction
        adj_coo = self.adj_matrix.tocoo()
        # personal note: check (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html) coo separates a sparse matrix into row, column, data (values)

        # Extract the row indices, column indices, and values
        row = adj_coo.row
        col = adj_coo.col
        self.data = adj_coo.data
        self.len_data = len(self.data) # the number of non-zero and non-diagonal components in augumented adjacent matrix, represenging the number of reactions for the infection across compartments

        # Combine row and column indices into a single array of indices
        self.indices = np.vstack((row, col)).T
        # Initialize recovery and infection arrays
        recovery = self.gamma * self.state[self.N:2*self.N]
        infection = np.zeros(self.len_data + self.N)

        # infection across compartments
            # i: susceptible S_i
            # j: infected I_j
        infection[:self.len_data] = (
            self.data * self.beta * self.state[row] * self.state[self.N + col] / self.volume
        )

        # infection inside compartments
        # note that the diagonal components are 0 above this point because the loops are not allowed in the network
        infection[self.len_data:] = (
            self.beta * (1 - self.epsilon) * self.state[:self.N] * self.state[self.N:2*self.N] / self.volume
        )

        # Combine infection and recovery into the rate vector
        rate = np.concatenate((infection, recovery))
        return rate

    def waiting_time(self,rate):
        # choose the waiting time for the next reaction
        sum_rate = np.sum(rate)
        parameter = 1 / sum_rate
        return exponential(parameter)
    
    def choose_reaction(self,rate):
        # choose the reaction which occurs next
        index = np.arange(len(rate))
        probabilities = rate / (np.sum(rate))
        fired_reaction = np.random.choice(index, size=1, p=probabilities)
        return fired_reaction
    
    def update(self,index):
        # update the state of the system 

        # infection across compartment (S_i + I_j -> I_i + I_j)
        if 0 <= index < self.len_data:
            # infection across compartment
            indices = self.indices[index]
            row = indices[0][0] # i
            col = indices[0][1] # j
            self.state[row] -= 1
            self.state[row + self.N] += 1
        # infection inside compartment (S_i + I_i -> 2 I_i)
        elif self.len_data <= index < self.len_data + self.N:
            comp_of_interest = index - self.len_data
            self.state[comp_of_interest] -= 1
            self.state[comp_of_interest + self.N] += 1
        # recovery (I_i -> R_i)
        elif self.len_data + self.N <= index < self.len_data + 2 * self.N:
            comp_of_interest = index - self.len_data - self.N
            self.state[comp_of_interest + self.N] -= 1
            self.state[comp_of_interest + 2 * self.N] += 1
        
        # if the value of the ```index''' is not in the range of the number of reactions, raise an error
        else: 
            raise ValueError("The chosen reaction is not defined")
        
    
    def gillespie(self):
        chemical_development = np.zeros((3 * self.N,1)) # to store the discrete time developments of the states
        self.state = np.copy(self.init_state)
        chemical_development[:,0] = self.state # the first state is the initial state
        time_section = [] # store the time points
        time = 0
        time_section.append(time)
        self.end = 0 # the time at which the simulation is terminated
        self.herd_immunity_flag = False # 

        while True:
            rate = self.gen_rate_vector()
            waiting_time = self.waiting_time(rate)
            time += waiting_time
            index = self.choose_reaction(rate)
            self.update(index)

            chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
            time_section.append(time)

            """# just sanity check
            num_in_comp = np.array([self.state[i] + self.state[i+self.N] + self.state[i+2 * self.N] for i in range(self.N)])
            print("num conserv",(num_in_comp - np.ones(self.N) * self.volume))
            if np.count_nonzero(num_in_comp - np.ones(self.N) * self.volume) != 0:
                print("warning: the number in each compartment is not conserved")"""

            # if the simulation reaches the specified end time, terminate the simulation
            if time >= self.time_end:
                self.end = self.time_end
                break

            # if the infection dies out
            # add the last state and the final time for successful interpolation
            # then terminate the simulation
            elif np.count_nonzero(self.state[self.N:2*self.N]) == 0:
                time_section.append(self.time_end)
                chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
                self.end = time
                break

            # if the herd immunity is considered, i.e. ```herd_immunity''' is True
            elif self.herd_immunity:
                # if the specified fraction of the immunised population is reached, terminate the simulation to produce the configuration for the immunised population
                if np.sum(self.state[self.N:3*self.N]) >= self.herd_immunity_fraction_stop*(self.volume*self.N):
                    self.herd_immunity_flag = True
                    break

        return np.array(chemical_development), np.array(time_section)
    
    def sampling(self):
        # code to run the gillespie algorithm for the specified number of iterations (```sample_num''')

        # initial setup for the simulation, including the generation of the adjacent matrix
        self.inf_end = self.time_end # to store the time at which the infection dies out at one iteration
        self.generate_adj_matrix()
        outbreak_size = np.zeros((self.N,self.sample_num))

        self.interpolant_time = np.linspace(0,self.time_end,self.time_end*10) # finer time to take 
        #samples = np.zeros((len(self.interpolant_time),self.N,self.sample_num))
        self.samples = [] # to store the trajectories of the states which lead to a major outbreak
        self.inf_end_list = [] # to store the time at which the infection dies out at each iteration as a list
        infected_num = np.zeros((self.N,self.sample_num)) # to store the number of infected individuals at each compartment at the end of each iteration

        # if we intend to collect the information about at which compartment outbreak occurs, initialise the outbreak list
        if self.collect_outbreak:
            self.outbreak_array = np.zeros((self.sample_num,self.N))

        # iterate the simulation for the specified number of iterations, ```sample_num'''
        for i in range(self.sample_num):
            print(i)
            chemical_development, time_section = self.gillespie()

            # interpolate the time development of the states to the finer time points
            f = interp1d(time_section,chemical_development,kind="nearest",axis = 1)
            interpolated = f(self.interpolant_time)

            # if there is a major outbreak at the starting compartment, store the trajectory in the ```samples''' list
            if  self.state[self.starting_compartment + 2 * self.N] > self.threshold_minor:
                self.samples.append(np.transpose(interpolated))

            # if we intend to collect the information about at which compartment outbreak occurs, store the information in the ```outbreak_array'''
            if self.collect_outbreak:
                outbreak_indicator = np.array([self.state[2 * self.N + i] > self.threshold_minor for i in range(self.N)])
                outbreak_comp = np.flatnonzero(outbreak_indicator)
                self.outbreak_array[i,outbreak_comp] = 1

            # store the size of outbreak at each compartment and at each iteration
            outbreak_size[:,i] = self.state[2 * self.N: 3 * self.N]
            infected_num[:,i] = self.state[2*self.N:3*self.N] - self.init_state[2*self.N:3*self.N]
            
            # store the time at which the infection dies out
            self.inf_end_list.append(self.end)
            
        self.inf_end = np.max(self.inf_end_list)
        print("inf_end",self.inf_end, "set end time", self.time_end) # sanity check. Refer to the explanation of the variable ```inf_end'''
        self.samples = np.array(self.samples)
        if len(self.samples) == 0:
            raise ValueError("samples are empty")
        mean_chemical_states = np.mean(self.samples,axis = 0)
        self.infected_num_total = np.sum(np.mean(infected_num,axis = 1)) # note that the name of this variable is rather confusing. Please refer to the explanation of the variable ```infected_num_total'''
        return outbreak_size, mean_chemical_states
