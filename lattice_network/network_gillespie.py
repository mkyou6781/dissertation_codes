import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import exponential
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

"""
code for calculating the timedevelopment of chemicals in the random chemical reaction network 
with simple Gillespie algorithm 
Note: the state is expressed with numpy array with the following format [S_1, ...S_N, I_1, ..., I_N, R_1, ..., R_N]
"""


class Network_Gillespie:
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
        #print("adjacent matrix",self.adj_matrix)
        # DO I NEED TO COPY?


    def gen_rate_vector(self):

        adj_coo = self.adj_matrix.tocoo()
        # personal note: check (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html) coo separates a sparse matrix into row, column, data (values)

        # Extract the row indices, column indices, and values
        row = adj_coo.row
        col = adj_coo.col
        self.data = adj_coo.data
        self.len_data = len(self.data)

        # Combine row and column indices into a single array of indices
        self.indices = np.vstack((row, col)).T
        # Initialize recovery and infection arrays
        recovery = self.gamma * self.state[self.N:2*self.N]
        infection = np.zeros(self.len_data + self.N)

        # across compartment infection
            # i: susceptible S_i
            # j: infected I_j
        infection[:self.len_data] = (
            self.data * self.beta * self.state[row] * self.state[self.N + col] / self.volume
        )

        
        infection[self.len_data:] = (
            self.beta * (1 - self.epsilon) * self.state[:self.N] * self.state[self.N:2*self.N] / self.volume
        )

        # Combine infection and recovery into the rate vector
        rate = np.concatenate((infection, recovery))
        return rate

    def waiting_time(self,rate):
        sum_rate = np.sum(rate)
        parameter = 1 / sum_rate
        return exponential(parameter)
    
    def choose_reaction(self,rate):
        index = np.arange(len(rate))
        probabilities = rate / (np.sum(rate))
        fired_reaction = np.random.choice(index, size=1, p=probabilities)
        return fired_reaction
    
    def update(self,index):
        # infection inside compartment
        # print(index)
        if 0 <= index < self.len_data:
            # print("infection across compartment occured")
            indices = self.indices[index]
            row = indices[0][0]
            col = indices[0][1]
            # IS THE ORDER RIGHT?
            self.state[row] -= 1
            self.state[row + self.N] += 1
        elif self.len_data <= index < self.len_data + self.N:
            #print("infection inside compartment occured")
            comp_of_interest = index - self.len_data
            self.state[comp_of_interest] -= 1
            self.state[comp_of_interest + self.N] += 1
        elif self.len_data + self.N <= index < self.len_data + 2 * self.N:
            #print("recovery occured")
            comp_of_interest = index - self.len_data - self.N
            self.state[comp_of_interest + self.N] -= 1
            self.state[comp_of_interest + 2 * self.N] += 1
        
        else: 
            raise ValueError("The chosen reaction is not defined")
        
    
    def gillespie(self):
        chemical_development = np.zeros((3 * self.N,1))
        self.state = np.copy(self.init_state)
        chemical_development[:,0] = self.state
        time_section = []
        time = 0
        time_section.append(time)
        self.end = 0
        self.herd_immunity_flag = False

        while True:
            #print(self.state)
            #print(time)
            rate = self.gen_rate_vector()
            waiting_time = self.waiting_time(rate)
            time += waiting_time
            index = self.choose_reaction(rate)
            self.update(index)
            #print(self.state)

            chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
            time_section.append(time)

            """# just sanity check
            num_in_comp = np.array([self.state[i] + self.state[i+self.N] + self.state[i+2 * self.N] for i in range(self.N)])
            print("num conserv",(num_in_comp - np.ones(self.N) * self.volume))
            if np.count_nonzero(num_in_comp - np.ones(self.N) * self.volume) != 0:
                print("warning: the number in each compartment is not conserved")"""

            if time >= self.time_end:
                self.end = self.time_end
                break

            #if the simulation terminates before the time end,
            # when the infection dies out
            # add the last state and the final time for successful interpolation
            elif np.count_nonzero(self.state[self.N:2*self.N]) == 0:
                time_section.append(self.time_end)
                chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
                self.end = time
                break

            elif self.herd_immunity:
                if np.sum(self.state[self.N:3*self.N]) >= self.herd_immunity_fraction_stop*(self.volume*self.N):
                    self.herd_immunity_flag = True
                    break

        return np.array(chemical_development), np.array(time_section)
    
    def sampling(self):
        self.inf_end = self.time_end
        self.generate_adj_matrix()
        epidemic_size = np.zeros((self.N,self.sample_num))

        self.interpolant_time = np.linspace(0,self.time_end,self.time_end*10) # finer time to take 
        #samples = np.zeros((len(self.interpolant_time),self.N,self.sample_num))
        self.samples = []
        self.inf_end_list = []
        infected_num = np.zeros((self.N,self.sample_num))
        if self.collect_outbreak:
            self.outbreak_list = np.zeros((self.sample_num,self.N))
        for i in range(self.sample_num):
            print(i)
            chemical_development, time_section = self.gillespie()
            f = interp1d(time_section,chemical_development,kind="nearest",axis = 1)
            interpolated = f(self.interpolant_time)

            if  self.state[self.starting_compartment + 2 * self.N] > self.threshold_minor:
                # in other words, if the trajectory lead to a major outbreak at C1
                self.samples.append(np.transpose(interpolated))
            if self.collect_outbreak:
                outbreak_indicator = np.array([self.state[2 * self.N + i] > self.threshold_minor for i in range(self.N)])
                outbreak_comp = np.flatnonzero(outbreak_indicator)
                self.outbreak_list[i,outbreak_comp] = 1

            for j in range(self.N):
                epidemic_size[j,i] = self.state[j + 2 * self.N]
            self.inf_end_list.append(self.end)
            infected_num[:,i] = self.state[2*self.N:3*self.N] - self.init_state[2*self.N:3*self.N]
        self.inf_end = np.max(self.inf_end_list)
        print("inf_end",self.inf_end)
        self.samples = np.array(self.samples)
        if len(self.samples) == 0:
            print("samples are empty")
        mean_chemical_states = np.mean(self.samples,axis = 0)
        self.infected_num_total = np.sum(np.mean(infected_num,axis = 1))
        return epidemic_size, mean_chemical_states
        

