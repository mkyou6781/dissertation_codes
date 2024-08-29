import numpy as np
from numpy.random import random
import networkx as nx
from scipy.special import lambertw

class Lattice_Monte_Carlo:
    def __init__(self, N, r_0,  epsilon,volume, sample_num, network_type,starting_compartment = 0, herd_immunity=False,immunised_network = None, threshld_minor = 100,degree=None,adj_matrix = None,alpha_given = None):
        self.N = N
        self.r_0 = r_0
        self.epsilon = epsilon
        self.volume = volume
        self.sample_num = sample_num
        self.network_type = network_type
        self.starting_compartment = starting_compartment - 1
        self.herd_immunity = herd_immunity
        self.immunised_network = immunised_network
        self.threshold_minor = threshld_minor
        self.degree = degree
        self.adj_matrix = adj_matrix # Note the network needs to be unweighted
        self.total_num = 0 # total number of infected population
        self.N_sqrt = int(np.sqrt(self.N))
        self.alpha_given = alpha_given
    
    def calc_alpha(self):
        r_infty = 1 + lambertw(-(1) * self.r_0 * np.exp(-self.r_0 * (1))) / (self.r_0)
        peak_size = r_infty.real *self.volume
        if self.network_type == "lattice":
            #print(peak_size)
            alpha = (1 - self.epsilon * (self.r_0 - 1) / 4) ** peak_size
        elif self.network_type == "3D-lattice":
            #print(peak_size)
            alpha = (1 - self.epsilon * (self.r_0 - 1) / 6) ** peak_size
        elif self.network_type == "all-to-all":
            alpha = (1 - self.epsilon * (self.r_0 - 1) / (self.N-1)) ** peak_size
        elif self.network_type == "random":
            alpha = (1 - self.epsilon * (self.r_0 - 1) / (self.degree)) ** peak_size
        return alpha, peak_size
    
    def inhomegenous_alpha(self, imm_origin,imm_neighbour):
        #print("function called")
        # imm means imunised
        # calculate the reproduction number of the neighbour
        r_neighbour = self.r_0 * ((self.volume - imm_neighbour)/self.volume)
        r_origin = self.r_0 * ((self.volume - imm_origin)/self.volume)
        #print("r",r_neighbour)
        if r_neighbour <= 1: # cannot have outbreak at the neigbour even if there is a critical introduction of infection
            alpha = 1
            neighbour_peak_size = 0
        elif r_origin <= 1: # cannot have outbreak at the origin 
            alpha = 1
            neighbour_peak_size = 0
        else:
            r_infty_origin = 1 + (1/self.r_0) * lambertw((-1)* self.r_0 * np.exp(-self.r_0 * (1 - imm_origin/self.volume)) * (self.volume - imm_origin)/(self.volume))
            origin_peak_size = r_infty_origin.real * self.volume
            if np.isnan(origin_peak_size):
                print("error detected", imm_origin)
            if self.network_type == "lattice":
                # calculate the epidemic size at the origin
                alpha = (1 - (self.epsilon/4)*(r_neighbour - 1)) ** (np.floor(origin_peak_size))
            elif self.network_type == "random":
                alpha = (1 - (self.epsilon/self.degree)*(r_neighbour - 1)) ** (np.floor(origin_peak_size))
            r_infty_neighbour = 1 + (1/self.r_0) * lambertw((-1)* self.r_0 * np.exp(-self.r_0 * (1 - imm_neighbour/self.volume)) * (self.volume - imm_neighbour)/(self.volume))
            neighbour_peak_size = r_infty_neighbour.real * self.volume
        return alpha, neighbour_peak_size

    def generate_adj_matrix(self):
        if self.adj_matrix is None:
            if self.network_type == "lattice":
                N_half = int(np.sqrt(self.N))
                
                # Create the 2-D lattice graph
                G = nx.grid_2d_graph(N_half, N_half,periodic = True)
                # Get the adjacency matrix
                adj_matrix = nx.adjacency_matrix(G)
                # Convert the adjacency matrix to COO format
            elif self.network_type == "3D-lattice":
                N_cbrt = int(np.cbrt(self.N))
                
                # Create the 2-D lattice graph
                G = nx.grid_graph(dim=[N_cbrt, N_cbrt, N_cbrt], periodic=True)
                # Get the adjacency matrix
                adj_matrix = nx.adjacency_matrix(G)

            elif self.network_type == "all-to-all":
                N_half = int(np.sqrt(self.N))
                
                # Create the 2-D lattice graph
                G = nx.complete_graph(self.N)
                # Get the adjacency matrix
                adj_matrix = nx.adjacency_matrix(G)
                # Convert the adjacency matrix to COO format
        else:
            adj_matrix = self.adj_matrix
        
        self.adj_matrix = adj_matrix
        # DO I NEED TO COPY?

        adj_coo = self.adj_matrix.tocoo()
        # personal note: check (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html) coo separates a sparse matrix into row, column, data (values)

        # Extract the row indices, column indices, and values
        self.row = adj_coo.row
        self.col = adj_coo.col
        self.data = adj_coo.data
        self.len_data = len(self.data)

        # Combine row and column indices into a single array of indices
        #self.indices = np.vstack((row, col)).T
    
    def check_infection(self,prev,curr):
        out = True
        diff = curr - prev
        if np.all(diff == 0):
            out = False
        elif np.all(curr == 1):
            out = False
        return out
    
    def get_adj(self,comp):
        adj = []
        for i in range(self.len_data):
            if self.row[i] == comp:
                adj.append(self.col[i])
        return adj
    
    def trigger(self,alpha):
        trigger = False
        p_1 = random()
        if (p_1 < 1 - alpha):
            trigger = True
        return trigger


    def monte_carlo(self):
        pseudo_infected_num = 0 # the number of infected populations in the system
        if self.herd_immunity:
            immunised = np.copy(self.immunised_network)
            outbreak_in_immunised = immunised > self.threshold_minor
        self.alpha, peak_size = self.calc_alpha()
        if self.alpha_given is not None:
            self.alpha = self.alpha_given
        self.generate_adj_matrix()
        self.lattice = np.zeros(self.N)
        self.lattice[self.starting_compartment] = 1
        self.neighbour_sum = 0 # number to count the non-outbreak neighbours for the analysis of transition
        prev = np.zeros(self.N)
        self.infected_num_per_comp = np.zeros(self.N)
        while self.check_infection(prev,self.lattice):
            diff = self.lattice - prev
            prev = np.copy(self.lattice)
            index_to_scan = np.flatnonzero(diff)
            for newly_infected in index_to_scan:
                adj = self.get_adj(newly_infected)
                #print("we are at",newly_infected)
                #print("adjacent compartments",adj)
                #print("state",self.lattice[adj])
                for comp in adj:
                    if self.herd_immunity:
                        self.alpha, peak_size = self.inhomegenous_alpha(immunised[newly_infected],immunised[comp])
                        print("alpha",self.alpha)
                        # obtain the alpha value from the number of immunisedd population in the compartment and the size of the outbreak size in the neighbour
                    else:
                        pass
                    # count the number of non-outbreak neighbours
                    if self.lattice[comp] == 0:
                        self.neighbour_sum += 1
                    if self.trigger(self.alpha) and self.lattice[comp] == 0:
                        self.lattice[comp] = 1
                        pseudo_infected_num += peak_size
                        self.infected_num_per_comp[comp] = peak_size
                        
        return self.lattice, pseudo_infected_num
    
    def sampling(self):
        self.outbreak_list = []
        self.outbreak_num = np.zeros(self.sample_num)
        self.expected_neighbour = 0 # expected number of non-outbreak neighbours
        #print(self.sample_num)
        self.pseudo_infected_num_array = np.zeros(self.sample_num)
        infected_num_array_per_comp = np.zeros((self.N,self.sample_num))
        for i in range(int(self.sample_num)):
            print(i)
            lattice, pseudo_infected_num = self.monte_carlo()
            #print("pseudo_infected_num",pseudo_infected_num)
            self.outbreak_num[i] = np.count_nonzero(lattice)
            self.outbreak_list.append(lattice)
            self.expected_neighbour += self.neighbour_sum /(np.sum(self.lattice) * self.sample_num)
            self.pseudo_infected_num_array[i] = pseudo_infected_num
            #print("pseudo_infected_num_array",self.pseudo_infected_num_array)
            #print("self.expected_neighbour",self.expected_neighbour)
            infected_num_array_per_comp[:,i] = self.infected_num_per_comp
        self.outbreak_list = np.array(self.outbreak_list)
        self.mean_infected_num_array_per_comp = np.mean(infected_num_array_per_comp,axis = 1)
        return self.outbreak_list, self.outbreak_num
