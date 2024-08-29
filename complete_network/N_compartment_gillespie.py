import numpy as np
from numpy.random import exponential
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy import sparse
#from matplotlib.lines import Line2D
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset


"""
code for calculating the timedevelopment of chemicals in the random chemical reaction network 
with simple Gillespie algorithm 
Note: the state is expressed with numpy array with the following format [S_1, ...S_N, I_1, ..., I_N, R_1, ..., R_N]

input:
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
    sample_time_criteria: list
        if the number corresponding to the compartment is included in the list, then the survival time is sampled only if there is a major outbreak at that compartment
        default: [0] 

"""

class N_Comp_Gillespie:
    def __init__(self,init_state,time_end,sample_num, beta,gamma,volume, epsilon, threshold_minor,N,sample_time_criteria=[0]):
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
        self.sample_time_criteria = sample_time_criteria

    def calc_rate(self):
        recovery = self.gamma * self.state[self.N:2*self.N]

        diagonal = np.multiply(self.state[:self.N], self.state[self.N:2*self.N]) * ((1-self.epsilon) * self.beta) / self.volume
        infection = np.outer(self.state[:self.N], self.state[self.N:2*self.N]) / self.volume
        infection *= self.beta * self.epsilon / (self.N - 1)
        np.fill_diagonal(infection, diagonal)
        rate = np.concatenate((infection.ravel(), recovery))
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
        #print(index)
        if 0 <= index < self.N * self.N:
            #print("infection occured")
            comp_of_interest_1 = index // self.N # i
            comp_of_interest_2 = index % self.N # j
            self.state[comp_of_interest_1] -= 1
            self.state[comp_of_interest_1 + self.N] += 1
        elif self.N * self.N <= index < self.N * self.N + self.N:
            #print("recovery occured")
            comp_of_interest = index - self.N ** 2
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
        #print(self.state)
        while True:
            #print(self.state)
            #print(time)
            rate = self.calc_rate()
            waiting_time = self.waiting_time(rate)
            time += waiting_time
            index = self.choose_reaction(rate)
            self.update(index)
            #print(self.state)

            chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
            time_section.append(time)

            # just for sanity check
            """num_in_comp = np.array([self.state[i] + self.state[i+self.N] + self.state[i+2 * self.N] for i in range(self.N)])
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
        
        return np.array(chemical_development), np.array(time_section)
    
    def sampling(self):
        epidemic_size = np.zeros((self.N,self.sample_num))

        self.interpolant_time = np.linspace(0,self.time_end,self.time_end*10) # finer time to take 
        #samples = np.zeros((len(self.interpolant_time),self.N,self.sample_num))
        ### need to fix this because if I put the large time_end to run the simulation, the size of the data becomes too large and it crashes
        self.samples = []
        self.inf_end_list = []
        self.delay_time_list = []
        for i in range(self.sample_num):
            print(i)
            chemical_development, time_section = self.gillespie()
            f = interp1d(time_section,chemical_development,kind="nearest",axis = 1)
            interpolated = f(self.interpolant_time)
            #print(interpolated.shape)

            if  self.state[2*self.N] > self.threshold_minor:
                self.delay_time = np.zeros(self.N)
                # in other words, if the trajectory lead to a major outbreak 
                self.samples.append(np.transpose(interpolated))
                # find the latest time each compartment had 0 infected individual
                for j in range(1,self.N): # avoid the first compartment
                    #print(interpolated[j+self.N,:])
                    if np.count_nonzero(interpolated[j+2 * self.N,:] > self.threshold_minor) == 0: # no outbreak
                        # add nan
                        self.delay_time[j] = np.nan
                    else:
                        # obtain the index at which the outbreak is evident
                        index = np.flatnonzero(interpolated[j+2*self.N,:] > self.threshold_minor)
                        #print("index",index)
                        infected_before_outbreak = interpolated[j+self.N,:index[0]] 
                        # add 1 so that the length of the array is the same as the interpolated time but not detected as 0
                        infected_before_outbreak = np.append(infected_before_outbreak,np.ones(len(self.interpolant_time) - len(infected_before_outbreak)))
                        zero_indices = np.flatnonzero([infected_before_outbreak < 1])
                        #print("zero_indices",zero_indices)
                        #self.delay_time[j] = self.interpolant_time[zero_indices[-1]]
                        # Create the main plot
                        """fig, ax = plt.subplots()
                        x,y = self.interpolant_time,interpolated[j+self.N,:]
                        ax.plot(x,y)

                        ax.set_xlabel(r"$t$",fontsize=12)
                        ax.set_ylabel(r"$I$",fontsize=12)
                        ax.vlines(self.delay_time[j],0,self.volume,colors="red",linestyles="--")
                        ax.set_ylim(0,np.max(interpolated[j+self.N]))
                        

                        # Define the region to zoom in
                        x1, x2, y1, y2 = self.delay_time[j]-2, self.delay_time[j]+2, 0, 5

                        # Create an inset axis
                        axins = inset_axes(ax, width="80%", height="80%", loc='upper right',
                                        bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                                        bbox_transform=ax.transAxes)

                        # Plot the same data on the inset axis
                        axins.plot(x, y)
                        axins.vlines(self.delay_time[j],0,self.volume,colors="red",linestyles="--")
                        # Manually add a rectangle in the main plot to show where the zoomed area is
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='k', facecolor='none', linewidth=1,linestyle="--")
                        ax.add_patch(rect)

                        # Set the limits for the inset axis to zoom in
                        axins.set_xlim(x1, x2)
                        axins.set_ylim(y1, y2)

                        # Optional: Add a rectangle in the main plot to show where the zoomed area is
                        ax.indicate_inset_zoom(axins)
                        # Add labels and legend
                        ax.legend()

                        # Show the plot
                        plt.show()"""
                #self.delay_time_list.append(self.delay_time)


            for j in range(self.N):
                epidemic_size[j,i] = self.state[j + 2 * self.N]
            
            sampling_criteria_for_survival_time = (self.state[2*self.N:3*self.N] - self.threshold_minor * np.ones(self.N)) > 0
            if np.all(sampling_criteria_for_survival_time[self.sample_time_criteria]):
                # in other words, if the trajectory lead to a major outbreak at the compartment of interest, sample survival time
                self.inf_end_list.append(self.end)
        self.samples = np.array(self.samples)
        self.inf_end = np.max(self.inf_end_list)
        mean_chemical_states = np.mean(self.samples,axis = 0)
        #self.delay_time_list = np.array(self.delay_time_list)

        return epidemic_size, mean_chemical_states
        

