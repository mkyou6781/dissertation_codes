import numpy as np
from numpy.random import exponential
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy import sparse


"""
class for calculating the timedevelopment of chemicals in the multi-compartment SIR model inscribed in the complete network with size N with simple Gillespie algorithm 
Note: the state is expressed with numpy array with the following format [S_1, ...S_N, I_1, ..., I_N, R_1, ..., R_N]
"""

class N_Comp_Gillespie:
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
        sample_time_criteria: list
            if the number corresponding to the compartment is included in the list, then the survival time is sampled only if there is a major outbreak at that compartment
            default: [0] 
    Returns:
        outbreak_size: numpy array (N by sample_num)
            the size of the outbreak
        mean_chemical_states: numpy array (time_end*10 by 3N)
            the mean of the chemical states given there is a major outbreak in the compartment 1 (R_1[-1] > threshold_minor)
        delay_time_list: list
            the delay time (the time at which the compartment has zero infected individual last time) for each compartment
        interpolant_time: numpy array
            the time points for interpolation
        inf_end_list: list
            the list of the time at which the infection dies out
        samples: numpy array (N by length of interpolant_time)
            the sample trajectories given there is a major outbreak
    """
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
        # calculate the rate of the reaction
        # rate of recovery I -> R
        recovery = self.gamma * self.state[self.N:2*self.N]

        # rate of infection within compartment S_i + I_i -> 2I_i
        diagonal = np.multiply(self.state[:self.N], self.state[self.N:2*self.N]) * ((1-self.epsilon) * self.beta) / self.volume

        # rate of infection between compartment S_i + I_j -> I_i + I_j
        infection = np.outer(self.state[:self.N], self.state[self.N:2*self.N]) / self.volume
        infection *= self.beta * self.epsilon / (self.N - 1)
        np.fill_diagonal(infection, diagonal)

        # concatenate the rate of the reaction
        rate = np.concatenate((infection.ravel(), recovery))
        return rate

    def waiting_time(self,rate):
        # calculate the waiting time
        sum_rate = np.sum(rate)
        parameter = 1 / sum_rate
        return exponential(parameter)
    
    def choose_reaction(self,rate):
        # choose the reaction
        index = np.arange(len(rate))
        probabilities = rate / (np.sum(rate))
        fired_reaction = np.random.choice(index, size=1, p=probabilities)
        return fired_reaction
    
    def update(self,index):
        if 0 <= index < self.N * self.N:
            #infection
            # first, find the values of i and j
            comp_of_interest_1 = index // self.N # i
            comp_of_interest_2 = index % self.N # j
            self.state[comp_of_interest_1] -= 1
            self.state[comp_of_interest_1 + self.N] += 1
        elif self.N * self.N <= index < self.N * self.N + self.N:
            #recovery
            comp_of_interest = index - self.N ** 2
            self.state[comp_of_interest + self.N] -= 1
            self.state[comp_of_interest + 2 * self.N] += 1
        else: 
            raise ValueError("The chosen reaction is not defined")
        
    
    def gillespie(self):
        # code for running the gillespie algorithm for one iteration
        chemical_development = np.zeros((3 * self.N,1)) # to store the discrete state of the chemical reaction over time
        self.state = np.copy(self.init_state) # to store the current state of the chemical reaction
        chemical_development[:,0] = self.state
        time_section = [] # to store the time points
        time = 0 # to store the current time
        time_section.append(time)
        self.end = 0
        while True:
            # calculate the rate of the reaction and update the state accordingly
            rate = self.calc_rate()
            waiting_time = self.waiting_time(rate)
            time += waiting_time
            index = self.choose_reaction(rate)
            self.update(index)

            chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1) # add the current state to checmical_development
            time_section.append(time) # add the current time to time_section

            # just for sanity check
            """num_in_comp = np.array([self.state[i] + self.state[i+self.N] + self.state[i+2 * self.N] for i in range(self.N)])
            print("num conserv",(num_in_comp - np.ones(self.N) * self.volume))
            if np.count_nonzero(num_in_comp - np.ones(self.N) * self.volume) != 0:
                print("warning: the number in each compartment is not conserved")"""

            if time >= self.time_end:
                # when the simulation terminates at the time end
                self.end = self.time_end
                break

            #if the following condition is satisfied, then the simulation is terminated before the time end
            # when the infection dies out
            # add the last state and the final time for successful interpolation
            elif np.count_nonzero(self.state[self.N:2*self.N]) == 0:
                time_section.append(self.time_end)
                chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
                self.end = time
                break
        
        return np.array(chemical_development), np.array(time_section)
    
    def sampling(self):
        ### code for sampling the trajectories by running the gillespie algorithm sample_num times
        outbreak_size = np.zeros((self.N,self.sample_num)) # to store the outbreak size

        self.interpolant_time = np.linspace(0,self.time_end,self.time_end*10) # time points for interpolation 
        
        # ignore the following comment for now
        #samples = np.zeros((len(self.interpolant_time),self.N,self.sample_num))
        ### need to fix this because if I put the large time_end to run the simulation, the size of the data becomes too large and it crashes


        self.samples = [] # to store the sample trajectories given there is a major outbreak
        self.inf_end_list = [] # to store the time of the infection dies out
        self.delay_time_list = [] # to store the delay time for each compartment
        for i in range(self.sample_num):
            print(i)
            # run the gillespie algorithm for one iteration to obtain the trajectory
            chemical_development, time_section = self.gillespie()
            f = interp1d(time_section,chemical_development,kind="nearest",axis = 1)
            interpolated = f(self.interpolant_time)

            # if the trajectory lead to a major outbreak
            if  self.state[2*self.N] > self.threshold_minor:
                self.delay_time = np.zeros(self.N) 
                self.samples.append(np.transpose(interpolated))

                # find the delay time, the latest time each compartment had 0 infected individual
                for j in range(1,self.N): # avoid the first compartment because if there is more than one outbreak, the first compartment is always the first one to have the outbreak

                    # check there is an outbreak at compartment j
                    # if not, add nan
                    if np.count_nonzero(interpolated[j+2 * self.N,:] > self.threshold_minor) == 0: # no outbreak
                        self.delay_time[j] = np.nan
                    else:
                        # obtain the index at which the outbreak is evident
                        index = np.flatnonzero(interpolated[j+2*self.N,:] > self.threshold_minor)

                        infected_before_outbreak = interpolated[j+self.N,:index[0]] 
                        # add 1 so that the length of the array is the same as the interpolated time but not detected as 0
                        infected_before_outbreak = np.append(infected_before_outbreak,np.ones(len(self.interpolant_time) - len(infected_before_outbreak)))
                        zero_indices = np.flatnonzero([infected_before_outbreak < 1])
                        print("zero_indices",zero_indices)
                        self.delay_time[j] = self.interpolant_time[zero_indices[-1]]
                        
                        # Create the main plot
                        # the following code is for plotting the time development of the infected individuals to check whether the delay time is correctly calculated
                        # if you want to check the delay time, uncomment the following code
                        
                        """from matplotlib.lines import Line2D
                        import matplotlib.pyplot as plt
                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
                        
                        fig, ax = plt.subplots()
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
                self.delay_time_list.append(self.delay_time)

            # store the size of outbreak: the number of recovered individuals in each compartment at the end of the simulation
            for j in range(self.N):
                outbreak_size[j,i] = self.state[j + 2 * self.N]
            

            # check if the trajectory lead to a major outbreak at the compartment of interest, given by sampling_criteria_for_survival_time sample survival time
            sampling_criteria_for_survival_time = (self.state[2*self.N:3*self.N] - self.threshold_minor * np.ones(self.N)) > 0
            if np.all(sampling_criteria_for_survival_time[self.sample_time_criteria]):
                self.inf_end_list.append(self.end)
        self.samples = np.array(self.samples)
        print(self.samples)
        print("inf_end_list",self.inf_end_list)
        self.inf_end = np.max(self.inf_end_list)
        mean_chemical_states = np.mean(self.samples,axis = 0)
        self.delay_time_list = np.array(self.delay_time_list)

        return outbreak_size, mean_chemical_states
        

