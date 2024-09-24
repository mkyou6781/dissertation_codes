import numpy as np
import scipy as sp
import pandas as pd
from numpy.random import exponential
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from calc_rate import calc_rate_coupling, calc_rate_diffusion
from update import update_coupling, update_diffusion

"""
code for calculating the time development of states in the two-compartment stochastic SIR model
"""


class Two_Comp_Gillespie:
    """
    class for the two-compartment stochastic SIR model

    parameters:
        init_state: numpy array (1 by 6)
            initial state of the system
        time_end: float
            the time point which the simulation ends
        sample_num: int
            the number of samples
        beta_1: float
            infection rate in compartment 1
        gamma_1: float
            recovery rate in compartment 1
        volume_1: float
            volume of compartment 1
        beta_2: float
            infection rate in compartment 2
        gamma_2: float
            recovery rate in compartment 2
        volume_2: float
            volume of compartment 2
        epsilon: float
            the coupling strength epsilon
        model: str
            the model to be used. Either "coupling" or "diffusion"
        threshold_minor: int
            threshold for major outbreak
    
    Returns:
        mean_chemical_states: numpy array (time_end*10 by 6)
            the mean of the chemical states given there is a major outbreak (R[-1] > threshold_minor)
        final_state_1: numpy array (1 by sample_num)
            the final recovered population of the first compartment
        final_state_2: numpy array (1 by sample_num)
            the final recovered population of the second compartment
    """
    def __init__(self,init_state,time_end,sample_num, beta_1,gamma_1,volume_1,beta_2,gamma_2,volume_2, epsilon, model, threshold_minor):
        self.init_state = init_state # two dimensional array
        self.init_sum = np.sum(init_state)
        self.time_end = time_end
        self.data = []  # To store data
        self.sample_num = sample_num
        self.gamma_1 = gamma_1
        self.beta_1 = beta_1
        self.volume_1 = volume_1
        self.gamma_2 = gamma_2
        self.beta_2 = beta_2
        self.volume_2 = volume_2
        self.epsilon = epsilon
        self.model = model
        self.threshold_minor = threshold_minor

    def calc_rate(self):
        if self.model == "coupling":
            rate = calc_rate_coupling(self.beta_1,self.gamma_1,self.volume_1,self.beta_2,self.gamma_2,self.volume_2, self.epsilon,self.state)
        elif self.model == "diffusion":
            rate = calc_rate_diffusion(self.beta_1,self.gamma_1,self.volume_1,self.beta_2,self.gamma_2,self.volume_2, self.epsilon,self.state)
        else:
            raise ValueError("model must be either 'coupling' or 'diffusion' ")
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
        if self.model == "coupling":
            self.state = update_coupling(self.state,index)
        elif self.model == "diffusion":
            self.state = update_diffusion(self.state,index)
        else:
            raise ValueError("model must be either 'coupling' or 'diffusion' ")
        
    
    def gillespie(self):
        chemical_development = np.zeros((6,1))
        self.state = np.copy(self.init_state)
        chemical_development[:,0] = self.state
        time_section = []
        time = 0
        time_section.append(time)
        #print(self.state)
        while True:
            #print(self.state)
            #print(time)
            rate = self.calc_rate()
            waiting_time = self.waiting_time(rate)
            time += waiting_time
            index = self.choose_reaction(rate)
            self.update(index)

            chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
            time_section.append(time)
            #print("after the update",self.state)
            if time >= self.time_end:
                break
            #if the simulation terminates before the time end, add the last state and the final time for successful interpolation
            elif ((self.state[1] == 0) and (self.state[4] == 0)):
                time_section.append(self.time_end)
                chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
                break

        return np.array(chemical_development), np.array(time_section)
    
    def sampling(self):
        self.final_state_1 = []
        self.final_state_2 = []

        self.interpolant_time = np.linspace(0,self.time_end,self.time_end*10) # finer time to take 
        #samples = np.zeros((len(self.interpolant_time),len(self.state),self.sample_num))
        self.samples = []
        for i in range(self.sample_num):
            print(i)
            chemical_development, time_section = self.gillespie()
            f = interp1d(time_section,chemical_development,kind="nearest",axis = 1)
            interpolated = f(self.interpolant_time)
            
            if (self.state[2] > self.threshold_minor) and (self.state[5] > self.threshold_minor):
                # in other words, if the trajectory lead to a major outbreak 
                self.samples.append(np.transpose(interpolated))
            
            self.final_state_1.append([self.state[2]])
            self.final_state_2.append([self.state[5]])
        self.samples = np.array(self.samples)
        mean_chemical_states = np.mean(self.samples,axis = 0)

        return mean_chemical_states,np.array(self.final_state_1).flatten(),np.array(self.final_state_2).flatten()
        

