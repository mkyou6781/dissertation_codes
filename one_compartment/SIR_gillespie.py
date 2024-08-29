import numpy as np
import scipy as sp
import pandas as pd
from numpy.random import exponential
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

"""
code for calculating the timedevelopment of chemicals in the random chemical reaction network 
with simple Gillespie algorithm 
and plotting the result
"""


class SIR_Gillespie:
    """
    class for developing the chemical over time
        
    parameters:
        N: int
            number of chemicals
        n: int
            number of nutrient/transporter
        g: int
            number of the two-body reaction
        D: float
            coefficient for the intake of nutrient
        init_state: numpy array (1 by N)
            initial state of the "number" of chemicals
        init_volume: float
            initial volume of the system
        nutrient_state: numpy array (1 by n)
            current "conc" of nutrient
        reactant: numpy array
            indices of chemical which work as a reactant
        catalyst: numpy array 
            indices of chemical which work as a catalyst
        product: numpy array
            indices of chemical which work as a catalyst 
        time_end: float
            the time point which the simulation ends
        threshold_minor: int
            threshold for major outbreak
    
    Returns:
        time_section: numpy array
            time points which a reaction fires
        chemical_development: numpy array (N,len(time_section))
            number of chemicals at each time point
        conc: numpy array (N,len(time_section))
            concentration of chemicals at each time point


    """
    def __init__(self,init_state,time_end,sample_num, gamma,beta,volume,threshold_minor):
        self.init_state = init_state # two dimensional array
        self.init_sum = np.sum(init_state)
        self.time_end = time_end
        self.data = []  # To store data
        self.sample_num = sample_num
        self.gamma = gamma
        self.beta = beta
        self.volume = volume
        self.threshold_minor = threshold_minor

    def calc_rate(self):
        rate = np.array([self.beta * self.state[0] * self.state[1]/self.volume,self.gamma * self.state[1]])
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
    
    def update(self,fired_reaction):
        if fired_reaction == 0: # infected
            self.state[0] -= 1
            self.state[1] += 1
        elif fired_reaction == 1:  # recovery
            self.state[1] -= 1
            self.state[2] += 1
    
    def gillespie(self):
        chemical_development = np.zeros((3,1))
        self.state = np.copy(self.init_state)
        chemical_development[:,0] = self.state
        time_section = []
        time = 0
        time_section.append(time)
        self.inf_end = 0
        while True:
            #print(self.state)
            rate = self.calc_rate()
            waiting_time = self.waiting_time(rate)
            time += waiting_time
            fired_reaction = self.choose_reaction(rate)
            self.update(fired_reaction)

            chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
            time_section.append(time)
            
            if time >= self.time_end:
                self.end = self.time_end
                break

            #if the simulation terminates before the time end,
            # when the infection dies out
            # add the last state and the final time for successful interpolation
            elif self.state[1] == 0:
                time_section.append(self.time_end)
                chemical_development = np.append(chemical_development,np.transpose(np.array([self.state])),axis=1)
                self.end = time
                # do not forget to add the end time and end state for interpolation
                break

            #print(time)
        self.inf_end = time

        return np.array(chemical_development), np.array(time_section)
    
    def sampling(self):
        cluster_size = []
        self.inf_end_list = []
        self.samples = []
        self.interpolant_time = np.linspace(0,self.time_end,self.time_end*10)
        for i in range(self.sample_num):
            print(i)
            chemical_development, time_section = self.gillespie()
            cluster_size.append(self.state[2])
            #print(time_section)
            #print(chemical_development)
            f = interp1d(time_section,chemical_development,kind="nearest",axis = 1)
            interpolated = f(self.interpolant_time)

            if  self.state[2] > self.threshold_minor:
                # in other words, if the trajectory lead to a major outbreak 
                self.samples.append(np.transpose(interpolated))
            self.inf_end_list.append(self.inf_end)
        
        #print(len(self.samples))
        
        self.samples = np.array(self.samples)
        self.inf_end = np.max(self.inf_end_list)
        mean_chemical_states = np.mean(self.samples,axis = 0)
        if len(self.samples) == 0:
            mean_chemical_states = np.zeros((len(self.interpolant_time),3))
        return np.array(cluster_size), self.interpolant_time,mean_chemical_states

