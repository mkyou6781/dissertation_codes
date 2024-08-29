import numpy as np
from scipy.integrate import odeint
import pandas as pd


class Two_Comp_Det:
    def __init__(self,init_state,time_end,sample_num, beta_1,gamma_1,volume_1,beta_2,gamma_2,volume_2, epsilon, model,dt):
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
        self.dt = dt

    def coupling_rate(self,state,t):
        dx = np.zeros(6)
        dx[0] = - (1-self.epsilon)*self.beta_1 * state[0]*state[1] / self.volume_1 - self.epsilon * self.beta_2 * state[0]*state[4] / self.volume_2
        dx[1] = (1-self.epsilon)*self.beta_1 * state[0]*state[1] / self.volume_1 - self.gamma_1 * state[1] + self.epsilon * self.beta_2 * state[4]*state[0] / self.volume_2
        dx[2] = self.gamma_1 * state[1]
        dx[3] = - (1-self.epsilon)*self.beta_2 * state[3]*state[4] / self.volume_2 - self.epsilon * self.beta_1* state[3]*state[1] / self.volume_1
        dx[4] = (1-self.epsilon)*self.beta_2 * state[3]*state[4] / self.volume_2 - self.gamma_2 * state[4] + self.epsilon * self.beta_1* state[3]*state[1] / self.volume_2
        dx[5] = self.gamma_2 * state[4]
        return dx

    def diffusion_rate(self,state,t):
        dx = np.zeros(6)
        dx[0] = - self.beta_1 * state[0]*state[1] / np.sum(state[:3]) + self.epsilon * state[3] - self.epsilon * state[0]
        dx[1] = self.beta_1 * state[0]*state[1] / np.sum(state[:3]) - self.gamma_1 * state[1] + self.epsilon * state[4] - self.epsilon * state[1]
        dx[2] = self.gamma_1 * state[1]
        dx[3] = - self.beta_2 * state[3]*state[4] / np.sum(state[3:]) + self.epsilon * state[0] - self.epsilon * state[3] 
        dx[4] = self.beta_2 * state[3]*state[4] / np.sum(state[3:]) - self.gamma_2 * state[4] + self.epsilon * state[1] - self.epsilon * state[4]
        dx[5] = self.gamma_2 * state[4]
        return dx
        
    def solve_ode(self):
        t = np.arange(0, self.time_end, self.dt)  # Time vector
        if self.model == "coupling":
            solution = odeint(self.coupling_rate, self.init_state, t)
        elif self.model == "diffusion":
            solution = odeint(self.diffusion_rate, self.init_state, t)
        else:
            raise ValueError
        return t, solution