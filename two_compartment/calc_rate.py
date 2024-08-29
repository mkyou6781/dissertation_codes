import numpy as np
import scipy as sp
import pandas as pd
from numpy.random import exponential
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

def calc_rate_coupling(beta_1,gamma_1,volume_1,beta_2,gamma_2,volume_2, epsilon,state):
    """
    calcualte the reaction rate of the coupled SIR model
    S_1 + I_1 -> 2 I_1
    I_1 -> R_1
    S_2 + I_2 -> 2 I_2
    I_2 -> R_2
    S_1 + I_2 -> I_1 + I_2
    I_1 + S_2 -> I_1 + I_2
    """
    return np.array([(1-epsilon)*beta_1 * state[0] * state[1]/volume_1,gamma_1 * state[1],(1-epsilon)*beta_2 * state[3] * state[4]/volume_2,gamma_2 * state[4],epsilon * beta_2 * state[0] * state[4]/volume_1,epsilon * beta_1 * state[1] * state[3]/volume_2])

def calc_rate_diffusion(beta_1,gamma_1,volume_1,beta_2,gamma_2,volume_2, epsilon,state):
    """
    calcualte the reaction rate of the coupled SIR model
    S_1 + I_1 -> 2 I_1
    I_1 -> R_1
    S_2 + I_2 -> 2 I_2
    I_2 -> R_2
    S_1 -> S_2
    S_2 -> S_1
    I_1 -> I_2
    I_2 -> I_1
    """
    return np.array([beta_1 * state[0] * state[1]/np.sum(state[:3]),gamma_1 * state[1],beta_2 * state[3] * state[4]/np.sum(state[3:]),gamma_2 * state[4],epsilon * state[0],epsilon * state[3],epsilon * state[1],epsilon * state[4]])