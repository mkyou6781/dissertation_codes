import numpy as np
import scipy as sp
import pandas as pd
from numpy.random import exponential
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

def update_coupling(state,index):
    """
    update the state of the system for the coupled SIR model
    S_1 + I_1 -> 2 I_1
    I_1 -> R_1
    S_2 + I_2 -> 2 I_2
    I_2 -> R_2
    S_1 + I_2 -> I_1 + I_2
    I_1 + S_2 -> I_1 + I_2
    """
    vec_reac_1 = np.array([-1,+1,0,0,0,0])
    vec_reac_2 = np.array([0,-1,+1,0,0,0])
    vec_reac_3 = np.array([0,0,0,-1,+1,0])
    vec_reac_4 = np.array([0,0,0,0,-1,+1])
    vec_reac_5 = np.array([-1,+1,0,0,0,0])
    vec_reac_6 = np.array([0,0,0,-1,+1,0])
    stoichiometric_matrix = np.vstack((vec_reac_1, vec_reac_2, vec_reac_3, vec_reac_4, vec_reac_5, vec_reac_6))

    if 0 <= index < len(stoichiometric_matrix):
        chosen_vector = stoichiometric_matrix[index].flatten()
        return state + chosen_vector
    else:
        raise ValueError("Index must be between 0 and 5")

def update_diffusion(state,index):
    """
    update the state of the system for the coupled SIR model
    S_1 + I_1 -> 2 I_1
    I_1 -> R_1
    S_2 + I_2 -> 2 I_2
    I_2 -> R_2
    S_1 -> S_2
    S_2 -> S_1
    I_1 -> I_2
    I_2 -> I_1
    """
    vec_reac_1 = np.array([-1,+1,0,0,0,0])
    vec_reac_2 = np.array([0,-1,+1,0,0,0])
    vec_reac_3 = np.array([0,0,0,-1,+1,0])
    vec_reac_4 = np.array([0,0,0,0,-1,+1])
    vec_reac_5 = np.array([-1,0,0,+1,0,0])
    vec_reac_6 = np.array([+1,0,0,-1,0,0])
    vec_reac_7 = np.array([0,-1,0,0,+1,0])
    vec_reac_8 = np.array([0,+1,0,0,-1,0])
    stoichiometric_matrix = np.vstack([vec_reac_1, vec_reac_2, vec_reac_3, vec_reac_4, vec_reac_5, vec_reac_6, vec_reac_7, vec_reac_8])

    if 0 <= index < len(stoichiometric_matrix):
        chosen_vector = stoichiometric_matrix[index].flatten()
        return state + chosen_vector
    else:
        raise ValueError("Index must be between 0 and 7")

