import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw, comb

"""
This code is for calculating the probability of the number of local outbreaks given the number of epsilons and the population in the compartment.

The code first calculates the value of alpha_{ij} with the function calc_alpha, which is the conditional probability of compartment j not having local outbreak given there is a local outbreak at compartment i. Note the value is homegonous, regardless of indices ij, because all the compartments are equivalent in the complete network model we deal with here. 

Then, the code calculates the polynomial of the probability mass distribution of the number of local outbreaks given there is a local outbreak at compartment 1 with the function N_polynomial. The algorithm is iterative, i.e. it calculates the polynomial at N = M from N = M-1.  

"""


def calc_alpha(r_0,epsilon,N,volume):
    r_infty = 1 + lambertw(-(1) * r_0 * np.exp(-r_0 * (1))) / (r_0)
    peak_size = r_infty.real * volume
    #print(peak_size)
    prob = (1 - epsilon * (r_0 - 1) / (N-1)) ** peak_size
    return prob

 

def N_polynomial(alpha,N):
    poly = np.array([alpha,(1-alpha)]) # N = 2
    #print(poly)
    if N == 2:
        pass
    else:
        for i in range(2,N):
            next_poly = np.zeros(i+1)
            for j in range(i):
                alpha_power = alpha ** (j+1)
                next_poly[j] = poly[j] * alpha_power * (comb(i,j)/comb(i-1,j))
            next_poly[i] = 1 - np.sum(next_poly[:-1])
            poly = next_poly

    return poly

def N_prob(r_0,epsilon,N,volume):
    alpha = calc_alpha(r_0,epsilon,N,volume)
    return N_polynomial(alpha,N)

# the function to calcualte the probability for the case N = 3. This is because we can get the analytic expression quite easily.
def three_comp_polynomial(alpha):
    prob_1 = alpha ** 2
    prob_2 = 2 * alpha ** 2 * (1 - alpha)
    prob_3 = (1 - alpha) ** 2 + 2 * alpha * (1 - alpha) ** 2
    return np.array([prob_1,prob_2,prob_3])

def three_comp_prob(r_0,epsilon,N):
    alpha = calc_alpha(r_0,epsilon,N)
    return three_comp_polynomial(alpha)

from sympy import symbols, Rational, binomial,expand

# the following code is for symbolic calculation to check the form of the polynomial
x = symbols('x')

def symbolic_N_prob(N):
    poly = np.array([x, 1 - x], dtype=object)  # N = 2
    if N == 2:
        return poly
    else:
        for i in range(2, N):
            next_poly = np.zeros(i + 1, dtype=object)
            for j in range(i):
                alpha_power = x ** (j + 1)
                next_poly[j] = poly[j] * alpha_power * Rational(binomial(i, j)) / Rational(binomial(i - 1, j))
            next_poly[i] = 1 - sum(next_poly[:-1])
            poly = next_poly

    return poly


# sanity check with N = 4
N = 4
prob = symbolic_N_prob(N)
print(prob)
print("division",prob[2]/(prob[2]+prob[3]))
print("division expanded",expand(prob[2]/(prob[2]+prob[3])))
expanded = []
for i in range(len(prob)):
    expanded.append(expand(prob[i]))
print(expanded)

# Save the result to a txt file
with open("/Users/bouningen0909/dissertation/week_6/data/prob_plot/polynomial/symbolic_{}_prob_expanded_result.txt".format(N), "w") as file:
    for p in expanded:
        file.write(str(p) + "\n")
