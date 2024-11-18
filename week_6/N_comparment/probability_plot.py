import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw, comb
import matplotlib.cm as cm

N = 40
volume = 300
beta = 2.0
gamma = 1.0
r_0 = beta / gamma

"""
Code to plot the probability of the number of local outbreaks for different epsilon and N values. Mainly used to check when there is an instability in the numerical calculation.
"""

### utility functions (probably better to unify and put in a separate file)
def calc_alpha(r_0,epsilon,N,volume):
    r_infty = 1 + lambertw(-(1) * r_0 * np.exp(-r_0 * (1))) / (r_0)
    peak_size = r_infty.real * volume
    print(peak_size)
    prob = (1 - epsilon * (r_0 - 1) / (N-1)) ** peak_size
    return prob

def three_comp_polynomial(alpha):
    prob_1 = alpha ** 2
    prob_2 = 2 * alpha ** 2 * (1 - alpha)
    prob_3 = (1 - alpha) ** 2 + 2 * alpha * (1 - alpha) ** 2
    return np.array([prob_1,prob_2,prob_3])

def three_comp_prob(r_0,epsilon,N):
    alpha = calc_alpha(r_0,epsilon,N)
    return three_comp_polynomial(alpha)

 

def N_polynomial(alpha,N):
    poly = np.array([alpha,(1-alpha)]) # N = 2
    print(poly)
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




# Plotting
# setting the range of epsilon values
exponents = np.linspace(-5,-1, 50)
#exponents = np.linspace(-4,-2.5, 10)
epsilons = np.power(10,exponents)  # Generate a range of epsilon values
colors = cm.coolwarm(np.linspace(0.4, 1, len(epsilons)))  # Use hsv colormap

plt.figure(figsize=(10, 6))

for epsilon, color in zip(epsilons, colors):
    prob = N_prob(r_0, epsilon, N, volume)
    if prob[-1] < prob[-2]:
        print("anomalous epsilon",epsilon)
    plt.plot(range(1, N+1), prob, color=color)
    #plt.scatter(range(1, N+1), prob, color=color,marker = ".")

# Create a colorbar
norm = plt.Normalize(vmin = exponents.min(), vmax=exponents.max())
sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, ticks=np.linspace(exponents.min(), exponents.max(), 10))
cbar.set_label(r'$Log (\epsilon)$')

plt.xlabel('Index')
plt.ylabel('Probability')
plt.title('N_prob for different epsilon values')

plt.savefig("/Users/bouningen0909/dissertation/week_6/data/prob_plot/N{}_plot_full.png".format(N))
plt.show()

print(N_prob(r_0,0.0001,N))



