import numpy as np
import matplotlib.pyplot as plt
from lattice_monte_carlo import Lattice_Monte_Carlo
from scipy.special import lambertw
import pandas as pd


N = 64
r_0 = 2
#epsilon_list = np.array([0.03,0.014,0.01,0.009,0.006,0.003,0.001])
#epsilon_list = np.linspace(0.01,0.02,10)
epsilon_list = [0.06,0.03,0.01,0.006,0.003,0.001]
volume = 300

sample_num = 1000
network_type = "3D-lattice"
starting_compartment = 1
N_sqrt = int(np.sqrt(N))

fig, ax = plt.subplots(3, 3, figsize=(15,15))

for k, epsilon in enumerate(epsilon_list):
    i = k // 3
    j = k % 3
    lattice_monte_carlo = Lattice_Monte_Carlo(N, r_0,  epsilon,volume, sample_num, network_type,starting_compartment)

    outbreak_list, outbreak_num = lattice_monte_carlo.sampling()
    alpha = lattice_monte_carlo.alpha

    outbreak_prob_by_comp = np.zeros(N)
    outbreak_count = np.zeros(N)
    outbreak_count = np.sum(outbreak_list,axis = 0)
    center_outbreak_num = np.max(outbreak_count)
    outbreak_prob_by_comp = outbreak_count / center_outbreak_num

    df = pd.DataFrame(np.transpose(np.copy(outbreak_list)), columns=[f"sample{i}" for i in range(sample_num)])

    # Save DataFrame to CSV
    csv_file_path = "/Users/bouningen0909/dissertation/week_7/data/lattice_N_81/monte_carlo/monte_carlo_outbreak_count{}_comp_eps{}R{}.csv".format(N,epsilon,r_0)
    df.to_csv(csv_file_path, index=False)

    # heatmap to show the probability of outbreak given there is a major outbreak at C1
    im = ax[i,j].imshow(outbreak_prob_by_comp.reshape(int(np.sqrt(N)),int(np.sqrt(N))), cmap="binary", interpolation='nearest', vmin=0, vmax=1)
    ax[i,j].set_title(rf"$\epsilon = ${epsilon}, $1 - \alpha = ${1 - alpha:.3f} ")
    # adding a color bar
    cbar = fig.colorbar(im, ax=ax[i,j],shrink=0.75)
    cbar.set_label('Probability')

    contour_levels = np.arange(0, 1.0, 0.1)

    # Overlay a contour plot
    contour = ax[i,j].contour(outbreak_prob_by_comp.reshape(int(np.sqrt(N)),int(np.sqrt(N))),levels=contour_levels, colors='red', linewidths=1)

    # Add labels to the contour lines
    ax[i,j].clabel(contour, inline=True, fontsize=8)

# Displaying parameters in the second subplot
params_text = f"""Parameters:
N : {N}
r_0: {r_0}
volume: {volume}
Sample Number: {sample_num}
"""
count = np.zeros(N+1)
for i in outbreak_num:
    count[int(i)] += 1
plt.bar(np.arange(1,N+1),count[1:]/(np.sum(count[1:])), color="blue")

# impose the data on the plot
'''csv_file_path = "/Users/bouningen0909/dissertation/week_7/data/num_outbreak25_comp_eps0.01R2.csv"
df = pd.read_csv(csv_file_path)
num_outbreak = df.to_numpy()
print(num_outbreak.shape)
plt.bar(np.arange(1,N+1),num_outbreak[1:,1]/(np.sum(num_outbreak[1:,1])), color="red",width=0.5,alpha=0.5)'''
plt.show()

"""ax[i,j+1].axis('off')
ax[i,j+1].text(0.5, 0.5, params_text, fontsize=9, ha='center', va='center')
plt.savefig("/Users/bouningen0909/dissertation/week_7/data/lattice_N_81/monte_carlo/monte_carloN{}_R_{}.png".format(N,int(r_0)))
plt.show()"""

