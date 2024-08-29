import numpy as np
import matplotlib.pyplot as plt
from lattice_monte_carlo import Lattice_Monte_Carlo
from scipy.special import lambertw
import pandas as pd
from scipy.integrate import trapezoid


N = 81
r_0 = 2
#epsilon_list = np.array([0.06,0.03,0.02,0.01,0.006,0.003,0.001])
# add additional epsilon values which take values between 0.01 and 0.02 with 10 values
#epsilon_list = np.linspace(0.000,0.06,100)
#epsilon_list = np.concatenate((epsilon_list,np.linspace(0.005,0.015,50)))
alpha_array = np.linspace(0,1,100)

volume = 300

sample_num = 1000
network_type = "lattice"
starting_compartment = 1
N_sqrt = int(np.sqrt(N))

prob_mass_list = []
one_minus_alpha_list = []
expected_neighbour_num_list = []

#epsilon_list = epsilon_list[::-1] 
for k,alpha in enumerate(alpha_array):
    print(k,"th value")
    lattice_monte_carlo = Lattice_Monte_Carlo(N, r_0,  0,volume, sample_num, network_type,starting_compartment,alpha_given=alpha)

    outbreak_list, outbreak_num = lattice_monte_carlo.sampling()
    expected_neighbour_num = lattice_monte_carlo.expected_neighbour
    expected_neighbour_num_list.append(expected_neighbour_num)
    alpha = lattice_monte_carlo.alpha
    #print("alpha",alpha)

    outbreak_prob_by_comp = np.zeros(N)
    outbreak_count = np.zeros(N)
    outbreak_count = np.sum(outbreak_list,axis = 0)
    center_outbreak_num = np.max(outbreak_count)
    outbreak_prob_by_comp = outbreak_count / center_outbreak_num
    #print(outbreak_prob_by_comp)
    prob_mass = np.sum(outbreak_prob_by_comp)
    prob_mass_list.append(prob_mass)

    one_minus_alpha_list.append(1 - alpha)

# Convert to numpy arrays
#epsilon_array = np.array(epsilon_list)
prob_mass_array = np.array(prob_mass_list)
#one_minus_alpha_array = np.array(one_minus_alpha_list)
one_minus_alpha_array = 1 - alpha_array
expected_neighbour_num_array = np.array(expected_neighbour_num_list)

# sort the value of epsilon_list, alpha_list and prob_mass_list in ascending order of one_minus_alpha
sorted_indices = np.argsort(one_minus_alpha_array)
sorted_one_minus_alpha_array = one_minus_alpha_array[sorted_indices]
sorted_expected_neighbour_num_array = expected_neighbour_num_array[sorted_indices]

"""# Get the sorted indices
sorted_indices = np.argsort(epsilon_array)

# Sort the arrays using the sorted indices
new_epsilon_array = epsilon_array[sorted_indices]
sorted_prob_mass_array = prob_mass_array[sorted_indices]
sorted_one_minus_alpha_array = one_minus_alpha_array[sorted_indices]
sorted_expected_neighbour_num_array = expected_neighbour_num_array[sorted_indices]

# Convert back to lists if needed
sorted_epsilon_list = new_epsilon_array.tolist()
sorted_prob_mass_list = sorted_prob_mass_array.tolist()
sorted_one_minus_alpha_list = sorted_one_minus_alpha_array.tolist()
sorted_expected_neighbour_num_list = sorted_expected_neighbour_num_array.tolist()


plt.plot(epsilon_list, prob_mass_list, color='blue', marker='.')
plt.hlines(N, min(epsilon_list), max(epsilon_list), colors='red', linestyles='dashed',label='N')
plt.hlines(N/2, min(epsilon_list), max(epsilon_list), colors='red', linestyles='dotted',label='N/2')
plt.hlines(0, min(epsilon_list), max(epsilon_list), colors='red', linestyles='dashed')
plt.title(f"Probability mass of outbreak (N={N})")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Probability mass")
plt.legend()
#plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/cubic_216/monte_carlo/N_{N}epsilon_prob_mass.png")
plt.show()"""

plt.plot(one_minus_alpha_list, prob_mass_list, color='blue', marker='.')
plt.hlines(N, min(one_minus_alpha_list),max(one_minus_alpha_list), colors='red', linestyles='dashed',label='N')
plt.hlines(N/2, min(one_minus_alpha_list),max(one_minus_alpha_list), colors='red', linestyles='dotted',label='N/2')
plt.hlines(0, min(one_minus_alpha_list),max(one_minus_alpha_list), colors='red', linestyles='dashed')
plt.title(f"Probability mass of outbreak (N={N})")
plt.xlabel(r"$1 - \alpha$")
plt.ylabel("Probability mass")
plt.legend()
#plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/cubic_216/monte_carlo/N_{N}alpha_prob_mass.png")
plt.show()

# obtain the value of epsilon and 1 - alpha which gives the value (N/4, 3N/4)
"""i = 0
while abs(prob_mass_list[i] - N/4) > 1e-1:
    #print(np.absolute(1 - alpha[i] - half_alpha))
    i += 1
one_minus_alpha_N_4 = one_minus_alpha_list[i]
epsilon_N_4 = epsilon_list[i]
i = 0
while abs(prob_mass_list[i] - 3*N/4) > 1e-1:
    #print(np.absolute(1 - alpha[i] - half_alpha))
    i += 1
one_minus_alpha_3N_4 = one_minus_alpha_list[i]
epsilon_3N_4 = epsilon_list[i]

print("sharpness alpha",one_minus_alpha_3N_4 - one_minus_alpha_N_4)
print("sharpness epsilon",epsilon_3N_4 - epsilon_N_4)"""

"""# save the expected_neighbour_num_list with epsilon_list, one_minus_alpha_list to a csv file
df = pd.DataFrame({ 'epsilon' : sorted_epsilon_list, 'one_minus_alpha' : sorted_one_minus_alpha_list, 'expected_neighbour_num' : sorted_expected_neighbour_num_list})
csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/lattice_transition/N_{N}epsilon_expected_neighbour_num.csv"
df.to_csv(csv_file_path, index=False)"""

df = pd.DataFrame({ 'one_minus_alpha' : sorted_one_minus_alpha_array, 'expected_neighbour_num' : sorted_expected_neighbour_num_array})
csv_file_path = f"/Users/bouningen0909/dissertation/week_7/data/lattice_transition/N_{N}epsilon_expected_neighbour_num(3D).csv"
df.to_csv(csv_file_path, index=False)


# get the numerical integral of the expected number of neighbours over one_minus_alpha
integral = trapezoid(sorted_expected_neighbour_num_array, sorted_one_minus_alpha_array)
# plot the expected number of neighbours against 1 - alpha
plt.plot(one_minus_alpha_list, expected_neighbour_num_list, color='blue', marker='.')
#plt.hlines(1, min(one_minus_alpha_list), max(one_minus_alpha_list), colors='red', linestyles='dashed',label='1')
plt.hlines(integral, min(one_minus_alpha_list), max(one_minus_alpha_list), colors='red', linestyles='dashed',label=f'average={integral:.3f}')
plt.xlabel(r"$1 - \alpha$",fontsize=12)
plt.ylabel("Expected number of neighbours",fontsize=12)
#plt.title(f"Expected number of neighbours (N={N})")
plt.legend(fontsize=12)
plt.savefig(f"/Users/bouningen0909/dissertation/week_7/data/lattice_transition/N_{N}alpha_expected_neighbour_num(3Dlattice).png",dpi=300)
plt.show()



