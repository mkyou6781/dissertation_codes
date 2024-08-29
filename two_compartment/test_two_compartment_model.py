from two_compartment_gillespie import Two_Comp_Gillespie
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw
from deterministic_dev import Two_Comp_Det
import matplotlib as mpl
from HIT import HIT, naive_HIT

vol_1 = 1000
#vol_2 = int(vol_1/4) 
vol_2 = 1000
init_state = (
    np.array([vol_1 ,1,0,vol_2 -1,1,0])
)
init_state = init_state.astype(int)
time_end = 100
sample_num = 100
volume = np.sum(init_state)
beta_1 = 1.5
gamma_1 = 1
volume_1 = vol_1
beta_2 = 1.5
gamma_2 = 1
volume_2 = vol_2
epsilon = 0.5
model = "coupling"
dt = 0.01
threshold_minor = 100

# Compute the coefficients
S_1 = vol_1
S_2 = vol_2
V_1 = vol_1
V_2 = vol_2
a_00 = gamma_1 / ((1 - epsilon) * beta_1 * S_1 / V_1 + epsilon * beta_1 * S_2 / V_1 + gamma_1)
a_20 = ((1 - epsilon) * beta_1 * S_1 / V_1) / ((1 - epsilon) * beta_1 * S_1 / V_1 + epsilon * beta_1 * S_2 / V_1 + gamma_1)
a_11 = (epsilon * beta_1 * S_2 / V_1) / ((1 - epsilon) * beta_1 * S_1 / V_1 + epsilon * beta_1 * S_2 / V_1 + gamma_1)

b_00 = gamma_2 / ((1 - epsilon) * beta_2 * S_2 / V_2 + epsilon * beta_2 * S_1 / V_2 + gamma_2)
b_02 = ((1 - epsilon) * beta_2 * S_2 / V_2) / ((1 - epsilon) * beta_2 * S_2 / V_2 + epsilon * beta_2 * S_1 / V_2 + gamma_2)
b_11 = (epsilon * beta_2 * S_1 / V_2) / ((1 - epsilon) * beta_2 * S_2 / V_2 + epsilon * beta_2 * S_1 / V_2 + gamma_2)

# compute the parameter of criticality in branching process
rho = (a_20+b_02) + np.sqrt((a_20 - b_02)**2 + a_11*b_11)
print("rho",rho)

# compute the basic reproduction number
a = (beta_1 * S_1)/(gamma_1 * V_1)
b = (beta_2 * S_2)/(gamma_2 * V_2)
r = (1-epsilon)*(a+b)/2 + np.sqrt((1-epsilon)**2*(a+b)**2 - 4*(1-2*epsilon)*a*b)/2
print("r",r)

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']) 
# Obtain the colors from the current color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sir = Two_Comp_Gillespie(init_state, time_end, sample_num, beta_1, gamma_1, volume_1, beta_2, gamma_2, volume_2, epsilon, model,threshold_minor)
chemical_development, time_section = sir.gillespie()

det = Two_Comp_Det(init_state, time_end, sample_num, beta_1, gamma_1, volume_1, beta_2, gamma_2, volume_2, epsilon, model, dt)
det_t, det_solution = det.solve_ode()
det_cluster_1 = det_solution[-1, 2]
det_cluster_2 = det_solution[-1, 5]
hit_time = HIT(det_t,det_solution,beta_1,beta_2,gamma_1,gamma_2,epsilon,volume_1,volume_2, time_end) # time at which the herd immunity threshold is passed (time_end if it is not achieved)
#naive_hit_time = naive_HIT(det_t,det_solution,volume_1,volume_2,r, time_end)

mean_chemical_states, cluster_sizes1, cluster_sizes2 = sir.sampling()
print(cluster_sizes1)
print(cluster_sizes2)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Flatten the axes array and use only the first 5 axes
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

# Plotting the histogram in the first subplot
count1 = np.zeros(vol_1)
count2 = np.zeros(vol_2)
minor_outbreak_1 = 0
minor_outbreak_2 = 0
for i in range(vol_1):
    count1[i] = np.count_nonzero(cluster_sizes1 == i)
    if i < threshold_minor:
        minor_outbreak_1 += np.count_nonzero(cluster_sizes1 == i)

for i in range(vol_2):
    count2[i] = np.count_nonzero(cluster_sizes2 == i)
    if i < threshold_minor:
        minor_outbreak_2 += np.count_nonzero(cluster_sizes2 == i)

def average_above_threshold(arr, threshold):
    # Find the elements that exceed the threshold
    above_threshold = arr[arr > threshold]
    
    # Calculate the average of these elements
    if above_threshold.size > 0:
        average = np.mean(above_threshold)
    else:
        raise ValueError("threshold for the minor outbreak is too large")
    
    return average

# Calculate the average of these elements
average1 = average_above_threshold(cluster_sizes1,threshold_minor)
average2 = average_above_threshold(cluster_sizes2,threshold_minor)

extinction_event = 0
for i in range(len(cluster_sizes1)):
    if (
        cluster_sizes1[i] < threshold_minor
        and cluster_sizes2[i] < threshold_minor
    ):
        extinction_event += 1
extinction_prob = extinction_event / sample_num
     
# Calculate the maximum values of the histograms
hist1, bins1 = np.histogram(cluster_sizes1, bins=100)
hist2, bins2 = np.histogram(cluster_sizes2, bins=100)
max_val1 = np.max(hist1)
max_val2 = np.max(hist2)

# Plot the histograms and vertical lines
ax1.hist(cluster_sizes1, bins=100, color="blue", alpha=0.5, label="cluster size of compartment 1")
ax1.vlines(average1, 0, max_val1, label="Mean of major outbreak", color='blue', linestyle=":")
ax1.vlines(det_cluster_1, 0, max_val1, label="Deterministic", color='blue', linestyle="--")
ax2.hist(cluster_sizes2, bins=100, color="red", alpha=0.5, label="cluster size of compartment 2")
ax2.vlines(average2, 0, max_val2, label="Mean of major outbreak", color='red', linestyle=":")
ax2.vlines(det_cluster_2, 0, max_val2, label="Deterministic", color='red', linestyle="--")

ax1.set_xlabel("Cluster Size")
ax1.set_ylabel("Frequency")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Cluster Size")
ax2.set_ylabel("Frequency")
ax2.grid(True)
ax2.legend()

max_val = np.max((max_val1,max_val2))
ax3.hist(cluster_sizes1, bins=100, color="blue", alpha=0.7, label="cluster size of compartment 1")
ax3.vlines(average1, 0, max_val, label="Mean of major outbreak", color='blue', linestyle=":")
ax3.vlines(det_cluster_1, 0, max_val, label="Deterministic", color='blue', linestyle="--")
ax3.hist(cluster_sizes2, bins=100, color="red", alpha=0.7, label="cluster size of compartment 2")
ax3.vlines(average2, 0, max_val, label="Mean of major outbreak", color='red', linestyle=":")
ax3.vlines(det_cluster_2, 0, max_val, label="Deterministic", color='red', linestyle="--")
ax3.set_xlabel("Cluster Size")
ax3.set_ylabel("Frequency")
ax3.grid(True)
ax3.legend()

labels = [r"$S_1$", r"$I_1$", r"$R_1$", r"$S_2$", r"$I_2$", r"$R_2$"]
for i in range(6):
    ax4.plot(sir.interpolant_time, mean_chemical_states[:, i], label=f"{labels[i]} stochastic", color=colors[i % len(colors)])
    ax4.plot(det_t, det_solution[:, i], label=f"{labels[i]} deterministic", linestyle="--", color=colors[i % len(colors)])
ax4.vlines(hit_time, 0,np.max(np.array([vol_1,vol_2])), label="HIT passed", linestyle="--", color="r")
#ax4.vlines(naive_hit_time, 0,np.max(np.array([vol_1,vol_2])), label="naive HIT passed", linestyle=":", color="r")
ax4.legend()
ax4.set_xlabel("Time")
ax4.set_ylabel("Number")

example_num = 5
labels = [r"$S_1$", r"$I_1$", r"$R_1$", r"$S_2$", r"$I_2$", r"$R_2$"]
for i in range(len(labels)):
    for j in range(example_num):
        if j == 0:
            ax5.plot(sir.interpolant_time, sir.samples[j,:, i], label=f"{labels[i]} stochastic", color=colors[i % len(colors)], alpha=0.5)
        else:
            ax5.plot(sir.interpolant_time, sir.samples[j,:, i], color=colors[i % len(colors)], alpha=0.5)
ax5.vlines(hit_time, 0,np.max(np.array([vol_1,vol_2])), label="HIT passed", linestyle="--", color="r")
ax5.legend()
ax5.set_title("{} Example trajectories".format(example_num))
ax5.set_xlabel("Time")
ax5.set_ylabel("Number")

# Displaying parameters in the second subplot
params_text = f"""Parameters:
[S1,I1,R1,S2,I2,R2] = {init_state}
beta_1: {beta_1}
gamma_1: {gamma_1}
beta_2: {beta_2}
gamma_2: {gamma_2}
epsilon: {epsilon}
Time End: {time_end}
Sample Number: {sample_num}
R_0 (cross infection model): {r:.4f}
rho : {rho:.4f}

Result:
Time when HIT passed: {hit_time}
threshold for minor outbreak: {threshold_minor}
minor outbreak (compartment 1): {minor_outbreak_1} 
minor outbreak (compartment 2): {minor_outbreak_2} 
major outbreak size1(deterministic): {det_cluster_1:.1f}
major outbreak size1(stochastic): {average1:.1f}
major outbreak size2(deterministic): {det_cluster_2:.1f}
major outbreak size2(stochastic): {average2:.1f}
extinction probability: {extinction_prob}
difference between two models:
major outbreak size1: {abs(det_cluster_1-average1):.3f}
major outbreak size2: {abs(det_cluster_2-average2): .3f}
"""

ax6.axis('off')
ax6.text(0.5, 0.5, params_text, fontsize=9, ha='center', va='center')

plt.savefig("/Users/bouningen0909/dissertation/week_four/data/plot{}_eps_{}.png".format(model,epsilon), bbox_inches="tight")
plt.show()
