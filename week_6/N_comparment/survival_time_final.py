import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from scipy.stats import ks_2samp

"""
Code to analyse the survival time of the infection. The code is for checking if the survival time of the two compartment model fits that of one compartment model plus some delay time. The hypothesis, that the survival time of the two compartment model (T) is the sum of the survival time of the one compartment model (T_{one compartment}) and the delay time (T_delay), is tested by comparing the cumulative probability of the survival time by KS test. 

The code reads the survival time of the one compartment model and the two compartment model from the csv file and calculates the cumulative probability of the survival time. The delay time is also imported from csv file to incorporate it in the analysis. The cumulative probability distribution are plotted for each epsilon and the KS test is performed to check if the two cumulative probability distributions are consistent.

The mean and the variance of the survival time and delay time of the two compartment model are calculated and shown on the subplot. 
"""

# parameters
epsilon_list = [0.1,0.06,0.03,0.01,0.006,0.003,0.001]
bin_num = 20
indp_sample_num = 1000
sample_num = 1000
r_0 = 2

# bounds for the survival time
upper_bound = 50
lower_bound = 10

path_header = "/Users/bouningen0909/dissertation/week_6/data/N_2_all_to_all_data"

# utility function for getting cumulatice probability
def get_cumulative_prob(array,lower_bound,upper_bound,length=100):
    # truncate the array
    print("length of arary",len(array))
    array = array[array < upper_bound]
    array = array[array > lower_bound]
    array_num = len(array)
    print("array_num",array_num)
    x = np.linspace(lower_bound,upper_bound,length)
    cumu_prob = np.zeros(length)
    for i in range(length):
        cumu_prob[i] = np.sum(array < x[i]) / array_num
    return x,cumu_prob

fig,ax = plt.subplots(3,3,figsize=(20,20))

# importing the survival time of the one compartment model
# note this needs to be prepared beforehand
one_comp_file_path = "dissertation_codes/one_compartment/data/survival_time_R2.csv"
one_comp_df = pd.read_csv(one_comp_file_path, header=None, names=['survival time'])

one_comp_df['survival time'] = pd.to_numeric(one_comp_df['survival time'], errors='coerce')

# Extract the data
one_comp_data = one_comp_df['survival time']
one_comp_data = one_comp_data[1:]
one_comp_mean = np.mean(one_comp_data)
one_comp_var = np.var(one_comp_data)

x,one_comp_prob = get_cumulative_prob(one_comp_data,lower_bound,upper_bound)

mean_list = []
variance_list = []
delay_variance_list = []
median_dict = {}

#epsilon_list = epsilon_list[::-1]
for i, epsilon in enumerate(epsilon_list):
    # Read the CSV file for two compartment model
    csv_file_path = path_header + f"/end_time_dist2_comp_eps{epsilon}R2_given_two_outbreak.csv"
    df = pd.read_csv(csv_file_path, header=None, names=['time till the infection ends'])

    df['time till the infection ends'] = pd.to_numeric(df['time till the infection ends'], errors='coerce')

    # Extract the data
    data = df['time till the infection ends']
    data = data[1:]
    data = np.array(data)

    # get the mean and variance of the data
    mean = np.mean(data)
    var = np.var(data)
    mean_list.append(mean)
    variance_list.append(var)

    # import the data about the delay time 
    delay_list_csv_path = path_header + f"/delay_time_list2_comp_eps{epsilon}R2_test.csv"
    delay_list = np.loadtxt(delay_list_csv_path, delimiter=",")
    delay_list = delay_list[:,1]
    median_dict[epsilon] = round(np.median(delay_list[~np.isnan(delay_list)]),3)

    # calculate the offset data where the delay time is subtracted from the corresponding survival time
    offset_data = []
    j = 0 # subtract the delay time from the survival time only when the delay time is not nan
    for delay in delay_list:
        if not np.isnan(delay):
            print("length of data",len(data))
            offset_data.append(data[j] - delay)
            j += 1
    offset_data = np.array(offset_data)
    print("offset data",offset_data)
   
    x, prob = get_cumulative_prob(data, lower_bound, upper_bound)

    # calculate the mean and the variance of the delay time
    delay_variance = np.var(delay_list[~np.isnan(delay_list)])
    delay_mean = np.mean(delay_list[~np.isnan(delay_list)])

    # obtain the cumulative probability of the offset data
    x,offset_prob = get_cumulative_prob(offset_data,lower_bound,upper_bound)

    # compare the offset data and the one compartment data by KS test
    stats_ks = ks_2samp(one_comp_data,offset_data)
    print(stats_ks)
    p_value = stats_ks[1]

    # plot the data
    index1 = i % 3
    index2 = i // 3
    ax[index2, index1].plot(x, one_comp_prob,color='red', label='One compartment')
    ax[index2, index1].plot(x,prob, color='blue', label=f'Two compartments',linestyle="--",alpha=0.4)
    ax[index2, index1].plot(x,offset_prob, color='blue', label=f'Two compartments without delay')

    # Add annotations for mean and variance
    ax[index2, index1].annotate(f'Mean: {mean:.2f} ({one_comp_mean:.2f})\nVariance: {var:.2f} ({one_comp_var:.2f})\n() is 1 compartment model\nDelay: {delay_mean:.3f}\nDelay Variance: {delay_variance:.3f}\n \n p-value: {p_value:.3e}', xy=(0.45, 0.7), xycoords='axes fraction',
                                fontsize=15, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    if index1 == 0:
        ax[index2,index1].set_ylabel('cumulative probability',fontsize=15)
    if index2 == 2:
        ax[index2,index1].set_xlabel('survival time',fontsize=15)
    ax[index2,index1].vlines(one_comp_mean,0,1,label="Survival time of 1 compartment",linestyle="--",color='red')
    #ax[index2,index1].vlines(34.2,0,1,label=r"2 $\times$ Survival time of 1 compartment",linestyle=":",color='red')
    if i == len(epsilon_list) - 1:
        ax[index2,index1].legend(fontsize=15)

    triggering_file_path = path_header + f"/num_outbreak2_comp_eps{epsilon}R2_test.csv"
    df = pd.read_csv(triggering_file_path, header=None, names=['outbreak_num', 'frequency'])
    #print(df)

    freq = pd.to_numeric(df['frequency'], errors='coerce')
    freq = freq[1:]
    #print(freq)
    cond_prob = freq[3]/(freq[2] + freq[3])
    ax[index2,index1].set_title(rf'$\epsilon$ = {epsilon}, P(2 outbreak | 1 outbreak)={cond_prob:.3f}',fontsize=15)
    ax[index2,index1].tick_params(axis='both', which='major', labelsize=14)  # Major ticks

'''# create the csv file while saves the mean and variance of the data for each epsilon
results_df = pd.DataFrame({
    'epsilon' : epsilon_list,
    'mean' : mean_list,
    'variance' : variance_list,
    'delay_variance' : delay_variance_list
})

# Save DataFrame to CSV
csv_file_path = path_header + f"/mean_var_delay_variance.csv"
results_df.to_csv(csv_file_path, index=False)'''

print(median_dict)
plt.tight_layout()
plt.savefig(path_header + f'/(proper)end_time_dist2_comp_cumulative_prob.png')
plt.show()