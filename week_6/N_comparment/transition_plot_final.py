from probability_analysis import N_prob, calc_alpha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
from scipy.special import factorial
from scipy.optimize import curve_fit

"""
This code is to analyse the transition point of the outbreak fraction (the number of local outbreaks divided by N) over the value of epsilon. 

We calculate the number of local outbreaks by the polynomial approximation by probability_analysis.py. The number of local outbreaks is calculated over dense values of epsilon. Simultaneously, the value of the alpha corresponding to each epsilon is calculated. Then the outbreak fraction is calculated. The values are cross-checked against the data from the stochastic simulation. 

The analysis proceeds to characterise the transition points. The transition point is identified by taking the half point (half_alpha) of the possible upperbound (1) and the lower bound (1/N). The width of the transition point is calculated by taking the difference between the 1st quartile and the 3rd quartile between the same upperbound and lowerbound. The width is to identify the sharpness of the transition. The half_alpha is fitted against the model (model_2) which is the function of N. The fitting is done by curve_fit. The fitting is plotted against the data.

The plot of the outbreak fraction is plotted against the value of 1-alpha. Then the plot of the half_alpha is plotted against N with the model fit and the log-log plot of the half_alpha is plotted against N with the model fit.
"""

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold']) 
# Obtain the colors from the current color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

epsilon_list_for_SS = np.array([0.03,0.01,0.006,0.003,0.001])
# add additional epsilon values which take values between 0.01 and 0.02 with 10 values
epsilon_list = np.concatenate((epsilon_list_for_SS,np.linspace(0.0001,0.1,1000)))

#N_list = [3,8,16,30]
N_list = [4,8,16,32]
#N_list = range(3,30)
volume = 300
r_0 = 2

path_header_input = "dissertation_codes/week_6/data"
path_header_output = "dissertation_codes/week_6/data/transition"

half_alpha_list = []


for N in N_list:
    if (N == 3) or (N == 16):
        outbreak_fraction_list_from_SS = []
        one_minus_alpha_list_SS = 1 - calc_alpha(r_0,epsilon_list_for_SS,N,volume)
        print(one_minus_alpha_list_SS)
        for epsilon in epsilon_list_for_SS:
            epidemic_size_csv_path = path_header_input + f"/N_{N}_all_to_all_data/small_vol/num_outbreak{N}_comp_eps{epsilon}R2_test.csv"

            epidemic_size = pd.read_csv(epidemic_size_csv_path)
            # convert to numpy array
            epidemic_size = epidemic_size.to_numpy()
            num_outbreak = epidemic_size[:,1]

            prob = num_outbreak[1:] / np.sum(num_outbreak[1:])

            # calculate the outbreak fraction for SS data
            outbreak_fraction = np.sum(np.multiply(np.arange(1,N+1),prob)) / N
            outbreak_fraction_list_from_SS.append(outbreak_fraction)
        # plot the SS data to compare with the model
        plt.scatter(one_minus_alpha_list_SS, outbreak_fraction_list_from_SS, color=colors[N_list.index(N) %len(colors)],marker="+")

    # convert the value of epsilon to 1 - alpha
    one_minus_alpha_list = 1 - calc_alpha(r_0,epsilon_list,N,volume)

    outbreak_fraction_list = []
    outbreak_fraction_list_from_SS = []
    for epsilon in epsilon_list:
        # calculate the probability of the number of local outbreaks with the polynomial approximation
        prob = N_prob(r_0,epsilon,N,volume)
        outbreak_num = np.arange(1,N+1)
        outbreak_fraction = np.sum(np.multiply(outbreak_num,prob)) / N
        outbreak_fraction_list.append(outbreak_fraction)
    
    """for epsilon in epsilon_list_for_SS:
        epidemic_size_csv_path = f"/Users/bouningen0909/dissertation/week_6/data/N_16_all_to_all_data/small_vol/num_outbreak16_comp_eps{epsilon}R2_test.csv"
        # import csv
        epidemic_size = pd.read_csv(epidemic_size_csv_path)
        # convert to numpy array
        epidemic_size = epidemic_size.to_numpy()
        num_outbreak = epidemic_size[:,1]

        prob = num_outbreak[1:] / np.sum(num_outbreak[1:])
        outbreak_fraction = np.sum(np.multiply(np.arange(1,N+1),prob))
        outbreak_fraction_list_from_SS.append(outbreak_fraction)"""
    # sort the list of epsilon, 1 - alpha and probability mass in ascending order of epsilon
    epsilon_list = np.array(epsilon_list)
    one_minus_alpha_list = np.array(one_minus_alpha_list)
    outbreak_fraction_list = np.array(outbreak_fraction_list)
    idx = np.argsort(epsilon_list)
    epsilon_list = epsilon_list[idx]
    one_minus_alpha_list = one_minus_alpha_list[idx]
    outbreak_fraction_list = outbreak_fraction_list[idx]

    # obtain the half alpha value and 1st and 3rd quartile value
    half = (1 - 1/N) / 2 + 1/N
    quart_1 = (1 - 1/N) / 4 + 1/N
    quart_3 = 3 * (1 - 1/N) / 4 + 1/N
    print("half",half)

    # obtain the values of outbreak fraction which are closest to the half, 1st quartile and 3rd quartile. Then find the corresponding 1-alpha values
    half_diff = np.abs(outbreak_fraction_list - half)
    half_alpha = one_minus_alpha_list[np.argmin(half_diff)]
    half_alpha_list.append(half_alpha)

    quart_1_diff = np.abs(outbreak_fraction_list - quart_1)
    quart_1_alpha = one_minus_alpha_list[np.argmin(quart_1_diff)]

    quart_3_diff = np.abs(outbreak_fraction_list - quart_3)
    quart_3_alpha = one_minus_alpha_list[np.argmin(quart_3_diff)]

    # plot the vertial line for the half alpha and the width of the transition
    plt.vlines(half_alpha,0,half,linestyle="--",color=colors[N_list.index(N) %len(colors)],label=rf"$N = {N}$"+r"$, 1 - \alpha_{1/2}$"+rf"$ = {half_alpha:.3f}$ width = {quart_3_alpha - quart_1_alpha:.3f}")
    plt.plot(one_minus_alpha_list, outbreak_fraction_list, color=colors[N_list.index(N) %len(colors)])
    
"""plt.hlines(1, min(epsilon_list), max(epsilon_list), colors='k', linestyles='dashed',label='1')
plt.hlines(0.5, min(epsilon_list), max(epsilon_list), colors='k', linestyles='dotted',label='0.5')
plt.hlines(0, min(epsilon_list), max(epsilon_list), colors='k', linestyles='dashed')"""
plt.xlabel(r"$1 - \alpha$",fontsize=12)
plt.ylabel("Outbreak Fraction",fontsize=12)
plt.scatter([],[],color="black",marker="+",label="Stochastic simulation")
plt.legend(fontsize=12)
#ax.set_xlim(0,1)
#ax[1].set_xlabel(r"$\epsilon$",fontsize=12)
#ax[1].set_ylabel("Probability mass",fontsize=12)
#ax[1].legend(fontsize=12)
#plt.savefig(path_header_output + "/transition_alpha.png",dpi=300)
plt.show()

# model fitting
N_list = np.array(N_list)
def model_2(N,a,c):
    return a * (c/factorial(N-1)) ** (1/(N-1))
# fit the data against model_2
popt, pcov = curve_fit(model_2,N_list,half_alpha_list)
print("popt",popt)
print("pcov",pcov)
model_2_data = model_2(N_list,popt[0],popt[1])
plt.scatter(N_list,half_alpha_list,marker="+")
N_list = np.array(N_list)
a_fit = round(popt[0],3)
c_fit = round(popt[1],3)
plt.plot(N_list
         ,model_2_data,color="red",linestyle="--",label=rf"$y = {a_fit} \times ( {c_fit}$" +r"$/(N-1)!)^{\{1/(N-1)\}}$")
plt.xlabel(r"$N$",fontsize=12)
plt.ylabel(r"$1 - \alpha_{1/2}$",fontsize=12)
plt.legend(fontsize=12)
plt.savefig(path_header_output + "/alpha_model7.png",dpi=300)
plt.show()

# plot the log-log plot of the half alpha against N
log_N_list = np.log(np.array(N_list))
log_half_alpha_list = np.log(np.array(half_alpha_list))
#plt.scatter(log_N_list,np.log(model),label="model")
#plt.scatter(log_N_list,np.log(model_2),label="model 2")
slope, intercept, r_value, p_value, std_err = stats.linregress(log_N_list,log_half_alpha_list)
print("slope",slope)
print("intercept",intercept)
print("r_value",r_value)
print("p_value",p_value)
print("std_err",std_err)


#plt.plot(log_N_list, np.log(model_2_data), color="green",label=rf"$y = {slope_2:.3f}x + {intercept_2:.3f}$")
plt.plot(log_N_list, np.log(model_2_data), color="red",linestyle="--",label=rf"$y = {popt[0]:2f} \times$" +r"$(N!)^{(-1/N)}$")
plt.scatter(log_N_list,log_half_alpha_list,marker="+")
plt.legend(fontsize=15)
plt.xlabel(r"$\ln(N)$",fontsize=12)
plt.ylabel(r"$\ln(1 - \alpha_{1/2})$",fontsize=12)
plt.savefig(path_header_output + "/log_log_alpha7.png",dpi=300)
plt.show()


    