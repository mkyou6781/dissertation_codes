import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from network_gillespie import Network_Gillespie
import scipy.sparse as sp
from network_r_rho import R_eff, rho, calc_q
from matplotlib.colors import Normalize

"""
This is a code for 


"""


volume = 300
N = 16
total_network_num = 30
trial_num = 8
beta = 2
gamma = 1
epsilon = 0.01
network_type = "lattice"
r_0 = beta / gamma

immunity_type = ["random", "natural"]
# immunity_type = ["random"]
color_list = ["blue", "orange"]

# Create the figure and axis outside the loop
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
# fig, ax = plt.subplots(1,3, figsize=(21,7))

path_header = "dissertation_codes/week_7/data/herd_immunity/"

# structure of the data
#            fraction   trial1_comp   trial2_comp    ...   trial1_infected_num  trial2_infected_num ...
# network1      0.4        5              3          ...     20                  30                 ...
# network2      0.4        7              71         ...     20                  30
# ...
# networkK      0.4        1              27          ...     20                   30                 ...
# network1      0.5        5              3          ...     20                  30                 ...
# ...
# networkK      0.5        1              27          ...     20                   30                 ...


# importing the data to analyse
# please put the path correctly
for immunity in immunity_type:
    if network_type == "all-to-all":
        if immunity == "random":
            csv_file_path = (
                path_header
                + f"complete_network/infected_num/complete_(random)infected_num_vs_fraction_{N}_I_1.csv"
            )
        elif immunity == "natural":
            csv_file_path = (
                path_header
                + f"complete_network/infected_num/complete_infected_num_vs_fraction_{N}_I_1.csv"
            )
    elif network_type == "lattice":
        if immunity == "random":
            csv_file_path = (
                path_header
                + f"lattice_network/infected_num/lattice_(random)infected_num_vs_fraction_{N}_eps_0.01_I_1.csv"
            )

        elif immunity == "natural":
            csv_file_path = (
                path_header
                + f"lattice_network/infected_num/lattice_infected_num_vs_fraction_{N}_eps_0.01_I_1.csv"
            )
    orig_df = pd.read_csv(csv_file_path)
    data = orig_df.to_numpy()
    data = data[:, 1:]  # Drop the first column because it corresponds to the labels

    # Normalize the data by the surviving population
    fraction_array = data[:, 0]
    comp_data = data[
        :, : 1 + trial_num
    ]  # data of the fraction of immunised population and the strating compartments
    unique_fraction_array = np.unique(fraction_array)  # remove duplicates
    surviving_population = volume * N * (1 - fraction_array)  # there are duplicates
    surviving_population_array = np.zeros(data[:, 1 + trial_num :].shape)
    for i in range(surviving_population_array.shape[0]):
        surviving_population_array[i, :] = surviving_population[i]
    # Normalize the data by surviving population by element wise division
    data[:, 1 + trial_num :] = np.divide(
        data[:, 1 + trial_num :], surviving_population_array
    )

    # Generate the network index labels
    network_name_all = [f"network{i}" for i in range(total_network_num)]
    trial_index = [f"trial{i}" for i in range(trial_num)]
    trial_comp_index = [f"trial_comp{i}" for i in range(trial_num)]

    # Create the column names for the DataFrame
    columns = ["fraction"] + trial_comp_index + trial_index
    # Create the DataFrame
    orig_df = pd.DataFrame(
        data, columns=columns, index=network_name_all * len(unique_fraction_array)
    )
    network_stats_df = orig_df.copy()
    # Calculate mean and variance over the networks
    # axis = 1 means the mean is calculated over the trial (given the network is fixed)
    network_stats_df["mean"] = network_stats_df.iloc[:, 1 + trial_num :].mean(axis=1)
    # same as the mean
    network_stats_df["variance"] = network_stats_df.iloc[:, 1 + trial_num :].var(axis=1)

    # Group the data by fraction and aggregate the network values
    grouped = network_stats_df.groupby("fraction").mean()
    # note that the compartment is weirdly averaged so ignore it

    # Prepare data for the boxplot
    box_data = [
        network_stats_df[network_stats_df["fraction"] == frac]
        .iloc[:, 1 + trial_num : -2]
        .values.flatten()
        for frac in grouped.index
    ]

    # Create the boxplot with custom colors
    positions = (
        np.arange(len(unique_fraction_array))
        + (-1 + 2 * immunity_type.index(immunity)) * 0.1
        + 0.2
    )
    boxplot = ax[0, 0].boxplot(
        box_data,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        notch=False,
        whiskerprops={"linewidth": 1},  # Set whiskers to have no length
        capprops={"linewidth": 1},  # Hide caps
        showfliers=False,
        whis=[0, 100],
    )

    # Set the same color for all boxes for each immunity type
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color_list[immunity_type.index(immunity)])

    # plot the mean as the point on the plot
    means = [np.mean(data) for data in box_data]
    ax[0, 0].plot(
        positions, means, marker="x", color="black", linestyle="none", markersize=10
    )

    # Add a scatter point for the legend
    ax[0, 0].scatter(
        [],
        [],
        color=color_list[immunity_type.index(immunity)],
        marker="s",
        label=f"{immunity} immunity",
    )
    # add grid lines
    ax[0, 0].grid(visible=True, axis="x")

    ### calculate R_eff and extinction probability q from the adjacency matrix and immunised network
    # note rho is not calculated because rho > 1 and R_eff > 1 are equivalent
    R_eff_for_each_fraction = []
    q_mean_for_each_fraction = []
    # add columns in orig_df to store the extinction probability
    q_index = ["ext_prob_trial{}".format(i) for i in range(trial_num)]
    orig_df[q_index] = pd.DataFrame(
        np.zeros((orig_df.shape[0], trial_num)), index=orig_df.index
    )

    variance_mean_list = (
        []
    )  # this is for the variance of the surviving population over the compartment to check the distribution of the immunity. The mean of such variance over networks with the fixed fraction of the immunised population is calculated
    for fraction in unique_fraction_array:
        if network_type == "all-to-all":
            if immunity == "random":
                network_file_path = (
                    path_header
                    + f"complete_network/N_16/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
                )
            elif immunity == "natural":
                network_file_path = (
                    path_header
                    + f"complete_network/N_16/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
                )
        if network_type == "lattice":
            if immunity == "random":
                network_file_path = (
                    path_header
                    + f"lattice_network/random_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
                )
            if immunity == "natural":
                network_file_path = (
                    path_header
                    + f"lattice_network/herd_immunity_network_N_{N}_comp_eps0.1R2.0_fraction{fraction}.csv"
                )
        # all the dummy variables which are not used
        init_state = np.zeros(3 * N)
        time_end = 100
        sample_num = 100
        threshold_minor = 100

        # obtain the adjacency matrix
        n_comp = Network_Gillespie(
            init_state,
            time_end,
            sample_num,
            beta,
            gamma,
            volume,
            epsilon,
            threshold_minor,
            N,
            network_type,
            collect_outbreak=True,
        )
        n_comp.generate_adj_matrix()
        adj_matrix = n_comp.adj_matrix
        # convert adj_matrix to dense matrix
        adj_matrix = sp.csr_matrix.todense(adj_matrix)
        # note that the adjacency matrix have their diagonal value 1
        # so we need to fill the diagonal mannually
        np.fill_diagonal(adj_matrix, (1 - epsilon))

        df = pd.read_csv(network_file_path)

        # Convert the DataFrame to a numpy array
        data_array = df.to_numpy()
        total_network_num = data_array.shape[0]
        q_array = np.zeros(
            (N, total_network_num)
        )  # extinction probability from branching process: q

        variance_mean = np.mean(
            np.var(data_array, axis=1)
        )  # check the description above
        variance_mean_list.append(variance_mean)
        R_eff_array = np.zeros(data_array.shape[0])
        q_mean_array = np.zeros(data_array.shape[0])
        for j in range(data_array.shape[0]):
            # Convert the DataFrame to a numpy array
            immunised_network = data_array[j]
            # immunised_network = np.zeros(N)

            r_eff = R_eff(
                adj_matrix, beta, gamma, epsilon, volume, N, immunised_network
            )  # effective reproduction number: R_eff
            R_eff_array[j] = r_eff

            q = calc_q(adj_matrix, beta, gamma, epsilon, volume, N, immunised_network) # extinction probability from branching process: q
            q_array[:, j] = q
            q_mean_array[j] = np.mean(
                q
            )  # the mean of the extinction probability over the compartments
            # extract the starting compartments for this trial
            comp = orig_df.loc[orig_df["fraction"] == fraction].iloc[
                j, 1 : 1 + trial_num
            ]
            comp = comp.to_numpy()
            # get the compartment and substitute the extinction prob correspond to that to the dataframe
            for k in range(trial_num):
                orig_df.iloc[
                    j
                    + np.where(unique_fraction_array == fraction)[0][0]
                    * total_network_num,
                    1 + 2 * trial_num + k,
                ] = q[int(comp[k] - 1)]

            # uncomment to plot the surviving population and extinction probability over the compartments as two-dimensional plot
            """if j == 0:
                fig2, ax2 = plt.subplots(1, 1, figsize=(5,5))
                vmin = 0  # You can set a specific value here
                vmax = 300 # You can set a specific value here

                # Normalize the colors
                norm = Normalize(vmin=vmin, vmax=vmax)
                im = ax2.imshow(immunised_network.reshape(int(np.sqrt(N)),int(np.sqrt(N))), cmap='viridis', aspect='auto',norm = norm)
                # add colorbar
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Immunised population')
                # show the extinction probability on the plot
                #colors = np.power(q,10)

                # Normalize the colors
                norm = Normalize(vmin=vmin, vmax=vmax)
                for i in range(N):
                    x = i % int(np.sqrt(N))
                    y = i // int(np.sqrt(N))
                    #sc = ax.scatter(x, y, s=colors[i] * 100, alpha=0.3, c=colors[i], cmap='hot',norm=norm)
                    # annotate the probability
                    ax2.text(x, y, f"{q[i]:.2f}", ha='center', va='center', color='black')
                if network_type == "all-to-all":
                    if immunity == "random":
                        fig_file_path = path_header + f"complete_network/N_16/(random)example_network{fraction}.png"
                    elif immunity == "natural":
                        fig_file_path = path_header + f"complete_network/N_16/example_network{fraction}.png"
                if network_type == "lattice":
                    if immunity == "random":
                        fig_file_path = path_header + f"lattice_network/(random)example_network{fraction}.png"
                    if immunity == "natural":
                        fig_file_path = path_header + f"lattice_network/example_network{fraction}.png"

                fig2.savefig(fig_file_path,dpi=300)"""

        R_eff_for_each_fraction.append(R_eff_array)
        q_mean_for_each_fraction.append(q_mean_array)

        # save the q_array to csv
        q_df = pd.DataFrame(
            q_array, columns=[f"network{i}" for i in range(total_network_num)]
        )
        q_index = [f"compartment{i}" for i in range(N)]
        if network_type == "all-to-all":
            if immunity == "random":
                q_file_path = (
                    path_header
                    + f"complete_network/N_16/(q)random_network_N_{N}_comp_eps{epsilon}R{r_0}_fraction{fraction}.csv"
                )
            elif immunity == "natural":
                q_file_path = (
                    path_header
                    + f"complete_network/N_16/(q)herd_immunity_network_N_{N}_comp_eps{epsilon}R{r_0}_fraction{fraction}.csv"
                )
        if network_type == "lattice":
            if immunity == "random":
                q_file_path = (
                    path_header
                    + f"lattice_network/(q)random_network_N_{N}_comp_eps{epsilon}R{r_0}_fraction{fraction}.csv"
                )
            if immunity == "natural":
                q_file_path = (
                    path_header
                    + f"lattice_network/(q)herd_immunity_network_N_{N}_comp_eps{epsilon}R{r_0}_fraction{fraction}.csv"
                )
        q_df.to_csv(q_file_path, index=q_index)
    # make a box plots for both R_eff
    # R_eff
    box_data = R_eff_for_each_fraction
    # Create the boxplot with custom colors
    # print("positions",positions.shape)
    boxplot = ax[0, 1].boxplot(
        box_data,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        notch=False,
        whiskerprops={"linewidth": 1},  # Set whiskers to have no length
        capprops={"linewidth": 1},  # Hide caps
        showfliers=False,
        whis=[0, 100],
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color_list[immunity_type.index(immunity)])
    # add mean point
    means = [np.mean(data) for data in box_data]
    ax[0, 1].plot(
        positions, means, marker="x", color="black", linestyle="none", markersize=10
    )
    # Add a scatter point for the legend
    ax[0, 1].scatter(
        [],
        [],
        color=color_list[immunity_type.index(immunity)],
        marker="s",
        label=f"{immunity} immunity",
    )
    ax[0, 1].grid(visible=True, axis="x")
    
    
    ### q
    box_data = q_mean_for_each_fraction
    boxplot = ax[1, 0].boxplot(
        box_data,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        notch=False,
        whiskerprops={"linewidth": 1},  # Set whiskers to have no length
        capprops={"linewidth": 1},  # Hide caps
        showfliers=False,
        whis=[0, 100],
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color_list[immunity_type.index(immunity)])
    # add mean point
    means = [np.mean(data) for data in box_data]
    ax[1, 0].plot(
        positions, means, marker="x", color="black", linestyle="none", markersize=10
    )
    # Add a scatter point for the legend
    ax[1, 0].scatter(
        [],
        [],
        color=color_list[immunity_type.index(immunity)],
        marker="s",
        label=f"{immunity} immunity",
    )
    ax[1, 0].grid(visible=True, axis="x")

    ### additional scatter plot between the fraction of total new infection and the extinction probability q at the corresponding starting compartment
    infection_fraction = orig_df.iloc[
        :, 1 + trial_num : 1 + 2 * trial_num
    ].values.flatten()
    infection_fraction = np.array(infection_fraction)
    ext_prob = orig_df.iloc[:, 1 + 2 * trial_num :].values.flatten()
    ext_prob = np.array(ext_prob)
    # calcualte the correlation between the infection fraction and the extinction probability
    mean_infection_fraction = np.mean(infection_fraction)
    mean_ext_prob = np.mean(ext_prob)
    cov = 0
    for k in range(len(infection_fraction)):
        cov += (infection_fraction[k] - mean_infection_fraction) * (
            ext_prob[k] - mean_ext_prob
        )
    correlation = (cov / (len(infection_fraction) - 1)) / (
        np.std(infection_fraction) * np.std(ext_prob)
    )

    ax[1, 1].scatter(
        infection_fraction,
        ext_prob,
        color=color_list[immunity_type.index(immunity)],
        label=f"{immunity} immunity \n Correlation = {correlation:.2f}",
        marker=".",
        alpha=0.3,
    )
    print(immunity, variance_mean_list)


ax[0, 0].scatter([], [], marker="x", color="black", label="mean")
# Set custom tick labels centered between the two boxplots for each fraction
ax[0, 0].set_xticks(np.arange(len(grouped.index)) + 0.2)
ax[0, 0].set_xticklabels([f"{frac:.2f}" for frac in grouped.index])

# Add labels and title
# plt.title('Box Plot of Network Values by Fraction')
ax[0, 0].set_xlabel(r"Fraction of immunised, $h$", fontsize=15)
ax[0, 0].set_ylabel("Fraction of total new infection", fontsize=15)
ax[0, 0].vlines(
    positions[3] - 0.1, -1, 2, colors="red", linestyles="dashed", label=r"$1 - 1/R_0$"
)
ax[0, 0].set_ylim(-0.02, 0.3)
ax[0, 0].legend(fontsize=15)
ax[0, 0].hlines(
    0,
    np.min(positions) - 1,
    np.max(positions) + 1,
    linestyles="dashed",
    color="black",
    alpha=0.3,
)
ax[0, 0].set_title("Total infection", fontsize=15)
# add the label (a) on the top left
ax[0, 0].text(
    -0.05, 1.1, "(a)", transform=ax[0, 0].transAxes, fontsize=17, va="top", ha="right"
)

ax[0, 1].hlines(
    1, np.min(positions) - 1, np.max(positions) + 1, linestyles="dashed", color="black"
)
ax[0, 1].set_xticks(np.arange(len(grouped.index)) + 0.2)
ax[0, 1].set_xticklabels([f"{frac:.2f}" for frac in grouped.index])
ax[0, 1].set_xlabel(r"Fraction of immunised, $h$", fontsize=15)
ax[0, 1].set_ylabel(r"$R_{eff}$", fontsize=15)
ax[0, 1].set_title("Effective reproduction number \n" + r"$R_{eff}$", fontsize=15)
ax[0, 1].scatter([], [], marker="x", color="black", label="mean")
ax[0, 1].vlines(
    positions[3] - 0.1, -1, 2, colors="red", linestyles="dashed", label=r"$1 - 1/R_0$"
)
ax[0, 1].set_ylim(0.6, 2)
ax[0, 1].legend(fontsize=15)
ax[0, 1].text(
    -0.05, 1.1, "(b)", transform=ax[0, 1].transAxes, fontsize=17, va="top", ha="right"
)

ax[1, 0].hlines(
    1, np.min(positions) - 1, np.max(positions) + 1, linestyles="dashed", color="black"
)
ax[1, 0].set_xlabel(r"Fraction of immunised, $h$", fontsize=15)
ax[1, 0].set_ylabel(r"$\bar{q}$", fontsize=15)
ax[1, 0].set_xticks(np.arange(len(grouped.index)) + 0.2)
ax[1, 0].set_xticklabels([f"{frac:.2f}" for frac in grouped.index])
ax[1, 0].set_title("Mean of extinction probabilities \n" + r"$\bar{q}$", fontsize=15)
ax[1, 0].scatter([], [], marker="x", color="black", label="mean")
ax[1, 0].vlines(
    positions[3] - 0.1, -1, 2, colors="red", linestyles="dashed", label=r"$1 - 1/R_0$"
)
ax[1, 0].set_ylim(0.65, 1.05)
ax[1, 0].legend(fontsize=15)
ax[1, 0].text(
    -0.05, 1.1, "(c)", transform=ax[1, 0].transAxes, fontsize=17, va="top", ha="right"
)

ax[1, 1].set_xlabel("Fraction of total new infection", fontsize=15)
ax[1, 1].set_ylabel(r"Extinction probability $q_i$", fontsize=15)
ax[1, 1].legend(fontsize=15, loc="upper right")
ax[1, 1].text(
    -0.05, 1.1, "(d)", transform=ax[1, 1].transAxes, fontsize=17, va="top", ha="right"
)

# make a scatter plot of trial{k} and the extinction probability
# extract the data from the orig_df

plt.tight_layout()
if network_type == "all-to-all":
    plt.savefig(
        path_header+"complete_network/infected_num/box_plot",
        dpi=300,
    )
elif network_type == "lattice":
    plt.savefig(
        path_header + "lattice_network/infected_num/box_plot",
        dpi=300,
    )
plt.show()
