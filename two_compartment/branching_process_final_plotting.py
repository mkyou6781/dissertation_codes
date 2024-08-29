import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the following two csv files 
file_path_1 = "/Users/bouningen0909/dissertation/week_4/data/branchingprocess_with_validation/branching_process.csv"
file_path_2 = "/Users/bouningen0909/dissertation/week_4/data/branchingprocess_with_validation/branching_process_32.csv"

df_1 = pd.read_csv(file_path_1)
df_2 = pd.read_csv(file_path_2)

#obtain the data from the columns beta_2,extinction1,extinction2,sol1,sol2 from df_1

extinction1_list = df_1["extinction1"]
extinction2_list = df_1["extinction2"]
sol1_list = df_1["sol1"]
sol2_list = df_1["sol2"]
beta_2_list = df_1["beta_2"]

# obtain the data from the columns beta_2,extinction from df_2
extinction_list_3 = df_2["extinction"]

# plot the data
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(beta_2_list, extinction1_list, label="simulation",marker = ".", color="r")
ax[0].scatter(beta_2_list, sol1_list, label="branching process",marker = "+", color="b")
ax[0].set_xlabel(r"$\beta_2$",fontsize=15)
ax[0].set_ylabel(r"$P(extinction)$",fontsize=15)
ax[0].set_title(r"$P(extinction \mid I_1(0) = 1, I_2(0)=0)$",fontsize=15)
ax[0].legend(fontsize=15)
ax[0].annotate("(a)", xy=(-0.05, 1.12), xycoords='axes fraction', fontsize=14,
               horizontalalignment='left', verticalalignment='top')

ax[1].scatter(beta_2_list, extinction2_list, label="simulation",marker = ".", color="r")
ax[1].scatter(beta_2_list, sol2_list, label="branching process",marker = "+", color="b")
ax[1].set_xlabel(r"$\beta_2$",fontsize=15)
ax[1].set_ylabel(r"$P(extinction)$",fontsize=15)
ax[1].set_title(r"$P(extinction \mid I_1(0) = 0, I_2(0)=1)$",fontsize=15)
ax[1].legend(fontsize=15)
ax[1].annotate("(b)", xy=(-0.05, 1.12), xycoords='axes fraction', fontsize=14,
               horizontalalignment='left', verticalalignment='top')

sol1_list = np.array(sol1_list)
sol2_list = np.array(sol2_list)
fit_list = sol1_list ** 3 * sol2_list ** 2
ax[2].scatter(beta_2_list, extinction_list_3, label="simulation",marker = ".", color="r")
ax[2].scatter(beta_2_list, fit_list, label="branching process",marker = "+", color="b")
ax[2].set_xlabel(r"$\beta_2$",fontsize=15)
ax[2].set_ylabel(r"$P(extinction)$",fontsize=15)
ax[2].set_title(r"$P(extinction \mid I_1(0) = 3, I_2(0)=2)$",fontsize=15)
ax[2].legend(fontsize=15)
ax[2].annotate("(c)", xy=(-0.05, 1.12), xycoords='axes fraction', fontsize=14,
               horizontalalignment='left', verticalalignment='top')
#plt.tight_layout()
plt.savefig("/Users/bouningen0909/dissertation/week_4/data/branchingprocess_with_validation/branching_processfinal_plot.png", bbox_inches="tight")
plt.show()
