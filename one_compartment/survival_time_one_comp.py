import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
code for plot the survival time of the system
path: str (the csv file has 1 column)
    the path of the csv file
"""

path = "dissertation_codes/one_compartment/data/survival_time_R1.3.csv"

df = pd.read_csv(path)
data = df.to_numpy()
print(data)

print(np.max(data))
plt.hist(data.flatten(),bins = 20)
plt.xlabel("Survival time",fontsize = 12)
plt.ylabel("Frequency",fontsize = 12)
plt.savefig("dissertation_codes/one_compartment/data/survival_time_hist.png",dpi = 300)
plt.show()