# This code produces the plot of the final magnetization for initialization on training and test examples
# L = 16 case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

############### load the data ###############

#first, get the temperature
parser = argparse.ArgumentParser(description='Train PLLModel with specified temperature.')
parser.add_argument('--temperature', type=float, required=True, help='Temperature value for the simulation')
args = parser.parse_args()
T = args.temperature

seed = 3397150145
data = pd.read_csv(f"../../Dati/Omega/Results/L16_seed{seed}/L16_seed{seed}_T{T:.2f}.txt", sep=" ", header = None, names = ["P", "m_train", "m_test", "E_train", "E_test", "gsdist_train", "gsdist_test"])
count_P_equals_1 = data[data["P"] == 1].shape[0]
print(f"Used {count_P_equals_1} runs")
data = data.groupby("P").mean().reset_index()
all_ms_np = data.to_numpy()

asymptotics = pd.read_csv(f"../../Dati/Omega/Results/L16_seed{seed}/asymptotics.txt", sep=" ", header = None, names = ["T", "m_train", "m_test", "m_perf", "E_training", "E_test", "E_perfect", "gsdist_train", "gsdist_test", "gsdist_perfect"])
asymptotics = asymptotics[asymptotics["T"] == T]

#magnetization_perfect = 0.8846750259399414 #value of the test_magnetization when J is the exact one
#############################################
############# create the plot ###############

# Convert all_ms to numpy array if it's not already

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(all_ms_np[:, 0], all_ms_np[:, 1], marker='o', linestyle='-', color='teal', label='Magnetization')
plt.plot(all_ms_np[:, 0], all_ms_np[:, 2], marker='o', linestyle='-', color='firebrick', label='Magnetization test')
# Plot horizontal line at magnetization perfect
plt.axhline(y=asymptotics["m_perf"].values[0], color='black', linestyle='--', label=r'$P = \infty$')
plt.axhline(y=asymptotics["m_train"].values[0], color='darkgray', linestyle='-.', label='P = 130072')


# Adding labels and title
plt.xlabel('P', fontsize=20)
plt.ylabel('Magnetization', fontsize=20)
plt.title(f'T = {T:.2f}\nN = 256, 5000 epochs with GD and Exponential Decay\n1000 zero-temperature steps', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adding grid
plt.grid(True)

# Adding legend
plt.legend(fontsize=20)

# Improving layout
plt.tight_layout()

# Saving the plot as a high-resolution image
#plt.savefig('magnetization_vs_P.png', dpi=300)

# Display the plot
plt.tight_layout()

#lin-scale
plt.xlim(0,2500)
plt.savefig(f"../Figures/stability_withP_L16_T{T:.2f}_lin.png")

#log-scale
plt.xscale('log')
plt.xlim(1,15000)
plt.savefig(f"../Figures/stability_withP_L16_T{T:.2f}_log.png")
#############################################