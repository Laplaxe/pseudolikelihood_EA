# This code produces the plot of the final magnetization for initialization on training and test examples
# L = 16 case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

############### load the data ###############

#first, get the temperature
parser = argparse.ArgumentParser(description='Train PLLModel with specified temperature.')
parser.add_argument('--temperatures', type=float, nargs='+', required=True, help='List of temperature values for the simulation')
args = parser.parse_args()
temperatures = args.temperatures

seed = 3397150145
N = 256

#############################################
############# create the plot ###############

# Convert all_ms to numpy array if it's not already

# Plotting
plt.figure(figsize=(10, 6))

colors = ["Teal", "Firebrick", "Goldenrod", "Black", "Darkgray"]

for n, T in enumerate(temperatures):
    data = pd.read_csv(f"../../Dati/Omega/Results/L16_seed{seed}/L16_seed{seed}_T{T:.2f}.txt", sep=" ", header = None, names = ["P", "m_train", "m_test", "m_GS", "E_train", "E_test", "E_gs", "gamma","R"])
    data = data.groupby("P").mean().reset_index()
    all_ms_np = data.to_numpy()
    plt.plot(all_ms_np[:, 0]/N, all_ms_np[:, 7], label = f"{T}", marker='o', markersize=10, linestyle='-', color=colors[n])


# Adding labels and title
plt.tight_layout()
#plt.xscale("log")
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel(r"$\alpha$", fontsize=28)
plt.ylabel(r"$\gamma$", fontsize=28)

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
plt.xscale("log")
plt.savefig(f"../Figures/gamma.png")

#############################################