# This code produces the plot of the final magnetization for initialization on training and test examples
# L = 8 case

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############### load the data ###############

data = pd.read_csv("../../Dati/Omega/Results/L8_seed42/L8_seed42", sep=" ", header = None, names = ["P", "m_train", "m_test"])
data = data.groupby("P").mean().reset_index()
all_ms_np = data.to_numpy()

magnetization_perfect = 0.8846750259399414 #value of the test_magnetization when J is the exact one
#############################################
############# create the plot ###############

# Convert all_ms to numpy array if it's not already

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(all_ms_np[:, 0], 2*all_ms_np[:, 1]-1, marker='o', linestyle='-', color='teal', label='Magnetization')
plt.plot(all_ms_np[:, 0], 2*all_ms_np[:, 2]-1, marker='o', linestyle='-', color='firebrick', label='Magnetization test')
# Plot horizontal line at magnetization perfect
plt.axhline(y=2*magnetization_perfect-1, color='black', linestyle='--', label=r'$P = \infty$')
#plt.axhline(y=2*magnetizations_largealpha_test.item()-1, color='darkgray', linestyle=':', label='Perfect Magnetization')


# Adding labels and title
plt.xlabel('P', fontsize=20)
plt.ylabel('Magnetization', fontsize=20)
plt.title('N = 64, 5000 epochs with GD and Exponential Decay\n1000 zero-temperature steps', fontsize=16)
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
plt.xlim(0,750)
plt.savefig("../Figures/stability_withP_L8_lin.png")

#log-scale
plt.xscale('log')
plt.xlim(1,15000)
plt.savefig("../Figures/stability_withP_L8_log.png")
#############################################