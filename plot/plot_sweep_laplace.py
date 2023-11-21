import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Use Seaborn styles for a more professional look
sns.set_style("whitegrid")

# read data from string
from io import StringIO


# data
# data = """
# sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
# 0.5,26.51,20.86,34.37,25.99,22.3,29.11
# 1,27.37,22.01,35.65,26.58,22.68,30.36
# 3,27.25,17.12,36.57,33.83,25.2,39.61
# 5,26.44,18.31,34.48,37.93,32.41,41.71
# 10,24.98,19.28,33.18,42.23,36.73,47.45
# 20,22.78,17.28,31.76,43.49,39.63,50.53
# 40,19.66,14.16,26.51,42.15,36.26,46.77
# """

# data = """
# sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
# 1,27.81,22,34,27.95,23.59,31.09
# 5,28.34,21.99,39.73,31.45,27.61,35.47
# 10,28.28,21.61,37.49,34.45,29.53,38.54
# 20,27.75,20.75,38.93,33.47,31.01,35.52
# 40,27.04,17.88,35.96,33.02,29.89,36.58
# 80,27.13,20.37,35.77,31.85,29.58,36.46
# """

data = """
sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
1,25.89,20.91,34.86,40.53,37.34,43.51
5,25.82,20.78,35.04,43.39,36.92,46.81
10,25.74,20.56,34.95,44.77,38.76,48.16
20,25.63,20.30,35.07,44.50,39.51,46.78
40,25.46,20.19,35.05,42.9,39.1,47.14
80,25.21,20.24,34.42,41.71,38.65,44.84
"""

# data= """
# sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
# 0.5,25.14,20.24,34.07,36.20,32.57,40.03
# 1,25.52,20.28,34.74,35.21,30.34,39.08
# 3,25.25,20.59,32.99,42.24,40.19,45.7
# 5,25.59,20.56,34.51,47.62,44.98,49.49
# 10,25.22,20.16,34.3,50.98,48.00,53.31
# 20,24.1,19.28,33.29,50.84,48.79,53.26
# 40,19.79,15.44,26.76,48.48,46.92,51.02
# """

# data = """sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
# 1,25.04,19.98,34.09,29.71,24.68,37.49
# 0.1,25.52,20.28,34.74,35.21,30.34,39.08
# 0.01,25.63,20.30,35.07,44.50,39.51,46.78
# 0.001,25.43,20.37,35.06,50.56,46.92,54.07
# """

# data = """sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
# 1,25.37,19.91,32.62,22.98,19,28.08
# 0.1,27.37,22.01,35.65,26.58,22.68,30.36
# 0.01,27.75,20.75,38.93,33.47,31.01,35.52
# 0.001,27.44,20.46,41.49,42.95,38.05,46.97
# """



# read data from string
df = pd.read_csv(StringIO(data))

# Define colors as RGB tuples
keynote_blue = (0/255, 118/255, 186/255)
keynote_orange = (254/255, 174/255, 0/255)

# plot data
fig, ax = plt.subplots()

# plot TEST and TRAIN data with error bars
for mode, color in zip(['TEST', 'TRAIN'], [keynote_blue, keynote_orange]):
    ax.plot(df['sigma'], df[f'{mode}_psnr'], label=mode, color=color, marker='o')
    ax.fill_between(df['sigma'], df[f'{mode}_min'], df[f'{mode}_max'], color=color, alpha=0.2)

# Set title and labels for axes for \alpha_{reg}
# alpha_{reg}
# ax.set_xlabel("$\\alpha_{reg}$", fontsize=14)
ax.set_xlabel("$\sigma$", fontsize=14)
ax.set_ylabel('PSNR', fontsize=14)

# Add a grid
ax.grid(True)
# plt.xscale('log')
# Add a legend
ax.legend(loc='upper right', fontsize=12)

# Show the plot with improved formatting
plt.gcf().set_size_inches(4, 3)
plt.tight_layout()
# make the plot small
# save the plot as pdf
plt.savefig('sweep.pdf')
# plt.show()