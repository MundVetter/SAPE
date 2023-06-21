import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use Seaborn styles for a more professional look
sns.set()

# data
data = """
sigma,TEST_psnr,TEST_min,TEST_max,TRAIN_psnr,TRAIN_min,TRAIN_max
0.5,26.51,20.86,34.37,25.99,22.3,29.11
1,27.37,22.01,35.65,26.58,22.68,30.36
3,27.25,17.12,36.57,33.83,25.2,39.61
5,26.44,18.31,34.48,37.93,32.41,41.71
10,24.98,19.28,33.18,42.23,36.73,47.45
20,22.78,17.28,31.76,43.49,39.63,50.53
40,19.66,14.16,26.51,42.15,36.26,46.77
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

# read data from string
from io import StringIO
df = pd.read_csv(StringIO(data))

# plot data
fig, ax = plt.subplots()

# plot TEST and TRAIN data with error bars
for mode, color in zip(['TEST', 'TRAIN'], ['blue', 'orange']):
    ax.plot(df['sigma'], df[f'{mode}_psnr'], label=mode, color=color, marker='o')
    ax.fill_between(df['sigma'], df[f'{mode}_min'], df[f'{mode}_max'], color=color, alpha=0.2)

# Set title and labels for axes
ax.set_xlabel('Mask $\sigma$', fontsize=14)
ax.set_ylabel('PSNR', fontsize=14)
ax.set_title('Laplace', fontsize=16)

# Add a grid
ax.grid(True)

# Add a legend
ax.legend(loc='lower right', fontsize=12)

# Show the plot with improved formatting
plt.tight_layout()
plt.show()