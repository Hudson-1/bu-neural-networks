import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load raw data
df_raw = pd.read_csv('Arduino/A_Data/gryo_def_raw.csv')
np_data_raw = df_raw.to_numpy().T
t_raw = np_data_raw[0]
x_raw = np_data_raw[1]
y_raw = np_data_raw[2]
z_raw = np_data_raw[3]

# Load filtered data
df_filtered = pd.read_csv('Arduino/A_Data/gryo_def_filtered.csv')
np_data_filtered = df_filtered.to_numpy().T
tf = np_data_filtered[0]
xf = np_data_filtered[1]
yf = np_data_filtered[2]
zf = np_data_filtered[3]


# Create a figure and six subplots (3 for raw vs. filtered, 3 for the difference)
plt.style.use('dark_background')
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True)

# Plot raw vs. filtered data on each subplot
ax1.plot(t_raw, x_raw, label='Raw')
ax1.plot(tf, xf, label='Filtered')
ax1.set_ylabel('x')
ax1.legend()

ax2.plot(t_raw, y_raw, label='Raw')
ax2.plot(tf, yf, label='Filtered')
ax2.set_ylabel('y')
ax2.legend()

ax3.plot(t_raw, z_raw, label='Raw')
ax3.plot(tf, zf, label='Filtered')
ax3.set_ylabel('z')
ax3.set_xlabel('time')
ax3.legend()

# Calculate the difference between raw and filtered data
x_diff = x_raw - xf
y_diff = y_raw - yf
z_diff = z_raw - zf

# Plot the difference data
ax4.plot(t_raw, x_diff, color='orange')
ax4.set_ylabel('Difference (x)')

ax5.plot(t_raw, y_diff, color='orange')
ax5.set_ylabel('Difference (y)')

ax6.plot(t_raw, z_diff, color='orange')
ax6.set_ylabel('Difference (z)')
ax6.set_xlabel('time')

# Show the figure
plt.tight_layout()
plt.show()
