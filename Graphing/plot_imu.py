import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('imu_data.csv')
np_data = (df.to_numpy()).T
# print(np_data[0].shape)
t = np_data[0]
x = np_data[1]
y = np_data[2]
z = np_data[3]

# Create a figure and three subplots
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# Plot x, y, and z on each subplot
ax1.plot(t, x)
ax1.set_ylabel('x')
ax2.plot(t, y)
ax2.set_ylabel('y')
ax3.plot(t, z)
ax3.set_ylabel('z')
ax3.set_xlabel('time')

# Show the figure
plt.show()
