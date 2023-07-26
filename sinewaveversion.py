import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create three different sine wave functions with some noise
x = np.linspace(0, 20*np.pi, 10000, endpoint=False) 
y1 = 30 * (np.sin(x)) + np.random.normal(0, 1, size=10000) * 0.1
y2 = 30 * (np.sin(2*x)) + np.random.normal(0, 1, size=10000) * 0.1
y3 = 30 * (np.sin(3*x)) + np.random.normal(0, 1, size=10000) * 0.1

#save data
df = pd.DataFrame({"t": x, "x": y1, "y": y2, "z": y3}) 
df.to_csv("sine_wave.csv", index=False) 

#graph
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(x, y1)
ax1.set_ylabel("x")
ax2.plot(x, y2)
ax2.set_ylabel("y")
ax3.plot(x, y3)
ax3.set_xlabel("t")
ax3.set_ylabel("z")
plt.suptitle("Three sine waves with noise")
# plt.savefig("sine_wave.png")
plt.show() 
