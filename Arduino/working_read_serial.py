import serial
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import time
s_t = time.time()
# Open the serial port
ser = serial.Serial("COM6", 115200)
# Create deques to store the data
n = int(1e4)
data_x = deque(maxlen=n)
data_y = deque(maxlen=n)
data_z = deque(maxlen=n)
t = deque(maxlen=n)

imu_data = [0,0,0]
instring = ""

# Define a function that records data from the serial port and returns a list of three values
def update_imu():
    global instring
    global imu_data
    global data_x, data_y, data_z, t
    # Read one line from the serial port
    in_char = ser.read()
    if in_char != b'\n' and in_char != b'\r':
        instring += chr(ord(in_char))
    elif in_char == b"\n":
        if len(instring) != 0:
            # print(instring)
            # Split the line by commas and convert to floats
            chans = instring.split(',')
            for i, chan in enumerate(chans):
                imu_data[i] = float(chan)
            t.append(time.time() - s_t)
            data_x.append(imu_data[0])
            data_y.append(imu_data[1])
            data_z.append(imu_data[2])
            instring = ""

def save_data():
    df = pd.DataFrame({"t" : np.asarray(t), "x": np.asarray(data_x), "y": np.asarray(data_y), "z": np.asarray(data_z)})
    df.to_csv("Data\imu_data.csv", index=False)


while True:
    print("\r{:^11.3f}%".format(len(data_x)*100 / n), end="")
    update_imu()
    if len(data_x) == n:
        save_data()
        break

print("\ncomplete")