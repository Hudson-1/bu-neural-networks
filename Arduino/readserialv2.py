import serial
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import time

s_t = time.time()
# Open the serial port
ser = serial.Serial("COM7", 115200)
# Create deques to store the data
n = int(1e4)
data_accel_f_x = deque(maxlen=n)
data_accel_f_y = deque(maxlen=n)
data_accel_f_z = deque(maxlen=n)
data_gyro_f_x = deque(maxlen=n)
data_gyro_f_y = deque(maxlen=n)
data_gyro_f_z = deque(maxlen=n)

data_accel_r_x = deque(maxlen=n)
data_accel_r_y = deque(maxlen=n)
data_accel_r_z = deque(maxlen=n)
data_gyro_r_x = deque(maxlen=n)
data_gyro_r_y = deque(maxlen=n)
data_gyro_r_z = deque(maxlen=n)
t = deque(maxlen=n)

imu_data = [0,0,0,0,0,0,0,0,0,0,0,0]
instring = ""

# Define a function that records data from the serial port and returns a list of three values
def update_imu():
    global instring
    global imu_data
    global data_accel_f_x, data_accel_f_y, data_accel_f_z, data_accel_r_x, data_gyro_f_y, data_gyro_f_z,data_accel_r_x,data_accel_r_y,data_accel_r_z,data_gyro_r_x,data_gyro_r_y,data_gyro_r_z, t
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
            data_accel_f_x.append(imu_data[0])
            data_accel_f_y.append(imu_data[1])
            data_accel_f_z.append(imu_data[2])

            data_gyro_f_x.append(imu_data[3])
            data_gyro_f_y.append(imu_data[4])
            data_gyro_f_z.append(imu_data[5])

            data_accel_r_x.append(imu_data[6])
            data_accel_r_y.append(imu_data[7])
            data_accel_r_z.append(imu_data[8])

            data_gyro_r_x.append(imu_data[9])
            data_gyro_r_y.append(imu_data[10])
            data_gyro_r_z.append(imu_data[11])
            instring = ""

def save_data():
    df = pd.DataFrame({"t" : np.asarray(t), "x": np.asarray(data_accel_f_x), "y": np.asarray(data_accel_f_y), "z": np.asarray(data_accel_f_z)})
    df.to_csv("accel_def_filtered.csv", index=False)

    df = pd.DataFrame({"t" : np.asarray(t), "x": np.asarray(data_gyro_f_x), "y": np.asarray(data_gyro_f_y), "z": np.asarray(data_gyro_f_y)})
    df.to_csv("gryo_def_filtered.csv", index=False)

    df = pd.DataFrame({"t" : np.asarray(t), "x": np.asarray(data_accel_r_x), "y": np.asarray(data_accel_r_y), "z": np.asarray(data_accel_r_z)})
    df.to_csv("accel_def_raw.csv", index=False)

    df = pd.DataFrame({"t" : np.asarray(t), "x": np.asarray(data_accel_r_x), "y": np.asarray(data_accel_r_y), "z": np.asarray(data_accel_r_z)})
    df.to_csv("gryo_def_raw.csv", index=False)

while True:
    print("\r{:^11.3f}%".format(len(data_accel_f_x)*100 / n), end="")
    update_imu()
    if len(data_accel_f_x) == n:
        save_data()
        break

print("\ncomplete")