import pandas as pd
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np

def quaternion_to_euler_angle_vectorized2(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z

# Read the CSV file and skip the first row
df = pd.read_csv("~/Downloads/synchronized_data_2.csv") 

# Create empty lists to store the Euler angles for imu and mocap
pitch_imu = []
yaw_imu = []
roll_imu = []

pitch_mocap = []
yaw_mocap = []
roll_mocap = []

# Loop through each row of the DataFrame
for index, row in df.iterrows():
    # Create a Quaternion object from the imu quaternion components
    q_imu = Quaternion(x=row['.x'],
                       y=row['.y'],
                       z=row['.z'],
                       w=row['.w'])

    # Convert the imu quaternion to Euler angles
    X_imu, Y_imu, Z_imu = quaternion_to_euler_angle_vectorized2(q_imu.w, q_imu.x, q_imu.y, q_imu.z)

    # Append the imu Euler angles to the corresponding lists
    pitch_imu.append(X_imu)
    yaw_imu.append(Y_imu)
    roll_imu.append(Z_imu)

    # Create a Quaternion object from the mocap quaternion components
    q_mocap = Quaternion(x=row['.pose.orientation.x'],
                         y=row['.pose.orientation.y'],
                         z=row['.pose.orientation.z'],
                         w=row['.pose.orientation.w'])

    # Convert the mocap quaternion to Euler angles
    X_mocap, Y_mocap, Z_mocap = quaternion_to_euler_angle_vectorized2(q_mocap.w, q_mocap.x, q_mocap.y, q_mocap.z)

    # Append the mocap Euler angles to the corresponding lists
    pitch_mocap.append(X_mocap)
    yaw_mocap.append(Y_mocap)
    roll_mocap.append(Z_mocap)

# Calculate the absolute difference between IMU and Mocap angles for each Euler angle
diff_pitch = np.abs(np.array(pitch_imu) - np.array(pitch_mocap))
diff_yaw = np.abs(np.array(yaw_imu) - np.array(yaw_mocap))
diff_roll = np.abs(np.array(roll_imu) - np.array(roll_mocap))

# Create separate plots for each Euler angle for imu
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(pitch_imu, label='IMU Pitch')
plt.plot(pitch_mocap, label='Mocap Pitch')
plt.xlabel('Index')
plt.ylabel('Pitch (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(yaw_imu, label='IMU Yaw')
plt.plot(yaw_mocap, label='Mocap Yaw')
plt.xlabel('Index')
plt.ylabel('Yaw (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(roll_imu, label='IMU Roll')
plt.plot(roll_mocap, label='Mocap Roll')
plt.xlabel('Index')
plt.ylabel('Roll (degrees)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the absolute difference between IMU and Mocap angles for each Euler angle
plt.figure(figsize=(10, 6))
plt.plot(diff_pitch, label='Pitch Difference')
plt.plot(diff_yaw, label='Yaw Difference')
plt.plot(diff_roll, label='Roll Difference')
plt.xlabel('Index')
plt.ylabel('Angle Difference (degrees)')
plt.legend()
plt.title('Absolute Difference between IMU and Mocap Euler Angles')
plt.grid(True)
plt.show()
