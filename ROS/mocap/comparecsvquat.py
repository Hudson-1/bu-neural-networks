import pandas as pd
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file and skip the first row
df = pd.read_csv("~/Downloads/synchronized_data_2.csv") 

# Create empty lists to store the Quaternion components for imu and mocap
quat_imu_x = []
quat_imu_y = []
quat_imu_z = []
quat_imu_w = []

quat_mocap_x = []
quat_mocap_y = []
quat_mocap_z = []
quat_mocap_w = []

# Loop through each row of the DataFrame
for index, row in df.iterrows():
    # Store imu quaternion components
    quat_imu_x.append(row['.x'])
    quat_imu_y.append(row['.y'])
    quat_imu_z.append(row['.z'])
    quat_imu_w.append(row['.w'])

    # Store mocap quaternion components
    quat_mocap_x.append(row['.pose.orientation.x'])
    quat_mocap_y.append(row['.pose.orientation.y'])
    quat_mocap_z.append(row['.pose.orientation.z'])
    quat_mocap_w.append(row['.pose.orientation.w'])

# Convert the lists of quaternion components to Quaternion objects
quaternions_imu = [Quaternion(x, y, z, w) for x, y, z, w in zip(quat_imu_x, quat_imu_y, quat_imu_z, quat_imu_w)]
quaternions_mocap = [Quaternion(x, y, z, w) for x, y, z, w in zip(quat_mocap_x, quat_mocap_y, quat_mocap_z, quat_mocap_w)]

# Calculate the absolute difference between IMU and Mocap quaternions for each component
diff_x = np.abs(np.array(quat_imu_x) - np.array(quat_mocap_x))
diff_y = np.abs(np.array(quat_imu_y) - np.array(quat_mocap_y))
diff_z = np.abs(np.array(quat_imu_z) - np.array(quat_mocap_z))
diff_w = np.abs(np.array(quat_imu_w) - np.array(quat_mocap_w))

# Create separate plots for each Quaternion component for imu
# plt.style.use('dark_background')
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(quat_imu_x, label='IMU Quaternion X')
plt.plot(quat_mocap_x, label='Mocap Quaternion X')
plt.xlabel('Index')
plt.ylabel('Quaternion Component')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(quat_imu_y, label='IMU Quaternion Y')
plt.plot(quat_mocap_y, label='Mocap Quaternion Y')
plt.xlabel('Index')
plt.ylabel('Quaternion Component')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(quat_imu_z, label='IMU Quaternion Z')
plt.plot(quat_mocap_z, label='Mocap Quaternion Z')
plt.xlabel('Index')
plt.ylabel('Quaternion Component')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(quat_imu_w, label='IMU Quaternion W')
plt.plot(quat_mocap_w, label='Mocap Quaternion W')
plt.xlabel('Index')
plt.ylabel('Quaternion Component')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the absolute difference between IMU and Mocap quaternions for each component
plt.figure(figsize=(15, 3))
plt.plot(diff_x, label='Quaternion X Difference')
plt.plot(diff_y, label='Quaternion Y Difference')
plt.plot(diff_z, label='Quaternion Z Difference')
plt.plot(diff_w, label='Quaternion W Difference')
plt.xlabel('Time')
plt.ylabel('Component Difference')
plt.legend()
plt.title('Absolute Difference between IMU and Mocap Quaternion Components')
plt.grid(True)
plt.show()
