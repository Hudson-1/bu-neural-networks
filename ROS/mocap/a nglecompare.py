import pandas as pd
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np

def quaternion_angle_difference(q1, q2):
    dot_product = np.dot(q1, q2)
    print(np.abs(dot_product))
    angle = 2 * np.arccos(np.abs(dot_product))
    return np.degrees(angle)

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

# Calculate the angle differences between IMU and Mocap quaternions
angle_differences = [quaternion_angle_difference(q1, q2) for q1, q2 in zip(quaternions_imu, quaternions_mocap)]

# Create a single plot for all Quaternion components (x, y, z, w) for imu and mocap
# plt.figure(figsize=(10, 8))
# plt.plot(quat_imu_x, label='IMU Quaternion X')
# plt.plot(quat_imu_y, label='IMU Quaternion Y')
# plt.plot(quat_imu_z, label='IMU Quaternion Z')
# plt.plot(quat_imu_w, label='IMU Quaternion W')
# plt.plot(quat_mocap_x, label='Mocap Quaternion X')
# plt.plot(quat_mocap_y, label='Mocap Quaternion Y')
# plt.plot(quat_mocap_z, label='Mocap Quaternion Z')
# plt.plot(quat_mocap_w, label='Mocap Quaternion W')
# plt.xlabel('Index')
# plt.ylabel('Quaternion Component')
# plt.legend()
# plt.grid(True)
# plt.title('Comparison of IMU and Mocap Quaternion Components')
# plt.show()

# Plot the quaternion angle differences
plt.figure(figsize=(15, 3))
plt.plot(angle_differences, label='Quaternion Angle Difference')
plt.xlabel('Time')
plt.ylabel('Angle Difference (degrees)')
plt.legend()
plt.title('Angle Difference between IMU and Mocap Quaternions')
plt.grid(True)
plt.show()
