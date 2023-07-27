import pandas as pd
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("~/Downloads/synchronized_data_2.csv") 

imu_norms = []
mocap_norms = []
norm_diff = []

for index, row in df.iterrows():
    q_imu = Quaternion(x=row['.x'],
                       y=row['.y'],
                       z=row['.z'],
                       w=row['.w'])

    imu_norm = q_imu.norm
    imu_norms.append(imu_norm)

    q_mocap = Quaternion(x=row['.pose.orientation.x'],
                         y=row['.pose.orientation.y'],
                         z=row['.pose.orientation.z'],
                         w=row['.pose.orientation.w'])

    mocap_norm = q_mocap.norm
    mocap_norms.append(mocap_norm)

    norm_diff.append((mocap_norm - imu_norm))

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(imu_norms, label='IMU Norm')
plt.xlabel('Index')
plt.ylabel('Quaternion Norm')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(mocap_norms, label='Mocap Norm')
plt.xlabel('Index')
plt.ylabel('Quaternion Norm')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(norm_diff, label='Norm Difference')
plt.xlabel('Index')
plt.ylabel('Difference in Norm')
plt.legend()
plt.grid(True)
plt.show()
