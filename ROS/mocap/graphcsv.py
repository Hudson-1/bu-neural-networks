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
df = pd.read_csv("~/Downloads/imu_2.csv") 

# Create empty lists to store the Euler angles and time values
pitch = []
yaw = []
roll = []
# time_values = []

# Loop through each row of the DataFrame
for index, row in df.iterrows():
    # Create a Quaternion object from the four components
    q = Quaternion(x=row['.x'], y=row['.y'], z=row['.z'], w=row['.w'])

    # Convert the quaternion to Euler angles
    X, Y, Z = quaternion_to_euler_angle_vectorized2(q.w, q.x, q.y, q.z)

    # Append the Euler angles and time value to the corresponding lists
    pitch.append(X)
    yaw.append(Y)
    roll.append(Z)
    # time_values.append(row['time'])

# # Convert time_values to a numpy array for better plotting
# time_values = np.array(time_values)


# print("Pitch:", pitch)
# print("Yaw:", yaw)
# print("Roll:", roll)
# print("Time Values:", time_values)

# Plot the Euler angles against time
print("Angles in Euler Now")
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(pitch)
plt.xlabel('Index')
plt.ylabel('Pitch (degrees)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(yaw)
plt.xlabel('Index')
plt.ylabel('Yaw (degrees)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(roll)
plt.xlabel('Index')
plt.ylabel('Roll (degrees)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Combine all Euler angles into a single plot
plt.figure(figsize=(10, 6))
plt.plot(pitch, label='Pitch')
plt.plot(yaw, label='Yaw')
plt.plot(roll, label='Roll')
plt.xlabel('Index')
plt.ylabel('Angle (degrees)')
plt.legend()
plt.title('Euler Angles')
plt.grid(True)
plt.show()