import pandas as pd
import numpy as np

# Read the imu data and skip the first row
df_imu = pd.read_csv("~/Downloads/imu_rosbagimu_topic.csv")

# Read the mocap data and skip the first row
df_mocap = pd.read_csv("~/Downloads/imu_rosbagmocap_topic.csv")

# Convert timestamps to datetime objects for both DataFrames
df_imu['time'] = pd.to_datetime(df_imu['time'])
df_mocap['time'] = pd.to_datetime(df_mocap['time'])


print(df_imu['time'])
print(df_mocap['time'])
# # Sort both DataFrames based on time to ensure accurate synchronization
# df_imu.sort_values('time', inplace=True)
# df_mocap.sort_values('time', inplace=True)

# def find_nearest_imu_row(mocap_time):
#     time_diff = abs(df_imu['time'] - mocap_time)
#     nearest_idx = time_diff.idxmin()
#     return df_imu.loc[nearest_idx, ['.x', '.y', '.z', '.w']].values

# # Find the nearest IMU values for each Mocap point
# df_mocap[['imu_x', 'imu_y', 'imu_z', 'imu_w']] = np.vstack(df_mocap['time'].apply(find_nearest_imu_row))

# # Save the synchronized data to a new CSV file
# df_mocap.to_csv("~/Downloads/synchronized_data_3.csv", index=False)
