import pandas as pd
import numpy as np

# Read the imu data and skip the first row
df_imu = pd.read_csv("~/Downloads/imu_rosbagimu_topic.csv")

# Read the mocap data and skip the first row
df_mocap = pd.read_csv("~/Downloads/imu_rosbagmocap_topic.csv")

# Convert timestamps to datetime objects for both DataFrames
df_imu['time'] = pd.to_datetime(df_imu['time'])
df_mocap['time'] = pd.to_datetime(df_mocap['time'])

# Sort both DataFrames based on time to ensure accurate synchronization
df_imu.sort_values('time', inplace=True)
df_mocap.sort_values('time', inplace=True)

# Perform the time-based synchronization using merge_asof with direction='forward'
df_synced = pd.merge_asof(df_imu, df_mocap, on='time', direction='backward', suffixes=('_mocap', '_imu'))

# Rename the quaternion columns for clarity
df_synced.rename(columns={
    '.pose.orientation.x_mocap': 'mocap_x',
    '.pose.orientation.y_mocap': 'mocap_y',
    '.pose.orientation.z_mocap': 'mocap_z',
    '.pose.orientation.w_mocap': 'mocap_w',
    '.x_imu': 'imu_x',
    '.y_imu': 'imu_y',
    '.z_imu': 'imu_z',
    '.w_imu': 'imu_w',
}, inplace=True)

# Save the synchronized data to a new CSV file
df_synced.to_csv("~/Downloads/synchronized_data_2.csv", index=False)