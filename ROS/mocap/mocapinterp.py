import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Read the imu data and skip the first row
df_imu = pd.read_csv("~/Downloads/imu_rosbagimu_topic.csv")

# Read the mocap data and skip the first row
df_mocap = pd.read_csv("~/Downloads/imu_rosbagmocap_topic.csv")

# Convert timestamps to datetime objects for both DataFrames
df_imu['time'] = pd.to_datetime(df_imu['time'])
df_mocap['time'] = pd.to_datetime(df_mocap['time'])

# Create interpolation functions for each of the quaternion channels
interp_x = interp1d(df_imu['time'].values, df_imu['.x'].values, kind='linear', fill_value='interpolate')
interp_y = interp1d(df_imu['time'].values, df_imu['.y'].values, kind='linear', fill_value='interpolate')
interp_z = interp1d(df_imu['time'].values, df_imu['.z'].values, kind='linear', fill_value='interpolate')
interp_w = interp1d(df_imu['time'].values, df_imu['.w'].values, kind='linear', fill_value='interpolate')

# Interpolate the IMU quaternion channels at mocap timestamps
df_mocap['imu_x'] = interp_x(df_mocap['time'].values)
df_mocap['imu_y'] = interp_y(df_mocap['time'].values)
df_mocap['imu_z'] = interp_z(df_mocap['time'].values)
df_mocap['imu_w'] = interp_w(df_mocap['time'].values)

# Save the synchronized data to a new CSV file
df_mocap.to_csv("~/Downloads/synchronized_data_interpolated.csv", index=False)
