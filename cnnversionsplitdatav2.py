# Import libraries
import pandas as pd
import numpy as np
import torch

# Read the csv file
df = pd.read_csv('sine_wave.csv')

# Convert to numpy array and drop the first column (t)
np_data = (df.to_numpy())[:,1:]

# Get the number of rows
n_data = np_data.shape[0]

# Set the window length
win_len = 31

# Initialize empty lists for independent and dependent data
IND_DATA = []
DEP_DATA = []

# Loop over the rows and create windows of data
for i in range(n_data - win_len):
    # Get the current window
    cur_win = np_data[i:i+win_len]
    
    # Append the first 30 rows to IND_DATA
    IND_DATA.append(cur_win[:-1])
    
    # Append the last row to DEP_DATA
    DEP_DATA.append(cur_win[-1])

# Convert the lists to numpy arrays
IND_DATA = np.asarray(IND_DATA)
IND_DATA = IND_DATA.transpose(0, 2, 1)
DEP_DATA = np.asarray(DEP_DATA)
DEP = DEP_DATA.transpose(0, 1)
# Print the shapes of IND_DATA and DEP_DATA
print(IND_DATA.shape)
print(DEP_DATA.shape)

# Save IND_DATA and DEP_DATA as csv files with column names
# Create a list of column names for IND_DATA
ind_cols = []
for i in range(1, 31):
    ind_cols.extend([f'x{i}', f'y{i}', f'z{i}'])

# Create a list of column names for DEP_DATA
dep_cols = ['x31', 'y31', 'z31']

# Save IND_DATA as a csv file named ind_data_compare.csv with shape (9969, 30, 3) and column names
pd.DataFrame(IND_DATA.reshape(9969, -1)).to_csv('ind_data_cnn.csv', index=False, header=ind_cols)

# Save DEP_DATA as a csv file named dep_data_compare.csv with shape (9969, 3) and column names
pd.DataFrame(DEP_DATA).to_csv('dep_data_cnn.csv', index=False, header=dep_cols)

df = pd.load_csv('ind_data_compare.csv')
print(df.head())
################################
### output data
### (9969, 30, 3)
### (9969, 3)

