import pandas as pd
import numpy as np

df = pd.read_csv('Data\sine_wave_compare.csv')
np_data = (df.to_numpy())[:,1:]
n_data = np_data.shape[0]
win_len = 5

IND_DATA = []
DEP_DATA = []

for i in range(n_data - win_len):
    cur_win = np_data[i:i+win_len]
    IND_DATA.append(cur_win[:-1])
    DEP_DATA.append(cur_win[-1])
    # print(cur_win.shape)
IND_DATA = np.asarray(IND_DATA)
DEP_DATA = np.asarray(DEP_DATA)
print(IND_DATA.shape)
print(DEP_DATA.shape)

#save data as csv
# Create a list of column names for IND_DATA
ind_cols = []
for i in range(1, 5):
    ind_cols.extend([f'x{i}', f'y{i}', f'z{i}'])

# Create a list of column names for DEP_DATA
dep_cols = ['x5', 'y5', 'z5']

# Save IND_DATA as a csv file named ind_data.csv with shape (9995, 4, 3) and column names
pd.DataFrame(IND_DATA.reshape(9995, -1)).to_csv('Data\ind_data_compare.csv', index=False, header=ind_cols)

# Save DEP_DATA as a csv file named dep_data.csv with shape (9995, 3) and column names
pd.DataFrame(DEP_DATA).to_csv('Data\dep_data_compare.csv', index=False, header=dep_cols)

################################
### output data
### (9995, 4, 3)
### (9995, 3)