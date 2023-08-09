import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # import the library

torch.set_default_dtype(torch.float64)


df = pd.read_csv('Data\ind_data_compare.csv')
np_data = (df.to_numpy())
np_data = torch.from_numpy(np_data)

df = pd.read_csv('Data\dep_data_compare.csv')
np_dep_data = (df.to_numpy())
np_dep_data = torch.from_numpy(np_dep_data)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12,3)
        #self.fc2 = nn.Linear(7,3)
    
    def forward(self,x):
        relu = nn.ReLU()
        x = self.fc1(x)
        #x = relu(x)
        #x = self.fc2(x)
        return x

net = Net()
net.load_state_dict(torch.load("Data\sine_wave_model.pth"))

predicted_points = net(np_data).detach().numpy()
actual_points = np_dep_data.numpy()

mae = torch.mean(torch.square(torch.from_numpy(predicted_points) - torch.from_numpy(actual_points)))
print(f"Mean square error: {mae:.3f}")

# plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(actual_points[:,0], c="blue", linestyle="-", label="Actual") 
ax1.set_ylabel("Y")
ax1.set_title("Actual data")
ax1.legend() 

ax2.plot(predicted_points[:,0], c="red", linestyle="--", label="Predicted") 
ax2.set_xlabel("T")
ax2.set_title("Predicted data")
ax2.legend()

ax3.plot(actual_points[:,0], c="blue", linestyle="-", label="Actual")
ax3.plot(predicted_points[:,0], c="red", linestyle="--", label="Predicted") 
ax3.set_title("Predicted vs Actual")
ax3.legend()

plt.show() 
    
print("Finished Comparing")
