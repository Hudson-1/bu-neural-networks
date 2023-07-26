import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # import the library

torch.set_default_dtype(torch.float64)

df = pd.read_csv('ind_data.csv')
np_data = (df.to_numpy())
np_data = torch.from_numpy(np_data)

df = pd.read_csv('dep_data.csv')
np_dep_data = (df.to_numpy())
np_dep_data = torch.from_numpy(np_dep_data)

batch_size = 2000

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
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum= 0.9)
criterion = nn.MSELoss()

loss_list = [] # create an empty list to store the loss values

for epoch in range(1500):
    running_loss = 0.0
    for i in range(np_data.shape[0]//batch_size):
        inputs = np_data[i:i+batch_size]
        labels = np_dep_data[i:i+batch_size]

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[{epoch + 1} loss: {running_loss:.3f}")
    loss_list.append(running_loss) 


plt.style.use('dark_background')
plt.plot(loss_list) 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.title("Loss function over epochs")
plt.show()

predicted_points = net(np_data).detach().numpy()
actual_points = np_dep_data.numpy()

plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(actual_points[:,0], c="blue", linestyle="-", label="Actual") 
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("True data")
ax1.legend() 

ax2.plot(predicted_points[:,0], c="red", linestyle="--", label="Predicted") 
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Predicted data")
ax2.legend()

ax3.plot(actual_points[:,0], c="blue", linestyle="-", label="Actual")
ax3.plot(predicted_points[:,0], c="red", linestyle="--", label="Predicted") 
ax3.set_xlabel("X") 
ax3.set_ylabel("Y") 
ax3.set_title("Predicted vs Actual points")
ax3.legend()

plt.show() 
    
torch.save(net.state_dict(), "sine_wave_model.pth")
print("Finished Training")
