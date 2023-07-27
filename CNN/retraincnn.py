import debugpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init

torch.set_default_dtype(torch.float64)

batch_size = 150

df = pd.read_csv('CNN/CNN_Data/ind_data_cnn.csv')
np_data = (df.to_numpy())
np_data = torch.from_numpy(np_data)
# print(np_data.size())
np_data = torch.reshape(np_data, (-1, 30, 3)).transpose(2,1)
print(np_data.size())
# print("np")
# print(np_data)
# ### artificial data ####
# ind = torch.rand(10000, 3, 30)
# dep = torch.rand(1000, 3)

df = pd.read_csv('CNN/CNN_Data/dep_data_cnn.csv')
np_dep_data = (df.to_numpy())
np_dep_data = torch.from_numpy(np_dep_data)
# print(np_dep_data.size())
# print("dep")
# print(np_dep_data)


class Net(nn.Module):
    # xavier initialization
    def init_weights(self, weights):
        torch.nn.init.xavier_uniform_(weights.weight)
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, padding=0, dilation=1)
        # self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding=0, dilation=1)
        # self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding=0, dilation=1)
        # self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, padding=0, dilation=1)
        self.pool = torch.nn.MaxPool1d(1,stride=1)
        self.fc1 = nn.Linear(32 * 30, 3)
        self.fc2 = nn.Linear(480, 3)
        self.relu = nn.ReLU()
        


    def forward(self, x):   
        x = self.conv1(x)
        x=  self.relu(x)
        # x = torch.nn.LayerNorm((batch_size,32,30))(x)
        # x = F.relu(self.conv2(x))
        # x = torch.nn.LayerNorm((batch_size,64,30))(x)
        # x = F.relu(self.conv3(x))
        # x = torch.nn.LayerNorm((batch_size,128,30))(x)
        # x = F.relu(self.conv4(x))
        # x = torch.nn.LayerNorm((batch_size,256,30))(x)
        # x = self.conv5(x)
        x = self.pool(x)
        x = torch.nn.Flatten()(x)
        x = self.fc1(x)
        # x=  self.relu(x)
        # x = self.fc2(x)
        # print("output")
        # print(x)
        return x
    
    
net = Net()

net.load_state_dict(torch.load("CNN/CNN_Data/sine_wave_model_cnn.pth"))


optimizer = optim.Adam(net.parameters(), lr=1e-3)#, momentum= 0.9)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loss_list = [] 

# net = Net()
net.train()

loss_list = [] 
# print("net",list(net.parameters()))
for epoch in range(10000):
    running_loss = 0.0
    for i in range(np_data.shape[0]//batch_size):

        optimizer.zero_grad()
        inputs = np_data[i:i+batch_size]
        # inputs = ind[i:i+batch_size]
        labels = np_dep_data[i:i+batch_size]
        # labels = dep[i:i+batch_size]
        # print(inputs, labels) 
        outputs = net(inputs)
        # print(inputs[0], labels[0]) 
        
        loss = criterion(outputs, labels)
        loss.backward() # compute gradients

        # for name, param in net.named_parameters():
        #     print(name, param.grad)

        optimizer.step() # update parameters
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
    
torch.save(net.state_dict(), "CNN/CNN_Data/sine_wave_model_cnn.pth")
print("Finished Training")