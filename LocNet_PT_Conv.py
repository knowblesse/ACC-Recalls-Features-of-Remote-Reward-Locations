import scipy.io
import torch
import torch.utils.data
import torch.nn as nn
import os
from matplotlib import pyplot as plt
import numpy as np
import time

t = time.time() # Start Timer

class NeuralNet(nn.Module):
    """
    Pytorch Neural Net Model Class
    """
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50,25)
        self.fc4 = nn.Linear(25,2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        return out

class MyDataset(torch.utils.data.Dataset):
    """
    Custom-written Dataset Class for DataLoader.
    Inherit torch.utils.data.Dataset Abstract Class
    Input Data must be in a list form. [X, y]
    """
    def __init__(self, data):
        self.data = data
    #override
    def __len__(self):
        return self.data[0].shape[0]
    #override
    def __getitem__(self, idx):
        return [self.data[0][idx, :],
                self.data[1][idx, :]]  # [spike data, XY Loc]

class NNModel:
    """
    Neural Network Model implementing pytorch
    """
    def __init__(self, batch_size=200, num_epochs=10):
        self.model = None
        self.numInput = None
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def Train(self, X_train, Y_train):

        if (self.model is None):
            self.numInput = X_train.shape[1]
            self.CreateModel()

        # Generate Torch.utils.data.Dataset
        train_dataset = MyDataset([torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()])

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        # Train the model
        total_step = len(train_loader)
        for epoch in range(self.num_epochs):
            for i, (spike, XY) in enumerate(train_loader):
                # Move tensors to the configured device
                spike = spike.reshape(-1, self.numInput).to(device)
                XY = XY.to(device)

                # Forward pass
                outputs = self.model(spike)
                loss = self.criterion(outputs, XY)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 50 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))

    def CreateModel(self):

        # Create Model
        self.model = NeuralNet(self.numInput).to(device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = learning_rate, momentum = 0.9, weight_decay=0e-6, nesterov=True)


##############################################################
######################### Main Code ##########################
##############################################################

# Hyper-parameters
learning_rate = 0.0001

# Loading data from data.mat file
if not(os.path.isfile(r'.\data.mat')):
    raise Exception('No File Error','data.mat file must be in the same path')
data = scipy.io.loadmat(r'.\datad.mat')
train = data['train'].T
valid = data['test'].T
loc_t = data['train_loc'].T
loc_v = data['test_loc'].T


# Device configuration
if torch.cuda.is_available():
    print('CUDA is available')
    print('Running in CUDA....')
    device = torch.device('cuda')
else:
    print('CUDA is unavailable')
    print('Running in CPU....')
    device = torch.device('cpu')

# Build and Train the Model with training dataset
nnModel = NNModel()
nnModel.Train(train, loc_t)

# Create Testing data for validation
test_dataset = MyDataset((torch.from_numpy(valid).float(), torch.from_numpy(loc_v).float()))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)

loc_v_pred = np.zeros([valid.shape[0],2]) # Array for storing predicted y value from the trained model

# Run Forward Prop. with testing data
with torch.no_grad(): # with-out tuning parameters
    for i, (spike, a) in enumerate(test_loader):
        # Move tensors to the configured device
        spike = spike.reshape(-1, spike.shape[1]).to(device)
        # Forward pass
        loc_v_pred[i,:] = nnModel.model(spike)

# # Plot Trail data
# numTrail = 20
# #for i in range(loc_v.shape[0]-numTrail):
# for i in range(1000):
#     plt.plot(loc_v[i:i+numTrail, 0], loc_v[i:i+numTrail, 1],'r')
#     plt.plot(loc_v_pred[i:i+numTrail, 0], loc_v_pred[i:i+numTrail, 1],'b')
#     plt.draw()
#     plt.xlim([-70,70])
#     plt.ylim([-70, 70])
#     plt.title(str(i))
#     plt.pause(0.000000001)
#     plt.clf()
print(time.time() -t) # stop timer and print elapsed time