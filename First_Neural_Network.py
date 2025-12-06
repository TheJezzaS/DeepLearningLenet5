import torch
#Defining the Sigmoid function
def activation(x):
    return 1/(1+torch.exp(-x))
#Generate a random (normal dist) tensor of 1 row and 5 columns
features = torch.randn((1, 5))
#Generate a tensor of the same form as a previous tensor
weights = torch.randn_like(features)
#Generate a Bias term
bias = torch.randn((1,1))
#Output of the simple neural network
y1 = activation(torch.sum(features*weights)+bias)
print(y1)
#We prefer to do this with matrix multiplication
#method1 torch.mm with reshape
y2 = activation(torch.mm(features,weights.reshape(5,1))+bias)
print(y2)
#method2 torch.mm with view
y3 = activation(torch.mm(features,weights.view(5,1))+bias)
print(y3)