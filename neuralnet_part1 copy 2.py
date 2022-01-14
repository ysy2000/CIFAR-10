# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified the code for ECE4453 Final Project on 11/19/2021

"""
This is the main entry point for your project. You should only modify code
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        lrate = 0.01

        ## self.model = nn.Sequential(
        # OrderedDict([(nn.Linear(in_size, 32)), 
        # (nn.ReLu()), (nn.ReLu(), nn.Linear(32, out_size))]))
        
        self.model = nn.Sequential(
            nn.Linear(in_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, out_size)
        )

        # hidden layer = 32

        self.optimiser = optim.SGD(
            self.parameters(), 
            lrate, 
            weight_decay = 1e-2 ## L2 norm
        )

        ## self.optimiser = optim.SGD(self.parameters(), lr=0.01, weight_decay = 1e-4)
        # raise NotImplementedError("You need to write init part!")

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """
        raise NotImplementedError("You need to write set_param part!")
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        raise NotImplementedError("You need to write get_param part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write forward part!")
        ## The forward() function should do a forward pass through your network. This means it should explicitly evaluate FW(x).
        ## This can be done by simply calling your nn.Sequential() object defined in __init__()

        return self.model(x)
        # return torch.ones(x.shape[0], 1)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        self.optimiser.zero_grad()      # clear gradient buffer
        yhat = self.forward(x)
        loss_value = self.loss_fn(yhat, y)
        # print(loss)
        loss_value.backward()
        self.optimiser.step()
    
        # loss_value.item()
        # loss_value.detach().cpu().numpy() 

        return loss_value.detach().cpu().numpy()
         ## returned loss: detached from the computational graph 
         ## after one execution of the step() function and proper garbage 
         ## collection can take place.

        # raise NotImplementedError("You need to write step part!")
        # return 0.0


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    
    # Data Standardization: 
    # Convergence speed and accuracies can be improved greatly 
    # by simply centralizing your input data 
    # by subtracting the sample mean and dividing 
    # by the sample standard deviation.
    #  
    # More precisely, 
    # you can alter your data matrix X by simply doing X:=(X−μ)/σ.
    
    print("Train_set: {}".format(train_set))
    print("Dev_set: {}".format(dev_set))
    lrate = 0.01
    
    train_mean = torch.mean(train_set)
    train_std = torch.std(train_set)
    train_set = (train_set - train_mean) / train_std

    dev_mean = torch.mean(dev_set)
    dev_std = torch.std(dev_set)
    dev_set = (dev_set - dev_mean) / dev_std

    # print(len(train_set[0]))
    net = NeuralNet(lrate, nn.CrossEntropyLoss(), len(train_set[0]), 2)
    # loss function = CrossEntropyLoss
    losses = []

    batch = int(len(train_set) / batch_size)
    print("batch: {}".format(batch))
    # Train
    for epoch in range(n_iter):
        running_loss = 0.0
        count = 0
        for i in range(batch-1):

            # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            # Tutorial was followed slightly

            # get the inputs; data is a list of [inputs, labels]
            labels = train_labels[batch_size * i : batch_size * (i + 1)]
            inputs = train_set[batch_size * i : batch_size * (i + 1)]

            loss = net.step(inputs, labels)
            
            running_loss += loss

        losses.append(running_loss)
    yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), 1)    # yhats = NONE
    
    return losses, yhats, net
    # raise NotImplementedError("You need to write fit part!")
    # return [],[],None
