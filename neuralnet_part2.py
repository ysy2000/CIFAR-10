# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified for ECE4453 Final Project on 11/19/2021

"""
This is the main entry point for ECE4453 Final Project. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
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
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        lrate = 1e-3        # best value for rule of thumb

        # insize = 3072 (32*32*3)
        # ountsize = 2 (binary)
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 7, 2, padding = 1),
            nn.MaxPool2d(2, stride = 2),
            nn.Flatten(),
            nn.Linear(1792, 256),
            nn.Tanh(),
            nn.Linear(256, 32),
            nn.Tanh(),            
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, out_size)
        )        
       
        self.optimiser = optim.Adam(
            self.parameters(), 
            lrate, 
            weight_decay = 5e-3 
        )

        # raise NotImplementedError("You need to write init part!")


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std

        x = x.reshape(-1, 3, 32, 32)
        return self.model(x)

        # raise NotImplementedError("You need to write forward part!")
        # return torch.ones(x.shape[0], 1)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        self.optimiser.zero_grad()      # clear gradient buffer
        yhat = self.forward(x)          # inference
        loss_value = self.loss_fn(yhat, y)  # get loss

        loss_value.backward()
        self.optimiser.step()

        return loss_value.detach().cpu().numpy() 
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

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    lrate = 1e-3    # best value for rule of thumb

    # loss function = CrossEntropyLoss = 0.8736 / NLLLoss = 0.8012
    net = NeuralNet(lrate, nn.CrossEntropyLoss(), len(train_set[0]), 2)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)

    losses = []

    batch = int(len(train_set) / batch_size)
    print("batch: {}".format(batch))
    print("n_iter: {}".format(n_iter))      # len(losses) == n_iter.

    # Train
    for epoch in range(int(n_iter / batch_size)):
        running_loss = 0.0  #init
        for i in range(batch):
            # input: list of [inputs, labels]
            labels = train_labels[batch_size * i : batch_size * (i + 1)]
            inputs = train_set[batch_size * i : batch_size * (i + 1)]

            loss = net.step(inputs, labels)
            
            running_loss += loss

        losses.append(running_loss)
    yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), 1)    
    
    return losses, yhats, net
    # raise NotImplementedError("You need to write fit part!")
    # return [],[],None
