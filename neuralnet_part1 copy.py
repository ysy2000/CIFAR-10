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
from project import compute_accuracies


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss = constant
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        num_layer = 2
        self.lrate = lrate      
        lrate = 0.01        # param_lrate = 0.01
        self.in_size = in_size          # 3072    # 32 * 32 * 3
        self.out_size = out_size        # param_out_size = 2

        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.conv1 = nn.Conv2d(3,6,5)   # x-z
        self.conv2 = nn.Conv2d(6,16,5)  # z-y
        ## you need to edit above number
        ## just two layer

        self.NeuralNet = NeuralNet
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.model = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, out_size))
        

        # raise NotImplementedError("You need to write init part!")

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """

        # i dont know
        self.params = params
        raise NotImplementedError("You need to write set_parameters part!")
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        raise NotImplementedError("You need to write get_params part!")

## x is data, y is labels
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        z = F.relu(self.conv1(x))
        y = F.relu(self.conv2(z))
        return y

    def step(self, x, labels):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        y_hat = self.forward(x)
        """
        predictions = self.model(x)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        """
        
        loss = self.loss_fn(labels, y_hat)
        return loss

## add self in parameter
def fit(self, train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training -> default = 500
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # def __init__(self, lrate, loss_fn, in_size, out_size):
    net = NeuralNet(lrate=0.01, loss_fn=nn.CrossEntropyLoss, in_size=len(train_set), out_size=len(train_labels))

    return [optimizer], [scheduler], net

    # raise NotImplementedError("You need to write fit part!")
    # return [],[],None
    ## _,predicted_labels, net = model.fit(model, train_set, train_labels, dev_set, args.max_iter)
    """
        optimizer = torch.optim.SGD(
        # self.model.parameters(),
        lr=self.hparams.learning_rate,
        weight_decay=self.hparams.weight_decay,
        momentum=0.9,
        nesterov=True,
    )
    total_steps = self.hparams.max_epochs * len(self.train_dataloader())
    scheduler = {
        "scheduler": WarmupCosineLR(
            optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
        ),
        "interval": "step",
        "name": "learning_rate",
    }"""
