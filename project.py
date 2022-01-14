# project.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Modified by Mahir Morshed for the spring 2021 semester
# Modified the code for ECE4453 Final Project on 11/19/2021

import sys
import argparse
import configparser
import copy

import numpy as np
import torch

import reader
import neuralnet_part1 as p1
import neuralnet_part2 as p2

"""
This file contains the main application that is run for this project.
"""

def compute_accuracies(predicted_labels,dev_set,dev_labels):
    yhats = predicted_labels
    if len(yhats) != len(dev_labels):
        print("Lengths of predicted labels don't match length of actual labels", len(yhats),len(dev_labels))
        return 0.,0.,0.,0.

    accuracy = np.mean(yhats == dev_labels)

    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(dev_labels))])
    fp = np.sum([yhats[i] != dev_labels[i] and yhats[i] == 1 for i in range(len(dev_labels))])
    fn = np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(dev_labels))])

    precision = tp / (tp + fp)
    recall = tp / (fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, f1, precision, recall

def main(args):
    reader.init_seeds(args.seed)

    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.dataset_file)

    p = p1 if args.part == 1 else p2

    train_set = torch.tensor(train_set, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    dev_set = torch.tensor(dev_set, dtype=torch.float32)

    _, predicted_labels, net = p.fit(train_set, train_labels, dev_set, args.max_iter)
    accuracy, f1, precision, recall = compute_accuracies(predicted_labels, dev_set, dev_labels)

    print("Accuracy:", accuracy)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    torch.save(net, "net.model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECE4453 Project: Neural Nets and PyTorch')

    parser.add_argument('--dataset', dest='dataset_file', type=str, default = 'mp3_data',
                        help='directory containing the training data')
    parser.add_argument('--max_iter',dest="max_iter", type=int, default = 500,
                        help='Maximum iterations: default 500')
    parser.add_argument('--part', dest="part", type=int, default=2,
                        help='Part 1 or Part 2')
    parser.add_argument('--seed', dest="seed", type=int, default=42,
                        help='seed source for randomness')


    args = parser.parse_args()
    main(args)
