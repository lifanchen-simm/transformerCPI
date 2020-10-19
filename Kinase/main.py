# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/25 10:03
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""

import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import pickle

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""


    with open('../dataset/kinase_train.txt',"rb") as f:
        data = pickle.load(f)
    dataset = shuffle_dataset(data, 1234)
    dataset_train, dataset_dev = split_dataset(data, 0.8)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-4
    weight_decay = 1e-4
    iteration = 30
    kernel_size = 9

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/AUCs--lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=9,n_layer=3,batch=64,balance,lookaheadradam'+ '.txt'
    file_model = 'output/model/' + 'lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=9,n_layer=3,batch=64,balance,lookaheadradam'
    AUC = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """Start training."""
    print('Training...')
    print(AUC)
    start = timeit.default_timer()
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
    max_AUC_dev = 0
    for epoch in range(1, iteration + 1):

        loss_train = trainer.train(dataset_train, device)
        AUC_dev, PRC_dev = tester.test(dataset_dev)
        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, PRC_dev]
        scheduler.step()
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
        print('\t'.join(map(str, AUCs)))
