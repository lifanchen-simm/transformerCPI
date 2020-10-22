# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 20:49
@author: LiFan Chen
@Filename: main_glu.py
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
from model_glu import *
import timeit


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


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
    # torch.backends.cudnn.deterministic = True
    DATASET = "human"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/word2vec_30/')
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_2= split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_2, 0.5)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-3
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 0.5
    iteration = 40
    kernel_size = 5

    encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output_human/result/AUCs--lr=1e-3,dropout=0.1,weight_decay=1e-4,kernel=5,n_layer=3,batch=64,decay_interval=5.txt'
    file_model = 'output_human/model/lr=1e-3,dropout=0.1,weight_decay=1e-4,kernel=5,n_layer=3,batch=64,decay_interval=5.pt'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    #max_AUC_dev = 0
    max_AUC_test = 0
    epoch_label = 0
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, device)
        AUC_dev,_,_ = tester.test(dataset_dev)
        AUC_test, precision_test, recall_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
        tester.save_AUCs(AUCs, file_AUCs)
        if  AUC_test > max_AUC_test:
            tester.save_model(model, file_model)
            max_AUC_test = AUC_test
            epoch_label = epoch
        print('\t'.join(map(str, AUCs)))

    print("The best model is epoch",epoch_label)
