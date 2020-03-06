# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/23 15:24
@author: LiFan Chen
@Filename: analysis.py
@Software: PyCharm
"""
file_path = "dataset_kiba/kiba_train.txt"
with open(file_path,"r") as f:
    data_list = f.read().strip().split('\n')
P = 0
N = 0
with open("dataset_kiba/train_c.txt","w") as f:
    for element in data_list:
        compound = element.split()[0]
        protein = element.split()[1]
        affinity = element.split()[2]
        if float(affinity) >= 12.1:
            affinity = 1.0
            P += 1
            #for i in range(4):
            f.write("{}\n".format(" ".join([compound, protein, str(affinity)])))
        else:
            affinity = 0.0
            N += 1
            f.write("{}\n".format(" ".join([compound,protein,str(affinity)])))
print(P,N)