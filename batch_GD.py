from utils import *
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
import time

class batch_GD():
    def __init__(self,proj):
        self.proj=proj
        if proj:
            self.title = 'Batch_GD_proj'
        else:
            self.title = 'Batch_GD'

    def train(self,epoch, radius,X, train_A, test_A, train_B, test_B,alpha=0):
        accuracy_train, accuracy_test, loss_train, loss_test = [], [], [], []

        # init
        l_train, _, acc_train = reg_hinge_loss(X,train_A,train_B,alpha)
        l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
        accuracy_train.append(acc_train), accuracy_test.append(acc_test)
        loss_train.append(l_train), loss_test.append(l_test)
        print(f"init : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")
        print("-"*25)

        # training loop
        start = time.time()
        for t in range(1,epoch):
            step = (1/t)**1/2
            l_train, grad, acc_train = reg_hinge_loss(X,train_A,train_B,alpha)
            l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
            
            X-= step*grad # gradient step
            if self.proj:
                X = l1_ball_proj(X,z=radius) # L1 projection

            if t % 10 == 0:
                print(f"epoch {t} : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

            accuracy_train.append(acc_train), accuracy_test.append(acc_test)
            loss_train.append(l_train), loss_test.append(l_test)
        stop = time.time()

        print("-"*25)
        print(f"final : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

        return accuracy_test, stop-start

