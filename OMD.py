from utils import *
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
import time


class OMD():
    def __init__(self):
        self.title = 'OMD'

    def train(self,epoch,radius,X, train_A, test_A, train_B, test_B, alpha=0,proj=True):
        
        # pick random sample for training
        train_samples = np.random.choice(len(train_B), epoch)
        accuracy_train, accuracy_test, loss_train, loss_test = [], [], [], []

        Y = copy.deepcopy(X)

        # init
        l_train, _, acc_train = reg_hinge_loss(X,train_A,train_B,alpha)
        l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
        accuracy_train.append(acc_train), accuracy_test.append(acc_test)
        loss_train.append(l_train), loss_test.append(l_test)
        print(f"init : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")
        print("-"*25)
        # training loop
        i=1
        start = time.time()
        for t in train_samples:
            step = (1/i)**(1/2)
            sample_A = np.expand_dims(train_A[t],axis=0)
            sample_B = np.expand_dims(train_B[t],axis=0)
            l_train, grad, acc_train = reg_hinge_loss(X,sample_A,sample_B,alpha)
            l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
            
            Y-= step*grad # Gradient step in the dual space ("lazy" version of OMD)
            X = l1_ball_proj(Y,z=radius) # L1 projection

            if i%100==0:
                print(f"epoch {i} : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

            accuracy_train.append(acc_train), accuracy_test.append(acc_test)
            loss_train.append(l_train), loss_test.append(l_test)
            i+=1
        stop = time.time()

        print("-"*25)
        print(f"final : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

        return accuracy_test, stop-start