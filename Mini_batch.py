from utils import *
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
import time


class Mini_batch():
    def __init__(self,batch_size, proj):
        self.proj = proj
        self.batch_size = batch_size
        if proj:
            self.title = "Mini_batch_proj"+str(batch_size)
        else:
            self.title = "Mini_batch_"+str(batch_size)

    def train(self,epoch,radius,X, train_A, test_A, train_B, test_B, alpha=0):
        # Mini batch training samples

        num_batches = len(train_B) // self.batch_size
        indices = np.random.permutation(len(train_B))
        batches = np.split(indices,num_batches)
        samples_batches = np.random.choice(num_batches, epoch)
        
        accuracy_train, accuracy_test, loss_train, loss_test = [], [], [], []

        # init loss
        l_train, _, acc_train = reg_hinge_loss(X,train_A,train_B,alpha)
        l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
        accuracy_train.append(acc_train), accuracy_test.append(acc_test)
        loss_train.append(l_train), loss_test.append(l_test)
        print(f"init : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")
        print("-"*25)
        # training loop
        i=1
        start = time.time()
        for t in samples_batches:
            
            step = 1/i**(1/2)
            train_batch = batches[t]
            sample_A = train_A[train_batch]
            sample_B = train_B[train_batch]
            l_train, grad, acc_train = reg_hinge_loss(X,sample_A,sample_B,alpha)
            l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
            
            X-= step*grad
            if self.proj:
                X = l1_ball_proj(X,z=radius) # L1 projection

            if i%10==0:
                print(f"epoch {i} : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

            accuracy_train.append(acc_train), accuracy_test.append(acc_test)
            loss_train.append(l_train), loss_test.append(l_test)
            i+=1
        stop = time.time()

        print("-"*25)
        print(f"final : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

        return accuracy_test, stop-start