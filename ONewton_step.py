from utils import *
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
import time


class ONS():
    def __init__(self):
        self.title = 'ONS'

    def train(self,epoch,radius,X, train_A, test_A, train_B, test_B, proj=True,alpha=0):
        # pick random sample for training
        train_samples = np.random.choice(len(train_B), epoch)
        accuracy_train, accuracy_test, loss_train, loss_test = [], [], [], []

        # Init hessian estimation 
        gamma = 0.05
        Y = copy.deepcopy(X)
        A = (1/gamma**2)*np.identity(len(X))
        Ainv = np.linalg.inv(A)

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
            step = 1/gamma
            sample_A = np.expand_dims(train_A[t],axis=0)
            sample_B = np.expand_dims(train_B[t],axis=0)

            # compute loss/ accuracy and gradient
            l_train, grad, acc_train = reg_hinge_loss(X,sample_A,sample_B,alpha)
            l_test, _, acc_test = reg_hinge_loss(X,test_A,test_B,alpha, test=True)
            
            # Update parameters
            grad_matrix = np.outer(grad,grad) # gradients matrix (hessian approximation)
            A += grad_matrix
            #Ainv -= np.linalg.multi_dot([Ainv,grad_matrix,Ainv])/(1+np.linalg.multi_dot([grad,Ainv,grad]))
            Ainv = np.linalg.inv(A)
            Y = X - step*np.dot(Ainv,grad) # multiple learning rate step
            
            # weighted proj
            X = l1_ball_proj(Y,z=radius,d=np.diag(np.diag(A)),weighted=True)


            if i%100==0:
                print(f"epoch {i} : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

            accuracy_train.append(acc_train), accuracy_test.append(acc_test)
            loss_train.append(l_train), loss_test.append(l_test)
            i+=1
        stop = time.time()

        print("-"*25)
        print(f"final : test loss {'%.2f'%l_test}, test acc {'%.2f'%acc_test}")

        return accuracy_test, stop-start