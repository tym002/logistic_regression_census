#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 00:13:35 2019

@author: tianyu
"""

import numpy as np
import random

def logreg_loss(X,y,w):
    '''
    Given a dataset of examples X and example labels y,
    logreg_loss calculates the loss function for logistic regression based on
    weight w
    
    Arguments: 
        X: training data, a (124,N) array, N is the number of total training example
        y: traing labels, a (N,1) array
        w: weights, a (124,1) array 
    '''
    n = y.shape[0]
    return np.sum(np.log(1+np.exp(-np.multiply(np.transpose(np.dot(np.transpose(w),X)),y))))/n


def logreg_error(X,y,w):
    '''
    Given a dataset of examples X and example labels y,
    logreg_error calculates the error rate for predictions based on weight w
    
    Arguments: 
        X: training data, a (124,N) array, N is the number of total training example
        y: traing labels, a (N,1) array
        w: weights, a (124,1) array 
    '''
    r = np.equal(np.sign(np.transpose(np.dot(np.transpose(w),X))),y)
    return list(r).count(False)/r.shape[0] 


def logreg_sgd(X,y,w,alpha,sigma,T):
    '''
    Given a dataset of examples X and example labels y, logreg_sgd runs SGD 
    on that dataset for T iterations, returning the resulting model
    
    Arguments: 
        X: training data, a (124,N) array, N is the number of total training example
        y: traing labels, a (N,1) array
        w: weights, a (124,1) array 
        alpha: learning rate 
        sigma: regularization constant
        T: number of steps
    '''
    n = y.shape[0]
    for i in range(0,T):
        batch = list(range(0,n))
        while len(batch) > 0:
            rn = random.randint(0,len(batch)-1)
            sample = batch[rn]
            Xi = np.reshape(X[:,sample],(124,1))
            yi = y[sample,0]
            batch.remove(sample)
            w = w + alpha*yi*Xi/(1+np.exp(yi*np.dot(np.transpose(w),Xi)))-sigma*2*np.sum(w)
    return w

if __name__ == "__main__":       
    X = np.load('train_data.npy')
    y = np.load('train_label.npy')
    Xt = np.load('test_data.npy')
    yt = np.load('test_label.npy')
    w = np.zeros((124,1))
    
    wp = logreg_sgd(X,y,w,0.001,0.001,10)
    print(logreg_error(X,y,wp))