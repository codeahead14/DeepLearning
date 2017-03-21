# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 00:19:17 2017

@author: GAURAV
"""

import cPickle
import numpy as np
import graphlab as gl
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy import ndimage

def softmax_probability(coefficients,features):
    if np.shape(coefficients)[1] != np.shape(features)[0]:
        coefficients = np.transpose(coefficients)
    score = np.dot(coefficients,features)
    return np.transpose(np.exp(score-np.amax(score,axis=0))/np.sum(np.exp(score-np.amax(score,axis=0)),axis=0))
    
def one_hot_encoding(output_label,classes):
    '''It is highly probable that our output_label array consists of all possible classes.
    Yet we will be using num_classes to be sent from outside.'''
    output_dim = np.shape(output_label)
    one_hot_encoded = np.zeros(shape=(output_dim[0],np.shape(classes)[0]))
    for i in xrange(output_dim[0]):
        ind = np.where(classes == output_label[i])[0][0]
        one_hot_encoded[i][ind] = 1
    return one_hot_encoded
    
def softmax_regression(coefficients,features,output_label,classes,step_size,lamda,max_iter=1000):
    features_dim = np.shape(features)
    coeff_dim = np.shape(coefficients)
    probs = np.zeros(shape=(coeff_dim[1],features_dim[0]))
    #cost = np.zeros(np.shape(output_label)[0])  # Cost - array of len (number of output labels)
    cost = []
    
    for count in xrange(max_iter): 
        probs = softmax_probability(coefficients,features.T)
        
        '''The following works provided output_label contains numerical data.
        For categorical classes convert all classes into corresponding indices and 
        the output labels accordingly.'''
        data_loss = -np.sum(np.log(probs[range(features_dim[0]),output_label])) #+ (lamda/2)*np.sum(coefficients*coefficients)
        cost.append(data_loss/features_dim[0])

        ''' Computing the gradient. Shape of Features(X) = [Number of data points, number of features].
        Shape of Probabilities = [number of data points, number of classes].
        Here we will have to compute the outer product of X & Probabilities. Each column will then correspond to 
        the cost derivative w.r.t to corresponding class'''
        probs[range(features_dim[0]),output_label] -= 1
        derivative = np.dot(features.T,probs) #+ lamda*coefficients.T

        coefficients = coefficients - step_size*derivative.T
    return coefficients,cost
    
if __name__ == "__main__":
    with open("mnist.pkl",'rb') as mnist_file:
        train_data,valid_data,test_data = cPickle.load(mnist_file)
    
    train_vec_in,train_label_out = train_data    
    classes = set(train_label_out)
    step_size = 1e-5
    coefficients = np.zeros(shape=(len(classes),np.shape(train_vec_in)[1]))
    model_coefficients,cost = softmax_regression(coefficients,train_vec_in, \
                train_label_out,classes,step_size,lamda=1,max_iter=100)