# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:29:31 2018

@author: 우람
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\시뮬')

S=100
K=100
T=30/365
r=0
v=0.01
kappa=2
theta=0.01
Lambda=0
q=0
sigma=0.1
rho=0
call=1.1345
N=100
n_steps=N
n_trials=1000

#%%

def simulate(n_trials, n_steps, S=100,K=100,T=30/365,r=0,v=0.01,kappa=2,theta=0.01,Lambda=0,sigma=0.1,rho=0, q=0,underlying_process='Heston model',antitheticVariates=False, boundaryScheme='full truncation'):
    dt = T / n_steps
    mu = r - q
    n_trials = n_trials
    n_steps = n_steps
    boundaryScheme = boundaryScheme
    kappa=kappa
    theta=(kappa*theta)/(kappa+Lambda)

    if (underlying_process == "geometric brownian motion"):
        #             first_step_prices = np.ones((n_trials,1))*np.log( S0)
        log_price_matrix = np.zeros((n_trials, n_steps))
        normal_matrix = np.random.normal(size=(n_trials, n_steps))
        if (antitheticVariates == True):
            n_trials *= 2
            n_trials = n_trials
            normal_matrix = np.concatenate((normal_matrix, -normal_matrix), axis=0)
        cumsum_normal_matrix = normal_matrix.cumsum(axis=1)
        #             log_price_matrix = np.concatenate((first_step_prices,log_price_matrix),axis=1)
        deviation_matrix = cumsum_normal_matrix *  sigma * np.sqrt(dt) + \
                           (mu -  sigma ** 2 / 2) * dt * np.arange(1, n_steps + 1)
        log_price_matrix = deviation_matrix + np.log( S)
        price_matrix = np.exp(log_price_matrix)
        price_zero = (np.ones(n_trials) *  S)[:, np.newaxis]
        price_matrix = np.concatenate((price_zero, price_matrix), axis=1)
        price_matrix = price_matrix

    elif (underlying_process == "CIR model"):
        # generate correlated random variables
        randn_matrix_v = np.random.normal(size=(n_trials, n_steps))
        if (antitheticVariates == True):
            n_trials *= 2
            n_trials = n_trials
            randn_matrix_v = np.concatenate((randn_matrix_v, -randn_matrix_v), axis=0)

        # boundary scheme fuctions
        if (boundaryScheme == "absorption"):
            f1 = f2 = f3 = lambda x: np.maximum(x, 0)
        elif (boundaryScheme == "reflection"):
            f1 = f2 = f3 = np.absolute
        elif (boundaryScheme == "Higham and Mao"):
            f1 = f2 = lambda x: x
            f3 = np.absolute
        elif (boundaryScheme == "partial truncation"):
            f1 = f2 = lambda x: x
            f3 = lambda x: np.maximum(x, 0)
        elif (boundaryScheme == "full truncation"):
            f1 = lambda x: x
            f2 = f3 = lambda x: np.maximum(x, 0)

        # simulate CIR process
        V_matrix = np.zeros((n_trials, n_steps + 1))
        V_matrix[:, 0] =  S

        for j in range( n_steps):
            V_matrix[:, j + 1] = f1(V_matrix[:, j]) -  kappa * dt * (f2(V_matrix[:, j]) -  theta) + \
                                  sigma * np.sqrt(f3(V_matrix[:, j])) * np.sqrt(dt) * randn_matrix_v[:, j]
            V_matrix[:, j + 1] = f3(V_matrix[:, j + 1])

        price_matrix = V_matrix
        price_matrix = price_matrix


    elif (underlying_process == "Heston model"):
        # generate correlated random variables
        randn_matrix_1 = np.random.normal(size=(n_trials, n_steps))
        randn_matrix_2 = np.random.normal(size=(n_trials, n_steps))
        randn_matrix_v = randn_matrix_1
        randn_matrix_S =  rho * randn_matrix_1 + np.sqrt(1 -  rho ** 2) * randn_matrix_2
        if (antitheticVariates == True):
            n_trials *= 2
            n_trials = n_trials
            randn_matrix_v = np.concatenate((randn_matrix_v, +randn_matrix_v), axis=0)
            randn_matrix_S = np.concatenate((randn_matrix_S, -randn_matrix_S), axis=0)

        # boundary scheme fuctions
        if (boundaryScheme == "absorption"):
            f1 = f2 = f3 = lambda x: np.maximum(x, 0)
        elif (boundaryScheme == "reflection"):
            f1 = f2 = f3 = np.absolute
        elif (boundaryScheme == "Higham and Mao"):
            f1 = f2 = lambda x: x
            f3 = np.absolute
        elif (boundaryScheme == "partial truncation"):
            f1 = f2 = lambda x: x
            f3 = lambda x: np.maximum(x, 0)
        elif (boundaryScheme == "full truncation"):
            f1 = lambda x: x
            f2 = f3 = lambda x: np.maximum(x, 0)

        # simulate stochastic volatility process
        V_matrix = np.zeros((n_trials, n_steps + 1))
        V_matrix[:, 0] =  v
        log_price_matrix = np.zeros((n_trials, n_steps + 1))
        log_price_matrix[:, 0] = np.log( S)
        for j in range( n_steps):
            #                 V_matrix[:,j+1] =  kappa* theta*dt + (1- kappa*dt)*V_matrix[:,j] +\
            #                      xi*np.sqrt(V_matrix[:,j]*dt)*randn_matrix_v[:,j]
            V_matrix[:, j + 1] = f1(V_matrix[:, j]) -  kappa * dt * (f2(V_matrix[:, j]) -  theta) + \
                                  sigma * np.sqrt(f3(V_matrix[:, j])) * np.sqrt(dt) * randn_matrix_v[:, j]
            V_matrix[:, j + 1] = f3(V_matrix[:, j + 1])
            log_price_matrix[:, j + 1] = log_price_matrix[:, j] + (mu - V_matrix[:, j] / 2) * dt + \
                                         np.sqrt(V_matrix[:, j] * dt) * randn_matrix_S[:, j]
        price_matrix = np.exp(log_price_matrix)
        price = (sum(np.maximum(price_matrix[:,-1]-K,0))/n_trials)*np.exp(-r*T)
        err= (np.maximum(price_matrix[:,-1]-K,0)).std()/np.sqrt(n_trials)


    return price, err


call=[]
error=[]
for n_trials in range(1000, 11000, 1000):
    price, err=simulate(n_trials, n_steps)
    call.append(price)
    error.append(err)
cu=pd.DataFrame(call)+1.96*pd.DataFrame(error)
cu.index=np.linspace(1000,10000,10)
cd=pd.DataFrame(call)-1.96*pd.DataFrame(error)
cd.index=np.linspace(1000,10000,10)
call=pd.DataFrame(call)
call.index=np.linspace(1000,10000,10)
true=pd.DataFrame(np.ones(10))*1.1345
true.index=np.linspace(1000,10000,10)


plt.plot(call, label='Heston Call')
plt.plot(cu, label='Upper CI')
plt.plot(true, label = 'True Price')
plt.plot(cd, label='Lower CI')
plt.show()
    