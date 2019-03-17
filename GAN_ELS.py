# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:26:38 2019

@author: kur7
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import time
import sys
from datetime import date
from datetime import timedelta
from datetime import datetime
import fix_yahoo_finance as yf

yf.pdr_override()

SX5E = yf.download("^STOXX50E", start="2007-01-01", end = '2018-02-13')['Close']  # EuroStoxx50 데이터 받기 
HSCEI = yf.download("^HSCE", start="2007-01-01", end = '2018-02-13')['Close'] # HSCEI 데이터 받기
#일별 데이터니까, 분포도 일별 분포다     끝나는 날은 ELS Pricing 시작하는 날로 잡음 
''' 가끔 위의 데이터가 안 받아질 때가 있음... 계속 시도해보면 5분 안에 해결 됨 '''

slim = tf.contrib.slim
stockCode = 'ELS_GAN'
saveDir = './Saver/' + stockCode            # 학습 결과를 저정할 폴더
saveFile = saveDir + '/' + stockCode        # 학습 결과를 저장할 파일

#%%

def createDataSet(code):
    data = pd.DataFrame(code)
    
    data['rtn'] = pd.DataFrame(data['Close']).apply(lambda x: np.log(x) - np.log(x.shift(1)))
    data = data.dropna()

    volatility = np.std(data['rtn'])
    lastPrice = data['Close'][-1]
    return np.array(data['rtn']), lastPrice, volatility

def KL(P, Q):
    histP, binsP = np.histogram(P, bins=150)
    histQ, binsQ = np.histogram(Q, bins=binsP)
    
    histP = histP / np.sum(histP) + 1e-8
    histQ = histQ / np.sum(histQ) + 1e-8

    kld = np.sum(histP * np.log(histP / histQ))
    return histP, histQ, kld

realData_E, lastPrice_E, volatility_E = createDataSet(SX5E)
realData_E = 10 * (realData_E - np.mean(realData_E)) / volatility_E #정규화 밑 스케일링.. 정규화 방식으로 다른것도 사용 가능
realData_E = realData_E.reshape(realData_E.shape[0], 1)
nDataRow_E = realData_E.shape[0]
nDataCol = realData_E.shape[1]

realData_H, lastPrice_H, volatility_H = createDataSet(HSCEI)
realData_H = 10 * (realData_H - np.mean(realData_H)) / volatility_H #마찬가지..
realData_H = realData_H.reshape(realData_H.shape[0], 1)
nDataRow_H = realData_H.shape[0]


nGInput = 20
nGHidden = 128
nDHidden = 128

tf.reset_default_graph()
def Generator(z, nOutput, nHidden=nGHidden, nLayer=1):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, nOutput, activation_fn=None)
    return x

def Discriminator(x, nOutput=1, nHidden=nDHidden, nLayer=1, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h = slim.stack(x, slim.fully_connected, [nHidden] * nLayer, activation_fn=tf.nn.relu)
        d = slim.fully_connected(h, nOutput, activation_fn=None)
    return d

def getNoise(m, n=nGInput):
    z = np.random.uniform(-1., 1., size=[m, n])
    return z


 #%%
Simnum=100000  #시뮬레이션 횟수!! 

def LetsLearn(Data, nSim = Simnum, nBatchCnt=10, nLearning= 10000, T=252*3):  
    # nBatchCnt: Mini-batch를 위해 input 데이터를 몇 개 블록으로 나눌건지
    
    with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
        x = tf.placeholder(tf.float32, shape=[None, nDataCol], name='x')
        z = tf.placeholder(tf.float32, shape=[None, nGInput], name='z')
        Gz = Generator(z, nOutput=nDataCol)
        Dx = Discriminator(x)
        DGz = Discriminator(Gz, reuse=True)
    D_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)) +
        tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz, labels=tf.zeros_like(DGz)))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz, labels=tf.ones_like(DGz)))
    
    thetaG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    thetaD = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    
    trainD = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list = thetaD)
    trainG = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list = thetaG)
    saver = tf.train.Saver()

    # 그래프를 실행한다
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # 기존 학습 결과를 적용한다.
#    if tf.train.checkpoint_exists(saveDir):
#        saver.restore(sess, saveFile)   

    histLossD = []      # Discriminator loss history 저장용 변수
    histLossG = []      # Generator loss history 저장용 변수
    histKL = []         # KL divergence history 저장용 변수
    fake = np.empty((nSim, T), float)
    nBatchSize = int(Data.shape[0] / nBatchCnt)  # 블록 당 Size
    nK = 1              # Discriminator 학습 횟수
    k = 0
    for i in range(nLearning):
        for n in range(nBatchCnt):
            nFrom = n * nBatchSize
            nTo = n * nBatchSize + nBatchSize
            
            if n == nBatchCnt - 1:
                nTo = Data.shape[0]
                   
            bx = Data[nFrom : nTo]
            bz = getNoise(m=bx.shape[0])
    
            if k < nK:
                # Discriminator를 nK-번 학습한다.
                _, lossDHist = sess.run([trainD, D_loss], feed_dict={x: bx, z : bz})
                k += 1
            else:
                # Generator를 1-번 학습한다.
                _, lossGHist = sess.run([trainG, G_loss], feed_dict={x: bx, z : bz})
                k = 0
        
        # 100번 학습할 때마다 Loss, KL의 history를 보관해 둔다
        if i % 100 == 0:
            p, q, kld = KL(bx, sess.run(Gz, feed_dict={z : bz}))
            histKL.append(kld)
            histLossD.append(lossDHist)
            histLossG.append(lossGHist)
            print("%d) D-loss = %.4f, G-loss = %.4f, KL = %.4f" % (i, lossDHist, lossGHist, kld))
    print("Finished! Making some Fake Data")    
    for i in range(nSim):
        fakeData = sess.run(Gz, feed_dict={z: getNoise(m=T)}) #1년을 252일로 잡고, ELS 만기인 3년 
        fake[i, :]=fakeData.T # 행 하나당, 시뮬레이션 한 번 
    tf.reset_default_graph()
    return histKL, histLossD, histLossG, fake

    
#%%
#학습 잘 됐는지 확인하기

histKL_E, histLossD_E, histLossG_E, fakeData_E = LetsLearn(realData_E)


plt.figure(figsize=(6, 3))
plt.plot(histLossD_E, label='Loss-D')
plt.plot(histLossG_E, label='Loss-G')
plt.legend()
plt.title("Loss history")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(histKL_E)
plt.title("KL divergence")
plt.show()


p, q, kld = KL(realData_E, fakeData_E[1,:].T) #시뮬레이션 한거 아무거나 한 행 불러서 비교해본것 
x = np.linspace(-3, 3, 150)
plt.plot(x, p, color='blue', linewidth=2.0, alpha=0.7, label='Real Data')
plt.plot(x, q, color='red', linewidth=2.0, alpha=0.7, label='Fake Data')

#%%
histKL_H, histLossD_H, histLossG_H, fakeData_H = LetsLearn(realData_H)

plt.figure(figsize=(6, 3))
plt.plot(histLossD_H, label='Loss-D')
plt.plot(histLossG_H, label='Loss-G')
plt.legend()
plt.title("Loss history")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(histKL_H)
plt.title("KL divergence")
plt.show()


p, q, kld = KL(realData_H, fakeData_H[2,:].T)
x = np.linspace(-3, 3, 150)
plt.plot(x, p, color='blue', linewidth=2.0, alpha=0.7, label='Real Data')
plt.plot(x, q, color='red', linewidth=2.0, alpha=0.7, label='Fake Data')


    #%%
# 인덱스 시뮬레이션 실행 
Face = 10000
rf = 0.0165 / 252 # 앞에서부터 일년을 252일로 잡음.. 
coupon=[0.027, 0.054, 0.081, 0.108, 0.135, 0.162]
K=[0.9, 0.9, 0.85, 0.85, 0.80, 0.80] 
KI=0.5

Euro = fakeData_E/10 ; Hsc =fakeData_H/10
E0 = lastPrice_E ; H0 = lastPrice_H
EPrice = np.array([E0]*Simnum) ; HPrice = np.array([H0]*Simnum)

def Simulation(data1, Price, S0, volatility, r=rf):
    for W in range(data1.shape[1]):
        S1= S0 * np.exp(volatility * data1[:,W] + (r - (volatility **2)/2))
        Price= np.vstack([Price,S1.reshape(-1,1).T])
        S0=S1
    return Price.T

EPrice = Simulation(Euro, EPrice, E0, volatility_E)
HPrice = Simulation(Hsc, HPrice, H0, volatility_H)



#%%
# ELS 평가


mod = sys.modules[__name__]

half_year = 126

for i in range(1, len(coupon)+1):
    buff = (HPrice[:,half_year*i] > lastPrice_H * K[i-1]) * (EPrice[:,half_year*i] >lastPrice_E*K[i-1]) 
    setattr(mod, 'early_redeem_{}'.format(i), buff )
    HPrice[:,half_year*i][buff]=np.nan
    HPrice = HPrice[~np.isnan(HPrice).any(axis=1)]
    EPrice[:,half_year*i][buff]=np.nan
    EPrice = EPrice[~np.isnan(EPrice).any(axis=1)]
    print('%.f차 조기상환 확률은' %i, 100*buff.sum()/Simnum, '% 입니다.')


fin_redeem = (np.min(HPrice, axis=1) > lastPrice_H *KI) * (np.min(EPrice, axis=1) >lastPrice_E *KI)
HPrice[:, 0][fin_redeem] = np.nan  #굳이 0이 아니어도 상관없음... 마지막 열만 아니면 됨.. 그냥 아무거나 넣음 
HPrice = HPrice[~np.isnan(HPrice).any(axis=1)]
EPrice[:,0][fin_redeem]=np.nan
EPrice = EPrice[~np.isnan(EPrice).any(axis=1)]
print('만기까지 낙인은 치지 않아서 만기에 상환될 확률은 ' , 100*fin_redeem.sum()/Simnum, '% 입니다.')
print('원금손실확률은 ', 100*len(EPrice)/Simnum, '% 입니다.')

E_Loss = EPrice[:, -1] / lastPrice_E - 1
H_Loss = HPrice[:, -1] / lastPrice_H -1
Loss = pd.concat([pd.Series(E_Loss), pd.Series(H_Loss)], axis=1)
Loss = Loss.min(axis=1)

redeem = [early_redeem_1, early_redeem_2, early_redeem_3, early_redeem_4, early_redeem_5, early_redeem_6]
for i in redeem:
    buff = i
    index = redeem.index(i)
    setattr(mod, 'price_{}'.format(index+1), 
            ( Face / (1 + ( rf / (252/ (126*(index+1) ) ) ) )) *(1+ coupon[index]) * buff.sum()/Simnum)

fin_price = ( Face / (1 + ( rf / (252/756) ) ) ) *(1+ coupon[5]) * fin_redeem.sum()/Simnum
loss_price =( ( Face / (1 + ( rf / (252/756) ) ) ) * (1 + Loss)  / Simnum).sum()


ELS=price_1 + price_2 + price_3+ price_4 + price_5 + price_6 +fin_price +loss_price
print('ELS의 현재가격은 ', ELS, '원 입니다.')



      