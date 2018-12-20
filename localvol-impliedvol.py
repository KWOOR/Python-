# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:26:52 2018

@author: 우람
"""


import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from scipy.optimize import root, fsolve, newton
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
import scipy.interpolate as spi
import os
from numpy import arange,array,ones#,random,linalg
from pylab import plot,show
from scipy import stats
import statsmodels.api as sm
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\수치해석')
market_data = pd.read_excel(r'ksp200call2.xlsx')
#market_data = pd.read_excel('local_vol.xlsx', skip_footer=4)
#market_data=market_data.iloc[1:]
#market_data.index=market_data.index/365
#market_data[0.01:]=market_data[0.01:].replace(0, np.nan)
#%%
#기초 함수
def bs_price(s,k,r,q,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    d_1 = (np.log(s/k) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d_2 = (np.log(s/k) + (r-q-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    option_price = x * s * np.exp(-q*t) * norm.cdf(x*d_1) -x*k*np.exp(-r*t) *norm.cdf(x*d_2);
    return option_price;

def bs_vega(s,k,r,q,t,sigma):
    d_1 = (np.log(s/k) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    vega = s * np.exp(-q*t) * norm.pdf(d_1)*np.sqrt(t)
    return vega

def implied_vol(s,k,r,q,t,optionprice,option_type,init=0.1,tol=1e-6):
    vol = init
    vega = bs_vega(s,k,r,q,t,vol)
    while abs(bs_price(s,k,r,q,t,vol,option_type)-optionprice)>tol:
        err = bs_price(s,k,r,q,t,vol,option_type)-optionprice
        vol = vol - err/vega
        vega = bs_vega(s,k,r,q,t,vol)
    return vol.round(3)


#def implied_vol(s,k,r,q,t,optionprice,option_type):
#    f = lambda x : bs_price(s,k,r,q,t,x,option_type) - optionprice
#    return (sp.optimize.brentq(f,-5,5))

def impvol_f1(mdata,a,b,c,d,e,f,g):
    f = a +c*np.exp(b*mdata.iloc[:,1])+d*mdata.iloc[:,0]+e*mdata.iloc[:,0]**2 +f*np.power(mdata.iloc[:,0],3) +g *np.power(mdata.iloc[:,0],4)
    return f

def impvol_f2(m,t,a,b,c,d,e,f,g):
    f = a +c*np.exp(b*t)+d*m+e*m**2 +f*np.power(m,3) +g *np.power(m,4)
    return f


#%%
#기초 변수
option_type = 'call'
s = 292.42
r = 0.0165
q = 0.017
#s=300.81

#옵션 데이터
#x=[]
#y=[]
#z=[]
#for i in range(7):
#    for j in range(25):
#        x.append(market_data.index[i])
#        y.append(market_data.columns[j])
#        z.append(market_data.iloc[i,j])
#empty=np.isfinite(z)
#x=(pd.Series(x)*empty).replace(0,np.nan).dropna()
#t=(pd.Series(y)*empty).replace(0,np.nan).dropna()
#p=(pd.Series(z)*empty).replace(0,np.nan).dropna()


x = market_data.iloc[:,0] #행사가격
t = market_data.iloc[:,1] #잔존만기
p = market_data.iloc[:,2] #시장가격
temp_size = len(market_data)
#행사가격, 잔존 만기별 내재변동성 추정
imp_vol = np.ones(temp_size)
for i in range(temp_size):
    imp_vol[i] = implied_vol(s,x[i],r,q,t[i],p[i],option_type)
imp_vol[np.isinf(imp_vol)] = np.nan
imp_vol[np.isneginf(imp_vol)] = np.nan
imp_vol = pd.DataFrame(imp_vol).interpolate(method='nearest', limit_direction='both').fillna(method='backfill')
imp_vol = imp_vol[0].values

#%%
#내재 변동성 함수 모델 설정 및 계수 추정
m = pd.DataFrame(np.log((s*np.exp((r-q)*t))/x),columns =['m'])
t = pd.DataFrame(t)
mdata = pd.concat([m,t],axis=1)
y = impvol_f1(mdata, 0.01, 0.01, 0.01,0.01,0.01,0.01,0.01) # 초기값
a = sp.optimize.curve_fit(impvol_f1, mdata, imp_vol, maxfev=3000)[0]
#현재 파이썬 코드에서 안되는 이유는 옵티마이즈가 잘 안돼. 밑에 a 는 매
a= [120.31,-0.0012,-120.2493,0.1923,1.3030,-0.2323,-1.8505]
#moneyness, 잔존만기 범위 설정 (구할 내재 변동성의 범위를 설정)
m = np.linspace(min(m.values)[0],max(m.values)[0],100)
t = np.linspace(min(t.values)[0],max(t.values)[0],100)
mdata1 = pd.concat([pd.DataFrame(m,columns=['m']),pd.DataFrame(t,columns=['t'])],axis=1)

#%%
#해상 moneyness, 잔존 만기 범위에 따른 행사가격
#t = np.linspace(0,3,365)
x = (s*np.exp((r-q)*t))/np.exp(m)

#내재변동성 함수 이용한 도함수 계산
imp_vol = pd.DataFrame(np.zeros((len(m),len(t))))
dt = pd.DataFrame(np.zeros((len(m),len(t))))
dx = pd.DataFrame(np.zeros((len(m),len(t))))
dxx = pd.DataFrame(np.zeros((len(m),len(t))))
d = pd.DataFrame(np.zeros((len(m),len(t))))
#%%
#implied vol
for i in range(len(m)):
    for j in range(len(t)):
        imp_vol.iloc[i,j] = impvol_f2(m[i],t[j],a[0],a[1],a[2],a[3],a[4],a[5],a[6])
        dt.iloc[i,j] = a[2]*a[1]*np.exp(a[1]*t[j])*(r-q)*(a[3]+2*a[4]*m[i]+3*a[5]*m[i]**2 + 4*a[6]*np.power(m[i],3))
        dx.iloc[i,j] = -(a[3]+2*a[4]*m[i]+3*a[5]*m[i]**2 + 4*a[6]*np.power(m[i],3))/x[i];
        dxx.iloc[i,j] = (a[3]+2*a[4]*(m[i]+1)+3*a[5]*m[i]*(m[i]+2)+ 4*a[6]*m[i]**2 * (m[i]+3)) / x[i]**2;
        d.iloc[i,j] = (np.log(s/x[i])+(r-q + 0.5*imp_vol.iloc[i,j]**2)*t[j])/(imp_vol.iloc[i,j]*np.sqrt(t[j]));
        
#%%
#local vol
local_vol = pd.DataFrame(np.zeros((len(x),len(t))))
for i in range(len(x)):
    for j in range(len(t)):
          local_vol.iloc[i,j] = np.sqrt((imp_vol.iloc[i,j]**2 + 2*imp_vol.iloc[i,j]*t[j]*\
                        (dt.iloc[i,j]+(r-q)*x[i]*dx.iloc[i,j]))/((1+x[i]*d.iloc[i,j]*dx.iloc[i,j]*\
               np.sqrt(t[j]))**2 + imp_vol.iloc[i,j]**2 * (x[i]**2) * t[j]*(dxx.iloc[i,j]-d.iloc[i,j]*\
               (dx.iloc[i,j]**2)*np.sqrt(t[j]))));
                        
#%%
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
xnew,tnew = np.meshgrid(x, t)
#local_vol = local_vol.dropna(axis =1)
#xnew = xnew[:,1:]
#tnew = tnew[:,1:]
surf = ax.plot_surface(xnew, tnew, local_vol.T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
