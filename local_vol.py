# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:53:25 2018

@author: 우람
"""


import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\수치해석')

#call_option 가격 불러오기
buff= pd.read_excel("local_vol.xlsx", skip_footer=4) #쓸데 없는 거 버리고..
buff=buff.iloc[1:]

# 이론적으로 맞지 않는 값들을 제거해줌.. 잔존만기가 길 수록 시장가가 높아야하는데, 그렇지 않은 경우 제거함.
#즉, 시장의 노이즈 제거  
'''for i in range(1,len(buff)):
    buff.iloc[i,:]=(buff.iloc[i,:]>buff.iloc[i-1,:])*buff.iloc[i,:]'''
buff[1:]= buff[1:].replace(0, np.nan)

call_index=range(0, 365*3)  #시간은 3년으로 늘림
call= pd.DataFrame(columns= buff.columns, index=call_index).replace(np.nan, True)*buff
call[1:]= call[1:].replace(0, np.nan) #잔존만기가 0일 때의 0값은 True값이므로 제외하고 replace함 

#%%  Call option inter&extrapolation  :linear
''' vol을 추정하기 위해선 call option의 가격이 연속적으로 있어야 하기에 option을 먼저 보간법으로 구했다!'''
''' 그러나 linear interpolation으로 옵션 가격을 추정한 후에 local vol을 구하면 안 된다!! 
 선형 결합이기 때문에.. C_KK를 구할 때, 앞의 값과 뒤의 값을 더하고 중간값에 2를 곱해서 빼는데.. 이 둘의 값이 같아서
 C_KK = 0이 된다!!'''
#interpolation & extrapolation을 smooth하게 하기 위해서 중간에 두 행을 먼저 보간법 처리해줌..
# 0일째의 가격은 시장에서 실제로 관찰된 값은 아니므로.. 시장에서 실제 관찰된 가격들을 보간법 처리
empty=call.iloc[20,:].dropna()
x_points=np.array(empty.index, dtype='float64')
y_points=np.array(empty.values)
lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
call.iloc[20,:]=lininter(np.linspace(270, 330, 25))

empty=call.iloc[48,:].dropna()
x_points=np.array(empty.index, dtype='float64')
y_points=np.array(empty.values)
lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
call.iloc[48,:]=lininter(np.linspace(270, 330, 25))


for i in range(0,25):
    empty= call.iloc[0:,i].dropna()
    x_points=np.array(empty.index, dtype='float64')
    y_points=np.array(empty.values)
    lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
    for j in range(0,1095):
        if lininter(j) >=0:
            call.iloc[j,i]=lininter(j)
        else:
            call.iloc[j,i]=0

#%% Call - Linear 후에 local_volatility estimate

vol=call.copy()
for i in range(1, 1095):
    for j in range(1,24):
        C_T= (call.iloc[i,j]-call.iloc[i-1,j])
        C_K=(0.0165*call.columns[j])*(call.iloc[i,j]-call.iloc[i,j-1])/2.5
        C_KK=(call.columns[j]**2)*(call.iloc[i,j+1]-2*call.iloc[i,j]+call.iloc[i,j-1])/(2.5**2)
        if C_KK !=0:
            vol.iloc[i,j]=np.sqrt(2*(C_T+C_K)/C_KK)
        else:
            vol.iloc[i,j]=np.nan
vol.iloc[0,:]=np.nan           ###########엉망진창
vol.iloc[:,0]=np.nan
vol.iloc[:,24]=np.nan
for i in range(1,24):
    empty= vol.iloc[0:,i].dropna()
    x_points=np.array(empty.index, dtype='float64')
    y_points=np.array(empty.values)
    lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
    for j in range(0,1095):
        if lininter(j) >=0:
            vol.iloc[j,i]=lininter(j)
        else:
            vol.iloc[j,i]=0
            
            
#%% Call option inter&extrapolation : Cubic Spline
            '''linear로 하면 오류가 나기에 cubic으로 시도'''
buff= pd.read_excel("local_vol.xlsx", skip_footer=4) 
buff[1:]= buff[1:].replace(0, np.nan)

call_index=range(0, 365*3)  #시간은 3년으로 늘림
call= pd.DataFrame(columns= buff.columns, index=call_index).replace(np.nan, True)*buff
call[1:]= call[1:].replace(0, np.nan) #잔존만기가 0일 때의 0값은 True값이므로 제외하고 replace함       
         
empty=call.iloc[20,:].dropna()
x_points=np.array(empty.index, dtype='float64')
y_points=np.array(empty.values)
lininter = sp.interpolate.CubicSpline(x_points, y_points, bc_type='natural')
call.iloc[20,:]=lininter(np.linspace(270, 330, 25))

empty=call.iloc[48,:].dropna()
x_points=np.array(empty.index, dtype='float64')
y_points=np.array(empty.values)
lininter = sp.interpolate.CubicSpline(x_points, y_points, bc_type='natural')
call.iloc[48,:]=lininter(np.linspace(270, 330, 25))

for i in range(0,25):
    empty= call.iloc[0:,i].dropna()
    x_points=np.array(empty.index, dtype='float64')
    y_points=np.array(empty.values)
    lininter = sp.interpolate.CubicSpline(x_points, y_points, bc_type='natural')
    for j in range(0,1095):
        if lininter(j) >=0:
            call.iloc[j,i]=lininter(j)
        else:
            call.iloc[j,i]=0

#%% Call - Cubic 후에 local_volatility estimate

vol=call.copy()
for i in range(1, 1095):
    for j in range(1,24):
        C_T= (call.iloc[i,j]-call.iloc[i-1,j])
        C_K=(0.0165*call.columns[j])*(call.iloc[i,j]-call.iloc[i,j-1])/2.5
        C_KK=(call.columns[j]**2)*(call.iloc[i,j+1]-2*call.iloc[i,j]+call.iloc[i,j-1])/(2.5**2)
        if C_KK !=0:
            vol.iloc[i,j]=np.sqrt(2*(C_T+C_K)/C_KK)
        else:
            vol.iloc[i,j]=np.nan
            #음수값이 많이 나옴.. Cubic Spline 함수가 증가하는 함수가 아니기에 발생.
            #뒤로 갈수록 옵션 가격이 0으로 뜨는게 많음..
vol.iloc[0,:]=np.nan
vol.iloc[:,0]=np.nan
vol.iloc[:,24]=np.nan
for i in range(1,24):
    empty= vol.iloc[0:,i].dropna()
    x_points=np.array(empty.index, dtype='float64')
    y_points=np.array(empty.values)
    lininter = sp.interpolate.CubicSpline(x_points, y_points, bc_type='natural')
    for j in range(0,1095):
        if lininter(j) >=0:
            vol.iloc[j,i]=lininter(j)
        else:
            vol.iloc[j,i]=0   #음수면 다 0으로 바꿨더니 망함... 

''' 여기까지가 Call을 먼저 추정하고, vol을 추정한 것!! 결과가 매우매우 안 좋음!!'''

#%% volatility 먼저 추정!!

buff.index=buff.index/365
vol=buff.copy()
for i in range(1, 7):
    for j in range(1,24):
        C_T= (buff.iloc[i,j]-buff.iloc[i-1,j])/(buff.index[i]-buff.index[i-1])
        C_K=(0.0165*buff.columns[j])*(buff.iloc[i,j]-buff.iloc[i,j-1])/2.5
        C_KK=(buff.columns[j]**2)*(buff.iloc[i,j+1]-2*buff.iloc[i,j]+buff.iloc[i,j-1])/(2.5**2)
        if np.round(C_KK,9) !=0: #파이썬새끼..... 0으로 인식했어야지...왜 0으로 인식을 안 해가지고...
            vol.iloc[i,j]=2*np.sqrt((C_T+C_K)/C_KK)
        else:
            vol.iloc[i,j]=np.nan
vol.iloc[0,:]=np.nan
vol.iloc[:,0]=np.nan
vol.iloc[:,24]=np.nan
'''그러나 여전히 음수를 가진 변동성이 계속 존재함... 무시하고 진행하면..'''
''' 음수를 다 nan값으로 바꿔주고 다시 보간법...'''

localvol_index=range(0, 365*3)  #시간은 3년으로 늘림
localvol= pd.DataFrame(columns= buff.columns, index=localvol_index).replace(np.nan, True)*vol

empty=localvol.iloc[20,:].dropna()
x_points=np.array(empty.index, dtype='float64')
y_points=np.array(empty.values)
lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
localvol.iloc[20,:]=lininter(np.linspace(270, 330, 25))

empty=localvol.iloc[83,:].dropna()
x_points=np.array(empty.index, dtype='float64')
y_points=np.array(empty.values)
lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
localvol.iloc[83,:]=lininter(np.linspace(270, 330, 25))

for i in range(0,25):
    empty= localvol.iloc[0:,i].dropna()
    x_points=np.array(empty.index, dtype='float64')
    y_points=np.array(empty.values)
    lininter = sp.interpolate.interp1d(x_points, y_points, fill_value='extrapolate')
    for j in range(0,1095):
        localvol.iloc[j,i]=lininter(j)
'''좀 더 깔끔해보이긴 하지만... 그래도 여전히 변동성이 음의 값을 갖는다'''


#%%

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x=[]
y=[]
z=[]
for i in range(7):
    for j in range(25):
        x.append(vol.index[i])
        y.append(vol.columns[j])
        z.append(vol.iloc[i,j])
empty=np.isfinite(z)
x=(pd.Series(x)*empty).replace(0,np.nan).dropna()
y=(pd.Series(y)*empty).replace(0,np.nan).dropna()
z=(pd.Series(z)*empty).replace(0,np.nan).dropna()
#y=y/1000
f=sp.interpolate.interp2d(x,y,z, kind='linear')

xnew=np.linspace(0,3, 100)
ynew=np.linspace(270,330, 100)
znew=f(xnew,ynew)

a,b=np.meshgrid(xnew, ynew)
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(a,b,f(xnew,ynew), rstride=2, cstride=2, cmap=cm.jet, alpha=0.7, linewidth=0.25) 

#%%  진짜 코드!!!
''' 여기가 진짜요!!!!!!!!!!!!!! '''

buff= pd.read_excel("local_vol.xlsx", skip_footer=4) #쓸데 없는 거 버리고..
buff=buff.iloc[1:]
buff[1:]= buff[1:].replace(0, np.nan)

buff.index=buff.index/365
vol=buff.copy()
for i in range(1, 7):
    for j in range(1,24):
        C_T= (buff.iloc[i,j]-buff.iloc[i-1,j])/(buff.index[i]-buff.index[i-1])
        C_K=(0.0165*buff.columns[j])*(buff.iloc[i,j]-buff.iloc[i,j-1])/2.5
        C_KK=(buff.columns[j]**2)*(buff.iloc[i,j+1]-2*buff.iloc[i,j]+buff.iloc[i,j-1])/(2.5**2)
        if np.round(C_KK,9) !=0: #파이썬새끼..... 0으로 인식했어야지...왜 0으로 인식을 안 해가지고...
            vol.iloc[i,j]=2*np.sqrt((C_T+C_K)/C_KK)
        else:
            vol.iloc[i,j]=np.nan
vol.iloc[0,:]=np.nan
vol.iloc[:,0]=np.nan
vol.iloc[:,24]=np.nan



vol=vol.dropna(axis=1, how='all')
vol=vol.dropna(axis=0, how='all')
vol=vol.dropna(axis=1)

vol_index=np.arange(0,1096)/365 #시간은 3년으로 늘림
locvol= pd.DataFrame(columns= buff.columns, index=vol_index).replace(np.nan, True)*vol
#locvol=locvol.interpolate(method='pchip', limit_direction='both')
#locvol=locvol.interpolate(method='pchip', limit_direction='both', axis=1)

locvol=locvol.interpolate(method='pchip', limit_direction='both')
locvol=locvol.interpolate(method='pchip', limit_direction='both', axis=1)

#xnew=np.linspace(0,3, 100)
xnew=np.arange(0,1096)/365
#ynew=np.linspace(270,330, 100)
ynew=np.linspace(270.0, 330.0, 25)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
#local_vol = local_vol.dropna(axis =1)
#xnew = xnew[:,1:]
#tnew = tnew[:,1:]
surf = ax.plot_surface(x, y, locvol.T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#%%

x=[]
y=[]
z=[]
for i in range(len(vol)):
    for j in range(7):
        x.append(vol.index[i])
        y.append(vol.columns[j])
        z.append(vol.iloc[i,j])
empty=np.isfinite(z)
x=(pd.Series(x)*empty).replace(0,np.nan).dropna()
y=(pd.Series(y)*empty).replace(0,np.nan).dropna()
z=(pd.Series(z)*empty).replace(0,np.nan).dropna()

f=sp.interpolate.interp2d(y,x,z, kind='quintic')



x=np.array(vol.index)
y=np.array(vol.columns)
z=vol.values



f=sp.interpolate.interp2d(y,x,z, fill_value=1)

xnew=np.linspace(0,3, 100)
ynew=np.linspace(270,330, 100)
znew=f(ynew,xnew)

a,b=np.meshgrid(ynew, xnew)
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(a,b,f(ynew,xnew), rstride=2, cstride=2, cmap=cm.jet, alpha=0.7, linewidth=0.25) 
