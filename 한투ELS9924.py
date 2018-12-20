# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:46:50 2018

@author: 우람
"""


import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import numpy as np
import datetime as dt 
import matplotlib.pyplot as plt
import scipy as sp
import math
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\수치해석')


sig1=0.27  #변동성(분산 아님) 0.27이었
sig2=0.217 #원래 0.217 이었음
rho=0.3503 #두 자산 간의 상관관계 원래 0.3503
r=0.0165  #무위험 이자율, CD91일물
K0=[12004.51,3340.93] #HSCEI, EuroStoxx50
S1max=K0[0]*2
S1min=0
S2max=K0[1]*2
S2min=0
F=100 #액면가
T=3 #만기
coupon=[0.027, 0.054, 0.081, 0.108, 0.135, 0.162] #쿠폰
K=[0.9, 0.9, 0.85, 0.85, 0.80, 0.80] #조기상환율?? 몇 %이상이어야 조기상환이 되는지
KI=0.50 #낙인 
pp=50 #6개월을 몇 번으로 쪼개는지
Nt=6*pp #조기상환기회가 6번 있음. 만기를 몇 번 쪼갤 것인지
Nx=100 #첫 번째 기초자산을 몇 번 쪼갤 것인지
Ny=100 #두 번째 기초자산을 몇 번 쪼갤 것인지
Nx0=round(Nx/2) #처음 가격을 노드에 찍은 것
Ny0=round(Ny/2) 
h=T/Nt #dt
k1=(S1max-S1min)/Nx #dx
k2=(S2max-S2min)/Ny #dy
q=0 #배당.. 없음 



#%% 주가 시뮬레이션

count_simulation=100000
timestep=Nt+1
t=np.linspace(0,3,timestep)
W1=np.random.normal(0,1,(count_simulation,int(timestep)))*np.sqrt(t[1]-t[0])
W1[:,0]=0
W1=W1.cumsum(axis=1)
W2=np.sqrt(t[1]-t[0])*(rho*np.random.normal(0,1,(count_simulation,int(timestep)))+np.sqrt(1-rho**2)*np.random.normal(0,1,(count_simulation,int(timestep))))
W2[:,0]=0
W2=W2.cumsum(axis=1)
stock1=K0[0]*np.exp((r-(sig1**2)/2)*t+sig1*W1)
stock2=K0[1]*np.exp((r-(sig2**2)/2)*t+sig2*W2)
count_KI=((stock1.min(axis=1)<=K0[0]*KI) + (stock2.min(axis=1)<=K0[1]*KI)).sum()
percent_KI=count_KI/count_simulation #인생동안 낙인 친 확률


#%%
#낙인 안 쳤을 때!!!

u = np.zeros((Nx+1,Ny+1, Nt+1));
u[math.ceil(Nx0*KI):, math.ceil(Ny0*KI):,0] = F*(1+coupon[5]);
for i in range(math.ceil(Nx0*KI)):
    for j in range(Ny+1):
        u[i, j, 0]=F*np.minimum(i/Nx0, j/Ny0)
        u[j, i, 0]=F*np.minimum(i/Nx0, j/Ny0)
        
        

def a_n(n,l=99,q=0, sig1=0.25,k1=70):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))
def b_n(n,l=99,q=0, sig1=0.25,r=0.02,k1=70, h=0.01):
    return 1/h+ (sig1**2)*((n*k1)**2)/(k1**2)+ (r*n*k1)/k1+0.5*r
def c_n(n,l=99,q=0, sig1=0.25, k1=70,r=0.02):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))- (r*n*k1)/k1
def d_n(n,l=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(u[n+1,l+1,m]-u[n+1,l,m]-u[n,l+1,m]+u[n,l,m])/(k1**2)+u[n,l,m]/h
    return x

def a_l(l,n=99,q=0, sig2=0.2,k2=60):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))
def b_l(l,n=99,q=0, sig2=0.2,r=0.02,k2=60, h=0.01):
    return 1/h+ (sig2**2)*((l*k2)**2)/(k2**2)+ (r*l*k2)/k2+0.5*r
def c_l(l,n=99,q=0, sig2=0.2, k2=60, r=0.02):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))- (r*l*k2)/k2
def d_l(l,n=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(buff[n+1,l+1,m+1]-buff[n+1,l,m+1]-buff[n,l+1,m+1]+buff[n,l,m+1])/(k2**2)+buff[n,l,m+1]/h
    return x        

def TDMAsolver(a, b, c, d):

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
               
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc    

#%%
     
buff=u.copy()
for m in range(Nt):
    if m==pp:
        u[math.ceil(Nx0*K[4]):, math.ceil(Ny0*K[4]):,pp] = F*(1+coupon[4]);
    elif m==2*pp:
        u[math.ceil(Nx0*K[3]):, math.ceil(Ny0*K[3]):,2*pp] = F*(1+coupon[3]);
    elif m==3*pp:
        u[math.ceil(Nx0*K[2]):, math.ceil(Ny0*K[2]):,3*pp] = F*(1+coupon[2]);
    elif m==4*pp:
        u[math.ceil(Nx0*K[1]):, math.ceil(Ny0*K[1]):,4*pp] = F*(1+coupon[1]);
    elif m==5*pp:
        u[math.ceil(Nx0*K[0]):, math.ceil(Ny0*K[0]):,5*pp] = F*(1+coupon[0]);
    for l in range(1,Ny):
        a=list()
        b=list()
        c=list()
        d=list()
        for n in range(len(u)):
            a.append(a_n(n=n,l=l,k1=k1))
            b.append(b_n(n=n,l=l,k1=k1))
            c.append(c_n(n=n,l=l,k1=k1))
        for n in range(len(u)):
            if n==Nx:                               
                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*u[n,l+1,m]-u[n-1,l+1,m]-(2*u[n,l,m]-u[n-1,l,m])- u[n,l+1,m]+u[n,l,m])/(k1**2)+u[n,l,m]/h)
            else:
                d.append(d_n(n=n,l=l,m=m,k1=k1))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        buff[:,l,m+1]= TDMAsolver(a[1:],b,c[:-1],d)
        buff[:,0,m+1]=2*buff[:,1,m+1]-buff[:,2,m+1]
        buff[:,-1,m+1]=2*buff[:,-2,m+1]-buff[:,-3,m+1]
        
    for n in range(1,Nx):
        a=list()
        b=list()
        c=list()
        d=list()
        for l in range(len(u)):
            a.append(a_l(l=l,n=n,k2=k2))
            b.append(b_l(l=l,n=n,k2=k2))
            c.append(c_l(l=l,n=n,k2=k2))
        for l in range(len(u)):
            if l==Ny:
                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*buff[n+1,l,m+1]-buff[n+1,l-1,m+1]-buff[n+1,l,m+1]-(2*buff[n,l,m+1]-buff[n,l-1,m+1])+buff[n,l,m+1])/(k2**2)+buff[n,l,m+1]/h)
            else:
                d.append(d_l(l=l,n=n,m=m,k2=k2))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        u[n,:,m+1]= TDMAsolver(a[1:],b,c[:-1],d)
        u[0,:,m+1]=2*u[1,:,m+1]-u[2,:,m+1]
        u[-1,:,m+1]=2*u[-2,:,m+1]-u[-3,:,m+1]
empty=copy.deepcopy(u)
    
'''낙인 안 칠 때 ELS Price'''
print("낙인 안 칠 때 ELS가격은",u[Nx0,Ny0,Nt])
NKI_ELS_Price=u[Nx0,Ny0,Nt]*(1-percent_KI) 
#print("copy", empty[Nx0,Ny0,Nt])
print("NKI_ELS_Price(확률까지 곱한 결과)", NKI_ELS_Price)

        
#%%  그림 그리기!!

xnew=np.linspace(0, S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, u[:,:,Nt].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       



#%%  낙인 쳤을 때 ELS 가격 구하기


def a_n(n,l=99,q=0, sig1=0.25,k1=70):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))
def b_n(n,l=99,q=0, sig1=0.25,r=0.02,k1=70, h=0.01):
    return 1/h+ (sig1**2)*((n*k1)**2)/(k1**2)+ (r*n*k1)/k1+0.5*r
def c_n(n,l=99,q=0, sig1=0.25, k1=70,r=0.02):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))- (r*n*k1)/k1
def d_n(n,l=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(u_KI[n+1,l+1,m]-u_KI[n+1,l,m]-u_KI[n,l+1,m]+u_KI[n,l,m])/(k1**2)+u_KI[n,l,m]/h
    return x

def a_l(l,n=99,q=0, sig2=0.2,k2=60):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))
def b_l(l,n=99,q=0, sig2=0.2,r=0.02,k2=60, h=0.01):
    return 1/h+ (sig2**2)*((l*k2)**2)/(k2**2)+ (r*l*k2)/k2+0.5*r
def c_l(l,n=99,q=0, sig2=0.2, k2=60,r=0.02):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))- (r*l*k2)/k2
def d_l(l,n=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(buff[n+1,l+1,m+1]-buff[n+1,l,m+1]-buff[n,l+1,m+1]+buff[n,l,m+1])/(k2**2)+buff[n,l,m+1]/h
    return x


u_KI = np.zeros((Nx+1,Ny+1, Nt+1));
u_KI[math.ceil(Nx0*K[5]):, math.ceil(Ny0*K[5]):,0] = F*(1+coupon[5]);
for i in range(math.ceil(Nx0*K[5])):
    for j in range(Ny+1):
        u_KI[i, j, 0]=F*np.minimum(i/Nx0, j/Ny0)
        u_KI[j, i, 0]=F*np.minimum(i/Nx0, j/Ny0)

buff=u_KI.copy()
for m in range(Nt):
    if m==pp:
        u_KI[math.ceil(Nx0*K[4]):, math.ceil(Ny0*K[4]):,pp] = F*(1+coupon[4]);
    elif m==2*pp:
        u_KI[math.ceil(Nx0*K[3]):, math.ceil(Ny0*K[3]):,2*pp] = F*(1+coupon[3]);
    elif m==3*pp:
        u_KI[math.ceil(Nx0*K[2]):, math.ceil(Ny0*K[2]):,3*pp] = F*(1+coupon[2]);
    elif m==4*pp:
        u_KI[math.ceil(Nx0*K[1]):, math.ceil(Ny0*K[1]):,4*pp] = F*(1+coupon[1]);
    elif m==5*pp:
        u_KI[math.ceil(Nx0*K[0]):, math.ceil(Ny0*K[0]):,5*pp] = F*(1+coupon[0]);
    for l in range(0,Ny):
        a=list()
        b=list()
        c=list()
        d=list()
        for n in range(len(u)):
            a.append(a_n(n=n,l=l,k1=k1))
            b.append(b_n(n=n,l=l,k1=k1))
            c.append(c_n(n=n,l=l,k1=k1))
        for n in range(len(u)):
            if n==Nx:                               
                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*u_KI[n,l+1,m]-u_KI[n-1,l+1,m]-(2*u_KI[n,l,m]-u_KI[n-1,l,m])- u_KI[n,l+1,m]+u_KI[n,l,m])/(k1**2)+u_KI[n,l,m]/h)
            else:
                d.append(d_n(n=n,l=l,m=m,k1=k1))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        buff[:,l,m+1]= TDMAsolver(a[1:],b,c[:-1],d)
        buff[:,0,m+1]=2*buff[:,1,m+1]-buff[:,2,m+1]
        buff[:,-1,m+1]=2*buff[:,-2,m+1]-buff[:,-3,m+1]
        
    for n in range(0,Nx):
        a=list()
        b=list()
        c=list()
        d=list()
        for l in range(len(u)):
            a.append(a_l(l=l,n=n,k2=k2))
            b.append(b_l(l=l,n=n,k2=k2))
            c.append(c_l(l=l,n=n,k2=k2))
        for l in range(len(u)):
            if l==Ny:
                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*buff[n+1,l,m+1]-buff[n+1,l-1,m+1]-buff[n+1,l,m+1]-(2*buff[n,l,m+1]-buff[n,l-1,m+1])+buff[n,l,m+1])/(k2**2)+buff[n,l,m+1]/h)
            else:
                d.append(d_l(l=l,n=n,m=m,k2=k2))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        u_KI[n,:,m+1]= TDMAsolver(a[1:],b,c[:-1],d)
        u_KI[0,:,m+1]=2*u_KI[1,:,m+1]-u_KI[2,:,m+1]
        u_KI[-1,:,m+1]=2*u_KI[-2,:,m+1]-u_KI[-3,:,m+1]
empty1=copy.deepcopy(u_KI)

'''낙인 쳤을 때 ELS Price'''
print("낙인 칠 때 ELS가격은",u_KI[Nx0,Ny0,Nt])
KI_ELS_Price=u_KI[Nx0,Ny0,Nt]*percent_KI
#print("copy", empty1[Nx0,Ny0,Nt])
print("NKI_ELS_Price(확률까지 곱한 결과)", KI_ELS_Price)

#%% 낙인 쳤을 때 그림 그리기
        
xnew=np.linspace(0,S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, u_KI[:,:,Nt].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       

#%%

ELS_Price= NKI_ELS_Price+KI_ELS_Price
ELS=u*(1-percent_KI)+u_KI*percent_KI
print("ELS 가격은 " , ELS_Price)
#print("KI쳤을 때 ELS가격은", u_KI[Nx0,Ny0,Nt])
#print("KI 안 쳤을 때 ELS가격은", u[Nx0,Ny0,Nt])
print("KI칠 확률", percent_KI)

#%% Greek

print("Delta X:" , (u[Nx0+1,Ny0,Nt]*(1-percent_KI)+u_KI[Nx0+1,Ny0,Nt]*percent_KI - ELS_Price)/k1)
print("Delta Y:" , (u[Nx0,Ny0+1,Nt]*(1-percent_KI)+u_KI[Nx0,Ny0+1,Nt]*percent_KI - ELS_Price)/k2)
print("Gamma X:" , (u[Nx0+1,Ny0,Nt]*(1-percent_KI)+u_KI[Nx0+1,Ny0,Nt]*percent_KI - 2*ELS_Price+ u[Nx0-1,Ny0,Nt]*(1-percent_KI)+u_KI[Nx0-1,Ny0,Nt]*percent_KI)/(k1**2))
print("Gamma Y:" , (u[Nx0,Ny0+1,Nt]*(1-percent_KI)+u_KI[Nx0,Ny0+1,Nt]*percent_KI - 2*ELS_Price+ u[Nx0,Ny0-1,Nt]*(1-percent_KI)+u_KI[Nx0,Ny0-1,Nt]*percent_KI)/k2**2)
print("Vega X:" "(sig1을 0.27에서 0.37로 변경시키니 가격이 98.14143304775145이 나옴)", (98.14143304775145-99.18094341075752)/0.1 )
print("Vega Y:" "(sig2을 0.217에서 0.317로 변경시키니 가격이 98.28416679374072이 나옴)", (98.28416679374072-99.18094341075752)/0.1 )
print("Sensitivity of Rho:", "(Rho를 0.3503에서 0.4503으로 변경시키니 가격이 99.17298732478966나옴.)", (99.17298732478966-99.18094341075752)/0.1)

empty=pd.DataFrame(u[:,Ny0,Nt]*(1-percent_KI)+u_KI[:,Ny0,Nt]*(percent_KI))
deltaX=pd.DataFrame((empty.shift(-1)-empty)/k1)
empty=pd.DataFrame(u[Nx0,:,Nt]*(1-percent_KI)+u_KI[Nx0,:,Nt]*(percent_KI))
deltaY=pd.DataFrame((empty.shift(-1)-empty)/k2)

deltaX.plot()
deltaY.plot()


empty=pd.DataFrame(ELS[:,Ny0,Nt]).shift(-1)- 2*pd.DataFrame(ELS[:,Ny0,Nt]) +pd.DataFrame(ELS[:,Ny0,Nt]).shift(1)
gammaX=empty/(k1**2)
empty=pd.DataFrame(ELS[Nx0,:,Nt]).shift(-1)- 2*pd.DataFrame(ELS[Nx0,:,Nt]) +pd.DataFrame(ELS[Nx0,:,Nt]).shift(1)
gammaY=empty/(k2**2)

gammaX.plot()
gammaY.plot()


empty=pd.DataFrame(ELS[Nx0,Ny0,:]).shift(-1)-pd.DataFrame(ELS[Nx0,Ny0,:])
theta=empty/h
theta.plot()
#%% ELS 그래프
xnew=np.linspace(0,S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, ELS[:,:,Nt].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       