# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:52:32 2018

@author: 우람
"""

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
r=0.0165  #무위험 이자율
K0=[12004.51,3340.93] #HSCEI, EuroStoxx50
S1max=K0[0]*2
S1min=0
S2max=K0[1]*2
S2min=0
F=10000 #액면가
T=3 #만기
coupon=[0.027, 0.054, 0.081, 0.108, 0.135, 0.162] #쿠폰
K=[0.9, 0.9, 0.85, 0.85, 0.80, 0.80] #조기상환율?? 몇 %이상이어야 조기상환이 되는지
KI=0.50  
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

u = np.zeros((Nt+1,Nx+1,Ny+1));
u[0,math.ceil(Nx0*KI):, math.ceil(Ny0*KI):] = F*(1+coupon[5]);
for i in range(math.ceil(Nx0*KI)):
    for j in range(Ny+1):
        u[0,i, j]=F*np.minimum(i/Nx0, j/Ny0)
        u[0,j, i]=F*np.minimum(i/Nx0, j/Ny0)
        
        

def a_n(n,l=99,q=0, sig1=0.25,k1=70):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))
def b_n(n,l=99,q=0, sig1=0.25,r=0.02,k1=70, h=0.01):
    return 1/h+ (sig1**2)*((n*k1)**2)/(k1**2)+ (r*n*k1)/k1+0.5*r
def c_n(n,l=99,q=0, sig1=0.25, k1=70,r=0.02):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))- (r*n*k1)/k1
def d_n(n,l=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(u[m,n+1,l+1]-u[m,n+1,l]-u[m,n,l+1]+u[m,n,l])/(k1**2)+u[m,n,l]/h
    return x

def a_l(l,n=99,q=0, sig2=0.2,k2=60):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))
def b_l(l,n=99,q=0, sig2=0.2,r=0.02,k2=60, h=0.01):
    return 1/h+ (sig2**2)*((l*k2)**2)/(k2**2)+ (r*l*k2)/k2+0.5*r
def c_l(l,n=99,q=0, sig2=0.2, k2=60, r=0.02):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))- (r*l*k2)/k2
def d_l(l,n=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(buff[m+1,n+1,l+1]-buff[m+1,n+1,l]-buff[m+1,n,l+1]+buff[m+1,n,l])/(k2**2)+buff[m+1,n,l]/h
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
        u[pp,math.ceil(Nx0*K[4]):, math.ceil(Ny0*K[4]):] = F*(1+coupon[4]);
    elif m==2*pp:
        u[2*pp,math.ceil(Nx0*K[3]):, math.ceil(Ny0*K[3]):] = F*(1+coupon[3]);
    elif m==3*pp:
        u[3*pp,math.ceil(Nx0*K[2]):, math.ceil(Ny0*K[2]):] = F*(1+coupon[2]);
    elif m==4*pp:
        u[4*pp,math.ceil(Nx0*K[1]):, math.ceil(Ny0*K[1]):] = F*(1+coupon[1]);
    elif m==5*pp:
        u[5*pp,math.ceil(Nx0*K[0]):, math.ceil(Ny0*K[0]):] = F*(1+coupon[0]);
    for l in range(1,Ny):
        a=list()
        b=list()
        c=list()
        d=list()
        for n in range(1,len(u.T)-1):
            a.append(a_n(n=n,l=l,k1=k1))
            b.append(b_n(n=n,l=l,k1=k1))
            c.append(c_n(n=n,l=l,k1=k1))
            d.append(d_n(n=n,l=l,m=m,k1=k1))
#        for n in range(1,len(u.T)-1):
#            if n==Nx:                   
#                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*u[m,n,l+1]-u[m,n-1,l+1]-(2*u[m,n,l]-u[m,n-1,l])- u[m,n,l+1]+u[m,n,l])/(k1**2)+u[m,n,l]/h)
#            else:
#                d.append(d_n(n=n,l=l,m=m,k1=k1))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        buff[m+1,1:Nx,l]= TDMAsolver(a[1:],b,c[:-1],d)
    buff[m+1,:,0]=2*buff[m+1,:,1]-buff[m+1,:,2]
    buff[m+1,:,-1]=2*buff[m+1,:,-2]-buff[m+1,:,-3]
    buff[m+1,0,:]=2*buff[m+1,1,:]-buff[m+1,1,:]
    buff[m+1,-1,:]=2*buff[m+1,-2,:]-buff[m+1,-3,:]
        
    for n in range(1,Nx):
        a=list()
        b=list()
        c=list()
        d=list()
        for l in range(1,len(u.T)-1):
            a.append(a_l(l=l,n=n,k2=k2))
            b.append(b_l(l=l,n=n,k2=k2))
            c.append(c_l(l=l,n=n,k2=k2))
            d.append(d_l(l=l,n=n,m=m,k2=k2))
#        for l in range(1,len(u.T)-1):
#            if l==Ny:
#                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*buff[m+1,n+1,l]-buff[m+1,n+1,l-1]-buff[m+1,n+1,l]-(2*buff[m+1,n,l]-buff[m+1,n,l-1])+buff[m+1,n,l])/(k2**2)+buff[m+1,n,l]/h)
#            else:
#                d.append(d_l(l=l,n=n,m=m,k2=k2))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        u[m+1,n,1:Ny]= TDMAsolver(a[1:],b,c[:-1],d)
    u[m+1,0,:]=2*u[m+1,1,:]-u[m+1,2,:]
    u[m+1,-1,:]=2*u[m+1,-2,:]-u[m+1,-3,:]
    u[m+1,:,0]=2*u[m+1,:,1]-u[m+1,:,2]
    u[m+1,:,-1]=2*u[m+1,:,-2]-u[m+1,:,-3]   
empty=copy.deepcopy(u)
    
'''낙인 안 칠 때 ELS Price'''
print("낙인 안 칠 때 ELS가격은",u[Nt,Nx0,Ny0])
NKI_ELS_Price=u[Nt,Nx0,Ny0]*(1-percent_KI) 
print("copy", empty[Nt,Nx0,Ny0])
print("NKI_ELS_Price", NKI_ELS_Price)

        
#%%  그림 그리기!!

xnew=np.linspace(0, S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, u[300,:,:].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       



#%%  낙인 쳤을 때 ELS 가격 구하기


def a_n(n,l=99,q=0, sig1=0.25,k1=70):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))
def b_n(n,l=99,q=0, sig1=0.25,r=0.02,k1=70, h=0.01):
    return 1/h+ (sig1**2)*((n*k1)**2)/(k1**2)+ (r*n*k1)/k1+0.5*r
def c_n(n,l=99,q=0, sig1=0.25, k1=70,r=0.02):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))- (r*n*k1)/k1
def d_n(n,l=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(u_KI[m,n+1,l+1]-u_KI[m,n+1,l]-u_KI[m,n,l+1]+u_KI[m,n,l])/(k1**2)+u_KI[m,n,l]/h
    return x

def a_l(l,n=99,q=0, sig2=0.2,k2=60):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))
def b_l(l,n=99,q=0, sig2=0.2,r=0.02,k2=60, h=0.01):
    return 1/h+ (sig2**2)*((l*k2)**2)/(k2**2)+ (r*l*k2)/k2+0.5*r
def c_l(l,n=99,q=0, sig2=0.2, k2=60,r=0.02):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))- (r*l*k2)/k2
def d_l(l,n=99, m=0,q=0, sig1=0.25, sig2=0.2, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(buff[m+1,n+1,l+1]-buff[m+1,n+1,l]-buff[m+1,n,l+1]+buff[m+1,n,l])/(k2**2)+buff[m+1,n,l]/h
    return x


u_KI = np.zeros((Nt+1,Nx+1,Ny+1));
u_KI[0,math.ceil(Nx0*K[5]):, math.ceil(Ny0*K[5]):] = F*(1+coupon[5]);
for i in range(math.ceil(Nx0*K[5])):
    for j in range(Ny+1):
        u_KI[0,i, j]=F*np.minimum(i/Nx0, j/Ny0)
        u_KI[0,j, i]=F*np.minimum(i/Nx0, j/Ny0)

buff=u_KI.copy()
for m in range(Nt):
    if m==pp:
        u_KI[pp,math.ceil(Nx0*K[4]):, math.ceil(Ny0*K[4]):] = F*(1+coupon[4]);
    elif m==2*pp:
        u_KI[2*pp,math.ceil(Nx0*K[3]):, math.ceil(Ny0*K[3]):] = F*(1+coupon[3]);
    elif m==3*pp:
        u_KI[3*pp,math.ceil(Nx0*K[2]):, math.ceil(Ny0*K[2]):] = F*(1+coupon[2]);
    elif m==4*pp:
        u_KI[4*pp,math.ceil(Nx0*K[1]):, math.ceil(Ny0*K[1]):] = F*(1+coupon[1]);
    elif m==5*pp:
        u_KI[5*pp,math.ceil(Nx0*K[0]):, math.ceil(Ny0*K[0]):] = F*(1+coupon[0]);
    for l in range(0,Ny):
        a=list()
        b=list()
        c=list()
        d=list()
        for n in range(1,len(u.T)-1):
            a.append(a_n(n=n,l=l,k1=k1))
            b.append(b_n(n=n,l=l,k1=k1))
            c.append(c_n(n=n,l=l,k1=k1))
            d.append(d_n(n=n,l=l,m=m,k1=k1))

#        for n in range(1,len(u.T)-1):
#            if n==Nx:                               
#                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*u_KI[m,n,l+1]-u_KI[m,n-1,l+1]-(2*u_KI[m,n,l]-u_KI[m,n-1,l])- u_KI[m,n,l+1]+u_KI[m,n,l])/(k1**2)+u_KI[m,n,l]/h)
#            else:
#                d.append(d_n(n=n,l=l,m=m,k1=k1))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        buff[m+1,1:Nx,l]= TDMAsolver(a[1:],b,c[:-1],d)
    buff[m+1,:,0]=2*buff[m+1,:,1]-buff[m+1,:,2]
    buff[m+1,:,-1]=2*buff[m+1,:,-2]-buff[m+1,:,-3]
    buff[m+1,0,:]=2*buff[m+1,1,:]-buff[m+1,1,:]
    buff[m+1,-1,:]=2*buff[m+1,-2,:]-buff[m+1,-3,:]

    
    for n in range(0,Nx):
        a=list()
        b=list()
        c=list()
        d=list()
        for l in range(1,len(u.T)-1):
            a.append(a_l(l=l,n=n,k2=k2))
            b.append(b_l(l=l,n=n,k2=k2))
            c.append(c_l(l=l,n=n,k2=k2))
            d.append(d_l(l=l,n=n,m=m,k2=k2))
#        for l in range(1,len(u.T)-1):
#            if l==Ny:
#                d.append(0.5*rho*sig1*sig2*n*k1*l*k2*(2*buff[m+1,n+1,l]-buff[m+1,n+1,l-1]-buff[m+1,n+1,l]-(2*buff[m+1,n,l]-buff[m+1,n,l-1])+buff[m+1,n,l])/(k2**2)+buff[m+1,n,l]/h)
#            else:
#                d.append(d_l(l=l,n=n,m=m,k2=k2))
        b[0]=2*a[0]+b[0]
        c[0]=c[0]-a[0]
        a[-1]=a[-1]-c[-1]
        b[-1]=b[-1]+2*c[-1]
        u_KI[m+1,n,1:Ny]= TDMAsolver(a[1:],b,c[:-1],d)
    u_KI[m+1,0,:]=2*u_KI[m+1,1,:]-u_KI[m+1,2,:]
    u_KI[m+1,-1,:]=2*u_KI[m+1,-2,:]-u_KI[m+1,-3,:]
    u_KI[m+1,:,0]=2*u_KI[m+1,:,1]-u_KI[m+1,:,2]
    u_KI[m+1,:,-1]=2*u_KI[m+1,:,-2]-u_KI[m+1,:,-3] 
empty1=copy.deepcopy(u_KI)

'''낙인 쳤을 때 ELS Price'''
print("낙인 칠 때 ELS가격은",u_KI[Nt,Nx0,Ny0])
KI_ELS_Price=u_KI[Nt,Nx0,Ny0]*percent_KI
print("copy", empty1[Nt,Nx0,Ny0])
print("NKI_ELS_Price", KI_ELS_Price)
KI_ELS_Price=u_KI[Nt,Nx0,Ny0]*percent_KI
#%% 낙인 쳤을 때 그림 그리기
        
xnew=np.linspace(0,S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, u_KI[Nt,:,:].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       

#%%

ELS_Price= NKI_ELS_Price+KI_ELS_Price
ELS=u*(1-percent_KI)+u_KI*(percent_KI)

xnew=np.linspace(0,S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, ELS[Nt,:,:].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       


print("ELS 가격은 " , ELS_Price)
print("KI쳤을 때 ELS가격은", u_KI[Nt,Nx0,Ny0])
print("KI 안 쳤을 때 ELS가격은", u[Nt,Nx0,Ny0])
print("KI칠 확률", percent_KI)

#%% Greek
#
#print("Delta X:" , (u[Nt,Nx0+1,Ny0]*(1-percent_KI)+u_KI[Nt,Nx0+1,Ny0]*percent_KI - ELS_Price)/k1)
#print("Delta Y:" , (u[Nt,Nx0,Ny0+1]*(1-percent_KI)+u_KI[Nt,Nx0,Ny0+1]*percent_KI - ELS_Price)/k2)
#print("Gamma X:" , (u[Nx0+1,Ny0,Nt]*(1-percent_KI)+u_KI[Nx0+1,Ny0,Nt]*percent_KI - 2*ELS_Price+ u[Nx0-1,Ny0,Nt]*(1-percent_KI)+u_KI[Nx0-1,Ny0,Nt]*percent_KI)/k1**2)
#print("Gamma Y:" , (u[Nx0,Ny0+1,Nt]*(1-percent_KI)+u_KI[Nx0,Ny0+1,Nt]*percent_KI - 2*ELS_Price+ u[Nx0,Ny0-1,Nt]*(1-percent_KI)+u_KI[Nx0,Ny0-1,Nt]*percent_KI)/k2**2)
#print("Vega X:" "(sig1을 0.27에서 0.37로 변경시키니 가격이 98.14143304775145이 나옴. 원래 가격은 99.1809434.. 였고)", (98.14143304775145-99.18094341075752)/0.1 )
#print("Vega Y:" "(sig2을 0.217에서 0.317로 변경시키니 가격이 98.28416679374072이 나옴. 원래 가격은 99.1809434.. 였고)", (98.28416679374072-99.18094341075752)/0.1 )
#print("Sensitivity of Rho:", "Rho를 0.3503에서 0.4503으로 변경시키니 가격이 99.17298732478966나옴.", (99.17298732478966-99.18094341075752)/0.1)

empty=pd.DataFrame(u[Nt,:,Ny0]*(1-percent_KI)+u_KI[Nt,:,Ny0]*(percent_KI))
deltaX=pd.DataFrame((empty-empty.shift(1))/k1)
empty=pd.DataFrame(u[Nt,Nx0,:]*(1-percent_KI)+u_KI[Nt,Nx0,:]*(percent_KI))
deltaY=pd.DataFrame((empty-empty.shift(1))/k2)

empty=pd.DataFrame(u[Nt,:,Ny0]*(1-percent_KI)+u_KI[Nt,:,Ny0]*(percent_KI))
deltaX=pd.DataFrame((empty-empty.shift(1))/k1)
deltaX.plot()

empty=pd.DataFrame(ELS[Nt,:,Ny0]).shift(-1)- 2*pd.DataFrame(ELS[Nt,:,Ny0]) +pd.DataFrame(ELS[Nt,:,Ny0]).shift(1)
gammaX=empty/(k1**2)
empty=pd.DataFrame(ELS[Nt, Nx0,:]).shift(-1)- 2*pd.DataFrame(ELS[Nt,Nx0,:]) +pd.DataFrame(ELS[Nt,Nx0,:]).shift(1)
gammaY=empty/(k2**2)

gammaX.plot()
gammaY.plot()

#%%

"""헤지 시점 주가를 𝑆0 , 헤지 평가(종가) 주가를 𝑆1 , 각각의 평가금액을 𝑁0, 𝑁1 그리고 𝑆0 에서의 
델타를 𝐷0라 할 때 감마수익은 𝐺(𝑆1) = 𝐷0 ⋅ (𝑆1 − 𝑆0) − 𝑁1 + 𝑁0 로 정의한다"""

""" 선물 롤오버 비용을 절감하기 위해 원월물의 거래량이 충분한 경우 원월물로 헷지했고, 
그렇지 못한 경우엔 근월물로 헷지했다. Euro Stoxx의 경우 9월물 6월물, HSCEI의 경우 9,6,3월물
(그러나 롤오버 비용을 고려하진 않을 것)
(환헷지 하거나 안 하거나 둘 다 있음)
alpha를 0.03으로 맞추면.. HSCEI는 계속 하락할 때만 사고 그러는데.. 그래서 손실 
alpha는 0.01로 해놓으면, 샀다팔았다를 그래도 좀 하면서 헷지수익은 발생하지만 쿠폰수익은 노노..
(예측했던 것보다 변동성이 낮게 나와서.. 샀다팔았다 하면서 수익을 많이 못 먹음"""

import time
from datetime import date
from datetime import timedelta
from datetime import datetime

futures=pd.read_excel('futures data modified.xlsx')
Euro=futures.iloc[:,0:3]
Euro['index']=0
for i in range(len(Euro)):
    Euro['index'][i]=(1095-(Euro.date[i]-Euro.date[186]).days)
Euro=Euro.dropna()
Euro['index']=Euro['index']/3.65
Euro.index=Euro['index']
Euro=Euro[:Nt]
Euro=Euro.iloc[:,1:]
Euro=Euro.iloc[:,:].astype(np.float64)
#Euro=Euro.iloc[:,0:2]
Euro['nodeprice']=Ny0*(Euro['price']/Euro['price'].iloc[-1])

Hscei=pd.DataFrame(futures.iloc[:,3:6])
Hscei.columns=['date','price','exchange']
Hscei['index']=0
for i in range(len(Hscei)):
    Hscei['index'][i]=(1095-(Hscei.date[i]-Hscei.date[180]).days)
Hscei=Hscei.dropna()
Hscei['index']=Hscei['index']/3.65
Hscei.index=Hscei['index']
Hscei=Hscei[:Nt]
Hscei=Hscei.iloc[:,1:]
Hscei=Hscei.iloc[:,:].astype(np.float64)
#Hscei=Hscei.iloc[:,0:2]
Hscei['nodeprice']=Nx0*(Hscei['price']/Hscei['price'].iloc[-1])

#pan_ELS=pd.Panel(ELS)


Euroex_avg=Euro['exchange'].sum()/len(Euro)
Hscex_avg=Hscei['exchange'].sum()/len(Hscei)
#%% delta Hedge W/ MS 3%

alpha=0.03
'''Daily Hedge를 원하면 -0.01로 해둬라'''

eurodelta=-(pd.DataFrame(ELS[:,Nx0,:])-pd.DataFrame(ELS[:,Nx0,:]).shift(1, axis=1))/(k2) #X축을 고정시키고 시간과 Y만 고려
x=np.array(eurodelta.index)
y=np.array(eurodelta.columns)
z=eurodelta.values
interdelta=sp.interpolate.interp2d(y,x,z) 
interdelta(Ny0+1,Nt) #shift해서 한 칸씩 밀리니까... 

Eurorec=Euro.iloc[:,1:].copy()
Eurorec['delta']=0
Eurorec=Eurorec.iloc[:,:].astype(np.float64)
Eurorec.loc[Nt]['delta']=interdelta(Ny0+1,Nt)


Euro['ret']=Euro['price']/Euro['price'].iloc[-1]-1
Euro['whether']=(abs(Euro['ret'])>alpha).replace(False,np.nan)

Eurohedge=Euro.iloc[::-1].dropna().iloc[:1,:]
Eurorec.loc[Eurohedge.index[0]]['delta']=interdelta(Eurohedge['nodeprice']+1,Eurohedge.index[0])

for i in range(len(Euro)):
    try:
        Euro_buff=Euro.copy()
        Euro_buff=Euro.loc[:Eurohedge.index[0]]
        Euro_buff=Euro_buff.iloc[:len(Euro_buff)-1,:]
        Euro_buff['whether']=np.array(pd.DataFrame(abs(Euro_buff['price'].values/Eurohedge['price'].values-1)>alpha).replace(False,np.nan))
        Eurohedge=Euro_buff.iloc[::-1].dropna().iloc[:1,:]
        Eurorec.loc[Eurohedge.index[0]]['delta']=interdelta(Eurohedge['nodeprice']+1,Eurohedge.index[0])
    except:
        break

#%%
hsceidelta=-(pd.DataFrame(ELS[:,:,Ny0])-pd.DataFrame(ELS[:,:,Ny0]).shift(1, axis=1))/(k1) #Y축을 고정시키고 시간과 Y만 고려
x=np.array(hsceidelta.index)
y=np.array(hsceidelta.columns)
z=hsceidelta.values
interdelta=sp.interpolate.interp2d(y,x,z) 
interdelta(Nx0+1,Nt) #shift해서 한 칸씩 밀리니까... 

Hsceirec=Hscei.iloc[:,1:].copy()
Hsceirec['delta']=0
Hsceirec=Hsceirec.iloc[:,:].astype(np.float64)
Hsceirec.loc[Nt]['delta']=interdelta(Nx0+1,Nt)


Hscei['ret']=Hscei['price']/Hscei['price'].iloc[-1]-1
Hscei['whether']=(abs(Hscei['ret'])>alpha).replace(False,np.nan)

Hsceihedge=Hscei.iloc[::-1].dropna().iloc[:1,:]
Hsceirec.loc[Hsceihedge.index[0]]['delta']=interdelta(Hsceihedge['nodeprice']+1,Hsceihedge.index[0])

for i in range(len(Hscei)):
    try:
        Hscei_buff=Hscei.copy()
        Hscei_buff=Hscei.loc[:Hsceihedge.index[0]]
        Hscei_buff=Hscei_buff.iloc[:len(Hscei_buff)-1,:]
        Hscei_buff['whether']=np.array(pd.DataFrame(abs(Hscei_buff['price'].values/Hsceihedge['price'].values-1)>alpha).replace(False,np.nan))
        Hsceihedge=Hscei_buff.iloc[::-1].dropna().iloc[:1,:]
        Hsceirec.loc[Hsceihedge.index[0]]['delta']=interdelta(Hsceihedge['nodeprice']+1,Hsceihedge.index[0])
    except:
        break

#%% Payoff 계산

#Eurorec=pd.concat([Eurorec.iloc[:,:3],Eurorec.iloc[:,4:]], axis=1)
#Eurorec['ret']=Eurorec['ret']+1
Eurorec=Eurorec.replace(0,np.nan).dropna()
Eurorec=Eurorec[5*pp:]

Eurorec['number']=-Eurorec['delta']/(Eurorec['exchange']*10) #틱을 원화로 바꿔주기 위해서 원화로 환율까지..  환헷지 안 했을 때
#Eurorec['number']=-Eurorec['delta']/(Euroex_avg*10) #평균환율로 환헷지 했다고 칠 때!!
Eurorec['number'].iloc[:-1]= (-Eurorec['delta']+Eurorec['delta'].shift(-1))/(Eurorec['exchange']*10) #환헷지 안 했을 때
#Eurorec['number'].iloc[:-1]= (-Eurorec['delta']+Eurorec['delta'].shift(-1))/(Euroex_avg*10) #환헷지 했을 때!
#Eurorec.iloc[:-1,-1]= -Eurorec['delta']+Eurorec['delta'].shift(-1)
Eurorec['payoff']=(-Euro['price']*Eurorec['number']).dropna()  #샀으니까 마이너스, 팔았으니까 플러스
Eurorec['cumpayoff']=Eurorec['payoff'][::-1].cumsum()
Eurorec['cumnum']=Eurorec['number'][::-1].cumsum()
Europayoff=Eurorec.iloc[0,-2]+(Euro['price']*Eurorec['cumnum']).dropna().iloc[:1]+(Eurorec['cumpayoff'].min()*(r/2))


#Hsceirec=pd.concat([Hsceirec.iloc[:,:3],Hsceirec.iloc[:,4:]], axis=1)
#Hsceirec['ret']=Hsceirec['ret']+1
Hsceirec=Hsceirec.replace(0,np.nan).dropna()
Hsceirec=Hsceirec[5*pp:]

Hsceirec['number']=-Hsceirec['delta']/(Hsceirec['exchange']*10) #환헷지 안 했을 때!!
#Hsceirec['number']=-Hsceirec['delta']/(Hscex_avg*50)  #환헷지 했을 때!
Hsceirec['number'].iloc[:-1]= (-Hsceirec['delta']+Hsceirec['delta'].shift(-1))/(Hsceirec['exchange']*10) #환헷지 안 했을 때!
#Hsceirec['number'].iloc[:-1]= (-Hsceirec['delta']+Hsceirec['delta'].shift(-1))/(Hscex_avg*10)  #환헷지 했을 때!
Hsceirec['payoff']=(-Hscei['price']*Hsceirec['number']).dropna()  #샀으니까 마이너스, 팔았으니까 플러스
Hsceirec['cumpayoff']=Hsceirec['payoff'][::-1].cumsum()
Hsceirec['cumnum']=Hsceirec['number'][::-1].cumsum()
Hsceipayoff=Hsceirec.iloc[0,-2]+(Hscei['price']*Hsceirec['cumnum']).dropna().iloc[:1]+(Hsceirec['cumpayoff'].min()*(r/2))

ELSpayoff=Europayoff.values+Hsceipayoff.values+F*(1+(r/2))-F*(1+coupon[0])
print('='*50, '\n', 'ELS의 Payoff는 발행금액이 ',F,' 이고, alpha는 ',alpha,' 일 때, ',ELSpayoff,' 이다. 이는 환 헷지 및 변동성 헷지를 하지 않았기 때문으로 추정한다.')


(Euro['price']/Euro['price'].shift(-1)).std()*np.sqrt(252)
(Hscei['price']/Hscei['price'].shift(-1)).std()*np.sqrt(252)

len(Eurorec)
len(Hsceirec)

#((pd.DataFrame(ELS[252,:,Ny0])-pd.DataFrame(ELS[252,:,Ny0]).shift(1))/k1).plot()


