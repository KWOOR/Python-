
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 00:02:02 2019
@author: 우람
"""

from scipy.stats import norm
from scipy.optimize import root, fsolve, newton
import scipy.interpolate as spi
import statsmodels.api as sm
from scipy import stats
import copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy as sp
import math
import os
import time
from datetime import date
from datetime import timedelta
from datetime import datetime
os.chdir('C:\\Users\\kur7\\Desktop\\ELS')

hscei_put=pd.read_excel('hscei_optiondata.xlsx').replace("#N/A Invalid Security", np.nan).dropna()
hscei_put.index=range(0,len(hscei_put))
for i in range(len(hscei_put)):
    hscei_put['Maturity'][i] = (datetime.strptime(hscei_put['Maturity'][i], 
             '%m/%d/%Y').date() - date(2018,2,13)).days/365
             
euro_put=pd.read_excel('euro_optiondata.xlsx').replace("#N/A Invalid Security", np.nan).dropna()
euro_put.index=range(0,len(euro_put))
for i in range(len(euro_put)):
    euro_put['Maturity'][i] = (datetime.strptime(euro_put['Maturity'][i], 
             '%m/%d/%Y').date() - date(2018,2,13)).days/365
                       
#쓸데 없는 거 버리고.. 0번째 열이 행사가, 1번째 열이 잔존만기(연율화... 0.13, 0.3 이런식)


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
pp=20 #6개월을 몇 번으로 쪼개는지
Nt=6*pp #조기상환기회가 6번 있음. 만기를 몇 번 쪼갤 것인지
Nx=100 #첫 번째 기초자산을 몇 번 쪼갤 것인지
Ny=100 #두 번째 기초자산을 몇 번 쪼갤 것인지
Nx0=round(Nx/2) #처음 가격을 노드에 찍은 것
Ny0=round(Ny/2) 
h=T/Nt #dt
k1=(S1max-S1min)/Nx #dx
k2=(S2max-S2min)/Ny #dy
q=0 #배당.. 없음 
option_type = 'put'

#%%

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


def a_n(n,l=99,q=0, sig1=0.27,k1=70):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))
def b_n(n,l=99,q=0, sig1=0.27,r=0.02,k1=70, h=0.01):
    return 1/h+ (sig1**2)*((n*k1)**2)/(k1**2)+ (r*n*k1)/k1+0.5*r
def c_n(n,l=99,q=0, sig1=0.27, k1=70,r=0.02):
    return -(sig1**2)*((n*k1)**2)/(2*(k1**2))- (r*n*k1)/k1
def d_n(u,n,l=99, m=0,q=0, sig1=0.27, sig2=0.217, rho=0.4,k1=70,k2=60,h=0.01):
    x= 0.5*rho*sig1*sig2*n*k1*l*k2*(u[m,n+1,l+1]-u[m,n+1,l]-u[m,n,l+1]+u[m,n,l])/(k1**2)+u[m,n,l]/h
    return x

def a_l(l,n=99,q=0, sig2=0.217,k2=60):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))
def b_l(l,n=99,q=0, sig2=0.217,r=0.02,k2=60, h=0.01):
    return 1/h+ (sig2**2)*((l*k2)**2)/(k2**2)+ (r*l*k2)/k2+0.5*r
def c_l(l,n=99,q=0, sig2=0.217, k2=60, r=0.02):
    return -(sig2**2)*((l*k2)**2)/(2*(k2**2))- (r*l*k2)/k2
def d_l(buff,l,n=99, m=0,q=0, sig1=0.27, sig2=0.217, rho=0.4,k1=70,k2=60,h=0.01):
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
'''Local Vol 쓸 때'''

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
    return vol


def impvol_f1(mdata,a,b,c,d,e,f,g): #mdata의 0번째 열은 행사가, 1번째 열은 잔존만기여야함
    f = a +c*np.exp(b*mdata.iloc[:,1])+d*mdata.iloc[:,0]+e*mdata.iloc[:,0]**2 +f*np.power(mdata.iloc[:,0],3) +g *np.power(mdata.iloc[:,0],4)
    return f


def impvol_f2(m,t,a,b,c,d,e,f,g): #m은 행사가 t는 만기
    f = a +c*np.exp(b*t)+d*m+e*m**2 +f*np.power(m,3) +g *np.power(m,4)
    return f


def optimizing(data, s):
    data['imp_vol']=np.nan
    for i in range(len(data)):
        data['imp_vol'][i]=implied_vol(s,data['Strike'][i],r,q,data['Maturity'][i],
                data['Price'][i],option_type)
    buff=data.replace(np.inf,np.nan)
    buff=buff.replace(-np.inf,np.nan) 
    buff['Strike'] = np.log((s*np.exp((r-q)*buff['Maturity']))/buff['Strike']) #행사가를 moneyness로 변환 (열이름은 그냥 행사가로 썼음...)
    b = sp.optimize.curve_fit(impvol_f1, buff.dropna(), buff.dropna()['imp_vol'], maxfev=3000)[0] #두번째 지수 
    return b

coeff1=optimizing(hscei_put, K0[0]) # HSCEI 옵션 데이터로 바꿔야함
coeff2=optimizing(euro_put, K0[1]) #유로스탁스 옵션 데이터로 바꿔야함
''' 이거 안 하면 에러 뜸... 이거 맞는 데이터 가져와서 바꾸면 에러 안 뜸'''

def impvol(strike,t,sival,s=K0[0]): # t는 만기, a는 위에서 최적화시킨 계수들의 값, s는 spot price
    m=np.log((s*np.exp((r-q)*t))/strike)
    return impvol_f2(m,t, sival[0], sival[1], sival[2], sival[3], sival[4], sival[5], sival[6])

def dt(strike,t,sival,s=K0[0]):
    m=np.log((s*np.exp((r-q)*t))/strike)
    return sival[2]*sival[1]*np.exp(sival[1]*t)*(r-q)*(sival[3]+2*sival[4]*m+3*sival[5]*m**2 + 4*sival[6]*np.power(m,3))

def dx(strike,t,sival,s=K0[0]):
    m=np.log((s*np.exp((r-q)*t))/strike)
    return -(sival[3]+2*sival[4]*m+3*sival[5]*m**2 + 4*sival[6]*np.power(m,3))/strike

def dxx(strike, t,sival,s=K0[0]):
    m=np.log((s*np.exp((r-q)*t))/strike)
    return  (sival[3]+2*sival[4]*(m+1)+3*sival[5]*m*(m+2)+ 4*sival[6]*m**2 * (m+3)) / (strike**2)

def ddd(strike, t,sival,s=K0[0]):
    return  (np.log(s/strike)+(r-q + 0.5*impvol(strike,t,sival,s)**2)*t)/(impvol(strike,t,sival,s)*np.sqrt(t))


def call_price(strike, maturity,sival, s=K0[0]):
    return bs_price(s,strike,r,q,maturity, impvol(strike,maturity,sival,s), option_type)

def local_vol(strike, maturity,sival, s=K0[0]):
    if strike ==0:
        strike =1
    if maturity==0:
        maturity=0.00001
    locvol=(impvol(strike, maturity,sival,s)**2 + 2*impvol(strike, maturity,sival,s)*maturity*(dt(strike, maturity,sival,s)+(r-q)*\
           strike*dx(strike,maturity,sival,s))) / ((1+strike*ddd(strike, maturity,sival,s)*dx(strike, maturity,sival,s)*\
           np.sqrt(maturity))**2 + impvol(strike,maturity,sival,s)**2 * (strike**2) * maturity*(dxx(strike,maturity,sival,s)-ddd(strike,maturity,sival,s)*\
           (dx(strike,maturity,sival,s)**2)*np.sqrt(maturity)))
    return np.sqrt(locvol)

   
def make_price_locvol(Nt, Nx, Ny, state=False): #Fals면 낙인 안 쳤을 때, True면 낙인 쳤을 때..
    if state==False:
        u = np.zeros((Nt+1,Nx+1,Ny+1));
        u[0,math.ceil(Nx0*KI):, math.ceil(Ny0*KI):] = F*(1+coupon[5]);
        for i in range(math.ceil(Nx0*KI)):
            for j in range(Ny+1):
                u[0,i, j]=F*np.minimum(i/Nx0, j/Ny0)
                u[0,j, i]=F*np.minimum(i/Nx0, j/Ny0)
    else:
        u = np.zeros((Nt+1,Nx+1,Ny+1));
        u[0,math.ceil(Nx0*K[5]):, math.ceil(Ny0*K[5]):] = F*(1+coupon[5]);
        for i in range(math.ceil(Nx0*K[5])):
            for j in range(Ny+1):
                u[0,i, j]=F*np.minimum(i/Nx0, j/Ny0)
                u[0,j, i]=F*np.minimum(i/Nx0, j/Ny0)        
        
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
                a.append(a_n(n=n,l=l,k1=k1,q=q, sig1=local_vol( n*k1 ,m/100, sival=coeff1, s=K0[0])))
                b.append(b_n(n=n,l=l,k1=k1,q=q,r=r, sig1=local_vol( n*k1 ,m/100, sival=coeff1, s=K0[0])))
                c.append(c_n(n=n,l=l,k1=k1,q=q,r=r, sig1=local_vol( n*k1 ,m/100, sival=coeff1, s=K0[0])))
                d.append(d_n(u,n=n,l=l,m=m,k1=k1, k2=k2,q=q, rho=rho,sig1=local_vol( n*k1 ,m/100, sival=coeff1, s=K0[0]), 
                             sig2=local_vol( l*k2 ,m/100, sival=coeff2, s=K0[1])))
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
                a.append(a_l(l=l,n=n,k2=k2,q=q, sig2=local_vol( l*k2 ,m/100, sival=coeff2, s=K0[1]) ))
                b.append(b_l(l=l,n=n,k2=k2,q=q,r=r, sig2=local_vol( l*k2 ,m/100, sival=coeff2, s=K0[1])))
                c.append(c_l(l=l,n=n,k2=k2,q=q,r=r, sig2=local_vol( l*k2 ,m/100, sival=coeff2, s=K0[1])))
                d.append(d_l(buff,l=l,n=n,m=m,k2=k2,k1=k1,q=q,rho=rho,sig2=local_vol( l*k2 ,m/100, sival=coeff2, s=K0[1]),
                             sig1=local_vol( n*k1 ,m/100, sival=coeff1, s=K0[0])))
            b[0]=2*a[0]+b[0]
            c[0]=c[0]-a[0]
            a[-1]=a[-1]-c[-1]
            b[-1]=b[-1]+2*c[-1]
            u[m+1,n,1:Ny]= TDMAsolver(a[1:],b,c[:-1],d)
        u[m+1,0,:]=2*u[m+1,1,:]-u[m+1,2,:]
        u[m+1,-1,:]=2*u[m+1,-2,:]-u[m+1,-3,:]
        u[m+1,:,0]=2*u[m+1,:,1]-u[m+1,:,2]
        u[m+1,:,-1]=2*u[m+1,:,-2]-u[m+1,:,-3]   
        
    return u



#%%
        
def make_price(Nt, Nx, Ny, state=False): #Fals면 낙인 안 쳤을 때, True면 낙인 쳤을 때..
    if state==False:
        u = np.zeros((Nt+1,Nx+1,Ny+1));
        u[0,math.ceil(Nx0*KI):, math.ceil(Ny0*KI):] = F*(1+coupon[5]);
        for i in range(math.ceil(Nx0*KI)):
            for j in range(Ny+1):
                u[0,i, j]=F*np.minimum(i/Nx0, j/Ny0)
                u[0,j, i]=F*np.minimum(i/Nx0, j/Ny0)
    else:
        u = np.zeros((Nt+1,Nx+1,Ny+1));
        u[0,math.ceil(Nx0*K[5]):, math.ceil(Ny0*K[5]):] = F*(1+coupon[5]);
        for i in range(math.ceil(Nx0*K[5])):
            for j in range(Ny+1):
                u[0,i, j]=F*np.minimum(i/Nx0, j/Ny0)
                u[0,j, i]=F*np.minimum(i/Nx0, j/Ny0)        
        
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
                a.append(a_n(n=n,l=l,k1=k1,q=q, sig1=sig1))
                b.append(b_n(n=n,l=l,k1=k1,q=q,r=r, sig1=sig1))
                c.append(c_n(n=n,l=l,k1=k1,q=q,r=r, sig1=sig1))
                d.append(d_n(u,n=n,l=l,m=m,k1=k1, k2=k2,q=q, rho=rho,sig1=sig1, sig2=sig2))
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
                a.append(a_l(l=l,n=n,k2=k2,q=q, sig2=sig2 ))
                b.append(b_l(l=l,n=n,k2=k2,q=q,r=r, sig2=sig2))
                c.append(c_l(l=l,n=n,k2=k2,q=q,r=r, sig2=sig2))
                d.append(d_l(buff,l=l,n=n,m=m,k2=k2,k1=k1,q=q,rho=rho,sig2=sig2, sig1=sig1))
            b[0]=2*a[0]+b[0]
            c[0]=c[0]-a[0]
            a[-1]=a[-1]-c[-1]
            b[-1]=b[-1]+2*c[-1]
            u[m+1,n,1:Ny]= TDMAsolver(a[1:],b,c[:-1],d)
        u[m+1,0,:]=2*u[m+1,1,:]-u[m+1,2,:]
        u[m+1,-1,:]=2*u[m+1,-2,:]-u[m+1,-3,:]
        u[m+1,:,0]=2*u[m+1,:,1]-u[m+1,:,2]
        u[m+1,:,-1]=2*u[m+1,:,-2]-u[m+1,:,-3]   
    return u

No_price=make_price(Nt, Nx, Ny)
price=make_price(Nt,Nx,Ny, True)
#%%
''' Local vol 안 씀 '''    
'''낙인 안 칠 때 ELS Price'''
print("낙인 안 칠 때 ELS가격은",No_price[Nt,Nx0,Ny0])
NKI_ELS_Price=No_price[Nt,Nx0,Ny0]*(1-percent_KI) 
print("NKI_ELS_Price", NKI_ELS_Price)
'''낙인 쳤을 때 ELS Price'''
print("낙인 칠 때 ELS가격은",price[Nt,Nx0,Ny0])
KI_ELS_Price=price[Nt,Nx0,Ny0]*percent_KI
print("NKI_ELS_Price", KI_ELS_Price)

ELS_Price= NKI_ELS_Price+KI_ELS_Price
ELS=No_price*(1-percent_KI)+price*(percent_KI)

xnew=np.linspace(0,S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, ELS[Nt,:,:].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       


print("ELS 가격은 " , ELS_Price)
print("KI쳤을 때 ELS가격은", price[Nt,Nx0,Ny0])
print("KI 안 쳤을 때 ELS가격은", No_price[Nt,Nx0,Ny0])
print("KI칠 확률", percent_KI)

#%%
'''Local vol 쓸 때! '''
#데이터가 없어서.... 임시로 다른 데이터를 넣어서 그런지 시간이  오오오오오오래 걸림

locvol_price1= make_price_locvol(Nt,Nx,Ny)
locvol_price2= make_price_locvol(Nt,Nx,Ny, True)


print("W/ Locvol, 낙인 안 칠 때 ELS가격은",locvol_price1[Nt,Nx0,Ny0])
NKI_ELS_Price_LV=locvol_price1[Nt,Nx0,Ny0]*(1-percent_KI) 
print("NKI_ELS_Price", NKI_ELS_Price_LV)
'''낙인 쳤을 때 ELS Price'''
print("W/ Locvol, 낙인 칠 때 ELS가격은",locvol_price2[Nt,Nx0,Ny0])
KI_ELS_Price_LV=locvol_price2[Nt,Nx0,Ny0]*percent_KI
print("KI_ELS_Price", KI_ELS_Price_LV)

ELS_Price_LV= NKI_ELS_Price_LV+KI_ELS_Price_LV
ELS_LV=locvol_price1*(1-percent_KI)+locvol_price2*(percent_KI)

xnew=np.linspace(0,S1max,Nx+1)
ynew=np.linspace(0,S2max,Ny+1)
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
x,y = np.meshgrid(xnew, ynew)
surf = ax.plot_surface(x, y, ELS_LV[Nt,:,:].T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()       


print("ELS 가격은 " , ELS_Price_LV)
print("KI쳤을 때 ELS가격은", locvol_price2[Nt,Nx0,Ny0])
print("KI 안 쳤을 때 ELS가격은", locvol_price1[Nt,Nx0,Ny0])
print("KI칠 확률", percent_KI)
#%%
''' Hedge 수익 보기 '''
futures=pd.read_excel('futures data modified.xlsx') #Euro와 Hscei지수가 같이 있음...

def make_futuresdf(data):
    df=pd.DataFrame(data)
    df.columns=['date','price','exchange']
    df['index']=0
    empty = df.date[(df.date=='2018-02-13').replace(False, np.nan).dropna().index.values]
    for i in range(len(df)):
        df['index'][i]=(1095-(df.date[i]-df.date[int(empty.index.values)]).days)
    df=df.dropna()
    df['index']=df['index']/3.65
    df.index=df['index']
    df=df[:Nt]
    df=df.iloc[:,1:]
    df=df.iloc[:,:].astype(np.float64)
    df['nodeprice']=Ny0*(df['price']/df['price'].iloc[-1])
    return df
    
Euro=make_futuresdf(futures.iloc[: ,0:3])
Hscei=make_futuresdf(futures.iloc[:,3:6])

Euroex_avg=Euro['exchange'].sum()/len(Euro)
Hscex_avg=Hscei['exchange'].sum()/len(Hscei)

#%%


alpha=0.01
'''Daily Hedge를 원하면 -0.01로 해둬라'''

def make_delta(alpha, data, underlying=1): #1이면 Euro, 다른건 Hscei라는 뜻... 
    if underlying ==1:
        delta = -(pd.DataFrame(ELS[:,Nx0,:])-pd.DataFrame(ELS[:,Nx0,:]).shift(1, axis=1))/(k2)
    else:
        delta = -(pd.DataFrame(ELS[:,:,Ny0])-pd.DataFrame(ELS[:,:,Ny0]).shift(1, axis=1))/(k1)
    x=np.array(delta.index)
    y=np.array(delta.columns)
    z=delta.values
    interdelta=sp.interpolate.interp2d(y,x,z)
    df_record= data.iloc[:,1:].copy()
    df_record['delta']=0
    df_record=df_record.iloc[:,:].astype(np.float64)
    df_record.loc[Nt]['delta']=interdelta(Ny0+1,Nt)
    data['ret']=data['price']/data['price'].iloc[-1]-1
    data['whether']=(abs(data['ret'])>alpha).replace(False,np.nan)
    df_hedge=data.iloc[::-1].dropna().iloc[:1,:]
    df_record.loc[df_hedge.index[0]]['delta']=interdelta(df_hedge['nodeprice']+1,df_hedge.index[0])
    
    for i in range(len(data)):
        try:
            data_buff=data.copy()
            data_buff=data.loc[:df_hedge.index[0]]
            data_buff=data_buff.iloc[:len(data_buff)-1,:]
            data_buff['whether']=np.array(pd.DataFrame(abs(data_buff['price'].values/df_hedge['price'].values\
                     -1)>alpha).replace(False, np.nan))
            df_hedge=data_buff.iloc[::-1].dropna().iloc[:1,:]
            df_record.loc[df_hedge.index[0]]['delta']=interdelta(df_hedge['nodeprice']+1, 
                         df_hedge.index[0])
        except:
            break
    return df_record

Eurorec=make_delta(alpha, Euro)
Hsceirec=make_delta(alpha, Hscei, 2)

#%%

def make_hedge(data, fx_hedge=True, whether_euro=True):
    data['delta'] = data['delta'].replace(0,np.nan)
    data=data.dropna()
    data=data[5*pp:]
    
    if fx_hedge==True:
        if whether_euro==True:
            data['number']=-data['delta']/(Euroex_avg*10)
            data['number'].iloc[:-1]= (-data['delta']+data['delta'].shift(-1))/(Euroex_avg*10)
            data['payoff']=(-Euro['price']*data['number']).dropna()
        if whether_euro==False:
            data['number']=-data['delta']/(Hscex_avg*50)
            data['number'].iloc[:-1]= (-data['delta']+data['delta'].shift(-1))/(Hscex_avg*50)  
            data['payoff']=(-Hscei['price']*data['number']).dropna()
            
    if fx_hedge == False:
        if whether_euro ==True:
            data['number']=-data['delta']/(data['exchange']*10)
            data['number'].iloc[:-1]= (-data['delta']+data['delta'].shift(-1))/(data['exchange']*10) 
            data['payoff']=(-Euro['price']*data['number']).dropna()
        if whether_euro ==False:
            data['number']=-data['delta']/(data['exchange']*50)
            data['number'].iloc[:-1]= (-data['delta']+data['delta'].shift(-1))/(data['exchange']*50)
            data['payoff']=(-Hscei['price']*data['number']).dropna()
    
    return data
    
Eurorec=make_hedge(Eurorec, fx_hedge=False) # fx_hedge는 동 기간의 평균 환율로 했다고 가정.. 원한다면 True 
Hsceirec=make_hedge(Hsceirec, fx_hedge=False, whether_euro=False)


    
    #%% Test

def calculate_payoff(df1, df2):
    payoff= pd.concat([df1['payoff'], df2['payoff']], axis=1).fillna(0)
    payoff.columns=['Euro_payoff', 'Hscei_payoff']
    payoff['Futures_Payoff']=payoff['Euro_payoff']+payoff['Hscei_payoff']
    payoff['Buffer']=np.exp(r*(300-np.array(payoff.index.tolist()))/100)*10000
    payoff['Buffer2']= (-pd.Series((np.array(payoff.index.tolist()))).shift(1)+ \
          pd.Series((np.array(payoff.index.tolist())))).values/100
    
    payoff['Bond']=np.nan
    payoff['Bond'].iloc[-1]=payoff['Buffer'].iloc[-1]
    for i in range(len(payoff)-2,-1,-1):
        payoff['Bond'].iloc[i] =( payoff['Bond'].iloc[i+1] + payoff['Futures_Payoff'].iloc[i+1])*\
        np.exp(r*payoff['Buffer2'].iloc[i+1])
    
    payoff['Futures_Cumpayoff']=payoff['Futures_Payoff'][::-1].cumsum()
    payoff['Total_Payoff']=payoff['Bond']+payoff['Futures_Cumpayoff']
    payoff=payoff.drop(['Buffer', 'Buffer2'], axis=1)

    return payoff    

payoff=calculate_payoff(Eurorec, Hsceirec)
print(payoff['Total_Payoff'].iloc[0])

#payoff= pd.concat([Eurorec['payoff'], Hsceirec['payoff']], axis=1).fillna(0)
#payoff.columns=['Euro_payoff', 'Hscei_payoff']
#payoff['Futures_Payoff']=payoff['Euro_payoff']+payoff['Hscei_payoff']
#payoff['Buffer']=np.exp(r*(300-np.array(payoff.index.tolist()))/100)*10000
#payoff['Buffer2']= (-pd.Series((np.array(payoff.index.tolist()))).shift(1)+ \
#      pd.Series((np.array(payoff.index.tolist())))).values/100
#
#payoff['Bond']=np.nan
#payoff['Bond'].iloc[-1]=payoff['Buffer'].iloc[-1]
#for i in range(len(payoff)-2,-1,-1):
#    payoff['Bond'].iloc[i] =( payoff['Bond'].iloc[i+1] + payoff['Futures_Payoff'].iloc[i+1])*\
#    np.exp(r*payoff['Buffer2'].iloc[i+1])
#
#payoff['Futures_Cumpayoff']=payoff['Futures_Payoff'][::-1].cumsum()
#payoff['Total_Payoff']=payoff['Bond']+payoff['Futures_Cumpayoff']











© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
Press h to open a hovercard with more details.