import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import util
import util.FD
import util.LST
import util.base_flow

plt.style.use('seaborn-paper')

ny=100
F=260.
#omega=0.26
beta=0.
Re_delta=1000.
nu=1.7208/Re_delta
omega = F/(nu*10**6)
x_start = 1./nu
Re = x_start
y_max=61.
#y=np.linspace(0,y_max,ny)
#delta=2.0001
yi=20.
#y = y_max*(1. + (np.tanh(delta*(y/y_max - 1.))/np.tanh(delta)))
y = util.FD.set_Cheby_stretched_y(y_max,ny,yi=yi)

params={}
params['ny']=ny
params['Re']=Re
params['omega']=omega
params['beta']=beta

diffs={}
Dy = util.FD.set_D_Chebyshev(y,d=1,need_map=True)
Dyy = util.FD.set_D_Chebyshev(y,d=2,need_map=True)
diffs['Dy']=Dy
diffs['Dyy']=Dyy

helper_mats={}
helper_mats['zero']=np.zeros((ny,ny))
helper_mats['I']=np.eye(ny)

baseflow = util.base_flow.blasius(y,x=x_start,nu=nu,plot=True)

params['flags']={'LST2':False,'LST3':False}
L,M,eigvals,eigfuncl,eigfuncr = util.LST.LST(params,diffs,baseflow,helper_mats)
params['flags']={'LST2':True ,'LST3':False}
L2,eigvals2,eigfuncl2,eigfuncr2 = util.LST.LST(params,diffs,baseflow,helper_mats)
params['flags']={'LST2':True ,'LST3':True }
L3,eigvals3,eigfuncl3,eigfuncr3 = util.LST.LST(params,diffs,baseflow,helper_mats)

fig,ax=plt.subplots(figsize=(3,3))
eigvals_scaled=1.7208*eigvals
eigvals_scaled2=1.7208*eigvals2
eigvals_scaled3=1.7208*eigvals3
ax.plot(eigvals_scaled.real,eigvals_scaled.imag,'o',label='LST1')
ax.plot(eigvals_scaled2.real,eigvals_scaled2.imag,'o',label='LST2')
ax.plot(eigvals_scaled3.real,eigvals_scaled3.imag,'.',label='LST3')
ax.set_xlim([-0.5,2])
ax.set_ylim([-2,5])
ax.set_xlabel(r'$\alpha_r$')
ax.set_ylabel(r'$\alpha_i$')
ax.legend(loc='best',numpoints=1)
fig.tight_layout()
fig.show()

alpha1=0.315+0.048j
fig,ax=plt.subplots()
eigval1,eigfuncr1 = sp.sparse.linalg.eigs(L,M=M,sigma=alpha1,k=1)
eigval2,eigfuncr2 = sp.sparse.linalg.eigs(L2,sigma=alpha1,k=1)
eigval3,eigfuncr3 = sp.sparse.linalg.eigs(L3,sigma=alpha1,k=1)
normi1=np.max(np.abs(eigfuncr1[:,0]))
normi2=np.max(np.abs(eigfuncr2[:,0]))
normi3=np.max(np.abs(eigfuncr3[:,0]))
ax.plot(np.abs(eigfuncr1[:ny,0])/normi1,y)
ax.plot(np.abs(eigfuncr2[:ny-2,0])/normi2,y[1:-1])
ax.plot(np.abs(eigfuncr3[:ny-2,0])/normi3,y[1:-1])
fig.tight_layout()
fig.show()
fig,ax=plt.subplots()
ax.plot(np.abs(eigfuncr1[ny:2*ny,0])/normi1,y)
ax.plot(np.abs(eigfuncr2[ny-2:2*(ny-2),0])/normi2,y[1:-1])
ax.plot(np.abs(eigfuncr3[ny-2:2*(ny-2),0])/normi3,y[1:-1])
fig.tight_layout()
fig.show()

input("Press Enter to continue...")
