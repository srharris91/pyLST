#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ..Classes import base_flow_class

def channel(y,x=1,Uinf=1,nu=1, plot=True):
    ''' Return velocity profile for plane Poiseuille flow
    Input:
        y: array of height of channel
        x: location along plate
        base_type: type of base_flow ['channel','plate']
    Output base flow for plane Poiseuille flow between two parallel plates or Blasius profile
        U: U mean velocity 1-y^2
        Uy: dU/dy of mean belocity -2y
        Uyy: d^2 U/dy^2 of mean velocity -2
    '''
    _y = y
    U = 1.-_y**2 # make a n vector of shape (n,1) so U will broadcast with D2 and D4 correctly
    Uy = -2.*_y # dU/dy of base flow
    Uyy = -2.*np.ones_like(_y) # d^2 U/dy^2 of base flow
    if plot:
        # plot values
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot(U,y,'.',label=r'$U$')
        #ax.plot(V,y,label='V')
        ax.plot(Uy,y,'.',label=r'$U_y$')
        ax.plot(Uyy,y,'.',label=r'$U_{yy}$')
        ax.set_ylabel(r'$y$')
        ax.legend(loc='best',numpoints=1) ;
        fig.tight_layout()
        fig.show()
    o = np.zeros_like(U)
    return base_flow_class(U=U.flatten(),Uy=Uy.flatten(),Uyy=Uyy.flatten(),V=o,P=o)

def tanh(y, diffs,plot=True):
    ''' Return velocity profile for plane tanh profile (see Michalke 1964)
        U = 0.5*(1+tanh(y))
    Input:
        y: array of domain
        base_type: type of base_flow ['channel','plate']
    Output base flow for plane Poiseuille flow between two parallel plates or Blasius profile
        U: U mean velocity 0.5*(1+tanh(y))
        Uy: dU/dy of mean belocity
        Uyy: d^2 U/dy^2 of mean velocity
    '''
    Dy = diffs['Dy']
    Dyy = diffs['Dyy']
    U = 0.5*(1.0+np.tanh(y))
    #Uy = 0.5*(1.0/np.cosh(y)**2) # dU/dy of base flow
    #Uyy = np.tanh(y)*(-1.0/np.cos(y)**2) # d^2 U/dy^2 of base flow
    Uy = Dy@U
    Uyy = Dyy@U
    if plot:
        # plot values
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot(U,y,'.',label=r'$U$')
        #ax.plot(V,y,label='V')
        ax.plot(Uy,y,'.',label=r'$U_y$')
        ax.plot(Uyy,y,'.',label=r'$U_{yy}$')
        ax.set_ylabel(r'$y$')
        ax.legend(loc='best',numpoints=1) ;
        fig.tight_layout()
        fig.show()
    o = np.zeros_like(U)
    return base_flow_class(U=U.flatten(),Uy=Uy.flatten(),Uyy=Uyy.flatten(),V=o,P=o)
def blasius(y,x=1,Uinf=1,nu=1,plot=True):
    '''
    Input:
        y: array of height of channel or flat plate
        x: location along plate
        base_type: type of base_flow ['channel','plate']
    Output base flow for plane Poiseuille flow between two parallel plates or Blasius profile
        U: U mean velocity
        Uy: dU/dy of mean belocity
        Uyy: d^2 U/dy^2 of mean velocity
        Ux: dU/dx of mean velocity
        V: wall normal mean velocity
        Vy: dVdy
        Vx: dVdx
    '''
    # assume nu=1 as default
    y_uniform=np.linspace(y.min(),y.max(),y.size*100)
    eta=y_uniform*np.sqrt(Uinf/(2.*nu*x))
    deta=np.diff(eta) # assume uniform grid would mean deta is all the same
    # IC for blasius f'''-ff'' = 0
    # or changed to coupled first order ODE
    #     f'' = \int -f*f'' deta
    #     f'  = \int f'' deta
    #     f   = \int f' deta
    # initialize and ICs
    # make lambda function
    f_fs = lambda fs: np.array([
        -fs[2]*fs[0], # f'' = \int -f*f'' deta
        fs[0],        # f'  = \int f'' deta
        fs[1],        # f   = \int f' deta
        1.])       # eta = \int 1 deta
    fs = np.zeros((eta.size,4))
    fs[0,0] = 0.332057336215195*np.sqrt(2.) #0.469600 # f''
    fs[0,1] = 0.       # f'
    fs[0,2] = 0.       # f
    fs[0,3] = eta[0]      # eta
    # step through eta
    for i,ideta in enumerate(deta):
        k1 = ideta*f_fs(fs[i]);
        k2 = ideta*f_fs(fs[i]+k1/2);
        k3 = ideta*f_fs(fs[i]+k2/2);
        k4 = ideta*f_fs(fs[i]+k3);
        fs[i+1] = fs[i] + (k1+(k2*2)+(k3*2)+k4)/6;
    #print('eta,f,fp,fpp = ')
    #print(fs[:,::-1])
    fpp=np.interp(y,y_uniform,fs[:,0])
    fp =np.interp(y,y_uniform,fs[:,1])
    f  =np.interp(y,y_uniform,fs[:,2])
    eta=np.interp(y,y_uniform,fs[:,3])
    fppp = np.gradient(fpp,eta)
    #print("eta = ",eta)
    #print("fp",fp)
    U  = Uinf*fp # f'
    Uy = fpp*np.sqrt(Uinf**3/(2.*nu*x))
    Uyy= fppp*(Uinf**2/(2.*nu*x))
    Ux = fpp*(-eta/(2.*x))
    V  = np.sqrt(nu*Uinf/(2.*x))*(eta*fp - f)
    Vy = Uinf/(2.*x) * eta*fpp
    Vx = np.sqrt(nu*Uinf/(8.*x**3)) * (-eta*fp + f - eta**2*fpp)
    baseflow = base_flow_class(U=U,Uy=Uy,Uyy=Uyy,Ux=Ux,V=V,Vy=Vy,Vx=Vx,P=np.zeros_like(U),ny=y.size)
    if plot:
        # plot values
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot(U,y,'.',label=r'$U$')
        #ax.plot(V,y,label='V')
        ax.plot(Uy,y,'.',label=r'$U_y$')
        ax.plot(Uyy,y,'.',label=r'$U_{yy}$')
        ax.set_ylabel(r'$y$')
        ax.legend(loc='best',numpoints=1) ;
        fig.tight_layout()
        fig.show()
    print('Vinf = ',V[-1],0.8604*Uinf*np.sqrt(nu/(x*Uinf)))
    return baseflow

def BL_FD_Implicit(params,diffs,baseflow,helper_mats):
    '''March boundary layer equations one step 
    
    The solution technique is adapted from  White Viscous Fluid Flow textbook (see pg. 226) 
    in the Laminar-Boundary-Layer Equations.  We add here, however, nonlinear terms to the RHS.
    This produces the equations as shown below for v-momentum, u-momentum, and continuity
    
    ∂P/∂y = -f_v
    u(∂u/∂x) + v(∂u/∂y) = -∂P/∂x + ν(∂^2u/∂y^2) - f_u
    ∂u/∂x + ∂v/∂y = 0

    This is solved numerically using the implicit/explicit choices outlined by White.

    ∂P_{i+1}/∂y = -f_v_{i}
    u_i(∂u_{i+1}/∂x) + v_i(∂u_i/∂y) = -∂P_{i+1}/∂x + ν(∂^2u_{i+1}/∂y^2) - f_u_{i}
    ∂u_{i+1}/∂x + ∂v_{i+1}/∂y = 0
    
    where f_v is the average nonlinear terms in the v-momentum equations.  
    Namely f_v = mean(u'(∂v'/∂x) + v'(∂v'/∂y) )
    Similarly f_u = mean(u'(∂u'/∂x) + v'(∂u'/∂y) ) for the u-momentum equations

    The solution takes 3 linear system solves in order of the equations listed above.  

    This should take on the unconditionally stable marching procedure. 
    A completely explicit marching procedure is described in White and worth investigating 
    if this is the source of code optimization slow downs.  
    Though, we will have to work with a conditionally stable procedure if that is implemented. 

    The Boundary Conditions include 
        1. P=0 at freestream for v-momentum
        2. u=0 at wall and u=u_∞ at freestream
        3. v=0 at wall for continuity
    
    Inputs:
        params:dict containing hx,x,grid,nu for the flow (though x is currently not used as a BC here)
        diffs:dict containing derivatives Dy and Dyy
        baseflow:base_flow_class containing the previous marched step
        helper_mats:dict containing fubar_base and fvbar_base
    Returns:
        baseflow_next:base_flow_class containing the next marched step (init with U,V,P and ny)
    
    '''

    # read in nonlinear terms
    fubar = helper_mats['fubar_base']
    fvbar = helper_mats['fvbar_base']
    # read in params
    hx = params['hx']
    x = params['x']
    ny = params['grid'].ny
    nu = params['nu']
    # read in diffs
    Dy = diffs['Dy']
    Dyy= diffs['Dyy']
    # read in baseflow from previous step
    U = baseflow.get_U()
    V = baseflow.get_V()
    P = baseflow.get_P()
    Ub= np.diag(U)
    Uinf=1. # assuming constant Uinf
    Uinfp1 = 1. # so next Uinf is also 1 at i+1 streamwise marched step

    # solve PDEs in order as described in docstring
    # solve V-momentum for P_{i+1}
    A = Dy.copy()
    b = -fvbar.copy()
    # set BCs
    A[ny-1,:] = 0. # P=0 at freestream
    A[ny-1,ny-1]= 1.
    b[ny-1] = 0.
    Pp1 = np.linalg.solve(A,b)

    # solve U-momentum for U_{i+1}
    A = Ub/hx - nu*Dyy
    b = (U**2)/hx - V*(Dy@U) + (Pp1-P)/(hx) - fubar
    # set BCs
    A[0,:] = 0. # U=0 at wall
    A[0,0] = 1.
    b[0] = 0.
    A[ny-1,:] = 0. # U=1 at freestream
    A[ny-1,ny-1]= 1.
    b[ny-1] = Uinfp1
    Up1 = np.linalg.solve(A,b)

    # solve continuity for V
    A = Dy.copy()
    b = -(Up1-U)/hx
    # set BCs
    A[0,:]=0. # V=0 at wall
    A[0,0]=1.
    b[0]=0
    #A[ny-1,:] = 0 # V=vinf at wall at next step
    #A[ny-1,ny-1]=1.
    #Vinfp1 = 0.8604*np.sqrt(nu/(x+hx))
    #b[ny-1]=Vinfp1
    Vp1 = np.linalg.solve(A,b)
    #P = np.zeros_like(Up1)

    # create base_flow_class and return
    baseflow_next = base_flow_class(U=Up1,V=Vp1,P=Pp1,ny=ny)
    return baseflow_next

# function for boundary layer equations
def set_BL_A(params,difs,baseflow,helper_mats):
    #ny = params['grid'].ny
    ny = baseflow.ny
    zero = helper_mats['zero']
    I = helper_mats['I']
    U = np.diag(baseflow.get_U())
    A = np.block([
        [U, zero, I],
        #[zero, U, zero],
        [zero, zero, zero],
        [I, zero, zero],
    ])
    return A
def set_BL_A_ByByy(params,diffs,baseflow,helper_mats):
    ''' Boundary layer equations, set A,By+Byy matrices
    Input:
        params:
        diffs:
        baseflow:base_flow_class: containing Q baseflow vector [U,V,P]
        helper_mats:
    Output base flow for plane Poiseuille flow between two parallel plates or Blasius profile
        F: F matrix in dQdx = F@Q
    '''
    # params
    nu = params['nu']
    #y = params['y']
    #ny= params['grid'].ny
    ny = baseflow.ny
    
    # diffs
    Dy = diffs['Dy']
    Dyy = diffs['Dyy']
    
    # baseflow
    U = np.diag(baseflow.get_U())
    V = np.diag(baseflow.get_V())
    
    # helper_mats
    zero=helper_mats['zero']
    I = helper_mats['I']
    
    ByByy = np.block([
        [-nu*Dyy+V@Dy, zero, zero],
        [zero, zero, Dy],
        [zero, Dy, zero],
    ])
    A = set_BL_A(params,diffs,baseflow,helper_mats)
    
    #print('Vinf = ',V[-1],0.8604*Uinf*np.sqrt(nu/(x*Uinf)))
    return A, ByByy

def BL_IE(params,diffs,baseflow_old,helper_mats):
    ''' march the boundary layer equations one step using Emplicit Euler'''
    Q_old = baseflow_old.get_Q()
    fubar = helper_mats['fubar_base']
    fvbar = helper_mats['fvbar_base']
    #ny=params['grid'].ny
    ny = baseflow_old.ny
    fbar = np.block([fubar,fvbar,np.zeros(ny)])
    print('made fbar')
    hx = params['hx']
    nu = params['nu']
    tol=1.0E-14
    iteration=0
    #Q_new=Q_old.copy()
    baseflow = base_flow_class(Q=Q_old,ny=ny)
    Uinf=baseflow.get_Q()[ny-1].copy()
    x = params['x']+hx
    converged=False
    max_iter=10000
    while not converged:
        Vinf = 0.8604*Uinf*np.sqrt(nu/(x*Uinf))
        #print('Vinf = ',Vinf)
        baseflow1=base_flow_class(Q=baseflow.get_Q(),ny=ny)
        A,ByByy = set_BL_A_ByByy(params,diffs,baseflow1,helper_mats)
        Ahx = A/hx
        b = Ahx@Q_old - fbar
        # set special BCs in momentum equations
        M = Ahx + ByByy
        M[0,:]=0.
        M[0,0]=1. # u=0 at wall
        b[0] = 0.
        M[ny-1,:]=0.
        M[ny-1,ny-1]=Uinf # u at freestream
        b[ny-1] = 1.
        M[ny,:]=0.
        M[ny,ny]=1. # v=0 at wall
        b[ny] = 0.
        M[2*ny-1,:]=0.
        M[2*ny-1,2*ny-1]=1. # v=inf at wall
        b[2*ny-1] = Vinf
        baseflow = base_flow_class(Q=np.linalg.solve(M,b),ny=ny)
        error = np.linalg.norm(baseflow1.get_Q()-baseflow.get_Q(),ord=np.inf)
        if error<tol: 
            print('norm = ',error)
            print('tol = ',tol)
            converged=True
        if iteration>=max_iter:
            converged=True
            print('Exit BL_IE due to max_iter = ',max_iter)
        #if not np.isfinite(Q_new).all():
            #print('failed isfinite test')
            #return Q_new,M,b
        iteration+=1
    print('BL_IE iterations = ',iteration)
    return baseflow

# march one step using semi-staggered grid adding all terms
def march_one_step(Ai,Di,y,yP,x,U,V,P, hx=0.960937500000,nu=4.92/800.,Re=800./4.92,
                    Dy=None,Dyy=None,DyP=None,DyyP=None,
                   tol=1E-10,maxiter=200,ImplicitEuler=True,
                   viscous_momentum_v=1.0,momentum_v=1.0,F1=None,F2=None,nonlinear=False,
                  UdVdxDNS=None, d2Udx2DNS=None,d2Vdx2DNS=None): 
    Q_i=np.block([U,V,P])
    ny=len(U)
    #U,V=Ublasius[1,:],Vblasius[1,:]
    Uinf=U[-1]
    #Vinf=V[-1]
    Vinf=0.8604*np.sqrt(nu/(x+hx))
    Pinf=P[-1]
    #U,V=Ublasius[0,:],Vblasius[0,:]
    Qk=np.copy(Q_i)

    zero=np.zeros(np.diag(U).shape)
    zeroP=zero[:,:-1]
    I=np.eye(*U.shape)
    IP=I[:,:-1]

    L2_error=400

    niter=0
    while (L2_error>=tol) and (niter<maxiter):
        # set U-mom LHS and ddx
        UmomentumLHS=np.block([
            np.diag(V)@Dy + -1./Re*Dyy,# U terms
            zero,# V terms
            zeroP,# P terms
        ])
        Umomentumddx=np.block([
            np.diag(U),# U terms
            zero,# V terms
            IP,# P terms
        ])

        # set V-mom LHS and ddx
        VmomentumLHS=np.block([
            zero,# U terms
            viscous_momentum_v*(-1.0/Re*Dyy + np.diag(V)@Dy),# V terms
            DyP,# P terms
        ])
        Vmomentumddx=np.block([
            zero,# U terms
            momentum_v*np.diag(U),#zero,# V terms
            zeroP,# P terms
        ])

        # set continuity LHS and ddx
        ContLHS=np.block([
            zero,# U terms
            Dy,# V terms
            zeroP,# P terms
        ])
        Contddx=np.block([
            I,# U terms
            zero,# V terms
            zeroP,# P terms
        ])

        # set A
        A = np.block([[UmomentumLHS],[VmomentumLHS],[ContLHS]])

        # scale ddx terms and split to LHS and RHS in Implicit Euler
        D = np.block([[Umomentumddx],[Vmomentumddx],[Contddx]])
        
        # form LHS and RHS operators
        if ImplicitEuler:
            LHS = A+D/hx

            # set RHS
            forcing_terms=np.zeros_like(np.hstack([U,V,V]))
            if nonlinear:
                forcing_terms[:ny]=-F1
                forcing_terms[ny:2*ny]=-F2
                forcing_terms[:ny]=forcing_terms[:ny] + 1./Re* d2Udx2DNS
                # set V-momentum UdVdx term
                forcing_terms[ny:2*ny]=forcing_terms[ny:2*ny]-UdVdxDNS+1./Re * d2Vdx2DNS
            #if niter<30:
                #forcing_terms[ny:2*ny]=-niter/30*0.0*U*(V-Q_i[ny:2*ny])/hx # U dVdx
            #else:
                #forcing_terms[ny:2*ny]=-0.0*U*(V-Q_i[ny:2*ny])/hx # U dVdx
            RHS = (D/hx)@Q_i + forcing_terms
        else: # Crank-Nicolson
            LHS = (D+Di)/(2.*hx) + A/2.
            forcing_terms=np.zeros_like(np.hstack([U,V,V]))
            RHS = ((D+Di)/(2.*hx) - Ai/2.)@Q_i + forcing_terms
            

        # set BCs
        # u wall and freestream
        LHS[[0,ny-1],:]=0
        LHS[0,0],LHS[ny-1,ny-1]=1,Uinf
        RHS[[0,ny-1]]=0,1 

        if True:
            # v wall
            LHS[ny,:]=0
            LHS[ny,ny]=1
            RHS[ny]=0 
            # v freestream
            # Neumann
            if False:
                LHS[2*ny-1,:]=0
                #LHS[2*ny-1,2*ny-1]=-1
                #LHS[2*ny-1,2*ny-2]=1
                LHS[2*ny-1,ny:2*ny]=Dy[-1,:]
                RHS[2*ny-1]=0
            else:
                LHS[2*ny-1,:]=0
                LHS[2*ny-1,2*ny-1]=1
                RHS[2*ny-1]=Vinf
        else: # replace continuity equation
            # v-wall
            LHS[2*ny,:]=0
            LHS[2*ny,ny]=1
            RHS[2*ny]=0 
            # freestream Neumann
            LHS[3*ny-1,:]=0
            LHS[3*ny-1,ny:2*ny]=Dy[-1,:]
            RHS[3*ny-1]=0

        # P infty 
        #LHS[-1,:]=0
        #LHS[-1,-1]=1
        #RHS[-1]=Pinf
        if True:
            # remove extra row of P BC
            LHS=LHS[:-1,:]
            RHS=RHS[:-1]
        else:
            # remove extra row of v-momentum for P BC
            LHS=np.delete(LHS,2*ny-1,axis=0)
            RHS=np.delete(RHS,2*ny-1)

        Q = np.linalg.solve(LHS,RHS)

        U=Q[:ny]
        V=Q[ny:2*ny]
        P=Q[2*ny:]

        L2_error = np.linalg.norm(np.abs(Qk-Q))
        Qk=np.copy(Q)
        niter+=1
    print('Final L2 error = ',L2_error)
    print('n iterations = ',niter)
    return A,D,U,V,P

