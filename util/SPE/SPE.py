import numpy as np
from ..Classes import base_flow_class
from ..math import inner_product
from ..math import max_zero_crossings_percent
import scipy as sp
from scipy import linalg
from ..LST import dLdomega, LST
from ..base_flow import blasius
from ..helper import ifflag
from ..helper import find_nearest_arg

def set_SPE2D_ExplicitOperator_with_BCs(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> np.ndarray:
    ''' set SPE right hand vector for qx = -A^-1 (C+By+Byy)q = f(q,x)
    Inputs:
        params:dict containting grid, Re, omega, and beta
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        helper_mats:dict containing zero and I matrices
    Returns:
        F:np.ndarray containing the F@q RHS operator
        
    '''

    # params
    omega=params['omega']
    beta=params['beta']
    Re=params['Re']
    #hx=params['hx']
    ny=params['grid'].ny
    # helper mats
    I=helper_mats['I']
    zero=helper_mats['zero']
    # diffs
    Dy=diffs['Dy']
    Dyy=diffs['Dyy']
    i=1.j
    # baseflow
    U=np.diag(baseflow.U)
    Uinvvec=np.divide(1.,baseflow.U,out=np.zeros_like(baseflow.U),where=baseflow.U!=0.)
    Uinv=np.diag(Uinvvec)
    Ux=np.diag(baseflow.Ux)
    Uy=np.diag(baseflow.Uy)
    V=np.diag(baseflow.V)
    Vy=np.diag(baseflow.Vy)
    # set operator
    delta=(-i*omega + 1./Re*beta**2)*I+ V@Dy - Dyy/Re
    Uinvdelta=Uinv@delta
    # SPE operator dqdx + FSPE@q = 0
    F = -np.block([
        [zero, Dy, i*beta*I, zero],
        [zero, Uinvdelta+Vy*Uinv, zero, Uinv@Dy],
        [zero, zero, Uinvdelta, Uinv*i*beta],
        [delta+Ux, -U@Dy+Uy, -i*beta*U, zero]
    ])
    #fSPE = -FSPE@q_old
    # set BCs on vector for Explicit Euler marching
    F[(0,ny-1,ny,2*ny,2*ny-1,3*ny-1),:] = 0
    return F

def set_SPE2D_operators(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> list:
    ''' set SPE operators for C*q + By*qy + Byy*qyy + A*qx = 0
    Inputs:
        params:dict containting grid, Re, omega, beta, and alpha
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        helper_mats:dict containing zero and I matrices
    Returns:
        C:np.ndarray containing the C*q operator
        By:np.ndarray containing the By*qy operator
        Byy:np.ndarray containing the Byy*qyy operator
        A:np.ndarray containing the A*qx operator
        
    '''

    # extract parameters from params
    ny=params['grid'].ny
    Re=params['Re']
    omega=params['omega']
    beta=params['beta']
    alpha=params['alpha']

    # extract finite difference operators
    Dy=diffs['Dy']
    Dyy=diffs['Dyy']

    # extract base flow
    U=np.diag(baseflow.U)
    Uy=np.diag(baseflow.Uy)
    Ux=np.diag(baseflow.Ux)
    V = np.diag(baseflow.V)
    Vy = np.diag(baseflow.Vy)

    # extract helper matrices
    I=helper_mats['I'] #np.eye(ny)
    zero=helper_mats['zero'] #np.zeros((ny,ny))

    # set imaginary i and delta
    i=1.j
    delta=-i*omega*I + i*alpha*U + ((I*alpha**2 + I*beta**2)*1./Re)

    # set and return C,By,Byy, and A
    C = np.block([
        # u         v           w           P
        [delta+Ux,  Uy,         zero,       i*alpha*I], # u-mom
        [zero,      delta+Vy,   zero,       zero     ], # v-mom
        [zero,      zero,       delta,      i*beta*I ], # w-mom
        [i*alpha*I, zero,       i*beta*I,   zero  ]  # continuity
    ])
    By = np.block([
        # u         v           w       P
        [V@Dy,      zero,       zero,   zero     ], # u-mom
        [zero,      V@Dy,       zero,   Dy       ], # v-mom
        [zero,      zero,       V@Dy,   zero     ], # w-mom
        [zero,      Dy,         zero,   zero     ]  # continuity
    ])
    Byy = np.block([
        # u         v         w      P
        [-1./Re*Dyy,zero,     zero,  zero     ], # u-mom
        [zero,      -1./Re*Dyy,zero, zero     ], # v-mom
        [zero,      zero,     -1./Re*Dyy,zero ], # w-mom
        [zero,      zero,     zero,  zero     ]  # continuity
    ])
    A = np.block([
        # u         v         w      P
        [U,        zero,     zero,  I        ], # u-mom
        [zero,      U,       zero,  zero     ], # v-mom
        [zero,      zero,     U,    zero     ], # w-mom
        [I,         zero,     zero,  zero     ]  # continuity
    ])
    return C,By,Byy,A

def set_SPE2D_ImplicitEuler(params:dict,C:np.ndarray,By:np.ndarray,Byy:np.ndarray,A:np.ndarray,q_old:np.ndarray,qxx_old=[None,]) -> list:
    ''' set SPE implicit euler integration step using SPE operators to return Aq=b linear system

    Inputs:
        params:dict containting grid (wall normal discrete points), Re (Reynolds number), and hx (streamwise step size)
        C:np.ndarray containing the C*q operator
        By:np.ndarray containing the By*qy operator
        Byy:np.ndarray containing the Byy*qyy operator
        A:np.ndarray containing the A*qx operator
        q_old:np.ndarray containing the current streamwise location perturbations quantities
        qxx_old: vector containing the d2qdx2 quantities of the perturbation quantities (to be iterated) (default to [None,] if not used)

    Returns:
        ASPE:np.ndarray left hand side operator for linear system solve
        bSPE:np.ndarray right hand vector for linear system solve
    '''

    # extract parameters
    ny=params['grid'].ny
    hx=params['hx']
    Re=params['Re']

    # Backward Euler and return linear system (without BCs)
    ASPE = C+By+Byy+(A*(1./hx))
    if qxx_old[0]==None:
        bSPE = ((1./hx)*A)@q_old
    else:
        bSPE = ((1./hx)*A)@q_old + 1./Re*qxx_old
    return ASPE,bSPE

def set_SPE2D_BCs(params:dict,ASPE:np.ndarray,bSPE:np.ndarray) -> list:
    ''' set SPE boundary conditions on linear system Aq=b zero at wall and freestream in momentum equations

    Inputs:
        params:dict containting grid (wall normal discrete points)
        ASPE:np.ndarray left hand side operator for linear system solve
        bSPE:np.ndarray right hand vector for linear system solve
    Returns:
        ASPE:np.ndarray left hand side operator for linear system solve, with BCs
        bSPE:np.ndarray right hand vector for linear system solve, with BCs

    '''

    # extract parameters
    ny=params['grid'].ny
    # set BCs and return
    # set BCs at wall u=v=w=0 and freestream
    ASPE[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
    ASPE[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
    bSPE[[0*ny-0,1*ny-0,2*ny-0]] = 0.
    bSPE[[1*ny-1,2*ny-1,3*ny-1]] = 0.
    ASPE[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.
    ASPE[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.
    #bPSE[[0*ny-0,1*ny-0,2*ny-0]] = 0.
    #bPSE[[1*ny-1,2*ny-1,3*ny-1]] = 0.
    return ASPE,bSPE


def march_SPE2D_one_step(params:dict,diffs:dict,baseflow:base_flow_class,q_old:np.ndarray,helper_mats:dict,qxx_old=[None,]) -> list:
    ''' March the 2D SPE equations one streamwise step

    This uses the functions
        - set_SPE2D_operators
        - set_SPE2D_ImplicitEuler
        - set_SPE2D_BCs
    and then solves the linear system for the next marched location
    
    Inputs:
        params:dict containting grid (wall normal discrete points), Re (Reynolds number), omega (temporal frequency), beta (spanwise wavenumber), hx (streamwise step size), and alpha (streamwise wavenumber)
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        q_old:np.ndarray containing the current streamwise location perturbations quantities
        helper_mats:dict containing zero and I matrices
        qxx_old: vector containing the d2qdx2 quantities of the perturbation quantities (to be iterated)

    Returns:
        qSPEsolved:np.ndarray next marched step state vector
    '''
    
    C,By,Byy,A = set_SPE2D_operators(params,diffs,baseflow,helper_mats)
    ASPE,bSPE = set_SPE2D_ImplicitEuler(params,C,By,Byy,A,q_old,qxx_old)
    ASPE,bSPE = set_SPE2D_BCs(params,ASPE,bSPE)

    qSPEsolved = np.linalg.solve(ASPE,bSPE)
    #qSPEsolved  = np.linalg.solve(APSE,bPSE)
    #qSPEsolved = np.linalg.inv(APSE)@bPSE
    #qSPEsolved = sp.sparse.linalg.spsolve(APSE,bPSE)
    #qSPEsolved = sp.sparse.linalg.dsolve.spsolve(APSE,bPSE)
    return qSPEsolved

def march_SPE2D_one_step_ExplicitEuler(params:dict,diffs:dict,baseflow:base_flow_class,q_old:np.ndarray,helper_mats:dict,qxx_old=[None,]) -> list:
    ''' March the 2D SPE equations one streamwise step using Explicit Euler equations

    Inputs:
        params:dict containting grid (wall normal discrete points), Re (Reynolds number), omega (temporal frequency), beta (spanwise wavenumber), hx (streamwise step size), and alpha (streamwise wavenumber)
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        q_old:np.ndarray containing the current streamwise location perturbations quantities
        helper_mats:dict containing zero and I matrices
        qxx_old: vector containing the d2qdx2 quantities of the perturbation quantities (to be iterated)

    Returns:
        q_new:np.ndarray next marched step state vector
    '''
    
    # params
    FSPE = set_SPE2D_ExplicitOperator_with_BCs(params,diffs,baseflow,helper_mats)
    #fSPE = -FSPE@q_old
    fSPE = FSPE@q_old
    # set BCs on vector for Explicit Euler marching
    #fSPE[(0,ny-1,ny,2*ny-1,2*ny,3*ny-1),] = 0
    # Explicit Euler marching and return new vector
    q_new = q_old + params['hx']*fSPE
    print('ExplicitEuler marching')
    return q_new

def march_SPE2D_one_step_RK4(params:dict,diffs:dict,baseflow:base_flow_class,q_old:np.ndarray,helper_mats:dict,qxx_old=[None,]) -> list:
    ''' March the 2D SPE equations one streamwise step using Runge-Kutta 4th order equations

    Inputs:
        params:dict containting grid (wall normal discrete points), Re (Reynolds number), omega (temporal frequency), beta (spanwise wavenumber), hx (streamwise step size), nu, and x
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        q_old:np.ndarray containing the current streamwise location perturbations quantities
        helper_mats:dict containing zero and I matrices
        qxx_old: vector containing the d2qdx2 quantities of the perturbation quantities (to be iterated)

    Returns:
        q_new:np.ndarray next marched step state vector
    '''
    
    # params
    hx=params['hx']
    nu=params['nu']
    x=params['x']
    y=params['grid'].y
    # helper_mats
    Eplus_EH = helper_mats['Eplus_EH']

    # solve baseflow at middle and at next step
    baseflow_half = blasius(y,x=x+hx/2.,nu=nu,plot=False)
    baseflow_next = blasius(y,x=x+hx,nu=nu,plot=False)

    # solve k1,k2,k3,k4
    F1 = set_SPE2D_ExplicitOperator_with_BCs(params,diffs,baseflow      ,helper_mats)
    F23 = set_SPE2D_ExplicitOperator_with_BCs(params,diffs,baseflow_half ,helper_mats)
    F4 = set_SPE2D_ExplicitOperator_with_BCs(params,diffs,baseflow_next ,helper_mats)
    k1 = hx*F1 @(Eplus_EH@(q_old      ))
    k2 = hx*F23@(Eplus_EH@(q_old+k1/2.))
    k3 = hx*F23@(Eplus_EH@(q_old+k2/2.))
    k4 = hx*F4 @(Eplus_EH@(q_old+k3   ))

    # RK4 marching and return new vector
    q_new = q_old + 1./6.*(k1+2.*(k2+k3)+k4)
    print('RK4 marching')
    return q_new

def set_SPE2D_E_oplus(params:dict,helper_mats:dict,eigvals,eigfuncl,eigfuncr):
    ''' set 2D E_oplus operator by using dLdomega and adjoints
        Inputs:
            params:dict containting grid (wall normal discrete points), Re (Reynolds number), omega (temporal frequency), beta (spanwise wavenumber), hx (streamwise step size), and alpha (streamwise wavenumber)
            helper_mats:dict containing zero and I matrices
            eigvals:
            eigfuncl:
            eigfuncr:
        Returns:
            eig_to_keep:np.array index of eigenvalues to keep, access downstream-traveling and physical modes by eigvals[eig_to_keep]
            cg:np.array group velocity for each non-infinite eigenvalue
            E_oplus:np.ndarray E_oplus operator with zeros for upstream-traveling modes (normalized for inner_product(E_oplus[:,0],eigfuncl[:,0]):=1)
    '''

    if ifflag(params,'LST2'):
        print('set_SPE2D LST2')
        omega=params['omega']
        dLdomega = helper_mats['dLdomega']
        M = helper_mats['M']
        # form E_oplus
        # form Eplus and E^H 
        E_oplus = np.copy(eigfuncr)
        # norm eigenvalues
        # initialize integer counters and C_norm and dalpha_domega
        #C_norm=np.zeros_like(eigvals)
        # normalize E
        for eigi in np.arange(len(eigvals)):
            C_norm = inner_product(E_oplus[:,eigi],M@eigfuncl[:,eigi])
            if C_norm==0:
                #print('C_norm=0 at eigi = ',eigi)
                #print(' alpha = ',eigvals[eigi])
                C_norm=1.
                E_oplus[:,eigi]=0.
            normalized_eigi=E_oplus[:,eigi]/C_norm
            if (np.abs(eigvals[eigi])!=np.inf):
                E_oplus[:,eigi]=normalized_eigi
            
        dalpha_domega = np.zeros_like(eigvals)
        left_going_count=0
        right_going_too_fast=0
        dalphadomega_inf=0
        cp_too_big=0
        cp_too_big_eigi=[]
        cp_too_small=0
        neig_imag_50=0
        eig_imag_20=0
        eig_real_too_big=0
        eig_imag_too_big=0
        cp_too_small_eigi=[]
        inf_going_count=0
        zero_going_count=0
        zero_going_disregard=[]
        eigs_to_keep=[]
        tol=1E-4
        if False: # typical LST2 projection
            for eigi in np.arange(len(eigvals)):
                if np.abs(eigvals[eigi])==np.inf: # remove infinity valued eigenvalues
                    inf_going_count+=1
                    E_oplus[:,eigi] = 0.
                    dalpha_domega[eigi] = 1e-16
                #elif np.abs(eigvals[eigi])<=1.E-7: # remove close to zero valued eigenvalues
                    #zero_going_count+=1
                    #E_dagger[:,eig] = 0.
                    #zero_going_disregard.append(eigi)
                    #print('remove zero_going_count')
                else:
                    #Emaxabs=np.max(np.abs(E[:,eigi]))
                    #E_oplus[:,eigi] = eigfuncr[:,eigi]#/Emaxabs
                    #C_norm[eigi] = inner_product(E_oplus[:,eigi],eigfuncl[:,eigi])
                    #dalpha_domega[eigi] = (dLdomega@E_oplus[:,eigi]).dot(eigfuncl[:,eigi].conj())/C_norm[eigi]
                    dalpha_domega[eigi] = inner_product(dLdomega@E_oplus[:,eigi],eigfuncl[:,eigi])#/C_norm[eigi]
                    if np.abs(dalpha_domega[eigi]) <= 1e-16 : # avoid division by zero
                        left_going_count+=1
                        E_oplus[:,eigi] = 0. 
                        dalpha_domega[eigi] = 1e-16
                    elif (1./(dalpha_domega[eigi].real))<=tol: # left going by group velocity
                        left_going_count+=1
                        E_oplus[:,eigi] = 0. 
                    #elif (1./dalpha_domega[eigi].real)>=1.: # right going too fast by group velocity
                        #right_going_too_fast+=1
                        #E_oplus[eigi,:] = 0. 
                    #elif (np.abs(dalpha_domega[eigi])==np.inf):
                        #dalphadomega_inf+=1
                        #E_oplus[eigi,:] = 0. 
                    #elif (omega/(eigvals[eigi].real))>=1.+tol:
                    elif (omega/(eigvals[eigi].real))>=1.+tol:
                        cp_too_big+=1
                        E_oplus[:,eigi] = 0. 
                        cp_too_big_eigi.append(eigi)
                    elif (omega/(eigvals[eigi].real))<=tol:
                    #elif (omega/(eigvals[eigi])).real<=tol:
                        cp_too_small+=1
                        E_oplus[:,eigi] = 0. 
                        cp_too_small_eigi.append(eigi)
                    elif eigvals[eigi].imag<-0.20:
                        neig_imag_50+=1
                        E_oplus[:,eigi] = 0. 
                        #print('(imag<-20) * 1.7208: ',eigvals[eigi]*1.7208)
                        #print('(imag<-20) : ',eigvals[eigi])
                        #print('alpha.imag<0 when eig=',eigvals[eigi],' at index=',eigi)
                        #print('should disregard due to imag<-4 and cg=',cg[eigi],' and cp=',cp[eigi],' with index=',eigi,' and alpha=',eigvals[eigi])
                        #if eigi==130:
                            #E_oplus[eigi,:] = 0. 
                            #print('disregarding eiginavalue ',eigi,' with cp=',cp[eigi],' and cg=',cg[eigi])
                    elif eigvals[eigi].imag>0.20:
                        eig_imag_20+=1
                        E_oplus[:,eigi] = 0. 
                        #print('alpha.imag>1.25 = ',eigvals[eigi])
                        #print(' cg = ',1./dalpha_domega[eigi])
                        #print(' cp = ',omega/eigvals[eigi])
                    ##elif np.abs(eigvals[eigi].real)>(np.pi*2./(50.*hx)):
                    #elif np.abs(eigvals[eigi].real)>20:
                        #eig_real_too_big+=1
                        #E_oplus[:,eigi] = 0. 
                    #elif (eigvals[eigi].imag)<-(1./(100.*hx)):
                        #eig_imag_too_big+=1
                        #E_oplus[eigi,:] = 0. 
                    else:
                        eigs_to_keep.append(eigi)
            #print('disregarding %d inf eigenvalues'%inf_going_count)
            #print('disregarding %d zero eigenvalues'%zero_going_count)
            #print('disregarding %d left going modes by group velocity'%left_going_count)
            #print('disregarding %d right going modes by group velocity too fast'%right_going_too_fast)
            #print('disregarding %d dalphadomega inf modes'%dalphadomega_inf)
            #print('disregarding %d cp too big by phase velocity'%cp_too_big)
            #print('disregarding %d cp too small by phase velocity'%cp_too_small)
            #print('disregarding %d alpha.imag<-20 modes'%neig_imag_50)
            #print('disregarding %d alpha.imag>1 modes'%eig_imag_20)
            #print('disregarding %d alpha.real too big modes'%eig_real_too_big)
            #print('disregarding %d alpha.imag too big modes'%eig_imag_too_big)
            cg=1./dalpha_domega
            #E_oplus = np.eye(eigfuncr.shape[0]) # no filtering...
            return eigs_to_keep,cg,E_oplus
        if True: # typical LST2 projection with zero crossings
            zcr_to_large = 0
            uvL2_to_large = 0
            for eigi in np.arange(len(eigvals)):
                if np.abs(eigvals[eigi])==np.inf: # remove infinity valued eigenvalues
                    inf_going_count+=1
                    E_oplus[:,eigi] = 0.
                    dalpha_domega[eigi] = 1e-16
                else:
                    dalpha_domega[eigi] = inner_product(dLdomega@E_oplus[:,eigi],eigfuncl[:,eigi])#/C_norm[eigi]
                    zcri = max_zero_crossings_percent(params,E_oplus[:,eigi])
                    u,v,w,p,au,av,aw,ap = E_oplus[:,eigi].reshape(8,-1)
                    uvL2 = np.linalg.norm(u) + np.linalg.norm(v)
                    if np.abs(dalpha_domega[eigi]) <= 1e-16 : # avoid division by zero
                        left_going_count+=1
                        E_oplus[:,eigi] = 0. 
                        dalpha_domega[eigi] = 1e-16
                    elif (1./(dalpha_domega[eigi].real))<=tol: # left going by group velocity
                        left_going_count+=1
                        E_oplus[:,eigi] = 0. 
                    elif (omega/(eigvals[eigi].real))>=1.+tol:
                        cp_too_big+=1
                        E_oplus[:,eigi] = 0. 
                        cp_too_big_eigi.append(eigi)
                    elif (omega/(eigvals[eigi].real))<=tol:
                        cp_too_small+=1
                        E_oplus[:,eigi] = 0. 
                        cp_too_small_eigi.append(eigi)
                    elif eigvals[eigi].imag<-1.023:
                        neig_imag_50+=1
                        E_oplus[:,eigi] = 0. 
                    elif eigvals[eigi].imag>1.023:
                        eig_imag_20+=1
                        E_oplus[:,eigi] = 0. 
                    #elif zcri >= 0.5:
                        #zcr_to_large += 1
                        #E_oplus[:,eigi] = 0. 
                    #elif uvL2 <= tol:
                        #uvL2_to_large += 1
                        #E_oplus[:,eigi] = 0. 
                    else:
                        eigs_to_keep.append(eigi)
            #print('disregarding %d inf eigenvalues'%inf_going_count)
            #print('disregarding %d zero eigenvalues'%zero_going_count)
            #print('disregarding %d left going modes by group velocity'%left_going_count)
            #print('disregarding %d right going modes by group velocity too fast'%right_going_too_fast)
            #print('disregarding %d dalphadomega inf modes'%dalphadomega_inf)
            #print('disregarding %d cp too big by phase velocity'%cp_too_big)
            #print('disregarding %d cp too small by phase velocity'%cp_too_small)
            print('disregarding %d zcr too large'%zcr_to_large)
            print('disregarding %d uvL2 too large'%uvL2_to_large)
            #print('disregarding %d alpha.imag<-20 modes'%neig_imag_50)
            #print('disregarding %d alpha.imag>1 modes'%eig_imag_20)
            #print('disregarding %d alpha.real too big modes'%eig_real_too_big)
            #print('disregarding %d alpha.imag too big modes'%eig_imag_too_big)
            cg=1./dalpha_domega.real
            #E_oplus = np.eye(eigfuncr.shape[0]) # no filtering...
            return eigs_to_keep,cg,E_oplus
        else:
            alphaTS = params['alphaTS']
            eigi = find_nearest_arg(eigvals,alphaTS)
            params['alphaTS'] = eigvals[eigi]
            eigfTS = E_oplus[:,eigi].copy()
            E_oplus[:] = 0.
            E_oplus[:,0] = eigfTS.copy()
            print('  only one mode')
            return None,None,E_oplus

    elif ifflag(params,'LSTNP'):
        print('set_SPE2D LSTNP')
        omega=params['omega']
        dLdomega = helper_mats['dLdomega']
        M = helper_mats['M']
        # form E_oplus
        # form Eplus and E^H 
        E_oplus = np.copy(eigfuncr)
        # norm eigenvalues
        # initialize integer counters and C_norm and dalpha_domega
        #C_norm=np.zeros_like(eigvals)
        # normalize E
        for eigi in np.arange(len(eigvals)):
            C_norm = inner_product(E_oplus[:,eigi],M@eigfuncl[:,eigi])
            if C_norm==0:
                #print('C_norm=0 at eigi = ',eigi)
                #print(' alpha = ',eigvals[eigi])
                C_norm=1.
                E_oplus[:,eigi]=0.
            normalized_eigi=E_oplus[:,eigi]/C_norm
            if (np.abs(eigvals[eigi])!=np.inf):
                E_oplus[:,eigi]=normalized_eigi
            
        dalpha_domega = np.zeros_like(eigvals)
        left_going_count=0
        right_going_too_fast=0
        dalphadomega_inf=0
        cp_too_big=0
        cp_too_big_eigi=[]
        cp_too_small=0
        neig_imag_50=0
        eig_imag_20=0
        eig_real_too_big=0
        eig_imag_too_big=0
        cp_too_small_eigi=[]
        inf_going_count=0
        zero_going_count=0
        zero_going_disregard=[]
        eigs_to_keep=[]
        tol=1E-4
        zcr_to_large = 0
        uvL2_to_large = 0
        for eigi in np.arange(len(eigvals)):
            if np.abs(eigvals[eigi])==np.inf: # remove infinity valued eigenvalues
                inf_going_count+=1
                E_oplus[:,eigi] = 0.
                dalpha_domega[eigi] = 1e-16
            else:
                dalpha_domega[eigi] = inner_product(dLdomega@E_oplus[:,eigi],eigfuncl[:,eigi])#/C_norm[eigi]
                zcri = max_zero_crossings_percent(params,E_oplus[:,eigi])
                u,v,w,p,au,av,aw,ap = E_oplus[:,eigi].reshape(8,-1)
                uvL2 = np.linalg.norm(u) + np.linalg.norm(v)
                if np.abs(dalpha_domega[eigi]) <= 1e-16 : # avoid division by zero
                    left_going_count+=1
                    E_oplus[:,eigi] = 0. 
                    dalpha_domega[eigi] = 1e-16
                elif np.abs(dalpha_domega[eigi].real) <= 1e-16 : # avoid division by zero
                    left_going_count+=1
                    E_oplus[:,eigi] = 0. 
                    dalpha_domega[eigi] = 1e-16
                elif (1./(dalpha_domega[eigi].real))<=tol: # left going by group velocity
                    left_going_count+=1
                    E_oplus[:,eigi] = 0. 
                elif (omega/(eigvals[eigi].real))>=1.+tol:
                    cp_too_big+=1
                    E_oplus[:,eigi] = 0. 
                    cp_too_big_eigi.append(eigi)
                elif (omega/(eigvals[eigi].real))<=tol:
                    cp_too_small+=1
                    E_oplus[:,eigi] = 0. 
                    cp_too_small_eigi.append(eigi)
                elif eigvals[eigi].imag<-0.1: #1.023:
                    neig_imag_50+=1
                    E_oplus[:,eigi] = 0. 
                elif eigvals[eigi].imag>0.1: #1.023:
                    eig_imag_20+=1
                    E_oplus[:,eigi] = 0. 
                #elif zcri >= 0.5:
                    #zcr_to_large += 1
                    #E_oplus[:,eigi] = 0. 
                #elif uvL2 <= tol:
                    #uvL2_to_large += 1
                    #E_oplus[:,eigi] = 0. 
                else:
                    eigs_to_keep.append(eigi)
        #print('disregarding %d inf eigenvalues'%inf_going_count)
        #print('disregarding %d zero eigenvalues'%zero_going_count)
        #print('disregarding %d left going modes by group velocity'%left_going_count)
        #print('disregarding %d right going modes by group velocity too fast'%right_going_too_fast)
        #print('disregarding %d dalphadomega inf modes'%dalphadomega_inf)
        #print('disregarding %d cp too big by phase velocity'%cp_too_big)
        #print('disregarding %d cp too small by phase velocity'%cp_too_small)
        print('disregarding %d zcr too large'%zcr_to_large)
        print('disregarding %d uvL2 too large'%uvL2_to_large)
        #print('disregarding %d alpha.imag<-20 modes'%neig_imag_50)
        #print('disregarding %d alpha.imag>1 modes'%eig_imag_20)
        #print('disregarding %d alpha.real too big modes'%eig_real_too_big)
        #print('disregarding %d alpha.imag too big modes'%eig_imag_too_big)
        cg=1./dalpha_domega.real
        #E_oplus = np.eye(eigfuncr.shape[0]) # no filtering...
        return eigs_to_keep,cg,E_oplus
    else:
        omega=params['omega']
        dLdomega = helper_mats['dLdomega']
        # form E_oplus
        # form Eplus and E^H 
        E_oplus = np.copy(eigfuncr)
        # norm eigenvalues
        # initialize integer counters and C_norm and dalpha_domega
        #C_norm=np.zeros_like(eigvals)
        # normalize E
        for eigi in np.arange(len(eigvals)):
            C_norm = inner_product(E_oplus[:,eigi],eigfuncl[:,eigi])
            normalized_eigi=E_oplus[:,eigi]/C_norm
            if (np.abs(eigvals[eigi])!=np.inf):
                E_oplus[:,eigi]=normalized_eigi
            
        dalpha_domega = np.zeros_like(eigvals)
        left_going_count=0
        right_going_too_fast=0
        dalphadomega_inf=0
        cp_too_big=0
        cp_too_big_eigi=[]
        cp_too_small=0
        neig_imag_50=0
        eig_imag_20=0
        eig_real_too_big=0
        eig_imag_too_big=0
        cp_too_small_eigi=[]
        inf_going_count=0
        zero_going_count=0
        zero_going_disregard=[]
        eigs_to_keep=[]
        tol=1E-4
        for eigi in np.arange(len(eigvals)):
            if np.abs(eigvals[eigi])==np.inf: # remove infinity valued eigenvalues
                inf_going_count+=1
                E_oplus[:,eigi] = 0.
            #elif np.abs(eigvals[eigi])<=1.E-7: # remove close to zero valued eigenvalues
                #zero_going_count+=1
                #E_dagger[eigi,:] = 0.
                #zero_going_disregard.append(eigi)
            else:
                #Emaxabs=np.max(np.abs(E[:,eigi]))
                #E_oplus[:,eigi] = eigfuncr[:,eigi]#/Emaxabs
                #C_norm[eigi] = inner_product(E_oplus[:,eigi],eigfuncl[:,eigi])
                #dalpha_domega[eigi] = (dLdomega@E_oplus[:,eigi]).dot(eigfuncl[:,eigi].conj())/C_norm[eigi]
                dalpha_domega[eigi] = inner_product(dLdomega@E_oplus[:,eigi],eigfuncl[:,eigi])#/C_norm[eigi]
                if (1./(dalpha_domega[eigi].real))<=tol: # left going by group velocity
                    left_going_count+=1
                    E_oplus[:,eigi] = 0. 
                #elif (1./dalpha_domega[eigi].real)>=1.: # right going too fast by group velocity
                    #right_going_too_fast+=1
                    #E_oplus[eigi,:] = 0. 
                #elif (np.abs(dalpha_domega[eigi])==np.inf):
                    #dalphadomega_inf+=1
                    #E_oplus[eigi,:] = 0. 
                elif (omega/(eigvals[eigi].real))>1.+tol:
                    cp_too_big+=1
                    E_oplus[:,eigi] = 0. 
                    cp_too_big_eigi.append(eigi)
                elif (omega/(eigvals[eigi].real))<=tol:
                    cp_too_small+=1
                    E_oplus[:,eigi] = 0. 
                    cp_too_small_eigi.append(eigi)
                elif eigvals[eigi].imag<-0.2:
                    neig_imag_50+=1
                    E_oplus[:,eigi] = 0. 
                    #print('(imag<-20) * 1.7208: ',eigvals[eigi]*1.7208)
                    #print('(imag<-20) : ',eigvals[eigi])
                    #print('alpha.imag<0 when eig=',eigvals[eigi],' at index=',eigi)
                    #print('should disregard due to imag<-4 and cg=',cg[eigi],' and cp=',cp[eigi],' with index=',eigi,' and alpha=',eigvals[eigi])
                    #if eigi==130:
                        #E_oplus[eigi,:] = 0. 
                        #print('disregarding eiginavalue ',eigi,' with cp=',cp[eigi],' and cg=',cg[eigi])
                elif eigvals[eigi].imag>0.2:
                    eig_imag_20+=1
                    E_oplus[:,eigi] = 0. 
                ##elif np.abs(eigvals[eigi].real)>(np.pi*2./(50.*hx)):
                #elif np.abs(eigvals[eigi].real)>20:
                    #eig_real_too_big+=1
                    #E_oplus[:,eigi] = 0. 
                #elif (eigvals[eigi].imag)<-(1./(100.*hx)):
                    #eig_imag_too_big+=1
                    #E_oplus[eigi,:] = 0. 
                else:
                    eigs_to_keep.append(eigi)
        #print('disregarding %d inf eigenvalues'%inf_going_count)
        #print('disregarding %d zero eigenvalues'%zero_going_count)
        #print('disregarding %d left going modes by group velocity'%left_going_count)
        #print('disregarding %d right going modes by group velocity too fast'%right_going_too_fast)
        #print('disregarding %d dalphadomega inf modes'%dalphadomega_inf)
        #print('disregarding %d cp too big by phase velocity'%cp_too_big)
        #print('disregarding %d cp too small by phase velocity'%cp_too_small)
        #print('disregarding %d alpha.imag<-20 modes'%neig_imag_50)
        #print('disregarding %d alpha.imag>20 modes'%eig_imag_20)
        #print('disregarding %d alpha.real too big modes'%eig_real_too_big)
        #print('disregarding %d alpha.imag too big modes'%eig_imag_too_big)
        cg=1./dalpha_domega
        #E_oplus = np.eye(eigfuncr.shape[0]) # no filtering...
        return eigs_to_keep,cg,E_oplus

# E^+ and E^H functions
def remove_zero_col(mat):
    '''returns a slice of the matrix that does not have zero columns '''
    #zero_cols=np.where(~mat.any(axis=0))[0]
    return mat[:,~np.all(mat == 0, axis=0)]

def add_with_zeros_and_fullrank(mat,orthogonalE):
    ''' Add a n by n-k matrix that is rank deficient, pad with random and make full rank.  Then perform QR decomposition and export the desired matrices
    
    Parameters:
       - A - Matrix A that is n by n-k
    Returns:
       -Q0 - matrix that is n by n,  where first columns are the orthogonal basis of A and the remaining n-k columns are padded with zeros to the right side
       -Qr - orthogonal matrix that is n by n, of the input A matrix padded with random numbers
   '''
    if orthogonalE: # works with orthogonal subspace
        n,k = mat.shape
        rand1=np.random.rand(n,n-k)+np.random.rand(n,n-k)*1.j
        mat_rand = np.block([
            [mat,rand1],
        ])
        mat_rand_orth,_ = sp.linalg.qr(mat_rand,pivoting=False)
        #return (np.block([mat_rand_orth[:,:k],np.zeros((n,n-k))]),mat_rand_orth)
        return mat_rand_orth[:,:k],mat_rand_orth[:,:k]
    else: # non-orthogonal E
        n,k = mat.shape
        rand1=np.random.rand(n,n-k)+np.random.rand(n,n-k)*1.j
        mat_rand = np.block([
            [mat,rand1],
        ])
        mat_rand_orth,_ = sp.linalg.qr(mat_rand,pivoting=False)
        #return (np.block([mat,np.zeros((n,n-k))]),np.block([mat,mat_rand_orth[:,k:]]))
        return (np.block([mat,np.zeros((n,n-k))]),np.block([mat,mat_rand_orth[:,k:]]))

def get_SPE2D_Eplus_and_EH_from_4ny_x_4ny(mat,orthogonalE=True):
    '''Return the Eplus and E^H of a matrix
    Take a matrix that has n rows and many zero columns,  
    remove the zero columns to obtain a rank deficient n by n-k matrix,  
    then perform QR decomposition and extract out matrices
    Parameters:
        - mat - matrix that is square with n rows and many zero columns
    Returns:
        - E^plus orthogonal basis of mat and padded with zeros to make it square
        - E^-1 inverse of basis of mat padded with random numbers
    '''
    Eplus,E =  add_with_zeros_and_fullrank(remove_zero_col(mat),orthogonalE=orthogonalE)
    if orthogonalE:
        return Eplus,E.conj().T # no need to inverse, can just conjugate transpose to get E^-1
    else:
        return Eplus,np.linalg.inv(E)

def get_SPE2D_Eplus_and_EH(params,helper_mats,eigvals,eigfuncl,eigfuncr,orthogonalE=True):
    ''' Returns eigenvalues to keep and the associated E^plus and E^H operators

    Inputs:
        params:dict containing items for set_SPE2D_E_oplus
        helper_mats:dict containing matrices for set_SPE2D_E_oplus and uvwP_from_LST and items for get_SPE2D_Eplus_and_EH_from_4ny_x_4ny
        eigvals:np.ndarray full eigenvalue spectrum of spatial LST
        eigfuncl:np.ndarray containing adjoint eigen functions
        eigfuncr:np.ndarray containing normal eigen functions, ith eigenfunction obtained from eigfuncr[:,i]
        orthogonalE=True: set to True if wanting an orthogonal E matrix to be returned
    Returns:
        eigs_to_keep:np.ndarray containing all the eigenvalues to keep
        cg:np.array group velocity for each eigenfunction
        Eplus:np.ndarray matrix containing downstream-traveling subspace and padded with zeros
        EH:np.ndarray complex conjugate transpose of (full system of downstream-traveling subspace and padded with random vectors)
    '''

    eigs_to_keep,cg,E_oplus = set_SPE2D_E_oplus(params,helper_mats,eigvals,eigfuncl,eigfuncr)
    Eplus,EH = get_SPE2D_Eplus_and_EH_from_4ny_x_4ny(helper_mats['uvwP_from_LST']@E_oplus)
    return eigs_to_keep,cg,Eplus,EH
    
def march_SPE2D_one_step_with_projection(params:dict,diffs:dict,baseflow:base_flow_class,q_old:np.ndarray,helper_mats:dict,qxx_old=[None,],projection=True) -> list:
    ''' marches the SPE equations one step
    Inputs:
        params: dict containing Re, omega, beta, hx, alpha, and grid
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        q_old:np.ndarray initial perturbation state to be marched
        helper_mats:dict containing zero and I matrices
        qxx_old=[None,]
    Returns:
        Eplus@EH@q_new - perturbation quantities at the next streamwise step and having filtered out the upstream-traveling solutions

    '''

    if projection:
        L,eigvals,eigfuncl,eigfuncr = LST(params,diffs,baseflow,helper_mats)
        #helper_mats['dLdomega'] = dLdomega(params,diffs,baseflow,helper_mats)

        eigs_to_keep,cg,Eplus,EH = get_SPE2D_Eplus_and_EH(params,helper_mats,eigvals,eigfuncl,eigfuncr)

        Eplus_EH=Eplus@EH
        helper_mats['Eplus_EH']=Eplus_EH
        if ifflag(params,'ExplicitEuler'):
            q_new = march_SPE2D_one_step_ExplicitEuler(params,diffs,baseflow,q_old,helper_mats,qxx_old)
        elif ifflag(params,'RK4'):
            q_new = march_SPE2D_one_step_RK4(params,diffs,baseflow,q_old,helper_mats,qxx_old)
        else:
            q_new = march_SPE2D_one_step(params,diffs,baseflow,q_old,helper_mats,qxx_old)
        return Eplus_EH@q_new
    else:
        q_new = march_SPE2D_one_step(params,diffs,baseflow,q_old,helper_mats,qxx_old)
        return q_new

def march_SPE2D_multistep_with_projection(params:dict,diffs:dict,baseflow:base_flow_class,q_init:np.ndarray,helper_mats:dict,qxx_old=[None,],projection=True) -> list:
    ''' marches the SPE equations multiple steps using the projection onto downstream-traveling subspace
    Inputs:
        params: dict containing Re, omega, beta, hx, alpha, grid, x_start, and steps
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        q_init:np.ndarray initial perturbation state to be marched
        helper_mats:dict containing zero and I matrices
        qxx_old=[None,]
    Returns:
        x:np.array containing streamwise locations of data
        q_istep:np.ndarray - perturbation quantities at all streamwise steps and having filtered out the upstream-traveling solutions

    '''
    # extract params
    steps=params['steps']
    hx=params['hx']
    ny=params['grid'].ny
    y=params['grid'].y
    nu=params['nu']
    x_start = params['x_start']
    # create arrays to fill
    x=x_start+np.arange(steps+1)*hx
    q_istep = np.zeros((steps+1,4*ny),dtype=np.complex)
    q_istep[0,:]=q_init

    # filter initial conditions to be one way
    #L,eigvals,eigfuncl,eigfuncr = LST(params,diffs,baseflow,helper_mats)
    #eigs_to_keep,cg,Eplus,EH = get_SPE2D_Eplus_and_EH(params,helper_mats,eigvals,eigfuncl,eigfuncr)
    #Eplus_EH=Eplus@EH
    #q_initoplus=Eplus_EH@q_init
    #q_istep[0,:]=q_initoplus
    # march SPE
    for xi in np.arange(steps):
        params['x']=x[xi]
        if ifflag(params,'RK4') or ifflag(params,'ExplicitEuler'):
            baseflow2 = blasius(y,x=x[xi],nu=nu,plot=False)
        else:
            baseflow2 = blasius(y,x=x[xi+1],nu=nu,plot=False)
        #baseflow2 = blasius(y,x=x_start,nu=nu,plot=False)
        #baseflow2.V[:]=baseflow2.Ux[:]=baseflow2.Vy[:]=0
        q_istep[xi+1,:] = march_SPE2D_one_step_with_projection(params,diffs,baseflow2,q_istep[xi,:],helper_mats,qxx_old,projection)
        print('marched x=',x[xi+1])
    return x,q_istep

def march_parallel_SPE2D_multistep_with_projection(params:dict,diffs:dict,baseflow:base_flow_class,q_init:np.ndarray,helper_mats:dict,qxx_old=[None,],projection=True) -> np.ndarray:
    ''' marches the parallel SPE equations in 2D multiple steps

    Inputs:
        params: dict containing Re, omega, beta, hx, alpha, grid, and steps
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y (make it 2D before running)
        q_init:np.ndarray initial perturbation state vector
        helper_mats:dict containing zero and I matrices
        qxx_old=[None,]

    Returns:
        x:np.ndarray 1D array of streamwise locations
        q_istep:np.ndarray 2D array containing perturbation quantities at each streamwise step.  xi^th state obtained with q_istep[xi,:]

    '''

    # extract params
    steps=params['steps']
    hx=params['hx']
    ny=params['grid'].ny

    # extract helper_mats
    if projection:
        Eplus_EH=helper_mats['Eplus']@helper_mats['EH']

    # create q_istep array
    x=np.arange(steps+1)*hx
    q_istep = np.zeros((steps+1,4*ny),dtype=np.complex)
    q_istep[0,:]=q_init


    if ifflag(params,'RK4'):
        hx = params['hx']
        F = set_SPE2D_ExplicitOperator_with_BCs(params,diffs,baseflow,helper_mats)
        
        for xi in np.arange(steps):
            if projection:
                q_old = q_istep[xi,:]
                k1 = hx*F@Eplus_EH@(q_old      )
                k2 = hx*F@Eplus_EH@(q_old+k1/2.)
                k3 = hx*F@Eplus_EH@(q_old+k2/2.)
                k4 = hx*F@Eplus_EH@(q_old+k3   )
                q_new = q_old + 1./6.*(k1+2.*(k2+k3)+k4)
                q_istep[xi+1,:] = Eplus_EH@q_new
                print('RK4 step completed at x = ',x[xi])
    elif ifflag(params,'ExplicitEuler'):
        hx = params['hx']
        F = set_SPE2D_ExplicitOperator_with_BCs(params,diffs,baseflow,helper_mats)
        for xi in np.arange(steps):
            if projection:
                q_old = q_istep[xi,:]
                q_new = q_old + hx*F@q_old
                q_istep[xi+1,:] = Eplus_EH@q_new
    else:
        C,By,Byy,A = set_SPE2D_operators(params,diffs,baseflow,helper_mats)
        
        for xi in np.arange(steps):
            ASPE,bSPE = set_SPE2D_ImplicitEuler(params,C,By,Byy,A,q_istep[xi,:],qxx_old)
            ASPE,bSPE = set_SPE2D_BCs(params,ASPE,bSPE)
            if projection:
                q_istep[xi+1,:] = Eplus_EH@np.linalg.solve(ASPE,bSPE)
            else:
                q_istep[xi+1,:] = np.linalg.solve(ASPE,bSPE)

    return x,q_istep


