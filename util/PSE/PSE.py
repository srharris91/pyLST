import numpy as np
import scipy as sp
# import scipy.sparse
from ..Classes import base_flow_class
from ..base_flow import blasius
from ..math import inner_product
from ..math import intq_dy
from ..helper import ifflag
from ..LST import LST
from ..SPE import set_SPE2D_E_oplus
from ..SPE import get_SPE2D_Eplus_and_EH_from_4ny_x_4ny

def step_PSE_test1(params,diffs,baseflow,q_old,qxx_old,hx,alpha,helper_mats):
    Re=params['Re']
    omega=params['omega']
    beta=params['beta']
    Dy=diffs['Dy']
    Dyy=diffs['Dyy']
    ny=Dy.shape[0]
    U=np.diag(baseflow.U)
    Uy=np.diag(baseflow.Uy)
    Ux=np.diag(baseflow.Ux)
    V = np.diag(baseflow.V)
    Vy = np.diag(baseflow.Vy)
    I=helper_mats['I'] #np.eye(ny)
    zero=helper_mats['zero'] #np.zeros((ny,ny))
    i=1.j
    #hx=0.0001
    #alpha=0.
    #alpha=alpha_original
    delta=-i*omega*I + i*alpha*U + ((I*alpha**2 + I*beta**2)*1./Re)
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

    # Backward Euler
    APSE = C+By+Byy+(A*(1./hx))
    bPSE = ((1./hx)*A)@q_old + 1./Re*qxx_old

    # set BCs
    # set BCs at wall u=v=w=0
    APSE[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
    APSE[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
    bPSE[[0*ny-0,1*ny-0,2*ny-0]] = 0.
    bPSE[[1*ny-1,2*ny-1,3*ny-1]] = 0.
    APSE[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.
    APSE[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.
    #bPSE[[0*ny-0,1*ny-0,2*ny-0]] = 0.
    #bPSE[[1*ny-1,2*ny-1,3*ny-1]] = 0.



    qPSEsolved = np.linalg.solve(APSE,bPSE)
    #qPSEsolved  = np.linalg.solve(APSE,bPSE)
    #qPSEsolved = np.linalg.inv(APSE)@bPSE
    #qPSEsolved = sp.sparse.linalg.spsolve(APSE,bPSE)
    #qPSEsolved = sp.sparse.linalg.dsolve.spsolve(APSE,bPSE)
    return qPSEsolved

def set_PSE2D_operators(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> list:
    ''' set PSE operators for C*q + By*qy + Byy*qyy + A*qx = 0
    Inputs:
        params:dict containting grid, Re, omega, beta, alpha, and flags neglect_dPdx_term
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
    #print(i,omega,I,alpha,U,beta,Re)
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
    if not ifflag(params,'neglect_dPdx_term'):
        print('dPdx_term=',True)
        A = np.block([
            # u         v         w      P
            [U,        zero,     zero,  I        ], # u-mom
            [zero,      U,       zero,  zero     ], # v-mom
            [zero,      zero,     U,    zero     ], # w-mom
            [I,         zero,     zero,  zero     ]  # continuity
        ])
    else:
        print('dPdx_term=',False)
        A = np.block([
            # u         v         w      P
            [U,        zero,     zero,  zero     ], # u-mom
            [zero,      U,       zero,  zero     ], # v-mom
            [zero,      zero,     U,    zero     ], # w-mom
            [I,         zero,     zero,  zero     ]  # continuity
        ])
    return C,By,Byy,A

def update_C_PSE2D_operator(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> list:
    ''' update PSE operator C in C*q + By*qy + Byy*qyy + A*qx = 0 with new alpha value
    Inputs:
        params:dict containting grid, Re, omega, beta, and alpha
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        helper_mats:dict containing zero and I matrices
    Returns:
        C:np.ndarray containing the C*q operator
        
    '''

    # extract parameters from params
    ny=params['grid'].ny
    Re=params['Re']
    omega=params['omega']
    beta=params['beta']
    alpha=params['alpha']

    # extract finite difference operators

    # extract base flow
    U=np.diag(baseflow.U)
    Uy=np.diag(baseflow.Uy)
    Ux=np.diag(baseflow.Ux)
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
    return C

def set_PSE2D_ImplicitEuler(params:dict,C:np.ndarray,By:np.ndarray,Byy:np.ndarray,A:np.ndarray,q_old:np.ndarray) -> list:
    ''' set PSE implicit euler integration step using PSE operators to return Aq=b linear system

    Inputs:
        params:dict containting grid (wall normal discrete points), Re (Reynolds number), and hx (streamwise step size)
        C:np.ndarray containing the C*q operator
        By:np.ndarray containing the By*qy operator
        Byy:np.ndarray containing the Byy*qyy operator
        A:np.ndarray containing the A*qx operator
        q_old:np.ndarray containing the current streamwise location perturbations quantities

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
    bSPE = ((1./hx)*A)@q_old
    return ASPE,bSPE

def set_PSE2D_BCs(params:dict,ASPE:np.ndarray,bSPE:np.ndarray) -> list:
    ''' set PSE boundary conditions on linear system Aq=b zero at wall and freestream in momentum equations

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
    return ASPE,bSPE


def update_alpha_closure(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats,q_old:np.ndarray) -> np.ndarray:
    ''' Update alpha and return new alpha value in params['alpha'] and return q_new
    Inputs:
        params:dict containting grid, alpha, and alpha_closure_tol, and flags neglect_dPdx_term, add_dPdx_term_RHS
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        helper_mats:dict containing zero and I matrices
        q_old:np.ndarray containing the current streamwise location perturbations quantities
    Returns: places new alpha value in params['alpha']
        q_new:np.ndarray containing state after iterating on closure terms

    '''
    def update_alpha_iter(params:dict,q_old:np.ndarray,q_new:np.ndarray):
        ''' update alpha for each iteration '''
        #y = params['y']
        alpha_new = params['alpha'] - (1.j/params['hx'])*(inner_product(q_new-q_old,q_new)/inner_product(q_new,q_new))
        return alpha_new

    if (ifflag(params,'neglect_dPdx_term')) and (ifflag(params,'add_dPdx_term_RHS')): # add dPdx to RHS for x-momentum equation
        print('add_dPdx_term_RHS = ',True)
        hx = params['hx']
        ny = params['grid'].ny
        q_dPdx = np.zeros(4*ny)#,dtype=np.complex)

    # set operators using old alpha
    C,By,Byy,A = set_PSE2D_operators(params,diffs,baseflow,helper_mats)
    APSE,bPSE = set_PSE2D_ImplicitEuler(params,C,By,Byy,A,q_old)
    APSE,bPSE = set_PSE2D_BCs(params,APSE,bPSE)
    q_new = np.linalg.solve(APSE,bPSE)
    #q_new = sp.sparse.linalg.dsolve.spsolve(APSE,bPSE)
    alpha_old = np.copy(params['alpha'])
    # get updated alpha value
    alpha_new = update_alpha_iter(params,q_old,q_new)

    # if changed less than tolerance, then return
    converged=False
    iteration=0
    while not converged:
        iteration+=1
        params['alpha']=np.copy(alpha_new)
        C = update_C_PSE2D_operator(params,diffs,baseflow,helper_mats)
        APSE,bPSE = set_PSE2D_ImplicitEuler(params,C,By,Byy,A,q_old)
        APSE,bPSE = set_PSE2D_BCs(params,APSE,bPSE)
        if (ifflag(params,'neglect_dPdx_term')) and (ifflag(params,'add_dPdx_term_RHS')): # add dPdx to RHS for x-momentum equation
            print('adding it to the RHS')
            P_old = helper_mats['P_from_SPE']@q_old.real
            P_new = helper_mats['P_from_SPE']@q_new.real
            q_dPdx[:ny] = (P_new - P_old)/hx 
            bPSE -= q_dPdx
        q_new = np.linalg.solve(APSE,bPSE)
        #q_new = sp.sparse.linalg.dsolve.spsolve(APSE,bPSE)
        alpha_old = np.copy(params['alpha'])
        alpha_new = update_alpha_iter(params,q_old,q_new)
        # check update value
        delta_alpha = np.abs(alpha_old-alpha_new)
        if delta_alpha<=params['alpha_closure_tol']:
            converged=True
    print('alpha closure iterations = ',iteration)
    #print(' direct solve')
    return q_new


def march_PSE2D_one_step(params:dict,diffs:dict,baseflow:base_flow_class,q_old:np.ndarray,helper_mats:dict) -> list:
    ''' March the 2D SPE equations one streamwise step

    This uses the functions
        - set_PSE2D_operators
        - set_PSE2D_ImplicitEuler
        - set_PSE2D_BCs
    and then solves the linear system for the next marched location
    
    Inputs:
        params:dict containting grid (wall normal discrete points), Re (Reynolds number), omega (temporal frequency), beta (spanwise wavenumber), hx (streamwise step size), and alpha (streamwise wavenumber), flags alpha_update
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        q_old:np.ndarray containing the current streamwise location perturbations quantities
        helper_mats:dict containing zero and I matrices

    Returns:
        qPSEsolved:np.ndarray next marched step state vector
    '''
    if ifflag(params,'alpha_update'):
        qPSEsolved = update_alpha_closure(params,diffs,baseflow,helper_mats,q_old)
    else:
        C,By,Byy,A = set_PSE2D_operators(params,diffs,baseflow,helper_mats)
        APSE,bPSE = set_PSE2D_ImplicitEuler(params,C,By,Byy,A,q_old)
        APSE,bPSE = set_PSE2D_BCs(params,APSE,bPSE)
        qPSEsolved = np.linalg.solve(APSE,bPSE)
        #qSPEsolved  = np.linalg.solve(APSE,bPSE)
        #qSPEsolved = np.linalg.inv(APSE)@bPSE
        #qSPEsolved = sp.sparse.linalg.spsolve(APSE,bPSE)
        #qPSEsolved = sp.sparse.linalg.dsolve.spsolve(APSE,bPSE)
    return qPSEsolved

def march_PSE2D_multistep(params:dict,diffs:dict,baseflow:base_flow_class,q_init:np.ndarray,helper_mats:dict,parallel=False,projection=False) -> list:
    ''' marches the PSE equations multiple steps
    Inputs:
        params: dict containing Re, steps, omega, beta, hx, alpha, ny, and x_start, and flags alpha_update
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        q_init:np.ndarray initial perturbation state to be marched
        helper_mats:dict containing zero and I matrices
    Returns:
        alphas:np.array containing all alpha values
        x:np.array containing all streamwise locations starting from x_start (shape steps+1)
        q_istep - perturbation quantities at the next streamwise step (shape steps+1,4*ny)

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
    alphas=np.zeros_like(x)+0.j
    alphas[0]=params['alpha']
    q_istep = np.zeros((steps+1,4*ny),dtype=np.complex)
    q_istep[0,:]=q_init
    baseflows=[baseflow,]

    # global baseflow test
    #DNS_baseflow = helper_mats['DNS_baseflow']
    Dy = diffs['Dy']
    if projection:
        print('get projecting operator for first step')
        L,M,eigvals,eigfuncl,eigfuncr = LST(params,diffs,baseflow,helper_mats)
        helper_mats['M'] = M
        eigs_to_keep,cg,E_oplus = set_SPE2D_E_oplus(params,helper_mats,eigvals,eigfuncl,eigfuncr)
        Eplus,EH = get_SPE2D_Eplus_and_EH_from_4ny_x_4ny(helper_mats['uvwP_from_LST']@E_oplus,orthogonalE=True)
        helper_mats['Eplus_EH'] = lambda q: (Eplus@(EH@q))

    # march SPE
    for xi in np.arange(steps):
        if not parallel: # non parallel
            baseflow2 = blasius(y,x=x[xi+1],nu=nu,plot=False)
            #baseflow2 = DNS_baseflow(params,diffs,baseflows[-1],helper_mats)
            if projection:
                print('get projecting operator')
                L,M,eigvals,eigfuncl,eigfuncr = LST(params,diffs,baseflow2,helper_mats)
                helper_mats['M'] = M
                eigs_to_keep,cg,E_oplus = set_SPE2D_E_oplus(params,helper_mats,eigvals,eigfuncl,eigfuncr)
                Eplus,EH = get_SPE2D_Eplus_and_EH_from_4ny_x_4ny(helper_mats['uvwP_from_LST']@E_oplus,orthogonalE=True)
                helper_mats['Eplus_EH'] = lambda q: (Eplus@(EH@q))
        else: # parallel 
            baseflow2 = baseflow
        params['x'] = x[xi+1]
        #baseflow2.Ux = (baseflow2.get_U() - baseflows[-1].get_U())/(params['hx'])
        #baseflow2.Uy = Dy@baseflow2.get_U()
        #baseflow2.Vy = Dy@baseflow2.get_V()
        #baseflow2 = blasius(y,x=x_start,nu=nu,plot=False)
        #baseflow2.V[:]=baseflow2.Vx[:]=baseflow2.Ux[:]=baseflow2.Vy[:]=0
        q_istep[xi+1,:] = march_PSE2D_one_step(params,diffs,baseflow2,q_istep[xi,:],helper_mats)
        if projection: # project the solution to stabilize the marching procedure
            print('projecting PSE')
            Eplus_EH = helper_mats['Eplus_EH']
            q_istep[xi+1,:] = Eplus_EH(q_istep[xi+1,:])
        alphas[xi+1] = params['alpha']
        baseflows.append(baseflow2)
        print('marched x=',x[xi+1])
    return alphas,x,q_istep

def step_parallel_PSE_test_dPdx(params,diffs,baseflow,q_old,qxx_old,hx,alpha,helper_mats,dPdx):
    steps=params['steps']
    Re=params['Re']
    omega=params['omega']
    beta=params['beta']
    Dy=diffs['Dy']
    Dyy=diffs['Dyy']
    ny=Dy.shape[0]
    U=np.diag(baseflow.U)
    Uy=np.diag(baseflow.Uy)
    Ux=np.diag(baseflow.Ux)
    V = np.diag(baseflow.V)
    Vy = np.diag(baseflow.Vy)
    I=helper_mats['I'] #np.eye(ny)
    zero=helper_mats['zero'] #np.zeros((ny,ny))
    i=1.j
    #hx=0.0001
    #alpha=0.
    #alpha=alpha_original
    delta=-i*omega*I + i*alpha*U + ((I*alpha**2 + I*beta**2)*1./Re)
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
        [U,        zero,     zero,  zero     ], # u-mom
        [zero,      U,       zero,  zero     ], # v-mom
        [zero,      zero,     U,    zero     ], # w-mom
        [I,         zero,     zero,  zero     ]  # continuity
    ])

    # Backward Euler
    APSE = C+By+Byy+(A*(1./hx))
    bPSE = ((1./hx)*A)@q_old + 1./Re*qxx_old - dPdx

    # set BCs
    # set BCs at wall u=v=w=0
    APSE[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
    APSE[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
    bPSE[[0*ny-0,1*ny-0,2*ny-0]] = 0.
    bPSE[[1*ny-1,2*ny-1,3*ny-1]] = 0.
    APSE[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.
    APSE[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.
    #bPSE[[0*ny-0,1*ny-0,2*ny-0]] = 0.
    #bPSE[[1*ny-1,2*ny-1,3*ny-1]] = 0.

    # march PSE in parallel flow
    q_istep = np.zeros((steps+1,4*ny),dtype=np.complex)
    q_istep[0,:]=q_init
    for xi in np.arange(steps):
        converged=False
        while not converged:
            bPSE = ((1./hx)*A)@q_istep[xi,:] + 1./Re*qxx_old - dPdx_n
            q_istep[xi+1,:] = np.linalg.solve(APSE,bPSE)
            dPdx_nm1=np.copy(dPdx_n)
            dPdx_n = helper_mats['P_from_SPE']@(q_istep[xi+1,:] - q_istep[xi,:])/hx
            print('nonlinear iteration = ',noniter)
            if np.norm(dPdx_n-dPdx_nm1)<1E-3:
                converged=True

    #qPSEsolved  = np.linalg.solve(APSE,bPSE)
    #qPSEsolved = np.linalg.inv(APSE)@bPSE
    #qPSEsolved = sp.sparse.linalg.spsolve(APSE,bPSE)
    #qPSEsolved = sp.sparse.linalg.dsolve.spsolve(APSE,bPSE)
    return qPSEsolved
