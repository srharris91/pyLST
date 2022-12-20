import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from ..Classes import base_flow_class
from ..helper import ifflag
# spatial LST for q=[u,v,w,P,au,av,aw,aP]
# spatial LST with rearrangement of q
def get_L_M(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> list:
    ''' Return L and M from Lq=alpha M q Orr-Sommerfield in primitive formulation.  Here, vector q is u,v,w,P,au,av,aw,aP.  and ordering of equations is 4 inflation matrices, then u-mom,v-mom,w-mom and continuity for 8 equations and 8 unknowns.
    
    Inputs:
        params: dict containing Re, omega, beta parameters and flags
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices

    Returns: L and M operators of L*q=alpha*M*q spatial Orr-Sommerfeld equation
        L: numpy 2d array 8ny x 8ny
        M: numpy 2d array 8ny x 8ny
    '''
    Re=params['Re']
    omega=params['omega']
    beta=params['beta']
    ny=params['ny']
    i=1.j
    if ifflag(params,'LST2'):
        if ifflag(params,'LST3'):
            # full operators
            Dy_full=diffs['Dy']
            U_full=np.diag(baseflow.U)
            zero_full=helper_mats['zero'] #np.zeros((ny,ny))
            # short and skinny
            Dy=Dy_full[1:-1,1:-1]
            Dyy=diffs['Dyy'][1:-1,1:-1]
            U=np.diag(baseflow.U)[1:-1,1:-1]
            Uy=np.diag(baseflow.Uy)[1:-1,1:-1]
            I=helper_mats['I'][1:-1,1:-1] #np.eye(ny)
            zero=helper_mats['zero'][1:-1,1:-1] #np.zeros((ny,ny))
            Delta=1./Re * (Dyy - I*beta**2) + i*I*omega
            # fat operators
            Dy_fat=Dy_full[1:-1,:]
            Dyy_fat=diffs['Dyy'][1:-1,:]
            U_fat=np.diag(baseflow.U)[1:-1,:]
            Uy_fat=np.diag(baseflow.Uy)[1:-1,:]
            I_fat=helper_mats['I'][1:-1,:] #np.eye(ny)
            zero_fat=helper_mats['zero'][1:-1,:] #np.zeros((ny,ny))
            # tall operators
            Dy_tall=diffs['Dy'][:,1:-1]
            Dyy_tall=diffs['Dyy'][:,1:-1]
            U_tall=np.diag(baseflow.U)[:,1:-1]
            Uy_tall=np.diag(baseflow.Uy)[:,1:-1]
            I_tall=helper_mats['I'][:,1:-1] #np.eye(ny)
            zero_tall=helper_mats['zero'][:,1:-1] #np.zeros((ny,ny))
            Delta_tall=1./Re * (Dyy_tall - I_tall*beta**2) + i*I_tall*omega
            L = np.block([
                # u                v                                            w         p              vx                wx          
                [zero           , i*Dy                                      , -beta*I   ,zero_fat     , zero            , zero              ],
                [zero           , zero                                      ,  zero     ,zero_fat     , I               , zero              ],
                [zero           , zero                                      , zero      ,zero_fat     , zero            , I                 ],
                [-i*Delta_tall  , i*Uy_tall - (i*(U_full @Dy_full)[:,1:-1]) , beta*U_tall,zero_full   , -1./Re*Dy_tall  ,-i*beta/Re*I_tall  ],
                [zero           , i*Re*Delta                                , zero      ,-i*Re*Dy_fat , Re*U            , zero              ],
                [zero           , zero                                      , i*Re*Delta,beta*Re*I_fat, zero            , Re*U              ],
                ])
            return L
        else:
            U=np.diag(baseflow.U)
            Uy=np.diag(baseflow.Uy)
            I=helper_mats['I'] #np.eye(ny)
            zero=helper_mats['zero'] #np.zeros((ny,ny))
            Dy=diffs['Dy']
            Dyy=diffs['Dyy']
            Delta=1./Re * (Dyy - I*beta**2) + i*I*omega
            L = np.block([
                # u         v                  vx          w          wx           p
                [zero    , i*Dy             , zero     , -beta*I   , zero       , zero],
                [zero    , zero             , I        ,  zero     , zero       , zero],
                [zero    , i*Re*Delta       , Re*U     , zero      , zero       , -i*Re*Dy],
                [zero    , zero             , zero     , zero      , I          , zero],
                [zero    , zero             , zero     , i*Re*Delta, Re*U       , beta*Re*I],
                [-i*Delta, i*Uy  - (i*U @Dy), -1./Re*Dy, beta*U    ,-i*beta/Re*I, zero],
                ])
            return L
    else:
        U=np.diag(baseflow.U)
        Uy=np.diag(baseflow.Uy)
        I=helper_mats['I'] #np.eye(ny)
        zero=helper_mats['zero'] #np.zeros((ny,ny))
        Dy=diffs['Dy']
        Dyy=diffs['Dyy']
        Delta=i*Re*omega*I + (Dyy-beta**2*I)
        L = np.block([
            # u   v   w   p
            # au  av  aw  ap
            [np.zeros((4*ny,4*ny)),                 np.eye(4*ny)],                     # inflation matrices
            [Delta, -Re*Uy,  zero,   zero,         -i*Re*U, zero,  zero,   -i*Re*I], # u-mom
            [zero,  Delta,    zero,   -Re*Dy,       zero,  -i*Re*U, zero,   zero],    # v-mom
            [zero,  zero,     Delta,  -i*Re*beta*I, zero,  zero,     -i*Re*U, zero],  # w-mom
            [zero,  Dy,       i*beta*I,  zero,      i*I,   zero,     zero,    zero]    # continuity
            ])
        M = np.block([
            # u   v   w   p
            # au  av  aw  ap
            [np.eye(4*ny),                          np.zeros((4*ny,4*ny)),],           # inflation matrices
            [zero,  zero,   zero,   zero,           I,     zero,   zero,   zero],      # u-mom
            [zero,  zero,   zero,   zero,           zero,  I,      zero,   zero],      # v-mom
            [zero,  zero,   zero,   zero,           zero,  zero,   I,      zero],      # w-mom
            [zero,  zero,   zero,   zero,           zero,  zero,   zero,   zero],      # continuity
            ])
        return L,M

def set_BCs(params,L,M=None,diffs=None):
    ''' set boundary conditions or spatial Orr-Sommerfeld eigenvalue problem.  u,v,w=0 at wall and freestream'''

    ny=params['ny']
    # set BCs at wall u=v=w=0
    if ifflag(params,'LST2'):
        if not ifflag(params,'LST3'): # don't need to call if LST3
            Dy=diffs['Dy']
            large_num=200001.
            walls=  (0*ny-0,1*ny-0,2*ny-0,3*ny-0,4*ny-0)
            freestreams=(1*ny-1,2*ny-1,3*ny-1,4*ny-1,5*ny-1)
            L[walls,:] = 0. # remove inflated systems too
            L[freestreams,:] = 0.
            L[walls,walls] = large_num
            L[freestreams,freestreams] = large_num
            print('shapes ',L.shape,Dy.shape,L[1*ny+1,1*ny:2*ny].shape,Dy[0,:].shape)
            L[1*ny+1,1*ny:2*ny] = Dy[0,:]  # dvdy = 0 at wall
            L[1*ny+1,1*ny+1   ]+= 1.       # dvdy = 0 at wall
            L[2*ny-2,1*ny:2*ny] = Dy[-1,:] # dvdy = 0 at freestream
            L[2*ny-2,1*ny+1   ]+= 1.       # dvdy = 0 at freestream
        return L
    else:
        #large_num=200001.
        L[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
        L[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
        L[(4*ny-0,5*ny-0,6*ny-0),:] = 0.
        L[(5*ny-1,6*ny-1,7*ny-1),:] = 0.
        M[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
        M[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
        M[(4*ny-0,5*ny-0,6*ny-0),:] = 0.
        M[(5*ny-1,6*ny-1,7*ny-1),:] = 0.
        L[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.
        L[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.
        L[(4*ny-0,5*ny-0,6*ny-0),(4*ny-0,5*ny-0,6*ny-0)] = 1.
        L[(5*ny-1,6*ny-1,7*ny-1),(5*ny-1,6*ny-1,7*ny-1)] = 1.
        #M[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1./large_num
        #M[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1./large_num
        #M[(4*ny-0,5*ny-0,6*ny-0),(4*ny-0,5*ny-0,6*ny-0)] = 1./large_num
        #M[(5*ny-1,6*ny-1,7*ny-1),(5*ny-1,6*ny-1,7*ny-1)] = 1./large_num
        return L,M

def LST(params:dict,diffs:dict,base_flow:base_flow_class,helper_mats:dict):
    ''' Solve the spatial Orr-Sommerfeld general eigenvalue problem L*q=alpha*M*q

    Inputs:
        params: dict containing Re, omega, and beta parameters and flags
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices

    Returns:
        eigvals: all eigenvalues of full spectrum of LST.  ith eigenvalue is eigvals[i]
        eigfuncl: left eigenfunctions (adjoint)
        eigfuncr: right eigenfunctions (normal)  access the ith eigenfunction as eigfuncr[:,i]

    '''

    if ifflag(params,'LST2'):
        if ifflag(params,'LST3'):
            L=get_L_M(params,diffs,base_flow,helper_mats)
            eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,left=True)
            return L,eigvals,eigfuncl,eigfuncr
        else:
            L=get_L_M(params,diffs,base_flow,helper_mats)
            L=set_BCs(params,L,diffs=diffs)
            eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,left=True)
            return L,eigvals,eigfuncl,eigfuncr
    else:
        L,M=get_L_M(params,diffs,base_flow,helper_mats)
        L,M=set_BCs(params,L,M)
        eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,b=M,left=True)
        return L,M,eigvals,eigfuncl,eigfuncr

def LST_alphas(params:dict,diffs:dict,base_flow:base_flow_class,helper_mats:dict,alphas):
    ''' Solve the spatial Orr-Sommerfeld general eigenvalue problem L*q=alpha*M*q solving for specific alpha

    Inputs:
        params: dict containing Re, omega, and beta parameters
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices
        alpha:np.ndarray target alpha values to solve for

    Returns:
        eigval: eigenvalue of LST closest to alpha
        eigfuncl: left eigenfunction (adjoint)
        eigfuncr: right eigenfunction (normal), ith eigenvalue obtained from eigfuncr[:,i]

    '''

    L,M=get_L_M(params,diffs,base_flow,helper_mats)
    L,M=set_BCs(L,M)
    LH=L.conj().T
    eigvals=[]
    eigfuncr=[]
    eigfuncl=[]
    for alpha in alphas:
        eigvali,eigfuncri = sp.sparse.linalg.eigs(L,M=M,sigma=alpha,k=1)
        eigvall,eigfuncli = sp.sparse.linalg.eigs(LH,M=M,sigma=eigvali.conjugate(),k=1)
        eigvals.append(eigvali[0])
        eigfuncr.append(eigfuncri[:,0])
        eigfuncl.append(eigfuncli[:,0])
    eigvals=np.array(eigvals)
    eigfuncr=np.array(eigfuncr).T
    eigfuncl=np.array(eigfuncl).T
    return L,M,eigvals,eigfuncl,eigfuncr

def dLdomega(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> np.ndarray:
    ''' Return the dLdomega of the LST used here
    Inputs:
        params:dict containting Re, omega, and beta
        diffs:dict containing Dy and Dyy derivative operators
        baseflow:base_flow_class containing the base flow profile at the current streamwise location
        helper_mats:dict containing zero and I matrices
    Returns:
        dLdomega:np.ndarray containing the dLdomega operator
    '''
    Re=params['Re']
    omega=params['omega']
    beta=params['beta']
    Dy=diffs['Dy']
    Dyy=diffs['Dyy']
    ny=Dy.shape[0]
    U=np.diag(baseflow.U)
    Uy=np.diag(baseflow.Uy)
    I=helper_mats['I'] #np.eye(ny)
    zero=helper_mats['zero'] #np.zeros((ny,ny))
    i=1.j
    dLdomega = np.block([
        # u         v          w     P      
        # au        av         aw    aP  
        [zero,        zero,  zero,   zero,  
             zero,   zero,   zero,   zero], # au-sub
        [zero,       zero,   zero,   zero,
            zero,    zero,   zero,   zero], # av-sub
        [zero,       zero,   zero,    zero,
            zero,    zero,   zero,   zero], # aw-sub
        [zero,       zero,   zero,    zero,
            zero,    zero,   zero,   zero], # aP-sub
        [Re*i*I,      zero,    zero,   zero,     
            zero,   zero,    zero,   zero], # u-mom
        [zero,       Re*I,   zero,   zero,  
             zero,   zero,   zero,   zero], # v-mom
        [zero,       zero,   Re*i*I,   zero,
            zero,   zero,    zero,   zero,],# w-mom
        [zero,       zero,   zero,   zero,
             zero,   zero,   zero,  zero ], # continuity
    ])
    # set BCs and wall and freestream of momentum equations
    dLdomega[(4*ny-0,5*ny-0,6*ny-0),:] = 0.
    dLdomega[(5*ny-1,6*ny-1,7*ny-1),:] = 0.
    return dLdomega
