import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from ..Classes import base_flow_class
from ..helper import ifflag

def LST_temporal(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> list:
    ''' Return L and M from Lq=omega M q Orr-Sommerfield in primitive formulation.  Here, vector q is u,v,w,P and ordering of equations is u-mom, v-mom, w-mom, continuity for 4 equations and 4 unknowns.
    Inputs:
        params: dict containing Re, alpha, beta parameters and flags, and grid class
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices

    Returns:
        L: numpy 2d array 4ny x 4ny
        M: numpy 2d array 4ny x 4ny
        eigvals: all eigenvalues of full spectrum of LST.  ith eigenvalue is eigvals[i]
        eigfuncl: left eigenfunctions (adjoint)
        eigfuncr: right eigenfunctions (normal)  access the ith eigenfunction as eigfuncr[:,i] with (shape 6ny x 6ny by padding with zeros)
        '''

    # get params,derivatives, and baseflow 
    Re=params['Re']
    α=params['alpha']
    β=params['beta']
    Dy=diffs['Dy']
    Dyy=diffs['Dyy']
    ny=Dy.shape[0]
    U=np.diag(baseflow.U)
    Uy=np.diag(baseflow.Uy)
    I=helper_mats['I'] #np.eye(ny)
    O=helper_mats['zero'] #np.zeros((ny,ny)
    i=1.j
    Δ = i*α*U + (α**2 + β**2)/Re*I - 1.0/Re*Dyy

    L = np.block([
        #u      v       w       p
        [Δ,     Uy,     O,      i*α*I],
        [O,     Δ,      O,      Dy   ],
        [O,     O,      Δ,      i*β*I],
        [i*α*I, Dy,     i*β*I,  O    ]
        ])
    M = np.block([
        #u      v       w       p
        [i*I,   O,      O,      O],
        [O,     i*I,    O,      O],
        [O,     O,      i*I,    O],
        [O,     O,      O,      O]
        ])
    # set BCs of u,v,w = 0 at walls and freestream
    L[(0*ny-0,1*ny-0,2*ny-0),:] = 0. # clear walls in momentum equations
    L[(1*ny-1,2*ny-1,3*ny-1),:] = 0. # clear freestream in momentum equations
    M[(0*ny-0,1*ny-0,2*ny-0),:] = 0. # clear walls in momentum equations RHS
    M[(1*ny-1,2*ny-1,3*ny-1),:] = 0. # clear freestream in momentum equations RHS
    L[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1. # set u,v,w = 0 at walls
    L[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1. # set u,v,w = 0 at freestream
    # solve eigenvalue problem for left and right eigenfunctions
    eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,b=M,left=True)
    # return L,M and eigenvalues, and left eigenfunctions (adjoint), and right eigenfunctions
    return L,M,eigvals,eigfuncl,eigfuncr

# spatial LST for q=[u,v,w,P,au,av,aw,aP]
# spatial LST with rearrangement of q
def get_L(params:dict,diffs:dict,baseflow:base_flow_class,helper_mats:dict) -> list:
    ''' Return L and M from Lq=alpha q Orr-Sommerfield in primitive formulation.  Here, vector q is u,v,w,P,av,aw.  and ordering of equations is continuity, v inflation, w inflation, then u-mom, v-mom, and w-mom for 6 equations and 6 unknowns.
    
    Inputs:
        params: dict containing Re, omega, beta parameters and flags, and grid class
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices

    Returns: L and M operators of L*q=alpha*q spatial Orr-Sommerfeld equation
        L: numpy 2d array 6ny-10 x 6ny-10
    '''
    if ifflag(params,'LST2'):
        Re=params['Re']
        omega=params['omega']
        beta=params['beta']
        Dy=diffs['Dy']
        Dyy=diffs['Dyy']
        ny=Dy.shape[0]
        U=np.diag(baseflow.U)
        Uy=np.diag(baseflow.Uy)
        I=helper_mats['I'] #np.eye(ny)
        zero=helper_mats['zero'] #np.zeros((ny,ny)
        i=1.j
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
    # for non-parallel LST
    elif ifflag(params,'LSTNP'):
        Re=params['Re']
        ω=params['omega']
        β=params['beta']
        Dy=diffs['Dy']
        Dyy=diffs['Dyy']
        ny=Dy.shape[0]
        U=np.diag(baseflow.U)
        Ux=np.diag(baseflow.Ux)
        Uy=np.diag(baseflow.Uy)
        Uxy=np.diag(Dy@baseflow.Ux)
        V=np.diag(baseflow.V)
        Vy=np.diag(baseflow.Vy)
        W=np.diag(baseflow.V*0.)
        Wx=np.diag(baseflow.V*0.)
        Wy=np.diag(baseflow.V*0.)
        Wxy=np.diag(baseflow.V*0.)
        I=helper_mats['I'] #np.eye(ny)
        O=helper_mats['zero'] #np.zeros((ny,ny)
        i=1.j
        #Delta=i*Re*omega*I + (Dyy-beta**2*I)
        def polyeig2(L0,L1,L2,helper_mats,**kwargs):
            '''Given a polynomial eigenvalue problem up to alpha^2, return a inflated matrices for a general eigenvalue problem
                Given: L0 q + α L1 q + α^2 L2 q = 0
                helper_mats is a dictionary that must contain identity 'I' and matching shape 'zero' matrix for infaltion
                **kwargs are keyword arguments that match the inputs for scipy.linalg.eig(L,b=M,**kwargs)
                Returns: L,M such that L qinf = αM inflated matrix
                [[0  I ]   [[q ]        [[I  0 ]   [[q ]  
                 [L0 L1]]   [αq]]  = α   [0 -L2]]   [αq]] 
            '''
            O = helper_mats['zero']
            I = helper_mats['I']
            #print('polyeig2 shapes',O.shape,I.shape,L0.shape,L1.shape,L2.shape)
            L = np.block([
                [O,  I],
                [L0, L1],
            ])
            M = np.block([
                [I,  O],
                [O, -L2]
            ])
            return L, M #scipy.linalg.eig(L,b=M,**kwargs)

        Δ = V@Dy - Dyy/Re + i*β*W + (-i*ω + β**2/Re)*I
        ABC2 = np.block([[I/Re,O,    O,    O],
                        [O,    I/Re, O,    O],
                        [O,    O,    I/Re, O],
                        [O,    O,    O,    O]])
        #B2 = C2 = D2 = np.zeros_like(A2)
        O4 = np.zeros_like(ABC2)
        D2 = dA2 = O4
        ABC1 = np.block([[i*U, O,    O,    i*I ],
                        [O,    i*U,  O,    O],
                        [O,    O,    i*U,  O],
                        [i*I,  O,    O,    O]])
        D1 = O4
        dA1 = np.block([[i*Ux, O,    O,   O],
                        [O,    i*Ux, O,   O],
                        [O,    O,   i*Ux, O],
                        [O,    O,   O,    O]])
        ABC0 = np.block([[Δ+Ux,Uy,   O,    O],
                        [O,    Δ+Vy, O,    Dy  ],
                        [Wx,   Wy,   Δ,    i*β*I],
                        [O,    Dy,   i*β*I,O]])
        D0 = np.block([ [U, O, O, I],
                        [O, U, O, O],
                        [O, O, U, O],
                        [I, O, O, O]])
        dA0 = np.block([[i*β*Wx, Uxy,    O,      O],
                        [O,      i*β*Wx, O,      O],
                        [O,      Wxy,    i*β*Wx, O],
                        [O,      O,      O,      O]])
        # set BCs
        ny = Dy.shape[0]
        wallbc = [0,ny,2*ny] # walls for u,v,w
        freebc = [ny-1,2*ny-1,3*ny-1] # free for u,v,w
        #for mat in [ABC0,ABC1,ABC2,D0,D1,D2,dA0,dA1,dA2]:
        for mat in [ABC0,ABC1,ABC2,D0,dA0,dA1]:
            mat[wallbc+freebc,:] = 0.
            mat[wallbc+freebc,wallbc+freebc] = 1.0
        # create inflation
        L0 = np.block([ [ABC0, D0],
                        [dA0,  ABC0]])
        L1 = np.block([ [ABC1, D1],
                        [dA1,  ABC1]])
        L2 = np.block([ [ABC2, D2],
                        [dA2,  ABC2]])
        L,M = polyeig2(L0,L1,L2,{'zero':np.zeros_like(L0),'I':np.eye(8*ny)})

        return L,M
        
    else:
        Re=params['Re']
        omega=params['omega']
        beta=params['beta']
        ny=params['grid'].ny
        i=1.j
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
            [zero           , i*Dy                                      , -beta*I   ,zero_fat     , zero            , zero              ],# cont
            [zero           , zero                                      ,  zero     ,zero_fat     , I               , zero              ],# v-sub
            [zero           , zero                                      , zero      ,zero_fat     , zero            , I                 ],# w-sub
            [-i*Delta_tall  , i*Uy_tall - (i*(U_full @Dy_full)[:,1:-1]) , beta*U_tall,zero_full   , -1./Re*Dy_tall  ,-i*beta/Re*I_tall  ],# u-mom
            [zero           , i*Re*Delta                                , zero      ,-i*Re*Dy_fat , Re*U            , zero              ],# v-mom
            [zero           , zero                                      , i*Re*Delta,beta*Re*I_fat, zero            , Re*U              ],# w-mom
            ])
        return L

def set_BCs(params,L,M,diffs=None):
    ''' set boundary conditions or spatial Orr-Sommerfeld eigenvalue problem.  u,v,w=0 at wall and freestream'''
    ny=params['grid'].ny
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
    ''' Solve the spatial Orr-Sommerfeld general eigenvalue problem L*q=alpha*q

    Inputs:
        params: dict containing Re, omega, and beta parameters and flags, and grid
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices

    Returns:
        eigvals: all eigenvalues of full spectrum of LST.  ith eigenvalue is eigvals[i]
        eigfuncl: left eigenfunctions (adjoint)
        eigfuncr: right eigenfunctions (normal)  access the ith eigenfunction as eigfuncr[:,i] with (shape 6ny x 6ny by padding with zeros)

    '''

    if ifflag(params,'LST2'):
        L,M=get_L(params,diffs,base_flow,helper_mats)
        L,M = set_BCs(params,L,M)
        helper_mats['M']=M
        eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,b=M,left=True)
        return L,M,eigvals,eigfuncl,eigfuncr
    elif ifflag(params,'LSTNP'):
        L,M=get_L(params,diffs,base_flow,helper_mats)
        helper_mats['M']=M
        eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,b=M,left=True)
        return L,M,eigvals,eigfuncl,eigfuncr
    else:
        L=get_L(params,diffs,base_flow,helper_mats)
        eigvals,eigfuncl,eigfuncr = sp.linalg.eig(L,left=True)
        # padd zeros to make u,v,w,av,aw full rank
        ny=params['grid'].ny
        zeros_to_insert=[0,ny-2,ny-2,2*(ny-2),2*(ny-2),3*(ny-2),3*(ny-2)+ny,4*(ny-2)+ny,4*(ny-2)+ny,5*(ny-2)+ny]
        eigfuncr = np.insert(eigfuncr,zeros_to_insert,0,axis=0)
        eigfuncl = np.insert(eigfuncl,zeros_to_insert,0,axis=0)
        return L,eigvals,eigfuncl,eigfuncr

def LST_alphas(params:dict,diffs:dict,base_flow:base_flow_class,helper_mats:dict,alphas):
    ''' Solve the spatial Orr-Sommerfeld general eigenvalue problem L*q=alpha*q solving for specific alpha

    Inputs:
        params: dict containing Re, omega, and beta parameters, and grid
        diffs: dict containing Dy and Dyy numpy operators acting on vectors of wall-normal coordinate y discretized values
        baseflow: base_flow_class containing the base flow as a function of wall-normal coordinate y
        helper_mats:dict containing zero and I matrices
        alpha:np.ndarray target alpha values to solve for

    Returns:
        L: L operator in eigenvalue problem
        eigval: eigenvalue of LST closest to alpha
        eigfuncl: left eigenfunction (adjoint)
        eigfuncr: right eigenfunction (normal), ith eigenvalue obtained from eigfuncr[:,i] (shape (6ny x 6ny) by padding with zeros)

    '''

    if ifflag(params,'LST2'):
        L,M=get_L(params,diffs,base_flow,helper_mats)
        L,M = set_BCs(params,L,M)
        LH=L.conj().T
        MH=M.conj().T
        eigvals=[]
        eigfuncr=[]
        eigfuncl=[]
        for alpha in alphas:
            eigvali,eigfuncri = sp.sparse.linalg.eigs(L,M=M,sigma=alpha,k=1)
            eigvall,eigfuncli = sp.sparse.linalg.eigs(LH,M=MH,sigma=eigvali.conjugate(),k=1)
            eigvals.append(eigvali[0])
            eigfuncr.append(eigfuncri[:,0])
            eigfuncl.append(eigfuncli[:,0])
        eigvals=np.array(eigvals)
        eigfuncr=np.array(eigfuncr).T
        eigfuncl=np.array(eigfuncl).T
        return L,M,eigvals,eigfuncl,eigfuncr
    elif ifflag(params,'LSTNP'):
        L,M=get_L(params,diffs,base_flow,helper_mats)
        #print('L = ',L)
        #print(L.shape,np.isfinite(L).sum())
        #print(M.shape,np.isfinite(M).sum())
        #print('M = ',M)
        #return L,M
        #L,M = set_BCs(params,L,M)
        LH=L.conj().T
        MH=M.conj().T
        eigvals=[]
        eigfuncr=[]
        eigfuncl=[]
        for alpha in alphas:
            eigvali,eigfuncri = sp.sparse.linalg.eigs(L,M=M,sigma=alpha,k=1)
            eigvall,eigfuncli = sp.sparse.linalg.eigs(LH,M=MH,sigma=eigvali.conjugate(),k=1)
            eigvals.append(eigvali[0])
            eigfuncr.append(eigfuncri[:,0])
            eigfuncl.append(eigfuncli[:,0])
        eigvals=np.array(eigvals)
        eigfuncr=np.array(eigfuncr).T
        eigfuncl=np.array(eigfuncl).T
        return L,M,eigvals,eigfuncl,eigfuncr
    else:
        L=get_L(params,diffs,base_flow,helper_mats)
        LH=L.conj().T
        eigvals=[]
        eigfuncr=[]
        eigfuncl=[]
        for alpha in alphas:
            eigvali,eigfuncri = sp.sparse.linalg.eigs(L,sigma=alpha,k=1)
            eigvall,eigfuncli = sp.sparse.linalg.eigs(LH,sigma=eigvali.conjugate(),k=1)
            eigvals.append(eigvali[0])
            eigfuncr.append(eigfuncri[:,0])
            eigfuncl.append(eigfuncli[:,0])
        eigvals=np.array(eigvals)
        eigfuncr=np.array(eigfuncr).T
        eigfuncl=np.array(eigfuncl).T
        # padd zeros to make u,v,w,av,aw full rank
        ny=params['grid'].ny
        zeros_to_insert=[0,ny-2,ny-2,2*(ny-2),2*(ny-2),3*(ny-2),3*(ny-2)+ny,4*(ny-2)+ny,4*(ny-2)+ny,5*(ny-2)+ny]
        eigfuncr = np.insert(eigfuncr,zeros_to_insert,0,axis=0)
        eigfuncl = np.insert(eigfuncl,zeros_to_insert,0,axis=0)
        return L,eigvals,eigfuncl,eigfuncr

def dLdomega(params:dict,diffs:dict,helper_mats:dict) -> np.ndarray:
    ''' Return the dLdomega of the LST used here (shape (6ny x 6ny))
    Inputs:
        params:dict containting Re, omega, and beta, and grid
        diffs:dict containing Dy and Dyy derivative operators
        helper_mats:dict containing zero and I matrices
    Returns:
        dLdomega:np.ndarray containing the dLdomega operator
    '''
    Re=params['Re']
    #omega=params['omega']
    #beta=params['beta']
    #ny=params['grid'].ny
    # full
    #zero_full=helper_mats['zero'] #np.zeros((ny,ny))
    # short and skinny
    I=helper_mats['I'] #[1:-1,1:-1] #np.eye(ny)
    zero=helper_mats['zero'] #np.zeros((ny,ny))
    # fat
    #I_fat=helper_mats['I'][1:-1,:] #np.eye(ny)
    #zero_fat=helper_mats['zero'][1:-1,:] #np.zeros((ny,ny))
    # tall
    #I_tall=helper_mats['I'][:,1:-1] #np.eye(ny)
    #zero_tall=helper_mats['zero'][:,1:-1] #np.zeros((ny,ny))
    #i=1.j
    if ifflag(params,'LST2'):
        ny=params['grid'].ny
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
            [zero,       Re*i*I,   zero,   zero,  
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
    elif ifflag(params,'LSTNP'):
        ny=params['grid'].ny
        i=1.j
        # uninflated dLdomega
        dLdomega = np.block([
            [-i*I, zero, zero, zero],
            [zero, -i*I, zero, zero],
            [zero, zero, -i*I, zero],
            [zero, zero, zero, zero],
            ])
        # inflate by non-parallel
        zero1 = np.zeros_like(dLdomega)
        dLdomega = np.block([   [dLdomega, zero1],
                                [zero1,    zero1],])
        # inflate by polynomial eigenvalue problem
        zero2 = np.zeros_like(dLdomega)
        dLdomega = np.block([   [zero2,     zero2],
                                [dLdomega,  zero2],])
        return dLdomega
    else:
        dLdomega = np.block([
            # u         v           w           p           vx      wx          
            [zero   , zero      , zero      ,zero       , zero  , zero  ],# cont
            [zero   , zero      , zero      ,zero       , zero  , zero  ],# v-sub
            [zero   , zero      , zero      ,zero       , zero  , zero  ],# w-sub
            [I      , zero      , zero      ,zero       , zero  , zero  ],# u-mom
            [zero   , -Re*I     , zero      ,zero       , zero  , zero  ],# v-mom
            [zero   , zero      , -Re*I     ,zero       , zero  , zero  ],# w-mom
            ])
        return dLdomega
