#!/usr/bin/env python

import numpy as np
from scipy.special import factorial


# Finite Difference functions
def get_D_Coeffs(s,d=2,h=1,TaylorTable=False):
    '''
    Solve arbitrary stencil points s of length N with order of derivatives d<N
    can be obtained from equation on MIT website
    http://web.media.mit.edu/~crtaylor/calculator.html
    where the accuracy is determined as the usual form O(h^(N-d))
    
    Inputs:
        s: array like input of stencil points e.g. np.array([-3,-2,-1,0,1])
        d: order of desired derivative
    '''
    # solve using Taylor Table instead
    if TaylorTable:
        # create Taylor Table
        N=s.size # stencil length
        b=np.zeros(N)
        A=np.zeros((N,N))
        A[0,:]=1. # f=0
        for row in np.arange(1,N):
            A[row,:]=1./factorial(row) * s**row # f^(row) terms
        b[d]=-1
        x = -np.linalg.solve(A,b)
        return x
        
        
    # use MIT stencil
    else: 
        # let's solve an Ax=b problem
        N=s.size # stencil length
        A=[]
        for i in range(N):
            A.append(s**i)
        b=np.zeros(N)
        b[d] = factorial(d)
        x = np.linalg.solve(np.matrix(A),b)
        return x

def set_D(y,yP=None,order=2,T=2.*np.pi,d=2,reduce_wall_order=True,output_full=False,periodic=False,uniform=True):
    '''
    Input:
        y: array of y values of channel
        order: order of accuracy desired (assuming even e.g. 2,4,6,...)
        d: dth derivative
        T: period if using Fourier
    Output:
        D: (n-2 by n) dth derivative of order O(h^order) assuming uniform y spacing
    '''
    if isinstance(order,int):
        h = y[1]-y[0] # uniform spacing
        if not uniform:
            xi=np.linspace(0,1,y.size)
            h=xi[1] - xi[0]
        n = y.size
        ones=np.ones(n)
        I = np.eye(n)
        # get coefficients for main diagonals
        N=order+d # how many pts needed for order of accuracy
        if N>n:
            raise ValueError('You need more points in your domain, you need %i pts and you only gave %i'%(N,n))
        Nm1=N-1 # how many pts needed if using central difference is equal to N-1
        if (d % 2 != 0): # if odd derivative
            Nm1+=1 # add one more point to central, to count the i=0 0 coefficient
        # stencil and get Coeffs for diagonals
        s = np.arange(Nm1)-int((Nm1-1)/2) # stencil for central diff of order
        smax=s[-1] # right most stencil used (positive range)
        Coeffs = get_D_Coeffs(s,d=d)
        # loop over s and add coefficient matrices to D
        D = np.zeros_like(I)
        si = np.nditer(s,('c_index',))
        while not si.finished:
            i = si.index
            if si[0]==0:
                diag_to_add = np.diag(Coeffs[i] * ones,k=si[0])
            else:
                diag_to_add = np.diag(Coeffs[i] * ones[:-abs(si[0])],k=si[0])

            D += diag_to_add
            if periodic:
                if si[0]>0:
                    diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]-n)
                elif si[0]<0:
                    diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]+n)
                if si[0]!=0:
                    D += diag_to_add
                    
            si.iternext()
        if not periodic:
            # alter BC so we don't go out of range on bottom of channel
            for i in range(0,smax):
                # for ith row, set proper stencil coefficients
                if reduce_wall_order:
                    if (d%2!=0): # if odd derivative
                        s = np.arange(Nm1-1)-i # stencil for shifted diff of order-1
                    else:
                        s = np.arange(Nm1)-i # stencil for shifted diff of order-1
                else:
                    s = np.arange(N)-i # stencil for shifted diff of order
                Coeffs = get_D_Coeffs(s,d=d)
                D[i,:] = 0. # set row to zero
                D[i,s+i] = Coeffs # set row to have proper coefficients

                # for -ith-1 row, set proper stencil coefficients
                if reduce_wall_order:
                    if (d%2!=0): # if odd derivative
                        s = -(np.arange(Nm1-1)-i) # stencil for shifted diff of order-1
                    else:
                        s = -(np.arange(Nm1)-i) # stencil for shifted diff of order-1
                else:
                    s = -(np.arange(N)-i) # stencil for shifted diff of order
                Coeffs = get_D_Coeffs(s,d=d)
                D[-i-1,:] = 0. # set row to zero
                D[-i-1,s-i-1] = Coeffs # set row to have proper coefficients

        if output_full:
            D = (1./(h**d)) * D # do return the full matrix
        else:
            D = (1./(h**d)) * D[1:-1,:] # do not return the top or bottom row
        if not uniform:
            D = map_D(D,y,order=order,d=d,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=uniform)
    elif order=='fourier':
        npts = y.size # number of points
        n = np.arange(0,npts)
        j = np.arange(0,npts)
        N,J = np.meshgrid(n,j,indexing='ij')
        D = 2.*np.pi/T*0.5*(-1.)**(N-J)*1./np.tan(np.pi*(N-J)/npts)
        D[J==N]=0
        
        if d==2:
            D=D@D
    return D 

def map_D(D,y,yP=None,order=2,d=2,reduce_wall_order=True,output_full=False,periodic=False,uniform=True,
          staggered=False,return_P_location=False,full_staggered=False):
    if not uniform:
        xi=np.linspace(0,1,y.size)
        if d==1: # if 1st derivative operator d(.)/dy = d(.)/dxi * dxi/dy
            if staggered and return_P_location==False:
                D1 = set_D_P(xi,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                             staggered=False,return_P_location=False)
                dydxi=D1@y
            elif full_staggered and return_P_location:
                if(0):
                    xi=np.linspace(0,1,y.size+yP.size)
                    D1 = set_D_P(xi,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=False,return_P_location=False)
                    yall=np.zeros(y.size+yP.size)
                    yall[::2]  = yP
                    yall[1::2] = y
                    dydxi=(D1@yall)[::2]
                elif(0):
                    xi=np.linspace(0,1,y.size)
                    xi2=np.insert(xi,0,-xi[1])
                    xiP=(xi2[1:]+xi2[:-1])/2.
                    D1 = set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=True,return_P_location=True,full_staggered=True)
                    dydxi=D1@y
                else:
                    dydxi = D@y # matrix multiply in python3
            else:
                dydxi = D@y # matrix multiply in python3
            dxidy = 1./dydxi # element wise invert
            return D*dxidy[:,np.newaxis] # d(.)/dy = d(.)/dxi * dxi/dy
        elif d==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
            xi=np.linspace(0,1,y.size)
            xiP=(xi[1:]+xi[:-1])/2.
            if staggered and return_P_location and full_staggered==False:
                D1=set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                            staggered=True,return_P_location=return_P_location)
                dydxi = D1@y
            elif full_staggered:
                if(0):
                    xiall=np.linspace(0,1,y.size+yP.size)
                    D1 = set_D_P(xiall,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=False,return_P_location=False)
                    yall=np.zeros(y.size+yP.size)
                    yall[::2]  = yP
                    yall[1::2] = y
                    dydxi=(D1@yall)[::2]
                else:
                    xi=np.linspace(0,1,y.size)
                    xiv=np.insert(xi,0,-xi[1])
                    xiP=(xiv[1:]+xiv[:-1])/2.
                    D1 = set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                 staggered=True,return_P_location=True,full_staggered=True)
                    dydxi=(D1@y)
            else:
                D1=set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                            staggered=False,return_P_location=return_P_location)
                dydxi = D1@y
            dxidy = 1./dydxi # element wise invert
            if staggered and full_staggered==False:
                D2=set_D_P(xi,xiP,order=order,d=2,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                            staggered=return_P_location,return_P_location=return_P_location)
                d2xidy2 = -(D2@y)*(dxidy)**3
                D1p=set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                            staggered=True,return_P_location=return_P_location)
                return (D*(dxidy[:,np.newaxis]**2)) + (D1p*d2xidy2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
            elif full_staggered:
                if(0):
                    xiall=np.linspace(0,1,y.size+yP.size)
                    D2 = set_D_P(xiall,order=order,d=2,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                 staggered=False,return_P_location=False,full_staggered=False)
                    yall=np.zeros(y.size+yP.size)
                    yall[::2]  = yP
                    yall[1::2] = y
                    d2xidy2 = -((D2@yall)[::2])*(dxidy)**3
                else:
                    xi=np.linspace(0,1,y.size)
                    xiv=np.insert(xi,0,-xi[1])
                    xiP=(xiv[1:]+xiv[:-1])/2.
                    D2 = set_D_P(xi,xiP,order=order,d=2,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                 staggered=True,return_P_location=True,full_staggered=True)
                    d2xidy2 = -(D2@y)*(dxidy)**3
                xi=np.linspace(0,1,y.size)
                xiv=np.insert(xi,0,-xi[1])
                xiP=(xiv[1:]+xiv[:-1])/2.
                D1p=set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                            staggered=staggered,return_P_location=return_P_location,full_staggered=full_staggered)
                return (D*(dxidy[:,np.newaxis]**2)) + (D1p*d2xidy2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
            else:
                d2xidy2 = -(D@y)*(dxidy)**3
                return (D*(dxidy[:,np.newaxis]**2)) + (D1*d2xidy2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
        else:
            print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',d)
    else:
        return D

def set_FD_stretched_y(y_max,ny,delta=2.0001):
    ''' Create a stretched mesh for Finite Difference operators using algebraic grid generation for CFD https://www.cfd-online.com/Wiki/Structured_mesh_generation
    Inputs:
        y_max: maximum desired domain location
        ny: number of desired grid points
        delta: stretching parameter
    Returns:
        y:np.array stretched grid from 0 to y_max
    '''

    y=np.linspace(0,y_max,ny)
    y=y_max*(1. + (np.tanh(delta*(y/y_max - 1.))/np.tanh(delta)))
    return y

def set_D_P(y,yP=None,staggered=False,return_P_location=False,full_staggered=False,order=2,d=2,reduce_wall_order=True,output_full=False,periodic=False,uniform=True):
    #DyyP=set_D_P(y,yP, order=4,d=2,output_full=True,uniform=False,staggered=True)
    '''
    Input:
        y: array of y values, (velocity points) or vertical velocity points if full staggered
        yP: array of y values, (pressure points) and horizontal velocity points if full staggered
        order: order of accuracy desired (assuming even e.g. 2,4,6,...)
        d: dth derivative
    Output:
        D: (n-2 by n) dth derivative of order O(h^order) assuming uniform y spacing
    '''
    if staggered:
        if full_staggered:
            h = y[0]-yP[0] # uniform spacing
        else:
            h = y[1]-yP[0] # uniform spacing
    else:
        h = y[1]-y[0] # uniform spacing
    if (not uniform) or full_staggered:
        if staggered:
            xi=np.linspace(0,1,y.size+yP.size)
        else:
            xi=np.linspace(0,1,y.size)
        h=xi[1] - xi[0]
    if staggered:
        n = 2*y.size-1
        if full_staggered:
            n = 2*y.size
    else:
        n=y.size
    ones=np.ones(n)
    I = np.eye(n)
    # get coefficients for main diagonals
    N=order+d # how many pts needed for order of accuracy
    if N>n:
        raise ValueError('You need more points in your domain, you need %i pts and you only gave %i'%(N,n))
    Nm1=N-1 # how many pts needed if using central difference is equal to N-1
    if (d % 2 != 0): # if odd derivative
        Nm1+=1 # add one more point to central, to count the i=0 0 coefficient
    if staggered and (d%2==0): # staggered and even derivative
        Nm1+=2  # add two more points for central
    # stencil and get Coeffs for diagonals
    if staggered:
        s = (np.arange(-Nm1+2,Nm1,2))#)-int((Nm1-1))) # stencil for central diff of order
    else:
        s = np.arange(Nm1)-int((Nm1-1)/2) # stencil for central diff of order
    #print('sc = ',s)
    smax=s[-1] # right most stencil used (positive range)
    Coeffs = get_D_Coeffs(s,d=d)
    # loop over s and add coefficient matrices to D
    D = np.zeros_like(I)
    si = np.nditer(s,('c_index',))
    while not si.finished:
        i = si.index
        if si[0]==0:
            diag_to_add = np.diag(Coeffs[i] * ones,k=si[0])
        else:
            diag_to_add = np.diag(Coeffs[i] * ones[:-abs(si[0])],k=si[0])

        D += diag_to_add
        if periodic:
            if si[0]>0:
                diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]-n)
            elif si[0]<0:
                diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]+n)
            if si[0]!=0:
                D += diag_to_add
                
        si.iternext()
    if not periodic:
        # alter BC so we don't go out of range on bottom 
        smax_range=np.arange(0,smax)
        for i in smax_range:
            # for ith row, set proper stencil coefficients
            if reduce_wall_order:
                if (d%2!=0): # if odd derivative
                    if staggered:
                        if i%2!=0: # odd row, for P_location
                            s = np.arange(-i,2*(Nm1-1)-i-1,2) # stencil for shifted diff of order-1
                        else: # even row, for velocity location
                            s = np.arange(-i+1,2*(Nm1-1)-i,2) # stencil for shifted diff of order-1
                    else:
                        s = np.arange(Nm1-1)-i # stencil for shifted diff of order-1
                else:
                    if staggered:
                        if i%2!=0: # odd row, for P_location
                            s = np.arange(-i+1,2*Nm1-i-2,2)-1 # stencil for shifted diff of order-1
                        else: # even row, for velocity location
                            s = np.arange(-i+1,2*Nm1-i-2,2) # stencil for shifted diff of order-1
                    else:
                        s = np.arange(Nm1)-i # stencil for shifted diff of order-1
            else:
                if staggered:
                    s = np.arange(-i+1,2*N-i,2) # stencil for shifted diff of order
                else:
                    s = np.arange(N)-i # stencil for shifted diff of order
            #print('i, s,scol = ',i,', ',s,',',s+i)
            Coeffs = get_D_Coeffs(s,d=d)
            D[i,:] = 0. # set row to zero
            D[i,s+i] = Coeffs # set row to have proper coefficients

            # for -ith-1 row, set proper stencil coefficients
            if reduce_wall_order:
                if (d%2!=0): # if odd derivative
                    if staggered:
                        if i%2!=0: # odd row, for P_location
                            s = -np.arange(-i+1,2*(Nm1-1)-i,2)+1 # stencil for shifted diff of order-1
                        else: # if even row, return velocity location
                            s = -np.arange(-i+1,2*(Nm1-1)-i,2) # stencil for shifted diff of order-1
                    else:
                        s = -(np.arange(Nm1-1)-i) # stencil for shifted diff of order-1
                else:
                    if staggered:
                        if i%2!=0: # odd row, for P_location
                            s = -np.arange(-i+1,2*Nm1-i-2,2)+1 # stencil for shifted diff of order-1
                        else: # even row, for velocity location
                            s = -np.arange(-i+1,2*Nm1-i-2,2) # stencil for shifted diff of order-1
                    else:
                        s = -(np.arange(Nm1)-i) # stencil for shifted diff of order-1
            else:
                if staggered:
                    s = -np.arange(-i+1,2*N-i,2) # stencil for shifted diff of order-1
                else:
                    s = -(np.arange(N)-i) # stencil for shifted diff of order
            #print('i,row, s,scol = ',i,',',-i-1,', ',s,',',s-i-1)
            Coeffs = get_D_Coeffs(s,d=d)
            D[-i-1,:] = 0. # set row to zero
            D[-i-1,s-i-1] = Coeffs # set row to have proper coefficients

    # filter out for only Pressure values
    if staggered:
        if full_staggered:
            if return_P_location:
                D = D[::2,1::2]
            else:
                D = D[1::2,::2]
        else:
            if return_P_location:
                D = D[1::2,::2]
            else:
                D = D[::2,1::2]
        
        #D[[0,-1],:]=D[[1,-2],:] # set dPdy at wall equal to dPdy off wall
    if output_full:
        D = (1./(h**d)) * D # do return the full matrix
    else:
        D = (1./(h**d)) * D[1:-1,:] # do not return the top or bottom row
    if not uniform:
            D = map_D(D,y,yP,order=order,d=d,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=uniform,
                      staggered=staggered,return_P_location=return_P_location,full_staggered=full_staggered)
    return D 

# set_D operators
def set_D_ops_2D(x,y,orders=[4,4]):
    ''' set derivative operators for 2D data field
    
    Inputs:
        x,y - numpy 1D arrays containing location values and lengths (will use meshgrid with indexing='ij')
        orders - list of length 2, order of derivative in x and y directions (default 4 each)
    Returns:
        ddx,ddy,d2dx2,d2dy2 - functions to compute the first and second derivatives for 2D field
    '''
    orderx,ordery=orders
    # set derivative operators for grid data
    if orderx>1:
        Dx =set_D(x,order=orderx,d=1,output_full=True,uniform=False)
        Dxx=set_D(x,order=orderx,d=2,output_full=True,uniform=False)
    else:
        Dxx=set_D(x,order=orderx+1,d=2,output_full=True,uniform=True)
    Dy =set_D(y,order=ordery,d=1,output_full=True,uniform=False)
    Dyy=set_D(y,order=ordery,d=2,output_full=True,uniform=False)
    #Dz =set_D(z,order=4,d=1,output_full=True,uniform=True,periodic=True)
    #Dzz=set_D(z,order=4,d=2,output_full=True,uniform=True,periodic=True)
    if orderx>1:
        ddx   = lambda a: Dx@a
    else:
        def ddx(a,x=x):
            dx=x[1]-x[0]
            da=np.zeros_like(a)
            da[1:] = a[1:]-a[:-1]
            da[0] = da[1]
            return da/dx
    d2dx2 = lambda a: Dxx@a
    ddy   = lambda a: a@Dy.T
    d2dy2 = lambda a: a@Dyy.T
    #ddz = lambda a: a@Dz.T
    #d2dz2=lambda z: a@Dzz.T
    return ddx,ddy,d2dx2,d2dy2

def set_D_ops_3D(x,y,z,orders=[4,4,4]):
    ''' set derivative operators for 3D data field
    
    Inputs:
        x,y,z - numpy 1D arrays containing location values and lengths (will use meshgrid with indexing='ij')
    Returns:
        ddx,ddy,ddz,d2dx2,d2dy2,d2dz2 - functions to compute the first and second derivatives for 3D
        orders - order of derivative in x,y, and z (list of length 3, must be even integers)
    '''
    orderx,ordery,orderz=orders
    # set derivative operators for grid data
    ddx,ddy,d2dx2,d2dy2 = set_D_ops_2D(x,y,orders=[orderx,ordery])
    # set 3D derivatives as well
    Dz =set_D(z,order=orderz,d=1,output_full=True,uniform=True,periodic=True)
    Dzz=set_D(z,order=orderz,d=2,output_full=True,uniform=True,periodic=True)
    # solve poisson equation for pressure at each of these points
    ddz = lambda a: a@Dz.T
    d2dz2=lambda z: a@Dzz.T
    def ddx_dim(a):
        nx,ny,nz = a.shape
        ddx_return = np.zeros_like(a)
        for zi in np.arange(nz):
            ddx_return[:,:,zi] = ddx(a[:,:,zi])
        return ddx_return
    def ddy_dim(a):
        nx,ny,nz = a.shape
        ddy_return = np.zeros_like(a)
        for zi in np.arange(nz):
            ddy_return[:,:,zi] = ddy(a[:,:,zi])
        return ddy_return
    def ddz_dim(a):
        nx,ny,nz = a.shape
        ddz_return = np.zeros_like(a)
        for xi in np.arange(nx):
            ddz_return[xi,:,:] = ddz(a[xi,:,:]) #@Dz.T
        return ddz_return
    def d2dx2_dim(a):
        nx,ny,nz = a.shape
        d2dx2_return = np.zeros_like(a)
        for zi in np.arange(nz):
            d2dx2_return[:,:,zi] = d2dx2(a[:,:,zi])
        return d2dx2_return
    def d2dy2_dim(a):
        nx,ny,nz = a.shape
        d2dy2_return = np.zeros_like(a)
        for zi in np.arange(nz):
            d2dy2_return[:,:,zi] = d2dy2(a[:,:,zi])
        return d2dy2_return
    def d2dz2_dim(a):
        nx,ny,nz = a.shape
        d2dz2_return = np.zeros_like(a)
        for xi in np.arange(nx):
            d2dz2_return[xi,:,:] = d2dz2(a[xi,:,:]) #@Dzz.T
        return d2dz2_return
    return ddx_dim,ddy_dim,ddz_dim,d2dx2_dim,d2dy2_dim,d2dz2_dim


# set Chebyshev operators
def set_D_Chebyshev(x,d=1,need_map=False):
    '''
    Input:
        x: array of y values satisfying xj=cos(pi j / N) for j=N,N-1,...,1,0
    Output: 
        D: derivative operator using discrete Chebyshev Transform and collocation points
    '''
    if need_map:
        D = map_D_Cheby(x,d=d,need_map=need_map)
    else:
        N=len(x)-1
        if x[0]==1.:
            order=1
        elif x[0]==-1.:
            order=-1
        D = np.zeros((N+1,N+1))
        c = np.ones(N+1)
        c[0]=c[N]=2.
        for j in np.arange(N+1):
            cj=c[j]
            xj=x[j]
            for k in np.arange(N+1):
                ck=c[k]
                xk=x[k]
                if j!=k:
                    D[j,k] = cj*(-1)**(j+k) / (ck*(xj-xk))
                elif ((j==k) and ((j!=0) and (j!=N))):
                    D[j,k] = -xj/(2.*(1.-xj**2))
                elif ((j==k) and (j==0 or j==N)):
                    D[j,k] = xj*(2.*N**2 + 1)/6.
        if d==2:
            D=D@D
    return D

def map_D_Cheby(x,d=1,need_map=False):
    if need_map:
        N=len(x)-1
        xi = np.cos(np.pi*np.arange(N+1)[::-1]/N)
        #xi=np.linspace(0,1,y.size)
        if d==1: # if 1st derivative operator d(.)/dy = d(.)/dxi * dxi/dy
            D=set_D_Chebyshev(xi,d=1,need_map=False)
            #dxdxi = D@x # matrix multiply in python3
            return np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
        elif d==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
            D1=set_D_Chebyshev(xi,d=1,need_map=False)
            D = D1@D1 # second derivative
            dxdxi = D1@x
            dxidx = 1./dxdxi # element wise invert
            #d2ydxi2 = D@y # d^2y/dxi^2
            #d2xidy2 = 1./d2ydxi2 # d^2xi/dy^2 = 1./(d^2y/dxi^2)
            d2xidx2 = -(D@x)*(dxidx)**3
            #print('d2xidy2 = ',d2xidy2)
            return (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
        else:
            print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',d)
    else:
        return D

def set_Cheby_stretched_y(y_max,ny,yi=10):
    ''' Create a stretched mesh for Chebyshev Difference operators using algebraic grid stretching suggested by Schmidt on page 486 of Stability and Transition in Shear Flows
    Inputs:
        y_max: maximum desired domain location
        ny: number of desired grid points
        yi: stretching parameter, put half of the points below this location in y
    Returns:
        y:np.array stretched grid from 0 to y_max with half of the points below yi
    '''

    xi = np.cos(np.pi*np.arange(ny)[::-1]/(ny-1))
    a=yi*y_max/(y_max-2*yi)
    b=1+2*a/y_max
    y=a*(1+xi)/(b-xi)
    return y

def set_Cheby_mapped_y(a,b,ny):
    ''' Create a stretched Chebyshev grid from [a,b] using default Chebfun like mapping
    Inputs:
        a: lower bound
        b: upper bound
        ny: number of points
    Outputs: 
        y:np.array stretched grid from a to b using ny points
    '''
    xi = np.cos(np.pi*np.arange(ny-1,-1,-1)/(ny-1))
    y = b*(xi+1)/2.0 + a*(1.0-xi)/2.0
    return y


# set Fourier discrete operator
def set_D_Fourier(t,d=1):
    ''' set derivative operator with respect to t discrete points.  Should be set by set_Fourier_t() function
    Inputs:
        t:np.array discrete points (uniform) defining a period of length t.max()+dt, where dt=t[1]-t[0]
        d=1: order of derivative (1 or 2)
    Returns:
        Dy: derivative operator with respect to t using Fourier collocation points in t.
    '''

    npts = t.size # number of points
    dt=t[1]-t[0] # uniform grid
    T = t[npts-1]+dt
    n = np.arange(0,npts)
    j = np.arange(0,npts)
    N,J = np.meshgrid(n,j,indexing='ij')
    D = np.empty(N.shape)
    JnN = (J!=N)
    JeN = (J==N)
    D[JnN] = 2.*np.pi/T*0.5*(-1.)**(N[JnN]-J[JnN])*1./np.tan(np.pi*(N[JnN]-J[JnN])/npts)
    D[JeN]=0. # fix diagonal entries
    if d==2: # if second derivative w.r.t. y
        D=D@D
    return D 

def set_Fourier_t(T=2.*np.pi,n=12):
    ''' set grid spacing for fourier collocated points.  Will return t vector from 0 to T-dt with n points.  
    Inputs:
        T=2.*np.pi: maximum domain location or period length
        n=12: number of points in domain
    Returns:
        t:np.array np.linspace(0,T,n+1)[:-1]
            
    '''
    t = np.linspace(0,T,n+1)[:-1]
    return t
    
def set_3D_D_dim(Dy,Dz,Dt):
    ''' take Dy,Dz,Dt 2D matrices and combine to make larger 2D matrices to act on q.flatten() array that is of shape q.shape = [nt,nz,ny]
    Inputs:
        Dy:np.ndarray 2D matrix operator that acts on ny by ny state variable
        Dz:np.ndarray 2D matrix operator that acts on nz by nz state variable
        Dt:np.ndarray 2D matrix operator that acts on nt by nt state variable
    Returns:
        Dy_dim:np.ndarray of size ny*nz*nt by ny*nz*nt to take derivative with respect to y
        Dz_dim:np.ndarray of size ny*nz*nt by ny*nz*nt to take derivative with respect to z
        Dt_dim:np.ndarray of size ny*nz*nt by ny*nz*nt to take derivative with respect to t
    '''

    ny = Dy.shape[0]
    nz = Dz.shape[0]
    nt = Dt.shape[0]
    # for Dy_dim
    Dy_dim = np.kron(np.eye(nz*nt),Dy)
    # for Dz_dim
    Dz_dim = np.kron(Dz,np.eye(ny))
    Dz_dim = np.kron(np.eye(nt),Dz_dim)
    # for Dt_dim
    Dt_dim = np.kron(Dt,np.eye(ny*nz))

    return Dy_dim,Dz_dim,Dt_dim

def set_2D_D_dim(Dy,Dt):
    ''' take Dy,Dt 2D matrices and combine to make larger 2D matrices to act on q.flatten() array that is of shape q.shape = [nt,ny]
    Inputs:
        Dy:np.ndarray 2D matrix operator that acts on ny by ny state variable
        Dt:np.ndarray 2D matrix operator that acts on nt by nt state variable
    Returns:
        Dy_dim:np.ndarray of size ny*nz*nt by ny*nz*nt to take derivative with respect to y
        Dt_dim:np.ndarray of size ny*nz*nt by ny*nz*nt to take derivative with respect to t
    '''

    ny = Dy.shape[0]
    nt = Dt.shape[0]
    # for Dy_dim
    Dy_dim = np.kron(np.eye(nt),Dy)
    # for Dt_dim
    Dt_dim = np.kron(Dt,np.eye(ny))

    return Dy_dim,Dt_dim
