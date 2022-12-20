import numpy as np

def inner_product(x,y):
    ''' Return the inner product of two vectors <x,y> or y^H x operation

    Inputs:
        x:np.ndarray containing the x in <x,y> or y^H x inner product operation
        y:np.ndarray containing the y in <x,y>

    Returns:
        y^H x: np.complex
    '''

    return y.conj()@x

def intq_dy(q:np.ndarray,y:np.ndarray):
    ''' Return integration of state vector q (shape 4*ny containing u,v,w,P discretized in y) by y using numpy.trapz
    Inputs:
        q:np.ndarray state vector (shape 4*ny containing u,v,w,P each discretized in y) by y
        y:np.ndarray wall normal discrete points (shape ny)
    Returns:
        integrate(u+v+w+P,y)
    '''

    ny=y.shape[0]
    q4=q.reshape(4,ny) # 4 primitive variables
    u=q4[0,:]
    v=q4[1,:]
    w=q4[2,:]
    P=q4[3,:]

    return np.trapz(u+v+w+P,x=y)

def max_zero_crossings_percent(params,q):
    ny = params['grid'].ny
    tol=1e-5
    #norm = q[np.argmax(np.abs(q))] # if not normed already
    #qik = q/norm # if not normed already
    qik = q.copy() # assume norm is already performed
    qik.real[np.abs(qik.real)<tol] = 0. # clip near zero
    qik.imag[np.abs(qik.imag)<tol] = 0. # clip near zero
    qik = qik.reshape(-1,ny) # reshape to u,v,w,p
    real_perc = (((np.diff(np.sign(qik.real)+(qik.real==0),axis=1))!=0).sum(axis=1)/(ny-1)).max() # calc # zero_crossings/(ny-1) for each u,v,w,P and take the max for real
    imag_perc = (((np.diff(np.sign(qik.imag)+(qik.imag==0),axis=1))!=0).sum(axis=1)/(ny-1)).max() # calc # zero_crossings/(ny-1) for each u,v,w,P and take the max for imaginary
    return np.maximum(real_perc,imag_perc) # return the maximum zero crossings
