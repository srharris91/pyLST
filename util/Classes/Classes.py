import numpy as np

class base_flow_class:
    ''' Base flow class containing velocity and pressure of current marched state and respective derivatives if desired
        
        Attributes - all default to None
        ==========
            U - streamwise velocity as function of y
            Uy- dU/dy as function of y
            Uyy-d2U/dy2 as function of y
            Ux -dU/dx as function of y
            V - wall-normal velocity as function of y
            Vy- dV/dy as function of y
            Vx- dV/dx as function of y
            P - Pressure as function of y
            Q - [U,V,P] array

    '''

    def __init__(self,ny=None,U=None,Uy=None,Uyy=None,Ux=None,V=None,Vy=None,Vx=None,P=None,Q=None):
        self.ny=ny
        self.U=U
        self.Uy=Uy
        self.Uyy=Uyy
        self.Ux=Ux
        self.V=V
        self.Vy=Vy
        self.Vx=Vx
        self.P=P
        self.Q=Q
    def get_Q(self):
        if np.any(self.Q==None):
            self.Q = np.block([self.U,self.V,self.P]).flatten()
        return self.Q
    def get_primitive(self):
        return self.get_U(), self.get_V(), self.get_P()
    def get_U(self):
        if np.any(self.U==None):
            self.U=self.Q[:self.ny]
        return self.U
    def get_V(self):
        if np.any(self.V==None):
            self.V=self.Q[self.ny:2*self.ny]
        return self.V
    def get_P(self):
        if np.any(self.P==None):
            self.P=self.Q[2*self.ny:3*self.ny-1]
        return self.P
    def get_Ux(self):
        return self.Ux
    def get_Uy(self):
        return self.Uy
    def get_Uyy(self):
        return self.Uyy
    def get_Vy(self):
        return self.Vy
    def get_Vx(self):
        return self.Vx


class grid4D_class:
    ''' grid class containing all discrete points
    Attributes - all default to None
    ==========
        x - streamwise grid (1D)
        y - wall-normal grid (1D)
        z - spanwise grid (1D)
        t - temporal grid
        X,Y,Z,T - meshgrid of x,y,z,t (4D)
    '''
    
    def __init__(self,x,y,z,t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.X,self.Y,self.Z,self.T = np.meshgrid(x,y,z,t,indexing='ij')

class grid3D_class:
    ''' grid class containing discrete points to be marched in streamwise directions
    Attributes 
    ==========
        t - temporal grid
        z - spanwise grid (1D)
        y - wall-normal grid (1D)
        ny,nz,nt - associated grid size
        T,Z,Y - meshgrid of t,z,y (3D) with indexing ij
    '''
    
    def __init__(self,y,z,t):
        self.y = y
        self.ny = y.size
        self.z = z
        self.nz = z.size
        self.t = t
        self.nt = t.size
        self.T,self.Z,self.Y = np.meshgrid(t,z,y,indexing='ij')

class grid2D_beta_class:
    ''' grid class containing discrete points to be marched in streamwise directions while holding beta const
    Attributes 
    ==========
        t - temporal grid
        y - wall-normal grid (1D)
        ny,nt - associated grid size
        T,Y - meshgrid of t,y (2D) with indexing ij
    '''
    
    def __init__(self,t,y):
        self.y = y
        self.ny = y.size
        self.t = t
        self.nt = t.size
        self.T,self.Y = np.meshgrid(t,y,indexing='ij')

    def __str__(self):
        return ''' grid2D_beta_class 
        nt,ny = {:g},{:g}'''.format(self.nt,self.ny)

class grid2D_omega_class:
    ''' grid class containing discrete points to be marched in streamwise directions while holding omega const
    Attributes 
    ==========
        z - spanwise grid (1D)
        y - wall-normal grid (1D)
        ny,nz - associated grid size
        Z,Y - meshgrid of z,y (2D) with indexing ij
    '''
    
    def __init__(self,y,z):
        self.y = y
        self.ny = y.size
        self.z = z
        self.nz = z.size
        self.Z,self.Y = np.meshgrid(z,y,indexing='ij')

class grid1D_omega_beta_class:
    ''' grid class containing discrete points to be marched in streamwise directions while holding omega and beta const
    Attributes
    ==========
        y - wall-normal grid (1D)
        ny- associated grid size
    '''
    
    def __init__(self,y):
        self.y = y
        self.ny = y.size

    def __str__(self):
        return ''' grid1D_omega_beta_class 
        ny = {:g}'''.format(self.ny)
