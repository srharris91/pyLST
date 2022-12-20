#!/usr/bin/env python

#from evtk.hl import gridToVTK
import evtk
import numpy as np

def export(filename,u,v,w,p,U,V,W,P,X,Y,Z):
    evtk.hl.gridToVTK(filename,X,Y,Z,pointData={"U":(U[:,:,np.newaxis]+u,V[:,:,np.newaxis]+v,W[:,:,np.newaxis]+w), "P":(P[:,:,np.newaxis]+p)})
