import scipy as sp
import scipy.sparse
from . import PetscBinaryIO
import os
os.environ['PETSC_DIR'] = "/home/shaun/GITHUBs/SPE/SPE/Submodules/SPI/petsc/arch-linux-cxx-opt"

def readMat(filename):
    '''assume that a Mat is the only saved value in sparse format in the file
    Will read the file and convert to scipy.sparse.csr_matrix and output
    '''
    io = PetscBinaryIO.PetscBinaryIO()
    data = io.readBinaryFile(filename)
    print('Objects in ',filename,' = ',len(data),', read the first entry')
    ((M,N),(I,J,V)) = data[0]
    return sp.sparse.csr_matrix((V,J,I),shape=(M,N))



