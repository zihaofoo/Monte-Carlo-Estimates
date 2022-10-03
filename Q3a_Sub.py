
import numpy as np
import scipy as sp
from scipy import sparse


def diffusioneqn(xgrid, F, k, source, rightbc)
    # diffusion.m: 
    
    # Solve 1-D diffusion equation with given diffusivity field k and left-hand flux F.
    
    # ARGUMENTS: 
    #     xgrid = vector with grid points
    #         F = flux at left-hand boundary, k*du/dx = -F 
    #    source = source term, either a vector of values at points in xgrid
    #             or a constant
    #   rightbc = Dirichlet BC on right-hand boundary
    #
    # Domain is given by xgrid (should be [0,1])
    #

    N = len(xgrid)
    h = xgrid[N - 1] - xgrid[N - 2] # assuming uniform grid

    # Set up discrete system f = Au + b using second-order FD
    A = sparse.csr_matrix(shape = (N - 1, N - 1), dtype = float)
    b = np.zeros((N-1, 1))
    if (np.isscalar(source)):
        f = -source * np.ones((N-1,1))
    else:
        f = -source[0: N - 2]
   
    
    # diagonal entries
    A = A - 2 * np.diag(k[0: N - 2]) - np.diag(k[1: N - 1]) - np.diag([k[0]; k[0: N - 3])

    # superdiagonal
    A = A + np.diag(k(1:N-2),1)  + diag(k(2:N-1),1);

    # subdiagonal
    A = A + diag(k(1:N-2),-1) + diag(k(2:N-1),-1);

    A = A / (2 * h^2);

    # Treat Neumann BC on left side
    A(1,2) = A(1,2) + k(1) / (h^2);
    b(1) = 2*F/h;

    % Treat Dirichlet BC on right side
    b(N-1) = rightbc * (k(N) + k(N-1)) / (2 * h^2);

    % Solve it: Au = f-b
    uinternal = A \ (f - b);

    usolution = [uinternal; rightbc];

    end