import numpy as np
import DoubleLayerWithHoles_new as DL
import time
from mpi4py import MPI
import pandas as pd

#Initialize the MPI-Communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Calling process of the communicator
size = comm.Get_size() # Number of processes in the communicator

faces = 64 #subdivision by number of faces
n = faces*2 #number of nodes
alpha = (1-np.sqrt(3/5))/2
#vi1 = DL.create_ellipse(1,1,0,0,0,n)
#vi2 = np.flip(DL.create_ellipse(0.1,0.1,0,0,0,n),1)
#vi = np.array((vi1, vi2))
#collocation_nodes = np.array((DL.create_collocation_nodes(vi[0],alpha), DL.create_collocation_nodes(vi[1],alpha)))

# Parameters for the outer ellipse
a = 2
b = 2
xm = 0
ym = 0
rot = 0
big_ellipse = DL.create_ellipse(a,b,xm,ym,rot,n)

# Parameters for the inner ellpses
a = 0.08
b = 0.6
distance = 1.04
angles = np.linspace(0,2*np.pi, 5, endpoint=False)
xm = distance*np.cos(angles)
ym = distance*np.sin(angles)
rot = angles

small_ellipses = [np.flip(DL.create_ellipse(a,b,xm[i],ym[i],rot[i],n),1) for i in range(5)]

# Stack all the boundaries together
vi = np.vstack((big_ellipse[None,:,:],small_ellipses))
collocation_nodes = np.array(([DL.create_collocation_nodes(vi[i],alpha) for i in range(vi.shape[0])]))

time_start_eigenvalues = time.time()
eigenvalues, eigenvectors = DL.calculate_eigenvalues_eigenvectors_with_holes_MPI(False, vi, collocation_nodes, alpha, collocation_nodes.shape[0]*collocation_nodes.shape[2], 20, 3.0, 0.5, 24, 0.0001)
time_end_eigenvalues = time.time()

time_eigenvalues = time_end_eigenvalues - time_start_eigenvalues

# Parameters of the square, in which all points inside the domain are searched
x_start = -2
x_end = 2
y_start = -2
y_end = 2
x_steps = 101
y_steps = 101

time_start_solution = time.time()
xy_inside, sol = DL.calculate_double_layer_inside_with_holes(eigenvalues[0], vi, collocation_nodes, alpha, eigenvectors[:,0], x_start, x_end, x_steps, y_start, y_end, y_steps)
time_end_solution = time.time()

time_solution = time_end_solution - time_start_solution
time_total = time_eigenvalues + time_solution

eigenvalues_data = pd.DataFrame({"Eigenvalues" : eigenvalues})
eigenvectors_data = pd.DataFrame(eigenvectors).add_prefix('Eigenvectors')
inside_data = pd. DataFrame(xy_inside.T).add_prefix('Inside')
sol_data = pd.DataFrame(sol).add_prefix('Solution')
EigenvaluesEigenvectors = pd.concat([eigenvalues_data, eigenvectors_data, inside_data, sol_data], axis=1, ignore_index=False)


time_data = pd.DataFrame({"Processes" : [size], "TimeEigenvalues": [time_eigenvalues], "TimeSolution" : [time_solution], "TimeTotal" : [time_total]})

if rank == 0:
    EigenvaluesEigenvectors.to_csv("EigenvaluesEigenvectorsInsideSol_MPI.csv", index=False)
    time_data.to_csv("RuntimeEigenvalueSolutionTotal_MPI.csv", index=False)
