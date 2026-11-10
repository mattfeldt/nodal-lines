import numpy as np
import DoubleLayerWithHoles_MPI_new_7 as DL
import Geometries as geo
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
n_masters = 4
lines_per_task = 15


vi1 = DL.create_ellipse(1,1,0,0,0,n)
vi2 = np.flip(DL.create_ellipse(0.1,0.1,0,0,0,n),1)
vi = np.array((vi1, vi2))
collocation_nodes = np.array((DL.create_collocation_nodes(vi[0],alpha), DL.create_collocation_nodes(vi[1],alpha)))
vi = geo.transform_3d_array_in_object_array_with_2d_arrays(vi)
collocation_nodes = geo.transform_3d_array_in_object_array_with_2d_arrays(collocation_nodes)

time_start_eigenvalues = time.time()
#eigenvalues, eigenvectors = DL.calculate_eigenvalues_eigenvectors_with_holes_MPI(False, vi, collocation_nodes, alpha, 2*3*faces, 20, 3.0, 0.5, 24, 0.0001)
eigenvalues, eigenvectors = DL.calculate_eigenvalues_eigenvectors_with_holes_MPI_diff_n_master_workers(False, vi, collocation_nodes, alpha, 2*3*faces, 20, 3.0, 0.5, 24, 0.0001, n_masters, lines_per_task)
time_end_eigenvalues = time.time()

time_eigenvalues = time_end_eigenvalues - time_start_eigenvalues

# Parameters of the square, in which all points inside the domain are searched
x_start = -1
x_end = 1
y_start = -1
y_end = 1
x_steps = 101
y_steps = 101

time_start_solution = time.time()
xy_inside, sol = DL.calculate_double_layer_inside_with_holes_diff_n_master_workers(eigenvalues[0], vi, collocation_nodes, alpha, eigenvectors[:,0], x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task)
#xy_inside, sol = DL.calculate_double_layer_inside_with_holes(eigenvalues[0], vi, collocation_nodes, alpha, eigenvectors[:,0], x_start, x_end, x_steps, y_start, y_end, y_steps)
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
