import numpy as np
import DoubleLayerWithHoles_MPI_new_8 as DL
import time
from mpi4py import MPI
import pandas as pd
from scipy.optimize import least_squares
from matplotlib import path
import scipy as sc
import Geometries_2 as geo
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

#Initialize the MPI-Communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Calling process of the communicator
size = comm.Get_size() # Number of processes in the communicator
faces = np.array([50,60,60,60])
#faces = np.array([22,40]) #subdivision by number of faces
# faces = 30
#faces = 70 #subdivision by number of faces
n = faces*2 #number of nodes
alpha = (1-np.sqrt(3/5))/2
max_mu = 10.5

# Parameters for the outer ellipse
a_big_ell = 2
b_big_ell = 2
xm_big_ell = 0
ym_big_ell = 0
rot_big_ell = 0
n_ellipses = 6
n_ellipses_single = 0
#big_ellipse = DL.create_ellipse(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,n)

# Parameters for the inner ellpses
# a_small_ell = 0.08
# b_small_ell = 0.6
# distance_small_ell = 1.04
#a_small_ell = _a_small_ell_
#b_small_ell = _b_small_ell_
#distance_small_ell = _distance_small_ell_
a_small_ell = 0.15584574005610
b_small_ell = 0.39126299243500
distance_small_ell = 1.07497307444190
angles_small_ell = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
xm_small_ell = distance_small_ell*np.cos(angles_small_ell)
ym_small_ell = distance_small_ell*np.sin(angles_small_ell)
rot_small_ell = angles_small_ell
eps = 0.2200436267753381
#small_ellipses = [np.flip(DL.create_ellipse(a_small_ell,b_small_ell,xm_small_ell[i],ym_small_ell[i],rot_small_ell[i],n),1) for i in range(n_ellipses)]

# Parameters for the inner hypotrochoid
xa_hyp = 0.8
xi_hyp = 0.6621178142898738
A_hyp = n_ellipses
r_hyp = (xa_hyp+xi_hyp)/(2*(A_hyp-1))
R_hyp = A_hyp*r_hyp
a_hyp = (A_hyp-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp)
xm_hyp = 0
ym_hyp = 0
rot_hyp = 0

n_masters = 4
lines_per_task = 15
#hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n),1)

# Stack all the boundaries together
#vi = np.vstack((big_ellipse[None,:,:],small_ellipses, hypotrochoid[None,:,:]))
#collocation_nodes = np.array(([DL.create_collocation_nodes(vi[i],alpha) for i in range(vi.shape[0])]))



def create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,xa_hyp,xi_hyp,xm_hyp,ym_hyp,rot_hyp,n):
    # Outer ellipse
    big_ellipse = DL.create_ellipse(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,n)
    # Inner ellipses
    angles_small_ell = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_small_ell = distance_small_ell*np.cos(angles_small_ell)
    ym_small_ell = distance_small_ell*np.sin(angles_small_ell)
    rot_small_ell = angles_small_ell
    small_ellipses = [np.flip(DL.create_ellipse(a_small_ell,b_small_ell,xm_small_ell[i],ym_small_ell[i],rot_small_ell[i],n),1) for i in range(n_ellipses)]
    # Hypotrochoid
    r_hyp = (xa_hyp+xi_hyp)/(2*(A_hyp-1));
    R_hyp = n_ellipses*r_hyp;
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp);
    hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n),1)
    # Stack all the boundaries together
    vi_first_eig = np.vstack((big_ellipse[None,:,:],small_ellipses, hypotrochoid[None,:,:]))
    collocation_nodes_first_eig = np.array(([DL.create_collocation_nodes(vi_first_eig[i],alpha) for i in range(vi_first_eig.shape[0])]))
    return vi_first_eig, collocation_nodes_first_eig

def create_6_ellipses(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,n):
    # Outer ellipse
    big_ellipse = DL.create_ellipse(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,n)
    # Inner ellipses
    angles_small_ell = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_small_ell = distance_small_ell*np.cos(angles_small_ell)
    ym_small_ell = distance_small_ell*np.sin(angles_small_ell)
    rot_small_ell = angles_small_ell
    small_ellipses = [np.flip(DL.create_ellipse(a_small_ell,b_small_ell,xm_small_ell[i],ym_small_ell[i],rot_small_ell[i],n),1) for i in range(n_ellipses)]
    # Stack all the boundaries together
    vi_second_eig = np.vstack((big_ellipse[None,:,:],small_ellipses))
    collocation_nodes_second_eig = np.array(([DL.create_collocation_nodes(vi_second_eig[i],alpha) for i in range(vi_second_eig.shape[0])]))
    return vi_second_eig, collocation_nodes_second_eig

#vi_first_eig, collocation_nodes_first_eig = create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n)
#vi_first_eig, collocation_nodes_first_eig = create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,6,xa_hyp,xi_hyp,0,0,0,n)

#vi_second_eig, collocation_nodes_second_eig = create_6_ellipses(a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,6,n)

def ray_tracing_overlaps(boundary, x_start, x_end, x_steps, y_start, y_end, y_steps):
    """
    Routine to find all the points of a square on an equidistant grid, that lie within the given domain without the given holes.

    Parameters
    ----------
    boundary : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    x_start : float
        Start value of a square in x-direction.
    x_end : float
        End value of a square in x-direction.
    x_steps : int
        Number of points in x-direction that should be checked to be inside the given domain without holes.
    y_start : float
        Start value of a square in y-direction.
    y_end : float
        End value of a square in y-direction.
    y_steps : int
        Number of points in y-direction that should be checked to be inside the given domain without holes.

    Returns
    -------
    xy_inside : numpy.ndarray
        Numpy array containing all points that lie inside of the given domain without the holes.
    """
    # Create a path for the outer boundary of the domain
    p1 = path.Path(boundary[0].T)
    # Create a path for the boundary of the holes
    p2 = [path.Path(boundary[i].T) for i in range(1, boundary.shape[0])]
    # Create the mesh of the square to be checked, whether they are inside or outside of the domain with holes.
    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    mesh = np.reshape(mesh,(x_steps*y_steps,2))
    # Check whether the mesh is inside or outside of the outer boundary of the domain (without holes).
    inside_with_hole = p1.contains_points(mesh)
    # Check to see if the mesh is inside or outside the holes.
    hole = np.array([p2[i].contains_points(mesh) for i in range(boundary.shape[0]-1)])
    # Count the number of overlaps between holes.
    overlap_holes = np.sum(np.sum(hole, axis = 0)>1)
    # Count the number of overlaps between the holes and the boundary of the domain
    overlap_holes_domain = np.sum(np.all(np.stack((inside_with_hole == False, np.any(hole == True, axis=0))),axis=0))
    n_overlaps = overlap_holes + overlap_holes_domain
    penalty = n_overlaps * 2
    return penalty

def set_solution_everywhere(xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps):
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    xy_inside_T = np.real(xy_inside.T)

    sol_everywhere = np.empty((mesh.shape[0],mesh.shape[1]),dtype=np.complex128)
    sol_everywhere.fill(np.nan)
    
    ave_n_rows_xy_inside, res_n_rows_xy_inside = divmod(xy_inside_T.shape[0], size)
    n_lines = np.concatenate((np.repeat(ave_n_rows_xy_inside+1, res_n_rows_xy_inside), np.repeat(ave_n_rows_xy_inside, size-res_n_rows_xy_inside)))
    # Caluculate cummulative number of lines the processes have to calculate starting from zero
    cumsum_lines = np.cumsum(n_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)

    eps = 0.000001
    mask_partial=np.abs(abs(mesh[:,:,None] - xy_inside_T[cumsum_lines[rank]:cumsum_lines[rank+1]])<eps).all(-1).any(-1)
    mask_global = np.empty((mesh.shape[0], mesh.shape[1]), dtype = bool)
    comm.Allreduce(mask_partial, mask_global, op=MPI.LOR)
    
    sol_everywhere[mask_global] = sol
    sol_everywhere.reshape((y_steps,x_steps))
    return sol_everywhere

def count_nodal_domains(xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps,barrier):
    sol_everywhere = set_solution_everywhere(xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps)
    sol_everywhere_real = np.real(sol_everywhere)
    pos_mask = sol_everywhere_real > 0
    neg_mask = sol_everywhere_real < 0
    
    pos_labeled, num_pos_domains = sc.ndimage.label(pos_mask)
    pos_labels, num_pos_labels = np.unique(pos_labeled, return_counts=True)
    mask_val_pos_labels = num_pos_labels>barrier
    num_val_pos_labels = sum(mask_val_pos_labels[1:])
    
    neg_labeled, num_neg_domains = sc.ndimage.label(neg_mask)
    neg_labels, num_neg_labels = np.unique(neg_labeled, return_counts=True)
    mask_val_neg_labels = num_neg_labels>barrier
    num_val_neg_labels = sum(mask_val_neg_labels[1:])
    
    if num_val_pos_labels != 0 and num_val_neg_labels != 0:
        num_nodal_domains = num_val_pos_labels + num_val_neg_labels
    elif num_val_pos_labels == 0 and num_val_neg_labels == 0:
        num_nodal_domains = 0
    else:
        num_nodal_domains = 1
    if rank == 0:
        print(f"num_val_pos_labels is {num_val_pos_labels}; num_val_neg_labels is {num_val_neg_labels}; num_nodal_domains is {num_nodal_domains}")
    #num_nodal_domains = num_val_pos_labels + num_val_neg_labels
    return num_nodal_domains

def calculate_second_eigenvalues(vi_second_eig, collocation_nodes_second_eig, alpha, n_masters, lines_per_task):
    # Calculations for the second eigenvalue (without hypotrochoid)
    n_eigenvalues = 0
    eigenvalues_list = []
    eigenvectors_list = []
    artifacts = []
    first_eigenvalue = False
    second_eigenvalue = False
    third_eigenvalue = False

    for mu_temp in np.arange(2.4, max_mu, 0.9):
        #if rank == 0:
        #    print("Before Calculation")
        if vi_second_eig.dtype == object:
            eigenvalues_temp, eigenvectors_temp = DL.calculate_eigenvalues_eigenvectors_with_holes_MPI_diff_n_master_workers(False, vi_second_eig, collocation_nodes_second_eig, alpha, np.sum([collocation_nodes_second_eig[i].shape[1] for i in range(collocation_nodes_second_eig.size)]), 24, mu_temp, 0.5, 80, 0.0001, n_masters, lines_per_task)
        elif collocation_nodes_second_eig.ndim == 3:
            eigenvalues_temp, eigenvectors_temp = DL.calculate_eigenvalues_eigenvectors_with_holes_MPI(False, vi_second_eig, collocation_nodes_second_eig, alpha, collocation_nodes_second_eig.shape[0]*collocation_nodes_second_eig.shape[2], 20, mu_temp, 0.5, 24, 0.0001)
        elif collocation_nodes_second_eig.ndim == 2:
            if rank == 0:
                print(f"I'm here",flush = True)
            eigenvalues_temp, eigenvectors_temp = DL.calculate_eigenvalues_eigenvectors(False, vi_second_eig, collocation_nodes_second_eig, alpha, collocation_nodes_second_eig.shape[1], 20, mu_temp, 0.5, 23, 0.0001)
        else:
            if rank == 0:
                print("Error")
            break
        #if rank == 0:
        #    print(f"Shape collocation_nodes is {collocation_nodes_second_eig.shape}")
        #    print(f"Shape eigenvalues_temp is {eigenvalues_temp.shape}")
        #    print(f"Shape eigenvectors_temp is {eigenvectors_temp.shape}")
        #if rank == 0:
        #    print("After Calculation")
        eigenvalues_idx_bigger = (mu_temp - 0.45 <= np.real(eigenvalues_temp))
        eigenvalues_idx_smaller = (np.real(eigenvalues_temp) <= mu_temp + 0.45)
        eigenvalues_idx_intervall = np.logical_and(eigenvalues_idx_bigger, eigenvalues_idx_smaller)
        eigenvalues_intervall = eigenvalues_temp[eigenvalues_idx_intervall]
        eigenvectors_intervall = eigenvectors_temp[:,eigenvalues_idx_intervall]
        eigenvalues_idx = np.abs(np.imag(eigenvalues_intervall))<0.0001
        eigenvalues_true = eigenvalues_intervall[eigenvalues_idx]
        eigenvectors_true = eigenvectors_intervall[:,eigenvalues_idx]
        #eigenvalues_idx = np.abs(np.imag(eigenvalues_temp))<0.0001
        #eigenvalues_true = eigenvalues_temp[eigenvalues_idx]
        #eigenvectors_true = eigenvectors_temp[:,eigenvalues_idx]
        if len(eigenvalues_list) >= 1:
            eigenvalues_list_array = np.concatenate([np.atleast_1d(x) for x in eigenvalues_list])
        else:
            eigenvalues_list_array = np.array([])
        no_duplicates = np.isin(np.real(eigenvalues_true).round(decimals=8), np.real(eigenvalues_list_array).round(decimals=8), invert=True)
        eigenvalues_true = eigenvalues_true[no_duplicates]
        eigenvectors_true = eigenvectors_true[:,no_duplicates]
        if collocation_nodes_second_eig.ndim == 2:
            n_colls = np.array([collocation_nodes_second_eig.shape[1]])
        else:
            n_colls = np.array([colls.shape[1] for colls in collocation_nodes_second_eig])
        sum_colls = np.insert(np.cumsum(n_colls),0,0)
        eps_null_vector = 1e-10
        #eigenvectors_without_null_vector = []
        #eigenvalues_without_null_vector = []
        eigenvectors_without_null_vector = np.empty((sum_colls[-1],0))
        eigenvalues_without_null_vector = np.array([])
        for i_eigenvector in range(eigenvectors_true.shape[1]):
            null_vector = False
            if vi_second_eig.dtype == object:
                n_boundaries = vi_second_eig.shape[0]
            elif vi_second_eig.ndim == 2:
                n_boundaries = 1
            else:
                n_boundaries = vi_second_eig.shape[0]
            for boundary in range(n_boundaries):
                mask_null_vector = np.abs(np.real(eigenvectors_true))[sum_colls[boundary]:sum_colls[boundary+1],i_eigenvector] < eps_null_vector
                if np.all(mask_null_vector):
                    null_vector = True
                    if rank == 0:
                        print(f"eigenvector {i_eigenvector} is nullvector",flush = True)
                    #eigenvectors_without_null_vector = np.delete(eigenvectors_without, i_eigenvector, axis = 1) 
                    break
            if null_vector == False:
                if rank == 0:
                    print(f"eigenvector {i_eigenvector} is NO nullvector",flush = True)
                #eigenvectors_without_null_vector = np.append(eigenvectors_without_null_vector, eigenvectors_true[:,i_eigenvector:i_eigenvector+1], axis = 0)
                #eigenvalues_without_null_vector = np.append(eigenvalues_without_null_vector, eigenvalues_true[i_eigenvector])
                eigenvectors_without_null_vector = np.hstack([eigenvectors_without_null_vector, eigenvectors_true[:,i_eigenvector:i_eigenvector+1]])
                eigenvalues_without_null_vector = np.hstack([eigenvalues_without_null_vector, eigenvalues_true[i_eigenvector]])
                #if rank == 0:
                #    print(f"eigenvectors_without_null_vector.shape is {eigenvectors_without_null_vector.shape}", flush = True)
        #if len(eigenvalues_without_null_vector) == 0:
        #    eigenvectors_without_null_vector = np.empty((0,sum_colls[-1], 1))
        #else:
        #    eigenvectors_without_null_vector = np.array(eigenvectors_without_null_vector)
        eigenvalues_without_null_vector = np.array(eigenvalues_without_null_vector)
        index_sort_eigen = np.argsort(eigenvalues_without_null_vector)
        eigenvalues_sort = eigenvalues_without_null_vector[index_sort_eigen]
        eigenvectors_sort = eigenvectors_without_null_vector[:,index_sort_eigen]
        unique_eigenvalues, indices_eigenvectors, indices, counts = np.unique(np.real(eigenvalues_sort).round(decimals=8), return_index=True, return_counts=True, return_inverse = True)
        unique_eigenvalues = eigenvalues_sort[indices_eigenvectors]
        if rank == 0:
            print(f"unique_eigenvalues is {unique_eigenvalues}",flush=True)
            print(f"unique_eigenvalues count is {counts}",flush=True)
        unique_eigenvectors = eigenvectors_sort[:,indices_eigenvectors]
        n_eigenvalues_temp = unique_eigenvalues.size
        #if rank == 0:
        #    print("Before check")
        if n_eigenvalues_temp > 0:
            for i in range(n_eigenvalues_temp):
                #if rank == 0: 
                #    print(f"n_eigenvalues_temp is {n_eigenvalues_temp}")
                if first_eigenvalue == False and counts[i]==1:
                    x_start = -2.05
                    x_end = 2.05
                    y_start = -2.05
                    y_end = 2.05
                    x_steps = 81
                    y_steps = 81
                    #if rank == 0:
                    #    print("If 1 before")
                    if vi_second_eig.dtype == object:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes_diff_n_master_workers(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task)
                    elif vi_second_eig.ndim == 3:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    elif vi_second_eig.ndim == 2:
                        y_inside, sol = DL.calculate_double_layer_inside(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    else:
                        if rank == 0:
                            print("Error")
                        break
                    if rank == 0:
                        print(f"Shape collocation_nodes is {collocation_nodes_second_eig.shape}")
                    abs_sol = np.abs(sol)
                    max_sol = np.max(abs_sol)

                    num_nodal_domains = count_nodal_domains(y_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps,25)
                    if rank == 0:
                        print(f"Number of nodal domains is {num_nodal_domains}",flush=True)
                    if rank == 0:
                        print(f"Max sol is {max_sol}")
                        print(f"Above are {np.sum(abs_sol > max_sol/2)}")
                        print(f"Eigenvalues are {unique_eigenvalues[i]}")
                    if num_nodal_domains ==1 and max_sol>0.00001:
                        first_eigenvalue = True
                        eigenvalues_list.append(unique_eigenvalues[i])
                        eigenvectors_list.append(unique_eigenvectors[:,i])
                        n_eigenvalues = n_eigenvalues + 1
                        if rank == 0:
                            print(f"First eigenvalue is {unique_eigenvalues[i]}",flush=True)
                    else:
                        artifacts.append(eigenvalues_sort[np.argwhere(indices == i)[:,0]])
                elif first_eigenvalue == True and second_eigenvalue == False:# and counts[i]<=2:
                    x_start = -2.05
                    x_end = 2.05
                    y_start = -2.05
                    y_end = 2.05
                    x_steps = 81
                    y_steps = 81
                    #if rank == 0:
                    #    print("If 2 before")
                    if rank == 0:
                        print(f"Shape collocation_nodes is {collocation_nodes_second_eig.shape}",flush=True)
                    if vi_second_eig.dtype == object:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes_diff_n_master_workers(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task)
                    elif vi_second_eig.ndim == 3:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    elif vi_second_eig.ndim ==2: 
                        y_inside, sol = DL.calculate_double_layer_inside(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    else:
                        if rank == 0:
                            print("Error",flush=True)
                        break
                    abs_sol = np.abs(sol)
                    max_sol = np.max(abs_sol)
                    #if rank <= 10:
                    #    print(f"Max sol is {max_sol}")
                    #    print(f"Above are {np.sum(abs_sol > max_sol/2)}")
                    #    print(f"Eigenvalues are {eigenvalues_sort[i]}")
                    
                    num_nodal_domains = count_nodal_domains(y_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps,10)
                    if rank == 0:
                        print(f"Current eigenvalue is {unique_eigenvalues[i]} and number of nodal domains is {num_nodal_domains}",flush=True)

                    if num_nodal_domains == 2  and max_sol>0.00001:
                        second_eigenvalue = True
                        indices_temp = np.argwhere(indices == i)[:,0]
                        eigenvalues_list.append(eigenvalues_sort[indices_temp])
                        eigenvectors_list.append(eigenvectors_sort[:,indices_temp])
                        n_eigenvalues = n_eigenvalues + 1
                    else:
                        artifacts.append(eigenvalues_sort[np.argwhere(indices == i)[:,0]])
                elif first_eigenvalue == True and second_eigenvalue == True and third_eigenvalue == False:# and counts[i]<=2:
                    x_start = -2.05
                    x_end = 2.05
                    y_start = -2.05
                    y_end = 2.05
                    x_steps = 81
                    y_steps = 81
                    #if rank == 0:
                    #    print("If 2 before")
                    if rank == 0:
                        print(f"Shape collocation_nodes is {collocation_nodes_second_eig.shape}",flush=True)
                    if vi_second_eig.dtype == object:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes_diff_n_master_workers(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task)
                    elif vi_second_eig.ndim == 3:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    elif vi_second_eig.ndim ==2: 
                        y_inside, sol = DL.calculate_double_layer_inside(unique_eigenvalues[i], vi_second_eig, collocation_nodes_second_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    else:
                        if rank == 0:
                            print("Error",flush=True)
                        break
                    abs_sol = np.abs(sol)
                    max_sol = np.max(abs_sol)
                    #if rank <= 10:
                    #    print(f"Max sol is {max_sol}")
                    #    print(f"Above are {np.sum(abs_sol > max_sol/2)}")
                    #    print(f"Eigenvalues are {eigenvalues_sort[i]}")
                    
                    num_nodal_domains = count_nodal_domains(y_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps,10)
                    if rank == 0:
                        print(f"Current eigenvalue is {unique_eigenvalues[i]} and number of nodal domains is {num_nodal_domains}",flush=True)
                    
                    if rank == 0:
                        print(f"I'm in third part 1 and max_sol is {max_sol}", flush = True)
                    if num_nodal_domains <= 4 and max_sol>0.00001:
                        if rank == 0:
                            print("I'm in third part 2", flush = True)
                        indices_temp = np.argwhere(indices == i)[:,0]
                        eigenvalues_list.append(eigenvalues_sort[indices_temp])
                        eigenvectors_list.append(eigenvectors_sort[:,indices_temp])

                        if len(eigenvalues_list) >= 1:
                            eigenvalues_list_array = np.concatenate([np.atleast_1d(x) for x in eigenvalues_list])
                        else:
                            eigenvalues_list_array = np.array([])
                        if rank == 0:
                            print(f"eigenvalues_list_array is {eigenvalues_list_array}", flush=True)
                            print(f"eigenvalues_list_array diff is {np.abs(np.real(eigenvalues_list_array[-1]-eigenvalues_list_array[1]))}", flush=True)
                        if np.abs(np.real(eigenvalues_list_array[-1]-eigenvalues_list_array[1])) >= 0.00001:
                            third_eigenvalue = True
                            if rank == 0:
                                print(f"third_eigenvalue is {third_eigenvalue}",flush = True)
                            n_eigenvalues = n_eigenvalues + 1
                    else:
                        artifacts.append(eigenvalues_sort[np.argwhere(indices == i)[:,0]])
                else:
                    artifacts.append(eigenvalues_sort[np.argwhere(indices == i)[:,0]])
                if rank == 0:
                    print(f"End Eigenvalues are {eigenvalues_list}")
                    print(f"End Artifacts are {artifacts}")

                #try:
                #    EigenvaluesEigenvectors = pd.read_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv")
                #    print("Try ", flush = True)
                #except:
                #    EigenvaluesEigenvectors = pd.DataFrame()
                #    print("Except ", flush = True)
                try:
                    eigenvalues = unique_eigenvalues[i]
                    eigenvalues_data = pd.DataFrame({"Eigenvalues" : [eigenvalues]})

                    eigenvectors = unique_eigenvectors[:,i]
                    eigenvectors_data = pd.DataFrame(eigenvectors).add_prefix('Eigenvectors')
                
                    inside_data = pd. DataFrame(y_inside.T).add_prefix('Inside')
                    sol_data = pd.DataFrame(sol).add_prefix('Solution')
                    new_EigenvaluesEigenvectors = pd.concat([eigenvalues_data, eigenvectors_data, inside_data, sol_data], axis=1, ignore_index=False)
                
                    #EigenvaluesEigenvectors = pd.concat([EigenvaluesEigenvectors, new_EigenvaluesEigenvectors], axis = 1)
    
                    #time_data = pd.DataFrame({"Processes" : [size], "TimeEigenvalues": [time_eigenvalues], "TimeSolution" : [time_solution], "TimeTotal" : [time_total]})

                    if rank == 0:
                        #EigenvaluesEigenvectors.to_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv", index=False)
                        try:
                            EigenvaluesEigenvectors = pd.read_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv")
                            print("Try ", flush = True)
                        except:
                            EigenvaluesEigenvectors = pd.DataFrame()
                            print("Except ", flush = True)

                        #eigenvalues = unique_eigenvalues[i]
                        #eigenvalues_data = pd.DataFrame({"Eigenvalues" : [eigenvalues]})

                        #eigenvectors = unique_eigenvectors[:,i]
                        #eigenvectors_data = pd.DataFrame(eigenvectors).add_prefix('Eigenvectors')
                
                        #inside_data = pd. DataFrame(y_inside.T).add_prefix('Inside')
                        #sol_data = pd.DataFrame(sol).add_prefix('Solution')
                        #new_EigenvaluesEigenvectors = pd.concat([eigenvalues_data, eigenvectors_data, inside_data, sol_data], axis=1, ignore_index=False)
                
                        EigenvaluesEigenvectors = pd.concat([EigenvaluesEigenvectors, new_EigenvaluesEigenvectors], axis = 1)
    
                        EigenvaluesEigenvectors.to_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv", index=False)
                        #time_data = pd.DataFrame({"Processes" : [size], "TimeEigenvalues": [time_eigenvalues], "TimeSolution" : [time_solution], "TimeTotal" : [time_total]})
                        #time_data.to_csv("RuntimeEigenvalueSolutionTotal_Dahne_MPI.csv", index=False)
                except:
                    if rank == 0:
                        print("Multiplicity of eigenvalue is too high; It's not saved in csv file", flush = True)

                #if third_eigenvalue == True:
                if second_eigenvalue == True:
                    #if rank == 0:
                    #    print("First break", flush = True)
                    break
        
        #if n_eigenvalues >= 3:
        #if third_eigenvalue == True:
        if second_eigenvalue == True:
            #print("Before break")
            #if rank == 0:
            #    print("Second break", flush = True)
            break
    eigenvalues = np.hstack((eigenvalues_list))
    if len(eigenvectors_list[0].shape) == 1 and len(eigenvectors_list[1].shape) > 1:
        eigenvectors = np.hstack((eigenvectors_list[0][:,None], eigenvectors_list[1]))
    elif len(eigenvectors_list[0].shape) == 1 and len(eigenvectors_list[1].shape) == 1:
        eigenvectors = np.hstack((eigenvectors_list[0][:,None], eigenvectors_list[1][:,None]))
    else:
        eigenvectors = np.hstack((eigenvectors_list))
    return eigenvalues[1:], eigenvectors[:,1:]

def calculate_first_eigenvalue(vi_first_eig, collocation_nodes_first_eig, alpha):
    # Calculations for the second eigenvalue (without hypotrochoid)
    n_eigenvalues = 0
    eigenvalues_list = []
    eigenvectors_list = []
    artifacts = []
    first_eigenvalue = False
    second_eigenvalue = False
    
    for mu_temp in np.arange(3,max_mu, 0.9):
        if collocation_nodes_first_eig.ndim == 3:
            eigenvalues_temp, eigenvectors_temp = DL.calculate_eigenvalues_eigenvectors_with_holes_MPI(False, vi_first_eig, collocation_nodes_first_eig, alpha, collocation_nodes_first_eig.shape[0]*collocation_nodes_first_eig.shape[2], 20, mu_temp, 0.5, 24, 0.0001)
        elif collocation_nodes_first_eig.ndim == 2:
            if rank == 0:
                print(f"I'm here",flush = True)
            eigenvalues_temp, eigenvectors_temp = DL.calculate_eigenvalues_eigenvectors(False, vi_first_eig, collocation_nodes_first_eig, alpha, collocation_nodes_first_eig.shape[1], 20, mu_temp, 0.5, 30, 0.0001)
        else:
            if rank == 0:
                print("Error")
        eigenvalues_idx = np.abs(np.imag(eigenvalues_temp))<0.00001
        eigenvalues_true = eigenvalues_temp[eigenvalues_idx]
        eigenvectors_true = eigenvectors_temp[:,eigenvalues_idx]
        no_duplicates = np.isin(np.real(eigenvalues_true).round(decimals=3), np.real(eigenvalues_list).round(decimals=3), invert=True)
        eigenvalues_true = eigenvalues_true[no_duplicates]
        eigenvectors_true = eigenvectors_true[:,no_duplicates]
        index_sort_eigen = np.argsort(eigenvalues_true)
        eigenvalues_sort = eigenvalues_true[index_sort_eigen]
        eigenvectors_sort = eigenvectors_true[:,index_sort_eigen] 
        unique_eigenvalues, indices_eigenvectors, indices, counts = np.unique(np.real(eigenvalues_sort).round(decimals=3), return_index=True, return_counts=True, return_inverse = True)
        unique_eigenvectors = eigenvectors_sort[:,indices_eigenvectors]
        #unique_eigenvalues, indices, counts = np.unique(np.real(eigenvalues_sort).round(decimals=3), return_counts=True, return_inverse = True)
        n_eigenvalues_temp = unique_eigenvalues.size
        if n_eigenvalues_temp > 0:
            for i in range(n_eigenvalues_temp):
                if first_eigenvalue == False and counts[i]==1:
                    x_start = -2.05
                    x_end = 2.05
                    y_start = -2.05
                    y_end = 2.05
                    x_steps = 31
                    y_steps = 31
                    if vi_first_eig.dtype == object:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes_diff_n_master_workers(unique_eigenvalues[i], vi_first_eig, collocation_nodes_first_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task)
                    elif vi_first_eig.ndim == 3:
                        y_inside, sol = DL.calculate_double_layer_inside_with_holes(unique_eigenvalues[i], vi_first_eig, collocation_nodes_first_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps) 
                    elif vi_first_eig.ndim == 2:
                        y_inside, sol = DL.calculate_double_layer_inside(unique_eigenvalues[i], vi_first_eig, collocation_nodes_first_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    else:
                        if rank == 0:
                            print("Error",flush = True)
                        break
                    #y_inside, sol = DL.calculate_double_layer_inside_with_holes(unique_eigenvalues[i], vi_first_eig, collocation_nodes_first_eig, alpha, unique_eigenvectors[:,i], x_start, x_end, x_steps, y_start, y_end, y_steps)
                    abs_sol = np.abs(sol)
                    max_sol = np.max(abs_sol)
                    #print(f"Max sol is {max_sol}")
                    #print(f"Above are {np.sum(abs_sol > max_sol/2)}")
                    #print(f"Eigenvalues are {eigenvalues_sort[i]}")
                    #if rank == 0:
                    #    print(f"Max sol is {max_sol}")
                    #    print(f"Above are {np.sum(abs_sol > max_sol/2)}")
                    #    print(f"Eigenvalues are {eigenvalues_sort[i]}")

                    num_nodal_domains = count_nodal_domains(y_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps)
                    if rank == 0:
                        print(f"Number of nodal domains is {num_nodal_domains}",flush=True)
                    if rank == 0:
                        print(f"Max sol is {max_sol}",flush = True)
                        print(f"Above are {np.sum(abs_sol > max_sol/2)}", flush = True)
                        print(f"Eigenvalues are {unique_eigenvalues[i]}", flush = True)

                    if num_nodal_domains == 1 and max_sol > 0.0001: 
                        first_eigenvalue = True
                        eigenvalues_list.append(unique_eigenvalues[i])
                        eigenvectors_list.append(unique_eigenvectors[:,i])
                        n_eigenvalues = n_eigenvalues + 1
                    else:
                        artifacts.append(eigenvalues_sort[np.argwhere(indices == i)[:,0]])
                else:
                    artifacts.append(eigenvalues_sort[np.argwhere(indices == i)[:,0]])
                if rank == 0:
                    print(f"End Eigenvalues are {eigenvalues_list}",flush=True)
                    print(f"End Artifacts are {artifacts}",flush=True)
            if n_eigenvalues >= 1:
                #print(f"Rank {rank} break")
                break
        #print("Here1")
        if n_eigenvalues >=1:
            #print(f"Rank {rank} break 2")
            break
    eigenvalues = np.hstack((eigenvalues_list))
    if len(eigenvectors_list[0].shape) == 1:
         eigenvectors = np.hstack((eigenvectors_list[0][:,None]))
    else:
        eigenvectors = np.hstack((eigenvectors_list))
    return eigenvalues[0]


def minimize_distance_of_eigenvalues(x,a_big_ell,b_big_ell,xa_hyp,xi_hyp,alpha,n_ellipses,n):
    a_small_ell,b_small_ell,distance_small_ell = x[0], x[1], x[2]
    if rank == 0:
        parameters = pd.read_csv('Parameters_Eigenvalues.csv', low_memory=False)
        parameters_new = pd.DataFrame({'a' : [a_small_ell], 'b' : [b_small_ell], 'distance' : [distance_small_ell], 'loss': [0], 'second_eig' : [0], 'first_eig': [0]})
        parameters = parameters._append(parameters_new)
        parameters.to_csv('Parameters_Eigenvalues.csv', index = False)

    vi_first_eig, collocation_nodes_first_eig = create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,xa_hyp,xi_hyp,0,0,0,n)
    vi_second_eig, collocation_nodes_second_eig = create_6_ellipses(a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,n)
    vi_pen_first_eig, collocation_nodes_pen_first_eig = create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,0,0,0,a_small_ell*1.01,b_small_ell*1.01,distance_small_ell,n_ellipses,xa_hyp,xi_hyp,0,0,0,n)
    penalty = ray_tracing_overlaps(vi_pen_first_eig, -2.1, 2.1, 203, -2.1, 2.1, 203)
    if penalty == 0:
        second_eigenvalues = calculate_second_eigenvalues(vi_second_eig, collocation_nodes_second_eig, alpha)
        first_eigenvalue = calculate_first_eigenvalue(vi_first_eig, collocation_nodes_first_eig, alpha)
        loss1 = np.abs(first_eigenvalue - second_eigenvalues[0])
        loss2 = xa_hyp + a_small_ell - distance_small_ell
        loss = loss1 + loss2
    else:
        loss = penalty
        first_eigenvalue = 0
        second_eigenvalues = 0
    if rank == 0:
        print(f"First Eigenvalue is {first_eigenvalue}")
        print(f"Second Eigenvalue is {second_eigenvalues}")
        parameters = pd.read_csv('Parameters_Eigenvalues.csv', low_memory=False)
        parameters_new = pd.DataFrame({'a' : [a_small_ell], 'b' : [b_small_ell], 'distance' : [distance_small_ell], 'loss': [loss], 'second_eig' : [second_eigenvalues], 'first_eig': [first_eigenvalue]})
        parameters = parameters._append(parameters_new)
        parameters.to_csv('Parameters_Eigenvalues.csv', index = False)
    return loss

time_start_eigenvalues = time.time()

a_small_ell_0 = 0.08
b_small_ell_0 = 0.4
distance_small_ell_0 = 1.04
xa_hyp_0 = 0.8
xi_hyp_0 = 0.7
x0 = np.array([a_small_ell_0,b_small_ell_0,distance_small_ell_0])
if rank == 0:
    parameters = pd.DataFrame({'a' : [a_small_ell], 'b' : [b_small_ell], 'distance' : [distance_small_ell], 'loss': [0], 'second_eig' : [0], 'first_eig': [0]})
    parameters.to_csv('Parameters_Eigenvalues.csv', index = False)

#res_lsq = least_squares(minimize_distance_of_eigenvalues, x0, args=(a_big_ell,b_big_ell, xa_hyp, xi_hyp, alpha,n), bounds=([0.001,0.001,0.5], [a_big_ell, b_big_ell, a_big_ell]), x_scale = [0.000001,0.000001,0.000001])

#vi_first_eig, collocation_nodes_first_eig = create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,6,xa_hyp,xi_hyp,0,0,0,n)
#vi_second_eig = geo.create_ellipses_with_connections(0.32,a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,n[0])
vi_second_eig, collocation_nodes_second_eig = geo.create_6_ellipses(a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,n[0],alpha)
vi_second_eig = geo.transform_3d_array_in_object_array_with_2d_arrays(vi_second_eig)
collocation_nodes_second_eig = geo.transform_3d_array_in_object_array_with_2d_arrays(collocation_nodes_second_eig)
#vi_second_eig, collocation_nodes_second_eig = geo.create_ellipses_with_connections_and_ellipses_diff_n(eps,a_big_ell,b_big_ell,0,0,0,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,n_ellipses_single,n,alpha)
#collocation_nodes_second_eig = DL.create_collocation_nodes(vi_second_eig,alpha) 
second_eigenvalues, second_eigenvectors = calculate_second_eigenvalues(vi_second_eig, collocation_nodes_second_eig, alpha, n_masters, lines_per_task)
# first_eigenvalue = calculate_first_eigenvalue(vi_first_eig, collocation_nodes_first_eig, alpha)


time_end_eigenvalues = time.time()

time_eigenvalues = time_end_eigenvalues - time_start_eigenvalues


#eigenvalues = np.array([first_eigenvalue, second_eigenvalues[0]])
eigenvalues = second_eigenvalues
eigenvalues_data = pd.DataFrame({"Eigenvalues" : [eigenvalues]})

eigenvectors = second_eigenvectors
eigenvectors_data = pd.DataFrame(eigenvectors).add_prefix('Eigenvectors')

EigenvaluesEigenvectors = pd.concat([eigenvalues_data, eigenvectors_data], axis=1, ignore_index=False)


time_data = pd.DataFrame({"Processes" : [size], "TimeEigenvalues": [time_eigenvalues]})

if rank == 0:
    #eigenvalues_data.to_csv("Eigenvalues_Dahne_MPI.csv", index=False)
    #EigenvaluesEigenvectors.to_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv", index=False)
    time_data.to_csv("RuntimeEigenvalueSolutionTotal_Dahne_MPI.csv", index=False)

# print(f"Ended")

if rank == 0:
    try:
        EigenvaluesEigenvectors = pd.read_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv")
        print("Try ", flush = True)
    except:
        EigenvaluesEigenvectors = pd.DataFrame()
        print("Except ", flush = True)

#EigenvaluesEigenvectors = pd.read_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv")
#EigenvaluesEigenvectors = pd.DataFrame()

# Parameters of the square, in which all points inside the domain are searched
x_start = -2.01
x_end = 2.01
y_start = -2.01
y_end = 2.01
x_steps = 461
y_steps = 461
#time_start_solution = time.time()
xy_inside, sol = DL.calculate_double_layer_inside_with_holes_diff_n_master_workers(eigenvalues[0], vi_second_eig, collocation_nodes_second_eig, alpha, eigenvectors[:,0], x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task)
#time_end_solution = time.time()

#time_solution = time_end_solution - time_start_solution
#time_total = time_eigenvalues + time_solution



def set_solution_everywhere(xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps):
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    xy_inside_T = np.real(xy_inside.T)

    sol_everywhere = np.empty((mesh.shape[0],mesh.shape[1]),dtype=np.complex128)
    sol_everywhere.fill(np.nan)
    
    ave_n_rows_xy_inside, res_n_rows_xy_inside = divmod(xy_inside_T.shape[0], size)
    n_lines = np.concatenate((np.repeat(ave_n_rows_xy_inside+1, res_n_rows_xy_inside), np.repeat(ave_n_rows_xy_inside, size-res_n_rows_xy_inside)))
    # Caluculate cummulative number of lines the processes have to calculate starting from zero
    cumsum_lines = np.cumsum(n_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)

    eps = 0.000001
    mask_partial=np.abs(abs(mesh[:,:,None] - xy_inside_T[cumsum_lines[rank]:cumsum_lines[rank+1]])<eps).all(-1).any(-1)
    mask_global = np.empty((mesh.shape[0], mesh.shape[1]), dtype = bool)
    comm.Allreduce(mask_partial, mask_global, op=MPI.LOR)
    
    sol_everywhere[mask_global] = sol
    sol_everywhere.reshape((y_steps,x_steps))
    return sol_everywhere

def proof_nodal_line_touches_boundary(boundaries,xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps):
    sol_everywhere = set_solution_everywhere(xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps)
    sol_everywhere_real = np.real(sol_everywhere)
    pos_mask = sol_everywhere_real > 0
    neg_mask = sol_everywhere_real < 0
    equal_mask = sol_everywhere_real == 0
    
    pos_labeled, num_pos_domains = sc.ndimage.label(pos_mask)
    pos_labels, num_pos_labels = np.unique(pos_labeled, return_counts=True)
    # Position at which the biggest area of positiv labels is (at position 0 are the negative values)
    idx_max_pos_label = np.argmax(num_pos_labels[1:])
    max_pos_label = pos_labels[idx_max_pos_label + 1]
    #if rank == 0:
    #    print(f"num_pos_labels is {num_pos_labels}", flush = True)
    #    print(f"max_pos_label is {max_pos_label}", flush = True)

    # set all positiv values which are not part of the big positiv are to a negative value
    mask_shift_to_negative_values = (pos_labeled != 0) & (pos_labeled != max_pos_label)
    sol_everywhere_real[mask_shift_to_negative_values] = -1
    #mask_val_pos_labels = num_pos_labels>barrier
    #num_val_pos_labels = sum(mask_val_pos_labels[1:])
    
    neg_labeled, num_neg_domains = sc.ndimage.label(neg_mask)
    neg_labels, num_neg_labels = np.unique(neg_labeled, return_counts=True)
    # Position at which the biggest area of negativ labels is (at position 0 are the positive values)
    idx_max_neg_label = np.argmax(num_neg_labels[1:])
    max_neg_label = neg_labels[idx_max_neg_label + 1]
    #if rank == 0:
    #    print(f"num_neg_labels is {num_neg_labels}", flush = True)
    #    print(f"max_neg_label is {max_neg_label}", flush = True)
    # set all negativ values which are not part of the big positiv are to a positive value
    mask_shift_to_positive_values = (neg_labeled != 0) & (neg_labeled != max_neg_label)
    sol_everywhere_real[mask_shift_to_positive_values] = 1
    
    pos_mask = sol_everywhere_real > 0
    neg_mask = sol_everywhere_real < 0
    
    pos_labeled, num_pos_domains = sc.ndimage.label(pos_mask)
    pos_labels, num_pos_labels = np.unique(pos_labeled, return_counts=True)
    # Position at which the biggest area of positiv labels is (at position 0 are the negative values)
    idx_max_pos_label = np.argmax(num_pos_labels[1:])
    max_pos_label = pos_labels[idx_max_pos_label + 1]
    #if rank == 0:
    #    print(f"num_pos_labels is {num_pos_labels}", flush = True)
    #    print(f"max_pos_label is {max_pos_label}", flush = True)

    # set all positiv values which are not part of the big positiv are to a negative value
    mask_shift_to_negative_values = (pos_labeled != 0) & (pos_labeled != max_pos_label)
    sol_everywhere_real[mask_shift_to_negative_values] = -1

    neg_labeled, num_neg_domains = sc.ndimage.label(neg_mask)
    neg_labels, num_neg_labels = np.unique(neg_labeled, return_counts=True)
    # Position at which the biggest area of negativ labels is (at position 0 are the positive values)
    idx_max_neg_label = np.argmax(num_neg_labels[1:])
    max_neg_label = neg_labels[idx_max_neg_label + 1]
    #if rank == 0:
    #    print(f"num_neg_labels is {num_neg_labels}", flush = True)
    #    print(f"max_neg_label is {max_neg_label}", flush = True)
    # set all negativ values which are not part of the big positiv are to a positive value
    mask_shift_to_positive_values = (neg_labeled != 0) & (neg_labeled != max_neg_label)
    sol_everywhere_real[mask_shift_to_positive_values] = 1
    
    equal_labeled, num_equal_domains = sc.ndimage.label(equal_mask)
    equal_labels, num_equal_labels = np.unique(equal_labeled, return_counts=True)
    #if rank == 0:
    #    print(f"equal_labels is {equal_labels} num_equal_labels is {num_equal_labels}", flush = True)


    pos_mask = sol_everywhere_real > 0
    neg_mask = sol_everywhere_real < 0
    equal_mask = sol_everywhere_real == 0
    
    pos_labeled, num_pos_domains = sc.ndimage.label(pos_mask)
    pos_labels, num_pos_labels = np.unique(pos_labeled, return_counts=True)
    #if rank == 0:
    #    print(f"num_pos_labels is {num_pos_labels}", flush = True)

    neg_labeled, num_neg_domains = sc.ndimage.label(neg_mask)
    neg_labels, num_neg_labels = np.unique(neg_labeled, return_counts=True)
    #if rank == 0:
    #    print(f"num_neg_labels is {num_neg_labels}", flush = True)


    for sub_label in range(1, num_equal_domains + 1):
        region = (equal_labeled == sub_label)
        # Extend region by one pixel
        dilated = sc.ndimage.binary_dilation(region)

        # Exterior region
        exterior_region = dilated & (~region)

        neighbor_values = sol_everywhere_real[exterior_region]

        pos_exterior = np.any(neighbor_values > 0)
        neg_exterior = np.any(neighbor_values < 0)
        
        if rank == 0:
            print(f"pos_exterior is {pos_exterior}",flush = True)
            print(f"neg_exterior is {neg_exterior}",flush = True)

        if pos_exterior and ~neg_exterior:
            sol_everywhere_real[region] = 1
            if rank == 0:
                print(f"pos_exterior is true neg_exterior is false",flush = True)
        if ~pos_exterior and neg_exterior:
            sol_everywhere_real[region] = -1
            if rank == 0:
                print(f"pos_exterior is true neg_exterior is false",flush = True)

    #mask_val_neg_labels = num_neg_labels>barrier
    #num_val_neg_labels = sum(mask_val_neg_labels[1:])
    
    x = np.linspace(x_start, x_end, x_steps)
    y = np.linspace(y_start, y_end, y_steps)
    X, Y = np.meshgrid(x,y)

    cs = plt.contour(X, Y, sol_everywhere_real, levels=[0])
    #nodal_line = [p.vertices for col in cs.collections for p in col.get_paths()]
    nodal_line = cs.allsegs[0]

    #boundary_paths = [path.Path(boundary[i].T, closed=True) for i in range(boundary.shape[0])]
    boundary_trees = []

    for boundary in boundaries:
        boundary_trees.append(cKDTree(boundary.T))

    if rank == 0:
        print(f"nodal_line is {nodal_line}", flush=True)
    tol = 2 * min(x[1]-x[0], y[1]-y[0])
    touches_boundary = False
    for nodal in nodal_line:
        for tree in boundary_trees:
            dists, _ = tree.query(nodal, k=1)
            if rank == 0:
                print(f"tol is {tol}", flush = True)
                print(f"np.min(dists) is {np.min(dists)}", flush=True)
            if np.any(dists < tol):
                touches_boundary = True
                break
        if touches_boundary:
            break
            #touches_boundary_temp = boundary_path.distance_to_path(path.Path(nodal)) < tol
            #touches_boundary_temp = boundary_path.distance_to_path(path.Path(nodal)) < tol
            
    if rank == 0:
        print(f"touches_boundary is {touches_boundary}", flush=True)
    #if num_val_pos_labels != 0 and num_val_neg_labels != 0:
    #    num_nodal_domains = num_val_pos_labels + num_val_neg_labels
    #elif num_val_pos_labels == 0 and num_val_neg_labels == 0:
    #    num_nodal_domains = 0
    #else:
    #    num_nodal_domains = 1
    #if rank == 0:
    #    print(f"num_val_pos_labels is {num_val_pos_labels}; num_val_neg_labels is {num_val_neg_labels}; num_nodal_domains is {num_nodal_domains}")
    #num_nodal_domains = num_val_pos_labels + num_val_neg_labels
    return nodal_line


nodal_line = proof_nodal_line_touches_boundary(vi_second_eig, xy_inside, sol, x_start,x_end,x_steps, y_start,y_end,y_steps)


eigenvalues_data = pd.DataFrame({"Eigenvalues_higher_precision" : eigenvalues})
eigenvectors_data = pd.DataFrame(eigenvectors).add_prefix('Eigenvectors_higher_precision')
inside_data = pd. DataFrame(xy_inside.T).add_prefix('Inside_higher_precision')
sol_data = pd.DataFrame(sol).add_prefix('Solution_higher_precision')
#EigenvaluesEigenvectors = pd.concat([eigenvalues_data, eigenvectors_data, inside_data, sol_data], axis=1, ignore_index=False)

nodal_line_data = pd.DataFrame(nodal_line[0]).add_prefix('nodal_line')
for i in range(1,len(nodal_line)):
    nodal_line_data_temp = pd.DataFrame(nodal_line[i]).add_prefix('nodal_line')
    nodal_line_data = pd.concat([nodal_line_data, nodal_line_data_temp], axis=1, ignore_index=False)


new_EigenvaluesEigenvectors = pd.concat([eigenvalues_data, eigenvectors_data, inside_data, sol_data, nodal_line_data], axis=1, ignore_index=False)
#EigenvaluesEigenvectors = pd.concat([EigenvaluesEigenvectors, new_EigenvaluesEigenvectors], axis = 1)

#time_data = pd.DataFrame({"Processes" : [size], "TimeEigenvalues": [time_eigenvalues], "TimeSolution" : [time_solution], "TimeTotal" : [time_total]})

if rank == 0:
    EigenvaluesEigenvectors = pd.concat([EigenvaluesEigenvectors, new_EigenvaluesEigenvectors], axis = 1)
    EigenvaluesEigenvectors.to_csv("EigenvaluesEigenvectorsInsideSol_Dahne_MPI.csv", index=False)
#    time_data.to_csv("RuntimeEigenvalueSolutionTotal_Dahne_MPI.csv", index=False)


