#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy
import scipy.special as sp
from scipy import integrate
from matplotlib import path
from mpi4py import MPI
import pandas as pd
import time
from itertools import chain
import sys
import os
#from scipy.integrate import quad_vec as quad_vec_origin


def create_polygon(n_gon,R,xm,ym,rot,n):
    """
    Routine to create the nodes (vertices and midpoints) of a regular polygon.

    Parameters
    ----------
    n_gon : int
        Number of vertices of the polygon.
    R : float
        Outer radius of the polygon.
    xm : float
        x-element of the center of the polygon.
    ym : float
        y-element of the center of the polygon.
    rot : float
        Rotation of the polyon in counterclockwise direction (rad).
    n : int
        Number of nodes to create.

    Returns
    -------
    f : numpy.ndarray
        Numpy array with x-axis elements of the nodes of a polygon in the first row and y-axis elements in the second row.
        Start and end points are NOT the same.
    """
    # Use the same amout of nodes per side
    n_ave, n_res = divmod(n,n_gon)
    # The vertices of the polygon all lie on a circle with radius R and rotation rot.
    vi_circle = create_ellipse(R,R,xm,ym,rot,n_gon)
    # Add the first column of vi_cirle to the end. 
    vi_circle_append = np.concatenate((vi_circle, vi_circle[:,0].reshape(2,1)), axis=1)
    # Create a straight line between two consecutive vertices without the endpoint. 
    ti = np.linspace(vi_circle_append[:,:-1],vi_circle_append[:,1:], n_ave, endpoint=False)
    # Transformation in the correct form.
    vi = np.concatenate(np.swapaxes(ti,0,2), axis=1)
    return vi

def create_rectangle(a,b,xm,ym,n):
    """
    Routine to create the nodes (vertices and midpoints) of a rectangel.

    Parameters
    ----------
    a : float
        Width of the rectangle.
    b : float
        Height of the rectangle.
    xm : float
        x-element of the center of the rectangle.
    ym : float
        y-element of the center of the rectangle.
    n : int
        Number of nodes to create.

    Returns
    -------
    f : numpy.ndarray
        Numpy array with x-axis elements of the nodes of a rectangle in the first row and y-axis elements in the second row.
        Start and end points are NOT the same.
    """
    n_ave, n_res = divmod(n,4)
    # If the rest value is 1 or 3, it is reduced to 0 or 2. The result is the same number of points in each direction.
    n_res = n_res - np.mod(n_res,2)
    
    if n_res == 2:
        if a >= b:
            nx = n_ave + 1
            ny = n_ave
        else:
            nx = n_ave
            ny = n_ave + 1
    else:
        nx = n_ave
        ny = n_ave
    ti_x = np.linspace(xm-a/2,xm+a/2,nx+1, endpoint=True)
    ti_y = np.linspace(ym-b/2,ym+b/2,ny+1, endpoint=True)
    x = np.concatenate([ti_x[:-1],(xm+a/2)*np.ones(ny), np.flip(ti_x)[:-1], (xm-a/2)*np.ones(ny)])
    y = np.concatenate([(ym-b/2)*np.ones(nx),ti_y[:-1],(ym+b/2)*np.ones(nx), np.flip(ti_y)[:-1]])
    vi = np.array([x, y])
    return vi

def create_ellipse(a,b,xm,ym,rot,n):
    """
    Routine to create the nodes (vertices and midpoints) of an ellipse.

    Parameters
    ----------
    a : float
        Width of the ellipse.
    b : float
        Height of the ellipse.
    xm : float
        x-element of the center of the ellipse.
    ym : float
        y-element of the center of the ellipse.
    rot : float
        Rotation of the polyon in counterclockwise direction (rad).
    n : int
        Number of nodes to create.

    Returns
    -------
    f : numpy.ndarray
        Numpy array with x-axis elements of the nodes of an ellipse in the first row and y-axis elements in the second row.
        Start and end points are NOT the same.
    """
    ti = np.linspace(0,2*np.pi,n, endpoint=False)
    vi = np.array([xm+a*np.cos(ti)* np.cos(rot)-b*np.sin(ti)*np.sin(rot), ym+a*np.cos(ti)* np.sin(rot)+b*np.sin(ti)*np.cos(rot)])
    return vi




def create_hypotrochoid(R,r,a,xm,ym,rot,n):
    """
    Routine to create the nodes (vertices and midpoints) of a hypotrochoid.
    Start and end points are NOT the same.

    Parameters
    ----------
    R : float
        Radius of the fixed circle.
    r : float
        Radius of the rolling circle, which rolls inside of the bigger circle
    a : float
        Normalized distance (by the radius of the revolving circle) from the center of the revolving circle to the drawing point.
    xm : float
        x-element of the center of the ellipse.
    ym : float
        y-element of the center of the ellipse.
    rot : float
        Rotation of the polyon in counterclockwise direction (rad).
    n : int
        Number of nodes to create.

    Returns
    -------
    f : numpy.ndarray
        Numpy array with x-axis elements of the nodes of a hypotrochoid in the first row and y-axis elements in the second row.
    """
    ti = np.linspace(0,2*np.pi,n, endpoint=False)
    #vi = np.array([xm+(R-r)*np.cos(ti)+a*r*np.cos((R/r-1)*ti), ym+(R-r)*np.sin(ti)-a*r*np.sin((R/r-1)*ti)])
    x = (R-r)*np.cos(ti)+a*r*np.cos((R/r-1)*ti)
    y = (R-r)*np.sin(ti)-a*r*np.sin((R/r-1)*ti)
    vi = np.array([xm+x* np.cos(rot)-y*np.sin(rot), ym+x* np.sin(rot)+y*np.cos(rot)])
    return vi




def calculate_reflection(vi):
    """
    Routine to calculate the coordinates of the vertices vi in the reverse direction.

    Parameters
    ----------
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary

    Returns
    -------
    numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary in the reverse direction
    """
    return np.flip(vi, 1)




def create_circle_parameterization(midpoint, R, t):
    """
    Routine to create the parameterization of a circle.

    Parameters
    ----------
    midpoint : float
        Midpoint of the circle
    R : float
        Radius of the circle.
    t : numpy.ndarray
        Array of elements between 0 and 2*np.pi for which the position on the circle is to be calculated

    Returns
    -------
    numpy.ndarray
        Numpy array containing positions on the circle with x-axis elements being the real values, 
        y-axis elements being the imaginary values.
    """
    return midpoint + R * np.exp(1j*t)




def create_circle_parameterization_derivative(R, t):
    """
    Routine to create the derivative of a circle parameterization.

    Parameters
    ----------
    R : float
        Radius of the circle.
    t : numpy.ndarray
        Array of elements between 0 and 2*np.pi for which position the derivative of the circle is to be calculated

    Returns
    -------
    numpy.ndarray
        Numpy array containing the derivative of the circle at the positions t.
        The x-axis elements are the real values, y-axis elements are the imaginary values.
    """
    return 1j* R * np.exp(1j*t)




def first_lagrange_basis (s):
    """
    Routine to compute the first Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the first Lagrange function must be calculated.

    Returns
    -------
    l1 : float
        Value of the first Lagrange function at the point s
    """
    u = 1-s
    l1 = u*(1-2*s)
    return l1




def second_lagrange_basis (s):
    """
    Routine to compute the second Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the second Lagrange function must be calculated.

    Returns
    -------
    l2 : float
        Value of the second Lagrange function at the point s
    """
    u = 1-s
    l2 = 4*s*u
    return l2




def third_lagrange_basis (s):
    """
    Routine to compute the third Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the third Lagrange function must be calculated.

    Returns
    -------
    l3 : float
        Value of the third Lagrange function at the point s
    """
    l3 = s*(2*s-1)
    return l3




def create_collocation_nodes(vi, alpha):
    r"""
    Routine to create the collocation nodes along the entire approximated boundary.
    The density function must be calculated at these points.
    For each face of the boundary, three collocation nodes are computed at the points
    $q_1 = \alpha$, $q_2 = 1/2$ and $q_3 = 1-\alpha$, where $0 < \alpha < 1/2$.

    Parameters
    ----------
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row and y-axis elements in the second row.
    """
    # Add the first column to the end of vi
    vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
    # Create an array with the shape 2x3*surfaces
    collocation_nodes = np.zeros((np.shape(vi)[0],int(np.shape(vi)[1]*3/2)))
    # Calculate the first collocation node of each part of the boundary line
    collocation_nodes[:,::3] = first_lagrange_basis(alpha)*vi[:,:-1:2] + second_lagrange_basis(alpha)*vi[:,1::2] + third_lagrange_basis(alpha)*vi_append[:,2::2]
    # Calculate the second collocation node of each part of the boundary line
    collocation_nodes[:,1::3] = first_lagrange_basis(0.5)*vi[:,:-1:2] + second_lagrange_basis(0.5)*vi[:,1::2] + third_lagrange_basis(0.5)*vi_append[:,2::2]
    # Calculate the third collocation node of each part of the boundary line
    collocation_nodes[:,2::3] = first_lagrange_basis(1-alpha)*vi[:,:-1:2] + second_lagrange_basis(1-alpha)*vi[:,1::2] + third_lagrange_basis(1-alpha)*vi_append[:,2::2]
    return collocation_nodes




def first_generalized_lagrange_basis (s, alpha):
    """
    Routine to compute the first generalized Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the first generalized Lagrange function must be calculated.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    gl1 : float
        Value of the first generalized Lagrange function at the point s
    """
    #s is single value
    u = 1-s
    gl1 = (u-alpha)/(1-2*alpha)*(1-2*s)/(1-2*alpha)
    return gl1




def second_generalized_lagrange_basis (s, alpha):
    """
    Routine to compute the second generalized Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the second generalized Lagrange function must be calculated.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    gl2 : float
        Value of the second generalized Lagrange function at the point s
    """
    #s is single value
    u = 1-s
    gl2 = 4*(s-alpha)/(1-2*alpha)*(u-alpha)/(1-2*alpha)
    return gl2




def third_generalized_lagrange_basis (s, alpha):
    """
    Routine to compute the third generalized Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the third generalized Lagrange function must be calculated.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    gl3 : float
        Value of the third generalized Lagrange function at the point s
    """
    #s is single value
    gl3 = (s-alpha)/(1-2*alpha)*(2*s-1)/(1-2*alpha)
    return gl3




def first_lagrange_basis_derivative (s):
    """
    Routine to calculate the derivative of the first Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the derivative of the first Lagrange function must be calculated.

    Returns
    -------
    dl1 : float
        Value of the derivative of the first Lagrange function at the point s
    """
    dl1 = 4*s-3
    return dl1




def second_lagrange_basis_derivative (s):
    """
    Routine to calculate the derivative of the second Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the derivative of the second Lagrange function must be calculated.

    Returns
    -------
    dl2 : float
        Value of the derivative of the second Lagrange function at the point s
    """
    dl2 = 4*(1-2*s)
    return dl2




def third_lagrange_basis_derivative (s):
    """
    Routine to calculate the derivative of the third Lagrange function at a given point s.

    Parameters
    ----------
    s : float
        Value at which the derivative of the third Lagrange function must be calculated.

    Returns
    -------
    dl3 : float
        Value of the derivative of the third Lagrange function at the point s
    """
    dl3 = 4*s-1
    return dl3




def create_external_normals (vi, s):
    """
    Routine to create the external normals at the boundary without the Jacobian.
    Each piece of the boundary is parameterized from 0 to 1.
    The value s indicates the place at which point the external normal for each piece of the boundary is to be calculated.

    Parameters
    ----------
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    s : float between 0 and 1
        Value at which the external normal must be calculated in each face of the boundary .

    Returns
    -------
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the external normals in the first row and y-axis elements in the second row.
    """
    
    # Add the first column to the end of vi
    vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
    #Calculates the directional derivative at the boundary at the points s
    external_normal = first_lagrange_basis_derivative(s)*vi[:,:-1:2] + second_lagrange_basis_derivative(s)*vi[:,1::2] + third_lagrange_basis_derivative(s)*vi_append[:,2::2]
    # Swap the first two rows
    external_normal[[0,1],:] = external_normal[[1,0],:]
    external_normal[1,:] = -external_normal[1,:]
    return external_normal



def create_integrand_double_layer (s, wavenumber, vi, collocation_nodes, alpha, check_singularity=False): 
    """
    Routine to create a matrix consisting of the integrands of the matrix of the double layer operator.
    The Jacobian is not used, because it cancels out with the Jacobian of the external normals.

    Parameters
    ----------
    s : float between 0 and 1
        Value at which the external normal must be calculated in each face of the boundary.
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    check_singularity : bool, default 'False'
        Boolean indicating whether to perform a check for singularities that can not be solved using ordinary integration.
        This occurs if the distances between the vertices and midpoints differ too much.

    Returns
    -------
    integrand : numpy.ndarray
        Numpy array containing all integrands of the double layer operator.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    eps =1e-05
    # Calculation of the exterior normal without the Jacobian
    external_normal = create_external_normals(vi[:,:-1], s)
    # Calculates the approximation of the boundary at point s \in [0,1] for each piece at the boundary
    approx_boundary = first_lagrange_basis(s)*vi[:,:-2:2] + second_lagrange_basis(s)*vi[:,1:-1:2] + third_lagrange_basis(s)*vi[:,2::2]
    # Computes a matrix that subtracts each x element of the approximate boundary from each x element of the collocation nodes 
    a = collocation_nodes[0][:,None] - approx_boundary[0][None,:]
    # Computes a matrix that subtracts each y element of the approximate boundary from each y element of the collocation nodes 
    b = collocation_nodes[1][:,None] - approx_boundary[1][None,:]
    r = np.sqrt(a*a+b*b) # multiplies elementwise and takes squareroot elementwise
    # Duplicate each column three times
    external_normal_big = np.repeat(external_normal, 3, axis=1)
    a_big = np.repeat(a, 3, axis=1)
    b_big = np.repeat(b, 3, axis=1)
    r_big = np.repeat(r, 3, axis=1)
    
    generalized_lagrange_functions = np.array((first_generalized_lagrange_basis(s,alpha),second_generalized_lagrange_basis(s,alpha),third_generalized_lagrange_basis(s,alpha)))
    # Generate x and y indices of a 3*facesx3*faces matrix
    x_indices, y_indices = np.ogrid[:int((np.shape(vi)[1]-1)*3/2), :int((np.shape(vi)[1]-1)*3/2)]
    
    # Without a check of singularity which can occur if the vertices and midpoints differ too much
    if check_singularity == False:
        # Calculate integrands without the main diagonal
        integrand1 = np.where((x_indices != y_indices) & ((y_indices % 3) == 0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[0], 0)
        integrand2 = np.where((x_indices != y_indices) & ((y_indices % 3) == 1), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[1], 0)
        integrand3 = np.where((x_indices != y_indices) & ((y_indices % 3) == 2), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[2], 0)
    
    # With a check of singularity which can occur if the vertices and midpoints differ too much
    else:
        mask_r0 = (r_big == 0)
        if np.any(mask_r0):
            print(f"Rank {rank} mask_r0 True", flush = True)
            #comm.Abort(1) # stops everything
            mask_s = (s <= alpha - eps) | ((s >= alpha + eps) & (s <= 1 - alpha - eps)) | (s >= 1 - alpha + eps)
            if mask_s:
                # Calculate integrands for the first collocation node of each boundary piece
                integrand1 = np.where(((y_indices % 3) == 0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[0], 0)
                # Calculate integrands for the second collocation node of each boundary piece
                integrand2 = np.where(((y_indices % 3) == 1), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[1], 0)
                # Calculate integrands for the third collocation node of each boundary piece
                integrand3 = np.where(((y_indices % 3) == 2), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[2], 0)
                integrand = integrand1 + integrand2 + integrand3
            else:
                # Calculate integrands for the first collocation node of each boundary piece
                integrand1 = np.where(((y_indices % 3) == 0) & (~mask_r0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[0], 0)
                # Calculate integrands for the second collocation node of each boundary piece
                integrand2 = np.where(((y_indices % 3) == 1) & (~mask_r0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[1], 0)
                # Calculate integrands for the third collocation node of each boundary piece
                integrand3 = np.where(((y_indices % 3) == 2) & (~mask_r0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[2], 0)
                integrand = integrand1 + integrand2 + integrand3
        else:
            # Calculate integrands for the first collocation node of each boundary piece
            integrand1 = np.where(((y_indices % 3) == 0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[0], 0)
            # Calculate integrands for the second collocation node of each boundary piece
            integrand2 = np.where(((y_indices % 3) == 1), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[1], 0)
            # Calculate integrands for the third collocation node of each boundary piece
            integrand3 = np.where(((y_indices % 3) == 2), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[2], 0)
            integrand = integrand1 + integrand2 + integrand3
    return integrand

def chunk_Bcast_array(array, comm, chunk_bytes=3000000):
    # How many elements can be Bcasted per iteration
    flat_array = array.ravel()
    chunk_elements = chunk_bytes // flat_array.itemsize
    if comm.rank == 0:
        print(f"chunk_elements is {chunk_elements}", flush = True)
        print(f"flat_array.size is {flat_array.size}", flush = True)
    for start_chunk in range(0,flat_array.size, chunk_elements):
        end_chunk = min(start_chunk+chunk_elements, flat_array.size)
        buf = np.ascontiguousarray(flat_array[start_chunk:end_chunk])
        if comm.rank == 0:
            print(f"Bis hier kein Problem", flush = True)
        comm.Bcast(buf, root=0)
        if comm.rank == 0:
            print(f"Bis hier auch noch kein Problem", flush = True)
        flat_array[start_chunk:end_chunk] = buf
        if comm.rank == 0:
            print(f"start chunk is {start_chunk}", flush = True)
            print(f"end chunk is {end_chunk}", flush = True)

def create_D_matrix (wavenumber, vi, collocation_nodes, alpha):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the i-th collocation node
    (i-th row) and at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for simply connected domains.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix : numpy.ndarray
        Numpy array containing all integrals of the double layer operator.
    """
    # Adaptive integration of the matrix from 0 to 1 without the points [alpha, 0.5, 1-alpha] due to singularities.
    D_matrix = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi, collocation_nodes, alpha), points = [alpha, 0.5, 1-alpha])
    return D_matrix




def create_D_matrix_with_holes (wavenumber, vi_holes, collocation_nodes_holes, alpha):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the i-th collocation node
    (i-th row) and at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    
    number_holes = vi_holes.shape[0]
    # Number of collocation-nodes at each boundary
    n_coll = collocation_nodes_holes.shape[2] 
    D_matrix_holes = np.zeros((number_holes*n_coll, number_holes*n_coll), dtype='complex')
    # Calculate the D-matrices with x on the k-th boundary, integrate over the i-th boundary. Write it in a bigger D-matrix.
    for k in range(number_holes):
        for i in range(number_holes):
            # Adaptive integration of the matrix from 0 to 1 without the points [alpha, 0.5, 1-alpha] due to singularities.
            D_matrix_holes[k*n_coll:(k+1)*n_coll,i*n_coll:(i+1)*n_coll]  = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], collocation_nodes_holes[k], alpha), points = [alpha, 0.5, 1-alpha])[0]
    return D_matrix_holes


def create_M_matrix (wavenumber, vi, collocation_nodes, alpha): 
    """
    Routine to create the M-matrix for a domain WITHOUT holes. The M-matrix is the D-matrix (boundary double layer)
    with the jump conditions, i.e. with a subtraction of an identity matrix with the same shape as the D-matrix.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    M_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator with jump conditions.
    """
    D_matrix = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi, collocation_nodes, alpha), points = [alpha, 0.5, 1-alpha])
    # D_matrix gets the D-matrix and eps
    M_matrix = D_matrix[0] - np.identity(D_matrix[0].shape[0])/2
    return M_matrix




def create_M_matrix_with_holes (wavenumber, vi_holes, collocation_nodes_holes, alpha): 
    """
    Routine to create the M-matrix for a domain WITH holes. The M-matrix is the D-matrix (boundary double layer)
    with the jump conditions, i.e. with a subtraction of an identity matrix with the same shape as the D-matrix.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    M_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator with jump conditions for q holes.
    """
    D_matrix_holes = create_D_matrix_with_holes (wavenumber, vi_holes, collocation_nodes_holes, alpha)
    M_matrix_holes = D_matrix_holes - np.identity(D_matrix_holes[0].size)/2 
    return M_matrix_holes




def calculate_rhs(wavenumber, collocation_nodes):
    """
    Routine to compute the right hand side of the linear equation system to calculate the density at the collocation points.
    The right hand side is the fundamental solution of the Helmholtz equation in two dimensions.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row and y-axis elements in the second row.

    Returns
    -------
    rhs : numpy.ndarray
        Numpy array containing the fundamental solution of the Helmholts equation at the collocation nodes.
    """
    r = np.sqrt(collocation_nodes[0]*collocation_nodes[0]+collocation_nodes[1]*collocation_nodes[1])
    rhs = 1j * sp.hankel1(0,wavenumber*r)/4
    return rhs




def calculate_density_exterior_problem(wavenumber, alpha, D_matrix, collocation_nodes):
    """
    Routine to calculate the density function at the given collocation nodes for the exterior Dirichlet problem,
    with the Dirichlet condition defined by the fundamental solution of the Helmholtz equation in two dimensions.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    D_matrix : numpy.ndarray
       Numpy array containing all integrals of the double layer operator.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row and y-axis elements in the second row.
    

    Returns
    -------
    density : numpy.ndarray
        Numpy array containing the values of the density function at the collocation nodes.
    """
    rhs = calculate_rhs(wavenumber, collocation_nodes)
    lhs = D_matrix[0] + np.identity(D_matrix[0].shape[0])/2 
    density = np.linalg.solve(lhs, rhs)
    return density




def calculate_density_interior_problem(wavenumber, alpha, D_matrix, collocation_nodes):
    """
    Routine to calculate the density function at the given collocation nodes for the interior Dirichlet problem,
    with the Dirichlet condition defined by the fundamental solution of the Helmholtz equation in two dimensions.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    D_matrix : numpy.ndarray
       Numpy array containing all integrals of the double layer operator.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row and y-axis elements in the second row.
    

    Returns
    -------
    density : numpy.ndarray
        Numpy array containing the values of the density function at the collocation nodes.
    """
    rhs = calculate_rhs(wavenumber, collocation_nodes)
    lhs = D_matrix[0] - np.identity(D_matrix[0].shape[0])/2 
    density = np.linalg.solve(lhs, rhs)
    return density




def beyn_algorithm (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps): 
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain without holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    # Create random matrix V_hat with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    if M_matrix == False:
        # Add the first column to the end of vi
        vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
        # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
        M_matrices = np.array([create_M_matrix (phi_r, vi_append, collocation_nodes, alpha) for phi_r in phi])
    else:
        M_matrices = np.array([M_matrix(phi_r) for phi_r in phi])
    # Inverse of the M-matrices
    M_matrices_inv = np.linalg.inv(M_matrices)
    # Prestep to calculate the summands for the tepezoidal rule
    A_0_summand_temp = np.dot(M_matrices_inv,V_hat)
    # Summands for the approximated A0
    A_0_summand = A_0_summand_temp*phi_derivative[:,np.newaxis, np.newaxis]
    # Summands for the approximated A1
    A_1_summand = phi[:,np.newaxis, np.newaxis] * A_0_summand
    # Approximated A0 and A1
    A_0 = np.sum(A_0_summand, axis=0)/(1j*N)
    A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
    # Calculate singular value decomposition of A0
    V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
    # Rank test
    Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
    rank = np.count_nonzero(Sigma0_eps)
    # Truncated singular value decomposition 
    Sigma_rank = Sigma0_eps[:rank]
    V_rank = V0[:,:rank]
    WH_rank = W0H[:rank,:]
    # Calculate matrix B
    B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
    # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
    eigenvalues, eigenvectors_B = np.linalg.eig(B)
    # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
    eigenvectors = np.dot(V_rank,eigenvectors_B)
    return eigenvalues, eigenvectors




def beyn_algorithm_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    # Number of collocation nodes per boundary
    n_coll = collocation_nodes.shape[1]
    # Add the first column of collocation nodes to the end.
    vi = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
        
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    comm.Bcast(V_hat, root = 0)
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    if M_matrix == False:
        # How many rows must be calculated by a process?
        ave_n_rows_D, res_n_rows_D = divmod(N*n_coll, size)
        # Calculate number of lines that each process has to calculate
        n_lines = np.concatenate((np.repeat(ave_n_rows_D+1, res_n_rows_D), np.repeat(ave_n_rows_D, size-res_n_rows_D)))
        # Caluculate cummulative number of lines the processes have to calculate starting from zero
        cumsum_lines = np.cumsum(n_lines)
        cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
        #For phi
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_matrix, start_line = divmod(cumsum_lines[:-1], n_coll)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_coll)
        #For hole
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_hole_matrix, start_hole_line = divmod(cumsum_lines[:-1], n_coll)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_hole_matrix, end_hole_line = divmod(cumsum_lines[1:]-1, n_coll)
    
        # Submatrix and row in the submatrix where the calculation for all the processes starts and ends.
        start_row_matrix, start_column_matrix = divmod(np.mod(start_hole_matrix, 1),1)
        end_row_matrix, end_column_matrix = divmod(np.mod(end_hole_matrix, 1),1)
        
        # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
        start_matrix_partial = start_matrix[rank]
        start_line_partial = start_hole_line[rank]
        end_matrix_partial = end_matrix[rank]
        end_line_partial = end_hole_line[rank]
        
        # Compute the submatrices of D on each process.
        # If only one submatrix needs to be calculated in the process
        if start_matrix_partial == end_matrix_partial:
            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi, collocation_nodes[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
        #If the rows to be calculated extend across multiple sub-matrices
        else: 
            for i in range(start_matrix_partial, end_matrix_partial+1):
                # Calculation of the first submatrix
                if i == start_matrix_partial:
                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                #Calculation of the last submatrices.
                elif i == end_matrix_partial:
                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                #Calculation of the submatrices between the first and last.
                else:
                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    
    
        # Buffer to store the entire D-matrix 
        D_matrix_holes = np.empty((N, n_coll, n_coll),dtype=np.complex128)
        # Number of elements per D-matrices in each process.
        sendcounts_matrices = n_lines*n_coll
        # Cumulative number of elements of the D matrices calculated up to the current process.
        displacements_matrices = cumsum_lines[:-1] * n_coll
        # Collect all parts of the D-Matrix and store them in a large
        comm.Gatherv(D_matrix_holes_partial, [D_matrix_holes, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
        # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
        D_matrix_holes_conc = np.concatenate(D_matrix_holes, axis = 1)
        M_matrix_holes = D_matrix_holes_conc - np.tile(np.identity(D_matrix_holes_conc[:,0].size),N)/2 
        M_matrix_holes_3d = np.reshape(M_matrix_holes,(n_coll,N,n_coll)).transpose(1,0,2)
    else:
        M_matrix_holes_3d = np.array([M_matrix(phi_r) for phi_r in phi])
    
    color_split = 0 if (rank < N and rank < size) else MPI.UNDEFINED
    sub_comm = comm.Split(color=color_split, key=rank)
    if sub_comm != MPI.COMM_NULL:
        if rank == 0:
            time_start1 = time.time()
        if M_matrix == False:
            M_matrix_holes_3d = np.ascontiguousarray(M_matrix_holes_3d)
            sub_comm.Bcast(M_matrix_holes_3d,root=0)
        sub_size = sub_comm.Get_size()
        sub_rank = sub_comm.Get_rank()
        # How many 2d matrices must be calculated by a process?
        ave_matrices, res_matrices = divmod(N, sub_size)
        # Calculate number of lines that each process has to calculate
        n_matrices = np.concatenate((np.repeat(ave_matrices+1, res_matrices), np.repeat(ave_matrices, sub_size-res_matrices)))
        cumsum_matrices = np.cumsum(n_matrices)
        cumsum_matrices = np.insert(cumsum_matrices,0,0).astype(np.int32)
        # Calculate the rhs of the LES Mx = V_hat phi_derivative to solve x, which is nessecary for x = M_inv V_hat phi_derivative
        rhs_solve = V_hat*phi_derivative[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis]
        # Summands for the approximated A0
        A_0_summand_partial = np.linalg.solve(M_matrix_holes_3d[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],:,:], rhs_solve)
        A_0_summand = np.empty((N,m,l), dtype = np.complex128)
        # Number of submatrices in each process.
        sendcounts_matrices = n_matrices*m*l
        # Cumulative number of submatrices calculated up to the current process.
        displacements_matrices = cumsum_matrices[:-1]*m*l
        sub_comm.Gatherv(A_0_summand_partial, [A_0_summand, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
        sub_comm.Free()
    if rank == 0:
        # Summands for the approximated A1
        A_1_summand = phi[:,np.newaxis, np.newaxis] * A_0_summand
        # Approximated A0 and A1
        A_0 = np.sum(A_0_summand, axis=0)/(1j*N)
        A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
        # Calculate singular value decomposition of A0
        V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
        #Rank test
        Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
        rank_test = np.count_nonzero(Sigma0_eps)
        # Truncated singular value decomposition 
        Sigma_rank = Sigma0_eps[:rank_test]
        V_rank = V0[:,:rank_test]
        WH_rank = W0H[:rank_test,:]
        # Calculate matrix B
        B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
        # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
        eigenvalues, eigenvectors_B = np.linalg.eig(B)
        eigenvalues_shape = np.array(eigenvalues.shape, dtype = np.int32)
        # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
        eigenvectors = np.dot(V_rank,eigenvectors_B)
        eigenvectors_shape = np.array(eigenvectors.shape, dtype = np.int32)
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
    comm.Bcast(eigenvalues_shape, root = 0)
    comm.Bcast(eigenvectors_shape, root = 0)
    if rank != 0:
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    return eigenvalues, eigenvectors



def calculate_D_matrix_without_holes_partial(vi, collocation_nodes, phi, alpha, start_matrix_partial, start_line_partial, end_matrix_partial, end_line_partial):
    if start_matrix_partial == end_matrix_partial:
        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi, collocation_nodes[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            # Calculation of the first submatrix
            if i == start_matrix_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
            #Calculation of the last submatrices.
            elif i == end_matrix_partial:
                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            #Calculation of the submatrices between the first and last.
            else:
                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
    return D_matrix_holes_partial

def beyn_algorithm_MPI_master_workers (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    # Number of collocation nodes per boundary
    n_coll = collocation_nodes.shape[1]

    # Number of sublines that have to be calculated
    n_lines = m*N

    # How many lines have to be calculated per task
    n_lines_per_task = min(lines_per_task, int(n_lines/size))
    # How many tasks with with the complete number of lines exist
    n_repeat_full_tasks = int(n_lines/n_lines_per_task)
    # How many lines are left over
    n_tasks_rest = n_lines - n_repeat_full_tasks * n_lines_per_task
    # Number of lines per task for all tasks in one array
    all_tasks_lines = np.repeat(n_lines_per_task, n_repeat_full_tasks)
    if n_tasks_rest > 0:
        all_tasks_lines = np.append(all_tasks_lines, n_tasks_rest)

    # Add the first column of collocation nodes to the end.
    vi = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
        
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    comm.Bcast(V_hat, root = 0)
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    if M_matrix == False:
        tag_start = 0
        tag_done = 1
        tag_exit = 2
        # Caluculate cummulative number of lines the processes have to calculate starting from zero
        cumsum_lines = np.cumsum(all_tasks_lines)
        cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
        #For phi
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_matrix, start_line = divmod(cumsum_lines[:-1], n_coll)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_coll)
        #For hole
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_hole_matrix, start_hole_line = divmod(cumsum_lines[:-1], n_coll)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_hole_matrix, end_hole_line = divmod(cumsum_lines[1:]-1, n_coll)
    
        # Submatrix and row in the submatrix where the calculation for all the processes starts and ends.
        start_row_matrix, start_column_matrix = divmod(np.mod(start_hole_matrix, 1),1)
        end_row_matrix, end_column_matrix = divmod(np.mod(end_hole_matrix, 1),1)
        
        all_tasks = np.array([(start_matrix[i],start_hole_line[i],end_matrix[i],end_hole_line[i]) for i in range(start_matrix.size)])
        
        # Compute the submatrices of D on each process.
        # If only one submatrix needs to be calculated in the process
        
        n_all_tasks = len(all_tasks)
        if n_all_tasks > size:
            if n_masters >= size:
                print("It's not allowed that the amout of master processes is equal or bigger than the total amount of processes", flush=True)
            if rank < n_masters:
                is_master = True
                is_task = False
            else :
                is_master = False
                is_task = True
            if is_master:
                color = 1
            else: 
                color = MPI.UNDEFINED
            master_comm = comm.Split(color=color, key=rank)


            if is_master:
                # Amount of ranks without master ranks
                task_first_index_send = size-n_masters
                # How many worker ranks has each master rank
                first_ave, first_rest = divmod(task_first_index_send,n_masters)
                task_index_send = np.concatenate((np.repeat(first_ave+1,first_rest), np.repeat(first_ave, n_masters-first_rest)))

                task_index_send_per_master = 0
                task_index_recv = 0
                n_recv_idx_per_master = 0
                
                # Task which will be scattered in the beginning to all worker tasks i
                first_tasks = all_tasks[:task_first_index_send]

                # With the master ranks (then 0)
                first_tasks_with_dummy = np.concatenate((np.zeros((n_masters,all_tasks[0].size),dtype=np.int64),first_tasks), axis=0, dtype = np.int64)
                
                # All the other tasks which haven't been scattered
                rest_tasks = all_tasks[task_first_index_send:]
                tasks_splitted = [rest_tasks[i::n_masters] for i in range(n_masters)]
                
                # Indices of the tasks which haven't been scattered
                idx_tasks_all_send = np.arange(task_first_index_send, n_all_tasks)

                # List of indices which have to be send seperated for each master rank
                idx_tasks_splitted = [idx_tasks_all_send[i::n_masters] for i in range(n_masters)]

                # List of indices which have to be send from this master rank
                idx_tasks_per_master = idx_tasks_splitted[rank]
                tasks_per_master = tasks_splitted[rank]
                n_send_tasks_per_master = len(tasks_per_master)
                n_recv_tasks_per_master = n_send_tasks_per_master + task_index_send[rank]


            first_task = np.empty((1,4), dtype = np.int64)
            if is_task: 
                first_tasks_with_dummy = np.zeros((size,all_tasks[0].size), dtype = np.int64)
            comm.Scatter(first_tasks_with_dummy, first_task, root = 0)

            if is_task:
                matrix_per_column = calculate_D_matrix_without_holes_partial(vi, collocation_nodes, phi, alpha, *first_task[0])
                if np.isnan(matrix_per_column).any():
                    print(f"I'm rank {rank} and nan is true",flush = True)
                    matrix_per_column = calculate_D_matrix_without_holes_partial_eps(vi, collocation_nodes, phi, alpha, *first_task[0])
                    if np.isnan(matrix_per_column).any():
                        print(f"I'm rank {rank} and nan is still true",flush=True)
                    else:
                        print(f"I'm rank {rank} and nan is NO MORE true",flush=True)

                comm.send((rank,rank-n_masters,matrix_per_column),dest=np.mod(rank,n_masters),tag=tag_done)

            if is_master:
                result_dict = {}
                idx_hard_break = 0
                while n_recv_idx_per_master < n_recv_tasks_per_master:
                    status = MPI.Status()
                    result_temp = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
                    task_rank, task_index_recv, task_result = result_temp
                    result_dict[task_index_recv] = task_result
                    if task_index_send_per_master < n_send_tasks_per_master:
                        idx_task = idx_tasks_per_master[task_index_send_per_master]
                        task = tasks_per_master[task_index_send_per_master]
                        comm.send((idx_task,task),dest=task_rank,tag=tag_start)
                        task_index_send_per_master = task_index_send_per_master + 1
                    else:
                        comm.send((None,None),dest=task_rank,tag=tag_exit)
                    task_index_recv = task_index_recv + 1
                    n_recv_idx_per_master = n_recv_idx_per_master + 1
                    idx_hard_break = idx_hard_break +1
                result_all_dicts = master_comm.gather(result_dict, root = 0)
                if rank == 0:
                    merged_dicts = dict(chain.from_iterable(d.items() for d in result_all_dicts))
                    items_sorted_all_results = dict(sorted(merged_dicts.items()))
                    D_matrix_holes_2d = np.vstack(list(items_sorted_all_results.values()))
                    D_matrix_holes = D_matrix_holes_2d.reshape(N,n_coll,n_coll)

            elif is_task:
                while True:
                    status = MPI.Status()
                    idx_task, task = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)
                    tag = status.Get_tag()
                    if tag == tag_exit:
                        break
                    elif tag == tag_start:
                        matrix_per_column = calculate_D_matrix_without_holes_partial(vi, collocation_nodes, phi, alpha, *task)
                        if np.isnan(matrix_per_column).any():
                            print(f"I'm rank {rank} and nan is true",flush=True)
                            matrix_per_column = calculate_D_matrix_without_holes_partial_eps(vi, collocation_nodes, phi, alpha, *task)
                            if np.isnan(matrix_per_column).any():
                                print(f"I'm rank {rank} and nan is still true",flush=True)
                            else:
                                print(f"I'm rank {rank} and nan is NO MORE true",flush=True)
                                
                        comm.send((rank,idx_task,matrix_per_column),dest=status.Get_source(),tag=tag_done)
            if rank == 0:
                D_matrix_holes_conc = np.concatenate(D_matrix_holes, axis = 1)
                M_matrix_holes = D_matrix_holes_conc - np.tile(np.identity(D_matrix_holes_conc[:,0].size),N)/2 
                M_matrix_holes_3d = np.reshape(M_matrix_holes,(n_coll,N,n_coll)).transpose(1,0,2)
            if rank != 0:
                M_matrix_holes_3d = np.empty((N, n_coll, n_coll),dtype=np.complex128)
    else:
        M_matrix_holes_3d = np.array([M_matrix(phi_r) for phi_r in phi])
    
    color_split = 0 if (rank < N and rank < size) else MPI.UNDEFINED
    sub_comm = comm.Split(color=color_split, key=rank)
    if sub_comm != MPI.COMM_NULL:
        if rank == 0:
            time_start1 = time.time()
        if M_matrix == False:
            M_matrix_holes_3d = np.ascontiguousarray(M_matrix_holes_3d)
            sub_comm.Bcast(M_matrix_holes_3d,root=0)
        sub_size = sub_comm.Get_size()
        sub_rank = sub_comm.Get_rank()
        # How many 2d matrices must be calculated by a process?
        ave_matrices, res_matrices = divmod(N, sub_size)
        # Calculate number of lines that each process has to calculate
        n_matrices = np.concatenate((np.repeat(ave_matrices+1, res_matrices), np.repeat(ave_matrices, sub_size-res_matrices)))
        cumsum_matrices = np.cumsum(n_matrices)
        cumsum_matrices = np.insert(cumsum_matrices,0,0).astype(np.int32)
        # Calculate the rhs of the LES Mx = V_hat phi_derivative to solve x, which is nessecary for x = M_inv V_hat phi_derivative
        rhs_solve = V_hat*phi_derivative[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis]
        # Summands for the approximated A0
        A_0_summand_partial = np.linalg.solve(M_matrix_holes_3d[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],:,:], rhs_solve)
        A_0_summand = np.empty((N,m,l), dtype = np.complex128)
        # Number of submatrices in each process.
        sendcounts_matrices = n_matrices*m*l
        # Cumulative number of submatrices calculated up to the current process.
        displacements_matrices = cumsum_matrices[:-1]*m*l
        sub_comm.Gatherv(A_0_summand_partial, [A_0_summand, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
        sub_comm.Free()
    if rank == 0:
        # Summands for the approximated A1
        A_1_summand = phi[:,np.newaxis, np.newaxis] * A_0_summand
        # Approximated A0 and A1
        A_0 = np.sum(A_0_summand, axis=0)/(1j*N)
        A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
        # Calculate singular value decomposition of A0
        V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
        #Rank test
        Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
        rank_test = np.count_nonzero(Sigma0_eps)
        # Truncated singular value decomposition 
        Sigma_rank = Sigma0_eps[:rank_test]
        V_rank = V0[:,:rank_test]
        WH_rank = W0H[:rank_test,:]
        # Calculate matrix B
        B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
        # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
        eigenvalues, eigenvectors_B = np.linalg.eig(B)
        eigenvalues_shape = np.array(eigenvalues.shape, dtype = np.int32)
        # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
        eigenvectors = np.dot(V_rank,eigenvectors_B)
        eigenvectors_shape = np.array(eigenvectors.shape, dtype = np.int32)
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
    comm.Bcast(eigenvalues_shape, root = 0)
    comm.Bcast(eigenvectors_shape, root = 0)
    if rank != 0:
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    return eigenvalues, eigenvectors

def beyn_algorithm_with_holes (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    if M_matrix == False:
        # Add the first column to the end of vi
        vi_append = np.concatenate((vi, vi[:,:,0].reshape(vi.shape[0],2,1)), axis=2)
        # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
        M_matrices = np.array([create_M_matrix_with_holes (phi_r, vi_append, collocation_nodes, alpha) for phi_r in phi])
    else:
        M_matrices = np.array([M_matrix(phi_r) for phi_r in phi])
    # Inverse of the M-matrices
    M_matrices_inv = np.linalg.inv(M_matrices)
    # Prestep to calculate the summands for the tepezoidal rule
    A_0_summand_temp = np.dot(M_matrices_inv,V_hat)
    # Summands for the approximated A0
    A_0_summand = A_0_summand_temp*phi_derivative[:,np.newaxis, np.newaxis]
    # Summands for the approximated A1
    A_1_summand = phi[:,np.newaxis, np.newaxis] * A_0_summand
    # Approximated A0 and A1
    A_0 = np.sum(A_0_summand, axis=0)/(1j*N)
    A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
    # Calculate singular value decomposition of A0
    V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
    #Rank test
    Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
    rank = np.count_nonzero(Sigma0_eps)
    # Truncated singular value decomposition 
    Sigma_rank = Sigma0_eps[:rank]
    V_rank = V0[:,:rank]
    WH_rank = W0H[:rank,:]
    # Calculate matrix B
    B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
    # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
    eigenvalues, eigenvectors_B = np.linalg.eig(B)
    # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
    eigenvectors = np.dot(V_rank,eigenvectors_B)
    return eigenvalues, eigenvectors

def find_close_value(matrix, value, tol=1e-4):
    mask = np.isclose(matrix.real, value.real, atol=tol) & np.isclose(matrix.imag, value.imag, atol=tol)
    positions = np.argwhere(mask)
    return positions

def beyn_algorithm_with_holes_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    # Number of holes+1
    n_holes = vi.shape[0]

    # Number of collocation nodes per boundary
    n_coll = collocation_nodes.shape[2]
    # Add the first column of collocation nodes for each boundary to the end.
    vi = np.concatenate((vi, vi[:,:,0].reshape(vi.shape[0],2,1)), axis=2)
    
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # V_hat must be the same on all processes so that parallelization of the Beyn algorithm is possible later.
    comm.Bcast(V_hat,root=0)
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
     
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    #phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    if M_matrix == False:
        # How many rows must be calculated by a process?
        ave_n_rows_D, res_n_rows_D = divmod(N*n_coll*n_holes*n_holes, size)
        # Calculate number of lines that each process has to calculate
        n_lines = np.concatenate((np.repeat(ave_n_rows_D+1, res_n_rows_D), np.repeat(ave_n_rows_D, size-res_n_rows_D)))
        # Caluculate cummulative number of lines the processes have to calculate starting from zero
        cumsum_lines = np.cumsum(n_lines)
        cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
        #For phi
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_matrix, start_line = divmod(cumsum_lines[:-1], n_holes*n_holes*n_coll)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_holes*n_holes*n_coll)
        #For hole
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_hole_matrix, start_hole_line = divmod(cumsum_lines[:-1], n_coll)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_hole_matrix, end_hole_line = divmod(cumsum_lines[1:]-1, n_coll)
    
        start_row_matrix, start_column_matrix = divmod(np.mod(start_hole_matrix, n_holes*n_holes),n_holes)
        end_row_matrix, end_column_matrix = divmod(np.mod(end_hole_matrix, n_holes*n_holes),n_holes)
        
        # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
        start_matrix_partial = start_matrix[rank]
        start_row_partial = start_column_matrix[rank]
        start_column_partial = start_row_matrix[rank]
        start_line_partial = start_hole_line[rank]
        end_matrix_partial = end_matrix[rank]
        end_row_partial = end_column_matrix[rank]
        end_column_partial = end_row_matrix[rank]
        end_line_partial = end_hole_line[rank]
        # Compute the submatrices of D on each process.
        # If only one submatrix needs to be calculated in the process, this means just one phi value (position at circle of Beyn algorithm) is used 
        if start_matrix_partial == end_matrix_partial:
            # If all the used collocation nodes (the values which should be calculated) are at the same boundary
            if start_column_partial == end_column_partial:
                # If all the used vertex nodes (where to integrate about) are at the same boundary
                if start_row_partial == end_row_partial:
                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_row_partial], collocation_nodes[start_column_partial,:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                else:
                    for row in range (start_row_partial, end_row_partial+1):
                        # With vertex nodes, which are at the first used boundary to integrate about
                        if row == start_row_partial:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        # With vertex nodes, which are at the last used boundary to integrate about
                        elif row == end_row_partial:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        # With vertex nodes in between the first and last boundary to integrate about
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            # If all the used collocation nodes (the values which should be calculated) are NOT at the same boundary 
            else:
                for column in range (start_column_partial, end_column_partial+1):
                    # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                    if column == start_column_partial:
                        for row in range (start_row_partial, n_holes):
                            if row == start_row_partial:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    # If all the used collocation nodes (the values which should be calculated) are at the last boundary
                    elif column == end_column_partial:
                        for row in range (end_row_partial + 1):
                            if row == end_row_partial:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    # If all the used collocation nodes (the values which should be calculated) are between the first and last boundary
                    else:
                        for row in range(n_holes):
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
        #If the rows to be calculated extend across multiple sub-matrices
        else: 
            for i in range(start_matrix_partial, end_matrix_partial+1):
                if i == start_matrix_partial:
                    for column in range (start_column_partial, n_holes):
                        # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                        if column == start_column_partial:
                            for row in range (start_row_partial, n_holes):
                                # Calculation of the first submatrix
                                if row == start_row_partial:
                                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                #Calculation of the submatrices between the first and last.
                                else:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        else:
                            for row in range (n_holes):
                                #Calculation of the submatrices between the first and last.
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                elif i == end_matrix_partial:
                    for column in range (end_column_partial+1):
                        if column == end_column_partial:
                            for row in range (end_row_partial+1):
                                # Calculation of the last submatrix
                                if row == end_row_partial:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                                #Calculation of the submatrices between the first and last.
                                else:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        else:
                            for row in range (n_holes):
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                else:
                    for column in range (n_holes):
                        for row in range (n_holes):
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)

        # Buffer to store the entire D-matrix 
        D_matrix_holes = np.empty((N, n_holes, n_holes, n_coll, n_coll),dtype=np.complex128)
        # Number of elements per D-matrices in each process.
        sendcounts_matrices = n_lines*n_coll
        # Cumulative number of elements of the D matrices calculated up to the current process.
        displacements_matrices = cumsum_lines[:-1] * n_coll
        # Collect all parts of the D-Matrix and store them in a large
        comm.Gatherv(D_matrix_holes_partial, [D_matrix_holes, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
        # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
        D_matrix_holes_conc = np.concatenate(np.concatenate(np.concatenate(D_matrix_holes, axis = 1), axis=1), axis=1)
        M_matrix_holes = D_matrix_holes_conc - np.tile(np.identity(D_matrix_holes_conc[:,0].size),N)/2 
        M_matrix_holes_3d = np.reshape(M_matrix_holes,(n_holes*n_coll,N,n_holes*n_coll)).transpose(1,0,2)
    else:
        M_matrix_holes_3d = np.array([M_matrix(phi_r) for phi_r in phi])

    color_split = 0 if (rank < N and rank < size) else MPI.UNDEFINED
    sub_comm = comm.Split(color=color_split,key=rank)
    
    if sub_comm != MPI.COMM_NULL:
        if M_matrix == False:
            M_matrix_holes_3d = np.ascontiguousarray(M_matrix_holes_3d)
            sub_comm.Bcast(M_matrix_holes_3d, root = 0)
        sub_rank = sub_comm.Get_rank() # Calling process of the communicator
        sub_size = sub_comm.Get_size() # Number of processes in the communicator
        # How many 2d matrices must be calculated by a process?
        ave_matrices, res_matrices = divmod(N, sub_size)
        # Calculate number of lines that each process has to calculate
        n_matrices = np.concatenate((np.repeat(ave_matrices+1, res_matrices), np.repeat(ave_matrices, sub_size-res_matrices)))
        cumsum_matrices = np.cumsum(n_matrices)
        cumsum_matrices = np.insert(cumsum_matrices,0,0).astype(np.int32)
        # Calculate the rhs of the LES Mx = V_hat phi_derivative to solve x, which is nessecary for x = M_inv V_hat phi_derivative
        rhs_solve = V_hat*phi_derivative[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis]
        # Summands for the approximated A0
        A_0_summand_partial = np.linalg.solve(M_matrix_holes_3d[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],:,:], rhs_solve)
        A_0_summand = np.empty((N,m,l), dtype = np.complex128)
        # Number of submatrices in each process.
        sendcounts_A_0_summand = n_matrices*m*l
        # Cumulative number of submatrices calculated up to the current process.
        displacements_A_0_summand = cumsum_matrices[:-1]*m*l
        
        sub_comm.Gatherv(A_0_summand_partial, [A_0_summand, sendcounts_A_0_summand, displacements_A_0_summand, MPI.DOUBLE_COMPLEX], root = 0)
        sub_comm.Free()
    if rank == 0:
        M_matrices_inv_test = np.linalg.inv(M_matrix_holes_3d)
        # Prestep to calculate the summands for the tepezoidal rule
        A_0_summand_temp_test = np.dot(M_matrices_inv_test,V_hat)
        # Summands for the approximated A0
        A_0_summand_test = A_0_summand_temp_test*phi_derivative[:,np.newaxis, np.newaxis] 
        # Summands for the approximated A1
        A_1_summand = phi[:,np.newaxis, np.newaxis] * A_0_summand
        # Approximated A0 and A1
        A_0 = np.sum(A_0_summand, axis=0)/(1j*N)
        A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
        # Calculate singular value decomposition of A0
        V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
        #Rank test
        Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
        rank_test = np.count_nonzero(Sigma0_eps)
        # Truncated singular value decomposition 
        Sigma_rank = Sigma0_eps[:rank_test]
        V_rank = V0[:,:rank_test]
        WH_rank = W0H[:rank_test,:]
        # Calculate matrix B
        B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
        # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
        eigenvalues, eigenvectors_B = np.linalg.eig(B)
        eigenvalues_shape = np.array(eigenvalues.shape, dtype = np.int32)
        # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
        eigenvectors = np.dot(V_rank,eigenvectors_B)
        eigenvectors_shape = np.array(eigenvectors.shape, dtype = np.int32)
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
    comm.Bcast(eigenvalues_shape, root = 0)
    comm.Bcast(eigenvectors_shape, root = 0)
    if rank != 0:
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    return eigenvalues, eigenvectors

def calculate_n_transition_lines_over_block(idx_add_lines,n_transition_lines):
    n_transition_lines_over_blocks = np.copy(n_transition_lines)
    boundaries = np.where(idx_add_lines[1:] - idx_add_lines[:-1] > 1)[0]
    upper = np.append(boundaries, idx_add_lines.size - 1)
    lower = np.insert(upper[:-1] + 1, 0, 0)
    idx_lower = idx_add_lines[lower] - 1
    idx_upper = idx_add_lines[upper]
    cumsum_lines = np.insert(np.cumsum(n_transition_lines_over_blocks),0,0)
    sum_interval = cumsum_lines[idx_upper+1]- cumsum_lines[idx_lower]
    n_transition_lines_over_blocks[idx_lower] = sum_interval
    len_idx_min_1 = idx_upper-idx_lower
    len_idx_total = np.sum(len_idx_min_1)
    idx_start = np.cumsum(len_idx_min_1) - len_idx_min_1
    idx_start_per_idx = np.repeat(idx_start, len_idx_min_1)
    idx_per_idx = np.arange(len_idx_total) - idx_start_per_idx 
    idx_delete = np.repeat(idx_lower+1,len_idx_min_1)+idx_per_idx
    n_transition_lines_over_blocks[idx_delete]=0
    return n_transition_lines_over_blocks

def fill_left_lines(arr, n):
    if n==0:
        return arr
    elif n!=0 and arr.size == 2 and arr[1]==0:
        return np.concatenate([arr[:n]+1,arr[n:]])
    else:
        return np.concatenate([np.array([arr[0]]),arr[1:n+1]+1,arr[n+1:]])
    
def beyn_algorithm_with_holes_MPI_diff_n (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    # Number of holes+1
    n_holes = vi.size

    # Number of collocation nodes per boundary
    n_colls = np.array([m.shape[1] for m in collocation_nodes])
    n_colls_N = np.tile(n_colls,N)
    # Add the first column of collocation nodes for each boundary to the end.
    #vi = np.concatenate((vi, vi[:,:,0].reshape(vi.shape[0],2,1)), axis=2)
    vi = list(map(lambda vi_temp: np.concatenate((vi_temp, vi_temp[:,0].reshape(2,1)),axis=1), vi))
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # V_hat must be the same on all processes so that parallelization of the Beyn algorithm is possible later.
    comm.Bcast(V_hat,root=0)
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
     
    if M_matrix == False:
        # How many lines need to be calculated in total?
        time_start = time.time()
        n_lines_total = N*m*n_holes
        weight = n_colls/(m*N)
        weight_N = np.tile(weight,N)
        n_proc_block = weight_N*size

        start_proc_block = np.zeros(n_holes*N)
        n_whole_proc_block = np.zeros(n_holes*N)
        end_proc_block = np.zeros(n_holes*N)
        
        for i in range(n_holes*N):
            if i == 0:
                end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block[i])
            else:
                if end_proc_block[i-1] != 0:
                    start_proc_block_temp = np.round(1 - end_proc_block[i-1],10)
                    if np.round(n_proc_block[i] - start_proc_block_temp,8)>=0 and n_whole_proc_block[i-1]>=0:
                        start_proc_block[i] = start_proc_block_temp
                        n_proc_block_temp = n_proc_block[i] - start_proc_block[i]
                        end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block_temp)
                    elif np.round(n_proc_block[i] + n_whole_proc_block[i-1],8)>=0 and n_whole_proc_block[i-1]<0:
                        start_proc_block[i] = - n_whole_proc_block[i-1]
                        n_proc_block_temp = n_proc_block[i] - start_proc_block[i]
                        end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block_temp)
                    else:
                        end_proc_block[i] = n_proc_block[i]
                        if n_whole_proc_block[i-1]>=0:
                            n_whole_proc_block[i] = -np.round(1- end_proc_block[i-1] - n_proc_block[i],10)
                        else: 
                            n_whole_proc_block[i] = n_whole_proc_block[i-1] + end_proc_block[i]
                else:
                    end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block[i])
        idx_add_lines = np.argwhere(n_whole_proc_block<0).flatten()
        n_whole_proc_block = np.round(n_whole_proc_block).astype(int)
        n_lines_per_proc_block = m/n_proc_block
        start_lines = n_lines_per_proc_block*start_proc_block
        end_lines = n_lines_per_proc_block*end_proc_block
        middle_lines = [np.repeat(n_lines_per_proc_block[i], max(0,n_whole_proc_block[i])) for i in range(n_whole_proc_block.size)]
        all_lines = [np.insert(sub_middle_lines, [0,sub_middle_lines.size], [sub_start_lines, sub_end_lines]).astype(int) for sub_middle_lines, sub_start_lines, sub_end_lines in zip(middle_lines, start_lines, end_lines)]
        sum_lines_per_column = np.array(list(map(lambda ai: np.sum(ai), all_lines)))
        left_lines = m-sum_lines_per_column
        lines_per_proc_per_block = list(map(fill_left_lines, all_lines, left_lines))
        n_transition_lines =  np.array(list(map(lambda lines_per_proc_per_block_begin, lines_per_proc_per_block_end: lines_per_proc_per_block_begin[-1]+lines_per_proc_per_block_end[0], lines_per_proc_per_block[:-1], lines_per_proc_per_block[1:])))
        n_transition_lines_over_blocks = calculate_n_transition_lines_over_block(idx_add_lines,n_transition_lines)
        
        n_lines_per_proc_without_last = list(map(lambda lines_per_proc_per_block_temp, n_transition_lines_temp: np.append(lines_per_proc_per_block_temp[1:-1], n_transition_lines_temp), lines_per_proc_per_block, n_transition_lines_over_blocks))
        
        n_lines_per_proc_with_null = np.concatenate([np.concatenate(n_lines_per_proc_without_last), lines_per_proc_per_block[-1][1:]])
        
        n_lines_per_proc_without_null = n_lines_per_proc_with_null[n_lines_per_proc_with_null!=0]
        cumsum_lines = np.cumsum(n_lines_per_proc_without_null)
        cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
        
        #For phi
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_matrix, start_line = divmod(cumsum_lines[:-1], n_holes*m)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_holes*m)
        start_column_matrix, start_line_column = divmod(start_line,m)
        end_column_matrix, end_line_column = divmod(end_line,m)

        ni_coll = list(map(lambda m: m.shape[1], collocation_nodes))
        cumsum_coll = np.cumsum(ni_coll)
        
        start_row_matrix = np.searchsorted(cumsum_coll, start_line_column,'right')
        end_row_matrix = np.searchsorted(cumsum_coll, end_line_column,'right')
        cumsum_coll_null = np.insert(cumsum_coll,0,0)
        start_hole_line = start_line_column - cumsum_coll_null[start_row_matrix]
        end_hole_line = end_line_column - cumsum_coll_null[end_row_matrix]
        
        
        # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
        start_matrix_partial = start_matrix[rank]
        start_row_partial = start_row_matrix[rank]
        start_column_partial = start_column_matrix[rank]
        start_line_partial = start_hole_line[rank]
        end_matrix_partial = end_matrix[rank]
        end_row_partial = end_row_matrix[rank]
        end_column_partial = end_column_matrix[rank]
        end_line_partial = end_hole_line[rank]
        time_end = time.time()
        diff_time = time_end -time_start
        matrix_per_column = {key: np.empty((0,n_coll)) for key, n_coll in enumerate(n_colls)}
        
        # Compute the submatrices of D on each process.
        # If only one submatrix needs to be calculated in the process, this means just one phi value (position at circle of Beyn algorithm) is used 
        factor_partial = abs(np.imag(sp.hankel1(0.0,4*phi[start_matrix_partial:end_matrix_partial+1])))
        factor = abs(np.imag(sp.hankel1(0.0,4*phi)))+abs(np.real(sp.hankel1(0.0,4*phi)))
        factor_sum = factor.sum()
        factor_partial_sum = factor_partial.sum()

        time_start_alg = time.time()
        if start_matrix_partial == end_matrix_partial:
            # If all the used collocation nodes (the values which should be calculated) are at the same boundary
            if start_column_partial == end_column_partial:
                # If all the used vertex nodes (where to integrate about) are at the same boundary
                if start_row_partial == end_row_partial:
                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[start_row_partial][:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                else:
                    for row in range (start_row_partial, end_row_partial+1):
                        # With vertex nodes, which are at the first used boundary to integrate about
                        if row == start_row_partial:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        # With vertex nodes, which are at the last used boundary to integrate about
                        elif row == end_row_partial:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        # With vertex nodes in between the first and last boundary to integrate about
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                D_matrix_holes_partial_temp_dict = matrix_per_column[start_column_partial]
                D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                matrix_per_column[start_column_partial] = D_matrix_holes_partial_dict
            # If all the used collocation nodes (the values which should be calculated) are NOT at the same boundary 
            else:
                for column in range (start_column_partial, end_column_partial+1):
                    # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                    if column == start_column_partial:
                        for row in range (start_row_partial, n_holes):
                            if row == start_row_partial:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        
                    # If all the used collocation nodes (the values which should be calculated) are at the last boundary
                    elif column == end_column_partial:
                        for row in range (end_row_partial + 1):
                            if row == end_row_partial and row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            elif row == end_row_partial and row != 0:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                            elif row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        
                    # If all the used collocation nodes (the values which should be calculated) are between the first and last boundary
                    else:
                        for row in range(n_holes):
                            if row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
        #If the rows to be calculated extend across multiple sub-matrices
        else: 
            for i in range(start_matrix_partial, end_matrix_partial+1):
                if i == start_matrix_partial:
                    for column in range (start_column_partial, n_holes):
                        # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                        if column == start_column_partial:
                            for row in range (start_row_partial, n_holes):
                                # Calculation of the first submatrix
                                if row == start_row_partial:
                                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                #Calculation of the submatrices between the first and last.
                                else:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        else:
                            for row in range (n_holes):
                                #Calculation of the submatrices between the first and last.
                                if row == 0:
                                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                else:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                        D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                        matrix_per_column[column] = D_matrix_holes_partial_dict
                elif i == end_matrix_partial:
                    for column in range (end_column_partial+1):
                        if column == end_column_partial:
                            for row in range (end_row_partial+1):
                                # Calculation of the last submatrix
                                if row == end_row_partial and row == 0:
                                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    
                                elif row == end_row_partial and row != 0:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                                #Calculation of the submatrices between the first and last.
                                elif row == 0:
                                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                else:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        else:
                            for row in range (n_holes):
                                if row == 0:
                                    D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                else:
                                    D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                    D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                        D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                        matrix_per_column[column] = D_matrix_holes_partial_dict
                else:
                    for column in range (n_holes):
                        for row in range (n_holes):
                            if row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                                
                        D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                        D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                        matrix_per_column[column] = D_matrix_holes_partial_dict
        time_end_alg = time.time()
        time_diff_alg = time_end_alg- time_start_alg
        factor_frac = factor_partial_sum/factor_sum
        sendcounts_matrices_partial = np.array(list(map(lambda matrix_per_column_temp: matrix_per_column_temp.shape[0]*matrix_per_column_temp.shape[1], matrix_per_column.values())))
        sendcounts_matrices = np.empty((size* n_holes), dtype=sendcounts_matrices_partial.dtype)
        comm.Gather(sendcounts_matrices_partial, sendcounts_matrices, root=0)
        comm.Bcast(sendcounts_matrices, root = 0)
        sendcounts_matrices = sendcounts_matrices.reshape((size, n_holes))
        displacements_matrices = np.insert(np.cumsum(sendcounts_matrices, axis=0),0,0,axis=0)[:-1,:]

        # Buffer to store the entire D-matrix
        D_matrix_holes = {key : np.empty((N* m, n_coll),dtype=D_matrix_holes_partial.dtype) for key, n_coll in enumerate(n_colls)}
        # Collect all parts of the D-Matrix and store them in a large
        for i in range(n_holes):
            comm.Gatherv(matrix_per_column[i], [D_matrix_holes[i], sendcounts_matrices[:,i], displacements_matrices[:,i], MPI.DOUBLE_COMPLEX], root = 0)
        D_matrix_holes_list = list(D_matrix_holes.values())
        # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
        D_matrix_holes_list_concatenate = np.concatenate(D_matrix_holes_list, axis = 1)
        D_matrix_holes_3d = D_matrix_holes_list_concatenate.reshape((N,m,m))
        M_matrix_holes_3d = D_matrix_holes_3d - (np.zeros((N,m,m))+np.eye(m)[None,:,:])/2 
        
    else:
        M_matrix_holes_3d = np.array([M_matrix(phi_r) for phi_r in phi])

    color_split = 0 if (rank < N and rank < size) else MPI.UNDEFINED
    sub_comm = comm.Split(color=color_split,key=rank)
    
    if sub_comm != MPI.COMM_NULL:
        if M_matrix == False:
            M_matrix_holes_3d = np.ascontiguousarray(M_matrix_holes_3d)
            sub_comm.Bcast(M_matrix_holes_3d, root = 0)
        sub_rank = sub_comm.Get_rank() # Calling process of the communicator
        sub_size = sub_comm.Get_size() # Number of processes in the communicator
        # How many 2d matrices must be calculated by a process?
        ave_matrices, res_matrices = divmod(N, sub_size)
        # Calculate number of lines that each process has to calculate
        n_matrices = np.concatenate((np.repeat(ave_matrices+1, res_matrices), np.repeat(ave_matrices, sub_size-res_matrices)))
        cumsum_matrices = np.cumsum(n_matrices)
        cumsum_matrices = np.insert(cumsum_matrices,0,0).astype(np.int32)
        # Calculate the rhs of the LES Mx = V_hat phi_derivative to solve x, which is nessecary for x = M_inv V_hat phi_derivative
        rhs_solve = V_hat*phi_derivative[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis]
        # Summands for the approximated A0
        A_0_summand_partial = np.linalg.solve(M_matrix_holes_3d[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],:,:], rhs_solve)
        A_1_summand_partial =  phi[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis] * A_0_summand_partial
        
        if N == sub_size:
            A_0_summand_partial_2d = A_0_summand_partial[0]
            A_0_factor = np.zeros_like(A_0_summand_partial_2d)
            sub_comm.Reduce(A_0_summand_partial_2d, A_0_factor, op=MPI.SUM, root = 0)
            A_1_summand_partial_2d = A_1_summand_partial[0]
            A_1_factor = np.zeros_like(A_1_summand_partial_2d)
            sub_comm.Reduce(A_1_summand_partial_2d, A_1_factor, op=MPI.SUM, root = 0)
        else: 
            A_0_summand_partial_2d = np.sum(A_0_summand_partial,axis = 0)
            A_0_factor = np.zeros_like(A_0_summand_partial_2d)
            sub_comm.Reduce(A_0_summand_partial_2d, A_0_factor, op=MPI.SUM, root = 0)
            A_1_summand_partial_2d = np.sum(A_1_summand_partial,axis = 0)
            A_1_factor = np.zeros_like(A_1_summand_partial_2d)
            sub_comm.Reduce(A_1_summand_partial_2d, A_1_factor, op=MPI.SUM, root = 0)
        sub_comm.Free()
    if rank == 0:
        # Approximated A0 and A1
        A_0 = A_0_factor/(1j*N)
        #A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
        A_1 = A_1_factor/(1j*N)
        # Calculate singular value decomposition of A0
        V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
        #Rank test
        Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
        rank_test = np.count_nonzero(Sigma0_eps)
        # Truncated singular value decomposition 
        Sigma_rank = Sigma0_eps[:rank_test]
        V_rank = V0[:,:rank_test]
        WH_rank = W0H[:rank_test,:]
        # Calculate matrix B
        B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
        # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
        eigenvalues, eigenvectors_B = np.linalg.eig(B)
        eigenvalues_shape = np.array(eigenvalues.shape, dtype = np.int32)
        # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
        eigenvectors = np.dot(V_rank,eigenvectors_B)
        eigenvectors_shape = np.array(eigenvectors.shape, dtype = np.int32)
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
    comm.Bcast(eigenvalues_shape, root = 0)
    comm.Bcast(eigenvectors_shape, root = 0)
    if rank != 0:
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    return eigenvalues, eigenvectors


def calculate_D_matrix_partial(vi, collocation_nodes, phi, alpha, start_matrix_partial, start_column_partial, start_row_partial, start_line_partial, end_matrix_partial, end_column_partial, end_row_partial, end_line_partial):
    # Number of holes+1
    n_holes = len(vi)

    # Number of collocation nodes per boundary
    n_colls = np.array([m.shape[1] for m in collocation_nodes])
    matrix_per_column = {key: np.empty((0,n_coll)) for key, n_coll in enumerate(n_colls)}
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process, this means just one phi value (position at circle of Beyn algorithm) is used 
    time_start_alg = time.time()
    if start_matrix_partial == end_matrix_partial:
        # If all the used collocation nodes (the values which should be calculated) are at the same boundary
        if start_column_partial == end_column_partial:
            # If all the used vertex nodes (where to integrate about) are at the same boundary
            if start_row_partial == end_row_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[start_row_partial][:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]

            else:
                for row in range (start_row_partial, end_row_partial+1):
                    # With vertex nodes, which are at the first used boundary to integrate about
                    if row == start_row_partial:
                        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                    # With vertex nodes, which are at the last used boundary to integrate about
                    elif row == end_row_partial:
                        D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    # With vertex nodes in between the first and last boundary to integrate about
                    else:
                        D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            D_matrix_holes_partial_temp_dict = matrix_per_column[start_column_partial]
            D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
            matrix_per_column[start_column_partial] = D_matrix_holes_partial_dict
        # If all the used collocation nodes (the values which should be calculated) are NOT at the same boundary 
        else:
            for column in range (start_column_partial, end_column_partial+1):
                # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                if column == start_column_partial:
                    for row in range (start_row_partial, n_holes):
                        if row == start_row_partial:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    
                # If all the used collocation nodes (the values which should be calculated) are at the last boundary
                elif column == end_column_partial:
                    for row in range (end_row_partial + 1):
                        if row == end_row_partial and row == 0:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        elif row == end_row_partial and row != 0:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        elif row == 0:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    
                # If all the used collocation nodes (the values which should be calculated) are between the first and last boundary
                else:
                    for row in range(n_holes):
                        if row == 0:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                matrix_per_column[column] = D_matrix_holes_partial_dict
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            if i == start_matrix_partial:
                for column in range (start_column_partial, n_holes):
                    # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                    if column == start_column_partial:
                        for row in range (start_row_partial, n_holes):
                            # Calculation of the first submatrix
                            if row == start_row_partial:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            #Calculation of the submatrices between the first and last.
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    else:
                        for row in range (n_holes):
                            #Calculation of the submatrices between the first and last.
                            if row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
            elif i == end_matrix_partial:
                for column in range (end_column_partial+1):
                    if column == end_column_partial:
                        for row in range (end_row_partial+1):
                            # Calculation of the last submatrix
                            if row == end_row_partial and row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                
                            elif row == end_row_partial and row != 0:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                            #Calculation of the submatrices between the first and last.
                            elif row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    else:
                        for row in range (n_holes):
                            if row == 0:
                                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            else:
                                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
            else:
                for column in range (n_holes):
                    for row in range (n_holes):
                        if row == 0:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                            
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
    time_end_alg = time.time()
    time_diff_alg = time_end_alg- time_start_alg
    return matrix_per_column


def calculate_D_matrix_partial_eps(vi, collocation_nodes, phi, alpha, start_matrix_partial, start_column_partial, start_row_partial, start_line_partial, end_matrix_partial, end_column_partial, end_row_partial, end_line_partial):
    # Number of holes+1
    n_holes = len(vi)

    # Number of collocation nodes per boundary
    n_colls = np.array([m.shape[1] for m in collocation_nodes])
    matrix_per_column = {key: np.empty((0,n_coll)) for key, n_coll in enumerate(n_colls)}
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process, this means just one phi value (position at circle of Beyn algorithm) is used 

    eps = 1e-6

    time_start_alg = time.time()
    if start_matrix_partial == end_matrix_partial:
        # If all the used collocation nodes (the values which should be calculated) are at the same boundary
        if start_column_partial == end_column_partial:
            # If all the used vertex nodes (where to integrate about) are at the same boundary
            if start_row_partial == end_row_partial:
                D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[start_row_partial][:,start_line_partial:end_line_partial+1], alpha))[0]
                D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[start_row_partial][:,start_line_partial:end_line_partial+1], alpha), points = [0.5])[0]
                D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[start_row_partial][:,start_line_partial:end_line_partial+1], alpha))[0]
                D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
            else:
                for row in range (start_row_partial, end_row_partial+1):
                    # With vertex nodes, which are at the first used boundary to integrate about
                    if row == start_row_partial:
                        D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,start_line_partial:], alpha))[0]
                        D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,start_line_partial:], alpha), points = [0.5])[0]
                        D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,start_line_partial:], alpha))[0]
                        D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                    # With vertex nodes, which are at the last used boundary to integrate about
                    elif row == end_row_partial:
                        D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                        D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [0.5])[0]
                        D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                        D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    # With vertex nodes in between the first and last boundary to integrate about
                    else:
                        D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:], alpha))[0]
                        D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                        D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[start_column_partial], collocation_nodes[row][:,:], alpha))[0]
                        D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            D_matrix_holes_partial_temp_dict = matrix_per_column[start_column_partial]
            D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
            matrix_per_column[start_column_partial] = D_matrix_holes_partial_dict
        # If all the used collocation nodes (the values which should be calculated) are NOT at the same boundary 
        else:
            for column in range (start_column_partial, end_column_partial+1):
                # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                if column == start_column_partial:
                    for row in range (start_row_partial, n_holes):
                        if row == start_row_partial:
                            D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,start_line_partial:], alpha))[0]
                            D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,start_line_partial:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,start_line_partial:], alpha))[0]
                            D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                        else:
                            D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    
                # If all the used collocation nodes (the values which should be calculated) are at the last boundary
                elif column == end_column_partial:
                    for row in range (end_row_partial + 1):
                        if row == end_row_partial and row == 0:
                            D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                            D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                            D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                        elif row == end_row_partial and row != 0:
                            D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                            D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                            D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        elif row == 0:
                            D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                        else:
                            D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    
                # If all the used collocation nodes (the values which should be calculated) are between the first and last boundary
                else:
                    for row in range(n_holes):
                        if row == 0:
                            D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                        else:
                            D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[start_matrix_partial], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                matrix_per_column[column] = D_matrix_holes_partial_dict
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            if i == start_matrix_partial:
                for column in range (start_column_partial, n_holes):
                    # If all the used collocation nodes (the values which should be calculated) are at the first boundary
                    if column == start_column_partial:
                        for row in range (start_row_partial, n_holes):
                            # Calculation of the first submatrix
                            if row == start_row_partial:
                                D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,start_line_partial:], alpha))[0]
                                D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,start_line_partial:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,start_line_partial:], alpha))[0]
                                D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                            #Calculation of the submatrices between the first and last.
                            else:
                                D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    else:
                        for row in range (n_holes):
                            #Calculation of the submatrices between the first and last.
                            if row == 0:
                                D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                            else:
                                D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
            elif i == end_matrix_partial:
                for column in range (end_column_partial+1):
                    if column == end_column_partial:
                        for row in range (end_row_partial+1):
                            # Calculation of the last submatrix
                            if row == end_row_partial and row == 0:
                                D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                                D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                                D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                                
                            elif row == end_row_partial and row != 0:
                                D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                                D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:end_line_partial+1], alpha))[0]
                                D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                            #Calculation of the submatrices between the first and last.
                            elif row == 0:
                                D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                            else:
                                D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    else:
                        for row in range (n_holes):
                            if row == 0:
                                D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                            else:
                                D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                                D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                                D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
            else:
                for column in range (n_holes):
                    for row in range (n_holes):
                        if row == 0:
                            D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial = D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3   
                        else:
                            D_matrix_holes_partial_temp_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha), points = [0.5])[0]
                            D_matrix_holes_partial_temp_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(phi[i], vi[column], collocation_nodes[row][:,:], alpha))[0]
                            D_matrix_holes_partial_temp = D_matrix_holes_partial_temp_1 + D_matrix_holes_partial_temp_2 + D_matrix_holes_partial_temp_3   
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                            
                    D_matrix_holes_partial_temp_dict = matrix_per_column[column]
                    D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
                    matrix_per_column[column] = D_matrix_holes_partial_dict
    time_end_alg = time.time()
    time_diff_alg = time_end_alg- time_start_alg
    return matrix_per_column

def beyn_algorithm_with_holes_MPI_diff_n_master_workers (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    #Initialize the MPI-Communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    # Number of holes+1
    n_holes = vi.size

    # Number of collocation nodes per boundary
    n_colls = np.array([m.shape[1] for m in collocation_nodes])
    n_lines = m*n_holes*N
    n_lines_per_task = min(lines_per_task, int(n_lines/size))
    n_repeat_full_tasks = int(n_lines/n_lines_per_task)
    n_tasks_rest = n_lines - n_repeat_full_tasks * n_lines_per_task
    all_tasks_lines = np.repeat(n_lines_per_task, n_repeat_full_tasks)
    if n_tasks_rest > 0:
        all_tasks_lines = np.append(all_tasks_lines, n_tasks_rest)
    
    # Add the first column of collocation nodes for each boundary to the end.
    vi = list(map(lambda vi_temp: np.concatenate((vi_temp, vi_temp[:,0].reshape(2,1)),axis=1), vi))
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # V_hat must be the same on all processes so that parallelization of the Beyn algorithm is possible later.
    comm.Bcast(V_hat,root=0)
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
     
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    if M_matrix == False:       
        tag_start = 0
        tag_done = 1
        tag_exit = 2
        # How many lines need to be calculated in total?
        time_start = time.time()
        n_lines_total = N*m*n_holes
        # Cumsum of amount of tasks with 0 at the beginning
        cumsum_lines = np.cumsum(all_tasks_lines)
        cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
        #For phi
        # Submatrix and row in the submatrix where the calculation starts for all processes.
        start_matrix, start_line = divmod(cumsum_lines[:-1], n_holes*m)
        # Submatrix and row in the submatrix where the calculation ends for all processes.
        end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_holes*m)
        start_column_matrix, start_line_column = divmod(start_line,m)
        end_column_matrix, end_line_column = divmod(end_line,m)

        ni_coll = list(map(lambda m: m.shape[1], collocation_nodes))
        cumsum_coll = np.cumsum(ni_coll)
        
        start_row_matrix = np.searchsorted(cumsum_coll, start_line_column,'right')
        end_row_matrix = np.searchsorted(cumsum_coll, end_line_column,'right')
        cumsum_coll_null = np.insert(cumsum_coll,0,0)
        start_hole_line = start_line_column - cumsum_coll_null[start_row_matrix]
        end_hole_line = end_line_column - cumsum_coll_null[end_row_matrix]

        all_tasks = np.array([(start_matrix[i],start_column_matrix[i], start_row_matrix[i],start_hole_line[i],end_matrix[i],end_column_matrix[i], end_row_matrix[i],end_hole_line[i]) for i in range(start_matrix.size)])
        time_end = time.time()
        diff_time = time_end -time_start

        n_all_tasks = len(all_tasks)
        if n_all_tasks > size:
            if n_masters >= size:
                print("It's not allowed that the amout of master processes is equal or bigger than the total amount of processes", flush=True)
            if rank < n_masters:
                is_master = True
                is_task = False
            else :
                is_master = False
                is_task = True
            if is_master:
                color = 1
            else: 
                color = MPI.UNDEFINED
            master_comm = comm.Split(color=color, key=rank)

            if is_master:
                task_first_index_send = size-n_masters
                first_ave, first_rest = divmod(task_first_index_send,n_masters)
                task_index_send = np.concatenate((np.repeat(first_ave+1,first_rest), np.repeat(first_ave, n_masters-first_rest)))
                task_index_send_per_master = 0
                task_index_recv = 0
                n_recv_idx_per_master = 0
        
                first_tasks = all_tasks[:task_first_index_send]
                first_tasks_with_dummy = np.concatenate((np.zeros((n_masters,all_tasks[0].size),dtype=np.int64),first_tasks), axis=0, dtype = np.int64)
                rest_tasks = all_tasks[task_first_index_send:]
                tasks_splitted = [rest_tasks[i::n_masters] for i in range(n_masters)]
                idx_tasks_all_send = np.arange(task_first_index_send, n_all_tasks)
                idx_tasks_splitted = [idx_tasks_all_send[i::n_masters] for i in range(n_masters)]
                idx_tasks_per_master = idx_tasks_splitted[rank]
                tasks_per_master = tasks_splitted[rank]
                n_send_tasks_per_master = len(tasks_per_master)
                n_recv_tasks_per_master = n_send_tasks_per_master + task_index_send[rank]
            
            first_task = np.empty((1,8), dtype = np.int64)
            if is_task: 
                first_tasks_with_dummy = np.zeros((size,all_tasks[0].size), dtype = np.int64)
            comm.Scatter(first_tasks_with_dummy, first_task, root = 0)
            if is_task:
                matrix_per_column = calculate_D_matrix_partial(vi, collocation_nodes, phi, alpha, *first_task[0])
                comm.send((rank,rank-n_masters,matrix_per_column),dest=np.mod(rank,n_masters),tag=tag_done)
            if master_comm != MPI.COMM_NULL:
                result_dict = {}
                idx_hard_break = 0
                while n_recv_idx_per_master < n_recv_tasks_per_master:
                    status = MPI.Status()
                    result_temp = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
                    task_rank, task_index_recv, task_result = result_temp
                    result_dict[task_index_recv] = task_result
                    #if task_index_send_per_master < n_recv_tasks_per_master+1:
                    if task_index_send_per_master < n_send_tasks_per_master:
                        idx_task = idx_tasks_per_master[task_index_send_per_master]
                        task = tasks_per_master[task_index_send_per_master]
                        comm.send((idx_task,task),dest=task_rank,tag=tag_start)
                        task_index_send_per_master = task_index_send_per_master + 1
                    else:
                        comm.send((None,None),dest=task_rank,tag=tag_exit)
                    task_index_recv = task_index_recv + 1
                    n_recv_idx_per_master = n_recv_idx_per_master + 1
                    idx_hard_break = idx_hard_break +1
                if rank == 0:
                    result_all_dicts = result_dict
                    for i in range(master_comm.size-1):
                        part = master_comm.recv(source= MPI.ANY_SOURCE)
                        result_all_dicts.update(part)
                else:
                    master_comm.send(result_dict, dest = 0)
                if rank == 0:
                    items_sorted_all_results = dict(sorted(result_all_dicts.items()))
                    all_results_array = np.array(list(items_sorted_all_results.values()))
            elif is_task:
                while True:
                    status = MPI.Status()
                    idx_task, task = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)
                    tag = status.Get_tag()
                    if tag == tag_exit:
                        break
                    elif tag == tag_start:
                        matrix_per_column = calculate_D_matrix_partial(vi, collocation_nodes, phi, alpha, *task)
                                
                        comm.send((rank,idx_task,matrix_per_column),dest=status.Get_source(),tag=tag_done)
        # Buffer to store the entire D-matrix
        if rank == 0:
            columns = []
            for key in range(n_holes):
                column_stacked = np.vstack([all_results_array[arr][key] for arr in range(all_results_array.size)])
                columns.append(column_stacked)
            D_matrix_holes = np.hstack(columns)
            D_matrix_holes_3d = D_matrix_holes.reshape((N,m,m))
            M_matrix_holes_3d = D_matrix_holes_3d - np.eye(m)[None,:,:]/2 
        if rank != 0:
            M_matrix_holes_3d = np.empty((N,m,m), dtype = np.complex128)
    else:
        M_matrix_holes_3d = np.array([M_matrix(phi_r) for phi_r in phi])

    color_split = 0 if (rank < N and rank < size) else MPI.UNDEFINED
    sub_comm = comm.Split(color=color_split,key=rank)
    if sub_comm != MPI.COMM_NULL:
        sub_rank = sub_comm.Get_rank() # Calling process of the communicator
        sub_size = sub_comm.Get_size() # Number of processes in the communicator
        # How many 2d matrices must be calculated by a process?
        ave_matrices, res_matrices = divmod(N, sub_size)
        # Calculate number of lines that each process has to calculate
        n_matrices = np.concatenate((np.repeat(ave_matrices+1, res_matrices), np.repeat(ave_matrices, sub_size-res_matrices)))
        cumsum_matrices = np.cumsum(n_matrices)
        cumsum_matrices = np.insert(cumsum_matrices,0,0).astype(np.int32)
        if rank == 0:
            print(f"cumsum_matrices is {cumsum_matrices}",flush=True)
        
        M_matrix_sub_3d = np.empty((n_matrices[sub_rank],m,m), dtype = np.complex128)
        sub_comm.Scatterv([M_matrix_holes_3d, n_matrices*m*m, cumsum_matrices[:-1]*m*m, MPI.DOUBLE_COMPLEX], M_matrix_sub_3d, root = 0)


        # Calculate the rhs of the LES Mx = V_hat phi_derivative to solve x, which is nessecary for x = M_inv V_hat phi_derivative
        rhs_solve = V_hat*phi_derivative[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis]
        # Summands for the approximated A0
        A_0_summand_partial = np.linalg.solve(M_matrix_sub_3d, rhs_solve)

        A_1_summand_partial =  phi[cumsum_matrices[sub_rank]:cumsum_matrices[sub_rank+1],np.newaxis, np.newaxis] * A_0_summand_partial
        
        if N == sub_size:
            A_0_factor = np.empty((m,l), dtype = np.complex128)
            sub_comm.Reduce(A_0_summand_partial[0], A_0_factor, op=MPI.SUM, root = 0)
            A_1_factor = np.empty((m,l), dtype = np.complex128)
            sub_comm.Reduce(A_1_summand_partial[0], A_1_factor, op=MPI.SUM, root = 0)
        else: 
            A_0_summand_partial_2d = np.sum(A_0_summand_partial,axis = 0)
            A_0_factor = np.empty((m,l), dtype = np.complex128)
            sub_comm.Reduce(A_0_summand_partial_2d, A_0_factor, op=MPI.SUM, root = 0)
            A_1_summand_partial_2d = np.sum(A_1_summand_partial,axis = 0)
            A_1_factor = np.empty((m,l), dtype = np.complex128)
            sub_comm.Reduce(A_1_summand_partial_2d, A_1_factor, op=MPI.SUM, root = 0)
        sub_comm.Free()
    if rank == 0:
        # Approximated A0 and A1
        A_0 = A_0_factor/(1j*N)
        #A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
        A_1 = A_1_factor/(1j*N)
        # Calculate singular value decomposition of A0
        V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
        #Rank test
        Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
        rank_test = np.count_nonzero(Sigma0_eps)
        # Truncated singular value decomposition 
        Sigma_rank = Sigma0_eps[:rank_test]
        V_rank = V0[:,:rank_test]
        WH_rank = W0H[:rank_test,:]
        # Calculate matrix B
        B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
        # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
        eigenvalues, eigenvectors_B = np.linalg.eig(B)
        eigenvalues_shape = np.array(eigenvalues.shape, dtype = np.int32)
        # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
        eigenvectors = np.dot(V_rank,eigenvectors_B)
        eigenvectors_shape = np.array(eigenvectors.shape, dtype = np.int32)
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
    comm.Bcast(eigenvalues_shape, root = 0)
    comm.Bcast(eigenvectors_shape, root = 0)
    if rank != 0:
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    if rank == 0:
        print("finish", flush = True)
    return eigenvalues, eigenvectors


def calculate_eigenvalues_eigenvectors(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain without holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.
    
    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    eigenvalues, eigenvectors = beyn_algorithm_MPI(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors = beyn_algorithm_MPI(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors


def calculate_eigenvalues_eigenvectors_master_workers(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain without holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.
    
    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints of the approximated boundary
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    eigenvalues, eigenvectors = beyn_algorithm_MPI_master_workers(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors = beyn_algorithm_MPI_master_workers(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task)
    return eigenvalues, eigenvectors


def calculate_eigenvalues_eigenvectors_with_holes(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    eigenvalues, eigenvectors = beyn_algorithm_with_holes (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):#& (l < m):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors = beyn_algorithm_with_holes (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors

def calculate_eigenvalues_eigenvectors_with_holes(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    eigenvalues, eigenvectors = beyn_algorithm_with_holes (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):#& (l < m):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors = beyn_algorithm_with_holes (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors



def calculate_eigenvalues_eigenvectors_with_holes_MPI(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    
    eigenvalues, eigenvectors = beyn_algorithm_with_holes_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):#& (l < m):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors = beyn_algorithm_with_holes_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors

def calculate_eigenvalues_eigenvectors_with_holes_MPI_diff_n(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    
    eigenvalues, eigenvectors = beyn_algorithm_with_holes_MPI_diff_n (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):#& (l < m):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors = beyn_algorithm_with_holes_MPI_diff_n (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors

def calculate_eigenvalues_eigenvectors_with_holes_MPI_diff_n_master_workers(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
    M_matrix : bool or numpy.ndarray
        Array for which the eigenvalues and eigenfunctions should be calculated.
        If False a matrix is generated based on the wavenumber, vertices and collocation nodes.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    m : int
        Shape of the nonlinear eigenvalue problem (mxm), which has to be solved.
    l : int 
        The size of the the random matrix is mxl.
        l must be choosen so that n(\gamma) \leq l \leq m holds.
        Here n(\gamma) determines the rank of the right and left eigenvectors that can be found in the given circle.
    midpoint : float
        Center of the circle in which to search for eigenvalues.
    R : float
        Radius of the circle in which to search for eigenvalues.
    N : int
        Number of terms used for the summation in the trapezoidal rule.
    eps : float
        Value to apply the truncated singular value decomposition. For values less than eps, the singular value is set to zero.
        
    Returns
    -------
    eigenvalues : numpy.ndarray
        Numpy array containing all eigenvalues of the nonlinear eigenvalue problem that lie in the given circle.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    if rank == 0:
        print("First step", flush = True)
    eigenvalues, eigenvectors = beyn_algorithm_with_holes_MPI_diff_n_master_workers (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task)
    while (eigenvalues.shape[0] == l) & ((l < m) | (l > m)):#& (l < m):
        if rank == 0:
            print("Second step", flush = True)
        if (m-l) >= 10:
            l=l+10
            if rank == 0:
                print("Third step", flush = True)
        else:
            l=l+1
            if rank == 0:
                print("Fourth step", flush = True)
        eigenvalues, eigenvectors = beyn_algorithm_with_holes_MPI_diff_n_master_workers (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps, n_masters, lines_per_task)
    return eigenvalues, eigenvectors



def create_D_matrix_result_MPI (wavenumber, vi, collocation_nodes, inside, alpha):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    
    # Number of points from which to calculate the result.
    n_inside = inside.shape[1]
    
    color_split = np.floor(rank/n_inside)
    sub_comm = comm.Split(color=color_split,key=rank)

    sub_rank = sub_comm.Get_rank() # Calling process of the communicator
    sub_size = sub_comm.Get_size() # Number of processes in the communicator
    
    # Number of collocation nodes per boundary
    n_coll = collocation_nodes.shape[1]
    D_matrix = np.zeros((n_inside, n_coll), dtype='complex')
    
    # How many rows must be calculated by a process?
    ave_n_rows_D, res_n_rows_D = divmod(n_inside, sub_size)
    # Calculate number of lines that each process has to calculate
    n_lines = np.concatenate((np.repeat(ave_n_rows_D+1, res_n_rows_D), np.repeat(ave_n_rows_D, sub_size-res_n_rows_D)))
    # Caluculate cummulative number of lines the processes have to calculate starting from zero
    cumsum_lines = np.cumsum(n_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_inside)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_inside)

    # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
    start_matrix_partial = start_matrix[sub_rank]
    start_line_partial = start_line[sub_rank]
    end_matrix_partial = end_matrix[sub_rank]
    end_line_partial = end_line[sub_rank]
    
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process
    D_matrix_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi, inside[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
    
    # Buffer to store the entire D-matrix 
    D_matrix = np.empty((n_inside, n_coll),dtype=np.complex128)
    # Number of elements per D-matrices in each process.
    sendcounts_matrices = n_lines*n_coll
    # Cumulative number of elements of the D matrices calculated up to the current process.
    displacements_matrices = cumsum_lines[:-1] * n_coll
    # Collect all parts of the D-Matrix and store them in a large
    sub_comm.Gatherv(D_matrix_partial, [D_matrix, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
    # Free the subcommunicator
    sub_comm.Free()

    comm.Bcast(D_matrix, root = 0)
    return D_matrix


def create_D_matrix_result_MPI_master_workers (wavenumber, vi, collocation_nodes, inside, alpha, n_masters, lines_per_task):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    
    # Number of points from which to calculate the result.
    n_lines = inside.shape[1]
    
    # Number of collocation nodes per boundary
    n_coll = collocation_nodes.shape[1]
   
    tag_start = 0
    tag_done = 1
    tag_exit = 2

    n_lines_per_task = min(lines_per_task, int(n_lines/size))
    if n_lines_per_task != 0:
        n_repeat_full_tasks = int(n_lines/n_lines_per_task)
        n_tasks_rest = n_lines - n_repeat_full_tasks * n_lines_per_task
        all_tasks_lines = np.repeat(n_lines_per_task, n_repeat_full_tasks)
        if n_tasks_rest > 0:
            all_tasks_lines = np.append(all_tasks_lines, n_tasks_rest)
    else: 
        all_tasks_lines = np.repeat(1, n_lines)
    
    cumsum_lines = np.cumsum(all_tasks_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)

    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_lines)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_lines)

    all_tasks = np.array([(start_matrix[i],start_line[i],end_matrix[i],end_line[i]) for i in range(start_matrix.size)])

    n_all_tasks = len(all_tasks)
    if n_all_tasks <= size:
        if rank < n_all_tasks:
            color = 1
        else:
            color = MPI.UNDEFINED
        sub_comm = comm.Split(color=color, key=rank)
        if sub_comm != MPI.COMM_NULL:
            sub_rank = sub_comm.Get_rank()
            sub_size = sub_comm.Get_size()
            sub_tasks = np.empty((1,4), dtype = all_tasks.dtype)
            sub_comm.Scatter(all_tasks, sub_tasks, root = 0)
            matrix_per_column = {sub_rank: integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi, inside[:,sub_tasks[0,1]:sub_tasks[0,3]+1], alpha), points = [alpha, 0.5, 1-alpha])[0]}
            D_matrix_holes_dict = sub_comm.gather(matrix_per_column,root=0)
            if rank == 0:
                merged_dicts = dict(chain.from_iterable(d.items() for d in D_matrix_holes_dict))
                items_sorted_all_results = dict(sorted(merged_dicts.items()))
                all_results_array = np.array(list(items_sorted_all_results.values()))

    if n_all_tasks > size:
        
        if n_masters >= size:
            print("It's not allowed that the amout of master processes is equal or bigger than the total amount of processes", flush=True)
        if rank < n_masters:
            is_master = True
            is_task = False
        else :
            is_master = False
            is_task = True
        if is_master:
            color = 1
        else: 
            color = MPI.UNDEFINED
        master_comm = comm.Split(color=color, key=rank)

        if is_master:
            task_first_index_send = size-n_masters
            first_ave, first_rest = divmod(task_first_index_send,n_masters)
            task_index_send = np.concatenate((np.repeat(first_ave+1,first_rest), np.repeat(first_ave, n_masters-first_rest)))
            task_index_send_per_master = 0
            task_index_recv = 0
            n_recv_idx_per_master = 0

            first_tasks = all_tasks[:task_first_index_send]
            first_tasks_with_dummy = np.concatenate((np.zeros((n_masters,all_tasks[0].size),dtype=np.int64),first_tasks), axis=0, dtype = np.int64)

            rest_tasks = all_tasks[task_first_index_send:]
            tasks_splitted = [rest_tasks[i::n_masters] for i in range(n_masters)]

            idx_tasks_all_send = np.arange(task_first_index_send, n_all_tasks)
            idx_tasks_splitted = [idx_tasks_all_send[i::n_masters] for i in range(n_masters)]
            idx_tasks_per_master = idx_tasks_splitted[rank]
            tasks_per_master = tasks_splitted[rank]
            n_send_tasks_per_master = len(tasks_per_master)
            n_recv_tasks_per_master = n_send_tasks_per_master + task_index_send[rank]


        first_task = np.empty((1,4), dtype = np.int64)


        if is_task:
            first_tasks_with_dummy = np.zeros((size,all_tasks[0].size), dtype = np.int64)
        comm.Scatter(first_tasks_with_dummy, first_task, root = 0)
        if is_task:

            
            matrix_per_column = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi, inside[:,first_task[0,1]:first_task[0,3]+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
            comm.send((rank,rank-n_masters,matrix_per_column),dest=np.mod(rank,n_masters),tag=tag_done)

        if is_master:
            result_dict = {}
            idx_hard_break = 0
            while n_recv_idx_per_master < n_recv_tasks_per_master:
                status = MPI.Status()
                result_temp = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
                task_rank, task_index_recv, task_result = result_temp

                result_dict[task_index_recv] = task_result

                if task_index_send_per_master < n_send_tasks_per_master:
                    idx_task = idx_tasks_per_master[task_index_send_per_master]
                    task = tasks_per_master[task_index_send_per_master]
                    comm.send((idx_task,task),dest=task_rank,tag=tag_start)
                    task_index_send_per_master = task_index_send_per_master + 1
                else:
                    comm.send((None,None),dest=task_rank,tag=tag_exit)
                task_index_recv = task_index_recv + 1
                n_recv_idx_per_master = n_recv_idx_per_master + 1
                idx_hard_break = idx_hard_break +1
            result_all_dicts = master_comm.gather(result_dict, root = 0)
            if rank == 0:
                merged_dicts = dict(chain.from_iterable(d.items() for d in result_all_dicts))
                items_sorted_all_results = dict(sorted(merged_dicts.items()))
                all_results_list = list(items_sorted_all_results.values())
                D_matrix = np.vstack(all_results_list)
        elif is_task:
            while True:
                status = MPI.Status()
                idx_task, task = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)

                tag = status.Get_tag()
                if tag == tag_exit:
                    break
                elif tag == tag_start:
                    matrix_per_column = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi, inside[:,task[1]:task[3]+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                    comm.send((rank,idx_task,matrix_per_column),dest=status.Get_source(),tag=tag_done)

    if rank != 0:
        D_matrix = np.empty((n_lines, n_coll), dtype = np.complex128)
    
    comm.Barrier()
    comm.Bcast(D_matrix, root=0)
    return D_matrix

def create_D_matrix_with_holes_result (wavenumber, vi_holes, collocation_nodes_holes, inside, alpha):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    # Number of holes+1
    n_holes = vi_holes.shape[0]
    # Number of points from which to calculate the result.
    n_inside = inside.shape[1]
    # Number of collocation nodes per boundary
    n_coll = collocation_nodes_holes.shape[2]
    D_matrix_holes = np.zeros((n_inside, n_holes*n_coll), dtype='complex')
    for i in range(n_holes):
        # Adaptive integration of the matrix from 0 to 1 without the points [alpha, 0.5, 1-alpha] due to singularities.
        D_matrix_holes[:,i*n_coll:(i+1)*n_coll]  = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside, alpha), points = [alpha, 0.5, 1-alpha])[0]
    return D_matrix_holes
    
def create_D_matrix_with_holes_result_MPI (wavenumber, vi_holes, collocation_nodes_holes, inside, alpha):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    # Number of holes+1
    n_holes = vi_holes.shape[0]
    # Number of points from which to calculate the result.
    n_inside = inside.shape[1]
    # Number of collocation nodes per boundary
    n_coll = collocation_nodes_holes.shape[2]
    D_matrix_holes = np.zeros((n_inside, n_holes*n_coll), dtype='complex')
    
    # How many rows must be calculated by a process?
    ave_n_rows_D, res_n_rows_D = divmod(n_inside*n_holes, size)
    # Calculate number of lines that each process has to calculate
    n_lines = np.concatenate((np.repeat(ave_n_rows_D+1, res_n_rows_D), np.repeat(ave_n_rows_D, size-res_n_rows_D)))
    # Caluculate cummulative number of lines the processes have to calculate starting from zero
    cumsum_lines = np.cumsum(n_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_inside)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_inside)

    # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
    start_matrix_partial = start_matrix[rank]
    start_line_partial = start_line[rank]
    end_matrix_partial = end_matrix[rank]
    end_line_partial = end_line[rank]
    
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process
    if start_matrix_partial == end_matrix_partial:
        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[start_matrix_partial], inside[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            # Calculation of the first submatrix
            if i == start_matrix_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside[:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
            # Calculation of the last submatrix
            elif i == end_matrix_partial:
                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside[:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            #Calculation of the submatrices between the first and last.
            else:
                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside, alpha), points = [alpha, 0.5, 1-alpha])[0]
                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
    
    # Buffer to store the entire D-matrix 
    D_matrix_holes = np.empty((n_holes, n_inside, n_coll),dtype=np.complex128)
    # Number of elements per D-matrices in each process.
    sendcounts_matrices = n_lines*n_coll
    # Cumulative number of elements of the D matrices calculated up to the current process.
    displacements_matrices = cumsum_lines[:-1] * n_coll
    # Collect all parts of the D-Matrix and store them in a large
    comm.Gatherv(D_matrix_holes_partial, [D_matrix_holes, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
    # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
    D_matrix_holes = np.concatenate(D_matrix_holes, axis=1)
    # print(f"D_matrix_holes shape is {D_matrix_holes.shape}")
    comm.Bcast(D_matrix_holes, root = 0)
    return D_matrix_holes


def create_D_matrix_with_holes_result_MPI_diff_n (wavenumber, vi_holes, collocation_nodes_holes, inside, alpha):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    # Number of holes+1
    n_holes = len(vi_holes)
    # Number of points from which to calculate the result.
    n_inside = inside.shape[1]

    # Number of collocation nodes per boundary
    n_colls = np.array([collocation_nodes.shape[1] for collocation_nodes in collocation_nodes_holes])
    
    # Weighting of a column: How wide is a column in relation to the total width of the matrix to be calculated?
    weight = n_colls/np.sum(n_colls)
    # How many processes are available for a block (column).
    n_proc_block = weight*size

    start_proc_block = np.zeros(n_holes)
    n_whole_proc_block = np.zeros(n_holes)
    end_proc_block = np.zeros(n_holes)

    for i in range(n_holes):
            if i == 0:
                end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block[i])
            else:
                if end_proc_block[i-1] != 0:
                    start_proc_block_temp = np.round(1 - end_proc_block[i-1],10)
                    if np.round(n_proc_block[i] - start_proc_block_temp,8)>=0 and n_whole_proc_block[i-1]>=0:
                        start_proc_block[i] = start_proc_block_temp
                        n_proc_block_temp = n_proc_block[i] - start_proc_block[i]
                        end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block_temp)
                    elif np.round(n_proc_block[i] + n_whole_proc_block[i-1],8)>=0 and n_whole_proc_block[i-1]<0:
                        start_proc_block[i] = - n_whole_proc_block[i-1]
                        n_proc_block_temp = n_proc_block[i] - start_proc_block[i]
                        end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block_temp)
                    else:
                        end_proc_block[i] = n_proc_block[i]
                        if n_whole_proc_block[i-1]>=0:
                            n_whole_proc_block[i] = -np.round(1- end_proc_block[i-1] - n_proc_block[i],10)
                        else: 
                            n_whole_proc_block[i] = n_whole_proc_block[i-1] + end_proc_block[i]
                else:
                    end_proc_block[i], n_whole_proc_block[i] = np.modf(n_proc_block[i])

    idx_add_lines = np.argwhere(n_whole_proc_block<0).flatten()
    n_whole_proc_block = np.round(n_whole_proc_block).astype(int)
    n_lines_per_proc_block = n_inside/n_proc_block
    start_lines = n_lines_per_proc_block*start_proc_block
    end_lines = n_lines_per_proc_block*end_proc_block
    middle_lines = [np.repeat(n_lines_per_proc_block[i], max(0,n_whole_proc_block[i])) for i in range(n_whole_proc_block.size)]
    all_lines = [np.insert(sub_middle_lines, [0,sub_middle_lines.size], [sub_start_lines, sub_end_lines]).astype(int) for sub_middle_lines, sub_start_lines, sub_end_lines in zip(middle_lines, start_lines, end_lines)]
    sum_lines_per_column = np.array(list(map(lambda ai: np.sum(ai), all_lines)))
    left_lines = n_inside-sum_lines_per_column
    lines_per_proc_per_block = list(map(fill_left_lines, all_lines, left_lines))
    n_transition_lines =  np.array(list(map(lambda lines_per_proc_per_block_begin, lines_per_proc_per_block_end: lines_per_proc_per_block_begin[-1]+lines_per_proc_per_block_end[0], lines_per_proc_per_block[:-1], lines_per_proc_per_block[1:])))
    if idx_add_lines.size != 0:
        n_transition_lines_over_blocks = calculate_n_transition_lines_over_block(idx_add_lines,n_transition_lines)
    else:
        n_transition_lines_over_blocks = np.copy(n_transition_lines)
        

    n_lines_per_proc_without_last = list(map(lambda lines_per_proc_per_block_temp, n_transition_lines_temp: np.append(lines_per_proc_per_block_temp[1:-1], n_transition_lines_temp), lines_per_proc_per_block, n_transition_lines_over_blocks))

    n_lines_per_proc_with_null = np.concatenate([np.concatenate(n_lines_per_proc_without_last), lines_per_proc_per_block[-1][1:]])
    n_lines_per_proc_without_null = n_lines_per_proc_with_null[n_lines_per_proc_with_null!=0]

    cumsum_lines = np.cumsum(n_lines_per_proc_without_null)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_inside)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_inside)

    # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
    start_matrix_partial = start_matrix[rank]
    start_line_partial = start_line[rank]
    end_matrix_partial = end_matrix[rank]
    end_line_partial = end_line[rank]

    matrix_per_column = {key: np.empty((0,n_coll)) for key, n_coll in enumerate(n_colls)}
    
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process
    if start_matrix_partial == end_matrix_partial:
        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[start_matrix_partial], inside[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
        D_matrix_holes_partial_temp_dict = matrix_per_column[start_matrix_partial]
        D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
        matrix_per_column[start_matrix_partial] = D_matrix_holes_partial_dict
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            # Calculation of the first submatrix
            if i == start_matrix_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside[:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
            # Calculation of the last submatrix
            elif i == end_matrix_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside[:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                #D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            #Calculation of the submatrices between the first and last.
            else:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside, alpha), points = [alpha, 0.5, 1-alpha])[0]
                #D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            D_matrix_holes_partial_temp_dict = matrix_per_column[i]
            D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
            matrix_per_column[i] = D_matrix_holes_partial_dict
    
    # Buffer to store the entire D-matrix 
    D_matrix_holes = {key : np.empty((n_inside, n_coll),dtype=D_matrix_holes_partial.dtype) for key, n_coll in enumerate(n_colls)}
    #D_matrix_holes = np.empty((n_holes, n_inside, n_coll),dtype=np.complex128)
    # Number of elements per D-matrices in each process.
    sendcounts_matrices_partial = np.array(list(map(lambda matrix_per_column_temp: matrix_per_column_temp.shape[0]*matrix_per_column_temp.shape[1], matrix_per_column.values())))
    sendcounts_matrices = np.empty((size* n_holes), dtype=sendcounts_matrices_partial.dtype)
    comm.Gather(sendcounts_matrices_partial, sendcounts_matrices, root=0)
    comm.Bcast(sendcounts_matrices, root = 0)
    sendcounts_matrices = sendcounts_matrices.reshape((size, n_holes))
    # Cumulative number of elements of the D matrices calculated up to the current process.
    displacements_matrices = np.insert(np.cumsum(sendcounts_matrices, axis=0),0,0,axis=0)[:-1,:]
    #displacements_matrices = cumsum_lines[:-1] * n_coll
    # Collect all parts of the D-Matrix and store them in a large
    for i in range(n_holes):
        comm.Gatherv(matrix_per_column[i], [D_matrix_holes[i], sendcounts_matrices[:,i], displacements_matrices[:,i], MPI.DOUBLE_COMPLEX], root = 0)
    # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
    D_matrix_holes_list = list(D_matrix_holes.values())
    D_matrix_holes_list_concatenate = np.concatenate(D_matrix_holes_list, axis = 1)
    D_matrix_holes = D_matrix_holes_list_concatenate.reshape((n_inside,np.sum(n_colls)))
    comm.Bcast(D_matrix_holes, root = 0)
    return D_matrix_holes

def calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, start_matrix_partial, start_line_partial, end_matrix_partial,  end_line_partial):
    # Number of holes+1
    n_holes = len(vi_holes)

    # Number of collocation nodes per boundary
    n_colls = np.array([m.shape[1] for m in collocation_nodes_holes])
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process, this means just one phi value (position at circle of Beyn algorithm) is used 
    eps = 1e-6

    time_start_alg = time.time()
    matrix_per_column = {key: np.empty((0,n_coll)) for key, n_coll in enumerate(n_colls)}
    
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process
    if start_matrix_partial == end_matrix_partial:
        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[start_matrix_partial], inside[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha], epsabs=1e-12, epsrel=1e-12, limit=300)[0]
        if np.isnan(D_matrix_holes_partial).any():
            print("I'm nan but I get repaired")
            D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(wavenumber, vi_holes[start_matrix_partial], inside[:,start_line_partial:end_line_partial+1], alpha))[0]
            D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(wavenumber, vi_holes[start_matrix_partial], inside[:,start_line_partial:end_line_partial+1], alpha), points = [0.5])[0]
            D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(wavenumber, vi_holes[start_matrix_partial], inside[:,start_line_partial:end_line_partial+1], alpha))[0]
            D_matrix_holes_partial =  D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3 
        D_matrix_holes_partial_temp_dict = matrix_per_column[start_matrix_partial]
        D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
        matrix_per_column[start_matrix_partial] = D_matrix_holes_partial_dict
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            # Calculation of the first submatrix
            if i == start_matrix_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside[:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]        
                if np.isnan(D_matrix_holes_partial).any():
                    print("I'm nan but I get repaired")
                    D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(wavenumber, vi_holes[i], inside[:,start_line_partial:], alpha))[0]
                    D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(wavenumber, vi_holes[i], inside[:,start_line_partial:], alpha), points = [0.5])[0]
                    D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(wavenumber, vi_holes[i], inside[:,start_line_partial:], alpha))[0]
                    D_matrix_holes_partial =  D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3 
            # Calculation of the last submatrix
            elif i == end_matrix_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside[:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                if np.isnan(D_matrix_holes_partial).any():
                    print("I'm nan but I get repaired")
                    D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(wavenumber, vi_holes[i], inside[:,:end_line_partial+1], alpha))[0]
                    D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(wavenumber, vi_holes[i], inside[:,:end_line_partial+1], alpha), points = [0.5])[0]
                    D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(wavenumber, vi_holes[i], inside[:,:end_line_partial+1], alpha))[0]
                    D_matrix_holes_partial =  D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3 
                #D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            #Calculation of the submatrices between the first and last.
            else:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(wavenumber, vi_holes[i], inside, alpha), points = [alpha, 0.5, 1-alpha])[0]
                if np.isnan(D_matrix_holes_partial).any():
                    print("I'm nan but I get repaired")
                    D_matrix_holes_partial_1 = integrate.quad_vec(create_integrand_double_layer, 0, alpha-eps, args=(wavenumber, vi_holes[i], inside, alpha))[0]
                    D_matrix_holes_partial_2 = integrate.quad_vec(create_integrand_double_layer, alpha+eps, 1-alpha-eps, args=(wavenumber, vi_holes[i], inside, alpha), points = [0.5])[0]
                    D_matrix_holes_partial_3 = integrate.quad_vec(create_integrand_double_layer, 1-alpha+eps, 1, args=(wavenumber, vi_holes[i], inside, alpha))[0]
                    D_matrix_holes_partial =  D_matrix_holes_partial_1 + D_matrix_holes_partial_2 + D_matrix_holes_partial_3 
                #D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            D_matrix_holes_partial_temp_dict = matrix_per_column[i]
            D_matrix_holes_partial_dict = np.concatenate((D_matrix_holes_partial_temp_dict, D_matrix_holes_partial), axis=0)
            matrix_per_column[i] = D_matrix_holes_partial_dict
    time_end_alg = time.time()
    time_diff_alg = time_end_alg- time_start_alg
    return matrix_per_column

def create_D_matrix_with_holes_result_MPI_diff_n_master_workers (wavenumber, vi_holes, collocation_nodes_holes, inside, alpha, n_masters, lines_per_task):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    # Number of holes+1
    n_holes = len(vi_holes)
    # Number of points from which to calculate the result.
    n_inside = inside.shape[1]

    tag_start = 0
    tag_done = 1
    tag_exit = 2

    # Number of collocation nodes per boundary
    n_colls = np.array([collocation_nodes.shape[1] for collocation_nodes in collocation_nodes_holes])
    n_colls_sum = np.sum(n_colls)
    
    n_lines = n_inside*n_holes
    n_lines_per_task = min(lines_per_task, int(n_lines/size))
    if n_lines_per_task != 0:
        n_repeat_full_tasks = int(n_lines/n_lines_per_task)
        n_tasks_rest = n_lines - n_repeat_full_tasks * n_lines_per_task
        all_tasks_lines = np.repeat(n_lines_per_task, n_repeat_full_tasks)
        if n_tasks_rest > 0:
            all_tasks_lines = np.append(all_tasks_lines, n_tasks_rest)
    else: 
        all_tasks_lines = np.repeat(1, n_lines)
    cumsum_lines = np.cumsum(all_tasks_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_inside)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_inside)

    all_tasks = np.array([(start_matrix[i],start_line[i],end_matrix[i],end_line[i]) for i in range(start_matrix.size)])
    n_all_tasks = len(all_tasks)

    if n_all_tasks <= size:
        if rank < n_all_tasks:
            color = 1
        else: 
            color = MPI.UNDEFINED
        sub_comm = comm.Split(color=color, key=rank)
        if sub_comm != MPI.COMM_NULL:
            sub_rank = sub_comm.Get_rank()
            sub_size = sub_comm.Get_size()
            sub_tasks = np.empty((1,4), dtype = all_tasks.dtype)
            sub_comm.Scatter(all_tasks, sub_tasks, root = 0)
            matrix_per_column = {sub_rank: calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, *sub_tasks[0])}
            D_matrix_holes_dict = sub_comm.gather(matrix_per_column,root=0)
            if rank == 0:
                merged_dicts = dict(chain.from_iterable(d.items() for d in D_matrix_holes_dict))
                items_sorted_all_results = dict(sorted(merged_dicts.items()))
                all_results_array = np.array(list(items_sorted_all_results.values()))

    if n_all_tasks > size:
        if n_masters >= size:
            print("It's not allowed that the amout of master processes is equal or bigger than the total amount of processes", flush=True)
        if rank < n_masters:
            is_master = True
            is_task = False
        else :
            is_master = False
            is_task = True
        if is_master:
            color = 1
        else: 
            color = MPI.UNDEFINED
        master_comm = comm.Split(color=color, key=rank)


        if is_master:
            task_first_index_send = size-n_masters
            first_ave, first_rest = divmod(task_first_index_send,n_masters)
            task_index_send = np.concatenate((np.repeat(first_ave+1,first_rest), np.repeat(first_ave, n_masters-first_rest)))
            task_index_send_per_master = 0
            task_index_recv = 0
            n_recv_idx_per_master = 0
            
            # Tasks to scatter previously
            first_tasks = all_tasks[:task_first_index_send]
            first_tasks_with_dummy = np.concatenate((np.zeros((n_masters,all_tasks[0].size),dtype=np.int64),first_tasks), axis=0, dtype = np.int64)
            
            # Tasks to send to the workers when masters get results from scattering
            rest_tasks = all_tasks[task_first_index_send:]
            # Task to send to the workeres seperated to the masters
            tasks_splitted = [rest_tasks[i::n_masters] for i in range(n_masters)]
            # indices of all task which have to be send by a master
            idx_tasks_all_send = np.arange(task_first_index_send, n_all_tasks)

            # indices of all task which have to be send by each master (seperated for each master)
            idx_tasks_splitted = [idx_tasks_all_send[i::n_masters] for i in range(n_masters)]
            # indices of all task which have to be send by the current master
            idx_tasks_per_master = idx_tasks_splitted[rank]
            # Tasks which have to be send by the current master
            tasks_per_master = tasks_splitted[rank]
            # Number of tasks which have to be send by the current master
            n_send_tasks_per_master = len(tasks_per_master)
            # Number of tasks which the master will receive from scatter plus the number of tasks, the current master will send
            n_recv_tasks_per_master = n_send_tasks_per_master + task_index_send[rank]

        
        first_task = np.empty((1,4), dtype = np.int64)
        if is_task: 
            first_tasks_with_dummy = np.zeros((size,all_tasks[0].size), dtype = np.int64)
        comm.Scatter(first_tasks_with_dummy, first_task, root = 0)
        if is_task:
            matrix_per_column = calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, *first_task[0])
            comm.send((rank,rank-n_masters,matrix_per_column),dest=np.mod(rank,n_masters),tag=tag_done)

        if is_master:
            left_over_idx = np.concatenate((np.arange(1,95*4,4), np.array(idx_tasks_per_master)))
            result_dict = {}
            idx_hard_break = 0
            while n_recv_idx_per_master < n_recv_tasks_per_master:
                status = MPI.Status()
                result_temp = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
                task_rank, task_index_recv, task_result = result_temp
                left_over_idx = left_over_idx[left_over_idx != task_index_recv]
                result_dict[task_index_recv] = task_result

                if task_index_send_per_master < n_send_tasks_per_master:
                    idx_task = idx_tasks_per_master[task_index_send_per_master]
                    task = tasks_per_master[task_index_send_per_master]
                    comm.send((idx_task,task),dest=task_rank,tag=tag_start)
                    task_index_send_per_master = task_index_send_per_master + 1
                else:
                    comm.send((None,None),dest=task_rank,tag=tag_exit)
                task_index_recv = task_index_recv + 1
                n_recv_idx_per_master = n_recv_idx_per_master + 1
                idx_hard_break = idx_hard_break +1
            
            result_all_dicts = master_comm.gather(result_dict, root = 0)
            if rank == 0:
                merged_dicts = dict(chain.from_iterable(d.items() for d in result_all_dicts))
                items_sorted_all_results = dict(sorted(merged_dicts.items()))
                all_results_array = np.array(list(items_sorted_all_results.values()))

        elif is_task:
            while True:
                status = MPI.Status()
                idx_task, task = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)

                tag = status.Get_tag()
                if tag == tag_exit:
                    break
                elif tag == tag_start:

                    matrix_per_column = calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, *task)
                    comm.send((rank,idx_task,matrix_per_column),dest=status.Get_source(),tag=tag_done)
    if rank == 0: 
        columns = []
        for key in range(n_holes):
            column_stacked = np.vstack([all_results_array[arr][key] for arr in range(all_results_array.size)])
            columns.append(column_stacked)
        D_matrix_holes = np.hstack(columns)
    if rank != 0:
        D_matrix_holes = np.empty((n_inside,n_colls_sum), dtype = np.complex128)
    D_matrix_holes = np.ascontiguousarray(D_matrix_holes,dtype = np.complex128)
    node_id = int(os.environ["SLURM_NODEID"])
    node_comm = comm.Split(color=node_id, key = rank)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()

    root_color = 1 if node_rank == 0 else MPI.UNDEFINED
    root_comm = comm.Split(color=root_color, key=rank)

    if node_rank == 0:
        root_comm.Bcast(D_matrix_holes, root=0)
    node_comm.Bcast(D_matrix_holes, root=0)
    return D_matrix_holes

def create_D_matrix_dense_points_at_boundary_with_holes_result_MPI_diff_n_master_workers (wavenumber, vi_holes, collocation_nodes_holes, inside, alpha, n_masters, lines_per_task):
    """
    Routine to create the D-matrix containing in each element the boundary double layer evaluated at the points: "inside"
    at the j-th piece of the boundary for the first, second and third generalized Lagrange functions.
    This method must be used for domains with holes.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    inside : numpy.ndarray
        Numpy array with x-axis elements of the points, for which the result shall be calculated, in the first row
        and y-axis elements in the second row.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.

    Returns
    -------
    D_matrix_holes : numpy.ndarray
        Numpy array containing all integrals of the double layer operator for q holes.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator

    # Number of holes+1
    n_holes = len(vi_holes)
    # Number of points from which to calculate the result.
    n_inside = inside.shape[1]

    tag_start = 0
    tag_done = 1
    tag_exit = 2

    # Number of collocation nodes per boundary
    n_colls = np.array([collocation_nodes.shape[1] for collocation_nodes in collocation_nodes_holes])
    n_colls_sum = np.sum(n_colls)
    
    n_lines = n_inside*n_holes
    n_lines_per_task = min(lines_per_task, int(n_lines/size))
    if n_lines_per_task != 0:
        n_repeat_full_tasks = int(n_lines/n_lines_per_task)
        n_tasks_rest = n_lines - n_repeat_full_tasks * n_lines_per_task
        all_tasks_lines = np.repeat(n_lines_per_task, n_repeat_full_tasks)
        if n_tasks_rest > 0:
            all_tasks_lines = np.append(all_tasks_lines, n_tasks_rest)
    else: 
        all_tasks_lines = np.repeat(1, n_lines)
    cumsum_lines = np.cumsum(all_tasks_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_inside)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_inside)

    all_tasks = np.array([(start_matrix[i],start_line[i],end_matrix[i],end_line[i]) for i in range(start_matrix.size)])
    n_all_tasks = len(all_tasks)
    

    if n_all_tasks <= size:
        if rank < n_all_tasks:
            color = 1
        else: 
            color = MPI.UNDEFINED
        sub_comm = comm.Split(color=color, key=rank)
        if sub_comm != MPI.COMM_NULL:
            sub_rank = sub_comm.Get_rank()
            sub_size = sub_comm.Get_size()
            sub_tasks = np.empty((1,4), dtype = all_tasks.dtype)
            sub_comm.Scatter(all_tasks, sub_tasks, root = 0)
            matrix_per_column = {sub_rank: calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, *sub_tasks[0])}
            D_matrix_holes_dict = sub_comm.gather(matrix_per_column,root=0)
            if rank == 0:
                merged_dicts = dict(chain.from_iterable(d.items() for d in D_matrix_holes_dict))
                items_sorted_all_results = dict(sorted(merged_dicts.items()))
                all_results_array = np.array(list(items_sorted_all_results.values()))



    if n_all_tasks > size:
        if n_masters >= size:
            print("It's not allowed that the amout of master processes is equal or bigger than the total amount of processes", flush=True)
        if rank < n_masters:
            is_master = True
            is_task = False
        else :
            is_master = False
            is_task = True
        if is_master:
            color = 1
        else: 
            color = MPI.UNDEFINED
        master_comm = comm.Split(color=color, key=rank)


        if is_master:
            task_first_index_send = size-n_masters
            first_ave, first_rest = divmod(task_first_index_send,n_masters)
            task_index_send = np.concatenate((np.repeat(first_ave+1,first_rest), np.repeat(first_ave, n_masters-first_rest)))
            task_index_send_per_master = 0
            task_index_recv = 0
            n_recv_idx_per_master = 0
            
            # Tasks to scatter previously
            first_tasks = all_tasks[:task_first_index_send]
            first_tasks_with_dummy = np.concatenate((np.zeros((n_masters,all_tasks[0].size),dtype=np.int64),first_tasks), axis=0, dtype = np.int64)
            
            # Tasks to send to the workers when masters get results from scattering
            rest_tasks = all_tasks[task_first_index_send:]
            # Task to send to the workeres seperated to the masters
            tasks_splitted = [rest_tasks[i::n_masters] for i in range(n_masters)]
            # indices of all task which have to be send by a master
            idx_tasks_all_send = np.arange(task_first_index_send, n_all_tasks)

            # indices of all task which have to be send by each master (seperated for each master)
            idx_tasks_splitted = [idx_tasks_all_send[i::n_masters] for i in range(n_masters)]
            # indices of all task which have to be send by the current master
            idx_tasks_per_master = idx_tasks_splitted[rank]
            # Tasks which have to be send by the current master
            tasks_per_master = tasks_splitted[rank]
            # Number of tasks which have to be send by the current master
            n_send_tasks_per_master = len(tasks_per_master)
            # Number of tasks which the master will receive from scatter plus the number of tasks, the current master will send
            n_recv_tasks_per_master = n_send_tasks_per_master + task_index_send[rank]

        
        first_task = np.empty((1,4), dtype = np.int64)
        if is_task: 
            first_tasks_with_dummy = np.zeros((size,all_tasks[0].size), dtype = np.int64)
        comm.Scatter(first_tasks_with_dummy, first_task, root = 0)
        if is_task:
            matrix_per_column = calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, *first_task[0])

            comm.send((rank,rank-n_masters,matrix_per_column),dest=np.mod(rank,n_masters),tag=tag_done)

        if is_master:
            left_over_idx = np.concatenate((np.arange(1,95*4,4), np.array(idx_tasks_per_master)))
            result_dict = {}
            idx_hard_break = 0
            while n_recv_idx_per_master < n_recv_tasks_per_master:
                status = MPI.Status()
                result_temp = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
                task_rank, task_index_recv, task_result = result_temp
                left_over_idx = left_over_idx[left_over_idx != task_index_recv]

                result_dict[task_index_recv] = task_result

                if task_index_send_per_master < n_send_tasks_per_master:
                    idx_task = idx_tasks_per_master[task_index_send_per_master]
                    task = tasks_per_master[task_index_send_per_master]
                    comm.send((idx_task,task),dest=task_rank,tag=tag_start)
                    task_index_send_per_master = task_index_send_per_master + 1
                else:
                    comm.send((None,None),dest=task_rank,tag=tag_exit)
                task_index_recv = task_index_recv + 1
                n_recv_idx_per_master = n_recv_idx_per_master + 1
                idx_hard_break = idx_hard_break +1

            
            result_all_dicts = master_comm.gather(result_dict, root = 0)
            if rank == 0:
                merged_dicts = dict(chain.from_iterable(d.items() for d in result_all_dicts))
                items_sorted_all_results = dict(sorted(merged_dicts.items()))
                all_results_array = np.array(list(items_sorted_all_results.values()))

        elif is_task:
            while True:
                status = MPI.Status()
                idx_task, task = comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG, status=status)

                tag = status.Get_tag()
                if tag == tag_exit:
                    break
                elif tag == tag_start:

                    matrix_per_column = calculate_D_matrix_result_partial(vi_holes, collocation_nodes_holes, inside, wavenumber, alpha, *task)
                    comm.send((rank,idx_task,matrix_per_column),dest=status.Get_source(),tag=tag_done)
    if rank == 0: 
        columns = []
        for key in range(n_holes):
            column_stacked = np.vstack([all_results_array[arr][key] for arr in range(all_results_array.size)])
            columns.append(column_stacked)
        D_matrix_holes = np.hstack(columns)
    if rank != 0:
        D_matrix_holes = np.empty((n_inside,n_colls_sum), dtype = np.complex128)
    D_matrix_holes = np.ascontiguousarray(D_matrix_holes,dtype = np.complex128)
    node_id = int(os.environ["SLURM_NODEID"])
    node_comm = comm.Split(color=node_id, key = rank)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()

    root_color = 1 if node_rank == 0 else MPI.UNDEFINED
    root_comm = comm.Split(color=root_color, key=rank)

    if node_rank == 0:
        root_comm.Bcast(D_matrix_holes, root=0)
    node_comm.Bcast(D_matrix_holes, root=0)
    return D_matrix_holes

def ray_tracing(boundary, x_start, x_end, x_steps, y_start, y_end, y_steps):
    """
    Routine to find all the points of a square on an equidistant grid, that lie within the given domain.

    Parameters
    ----------
    boundary : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundary.
    x_start : float
        Start value of a square in x-direction.
    x_end : float
        End value of a square in x-direction.
    x_steps : int
        Number of points in x-direction that should be checked to be inside the given domain.
    y_start : float
        Start value of a square in y-direction.
    y_end : float
        End value of a square in y-direction.
    y_steps : int
        Number of points in y-direction that should be checked to be inside the given domain.

    Returns
    -------
    xy_inside : numpy.ndarray
        Numpy array containing all points that lie inside of the given domain.
    """
    # Create a path for the boundary of the domain
    p = path.Path(boundary.T)
    # Create the mesh of the square to be checked, whether they are inside or outside of the domain.
    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    mesh = np.reshape(mesh,(x_steps*y_steps,2))
    # Check whether the mesh is inside or outside of the boundary of the domain.
    inside = p.contains_points(mesh)
    inside = np.reshape(inside, (y_steps, x_steps))
    xy_inside = np.stack((xv[inside], yv[inside]), axis=0)
    return xy_inside


def ray_tracing_with_holes(boundary, x_start, x_end, x_steps, y_start, y_end, y_steps):
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
    holes = hole.any(axis=0) # Any function is like logical_or function. Array is true wherever a hole is.
    # Calculate which points are inside the given domain (holes are outside -> False)
    inside = np.logical_xor(inside_with_hole, holes)
    inside = np.reshape(inside, (y_steps, x_steps))
    xy_inside = np.stack((xv[inside], yv[inside]), axis=0)
    return xy_inside

def ray_tracing_with_holes_diff_n(boundary, x_start, x_end, x_steps, y_start, y_end, y_steps, radius = 0.0):
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
    p2 = [path.Path(boundary[i].T) for i in range(1, boundary.size)]
    # Create the mesh of the square to be checked, whether they are inside or outside of the domain with holes.
    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    mesh = np.reshape(mesh,(x_steps*y_steps,2))
    # Check whether the mesh is inside or outside of the outer boundary of the domain (without holes).
    inside_with_hole = p1.contains_points(mesh, radius=radius)
    # Check to see if the mesh is inside or outside the holes.
    hole = np.array([p2[i].contains_points(mesh, radius=radius) for i in range(boundary.size-1)])
    holes = hole.any(axis=0) # Any function is like logical_or function. Array is true wherever a hole is.
    # Calculate which points are inside the given domain (holes are outside -> False)
    inside = np.logical_xor(inside_with_hole, holes)
    inside = np.reshape(inside, (y_steps, x_steps))
    xy_inside = np.stack((xv[inside], yv[inside]), axis=0)
    return xy_inside

def calculate_normals(points):
    tangents = np.diff(points, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:,None]
    return np.column_stack([-tangents[:,1], tangents[:,0]])

def calculate_band_points_around_path(points, width = 0.2, n_per_points=5):
    normal_points = calculate_normals(points)

    band = []

    for i in range(len(points) -1):
        point = points[i]
        normal_point = normal_points[i]

        for j in range(n_per_points):
            offset = np.random.uniform(-width, width)
            band.append(point+offset*normal_point)

    return np.array(band)

def ray_tracing_dense_points_at_boundary_with_holes_diff_n(boundary, x_start, x_end, x_steps, y_start, y_end, y_steps,width = 0.2, n_per_points=5):
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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    # Create a path for the outer boundary of the domain
    p1 = path.Path(boundary[0].T)
    # Create a path for the boundary of the holes
    p2 = [path.Path(boundary[i].T) for i in range(1, boundary.size)]
    # Create the mesh of the square to be checked, whether they are inside or outside of the domain with holes.
    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    mesh = np.reshape(mesh,(x_steps*y_steps,2))

    additional_points1 = p1.interpolated(200).vertices
    band1 = calculate_band_points_around_path(additional_points1, width, n_per_points)
    
    mesh1 = np.vstack((mesh,band1))

    additional_points2 = []
    band2 = []
    for i in range(len(p2)):
        p2_temp = p2[i].interpolated(200).vertices
        additional_points2.append(p2_temp)
        band2_temp = calculate_band_points_around_path(p2_temp, width, n_per_points)
        band2.append(band2_temp)

    mesh2 = np.vstack(band2)

    mesh_all = np.vstack((mesh1,mesh2))
    comm.Bcast(mesh_all, root=0) 

    # Check whether the mesh is inside or outside of the outer boundary of the domain (without holes).
    #inside_with_hole = p1.contains_points(mesh, radius = -10e-3)
    inside_with_hole = p1.contains_points(mesh_all)
    # Check to see if the mesh is inside or outside the holes.
    hole = np.array([p2[i].contains_points(mesh_all) for i in range(boundary.size-1)])
    holes = hole.any(axis=0) # Any function is like logical_or function. Array is true wherever a hole is.
    # Calculate which points are inside the given domain (holes are outside -> False)
    inside = np.logical_xor(inside_with_hole, holes)
    xy_inside = mesh_all[inside]
    xy_inside_T = xy_inside.T
    mesh_inside = mesh_all[inside]
    return mesh_inside, xy_inside_T


def calculate_double_layer_inside(wavenumber, vi, collocation_nodes, alpha, eigenvectors, x_start, x_end, x_steps, y_start, y_end, y_steps):
    """
    Routine to calculate the double layer potential inside of the given domain and inside a square, defined by the parameters.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpointsof the approximated boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues. 
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
        Numpy array containing all points that lie inside of the given domain.
    sol : numpy.ndarray
        Numpy array containing the double layer potential at the points xy_inside.
    """
    # Calculate all the points that are inside the boundary of the given domain and the square.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    xy_inside = ray_tracing(vi, x_start, x_end, x_steps, y_start, y_end, y_steps)
    # Add the first column to the end of vi
    vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
    # Calculate the D-matrix at the points xy_inside
    # D_matrix, eps = create_D_matrix(wavenumber, vi_append,  xy_inside, alpha)
    D_matrix = create_D_matrix_result_MPI(wavenumber, vi_append, collocation_nodes, xy_inside, alpha)
    # Calculate the double layer potential at the points xy_inside
    sol = np.dot(D_matrix, eigenvectors)
    return xy_inside, sol


def calculate_double_layer_inside_master_workers(wavenumber, vi, collocation_nodes, alpha, eigenvectors, x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task):
    """
    Routine to calculate the double layer potential inside of the given domain and inside a square, defined by the parameters.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi : numpy.ndarray
        Array containing the vertices and midpointsof the approximated boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    eigenvectors : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues. 
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
        Numpy array containing all points that lie inside of the given domain.
    sol : numpy.ndarray
        Numpy array containing the double layer potential at the points xy_inside.
    """
    # Calculate all the points that are inside the boundary of the given domain and the square.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    size = comm.Get_size() # Number of processes in the communicator
    xy_inside = ray_tracing(vi, x_start, x_end, x_steps, y_start, y_end, y_steps)
    # Add the first column to the end of vi
    vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
    # Calculate the D-matrix at the points xy_inside
    # D_matrix, eps = create_D_matrix(wavenumber, vi_append,  xy_inside, alpha)
    D_matrix = create_D_matrix_result_MPI_master_workers(wavenumber, vi_append, collocation_nodes, xy_inside, alpha, n_masters, lines_per_task)
    # Calculate the double layer potential at the points xy_inside
    sol = np.dot(D_matrix, eigenvectors)
    return xy_inside, sol


def calculate_double_layer_inside_with_holes(wavenumber, vi_holes, collocation_nodes_holes, alpha, eigenvectors_holes, x_start, x_end, x_steps, y_start, y_end, y_steps):
    """
    Routine to calculate the double layer potential inside of the given domain (not in holes) and inside a square,
    defined by the parameters.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    eigenvectors_holes : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues,
        here the wavenumber. 
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
    sol : numpy.ndarray
        Numpy array containing the double layer potential at the points xy_inside.
    """
    # Calculate all the points that are inside the boundary of the given domain (without holes) and the square.
    xy_inside = ray_tracing_with_holes(vi_holes, x_start, x_end, x_steps, y_start, y_end, y_steps)
    #Add the first column to the end of vi
    vi_holes_append = np.concatenate((vi_holes, vi_holes[:,:,0].reshape(vi_holes.shape[0],2,1)), axis=2)
    # Calculate the D-matrix at the points xy_inside
    D_matrix = create_D_matrix_with_holes_result_MPI (wavenumber, vi_holes_append, collocation_nodes_holes, xy_inside, alpha)
    # Calculate the double layer potential at the points xy_inside
    sol = np.dot(D_matrix, eigenvectors_holes)
    return xy_inside, sol

def calculate_double_layer_inside_with_holes_diff_n(wavenumber, vi_holes, collocation_nodes_holes, alpha, eigenvectors_holes, x_start, x_end, x_steps, y_start, y_end, y_steps, radius=0.0):
    """
    Routine to calculate the double layer potential inside of the given domain (not in holes) and inside a square,
    defined by the parameters.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    eigenvectors_holes : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues,
        here the wavenumber. 
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
    sol : numpy.ndarray
        Numpy array containing the double layer potential at the points xy_inside.
    """
    # Calculate all the points that are inside the boundary of the given domain (without holes) and the square.
    xy_inside = ray_tracing_with_holes_diff_n(vi_holes, x_start, x_end, x_steps, y_start, y_end, y_steps, radius)
    #Add the first column to the end of vi
    vi_holes_append = list(map(lambda vi_temp: np.concatenate((vi_temp, vi_temp[:,0].reshape(2,1)),axis=1), vi_holes))
    #vi_holes_append = np.concatenate((vi_holes, vi_holes[:,:,0].reshape(vi_holes.shape[0],2,1)), axis=2)
    # Calculate the D-matrix at the points xy_inside
    D_matrix = create_D_matrix_with_holes_result_MPI_diff_n (wavenumber, vi_holes_append, collocation_nodes_holes, xy_inside, alpha)
    # Calculate the double layer potential at the points xy_inside
    sol = np.dot(D_matrix, eigenvectors_holes)
    return xy_inside, sol

def calculate_double_layer_inside_with_holes_diff_n_master_workers(wavenumber, vi_holes, collocation_nodes_holes, alpha, eigenvectors_holes, x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task, radius=0.0):
    """
    Routine to calculate the double layer potential inside of the given domain (not in holes) and inside a square,
    defined by the parameters.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    eigenvectors_holes : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues,
        here the wavenumber. 
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
    sol : numpy.ndarray
        Numpy array containing the double layer potential at the points xy_inside.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    # Calculate all the points that are inside the boundary of the given domain (without holes) and the square.
    xy_inside = ray_tracing_with_holes_diff_n(vi_holes, x_start, x_end, x_steps, y_start, y_end, y_steps, radius)
    #Add the first column to the end of vi
    vi_holes_append = list(map(lambda vi_temp: np.concatenate((vi_temp, vi_temp[:,0].reshape(2,1)),axis=1), vi_holes))
    # Calculate the D-matrix at the points xy_inside
    D_matrix = create_D_matrix_with_holes_result_MPI_diff_n_master_workers (wavenumber, vi_holes_append, collocation_nodes_holes, xy_inside, alpha, n_masters, lines_per_task)
    # Calculate the double layer potential at the points xy_inside
    sol = np.dot(D_matrix, eigenvectors_holes)
    return xy_inside, sol


def calculate_double_layer_inside_with_holes_dense_points_at_boundary_diff_n_master_workers(wavenumber, vi_holes, collocation_nodes_holes, alpha, eigenvectors_holes, x_start, x_end, x_steps, y_start, y_end, y_steps, n_masters, lines_per_task,width = 0.2, n_per_points=5):
    """
    Routine to calculate the double layer potential inside of the given domain (not in holes) and inside a square,
    defined by the parameters.

    Parameters
    ----------
    wavenumber : float
        Wavenumber of the Helmholtz equation. The squared wavenumber is the eigenvalue of the Dirichlet problem.
    vi_holes : numpy.ndarray
        Array containing the vertices and midpoints for each of the approximated boundaries.
    collocation_nodes_holes : numpy.ndarray
        Numpy array with x-axis elements of the collocation nodes in the first row
        and y-axis elements in the second row for each boundary.
    alpha : float
        A value between 0 and 1/2 that ensures that the collocation nodes are always lying inside the pieces of the boundary.
        Choosing an optimal alpha can improve the overall convergence rate.
    eigenvectors_holes : numpy.ndarray
        Numpy array containing all eigenvectors of the nonlinear eigenvalue problem that belong to the given eigenvalues,
        here the wavenumber. 
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
    sol : numpy.ndarray
        Numpy array containing the double layer potential at the points xy_inside.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Calling process of the communicator
    # Calculate all the points that are inside the boundary of the given domain (without holes) and the square.
    mesh, xy_inside = ray_tracing_dense_points_at_boundary_with_holes_diff_n(vi_holes, x_start, x_end, x_steps, y_start, y_end, y_steps,width, n_per_points)
    #Add the first column to the end of vi
    vi_holes_append = list(map(lambda vi_temp: np.concatenate((vi_temp, vi_temp[:,0].reshape(2,1)),axis=1), vi_holes))
    # Calculate the D-matrix at the points xy_inside
    D_matrix = create_D_matrix_dense_points_at_boundary_with_holes_result_MPI_diff_n_master_workers (wavenumber, vi_holes_append, collocation_nodes_holes, xy_inside, alpha, n_masters, lines_per_task)
    # Calculate the double layer potential at the points xy_inside
    sol = np.dot(D_matrix, eigenvectors_holes)
    return mesh, xy_inside, sol
