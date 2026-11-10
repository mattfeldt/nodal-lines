#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy
import scipy.special as sp
from scipy import integrate
from matplotlib import path
from mpi4py import MPI
import pandas as pd
import time


# In[4]:

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


# In[5]:


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


# In[6]:


def calculate_reflection(vi):
    return np.flip(vi, 1)


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


def create_collocation_nodes(vi, alpha):
    """
    Routine to create the collocation nodes along the entire approximated boundary.
    The density function must be calculated at these points.
    For each face of the boundary, three collocation nodes are computed at the points
    q_1 = \alpha, q_2 = 1/2 and q_3 = 1-\alpha, where 0 < \alpha < 1/2.

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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:

def create_integrand_double_layer (s, wavenumber, vi, collocation_nodes, alpha): 
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

    Returns
    -------
    integrand : numpy.ndarray
        Numpy array containing all integrands of the double layer operator.
    """
    
    # Calculation of the exterior normal without the Jacobian
    external_normal = create_external_normals(vi[:,:-1], s)
    # Add the first column to the end of vi
    #vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
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

    # Calculate integrands without the main diagonal
    #integrand1 = np.where((x_indices != y_indices) & ((y_indices % 3) == 0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[0], 0)
    #integrand2 = np.where((x_indices != y_indices) & ((y_indices % 3) == 1), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[1], 0)
    #integrand3 = np.where((x_indices != y_indices) & ((y_indices % 3) == 2), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[2], 0)
    
    # Calculate integrands for the first collocation node of each boundary piece
    integrand1 = np.where(((y_indices % 3) == 0), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[0], 0)
    # Calculate integrands for the second collocation node of each boundary piece
    integrand2 = np.where(((y_indices % 3) == 1), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[1], 0)
    # Calculate integrands for the third collocation node of each boundary piece
    integrand3 = np.where(((y_indices % 3) == 2), 1j*wavenumber*sp.hankel1(1,wavenumber*r_big)/(4*r_big)*(a_big*external_normal_big[0] + b_big*external_normal_big[1])*generalized_lagrange_functions[2], 0)
    integrand = integrand1 + integrand2 + integrand3
    return integrand

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


# In[22]:


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
    n_coll = collocation_nodes_holes.shape[2] # number of collocation-nodes at each boundary
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


# In[24]:


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


# In[25]:


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


# In[26]:


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


# In[27]:


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


# In[28]:


def beyn_algorithm (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps): 
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain without holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
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


# In[29]:


def beyn_algorithm_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
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
    # print(f"n_coll is {n_coll}")

    vi = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
        
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # How many rows must be calculated by a process?
    ave_n_rows_D, res_n_rows_D = divmod(N*n_coll, size)
    # Calculate number of lines that each process has to calculate
    n_lines = np.concatenate((np.repeat(ave_n_rows_D+1, res_n_rows_D), np.repeat(ave_n_rows_D, size-res_n_rows_D)))
    # Caluculate cummulative number of lines the processes have to calculate starting from zero
    cumsum_lines = np.cumsum(n_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # print(f"cumsum_lines is {cumsum_lines}")
    #For phi
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_coll)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_coll)
    # print(f"start_matrix is {start_matrix}")
    # print(f"end_matrix is {end_matrix}")
    # print(f"start_line is {start_line}")
    # print(f"end_line is {end_line}")
    #For hole
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_hole_matrix, start_hole_line = divmod(cumsum_lines[:-1], n_coll)
    # print(f"start_hole_matrix is {start_hole_matrix}")
    # print(f"start_hole_line is {start_hole_line}")
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_hole_matrix, end_hole_line = divmod(cumsum_lines[1:]-1, n_coll)
    # print(f"end_hole_matrix is {end_hole_matrix}")
    # print(f"end_hole_line is {end_hole_line}")
    # start_hole_in_phi_matrix = np.mod(start_hole_matrix, n_holes)
    # end_hole_in_phi_matrix = np.mod(end_hole_matrix, n_holes)
    # # in Submatrix phi, submatrix in column and row (holes)
    # print(f"start_hole_in_phi_matrix is {start_hole_in_phi_matrix}")
    # print(f"start_hole_in_phi_matrix is {type(start_hole_in_phi_matrix)}")
    # print(f"end_hole_in_phi_matrix is {end_hole_in_phi_matrix}")
    # print(n_holes)
    # start_column_matrix, start_row_matrix = np.mod(start_hole_in_phi_matrix, n_holes)
    # end_column_matrix, end_row_matrix = np.mod(end_hole_in_phi_matrix, n_holes)

    start_row_matrix, start_column_matrix = divmod(np.mod(start_hole_matrix, 1),1)
    end_row_matrix, end_column_matrix = divmod(np.mod(end_hole_matrix, 1),1)
    # in Submatrix phi, submatrix in column and row (holes)
    # print(f"start_row_matrix is {start_row_matrix}")
    # print(f"start_column_matrix is {start_column_matrix}")
    # print(f"end_row_matrix is {end_row_matrix}")
    # print(f"end_column_matrix is {end_column_matrix}")
    #start_column_matrix, start_row_matrix = np.mod(start_hole_in_phi_matrix, n_holes)
    #end_column_matrix, end_row_matrix = np.mod(end_hole_in_phi_matrix, n_holes)
    
    # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
    # start_matrix_partial = start_matrix[rank]
    # start_column_partial = start_column_matrix[rank]
    # start_row_partial = start_row_matrix[rank]
    # start_line_partial = start_hole_line[rank]
    # end_matrix_partial = end_matrix[rank]
    # end_column_partial = end_column_matrix[rank]
    # end_row_partial = end_row_matrix[rank]
    # end_line_partial = end_hole_line[rank]
    start_matrix_partial = start_matrix[rank]
    # start_row_partial = start_column_matrix[rank]
    # start_column_partial = start_row_matrix[rank]
    start_line_partial = start_hole_line[rank]
    end_matrix_partial = end_matrix[rank]
    # end_row_partial = end_column_matrix[rank]
    # end_column_partial = end_row_matrix[rank]
    end_line_partial = end_hole_line[rank]
    if rank == 1:
        print(f"start_matrix_partial is {start_matrix_partial}")
        # print(f"start_column_partial is {start_column_partial}")
        # print(f"start_row_partial is {start_row_partial}")
        print(f"start_line_partial is {start_line_partial}")
        print(f"end_matrix_partial is {end_matrix_partial}")
        # print(f"end_column_partial is {end_column_partial}")
        # print(f"end_row_partial is {end_row_partial}")
        print(f"end_line_partial is {end_line_partial}")
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process

    if start_matrix_partial == end_matrix_partial:
        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi, collocation_nodes[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        # D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            if i == start_matrix_partial:
                # Calculation of the first submatrix
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
            #Calculation of the submatrices between the first and last.
            elif i == end_matrix_partial:
                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
            else:
                D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[i], vi, collocation_nodes[:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                
    #D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi, collocation_nodes[:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
    
    # Buffer to store the entire D-matrix 
    D_matrix_holes = np.empty((N, n_coll, n_coll),dtype=np.complex128)
    # Number of elements per D-matrices in each process.
    sendcounts_matrices = n_lines*n_coll
    # Cumulative number of elements of the D matrices calculated up to the current process.
    displacements_matrices = cumsum_lines[:-1] * n_coll
    # Collect all parts of the D-Matrix and store them in a large
    # print(f"sendcounts_matrices is {sendcounts_matrices}")
    # print(f"displacements_matrices is {displacements_matrices}")
    # print(f"Rank {rank}: D_matrix_holes_partial is {D_matrix_holes_partial.shape}")
    # print(f"D_matrix_holes is {D_matrix_holes.shape}")
    # if rank == 0:
    #     print(f"D_matrix_holes_partial is {D_matrix_holes_partial}")
    comm.Gatherv(D_matrix_holes_partial, [D_matrix_holes, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
    # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
    D_matrix_holes_conc = np.concatenate(D_matrix_holes, axis = 1)
    # if rank == 0:
    #     print(f"D_matrix_holes_conc is {D_matrix_holes_conc}")
    #     print(f"D_matrix_holes_conc shape is {D_matrix_holes_conc.shape}")
    # print(f"D_matrix_holes_conc shape is {D_matrix_holes_conc.shape}")
    M_matrix_holes = D_matrix_holes_conc - np.tile(np.identity(D_matrix_holes_conc[:,0].size),N)/2 
    M_matrix_holes_3d = np.reshape(M_matrix_holes,(n_coll,N,n_coll)).transpose(1,0,2)
    # print(f"D_matrix_holes_3d shape is {M_matrix_holes_3d.shape}")

    if rank == 0:
        print("First step is done")
    if rank == 0: 
        M_matrices_inv = np.linalg.inv(M_matrix_holes_3d)
        print("Matrix_inv done")
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
        print("SVD done")
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
        #print(f"Eigenvalues are {eigenvalues}")
        #print(f"Shape of Eigenvalues is {eigenvalues.shape} and shape of eigenvectors is {eigenvectors_shape}")
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
        #eigenvalues = np.empty([1], dtype = np.complex128)
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


# In[35]:


def beyn_algorithm_with_holes_MPI_faster (vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
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
    
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    #V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    V_hat = np.arange(m*l).reshape(m,l)+1j*np.arange(m*l).reshape(m,l)
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    #M_matrices = np.array([M_test (phi_r) for phi_r in phi])
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    #start_time = MPI.Wtime()
    if rank == 0:
        # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
        phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
        #print(f"I'm {rank + 1} Phi is {phi} and has type {type(phi)}")
        ave_scatter_phi, res_scatter_phi = divmod(phi.shape[0], size)
        # Number of elements of phi sent to the different processes
        sendcounts = [ave_scatter_phi+1 for i in range(res_scatter_phi)] + [ave_scatter_phi for i in range(size-res_scatter_phi)]
        sendcounts = np.array(sendcounts,dtype=np.int32)
        # Number of elements away from the first element in the array at which to begin the new, segmented
        displacements = np.array([sum(sendcounts[:j]) for j in range (size)])
    else:
        phi = np.empty([1], dtype = np.complex128)
        sendcounts = np.zeros([size],dtype=np.int32)
        displacements = np.empty([1])
    comm.Bcast(sendcounts, root = 0)
    phi_partial = np.empty([sendcounts[rank]], dtype = np.complex128)
    comm.Scatterv([phi, sendcounts, displacements, MPI.DOUBLE_COMPLEX], phi_partial, root = 0)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    # Add the first column to the end of vi
    vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
    M_partial = np.array([create_M_matrix_with_holes (phi_r, vi_append, collocation_nodes, alpha) for phi_r in phi_partial])
    M_partial_shape = M_partial.shape
    M_matrices = np.empty([sum(sendcounts),M_partial_shape[1],M_partial_shape[2]]).astype(np.complex128) # , dtype = complex?
    M_shape_each = M_partial_shape[1]*M_partial_shape[2]
    sendcounts_matrices = M_shape_each*sendcounts
    displacements_matrices = M_shape_each*displacements

    phi_derivative_partial = np.empty([sendcounts[rank]], dtype = np.complex128)
    comm.Scatterv([phi_derivative, sendcounts, displacements, MPI.DOUBLE_COMPLEX], phi_derivative_partial, root = 0)

    # Inverse of the M-matrices
    M_inv_partial = np.linalg.inv(M_partial)
    # Prestep to calculate the summands for the tepezoidal rule
    A_0_summand_temp_partial = np.dot(M_inv_partial,V_hat)
    #print(f"I'm rank {rank} and A_0_summand_temp_partial is {A_0_summand_temp_partial}")
    # Summands for the approximated A0
    A_0_summand_partial = A_0_summand_temp_partial*phi_derivative_partial[:,np.newaxis, np.newaxis]
    # Summands for the approximated A1
    A_1_summand_partial = phi_partial[:,np.newaxis, np.newaxis] * A_0_summand_partial
    #print(f"I'm rank {rank} and A_0_summand_partial is {A_0_summand_partial}")
    #print(f"I'm rank {rank} and A_1_summand_partial is {A_1_summand_partial}")
    # Approximated A0 and A1
    A_0_partial = np.sum(A_0_summand_partial, axis=0)/(1j*N)
    A_1_partial = np.sum(A_1_summand_partial, axis=0)/(1j*N)
    #print(f"I'm rank {rank} and A_0_partial is {A_0_partial}")

    if rank == 0:
        A_0 = np.empty_like(A_0_partial)
        A_1 = np.empty_like(A_1_partial)
    else:
        A_0 = np.empty([1])
        A_1 = np.empty([1])

    comm.Reduce(A_0_partial, A_0, op = MPI.SUM, root = 0)
    comm.Reduce(A_1_partial, A_1, op = MPI.SUM, root = 0)
    #end_time = MPI.Wtime()
    if rank == 0: 
        # Inverse of the M-matrices
        #M_matrices_inv = np.linalg.inv(M_matrices)
        # Prestep to calculate the summands for the tepezoidal rule
        #A_0_summand_temp = np.dot(M_matrices_inv,V_hat)
        # Summands for the approximated A0
        #A_0_summand = A_0_summand_temp*phi_derivative[:,np.newaxis, np.newaxis]
        # Summands for the approximated A1
        #A_1_summand = phi[:,np.newaxis, np.newaxis] * A_0_summand
        # Approximated A0 and A1
        #A_0 = np.sum(A_0_summand, axis=0)/(1j*N)
        #A_1 = np.sum(A_1_summand, axis=0)/(1j*N)
        # Calculate singular value decomposition of A0
        V0, Sigma0, W0H = np.linalg.svd(A_0, full_matrices=False)
        #Rank test
        Sigma0_eps = np.where(Sigma0 < eps, 0, Sigma0)
        rank_test = np.count_nonzero(Sigma0_eps)
        # Truncated singular value decomposition 
        Sigma_rank = Sigma0_eps[:rank_test]
        V_rank = V0[:,:rank_test]
        WH_rank = W0H[:rank_test,:]
        #print(f"Shape of WH_rank is {WH_rank.shape}")
        # Calculate matrix B
        B = np.linalg.multi_dot([V_rank.T.conj(), A_1, WH_rank.T.conj(), np.diag(1/Sigma_rank)])
        # Solve the linear eigenvalue problem to obtain the eigenvalues of the nonlinear problem
        eigenvalues, eigenvectors_B = np.linalg.eig(B)
        eigenvalues_shape = np.array(eigenvalues.shape, dtype = np.int32)
        # Calculate the eigenvectors of the nonlinear eigenvalue problem based on the linear ones
        eigenvectors = np.dot(V_rank,eigenvectors_B)
        eigenvectors_shape = np.array(eigenvectors.shape, dtype = np.int32)
        #print(f"Shape of Eigenvalues is {eigenvalues.shape} and shape of eigenvectors is {eigenvectors_shape}")
    else:
        #eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
        eigenvalues = np.empty([1], dtype = np.complex128)
    comm.Bcast(eigenvalues_shape, root = 0)
    #print(f"I'm rank {rank} Eigenvalues after: {eigenvalues}")
    #print(f"I'm rank {rank} Eigenvectors_shape before: {eigenvectors_shape}")
    comm.Bcast(eigenvectors_shape, root = 0)
    #print(f"I'm rank {rank} Eigenvectors_shape after: {eigenvectors_shape}")
    if rank != 0:
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
    #print(f"I'm rank {rank} Eigenvalues before: {eigenvalues}")
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    #print(f"I'm rank {rank} Eigenvectors before: {eigenvectors}")
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    #print(f"I'm rank {rank} Eigenvectors after: {eigenvectors}")
    #if rank == 0:
    #    print(f"Eigenvalues are {eigenvalues} and eigenvectors are {eigenvectors}")
    return eigenvalues, eigenvectors

def beyn_algorithm_with_holes_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by transforming it into a linear one.
    This method should be used for a domain with holes.
    The eigenvalue are searched in the inner part of a circle with the radius R and the center in midpoint (complex plane).
    The method is based on Keldysh's theorem with A0: f(z) = 1 and A1: f(z) = z.
    To approximate the integrals the trapezoidal rule is used.

    Parameters
    ----------
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
    # print(f"n_coll is {n_coll}")

    vi = np.concatenate((vi, vi[:,:,0].reshape(vi.shape[0],2,1)), axis=2)
    
    # Create random matrix V_dach with shape mxl, where n(\gamma) <= l <= m
    V_hat = np.random.random((m,l))+1j*np.random.random((m,l))
    # Calculate matrix A_0 and A_1 with trapezoidal rule and N equidistant nodes
    equidistant_nodes = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # Calculate the derivative of the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi_derivative = create_circle_parameterization_derivative(R, equidistant_nodes)
    # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    
    # if rank == 0:
    #     # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    #     phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    #     ave_scatter_phi, res_scatter_phi = divmod(phi.shape[0], size)
    #     # Number of elements of phi sent to the different processes
    #     sendcounts = [ave_scatter_phi+1 for i in range(res_scatter_phi)] + [ave_scatter_phi for i in range(size-res_scatter_phi)]
    #     sendcounts = np.array(sendcounts,dtype=np.int32)
    #     # Number of elements away from the first element in the array at which to begin the new, segmented
    #     displacements = np.array([sum(sendcounts[:j]) for j in range (size)])
    # else:
    #     phi = np.empty([1], dtype = np.complex128)
    #     sendcounts = np.zeros([size],dtype=np.int32)
    #     displacements = np.empty([1])
    
    # Calculate the parameterization of a circle with center midpoint and the radius R at the equidistant points.
    phi = create_circle_parameterization(midpoint, R, equidistant_nodes)
    # How many rows must be calculated by a process?
    ave_n_rows_D, res_n_rows_D = divmod(N*n_coll*n_holes*n_holes, size)
    # Calculate number of lines that each process has to calculate
    n_lines = np.concatenate((np.repeat(ave_n_rows_D+1, res_n_rows_D), np.repeat(ave_n_rows_D, size-res_n_rows_D)))
    # Caluculate cummulative number of lines the processes have to calculate starting from zero
    cumsum_lines = np.cumsum(n_lines)
    cumsum_lines = np.insert(cumsum_lines,0,0).astype(np.int32)
    # print(f"cumsum_lines is {cumsum_lines}")
    #For phi
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_matrix, start_line = divmod(cumsum_lines[:-1], n_holes*n_holes*n_coll)
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_matrix, end_line = divmod(cumsum_lines[1:]-1, n_holes*n_holes*n_coll)
    # print(f"start_matrix is {start_matrix}")
    # print(f"end_matrix is {end_matrix}")
    # print(f"start_line is {start_line}")
    # print(f"end_line is {end_line}")
    #For hole
    # Submatrix and row in the submatrix where the calculation starts for all processes.
    start_hole_matrix, start_hole_line = divmod(cumsum_lines[:-1], n_coll)
    # print(f"start_hole_matrix is {start_hole_matrix}")
    # print(f"start_hole_line is {start_hole_line}")
    # Submatrix and row in the submatrix where the calculation ends for all processes.
    end_hole_matrix, end_hole_line = divmod(cumsum_lines[1:]-1, n_coll)
    # print(f"end_hole_matrix is {end_hole_matrix}")
    # print(f"end_hole_line is {end_hole_line}")
    # start_hole_in_phi_matrix = np.mod(start_hole_matrix, n_holes)
    # end_hole_in_phi_matrix = np.mod(end_hole_matrix, n_holes)
    # # in Submatrix phi, submatrix in column and row (holes)
    # print(f"start_hole_in_phi_matrix is {start_hole_in_phi_matrix}")
    # print(f"start_hole_in_phi_matrix is {type(start_hole_in_phi_matrix)}")
    # print(f"end_hole_in_phi_matrix is {end_hole_in_phi_matrix}")
    # print(n_holes)
    # start_column_matrix, start_row_matrix = np.mod(start_hole_in_phi_matrix, n_holes)
    # end_column_matrix, end_row_matrix = np.mod(end_hole_in_phi_matrix, n_holes)

    start_row_matrix, start_column_matrix = divmod(np.mod(start_hole_matrix, n_holes*n_holes),n_holes)
    end_row_matrix, end_column_matrix = divmod(np.mod(end_hole_matrix, n_holes*n_holes),n_holes)
    # in Submatrix phi, submatrix in column and row (holes)
    # print(f"start_row_matrix is {start_row_matrix}")
    # print(f"start_column_matrix is {start_column_matrix}")
    # print(f"end_row_matrix is {end_row_matrix}")
    # print(f"end_column_matrix is {end_column_matrix}")
    #start_column_matrix, start_row_matrix = np.mod(start_hole_in_phi_matrix, n_holes)
    #end_column_matrix, end_row_matrix = np.mod(end_hole_in_phi_matrix, n_holes)
    
    # Submatrix and row in the submatrix where the calculation for the current processes starts and ends.
    # start_matrix_partial = start_matrix[rank]
    # start_column_partial = start_column_matrix[rank]
    # start_row_partial = start_row_matrix[rank]
    # start_line_partial = start_hole_line[rank]
    # end_matrix_partial = end_matrix[rank]
    # end_column_partial = end_column_matrix[rank]
    # end_row_partial = end_row_matrix[rank]
    # end_line_partial = end_hole_line[rank]
    start_matrix_partial = start_matrix[rank]
    start_row_partial = start_column_matrix[rank]
    start_column_partial = start_row_matrix[rank]
    start_line_partial = start_hole_line[rank]
    end_matrix_partial = end_matrix[rank]
    end_row_partial = end_column_matrix[rank]
    end_column_partial = end_row_matrix[rank]
    end_line_partial = end_hole_line[rank]
    # if rank == 3:
    #     print(f"start_matrix_partial is {start_matrix_partial}")
    #     print(f"start_column_partial is {start_column_partial}")
    #     print(f"start_row_partial is {start_row_partial}")
    #     print(f"start_line_partial is {start_line_partial}")
    #     print(f"end_matrix_partial is {end_matrix_partial}")
    #     print(f"end_column_partial is {end_column_partial}")
    #     print(f"end_row_partial is {end_row_partial}")
    #     print(f"end_line_partial is {end_line_partial}")
    # Compute the submatrices of D on each process.
    # If only one submatrix needs to be calculated in the process
   
    if start_matrix_partial == end_matrix_partial:
        if start_column_partial == end_column_partial:
            if start_row_partial == end_row_partial:
                D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[start_row_partial], collocation_nodes[start_column_partial,:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
            else:
                for row in range (start_row_partial, end_row_partial+1):
                    if row == start_row_partial:
                        D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                    elif row == end_row_partial:
                        D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                    else:
                        D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]

                        # D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[start_column_partial,:,start_line_partial:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
        else:
            for column in range (start_column_partial, end_column_partial+1):
                if column == start_column_partial:
                    for row in range (start_row_partial, n_holes):
                        if row == start_row_partial:
                            D_matrix_holes_partial = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,start_line_partial:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                elif column == end_column_partial:
                    for row in range (end_row_partial + 1):
                        if row == end_row_partial:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:end_line_partial+1], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                        else:
                            D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                            D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
                else:
                    for row in range(n_holes):
                        D_matrix_holes_partial_temp = integrate.quad_vec(create_integrand_double_layer, 0, 1, args=(phi[start_matrix_partial], vi[row], collocation_nodes[column,:,:], alpha), points = [alpha, 0.5, 1-alpha])[0]
                        D_matrix_holes_partial = np.concatenate((D_matrix_holes_partial, D_matrix_holes_partial_temp), axis=0)
    #If the rows to be calculated extend across multiple sub-matrices
    else: 
        for i in range(start_matrix_partial, end_matrix_partial+1):
            if i == start_matrix_partial:
                for column in range (start_column_partial, n_holes):
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
                        for row in range (end_row_partial+1):
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
    # print(f"sendcounts_matrices is {sendcounts_matrices}")
    # print(f"displacements_matrices is {displacements_matrices}")
    # print(f"Rank {rank}: D_matrix_holes_partial is {D_matrix_holes_partial.shape}")
    # print(f"D_matrix_holes is {D_matrix_holes.shape}")
    comm.Gatherv(D_matrix_holes_partial, [D_matrix_holes, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
    # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
    D_matrix_holes_conc = np.concatenate(np.concatenate(np.concatenate(D_matrix_holes, axis = 1), axis=1), axis=1)
    # if rank == 0:
    #     print(f"D_matrix_holes_conc is {D_matrix_holes_conc}")
    #     print(f"D_matrix_holes_conc shape is {D_matrix_holes_conc.shape}")
    # print(f"D_matrix_holes_conc shape is {D_matrix_holes_conc.shape}")
    M_matrix_holes = D_matrix_holes_conc - np.tile(np.identity(D_matrix_holes_conc[:,0].size),N)/2 
    M_matrix_holes_3d = np.reshape(M_matrix_holes,(n_holes*n_coll,N,n_holes*n_coll)).transpose(1,0,2)
    # print(f"D_matrix_holes_3d shape is {M_matrix_holes_3d.shape}")
    # comm.Bcast(sendcounts, root = 0)
    # phi_partial = np.empty([sendcounts[rank]], dtype = np.complex128)
    # comm.Scatterv([phi, sendcounts, displacements, MPI.DOUBLE_COMPLEX], phi_partial, root = 0)
    # # Add the first column to the end of vi
    # vi_append = np.concatenate((vi, vi[:,:,0].reshape(vi.shape[0],2,1)), axis=2)
    # # Calculate the boundary double layer including the jump conditions for the interior Dirichlet problem for each wavenumber phi.
    # M_partial = np.array([create_M_matrix_with_holes (phi_r, vi_append, collocation_nodes, alpha) for phi_r in phi_partial])
    # M_partial_shape = M_partial.shape

    # if size >= N:
    #     print("Here 1")
    #     print(f"Rank before {rank}")
    #     if rank < N:
    #         print(f"Rank after {rank}")
    #         M_matrices_inv = np.empty_like(M_matrix_holes_3d)
    #         # Inverse of the M-matrices
    #         M_partial_inv = np.linalg.inv(M_matrix_holes_3d[rank])
    #         M_shape_each = M_matrix_holes_3d.shape[1]*M_matrix_holes_3d.shape[2]
    #         sendcounts = np.ones((N),dtype=int)
    #         sendcounts_matrices = M_shape_each*sendcounts
    #         displacements = np.array([sum(sendcounts[:j]) for j in range (N)])
    #         displacements_matrices = M_shape_each*displacements
            
    #         print(f"sendcounts_matrices is {sendcounts_matrices}")
    #         print(f"displacements_matrices is {displacements_matrices}")
    #         print(f"M_partial_inv is {M_partial_inv.shape}")
    #         print(f"M_matrices_inv is {M_matrices_inv.shape}")
    #         comm.Gatherv(M_partial_inv, [M_matrices_inv, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
    
    # elif size < N:
    #     print("Here 2")
    #     ave_scatter_phi, res_scatter_phi = divmod(phi.shape[0], size)
    #     # Number of elements of phi sent to the different processes
    #     sendcounts = [ave_scatter_phi+1 for i in range(res_scatter_phi)] + [ave_scatter_phi for i in range(size-res_scatter_phi)]
    #     sendcounts = np.array(sendcounts,dtype=np.int32)
    #     # Number of elements away from the first element in the array at which to begin the new, segmented
    #     displacements = np.array([sum(sendcounts[:j]) for j in range (size)])
        
    #     M_matrices_inv = np.empty_like(M_matrix_holes_3d)
    #     M_shape_each = M_matrix_holes_3d.shape[1]*M_matrix_holes_3d.shape[2]
    #     sendcounts_matrices = M_shape_each*sendcounts
    #     displacements_matrices = M_shape_each*displacements
        
    #     index_inv_matrix = np.arange(displacements[rank],displacements[rank]+sendcounts[rank])
    #     # Inverse of the M-matrices
    #     print(f"index_inv_matrix is {index_inv_matrix}")
    #     M_partial_inv = np.array([np.linalg.inv(M_matrix_holes_3d[i])for i in index_inv_matrix])
        
    #     #M_partial_inv = np.array[np.linalg.inv(M_matrix_holes_3d[i]) for i in index_inv_matrix])
    #     print(f"sendcounts_matrices is {sendcounts_matrices}")
    #     print(f"displacements_matrices is {displacements_matrices}")
    #     M_partial_inv = np.concatenate(M_partial_inv, axis=0)
    #     print(f"M_partial_inv is {M_partial_inv.shape}")
    #     print(f"M_matrices_inv is {M_matrices_inv.shape}")
        
    #     comm.Gatherv(M_partial_inv, [M_matrices_inv, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX], root = 0)
    #     print("Test")
    #comm.Allgatherv(M_partial, [M_matrices, sendcounts_matrices, displacements_matrices, MPI.DOUBLE_COMPLEX])
    #end_time = MPI.Wtime()
    #MPI.Finalize()
    #print(f"M_matrices are: {M_matrices}")

    if rank == 0:
        print("First step is done")
    if rank == 0: 
        M_matrices_inv = np.linalg.inv(M_matrix_holes_3d)
        print("Matrix_inv done")
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
        print("SVD done")
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
        #print(f"Eigenvalues are {eigenvalues}")
        #print(f"Shape of Eigenvalues is {eigenvalues.shape} and shape of eigenvectors is {eigenvectors_shape}")
    else:
        eigenvalues_shape = np.zeros([1],dtype=np.int32)
        eigenvectors_shape = np.zeros([2],dtype=np.int32)
        #eigenvalues = np.empty([1], dtype = np.complex128)
    comm.Bcast(eigenvalues_shape, root = 0)
    comm.Bcast(eigenvectors_shape, root = 0)
    if rank != 0:
        eigenvalues = np.empty(eigenvalues_shape, dtype = np.complex128)
        eigenvectors = np.empty(eigenvectors_shape, dtype = np.complex128)
    comm.Bcast([eigenvectors, MPI.DOUBLE_COMPLEX], root = 0)
    comm.Bcast([eigenvalues, MPI.DOUBLE_COMPLEX], root = 0)
    return eigenvalues, eigenvectors


# In[31]:


def calculate_eigenvalues_eigenvectors(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain without holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.
    
    Parameters
    ----------
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
        eigenvalues, eigenvectors_B = beyn_algorithm_MPI(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors


# In[32]:


def calculate_eigenvalues_eigenvectors_with_holes(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
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
    while (eigenvalues.shape[0] == l) & (l < m):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors_B = beyn_algorithm_with_holes (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
    return eigenvalues, eigenvectors


# In[33]:


def calculate_eigenvalues_eigenvectors_with_holes_MPI(M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps):
    r"""
    Routine to find the eigenvalues and eigenfunctions of the nonlinear eigenvalue problem by using Beyn's algorithm.
    This method should be used for a domain with holes.
    If the set of computed eigenvalues is equal to l and l is less than the shape of the matrix M, then l must be increased
    to ensure, that all eigenvalues of the matrix can be found in the area given by the circle.

    Parameters
    ----------
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
    while (eigenvalues.shape[0] == l) & (l < m):
        if (m-l) >= 10:
            l=l+10
        else:
            l=l+1
        eigenvalues, eigenvectors_B = beyn_algorithm_with_holes_MPI (M_matrix, vi, collocation_nodes, alpha, m, l, midpoint, R, N, eps)
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
    #print(f"size {size}")
    
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
    # Concatenate the n_holes submatrices so that shape = (n_inside, n_coll*n_holes)
    # D_matrix = np.concatenate(D_matrix, axis=1)
    # print(f"D_matrix_holes shape is {D_matrix_holes.shape}")
    # Free the subcommunicator
    sub_comm.Free()

    comm.Bcast(D_matrix, root = 0)
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
    #print(f"size {size}")
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
    #xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps), indexing='ij')
    xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps))
    mesh = np.stack((xv, yv), axis=2)
    mesh = np.reshape(mesh,(x_steps*y_steps,2))
    # Check whether the mesh is inside or outside of the boundary of the domain.
    inside = p.contains_points(mesh)
    inside = np.reshape(inside, (y_steps, x_steps))
    xy_inside = np.stack((xv[inside], yv[inside]), axis=0)
    return xy_inside

# def calculate_double_layer_inside(wavenumber, vi, alpha, eigenvectors, x_start, x_end, x_steps, y_start, y_end, y_steps):
#     xy_inside = ray_tracing(vi, x_start, x_end, x_steps, y_start, y_end, y_steps)
#     # Add the first column to the end of vi
#     vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
#     D_matrix, eps = create_D_matrix (wavenumber, vi_append, xy_inside, alpha)
#     #create_integrand_double_layer (s, wavenumber, vi, xy_inside, alpha) # statt colocation nodes das mesh an den stellen, wo inside = 1
#     sol = np.dot(D_matrix, eigenvectors)
#     #xv, yv = np.meshgrid(np.linspace(x_start,x_end,x_steps), np.linspace(y_start,y_end,y_steps), indexing='ij')
#     return xy_inside, sol

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
    if rank == 0:
        print(f"xy_inside.shape is {xy_inside.shape}")
    # Add the first column to the end of vi
    vi_append = np.concatenate((vi, vi[:,0].reshape(2,1)), axis=1)
    # Calculate the D-matrix at the points xy_inside
    # D_matrix, eps = create_D_matrix(wavenumber, vi_append,  xy_inside, alpha)
    D_matrix = create_D_matrix_result_MPI(wavenumber, vi_append, collocation_nodes, xy_inside, alpha)
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
    #time_start1 = time.time()
    # Calculate all the points that are inside the boundary of the given domain (without holes) and the square.
    xy_inside = ray_tracing_with_holes(vi_holes, x_start, x_end, x_steps, y_start, y_end, y_steps)
    #time_end1 = time.time()
    #Add the first column to the end of vi
    vi_holes_append = np.concatenate((vi_holes, vi_holes[:,:,0].reshape(vi_holes.shape[0],2,1)), axis=2)
    # Calculate the D-matrix at the points xy_inside
    #D_matrix = create_D_matrix_with_holes_result (wavenumber, vi_holes_append, collocation_nodes_holes, xy_inside, alpha)
    #time_start2 = time.time()
    D_matrix = create_D_matrix_with_holes_result_MPI (wavenumber, vi_holes_append, collocation_nodes_holes, xy_inside, alpha)
    #time_end2 = time.time()
    # Calculate the double layer potential at the points xy_inside
    #time_start3 = time.time()
    sol = np.dot(D_matrix, eigenvectors_holes)
    #time_end3 = time.time()
    #print(f"Time Ray Tracing: {time_end1-time_start1}")
    #print(f"Time Create D Matrix: {time_end2-time_start2}")
    #print(f"Time Dot Product: {time_end3-time_start3}")
    return xy_inside, sol

