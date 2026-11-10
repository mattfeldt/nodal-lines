import numpy as np
import DoubleLayerWithHoles_MPI_new_8 as DL
from scipy.special import comb

def transform_3d_array_in_object_array_with_2d_arrays(vi):
    n_objects = vi.shape[0]
    vi_object = np.empty(n_objects, dtype=object)
    for i in range(n_objects):
        vi_object[i] = vi[i]
    return vi_object


def create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,xa_hyp,xi_hyp,xm_hyp,ym_hyp,rot_hyp,n,alpha):
    """
    Routine to create the nodes (vertices and midpoints) and the collocation nodes for the following geometry.
    

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

def create_6_ellipses(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,n,alpha):
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

def Beziercurve_6th_degree_upper(eps, a_circle,b_circle,a_ellipse,b_ellipse,xm_ellipse,ym_ellipse,rot_ellipse,n):
    # Circle
    # Continuous
    P0 = np.array([a_circle*np.cos(rot_ellipse+eps), b_circle*np.sin(rot_ellipse+eps)])
    # Smooth
    T0 = -np.array([-a_circle*np.sin(rot_ellipse+eps), b_circle*np.cos(rot_ellipse+eps)])
    P1 = P0 + T0/6
    # Constant curvature
    K0 = np.array([-a_circle*np.cos(rot_ellipse+eps), -b_circle*np.sin(rot_ellipse+eps)])
    P2 = P0 + 2/6*T0 +1/30*K0
    
    # Ellipse
    # Continuous
    P5 = np.array([xm_ellipse+a_ellipse*np.cos(eps)* np.cos(rot_ellipse)-b_ellipse*np.sin(eps)*np.sin(rot_ellipse),
                   ym_ellipse+a_ellipse*np.cos(eps)* np.sin(rot_ellipse)+b_ellipse*np.sin(eps)*np.cos(rot_ellipse)])
    # Smooth
    T1 = np.array([-a_ellipse*np.sin(eps)* np.cos(rot_ellipse)-b_ellipse*np.cos(eps)*np.sin(rot_ellipse),
                   -a_ellipse*np.sin(eps)* np.sin(rot_ellipse)+b_ellipse*np.cos(eps)*np.cos(rot_ellipse)])
    P4 = P5 - T1/6
    # Constant curvature
    K1 = np.array([-a_ellipse*np.cos(eps)* np.cos(rot_ellipse)+b_ellipse*np.sin(eps)*np.sin(rot_ellipse),
                   -a_ellipse*np.cos(eps)* np.sin(rot_ellipse)-b_ellipse*np.sin(eps)*np.cos(rot_ellipse)])
    P3 = P5 - 2/6*T1 + 1/30 * K1

    T2 = (P0-P5)/np.linalg.norm(P0-P5)
    #P6 = (P0+P5)/2 + 0.02*np.array([T2[1],-T2[0]])
    #P6 = P0*1/2+P5*1/2 + 0.02*np.array([T2[1],-T2[0]])
    P6 = (P0+P5)/2

    P = np.array([P0, P1, P2, P6, P3, P4, P5])
    # Generate Beziercurve
    ti = np.linspace(0,1,n,endpoint=False)
    degree=6
    B_t = np.zeros((2,len(ti)))
    for i in range(degree + 1):
        B_t = B_t+np.outer(P[i], comb(degree, i)*pow((1-ti),degree-i) *pow(ti,i))
    return B_t

def Beziercurve_6th_degree_lower(eps, a_circle,b_circle,a_ellipse,b_ellipse,xm_ellipse,ym_ellipse,rot_ellipse,n):
    # Circle
    # Continuous
    P0 = np.array([a_circle*np.cos(rot_ellipse-eps), b_circle*np.sin(rot_ellipse-eps)])
    # Smooth
    T0 = np.array([-a_circle*np.sin(rot_ellipse-eps), b_circle*np.cos(rot_ellipse-eps)])
    P1 = P0 + T0/6
    # Constant curvature
    K0 = np.array([-a_circle*np.cos(rot_ellipse-eps), -b_circle*np.sin(rot_ellipse-eps)])
    P2 = P0 + 2/6*T0 +1/30*K0

    # Ellipse
    # Continuous
    P5 = np.array([xm_ellipse+a_ellipse*np.cos(-eps)* np.cos(rot_ellipse)-b_ellipse*np.sin(-eps)*np.sin(rot_ellipse),
                   ym_ellipse+a_ellipse*np.cos(-eps)* np.sin(rot_ellipse)+b_ellipse*np.sin(-eps)*np.cos(rot_ellipse)])
    # Smooth
    T1 = -np.array([-a_ellipse*np.sin(-eps)* np.cos(rot_ellipse)-b_ellipse*np.cos(-eps)*np.sin(rot_ellipse),
                   -a_ellipse*np.sin(-eps)* np.sin(rot_ellipse)+b_ellipse*np.cos(-eps)*np.cos(rot_ellipse)])
    P4 = P5 - T1/6
    # Constant curvature
    K1 = np.array([-a_ellipse*np.cos(-eps)* np.cos(rot_ellipse)+b_ellipse*np.sin(-eps)*np.sin(rot_ellipse),
                   -a_ellipse*np.cos(-eps)* np.sin(rot_ellipse)-b_ellipse*np.sin(-eps)*np.cos(rot_ellipse)])
    P3 = P5 - 2/6*T1 + 1/30 * K1

    T2 = (P0-P5)/np.linalg.norm(P0-P5)
    #P6 = (P0+P5)/2 + 0.02*np.array([T2[1],-T2[0]])
    #P6 = P0*1/2+P5*1/2 + 0.02*np.array([T2[1],-T2[0]])
    P6 = (P0+P5)/2

    P = np.array([P0, P1, P2, P6, P3, P4, P5])
    # Generate Beziercurve
    ti = np.linspace(0,1,n,endpoint=True)
    degree=6
    B_t = np.zeros((2,len(ti)))
    for i in range(degree + 1):
        B_t = B_t+np.outer(P[i], comb(degree, i)*pow((1-ti),degree-i) *pow(ti,i))
    return B_t

def create_ellipse_section(a,b,xm,ym,rot,start_angle,end_angle,n):
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
    ti = np.linspace(start_angle,end_angle,n, endpoint=False)
    vi = np.array([xm+a*np.cos(ti)* np.cos(rot)-b*np.sin(ti)*np.sin(rot), ym+a*np.cos(ti)* np.sin(rot)+b*np.sin(ti)*np.cos(rot)])
    return vi

def create_ellipses_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses,n):
    rot_ellipse = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_ellipse = distance_ellipse*np.cos(rot_ellipse)
    ym_ellipse = distance_ellipse*np.sin(rot_ellipse)
    start_angles = np.linspace(0,2*np.pi,n_ellipses, endpoint=False)+eps
    end_angles = np.linspace(0,2*np.pi,n_ellipses+1, endpoint=True)[1:]-eps
    n_for_circle = n-4*int(n/6)
    big_ellipse_sections = np.array([create_ellipse_section(a_circle,b_circle,xm_circle,ym_circle,rot_circle,start_angles[i], end_angles[i],n_for_circle) for i in range(n_ellipses)])
    Beziercurves_lower = np.array([Beziercurve_6th_degree_lower(eps, a_circle,b_circle,a_ellipse,b_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i],n-1) for i in range(n_ellipses)])
    # To start with second ellipse
    Beziercurves_lower_1 = np.roll(Beziercurves_lower,(-1,0,0),axis=(0,1,2))
    small_ellipses_sections = np.array([np.flip(create_ellipse_section(a_ellipse,b_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i],eps,2*np.pi-eps,n),1) for i in range(n_ellipses)])
    # To start with second ellipse
    small_ellipses_sections_1 = np.roll(small_ellipses_sections,(-1,0,0),axis=(0,1,2))
    Beziercurves_upper = np.array([np.flip(Beziercurve_6th_degree_upper(eps, a_circle,b_circle,a_ellipse,b_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i],n),1) for i in range(n_ellipses)])
    # To start with second ellipse
    Beziercurves_upper_1 = np.roll(Beziercurves_upper,(-1,0,0),axis=(0,1,2))
    # Delete last element of the upper Beziercurve. Otherwise duplicate with startelement of big ellipse
    Beziercurves_upper_1 = Beziercurves_upper_1[:,:,:-1]
    ellipses_with_connections = np.dstack((big_ellipse_sections, Beziercurves_lower_1,small_ellipses_sections_1, Beziercurves_upper_1))
    ellipses_with_connections_2d = np.concatenate(ellipses_with_connections, axis=1)
    return ellipses_with_connections_2d

def create_ellipses_with_connections_and_ellipses_diff_n(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses_bezier,n_ellipses_single,n,alpha):
    ellipses_with_connections_2d = create_ellipses_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses_bezier,n[0])
    rot_ellipse = np.linspace(np.pi/n_ellipses_single,2*np.pi*(1+1/(2*n_ellipses_single)), n_ellipses_single, endpoint=False)
    xm_ellipse = distance_ellipse*np.cos(rot_ellipse)
    ym_ellipse = distance_ellipse*np.sin(rot_ellipse)
    single_ellipses = [np.flip(DL.create_ellipse(a_ellipse,b_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i],n[1]),1) for i in range(n_ellipses_single)]
    vi_ellipses_with_connections_and_single = np.empty(n_ellipses_single + 1, dtype=object)
    vi_ellipses_with_connections_and_single[0] = ellipses_with_connections_2d
    for i in range(n_ellipses_single):
        vi_ellipses_with_connections_and_single[i+1] = single_ellipses[i]
    #vi_ellipses_with_connections_hypotrochoid = np.vstack((vi_ellipses_with_connections[None,:,:], vi_hypotrochoid[None,:,:]))
    #collocation_nodes_ellipses_with_connections_hypotrochoid = np.array(([DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[i],alpha) for i in range(vi_ellipses_with_connections_hypotrochoid.shape[0])]))
    collocation_nodes_ellipses_with_connections_and_single = np.empty(n_ellipses_single + 1, dtype=object)
    collocation_nodes_ellipses_with_connections_and_single[0] = DL.create_collocation_nodes(vi_ellipses_with_connections_and_single[0],alpha)
    for i in range(n_ellipses_single):
        collocation_nodes_ellipses_with_connections_and_single[i+1] = DL.create_collocation_nodes(vi_ellipses_with_connections_and_single[i+1],alpha)
    return vi_ellipses_with_connections_and_single, collocation_nodes_ellipses_with_connections_and_single

def create_ellipses_with_connections_and_ellipses_hypotrochoid_diff_n(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses_bezier,n_ellipses_single,xa_hyp,xi_hyp,n,alpha):
    # Outer Ellipse witn n_ellipses inner ellipses and Beziercurves in between
    ellipses_with_connections_2d = create_ellipses_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses_bezier,n[0])
    # Inner ellipses without connections
    rot_ellipse = np.linspace(np.pi/n_ellipses_single,2*np.pi*(1+1/(2*n_ellipses_single)), n_ellipses_single, endpoint=False)
    xm_ellipse = distance_ellipse*np.cos(rot_ellipse)
    ym_ellipse = distance_ellipse*np.sin(rot_ellipse)
    single_ellipses = [np.flip(DL.create_ellipse(a_ellipse,b_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i],n[1]),1) for i in range(n_ellipses_single)]
    # Hypotrochoid
    n_ellipses = n_ellipses_bezier + n_ellipses_single
    rot_hyp = 0
    xm_hyp = 0
    ym_hyp = 0
    r_hyp = (xa_hyp+xi_hyp)/(2*(n_ellipses-1))
    R_hyp = (n_ellipses)*r_hyp
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp)
    vi_hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n[-1]),1)

    vi_ellipses_with_connections_and_single_hypo = np.empty(n_ellipses_single + 2, dtype=object)
    vi_ellipses_with_connections_and_single_hypo[0] = ellipses_with_connections_2d
    for i in range(n_ellipses_single):
        vi_ellipses_with_connections_and_single_hypo[i+1] = single_ellipses[i]
    vi_ellipses_with_connections_and_single_hypo[-1] = vi_hypotrochoid
    #vi_ellipses_with_connections_hypotrochoid = np.vstack((vi_ellipses_with_connections[None,:,:], vi_hypotrochoid[None,:,:]))
    #collocation_nodes_ellipses_with_connections_hypotrochoid = np.array(([DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[i],alpha) for i in range(vi_ellipses_with_connections_hypotrochoid.shape[0])]))
    collocation_nodes_ellipses_with_connections_and_single_hypo = np.empty(n_ellipses_single + 2, dtype=object)
    collocation_nodes_ellipses_with_connections_and_single_hypo[0] = DL.create_collocation_nodes(vi_ellipses_with_connections_and_single_hypo[0],alpha)
    for i in range(n_ellipses_single):
        collocation_nodes_ellipses_with_connections_and_single_hypo[i+1] = DL.create_collocation_nodes(vi_ellipses_with_connections_and_single_hypo[i+1],alpha)
    collocation_nodes_ellipses_with_connections_and_single_hypo[-1] = DL.create_collocation_nodes(vi_ellipses_with_connections_and_single_hypo[-1],alpha)
    return vi_ellipses_with_connections_and_single_hypo, collocation_nodes_ellipses_with_connections_and_single_hypo

def create_ellipses_with_connections_and_hypotrochoid(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses,xa_hyp, xi_hyp,n,alpha):
    vi_ellipses_with_connections = create_ellipses_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses,n)
    n_ellipses_with_connections = vi_ellipses_with_connections.shape[1]
    #if rank == 0:
    #    print(f"shape of ellipses with connections = {n_ellipses_with_connections}")
    # Hypotrochoid
    rot_hyp = 0
    xm_hyp = 0
    ym_hyp = 0
    r_hyp = (xa_hyp+xi_hyp)/(2*(n_ellipses-1))
    R_hyp = n_ellipses*r_hyp
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp)
    vi_hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n_ellipses_with_connections),1)
    # Stack all the boundaries together
    vi_ellipses_with_connections_hypotrochoid = np.vstack((vi_ellipses_with_connections[None,:,:], vi_hypotrochoid[None,:,:]))
    collocation_nodes_ellipses_with_connections_hypotrochoid = np.array(([DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[i],alpha) for i in range(vi_ellipses_with_connections_hypotrochoid.shape[0])]))
    return vi_ellipses_with_connections_hypotrochoid, collocation_nodes_ellipses_with_connections_hypotrochoid

def create_ellipses_with_connections_and_hypotrochoid_diff_n(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses,xa_hyp, xi_hyp,n,alpha):
    #print(f"n is {n}")
    vi_ellipses_with_connections = create_ellipses_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,distance_ellipse,n_ellipses,n[0])
    #n_ellipses_with_connections = vi_ellipses_with_connections.shape[1]
    #if rank == 0:
    #    print(f"shape of ellipses with connections = {n_ellipses_with_connections}")
    # Hypotrochoid
    rot_hyp = 0
    xm_hyp = 0
    ym_hyp = 0
    r_hyp = (xa_hyp+xi_hyp)/(2*(n_ellipses-1))
    R_hyp = n_ellipses*r_hyp
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp)
    vi_hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n[1]),1)
    # Stack all the boundaries together
    vi_ellipses_with_connections_hypotrochoid = np.empty(2, dtype=object)
    vi_ellipses_with_connections_hypotrochoid[0] = vi_ellipses_with_connections
    vi_ellipses_with_connections_hypotrochoid[1] = vi_hypotrochoid
    #vi_ellipses_with_connections_hypotrochoid = np.vstack((vi_ellipses_with_connections[None,:,:], vi_hypotrochoid[None,:,:]))
    #collocation_nodes_ellipses_with_connections_hypotrochoid = np.array(([DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[i],alpha) for i in range(vi_ellipses_with_connections_hypotrochoid.shape[0])]))
    collocation_nodes_ellipses_with_connections_hypotrochoid = np.empty(2, dtype=object)
    collocation_nodes_ellipses_with_connections_hypotrochoid[0] = DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[0],alpha)
    collocation_nodes_ellipses_with_connections_hypotrochoid[1] = DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[1],alpha)
    return vi_ellipses_with_connections_hypotrochoid, collocation_nodes_ellipses_with_connections_hypotrochoid

def create_n_kites_with_hypotrochoid(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_kite,b_small_kite,c_small_kite,distance_small_kite,n_ellipses,xa_hyp,xi_hyp,xm_hyp,ym_hyp,rot_hyp,n,alpha):
    """
    Routine to create the nodes (vertices and midpoints) and the collocation nodes for the following geometry.
    

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
    # Outer ellipse
    big_ellipse = DL.create_ellipse(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,n)
    # Inner kites                                                                                                  
    rot_small_kite = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_small_kite = distance_small_kite*np.cos(rot_small_kite)                                                           
    ym_small_kite = distance_small_kite*np.sin(rot_small_kite)                                                           
    small_kites = [np.flip(create_kite(a_small_kite,b_small_kite,c_small_kite,xm_small_kite[i],ym_small_kite[i],rot_small_kite[i] - np.pi/2,n),1) for i in range(n_ellipses)]    
    # Hypotrochoid                                                                                                    
    r_hyp = (xa_hyp+xi_hyp)/(2*(n_ellipses-1));                                                                       
    R_hyp = n_ellipses*r_hyp;                                                                                         
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp);                                                           
    hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n),1)                       
    # Stack all the boundaries together                                                                               
    vi_first_eig = np.vstack((big_ellipse[None,:,:],small_kites, hypotrochoid[None,:,:]))                          
    collocation_nodes_first_eig = np.array(([DL.create_collocation_nodes(vi_first_eig[i],alpha) for i in range(vi_first_eig.shape[0])]))
    return vi_first_eig, collocation_nodes_first_eig                      

def create_n_kites(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_kite,b_small_kite,c_small_kite,distance_small_kite,n_ellipses,n,alpha):
    """ 
    Routine to create the nodes (vertices and midpoints) and the collocation nodes for the following geometry.        
    
                                                                                                                      
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
    # Outer ellipse
    big_ellipse = DL.create_ellipse(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,n)
    # Inner kites                                                                                                  
    rot_small_kite = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_small_kite = distance_small_kite*np.cos(rot_small_kite)                                                           
    ym_small_kite = distance_small_kite*np.sin(rot_small_kite)                                                           
    small_kites = [np.flip(create_kite(a_small_kite,b_small_kite,c_small_kite,xm_small_kite[i],ym_small_kite[i],rot_small_kite[i] - np.pi/2,n),1) for i in range(n_ellipses)]    
    # Stack all the boundaries together                                                                               
    vi_first_eig = np.vstack((big_ellipse[None,:,:],small_kites))                          
    collocation_nodes_first_eig = np.array(([DL.create_collocation_nodes(vi_first_eig[i],alpha) for i in range(vi_first_eig.shape[0])]))
    return vi_first_eig, collocation_nodes_first_eig                      

def create_kite(a,b,c,xm,ym,rot,n):
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
    vi = np.array([xm-a*np.sin(ti)* np.cos(rot)-b*(np.cos(ti)+c*np.cos(2*ti))*np.sin(rot), ym-a*np.sin(ti)* np.sin(rot)+b*(np.cos(ti)+c*np.cos(2*ti))*np.cos(rot)])
    return vi

def create_6_ellipses_with_hypotrochoid(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,a_small_ell,b_small_ell,distance_small_ell,n_ellipses,xa_hyp,xi_hyp,xm_hyp,ym_hyp,rot_hyp,n,alpha):
    """
    Routine to create the nodes (vertices and midpoints) and the collocation nodes for the following geometry.
    

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
    # Outer ellipse
    big_ellipse = DL.create_ellipse(a_big_ell,b_big_ell,xm_big_ell,ym_big_ell,rot_big_ell,n)
    # Inner ellipses
    rot_small_ell = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_small_ell = distance_small_ell*np.cos(rot_small_ell)
    ym_small_ell = distance_small_ell*np.sin(rot_small_ell)
    small_ellipses = [np.flip(DL.create_ellipse(a_small_ell,b_small_ell,xm_small_ell[i],ym_small_ell[i],rot_small_ell[i],n),1) for i in range(n_ellipses)]
    # Hypotrochoid
    r_hyp = (xa_hyp+xi_hyp)/(2*(n_ellipses-1));
    R_hyp = n_ellipses*r_hyp;
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp);
    hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n),1)
    # Stack all the boundaries together
    vi_first_eig = np.vstack((big_ellipse[None,:,:],small_ellipses, hypotrochoid[None,:,:]))
    collocation_nodes_first_eig = np.array(([DL.create_collocation_nodes(vi_first_eig[i],alpha) for i in range(vi_first_eig.shape[0])]))
    return vi_first_eig, collocation_nodes_first_eig

def create_wave(a,b,c,d,xm,ym,rot,n):
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
    print(n)
    ti = np.linspace(0,2*np.pi,n, endpoint=False)
    vi = np.array([xm-a*np.sin(ti)* np.cos(rot)-b*(np.cos(ti)+c*np.cos(2*ti)+d*np.cos(4*ti))*np.sin(rot), ym-a*np.sin(ti)* np.sin(rot)+b*(np.cos(ti)+c*np.cos(2*ti)+d*np.cos(4*ti))*np.cos(rot)])
    return vi

def create_wave_section(a,b,c,d,xm,ym,rot,start_angle,end_angle,n):
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
    rot_temp = rot - np.pi/2
    ti = np.linspace(start_angle,end_angle,n, endpoint=False)
    vi = np.array([xm-a*np.sin(ti)* np.cos(rot_temp)-b*(np.cos(ti)+c*np.cos(2*ti)+d*np.cos(4*ti))*np.sin(rot_temp), ym-a*np.sin(ti)* np.sin(rot_temp)+b*(np.cos(ti)+c*np.cos(2*ti)+d*np.cos(4*ti))*np.cos(rot_temp)])
    return vi

def Beziercurve_6th_degree_upper_waves(eps, a_circle,b_circle,a_ellipse,b_ellipse,c_ellipse,d_ellipse,xm_ellipse,ym_ellipse,rot_ellipse,n):
    rot_wave = rot_ellipse - np.pi/2
    # Circle
    # Continuous
    P0 = np.array([a_circle*np.cos(rot_ellipse+eps), b_circle*np.sin(rot_ellipse+eps)])
    # Smooth
    T0 = -np.array([-a_circle*np.sin(rot_ellipse+eps), b_circle*np.cos(rot_ellipse+eps)])
    P1 = P0 + T0/6
    # Constant curvature
    K0 = np.array([-a_circle*np.cos(rot_ellipse+eps), -b_circle*np.sin(rot_ellipse+eps)])
    P2 = P0 + 2/6*T0 +1/30*K0

    # Ellipse
    # Continuous
    P5 = np.array([xm_ellipse-a_ellipse*np.sin(eps)* np.cos(rot_wave)-b_ellipse*(np.cos(eps)+c_ellipse*np.cos(2*eps)+d_ellipse*np.cos(4*eps))*np.sin(rot_wave),
                   ym_ellipse-a_ellipse*np.sin(eps)* np.sin(rot_wave)+b_ellipse*(np.cos(eps)+c_ellipse*np.cos(2*eps)+d_ellipse*np.cos(4*eps))*np.cos(rot_wave)])
    # Smooth
    T1 = np.array([-a_ellipse*np.cos(eps)* np.cos(rot_wave)+b_ellipse*(np.sin(eps)+2*c_ellipse*np.sin(2*eps)+4*d_ellipse*np.sin(4*eps))*np.sin(rot_wave),
                   -a_ellipse*np.cos(eps)* np.sin(rot_wave)-b_ellipse*(np.sin(eps)+2*c_ellipse*np.sin(2*eps)+4*d_ellipse*np.sin(4*eps))*np.cos(rot_wave)])
    P4 = P5 - T1/6
    # Constant curvature
    K1 = np.array([+a_ellipse*np.sin(eps)* np.cos(rot_wave)+b_ellipse*(np.cos(eps)+4*c_ellipse*np.cos(2*eps)+16*d_ellipse*np.cos(4*eps))*np.sin(rot_wave),
                   a_ellipse*np.sin(eps)* np.sin(rot_wave)-b_ellipse*(np.cos(eps)+4*c_ellipse*np.cos(2*eps)+16*d_ellipse*np.cos(4*eps))*np.cos(rot_wave)])
    P3 = P5 - 2/6*T1 + 1/30 * K1

    T2 = (P0-P5)/np.linalg.norm(P0-P5)
    #P6 = (P0+P5)/2 + 0.02*np.array([T2[1],-T2[0]])
    #P6 = P0*1/2+P5*1/2 + 0.02*np.array([T2[1],-T2[0]])
    P6 = (P0+P5)/2

    P = np.array([P0, P1, P2, P6, P3, P4, P5])
    # Generate Beziercurve
    ti = np.linspace(0,1,n,endpoint=False)
    degree=6
    B_t = np.zeros((2,len(ti)))
    for i in range(degree + 1):
        B_t = B_t+np.outer(P[i], comb(degree, i)*pow((1-ti),degree-i) *pow(ti,i))
    return B_t

def Beziercurve_6th_degree_lower_waves(eps, a_circle,b_circle,a_ellipse,b_ellipse,c_ellipse,d_ellipse,xm_ellipse,ym_ellipse,rot_ellipse,n):
    rot_wave = rot_ellipse - np.pi/2
    # Circle
    # Continuous
    P0 = np.array([a_circle*np.cos(rot_ellipse-eps), b_circle*np.sin(rot_ellipse-eps)])
    # Smooth
    T0 = np.array([-a_circle*np.sin(rot_ellipse-eps), b_circle*np.cos(rot_ellipse-eps)])
    P1 = P0 + T0/6
    # Constant curvature
    K0 = np.array([-a_circle*np.cos(rot_ellipse-eps), -b_circle*np.sin(rot_ellipse-eps)])
    P2 = P0 + 2/6*T0 +1/30*K0

    # Ellipse
    # Continuous
    P5 = np.array([xm_ellipse-a_ellipse*np.sin(-eps)* np.cos(rot_wave)-b_ellipse*(np.cos(-eps)+c_ellipse*np.cos(-2*eps)+d_ellipse*np.cos(-4*eps))*np.sin(rot_wave),
                   ym_ellipse-a_ellipse*np.sin(-eps)* np.sin(rot_wave)+b_ellipse*(np.cos(-eps)+c_ellipse*np.cos(-2*eps)+d_ellipse*np.cos(-4*eps))*np.cos(rot_wave)])
    # Smooth
    T1 = np.array([+a_ellipse*np.cos(-eps)* np.cos(rot_wave)-b_ellipse*(np.sin(-eps)+2*c_ellipse*np.sin(-2*eps)+4*d_ellipse*np.sin(-4*eps))*np.sin(rot_wave),
                   +a_ellipse*np.cos(-eps)* np.sin(rot_wave)+b_ellipse*(np.sin(-eps)+2*c_ellipse*np.sin(-2*eps)+4*d_ellipse*np.sin(-4*eps))*np.cos(rot_wave)])
    P4 = P5 - T1/6
    # Constant curvature
    K1 = np.array([+a_ellipse*np.sin(-eps)* np.cos(rot_wave)+b_ellipse*(np.cos(-eps)+4*c_ellipse*np.cos(-2*eps)+16*d_ellipse*np.cos(-4*eps))*np.sin(rot_wave),
                   a_ellipse*np.sin(-eps)* np.sin(rot_wave)-b_ellipse*(np.cos(-eps)+4*c_ellipse*np.cos(-2*eps)+16*d_ellipse*np.cos(-4*eps))*np.cos(rot_wave)])

    P3 = P5 - 2/6*T1 + 1/30 * K1

    T2 = (P0-P5)/np.linalg.norm(P0-P5)
    #P6 = (P0+P5)/2 + 0.02*np.array([T2[1],-T2[0]])
    #P6 = P0*1/2+P5*1/2 + 0.02*np.array([T2[1],-T2[0]])
    P6 = (P0+P5)/2

    P = np.array([P0, P1, P2, P6, P3, P4, P5])
    # Generate Beziercurve
    ti = np.linspace(0,1,n,endpoint=True)
    degree=6
    B_t = np.zeros((2,len(ti)))
    for i in range(degree + 1):
        B_t = B_t+np.outer(P[i], comb(degree, i)*pow((1-ti),degree-i) *pow(ti,i))
    return B_t

def create_waves_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,c_ellipse,d_ellipse,distance_ellipse,n_ellipses,n):
    rot_ellipse = np.linspace(0,2*np.pi, n_ellipses, endpoint=False)
    xm_ellipse = distance_ellipse*np.cos(rot_ellipse)
    ym_ellipse = distance_ellipse*np.sin(rot_ellipse)
    start_angles = np.linspace(0,2*np.pi,n_ellipses, endpoint=False)+eps
    end_angles = np.linspace(0,2*np.pi,n_ellipses+1, endpoint=True)[1:]-eps
    n_for_circle = n-4*int(n/6)
    big_ellipse_sections = np.array([create_ellipse_section(a_circle,b_circle,xm_circle,ym_circle,rot_circle,start_angles[i], end_angles[i],n_for_circle) for i in range(n_ellipses)])
    #big_ellipse_sections = np.array([create_ellipse_section(a_circle,b_circle,xm_circle,ym_circle,rot_circle,start_angles[i], end_angles[i],n) for i in range(n_ellipses)])
    Beziercurves_lower = np.array([Beziercurve_6th_degree_lower_waves(eps, a_circle,b_circle,a_ellipse,b_ellipse,c_ellipse,d_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i]  ,n-1) for i in range(n_ellipses)])
    #print(Beziercurves_lower)
    # To start with second ellipse
    Beziercurves_lower_1 = np.roll(Beziercurves_lower,(-1,0,0),axis=(0,1,2))
    small_ellipses_sections = np.array([np.flip(create_wave_section(a_ellipse,b_ellipse,c_ellipse,d_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i],eps,2*np.pi-eps,n),1) for i in range(n_ellipses)])
    # To start with second ellipse
    small_ellipses_sections_1 = np.roll(small_ellipses_sections,(-1,0,0),axis=(0,1,2))
    Beziercurves_upper = np.array([np.flip(Beziercurve_6th_degree_upper_waves(eps, a_circle,b_circle,a_ellipse,b_ellipse,c_ellipse,d_ellipse,xm_ellipse[i],ym_ellipse[i],rot_ellipse[i] ,n),1) for i in range(n_ellipses)])
    # To start with second ellipse
    Beziercurves_upper_1 = np.roll(Beziercurves_upper,(-1,0,0),axis=(0,1,2))
    # Delete last element of the upper Beziercurve. Otherwise duplicate with startelement of big ellipse
    Beziercurves_upper_1 = Beziercurves_upper_1[:,:,:-1]
    ellipses_with_connections = np.dstack((big_ellipse_sections, Beziercurves_lower_1,small_ellipses_sections_1, Beziercurves_upper_1))
    ellipses_with_connections_2d = np.concatenate(ellipses_with_connections, axis=1)
    return ellipses_with_connections_2d

def create_waves_with_connections_and_hypotrochoid_diff_n(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,c_ellises,d_ellipses,distance_ellipse,n_ellipses,xa_hyp, xi_hyp,n,alpha):
    #print(f"n is {n}")
    vi_waves_with_connections = create_waves_with_connections(eps, a_circle,b_circle,xm_circle,ym_circle,rot_circle,a_ellipse,b_ellipse,c_ellises,d_ellipses,distance_ellipse,n_ellipses,n[0])
    #n_ellipses_with_connections = vi_ellipses_with_connections.shape[1]
    #if rank == 0:
    #    print(f"shape of ellipses with connections = {n_ellipses_with_connections}")
    # Hypotrochoid
    rot_hyp = 0
    xm_hyp = 0
    ym_hyp = 0
    r_hyp = (xa_hyp+xi_hyp)/(2*(n_ellipses-1))
    R_hyp = n_ellipses*r_hyp
    a_hyp = (n_ellipses-1)*(xa_hyp-xi_hyp)/(xa_hyp+xi_hyp)
    vi_hypotrochoid = np.flip(DL.create_hypotrochoid(R_hyp,r_hyp,a_hyp,xm_hyp,ym_hyp,rot_hyp,n[1]),1)
    # Stack all the boundaries together
    vi_waves_with_connections_hypotrochoid = np.empty(2, dtype=object)
    vi_waves_with_connections_hypotrochoid[0] = vi_waves_with_connections
    vi_waves_with_connections_hypotrochoid[1] = vi_hypotrochoid
    #vi_ellipses_with_connections_hypotrochoid = np.vstack((vi_ellipses_with_connections[None,:,:], vi_hypotrochoid[None,:,:]))
    #collocation_nodes_ellipses_with_connections_hypotrochoid = np.array(([DL.create_collocation_nodes(vi_ellipses_with_connections_hypotrochoid[i],alpha) for i in range(vi_ellipses_with_connections_hypotrochoid.shape[0])]))
    collocation_nodes_waves_with_connections_hypotrochoid = np.empty(2, dtype=object)
    collocation_nodes_waves_with_connections_hypotrochoid[0] = DL.create_collocation_nodes(vi_waves_with_connections_hypotrochoid[0],alpha)
    collocation_nodes_waves_with_connections_hypotrochoid[1] = DL.create_collocation_nodes(vi_waves_with_connections_hypotrochoid[1],alpha)
    return vi_waves_with_connections_hypotrochoid, collocation_nodes_waves_with_connections_hypotrochoid

