import dolfin as dl
import numpy as np

def _bisection(u, a, b, u_target, xi):
    u_a = u(xi, a)
    u_b = u(xi, b)
    assert u_a < u_target
    assert u_b > u_target
    c = .5*a + .5*b
    u_c = u(xi, c)
    if np.abs( u_c - u_target ) < 1e-6:
        return c
    elif u_c > u_target:
        return _bisection(u,a,c, u_target, xi)
    else:
        return _bisection(u,c,b, u_target, xi)
        
        
def spreadFunction(u, mesh, x_max = None):
    """
    This routine computes the following quantities as a function of the distance x from the inflow:
    - The centerline velocity: u_cl
    - The integral jet thickness: L
    - The spread function (derivative of jet thickness): S
    - The jet thickness: y_1/2
    
    INPUTS:
    - the velocity u
    - the mesh mesh
    """
    boundary_mesh = dl.BoundaryMesh(mesh,"exterior")
    x = boundary_mesh.coordinates()
    x_coord = x[np.abs(x[:,1]) < 1e-9, 0]
    if x_max is not None:
        x_coord = x_coord[x_coord <= x_max]
    e1 = dl.Expression(("1.", "0."))
    u1, u2 = u.split(deepcopy = True)
    
    Vh_grad = dl.FunctionSpace(mesh, 'RT', 1)
    grad_u1 = dl.Function(Vh_grad)
    test = dl.TestFunction(Vh_grad)
    n = dl.FacetNormal(mesh)
    dl.solve(dl.inner(grad_u1, test)*dl.dx + u1*dl.div(test)*dl.dx - u1*dl.dot(test,n)*dl.ds == 0, grad_u1)
        
    axis = dl.AutoSubDomain(lambda x: dl.near(x[1], 0.))
    axis_mesh = dl.SubMesh(boundary_mesh, axis)
    Vh_axis = dl.FunctionSpace(axis_mesh, "CG", 2)
    u1_axis = dl.interpolate(u1, Vh_axis)
    Vh_axis_grad = dl.FunctionSpace(axis_mesh, 'CG', 1)
    du1_dx = dl.Function(Vh_axis_grad)
    test_axis = dl.TestFunction(Vh_axis_grad)
    left_point = dl.AutoSubDomain( lambda x, on_boundary: dl.near(x[0], x_coord[0]) and on_boundary)
    right_point = dl.AutoSubDomain( lambda x, on_boundary: dl.near(x[0], x_coord[-1]) and on_boundary)
    bb_marker = dl.FacetFunction("size_t", axis_mesh)
    bb_marker.set_all(0)
    left_point.mark(bb_marker, 1)
    right_point.mark(bb_marker, 2)
    dss = dl.Measure("ds")[bb_marker]
    
    dl.solve(du1_dx*test_axis*dl.dx + u1_axis*test_axis.dx(0)*dl.dx 
             +u1_axis*test_axis*dss(1) - u1_axis*test_axis*dss(2) == 0, du1_dx)
            
    u_cl = np.zeros(x_coord.shape)
    L = np.zeros(x_coord.shape)
    S = np.zeros(x_coord.shape)
    y_half = np.zeros(x_coord.shape)
    
    i = 0
    for xi in x_coord:
        line_segment = dl.AutoSubDomain(lambda x: dl.near(x[0], xi))
        markers = dl.FacetFunction("size_t", mesh)
        markers.set_all(0)
        line_segment.mark(markers, 1)
        
        if i == 0 or ( i==(x_coord.shape[0]-1) and x_max is None ):
            ds = dl.ds[markers]
            int_u1 = dl.assemble(u1*ds(1))
            int_du1_dx = dl.assemble(dl.dot(grad_u1,e1)*ds(1))
        else:
            dS = dl.dS[markers]
            int_u1 = dl.assemble(dl.avg(u1)*dS(1))
            int_du1_dx = dl.assemble(dl.avg( dl.dot(grad_u1,e1) )*dS(1))
            
            
        u_cl[i] = u1( (xi,0.) )
        du_cl_dx = du1_dx( (xi, 0.) )
        L[i] = int_u1/u_cl[i]
        S[i] = (int_du1_dx*u_cl[i] - int_u1*du_cl_dx)/(u_cl[i]*u_cl[i])
        y_half[i] = _bisection(u1, 9., 0., .5*u_cl[i], xi)
        
        i += 1
        
    out = np.zeros( (x_coord.shape[0],5), dtype = x_coord.dtype)
    out[:,0] = x_coord
    out[:,1] = u_cl
    out[:,2] = L
    out[:,3] = S
    out[:,4] = y_half
        
    return out

def selfsimilarprofiles(u, nu_t, x_coord, u_cl, y_half, mesh):
    """
    This routine computes
    - the adimensional y-coord y_hat = y / y_1/2
    - the adimensional velocity u_hat = u/u_cl
    - the adimensional viscosity nu_t_hat = nu_t / (u_hat * y_hat)
    
    INPUTS:
    - the velocity field u
    - the turbulent viscosity field nu_t
    - the list of x coordinates x_coord
    - the centerline velocities u_cl
    - the jet thickness y_1/2
    - the mesh
    """
    boundary_mesh = dl.BoundaryMesh(mesh,"exterior")
    x = boundary_mesh.coordinates()
    y_coord = x[ np.abs(x[:,0] - x_coord[0]) < 1e-9, 1 ]
    u1, u2 = u.split(deepcopy = True)
    
    u_hat = np.zeros((y_coord.shape[0], x_coord.shape[0]))
    y_hat = np.zeros((y_coord.shape[0], x_coord.shape[0]))
    nu_t_hat = np.zeros((y_coord.shape[0], x_coord.shape[0]))
    
    j = 0
    for xj in x_coord:
        u_clj = u_cl[j]
        y_halfj = y_half[j]
        i=0
        for yi in y_coord:
            y_hat[i,j] = yi/y_halfj
            u_hat[i,j] = u1( (xj,yi) )/u_clj
            nu_t_hat[i,j] = nu_t( (xj, yi) )/( u_clj*y_halfj )
            i+=1
        j += 1
        
    return  y_hat, u_hat, nu_t_hat
    