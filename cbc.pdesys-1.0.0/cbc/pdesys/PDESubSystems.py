__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-08"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

import ufl
import copy
from dolfin import *
from numpy import maximum, minimum, array, zeros
from collections import defaultdict
from time import time
import operator

import os

#parameters["linear_algebra_backend"] = "Epetra"
parameters["linear_algebra_backend"] = "PETSc"
#parameters['form_compiler']['representation'] = 'quadrature'
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
# Cache for work arrays
_work = {}

# Wrap Krylov solver because of the two different calls depending on whether the preconditioner has been set.
KrylovSolver.solve1 = KrylovSolver.solve
def cbcsolve(self, A, x, b):
    if self.preconditioned_solve:
        self.solve1(x, b)
    else:
        self.solve1(A, x, b)
KrylovSolver.solve = cbcsolve

class PDESubSystemBase:
    """Subclasses in the PDESubSystem hierarchy 1) define one variational 
    form, for (parts of) a PDE system, 2) assemble associated linear system, 
    and 3) solve the system. Some forms arise in many PDE problems and
    collecting such forms in a common library, together with assembly and solve
    functionality, makes the forms reusable across many PDE solvers.
    This is the rationale behind the PDESubSystem hierarchy.

    A particular feature of classes in the PDESubSystem hierarchy is the
    ease of avoiding assembly and optimizing solve steps if possible.
    For example, if a particular form is constant in time, the
    PDESubSystem subclass can easily assemble the associated matrix or vector
    only once. If the form is also shared among PDE solvers, the various
    solvers will automatically take advantage of only a single assembly.
    Similarly, for direct solution methods for linear systems, the matrix can
    be factored only once.
    
    Parameters:
    
    solver_namespace     = Namespace of solver class that holds the solution vector
    sub_system           = The sub_system we have a variational form for
    bcs                  = List of boundary conditions
    normalize            = normalization function        
    
    """
    
    def __init__(self, solver_namespace, sub_system, bcs=[], normalize=None, **kwargs):        
        self.solver_namespace = solver_namespace       
        self.query_args(sub_system)
        self.prm = Subdict(solver_namespace, self.name,
            reassemble_lhs=True,  # True if lhs depends on any q_, q_1, ..
            reassemble_rhs=True,  # True if rhs depends on any q_, q_1, ..
            reassemble_lhs_inner=True, # True if lhs depends on q_
            reassemble_rhs_inner=True, # True if rhs depends on q_
            reassemble_precond=True,
            assemble_system=False,
            max_inner_iter=1,
            max_inner_err=1e-5,
            reset_sparsity=True,
            wall_value=1e-12
            )
        self.prm.update(**kwargs)
        self.bcs = bcs
        self.normalize = normalize
        self.exterior_facet_domains = kwargs.get('exterior_facet_domains', None)
        self.F = kwargs.get('F', None)
        self.a = None
        # Define matrix and vector for lhs and rhs respectively
        self.A = Matrix() # Matrix() can be used for > 0.9.8
        self.A.initialized = False
        self.b = Vector()
        self.B = Matrix() # Possible user defined preconditioner 
        # Define matrix and vector for unchanging parts of tensor
        self.A1 = None
        self.b1 = None        
        info_green(self._info())
                
    def query_args(self, sub_system):
        """Check that the correct parameters have been suplied."""
        if not isinstance(sub_system, list):
           raise TypeError("expected a list for sub_system")

        self.name = ''.join(sub_system)
        self.V = self.solver_namespace[self.name + '_'].function_space()
        self.x = self.solver_namespace[self.name + '_'].vector()
        self.sub_system = sub_system
        self.work = self.get_work_vector()

    def solve(self, max_iter=None, assemble_A=None, assemble_b=None):
        """One or more assemble/solve of the variational form.
        If the tensor is uninitialized or the iteration type is Newton, 
        then assemble regardless of call. Otherwise use self.prm.
        
        If the user does not specify the number of iterations, then use 
        prm['max_inner_iter']. prm['max_inner_iter'] should only (!!) 
        be > 1 if the form depends on the solution q_ and then it 
        should always be reassembled. 
        """
        if not self.A.initialized or self.prm['iteration_type'] == 'Newton':
            assemble_A = True
            assemble_b = True
            self.A.initialized = True
        else:
            assemble_A = assemble_A if assemble_A!=None else self.prm['reassemble_lhs']
            assemble_b = assemble_b if assemble_b!=None else self.prm['reassemble_rhs']
        max_iter = max_iter or self.prm['max_inner_iter']
        j = 0
        err = 1
        solve = eval('self.solve_%s_system' %(self.prm['iteration_type']))
        while err > self.prm['max_inner_err'] and j < max_iter:
            res, dx = solve(assemble_A, assemble_b)
            j += 1
            ndx = norm(dx)
            if max_iter > 1: 
                info_red(" %8s Inner iter %s, error %s, %s " 
                         %(self.name, j, res, ndx))
                         
            err = max(res, ndx)
        
        return res, dx

    def solve_Picard_system(self, assemble_A, assemble_b):
        """One assemble and solve of Picard system."""
        self.prepare()
        if self.prm['assemble_system']:
            self.A, self.b = assemble_system(self.a, self.L, bcs=self.bcs,
                     A_tensor=self.A, b_tensor=self.b, 
                     exterior_facet_domains=self.exterior_facet_domains,
                     reset_sparsity=self.prm['reset_sparsity'])
            self.prm['reset_sparsity'] = False
        else:
            if assemble_A: self.assemble(self.A)
            if assemble_b: self.assemble(self.b)
            if assemble_A and assemble_b:
                [bc.apply(self.A, self.b) for bc in self.bcs]
            elif assemble_A:
                [bc.apply(self.A) for bc in self.bcs]
            elif assemble_b:
                [bc.apply(self.b) for bc in self.bcs]
        x_star = self.work  # more informative name
        x_star[:] = self.x[:]    # start vector for iterative solvers
        self.setup_solver(assemble_A, assemble_b)
        self.linear_solver.solve(self.A, x_star, self.b)
        if self.normalize: self.normalize(x_star)
        omega = self.prm['omega']
        err = x_star - self.x
        if abs(omega-1.) < 1.e-8:
            self.x[:] = x_star[:]
        else:
            # x = (1-omega)*x + omega*x_star = x + omega*(x_star-x):
            x_star.axpy(-1., self.x);  self.x.axpy(omega, x_star)  # relax
        res = residual(self.A, self.x, self.b)
        self.update()
        return res, err

    def solve_Newton_system(self, *args):
        """One assemble and solve of Newton system."""
        self.prepare()
        self.assemble(self.A)
        self.assemble(self.b)
        #if self.normalize: self.normalize(b)
        if not hasattr(self, 'first'):
            [bc.apply(self.x) for bc in self.bcs]
            self.first = True
        [bc.apply(self.A, self.b, self.x) for bc in self.bcs]
        dx = self.work   # more informative name
        dx.zero()        # start vector for iterative solvers
        self.linear_solver.solve(self.A, dx, self.b)
        if self.normalize: self.normalize(dx)
        omega = self.prm['omega']
        self.x.axpy(omega, dx)  # relax
        self.update()
        return norm(self.b), dx
        #return 1, dx

    def assemble(self, M):
        """Assemble tensor."""
        if isinstance(M, Matrix):
            M = assemble(self.a, tensor=M,
                     exterior_facet_domains=self.exterior_facet_domains,
                     reset_sparsity=self.prm['reset_sparsity'])
            # It is possible to preassemble parts of the matrix in A1. In that case just add the preassembled part
            if not self.A1 is None:
                M.axpy(1., self.A1, True)
            self.prm['reset_sparsity'] = False
        elif isinstance(M, Vector):
            M = assemble(self.L, tensor=M,
                     exterior_facet_domains=self.exterior_facet_domains)       
            # It is possible to preassemble parts of the vector in b1. If so add it here
            if not self.b1 is None:
                M.axpy(1., self.b1)
    
    def __call__(self):
        """Return complete lhs and rhs forms."""
        return self.a, self.L
        
    def get_solver(self):
        """Return linear solver. 
        """
        if self.prm['linear_solver'] == 'lu':
            return LUSolver()
        else:
            sol = KrylovSolver(self.prm['linear_solver'], self.prm['precond'])
            sol.preconditioned_solve = False
            return sol
            
    def setup_solver(self, assemble_A, assemble_b):
        """Some pdesubsystems do not require recomputing factorization or
        preconditioner. If the coefficient matrix is reassembled, then a new
        factorization must take place. Called prior to solve.
        """        
        sol = self.linear_solver
        prm_sol = sol.parameters 
        if type(sol) is KrylovSolver:
            prm_sol['monitor_convergence'] = self.prm['monitor_convergence']
            prm_sol['error_on_nonconvergence'] = False
            prm_sol['nonzero_initial_guess'] = True
            prm_sol['report'] = False
            if self.prm['monitor_convergence']:
                info_red('   Monitoring convergence for ' + self.name)
            
        if not assemble_A:
            # If A is not assembled, then neither is the preconditioner
            if type(sol) is KrylovSolver:
                prm_sol['preconditioner']['reuse'] = True
            elif type(sol) is LUSolver:
                prm_sol['reuse_factorization'] = True
                
        else: 
            if type(sol) is KrylovSolver:
                prm_sol['preconditioner']['reuse'] = False
                
                # Check for user defined preconditioner
                sol.B = self.get_precond(**self.solver_namespace)
                if sol.B: 
                    sol.set_operators(self.A, sol.B)
                    sol.preconditioned_solve = True
                    
            elif type(sol) is LUSolver:
                prm_sol['reuse_factorization'] = False

    def update(self):
        """Update pdesubsystem after solve. Called at end of solve_%s_system."""
        pass
    
    def prepare(self):
        """Prepare pdesubsystem for solve. Called at beginning of solve_%s_system."""
        pass
            
    def _info(self):
        return "Adding PDESubSystem: %s" %(self.__class__.__name__)

    def conv(self, v, u, w, convection_form = 'Standard'):
        """Alternatives for convection discretization."""
        #info_green(convection_form + ' convection')
        if convection_form == 'Standard':
            return inner(v, dot(w, nabla_grad(u)))
            #return inner(v, dot(grad(u), w))            
            
        elif convection_form == 'Divergence':
            return inner(v, nabla_div(outer(w, u)))
            
        elif convection_form == 'Divergence by parts':
            # Use with care. ds term could be important
            return -inner(grad(v), outer(w, u))
            
        elif convection_form == 'Skew':
            return 0.5*(inner(v, dot(u, nabla_grad(u))) + inner(v, nabla_div(outer(w, u))))

    def get_work_vector(self):
        """Return a work vector. Check first in cached _work."""
        name = self.V.ufl_element()
        if name in _work:
            return _work[name]
        else:
            info_green('Creating new work vector for {0:s}'.format(self.name))
            _work[name] = Vector(self.x)
            return _work[name]

    def get_form(self, form_args):
        """Set the variational form F.
        There are three ways of providing F:
            1) return F from method form
            2) Provide F as a string through keyword F
            3) Provide F as ufl.form.Form through keyword F
            
        The procedure is to check first in 1), then 2) and finally 3).
        """
        F = self.form(**form_args)
        if F:
            self.F = F
        elif isinstance(self.F, str):
            self.F = eval(self.F, globals(), form_args)
        elif isinstance(self.F, ufl.form.Form):
            pass

    def form(self, *args, **kwargs):
        """Return the variational form to be solved."""
        return False

    def add_exterior(self, *args, **kwargs):
        """Add weak boundary conditions."""
        pass

    def get_precond(self, *args, **kwargs):
        """Get a special preconditioner."""
        return None

class PDESubSystem(PDESubSystemBase):
    """Base class for most PDESubSystems"""
    def __init__(self, solver_namespace, sub_system, bcs=[], normalize=None, **kwargs):
        PDESubSystemBase.__init__(self, solver_namespace, sub_system, bcs, normalize, **kwargs)

        if not isinstance(self.bcs, list):
            raise TypeError("expecting a list of boundary conditions")
                        
        self.linear_solver = self.get_solver()
                                        
        self.define()
                        
    def define(self):
        
        form_args = self.solver_namespace.copy()
        if self.prm['iteration_type'] == 'Picard':
            self.get_form(form_args)
            self.a, self.L = lhs(self.F), rhs(self.F)
            
        else:
            # Set up Newton system by switching Function for TrialFunction
            for name in self.sub_system:
                form_args[name + '_'] = self.solver_namespace[name]
            self.get_form(form_args)
            u_ = self.solver_namespace[self.name + '_']
            u  = self.solver_namespace[self.name]
            F_ = action(self.F, coefficient=u_)
            J_ = derivative(F_, u_, u)
            self.a, self.L = J_, -F_

class DerivedQuantity(PDESubSystemBase):
    """Base class for derived quantities.        
    Derived quantities are all computed through forms like
    
    F = inner(u, v)*dx + L*v*dx
    
    where L is a function of some primary unknowns.
    For example, the turbulent viscosity in the Standard k-epsilon model
    is a derived quantity that can be computed like:
    
    u = TrialFunction(V)
    v = TestFunction(V)        
    F = inner(u, v)*dx - 0.09*k_**2/e_*v*dx
    # (k_ and e_ are the latest approximations to k and epsilon)
    a, L = lhs(F), rhs(F)
    A = assemble(a)
    b = assemble(L)
    solve(A, x, b)
    
    The purpose of this class is to simplify the interface for setting 
    up and solving for derived quantities. Using this class you can in 
    the solverclass set up the turbulent viscosity like
    
    nut_parameter = DerivedQuantity(dict(Cmu=Cmu, k_=k_, e_=e_), 'nut_', 
                                    V, "Cmu*k_**2/e_", apply='project')
        
    To compute one iteration of the turbulent viscosity with under-
    relaxation simply do:
    
    nut_parameter.solve()
    
    However, you don't have to set up a linear system and solve
    it. Another option is simply to use the formula as it is by
    setting self.nut_ = Cmu*k_**2/e_.  This is achieved by using the
    keyword apply = 'use_formula':
    
    nut_parameter.prm['apply'] = 'use_formula'
    nut_parameter.solve()  # sets nut_ = Cmu*k_**2/e_ in the provided 
    namespace (first argument)
    
    After redefining the nut_ parameter you must remember to call the
    solver's define method, because all the pdesubsystems containing nut_
    will be affected.
    
    Note 1. Through project it is possible to solve with under-
    relaxation, which often is neccessary to achieve convergence.
    
    Note 2. This base class assigns Dirichlet BCs on Walls with value
    self.prm['wall_value'] (default=1e-12). It also assumes that the
    derived quantity is larger than zero. Quantities that do not
    assign Dirichlet BC's on walls or are not bounded by zero should
    overload the update and create_BCs methods. See, e.g.,
    DerivedQuantity_NoBC.
        
    """
    def __init__(self, solver_namespace, name, space, formula, **kwargs):
        PDESubSystemBase.__init__(self, solver_namespace, name, **kwargs)

        self.formula, self.V = formula, space                     
        self.prm['iteration_type'] = 'Picard'
        self.bounded = kwargs.get('bounded', True)
        self.linear_solver = self.get_solver()
        self.dq = None  # unknown Function, if needed
        self.eval_formula()
        self.prm['reassemble_lhs'] = False
        self.initialize()

    def query_args(self, sub_system):
        if not isinstance(sub_system, str):
           raise TypeError("expected a str for sub_system")
        self.name = sub_system

    def eval_formula(self):
        #self._form = eval(self.formula, globals(), self.namespace)
        # Experimental:
        # The use of a conditional directly on a bounded derived quantity 
        # can help avoid potential nan's, because the conditional is then 
        # used in other forms where it may cause trouble if zero or negative.
        # For example if the formula appears under a square root sign.
        if self.bounded:
            if not isinstance(self.V, FunctionSpace):
                warning("Conditional works only for scalars." \
                        "Bounding only through update.")
                self._form = eval(self.formula, globals(), self.solver_namespace)
            else:
                # Use conditional to bound the derived quantity
                # New bug? Don't know why this does not work any more MM-071211
                #self._form = eval("max_(%s, Constant(1.e-12, cell=V['dq'].cell()))" 
                #                   %(self.formula), globals(), self.solver_namespace)
                self._form = eval(self.formula, globals(), self.solver_namespace)                   
        else:
            self._form = eval(self.formula, globals(), self.solver_namespace)
        
    def solve(self, *args, **kwargs):
        """Call use_formula, project, or compute_dofs."""
        getattr(self, self.prm['apply'])() 
        
    def use_formula(self):
        """Return formula, but check first if it already exists in solver."""
        if self.solver_namespace[self.name + '_'] is self._form:
            pass
        else:
            self.solver_namespace[self.name + '_'] = self._form

    def make_function(self):
        self.dq = Function(self.V)
        setattr(self, self.name, self.dq)  # attr w/real name
        self.solver_namespace[self.name + '_'] = self.dq
        self.x = self.dq.vector()
        if MPI.num_processes() > 1:
            info_red('make_function does not work in parallell!')
        self.b = Vector(self.x)
        self.work = self.get_work_vector()
            
    def project(self):
        """Solve (dq, v) = (formula, v) for all v in self.V."""
        if self.dq is None:
            self.make_function()
            self.define_projection()
        PDESubSystemBase.solve(self)

    def define_projection(self):
        V = self.V
        dq = TrialFunction(V)
        v = TestFunction(V)
        self.boundaries = self.solver_namespace['boundaries']
        self.bcs = self.create_BCs(self.boundaries)
        F = self.projection_form(dq, v, self._form)
        self.a, self.L = lhs(F), rhs(F)

    def projection_form(self, dq, v, formula):
        return inner(dq, v)*dx - inner(formula, v)*dx

    def define_arrays(self):
        # preprocessing for compute_dofs
        if self.dq is None:
            self.make_function()

        self.namespace_arrays = {}
        for name in self.solver_namespace:
            if not name in self.formula:
                continue # next iter.copy()
                
            var = self.solver_namespace[name]
            if isinstance(var, Function):
                if hasattr(var, 'get_array_slice'):
                    self.namespace_arrays[name] = var.get_array_slice(var)
                else:  # standard Function (not a subFunction without vector)
                    self.namespace_arrays[name] = var.vector().array()

            elif type(self.solver_namespace[name]) is Constant:
                self.namespace_arrays[name] = self.solver_namespace[name](0)

        # Also supply ufl and numpy names
        import numpy
        self.namespace_arrays.update(vars(ufl))
        self.namespace_arrays.update(vars(numpy))
        
    def compute_dofs(self):
        self.define_arrays()
        x_star = self.work
        x_star.set_local(eval(self.formula, self.namespace_arrays))
        x_star.axpy(-1., self.x)
        self.x.axpy(self.prm['omega'], x_star)
        
    def initialize(self):
        if self.prm['apply'] == 'use_formula':
            self.solver_namespace[self.name+'_'] = self._form
        elif self.prm['apply'] in ['project', 'compute_dofs']: 
            # Initial value should be computed without underrelaxation
            dummy = self.prm['omega']
            self.prm['omega'] = 1.
            self.solve()
            self.prm['omega'] = dummy
   
    def create_BCs(self, bcs):
        """Create boundary conditions for derived quantity based on boundaries 
        in list bcs. Assigned boundary conditions for Walls is set to 
        prm['wall_value']. VelocityInlets, ConstantPressure, Outlet and
        Symmetry are do-nothing. Periodic boundaries are handled as always.
        """
        bcu = []
        val = self.prm['wall_value']
        for bc in bcs:
            if bc.type() == 'Wall':
                if isinstance(self.V, FunctionSpace):
                    func = Constant(val)
                elif isinstance(self.V, VectorFunctionSpace):
                    func = Constant((val, )*self.V.num_sub_spaces())
                elif isinstance(self.V, TensorFunctionSpace):
                    dim = self.V.mesh().topology().dim()
                    func = Expression(((str(val), )*dim, )*dim)
                else:
                    raise NotImplementedError
                add_BC(bcu, self.V, bc, func)
            elif bc.type() == 'Periodic':
                add_BC(bcu, self.V, bc, None)
            else:
                info("No assigned boundary condition for %s -- skipping..." %
                     (bc.__class__.__name__))
        return bcu
        
    def update(self):
        if self.bounded:
            bound(self.x)
        
    def _info(self):
        return "Derived Quantity: %s" %(self.name)

class DerivedQuantity_NoBC(DerivedQuantity):
    """
    Derived quantity where default is no assigned boundary conditions.
    """
    def create_BCs(self, bcs):
        bcu = []
        for bc in bcs:
            if bc.type() == 'Periodic':
                bcu.append(PeriodicBC(self.V, bc))
                bcu[-1].type = bc.type
        return bcu

#class DerivedQuantity_NoBC(DerivedQuantity):
    #"""
    #Derived quantity where default is no assigned boundary conditions.
    #"""
    #def create_BCs(self, bcs):
        #return []
        
class DerivedQuantity_grad(DerivedQuantity):
    """Derived quantity using the gradient of the test function."""
    def projection_form(self, dq, v, formula):
        return inner(dq, v)*dx - inner(formula, grad(v))*dx
    
    def update(self):
        pass

class TurbModel(PDESubSystem):
    """Base class for all turbulence models."""

    def update(self):
        bound(self.x, 1e8)
        
class extended_normalize:
    """Normalize part or whole of vector.

    V    = Functionspace we normalize in

    u    = Function where part is normalized

    part = The index of the part of the mixed function space
           that we want to normalize.
        
    For example. When solving for velocity and pressure coupled in the
    Navier-Stokes equations we sometimes (when there is only Neuman BCs 
    on pressure) need to normalize the pressure.

    Example of use:
    mesh = UnitSquare(1, 1)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    VQ = V * Q
    up = Function(VQ)
    normalize_func = extended_normalize(VQ, 2)
    up.vector()[:] = 2.
    print 'before ', up.vector().array().astype('I')
    normalize_func(up.vector())
    print 'after ', up.vector().array().astype('I')

    results in: 
        before [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]   
        after  [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0]
    """
    def __init__(self, V, part='entire vector'):
        self.part = part
        if isinstance(part, int):
            self.u = Function(V)
            v = TestFunction(V)
            self.c = assemble(Constant(1., cell=V.cell())*dx, mesh=V.mesh())        
            self.pp = ['0']*self.u.value_size()
            self.pp[part] = '1'
            self.u0 = interpolate(Expression(self.pp, element=V.ufl_element()), V)
            self.x0 = self.u0.vector()
            self.C1 = assemble(v[self.part]*dx)
        else:
            self.u = Function(V)
            self.vv = self.u.vector()
        
    def __call__(self, v):
        if isinstance(self.part, int):
            # assemble into c1 the part of the vector that we want to normalize
            c1 = self.C1.inner(v)
            if abs(c1) > 1.e-8:
                # Perform normalization
                self.x0[:] = self.x0[:]*(c1/self.c)
                v.axpy(-1., self.x0)
                self.x0[:] = self.x0[:]*(self.c/c1)
        else:
            # normalize entire vector
            #dummy = normalize(v) # does not work in parallel
            #self.vv = Vector(v)
            self.vv[:] = 1./v.size()
            c = v.inner(self.vv)
            self.vv[:] = c
            v.axpy(-1., self.vv)

class FlowSubDomain(AutoSubDomain):
    """Wrapper class that creates a SubDomain compatible with CBC.PDESys's
    declaration of boundaries in terms of its type. This information is 
    used by the PDESystem to create boundary conditions.

    inside_function = inside method taking either x or x, on_boundary as args
                        e.g., lambda x, on_boundary: near(x[0], 0) and on_boundary
                        for an inside method where x[0] is close to zero
                        
                func = values for Dirichlet bcs. 
                        Dictionary using system_names as keys
                        
                mf = FacetFunction identifying boundaries
                
            bc_type = type of boundary. Currently recognized:
                        VelocityInlet 
                        Wall          
                        Periodic      
                        Weak boundary conditions that require meshfunction:
                        Outlet        
                        ConstantPressure
                        (Symmetry)
                        (Slip)
                        
        periodic_map = Method that contains periodicity (see PeriodicBC). 
                        Example:
                        def periodic_map(x, y):
                            y[0] = x[0] - 1
                            y[1] = x[1]
    """
    
    def __init__(self, inside_function, bc_type='Wall', func=None, 
                 mf=None, mark=True, periodic_map=None):
        AutoSubDomain.__init__(self, inside_function)
        self.bc_type = bc_type
        self.type = lambda: self.bc_type
        
        if func: self.func = func
            
        if mf: 
            self.mf = mf
            if hasattr(mf, 'boundary_indicator'):
                mf.boundary_indicator += 1
            else:
                mf.boundary_indicator = 1
            bid = self.bid = mf.boundary_indicator

        if mark and mf:
            self.mark(self.mf, bid)
            
        if periodic_map:
            self.map = periodic_map
        
    def apply(self, *args):
        """Some boundary conditions ('ConstantPressure', 'Outlet', 'Symmetry', 
        'Slip') are sometimes enforced weakly. Hence, these boundary conditions 
        should not modify tensors and by using this function they will correctly
        do-nothing. This apply method is not called in case this subdomain is 
        used to create a strong BC (like DirichletBC for pressure for a 
        ConstantPressure), because the DirichletBC has its own apply method.
        """
        pass

class MeshSubDomain(SubDomain):
    """Wrapper class that creates a SubDomain compatible with CBC.PDESys's
    declaration of boundaries in terms of its type. This information is 
    used by the PDESystem class to create boundary conditions.
    
    To be able to use this subdomain class, the boundary information must
    be part of the mesh as MeshValueCollections, i.e., the function
    mesh.domains().is_empty() must return False.
    
                 bid = Boundary indicator (int)
                        
                func = values for Dirichlet bcs. 
                        Dictionary using system_names as keys
                        
            bc_type = type of boundary. Currently recognized:
                        VelocityInlet
                        Wall         
                        Periodic     
                        Weak boundary conditions:
                        Outlet          
                        ConstantPressure
                        (Symmetry)       
                        (Slip)           
                        
        periodic_map = Method that contains periodicity (see PeriodicBC). 
                        Example:
                        def periodic_map(x, y):
                            y[0] = x[0] - 1
                            y[1] = x[1]
    """
    
    def __init__(self, bid, bc_type='Wall', func=None, periodic_map=None):
        SubDomain.__init__(self)
        self.bid = bid
        self.bc_type = bc_type
        
        self.type = lambda: self.bc_type
        
        if func: self.func = func
            
        if periodic_map: self.map = periodic_map
        
    def apply(self, *args):
        """Some boundary conditions ('ConstantPressure', 'Outlet', 'Symmetry', 
        'Slip') are sometimes enforced weakly. Hence, these boundary conditions 
        should not modify tensors and by using this function they will correctly
        do-nothing. This apply method is not called in case this subdomain is 
        used to create a strong BC (like DirichletBC for pressure for a 
        ConstantPressure), because the DirichletBC has its own apply method.
        """
        pass

def solve_nonlinear(pdesubsystems, max_iter=1, max_err=1e-7, logging=True):
        """Generic solver for system of equations"""
        err = 1.
        j = 0
        err_s = " %4.4e %4.4e |"
        
        # Assemble on the first iteration?
        for pdesubsystem in pdesubsystems:
            pdesubsystem.assemble_A = pdesubsystem.prm['reassemble_lhs']
            pdesubsystem.assemble_b = pdesubsystem.prm['reassemble_rhs']
            
        total_err = ""
        # Iterate over system of equations
        while err > max_err and j < max_iter:
            j += 1 
            total_err = ""
            err = 0.            
            for pdesubsystem in pdesubsystems:
                
                res, dx = pdesubsystem.solve(assemble_A=pdesubsystem.assemble_A,
                                             assemble_b=pdesubsystem.assemble_b) 
                ndx = norm(dx)
                total_err += err_s %(res, ndx)                
                err = max(err, max(res, ndx))
                        
                # Assemble system on the next inner iteration?
                # Do nothing if pdesubsystem.assemble_% is False.
                if pdesubsystem.assemble_A:
                    pdesubsystem.assemble_A = pdesubsystem.prm['reassemble_lhs_inner']
                if pdesubsystem.assemble_b:
                    pdesubsystem.assemble_b = pdesubsystem.prm['reassemble_rhs_inner']
                                              
            # print result
            if logging:
                info_green("    Iter    %s error | " %(j)+ 
                          ' | '.join([pdesubsystem.name 
                                      for pdesubsystem in pdesubsystems]) +
                          " | " + total_err)
            
        return total_err, j
    
def max_(a, b):
    return conditional(gt(a, b), a, b)

def min_(a, b):
    return conditional(lt(a, b), a, b)
    
# Define strain-rate tensor
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

# Define rotation-rate tensor
def omega(u):
    return 0.5*(grad(u) - grad(u).T)

# Define stress
def sigma(u, p, nu):
    return 2*nu*epsilon(u) - p*Identity(u.cell().d)
    
def bound(x, maxf=1e10, minf=1e-10):
    x.set_local(minimum(maximum(minf, x.array()), maxf))        

def matrix_division(A, A_):
    F = []
    for i in range(A.rank()):
        for j in range(A.rank()): 
            F.append(A[i, j]/A_[i, j])
    return reduce(operator.add, F) 
    
def recursive_update(dst, src):
    """Update dict dst with items from src deeply ("deep update")."""
    for key, val in src.items():
        if key in dst and isinstance(val, dict) and isinstance(dst[key], dict):
            dst[key] = recursive_update(dst[key], val)
        else:
            dst[key] = val
    return dst
    
def add_BC(bc_list, V, bc, func):
    """Add boundary condition to provided list."""
    if bc.type() in ('Wall', 'VelocityInlet', 'ConstantPressure', 'Outlet'):
        if hasattr(bc, 'mf'):
            if isinstance(func, (str, list, tuple)):
                bc_list.append(DirichletBC(V, Expression(func), bc.mf, bc.bid))
            else:
                bc_list.append(DirichletBC(V, func, bc.mf, bc.bid))
            bc_list[-1].mf = bc.mf
            bc_list[-1].bid = bc.bid
        elif not V.mesh().domains().is_empty(): # Has MeshValueCollections
            bc_list.append(DirichletBC(V, func, bc.bid))
        else:
            bc_list.append(DirichletBC(V, func, bc))
    elif bc.type() == 'Periodic':
        bc_list.append(PeriodicBC(V, bc))
        
    bc_list[-1].type = bc.type

class Subdict(dict):
    """Dictionary that looks for missing keys in the solver_namespace"""
    def __init__(self, solver_namespace, sub_name, **kwargs):
        dict.__init__(self, **kwargs)
        self.solver_namespace = solver_namespace
        self.sub_name = sub_name
    
    def __missing__(self, key):
        try:
            self[key] = self.solver_namespace['prm'][key][self.sub_name]
            info_green("Adding ['{}']['{}'] = {} to pdesubsystem {}".format(key, 
                       self.sub_name, self[key], ''.join(self.sub_name)))
        except:
            self[key] = self.solver_namespace['prm'][key]
            info_green("Adding ['{}'] = {} to pdesubsystem {}".format(key, 
                       self[key], ''.join(self.sub_name)))
        return self[key]
        
class Initdict(dict):
    """Dictionary that looks for key 'u0' in 'u'[0]."""
    
    def __missing__(self, key):
        try:
            index = eval(key[-1])
            if isinstance(index, int):
                self[key] = self[key[:-1]][index]
            return self[key]
        except:
            raise KeyError

# The following helper functions are available in dolfin versions >= 0.9.9
# They are redefined here for printing only on process 0. 
RED   = "\033[1;37;31m%s\033[0m"
BLUE  = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_blue(s):
    if MPI.process_number()==0:
        print BLUE % s

def info_green(s):
    if MPI.process_number()==0:
        print GREEN % s
    
def info_red(s):
    if MPI.process_number()==0:
        print RED % s
