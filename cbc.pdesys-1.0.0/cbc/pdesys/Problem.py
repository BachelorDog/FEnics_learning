__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-01-21"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""Super class for solving systems of PDEs."""

from cbc.pdesys.PDESubSystems import *
from os import getpid
from commands import getoutput

default_problem_parameters = dict(time_integration='Transient',
                                  max_iter=1,
                                  max_err=1e-7,
                                  iter_first_timestep=1,
                                  T=1.,
                                  dt=0.01)

class Problem:
    
    def __init__(self, mesh=None, parameters=default_problem_parameters):
        self.prm = parameters
        self.mesh = mesh     
        self.boundaries = []                            # List of subdomains
        self.q0 = {}                                    # Dictionary for initializing pdesystems
        self.t = 0                                      # Simulation time
        self.tstep = 0                                  # Time step
        self.total_number_iters = 0                     #
        self.num_timesteps = 0                          #
        self.pdesystemlist = []
        self.pdesystems = {}
        
    def add_pdesystem(self, pdesystem, name):
        if name in self.pdesystemlist:
            name = name + '_'
        self.pdesystemlist.append(name)
        self.pdesystems[name] = pdesystem
        
    def remove_pdesystem(self, name):
        del self.pdesystem[name]
        self.pdesystemlist.remove(name)
        
    def solve(self, pdesystems=None, func = 'advance', logging=True, **kwargs):
        """Call either:
            solve_Transient_advance      --  Advance solution to time T
            solve_Steady_advance         --  Steady iteration
        or user defined 
            solve_Transient_something
            solve_Steady_something                
        """
        self.prm.update(**kwargs)
        if not pdesystems:
            pdesystems = [self.pdesystems[name] for name in self.pdesystemlist]
        return eval('self.solve_%s_%s(pdesystems, logging)'
                    %(self.prm['time_integration'], func))

    def solve_Steady_advance(self, pdesystems, logging):
        """Iterate solution max_iter iterations or until convergence."""        
        err = 1
        j = 0
        
        self.prepare()

        while err > self.prm['max_err'] and j < self.prm['max_iter']:
            err = 0.
            j += 1
                        
            spr = ''
            for pdesystem in pdesystems:
                
                pdesystem.prepare()
                
                spr += pdesystem.solve_inner(max_iter=pdesystem.prm['max_iter'],
                                             max_err=pdesystem.prm['max_err'],
                                             logging=False)
                
                pdesystem.solve_derived_quantities()
                
                pdesystem.update()
                
            self.total_number_iters += 1
                
            # Print result
            if logging:
                info_blue("    Iter %4s error | " %(self.total_number_iters) + 
                        ' | '.join([name
                                    for pdesystem in pdesystems
                                    for name in pdesystem.system_names]) +
                        " | " + spr)
                
                err = max([eval(s) for s in spr.replace('|','').split()] + [err])
        
        self.update()        

    def solve_Transient_advance(self, pdesystems, logging):
        """Integrate solution in time.
           
           A problem contains a list of PDESystems (e.g., for Navier-Stokes, 
           k-epsilon and a passive scalar). Each PDESystem contains a dictionary
           of PDESubSystems, called pdesubsystems. E.g., pdesubsystems['u'] and 
           pdesubsystems['p'] for velocity and pressure in a segregated 
           Navier-Stokes PDESystem.
           
           The procedure used to integrate all PDESystems in time is as follows
        
                t += dt
                
                On each timestep perform a maximum of problem.prm['max_iter']
                iterations over all PDESystems before moving to the next timestep.
                
                For each PDESystem perform a loop over the pdesubsystems. The 
                PDESubsystems in each PDESystem class are looped over 
                PDESystem.prm['max_iter'] times before moving to the next 
                PDESystem. PDESystem.prm['max_iter'] is usually 1.
                
                Each PDESubSystem is solved pdesubsystems.prm['max_inner_iter'] 
                times. pdesubsystems.prm['max_inner_iter'] is usually 1.
                                     
        """        
        err = 1
        j = 0
        num_pdesystems = len(pdesystems)
        
        while self.t < (self.prm['T'] - self.tstep*DOLFIN_EPS):
            self.t = self.t + self.prm['dt']
            self.tstep = self.tstep + 1
            
            self.prepare()
            err = 1e10
            j = 0
            # On the first timestep it may be necessary to use more timesteps
            if self.tstep==1:
                max_iter = max(self.prm['iter_first_timestep'], self.prm['max_iter'])
            else:
                max_iter = self.prm['max_iter']
            while err > self.prm['max_err'] and j < max_iter:
                err = 0.
                j += 1                
                tot_number_iters = 0
                spr = ''
                for pdesystem in pdesystems:
                    # Solve all schemes in pdesystem a given number of times
                    
                    pdesystem.prepare()
                    
                    spr += pdesystem.solve_inner(max_iter=pdesystem.prm['max_iter'], 
                                                 max_err=pdesystem.prm['max_err'],
                                                 logging=logging)
                    
                    pdesystem.solve_derived_quantities()
                    
                    pdesystem.update()
                                    
                    tot_number_iters += pdesystem.total_number_iters
                            
                # Print result
                if logging:
                    info_blue("    Iter %4s error | " %(tot_number_iters/num_pdesystems) + 
                            ' | '.join([name
                                        for pdesystem in pdesystems
                                        for name in pdesystem.system_names]) +
                            " | " + spr)
                    
                err = max([eval(s) for s in spr.replace('|','').split()] + [err])
                    
            for pdesystem in pdesystems: pdesystem.Transient_update()
            
            info_green('Time = %s, End time = %s' %(self.t, self.prm['T']))
            
            self.update()

    def initialize(self, pdesystem):
        """Initialize the solution in a PDESystem.
        This default implementation uses the dictionary attribute q0 
        that may contain tuples of strings, Constants or Expressions,
        e.g., self.q0 = {'u': ('x[1](1-x[1])', '0'),
                         'p': '0'}
        or
              self.q0 = {'u': Expression(('x[1](1-x[1])', '0')),
                         'p': Constant(0)}
        """
        if self.q0 == {}: return False
        
        q0 = self.q0
        if not isinstance(q0, dict): raise TypeError('Initialize by specifying the dictionary Problem.q0')
                                
        for sub_system in pdesystem.system_composition:
            name = ''.join(sub_system) # e.g., 'u' and 'p' for segregated solver or 'up' for coupled
            
            try:
                q = q0[name]
                if isinstance(q, (Expression, Constant)):
                    qi = interpolate(q, pdesystem.V[name])
                elif isinstance(q, (float, int)):
                    qi = interpolate(Constant(q), pdesystem.V[name])
                else:
                    qi = interpolate(Expression(q), pdesystem.V[name])

            except KeyError:
                if all(i in q0 for i in sub_system):# Add together individual parts to mixed system, e.g., use u and p for sub_system up
                    qi = []
                    for ss in sub_system: # For coupled just add individual lists
                        q = q0[ss]
                        if isinstance(q, (str)):
                            qi.append(q)
                        elif isinstance(q, (float, int)):
                            qi.append(str(q))
                        elif isinstance(q, Constant):
                            v = zeros(q.value_size())
                            x = zeros(q.value_size())
                            q.eval(v, x)
                            qi += [str(i) for i in v]
                        else:
                            qi += list(q)
                    qi = interpolate(Expression(qi), pdesystem.V[name])
                else:
                    info_red('Initial values not provided for all components of sub_system ')
                    return False
            except:
                info_red('Error in initialize! Provide tuples of strings, Constants or Expressions.')
                return False
            
            # Initialize solution:
            pdesystem.q_[name].vector()[:] = qi.vector()[:] 
            if self.prm['time_integration']=='Transient':
                pdesystem.q_1[name].vector()[:] = qi.vector()[:] 
                pdesystem.q_2[name].vector()[:] = qi.vector()[:]
                
        return True
        
    def prepare(self, *args):
        """Called at the beginning of a timestep for transient simulations or
        before iterations in steady state simulations."""
        pass
    
    def update(self, *args):
        """Called at the end of a timestep for transient simulations or at the
        end of iterations for steady simulations."""
        pass
    
    def setup(self, pdesystem):
        pass
        
    def body_force(self):
        return Constant((0.,)*self.mesh.geometry().dim())
    #def body_force(self):
        #raise NotImplementedError('Set the body force in derived class')
        
    def getMyMemoryUsage(self):
        mypid = getpid()
        mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
        return mymemory
    
def dump_result(problem, solver, cputime, error, filename = "results/results.log"):
    import os, time
    num_dofs = 0
    for name in solver.system_names:
        num_dofs += solver.V[name].dim()
       
    full_path = os.path.abspath(filename)    
    if os.path.exists(full_path): # Append to file if it exists
        file = open(full_path, 'a')

    else:
        (full_dir, fn) = os.path.split(full_path)
        if not os.path.exists(full_dir): # Create folders if they don't exist
            os.makedirs(full_dir)
        file = open(full_path, 'w')      # Create file

    if MPI.process_number() == 0:    
        
        file.write("%s, %s, %s, %d, %.15g, %.15g,  %d\n" %
                (time.asctime(), problem.__class__.__name__, solver.__class__.__name__, num_dofs, cputime, error, MPI.num_processes()))
        file.close()
