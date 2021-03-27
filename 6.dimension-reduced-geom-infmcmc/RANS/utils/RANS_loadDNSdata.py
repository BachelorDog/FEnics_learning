'''
Created on Feb 6, 2016

@author: uvilla
'''
import numpy as np
from scipy.interpolate import RectBivariateSpline
import dolfin as dl

class VelocityDNS(dl.Expression):
    def __init__(self, x, y, U, V, coflow = 0., symmetrize=True, npoints = [81,1000], bbox=[0., 20., -10., 10], kxy = [3,3]):
        
        self.x = x.reshape(npoints[0],npoints[1])[:,0]
        self.y = y.reshape(npoints[0],npoints[1])[0,:]
        self.U = U.reshape(npoints[0],npoints[1])
        self.V = V.reshape(npoints[0],npoints[1])
        self.coflow = coflow
        
        if symmetrize:
            self.U = .5*self.U + .5*self.U[:,::-1]
            self.V = .5*self.V - .5*self.V[:,::-1]
        
        self.ff_U = RectBivariateSpline(self.x, self.y, self.U, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_V = RectBivariateSpline(self.x, self.y, self.V, bbox=bbox, kx=kxy[0], ky=kxy[1])
        
    def value_shape(self):
        return (2,)
    
    def eval(self,values, x):
        values[0] = self.ff_U(x[0],x[1]) + self.coflow
        values[1] = self.ff_V(x[0],x[1])
        
class KDNS(dl.Expression):
    def __init__(self, x, y, k, apply_log=False, symmetrize=True, npoints = [81,1000], bbox=[0., 20., -10., 10], kxy = [3,3]):
        
        self.x = x.reshape(npoints[0],npoints[1])[:,0]
        self.y = y.reshape(npoints[0],npoints[1])[0,:]
                    
        k = k.reshape(npoints[0],npoints[1])

        if symmetrize:
            k = .5*(k + k[:,::-1])
        
        if apply_log:    
            self.k = np.log(k)
        else:
            self.k = k
            
        self.apply_log = apply_log
        
        self.ff = RectBivariateSpline(self.x, self.y, self.k, bbox=bbox, kx=kxy[0], ky=kxy[1])
        
    def eval(self,values, x):
        values[0] = self.ff(x[0],x[1])
        
class ProductionDNS(dl.Expression):
    def __init__(self,x,y,U,V,uu,vv,uv,symmetrize=True, apply_log=False, min_v = 1e-10, npoints = [81,1000], bbox=[0., 20., -10., 10], kxy = [3,3]):
        self.x = x.reshape(npoints[0],npoints[1])[:,0]
        self.y = y.reshape(npoints[0],npoints[1])[0,:]
        
        self.U = U.reshape(npoints[0],npoints[1])
        self.V = V.reshape(npoints[0],npoints[1])
        self.uu = uu.reshape(npoints[0],npoints[1])
        self.vv = vv.reshape(npoints[0],npoints[1])
        self.uv = uv.reshape(npoints[0],npoints[1])
        
        if symmetrize == True:
            self.U = .5*self.U + .5*self.U[:,::-1]
            self.V = .5*self.V - .5*self.V[:,::-1]
            self.uu = .5*(self.uu + self.uu[:,::-1])
            self.vv = .5*(self.vv + self.vv[:,::-1])
            self.uv = .5*(self.uv - self.uv[:,::-1])
                    
        self.ff_U = RectBivariateSpline(self.x, self.y, self.U, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_V = RectBivariateSpline(self.x, self.y, self.V, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_uu = RectBivariateSpline(self.x, self.y, self.uu, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_vv = RectBivariateSpline(self.x, self.y, self.vv, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_uv = RectBivariateSpline(self.x, self.y, self.uv, bbox=bbox, kx=kxy[0], ky=kxy[1])
        
        self.apply_log = apply_log
        self.min_v = min_v
        
    def eval(self,values, x):

        #eps = 1e-6
        #u_x = 1./(2.*1e-6)*( self.ff_U(x[0]+eps,x[1]) - self.ff_U(x[0] - eps,x[1]) )
        #u_y = 1./(2.*1e-6)*( self.ff_U.ev(x[0],x[1]+eps) - self.ff_U.ev(x[0],x[1] - eps) )
        #v_x = 1./(2.*1e-6)*( self.ff_V(x[0]+eps,x[1]) - self.ff_V(x[0] - eps,x[1]) )
        #v_y = 1./(2.*1e-6)*( self.ff_V.ev(x[0],x[1]+eps) - self.ff_V.ev(x[0],x[1] - eps) )
        u_x = self.ff_U(x[0],x[1], dx=1)
        u_y = self.ff_U(x[0],x[1], dy=1)
        v_x = self.ff_V(x[0],x[1], dx=1)
        v_y = self.ff_V(x[0],x[1], dy=1)
                    
        val = -u_x*self.ff_uu(x[0], x[1]) -  v_y*self.ff_vv(x[0], x[1]) - (u_y+v_x)*self.ff_uv(x[0], x[1])
        if self.apply_log:
            values[0] = np.log( max(val, self.min_v) )
        else:
            values[0] = max(val, 0)
                
class ReynoldsStress(dl.Expression):
    def __init__(self,x,y,uu,vv, uv, symmetrize = True, npoints = [81,1000], bbox=[0., 20., -10., 10], kxy = [3,3]):
        self.x = x.reshape(npoints[0],npoints[1])[:,0]
        self.y = y.reshape(npoints[0],npoints[1])[0,:]
        
        self.uu = uu.reshape(npoints[0],npoints[1])
        self.vv = vv.reshape(npoints[0],npoints[1])
        self.uv = uv.reshape(npoints[0],npoints[1])
        
        if symmetrize:
            self.uu = .5*(self.uu + self.uu[:,::-1])
            self.vv = .5*(self.vv + self.vv[:,::-1])
            self.uv = .5*(self.uv - self.uv[:,::-1])
        
        
        self.ff_uu = RectBivariateSpline(self.x, self.y, self.uu, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_vv = RectBivariateSpline(self.x, self.y, self.vv, bbox=bbox, kx=kxy[0], ky=kxy[1])
        self.ff_uv = RectBivariateSpline(self.x, self.y, self.uv, bbox=bbox, kx=kxy[0], ky=kxy[1])
        
    def value_shape(self):
        return (2,2,)
    
    def eval(self,values, x):
        values[0] = self.ff_uu(x[0],x[1])
        values[1] = self.ff_uv(x[0],x[1])
        values[2] = values[1]
        values[3] = self.ff_vv(x[0],x[1])
        
        
def loadDNSData(filename, l_star = 0.1, C_mu = 0.09, coflow = 0.):
    """
    Load DNS from file *filename*.
    INPUTS:
    
    - filename: name of file of DNS data.
    - l_star: default mixing lenght (to compute e from k)
    - C_mu: constant in the k-e closure model
    - coflow: coflow to be added to the DNS data
    """
    apply_log=False
    x, y, U, V, uu, vv, ww, uv, k = np.loadtxt(filename,skiprows=2, unpack=True)
    u_fun = VelocityDNS(x=x, y=y, U=U, V=V, coflow=coflow)
    k_fun = KDNS(x=x, y=y, k=k, apply_log=apply_log)
    
    if apply_log:
        e_fun = dl.Expression("log(A) - log(B) + 1.5*C", A=C_mu, B=l_star, C = k_fun)
    else:
        e_fun = dl.Expression("A/B*pow(C, 1.5)", A=C_mu, B=l_star, C = k_fun)

    return u_fun, k_fun, e_fun
