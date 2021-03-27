import dolfin as dl

class GradingFunctionLin(dl.Expression):
    """
    Grade a uniform unit square mesh so that:
    - the mesh size is constant from 0 to cut_point[0]
    - it increases (slope > 0) or decrease (slope < 0) linearly from cut_point[0] to cut_point[1]
    - it is constant again from cut_point[1] to 1
    
    h profile (assuming a uniform mesh as input and slope > 0):
               ________    
              /
             /
    ________/
    0      c0 c1      1
    
    See FreeJet_Geometry in RANS_geometry for an usage example
    """
    def __init__(self, coordinate, cut_point, slope):
        self.i = coordinate
        self.cut_point = cut_point
        self.slope = slope
        
        self.scaling = self._integral(1.)
        
    def _spacing(self, x):
        
        if x < self.cut_point[0]:
            h = 1.
        elif x < self.cut_point[1]:
            h = 1. + self.slope*(x - self.cut_point[0])
        else:
            h = 1. + self.slope*(self.cut_point[1] - self.cut_point[0])
            
        return h
    
    def _integral1(self, x):
        assert x <= self.cut_point[0]
        return self._spacing(x)*x
    
    def _integral2(self, x):
        assert x >= self.cut_point[0] and x <= self.cut_point[1]
        xdiff = x - self.cut_point[0]
        val = self._integral1(self.cut_point[0]) + xdiff + .5*self.slope*(xdiff**2)
        return val
    
    def _integral3(self, x):
        assert x >= self.cut_point[1]
        xdiff = x - self.cut_point[1]
        val = self._integral2(self.cut_point[1]) + self._spacing(x)*xdiff
        return val
           
    def _integral(self, x):
        if x < self.cut_point[0]:
            val = self._integral1(x)
        elif x < self.cut_point[1]:
            val = self._integral2(x)
        else:
            val = self._integral3(x)
               
        return val
    
    def value_shape(self):
        return (2,)
        
    def eval(self, values, x):
        for i in range(len(values)):
            values[i] = 0
        values[self.i] = self._integral(x[self.i])/self.scaling - x[self.i]
        
class Remap(dl.Expression):
    """
    Remap a mesh in [0,1] by [0,1] to [a, b] by [c d]
    box = [a, b, c, d]
    
    See FreeJet_Geometry in RANS_geometry for an usage example
    """
    def __init__(self, box):
        self.x0 = box[0]
        self.y0 = box[2]
        self.mx = box[1] - box[0]
        self.my = box[3] - box[2]
        
    def value_shape(self):
        return (2,)
    
    def eval(self, values, x):
        values[0] = self.x0 + self.mx*x[0] - x[0]
        values[1] = self.y0 + self.my*x[1] - x[1]
    