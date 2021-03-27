# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import dolfin as dl
import sys
sys.path.append( "../../" )
from hippylib import *
import numpy as np

cpp_mollifier = '''
class MyFunc : public Expression
{
public:

  MyFunc() :
  Expression(),
  locations(10)
  {
  locations[0] = .1;
  locations[1] = .1;
  locations[2] = .1;
  locations[3] = .9;
  locations[4] = .5;
  locations[5] = .5;
  locations[6] = .9;
  locations[7] = .1;
  locations[8] = .9;
  locations[9] = .9;
  }

void eval(Array<double>& values, const Array<double>& x) const
  {
        int ndim(2);
        int nlocs = locations.size()/ndim;
        Array<double> dx(ndim);
        double e(0), val(0);
        for(int ip = 0; ip < nlocs; ++ip)
        {
            for(int idim = 0; idim < ndim; ++idim)
                dx[idim] = x[idim] - locations[2*ip+idim];
                
            e = pow( dx[0]*dx[0]*c00 + dx[1]*dx[1]*c11 + 2*dx[0]*dx[1]*c01, .5*o);
            val += exp( -e/l);
        }
        values[0] = val;
  }
  
  double l;
  double c00;
  double c01;
  double c11;
  std::vector<double> locations;
  double o;
  
};
'''

if __name__ == "__main__":
    dl.set_log_active(False)
    ndim = 2
    nx = 64
    ny = 64
    mesh = dl.UnitSquareMesh(nx, ny)
    Vh = dl.FunctionSpace(mesh, "CG", 1)
    
    e = dl.Expression(cpp_mollifier)
    e.o = 2
    e.l = math.pow(.2, e.o)
    e.c00 = 2.
    e.c01 = 0.
    e.c11 = .5
    
    m = dl.interpolate(e, Vh)
    
    dl.plot(m)
    dl.interactive()