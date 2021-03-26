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

from dolfin import compile_extension_module
import os

abspath = os.path.dirname( os.path.abspath(__file__) )
sdir = os.path.join(abspath,"cpp_rand")
header_file = open(os.path.join(sdir,"PRNG.h"), "r")
code = header_file.read()
header_file.close()
cpp_sources = ["PRNG.cpp"]  
cpp_module = compile_extension_module(
code=code, source_directory=sdir, sources=cpp_sources,
include_dirs=[".",  sdir])

Random = cpp_module.Random(1)

