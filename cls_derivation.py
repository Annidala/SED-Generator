# -*- coding: utf-8 -*-
import numpy as np
import sympy as sy
from sympy.utilities.codegen import codegen
import class_densities as cde
import cls_operators as co


class MakeMechanicalTensors():
    def __init__(self, params, sym = False, isochor = True):
        self.params = params
        self.sym = sym
        self.SD = cde.SymbolicDensities(self.params, self.sym)
        self.sub = [(self.SD.tens.c21, self.SD.tens.c12), 
               (self.SD.tens.c32, self.SD.tens.c23), 
               (self.SD.tens.c31, self.SD.tens.c13)]
        
    def stress_tensor(self):
        '''
        Returns a PK2 stress tensor by differentiating the density with regards 
        to C (Cauchy Green left deformation tensor)
        '''
        T = sy.derive_by_array(self.SD.density, self.SD.tens.C).tomatrix()
        return T.subs(self.sub)
    
    def tangent_matrix(self):
        '''
        Lagrangian elasticity tensor
        '''
        Ce = sy.derive_by_array(self.stress_tensor(), self.SD.tens.C)
        return Ce.subs(self.sub)    
    
    def python_function(self):
        '''
        Returns Python functions of the density, PK2 stress and tangent matrix
        '''
        tens = (self.SD.tens.c11, self.SD.tens.c22, self.SD.tens.c33, 
                self.SD.tens.c23, self.SD.tens.c13, self.SD.tens.c12)
        e = np.vectorize(sy.lambdify(tens, (self.SD.density).subs(self.sub)),
                         otypes = [list])
        f = np.vectorize(sy.lambdify(tens, 
                                     self.stress_tensor()), 
                         otypes = [np.ndarray])
        g = np.vectorize(sy.lambdify(tens, 
                                     self.tangent_matrix()), 
                         otypes = [np.ndarray])
        return e, f, g

    def cpp_function(self):
        '''
        Returns C++ functions of the density, pk2 stress and tangent matrix
        Args:
        Optional : dict of filename for the dens, stress and tangent_matrix
        '''
        d = codegen(("dens", self.SD.density), 
                                       language = "C99", 
                                       prefix = 'dens', 
                                       to_files = True, 
                                       header = False, 
                                       empty = True
                                       ) 
        pk2 = codegen(('PK2stress', self.stress_tensor()), 
                                       language = "C99", 
                                       prefix = 'stress', 
                                       to_files = True, 
                                       header = False, 
                                       empty = True) 

        tcm = codegen(
                ('tangent_matrix', sy.Matrix((self.tangent_matrix())[:,0,:,0])), 
                                        language = "C99", 
                                        prefix = 'tangent matrix', 
                                        to_files = True, 
                                        header = False, 
                                        empty = True) 
        print('Functions saved')
