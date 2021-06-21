import sympy as sy
import class_densities as cde
import numpy as np
from sympy.utilities.codegen import codegen
import matplotlib.pyplot as plt
import cls_derivation as cder

params = {'dens':['neohookean', 'hgo', 'quasi_incomp_poly'],
          'coeffs':{'C1':[sy.Symbol('C')],
                    'k1':[sy.Symbol('k1')],
                    'k2':[sy.Symbol('k2')],
                    'hgo_theta':[sy.Symbol('t')], 
                    'K':[sy.Symbol('K')], 
                    'pow_k':[2]}
          }

#params = {'dens':['kelvin_hyperelas_dev'], 
          #'coeffs':{'l1':[sy.Symbol('a'), sy.Symbol('b')],
                    #'l2':[sy.Symbol('a'),sy.Symbol('b')],
                    #'l3':[sy.Symbol('a'),sy.Symbol('b') ],
                    #'l4':[sy.Symbol('a'),sy.Symbol('b')],
                    #'l5':[sy.Symbol('a'),sy.Symbol('b')], 
                    #'a':[sy.Symbol('x'),sy.Symbol('y')]}}

#params = {'dens':['yeoh', 'incomp'], 
          #'coeffs':{'C0':sy.Symbol('a'),
                    #'C1':sy.Symbol('b'),
                    #'C2':0, 'p': sy.Symbol('p')}}
          
          
machin = cder.MakeMechanicalTensors(params, sym = False, isochor = True)
#machin.cpp_function()
