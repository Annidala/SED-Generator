import sympy as sy
import class_densities as cde
import numpy as np
from sympy.utilities.codegen import codegen
import matplotlib.pyplot as plt
import cls_derivation as cder

#params = {'dens':['neohookean', 'hgo', 'quasi_incomp_poly'],
          #'coeffs':{'C1':[sy.Symbol('C')],
                    #'k1':[sy.Symbol('k1')],
                    #'k2':[sy.Symbol('k2')],
                    #'hgo_theta':[sy.Symbol('t')], 
                    #'K':[sy.Symbol('K')], 
                    #'pow_k':[2]}
          #}

#params = {'dens':['kelvin_hyperelas_dev',  'quasi_incomp_poly'], 
          #'coeffs':{'l1':[sy.Symbol('l1')],
                    #'l2':[sy.Symbol('l2')],
                    #'l3':[sy.Symbol('l3') ],
                    #'l4':[sy.Symbol('l4')],
                    #'l5':[sy.Symbol('l5')], 
                    #'a':[2],
                    #'K':[10e3], 
                    #'pow_k':[2]}}

params = {'dens':['fourth_order_ellips'], 
          'coeffs':{'c':[sy.Symbol('c')],
                    'N':[sy.Symbol('N')],
              'r1':[sy.Symbol('r')],
                    'r2':[sy.Symbol('r')],
                    'foe_theta':[0], 
                    'K':[10e3], 
                    'pow_k':[2]}}

#params = {'dens':['yeoh', 'quasi_incomp_poly'], 
          #'coeffs':{'C0':sy.Symbol('a'),
                    #'C1':sy.Symbol('b'),
                    #'C2':0, 
                    #'K':[10e3], 
                    #'pow_k':[2]}}
          
          
machin = cder.MakeMechanicalTensors(params, sym = True, isochor = False)
machin.cpp_function(name = {'dens': 'dens_ellips', 
                            'PK2': 'pk2_stress_ellips', 
                            'tcm':'tcm_ellips'})
