# -*- coding: utf-8 -*-
import numpy as np
import sympy as sy
import cls_operators as co
import sympy as sy

def non_gauss_density(x, N):
    x = x**2/N
    L0 = 3 ; L1 = 9/5. ; 
    L2 = 297/175. ; L3 = 1539/875. 
    L4 = 126117/67375. ; L5 = 43733439/21896875.
    L6 = 231321177/109484375. ; L7 = 20495009043/9306171875. 
    L8 = 1073585186448381/476522530859375. ; L9 = 4387445039583/1944989921875.
    return (x**1*L0/2 + x**2*L1/4 +
            x**3*L2/6 + x**4*L3/8 +
            x**5*L4/10 + x**6*L5/12 +
            x**7*L6/14 + x**8*L7/16+
            x**9*L8/18 + x**10*L9/20)

foo = co.FourthOrderOperators()

class SymbolicDensities():
    '''
    '''
    def __init__(self, params, sym = False, deviat = True):
        self.sym = sym
        self.isochor = deviat
        self.model = params['dens']
        self.coeffs = params['coeffs']
        self.tens = co.SymbolicTensors(self.sym, self.isochor)
        self.density = self.make_density()
        
    def make_density(self):
        dens = 0
        for element in (self.model):
            dens+= getattr(self, element)()
        return dens

    # deviatoric densities
    def neohookean(self):
        '''
        parameters 'C0', 'C1', 'C2'
        Sym = False
        '''
        C1= self.coeffs['C1']
        W = C1[0]*(self.tens.I1()-3)
        return W
    
    def yeoh(self):
        '''
        parameters 'C0', 'C1', 'C2'
        Sym = False
        '''
        C0, C1, C2 = self.coeffs['C0'], self.coeffs['C1'], self.coeffs['C2']
        W = C0*(self.tens.I1()-3) + C1*(self.tens.I1()-3)**2 + C2*(
            self.tens.I1()-3)**3
        return W
    
    def mooney_rivlin(self):
        '''
        Parameters 'C1' and 'C2'
        Sym = False
        '''
        C1, C2 = self.coeffs['C1'], self.coeffs['C2']
        W = C1*(self.tens.I1()-3) + C2*(self.tens.I2() - 3)
        return W
    
    def kelvin_hyperelas_dev(self):
        '''
        Parameters 'li' w/ i from 1 to 5 and 'a'
        Sym = True
        '''
        l1, l2, l3, l4, l5 , a = self.coeffs['l1'], self.coeffs['l2'
            ], self.coeffs['l3'], self.coeffs['l4'], self.coeffs['l5'
                ], self.coeffs['a'] # Kelvin modulus and power order of the law
        proj = foo.get_projectors()[1] 
        # kelvin projectors - preferably choose only deviatoric projectors
        vaps = np.array([l1, l2, l3, l4, l5])
        vapp = vaps/np.sum(vaps, axis = 0)
        if self.isochor:
            E = 1/2*(self.tens.isochoric_tensor() - sy.Matrix([1, 1, 1, 0, 0, 0]))
        else:
            E = 1/2*(self.tens.C - sy.Matrix([1, 1, 1, 0, 0, 0]))
        W = 0
        for i, powa in enumerate(a):
            vapu = vapp[i]
            dens = 0
            for j, l in enumerate(vapu):
                dens+= ((E.T)*l*proj[j]*E)[0]
            W+= 1/(2*(powa+1))*(vaps[i]).sum()*dens**(powa+1)
        return W
    
    def fourth_order_ellips(self):
        '''
        Parameters:
        'c': rigidity of a chain
        'N': average length of a chain
        'r1': semi-ellipsoid axis (r1<1)
        'r2': second semi-ellipsoid axis (r2<1)
        'theta': angle of 
        Sym = True
        '''
        c, N, r1, r2, theta = self.coeffs['c'], self.coeffs['N'], self.coeffs[
            'r1'], self.coeffs['r2'], self.coeffs['foe_theta']
        if self.isochor:
            E = 1/2*(self.tens.isochoric_tensor() - sy.Matrix([1, 1, 1, 0, 0, 0]))
        else:
            E = 1/2*(self.tens.C - sy.Matrix([1, 1, 1, 0, 0, 0]))
        W = 0
        for i in range(len(c)):
            U = foo.create_ellips(r1[i], r2[i], theta[i])
            Eu = U*E
            nu = (Eu.T*Eu)[0] + 1
            W += c[i]*N[i]*non_gauss_density(nu, N[i])
        return W
        
    def hgo(self):
        k1, k2, theta = self.coeffs['k1'], self.coeffs['k2'
            ], self.coeffs['hgo_theta']
        W = 0
        if self.isochor:
            c = self.tens.isochoric_tensor()
        else:
            c = self.tens.C
        for i, k in enumerate(k1):
            u1 = sy.Matrix([sy.cos(theta[i]), sy.sin(theta[i]), 0])
            u2 = sy.Matrix([sy.cos(-theta[i]), sy.sin(-theta[i]), 0])
            I4 = (u1.T*c*u1)[0]
            I6 = (u2.T*c*u2)[0]
            W+= sy.Rational(1,2)*k/k2[i]*(sy.exp(k*(I4-1)**2) 
                                          + sy.exp(k*(I6 -1)**2) - 2)
        return W
    
    # volumetric densities
    def quasi_incomp_poly(self):
        '''
        Parameter : 'K'
        Sym = True or False
        '''
        K, pk = self.coeffs['K'], self.coeffs['pow_k']
        W = 0
        for i in range(len(K)):
            W += K[i]*(self.tens.I3() - 1)**pk[i]
        return W
    
    def incomp(self):
        '''
        Penalty function with hydrostatic pressure p
        Parameter : 'p'
        Sym = True or False
        '''
        p = self.coeffs['p']
        W = p[0]*(self.tens.I3() - 1)
        return W
