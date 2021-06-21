# -*- coding: utf-8 -*-
import sympy as sy
import numpy as np

P1 = sy.Rational(1,6)*sy.Matrix([[4,-2,-2,0,0,0],
               [-2,1,1,0,0,0],
               [-2,1,1,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]])
P2 = sy.Rational(1,2)*sy.Matrix([[0,0,0,0,0,0],
            [0,1,-1,0,0,0],
            [0,-1,1,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]])
P3 = sy.Matrix([[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]])
P4 = sy.Matrix([[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,0]])
P5 = sy.Matrix([[0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,1]])
Ph = sy.Rational(1,3)*sy.Matrix([[1,1,1,0,0,0],
                                 [1,1,1,0,0,0],
                                 [1,1,1,0,0,0],
                                 [0,0,0,0,0,0],
                                 [0,0,0,0,0,0],
                                 [0,0,0,0,0,0]])

def passage_mandelmatrix(m):
    '''
    Creates a 6x6 4th-order tensor from a 3x3 2nd-order tensor
    input: m is a 3x3 array
    '''
    M = sy.Matrix([[m[0,0]**2, m[0,1]**2, m[0,2]**2, 
                    sy.sqrt(2)*m[0,1]*m[0,2], 
                    sy.sqrt(2)*m[0,0]*m[0,2], 
                    sy.sqrt(2)*m[0,0]*m[0,1]],
    [m[1,0]**2, m[1,1]**2, m[1,2]**2, sy.sqrt(2)*m[1,1]*m[1,2], 
        sy.sqrt(2)*m[1,0]*m[1,2], sy.sqrt(2)*m[1,1]*m[1,0]],
    [m[2,0]**2, m[2,1]**2, m[2,2]**2, sy.sqrt(2)*m[2,2]*m[2,1], 
        sy.sqrt(2)*m[2,2]*m[2,0], sy.sqrt(2)*m[2,1]*m[2,0]],
    [sy.sqrt(2)*m[1,0]*m[2,0], sy.sqrt(2)*m[1,1]*m[2,1], 
        sy.sqrt(2)*m[1,2]*m[2,2], m[1,1]*m[2,2] + m[1,2]*m[2,1], 
        m[1,2]*m[2,0] + m[1,0]*m[2,2], m[1,1]*m[2,0] + m[1,0]*m[2,1]],
    [sy.sqrt(2)*m[0,0]*m[2,0], sy.sqrt(2)*m[0,1]*m[2,1], 
        sy.sqrt(2)*m[0,2]*m[2,2], m[0,1]*m[2,2] + m[0,2]*m[2,1], 
        m[0,0]*m[2,2] + m[2,0]*m[0,2], m[0,0]*m[2,1] + m[2,0]*m[0,1]],
    [sy.sqrt(2)*m[0,0]*m[1,0], sy.sqrt(2)*m[0,1]*m[1,1], 
        sy.sqrt(2)*m[0,2]*m[1,2], m[0,1]*m[1,2] + m[1,1]*m[0,2], 
        m[0,0]*m[1,2] + m[1,0]*m[0,2], m[0,0]*m[1,1] + m[1,0]*m[0,1]]])
    return M

class FourthOrderOperators():
    """
    The present class returns any 4th-rank tensors written in the 6-dimensional 
    second-rank tensor notation [1]
    
    References
    ----------
    [1] M. M. Mehrabadi and S. C. Cowin, Eigentensors of linear anisotropic 
    elastic materials, Q. J. Mech. Appl. Math., vol. 44, no. 2, p. 331, 1991.
    Parameters
    ----------
    C : Default None. When specified C is the fourth-rank elasticity tensor 
    written in a 6-dimensional tensor space. It is a (6,6) symmetric Matrix.
    Methods:
    - get_proj : diagonalize the C tensor, returns the vap and vep of the 
    elasticity tensor. Returns the isotropic vep if C is not specified
    - get_rot_tens: returns the Auld rotation matrix (6x6) given an angle and
    axis of rotation ('x', 'y', 'z'). Default parameters: theta = 0, axis = 'z'
    - passage_mandelmatrix : returns the Auld matrix (6x6) given any 3x3 
    passage matrix
    """
    def __init__(self, C = None):
        '''
        If provided, C is an elasticity tensor and therefore provided as a 6x6 
sympy matrix 
        '''
        self.C = C
        self.vap = self.get_projectors()[0]
        (self.P1, self.P2, self.P3, 
            self.P4, self.P5, self.Ph)  = self.get_projectors()[1] 
        self.projs = self.get_projectors()
        self.rot = self.get_rot_tens()
        
    def get_projectors(self):
        """
        This class method gives the Kelvin projectors and modules associated 
        with the elasticity tensor C
        If the parameters C is not specified the function only returns isotropic
        projectors. (this has to be corrected in the future)
        
        Returns
        -------
        vap: eigenvalues of tensor C
        vep: eigenvectors of tensor C
        """
        if self.C == None: 
            vep = (P1, P2, P3, P4, P5, Ph)
            vap = 0
        else:
            vap, vep = sy.linalg.eig(C)
        return vap, vep
        
    def get_rot_tens(self, theta = sy.Symbol('t'), axe = 'z'):
        """
        The method returns the rotation matrix along a precised axis in 
        the 6 dimensional tensor notation [1]
        
        Parameters
        ----------
        theta: float, an angle in radians
        axe: str, the axe of rotation, in the present version, axe is equal 
        to 'x', 'y' or 'z'
        
        Returns
        -------
        R: a (6,6) rotation Matrix. Works only to rotate mandel written 
        2nd-order tensors.
        
        [1] B. A. Auld : Acoustic Fields and Waves in Solids, vol. 2. 
        Robert E.Kreiger Publishing Compagny, Inc., Kriger Drive, Florida 3950,
        1973. ISBN 0-89874-782-1.
        """
        c = sy.cos(theta)
        s = sy.sin(theta)

        if axe =='z':
            m = sy.Matrix([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]])
        elif axe =='y':
            m = sy.Matrix([[c, 0, -s],
                           [0, 1, 0],
                           [s, 0 , c]])
        elif axe =='x':
            m = sy.Matrix([[1, 0, 0],
                           [0, c,  -s],
                           [0, s, c]])
        else:
            print('error, axis should be x, y or z')
        R = passage_mandelmatrix(m)
        return R

    
    def create_ellips(self, r1, r2, theta = sy.Symbol('t')):
        """
        create an ellipsoid fourth-order matrix
        ellipsoid in the e1/e2 plane with a theta inclination
        """
        p = sy.Matrix([[1,0,0],[0, r1**0.5, 0], [0, 0, r2**0.5]])
        P = passage_mandelmatrix(p)
        Pt = self.get_rot_tens(theta).T*(P*(self.get_rot_tens(theta)))
        return Pt
    
class SymbolicTensors():
    '''
    The SymbolicTensors class returns 2nd-order tensors in the Mandel or 
    Cartesian representation.
    Args :
    sym : Boolean - True enforces a 6x1 Mandel representation, 
          False  the tensor is non-symmetric and therefore written in Cartesian
    Init returns:
    C : a symbolic Matrix of shape in 6x1 or 3x3.
    Methods include the computation of invariants, inverse and isochororic 
    tensors.
    '''
    def __init__(self, sym = False, isochor = True):
        self.sym = sym
        self.isochor = isochor
        (self.c11, self.c22, self.c33, 
            self.c23, self.c13, self.c12, 
            self.c32, self.c31, self.c21) = sy.symbols(
                'c11, c22, c33, c23, c13, c12, c32, c31, c21')
        self.C = self.make_tensor()
        
    def make_tensor(self):
        '''
        The method returns a 6x1 Mandel tensor (vector shape) or a non-symmetric
        3x3 tensor depending on the class input sym
        '''
        if self.sym:
            c = sy.Matrix([self.c11, self.c22, self.c33, 
                           self.c23, self.c13, self.c12])
        else:
            c = sy.Matrix([[self.c11, self.c12, self.c13],
              [self.c21, self.c22, self.c23],
              [self.c31, self.c32, self.c33]])
        return c
    
    def sym_index_tensor(self):
        '''
        The method provides a 3x3 symmetric tensor 
        '''
        C = sy.Matrix([[self.c11, self.c12, self.c13],
              [self.c12, self.c22, self.c23],
              [self.c13, self.c23, self.c33]])
        return C
    
    def I1(self):
        '''
        Method to obtain the first invariant, ie trace of the matrix
        '''
        if self.sym:
            trace = sy.trace(self.sym_index_tensor())
        else:
            trace = sy.trace(self.C)

        if self.isochor:
            trace = (self.I3())**sy.Rational(-1,3)*trace
        return trace
    
    def I2(self):
        if self.sym:
            I2 = sy.Rational(1,2)*(sy.trace(self.sym_index_tensor())**2 - 
                                   sy.trace(self.sym_index_tensor()**2))
        else:
            I2 = sy.Rational(1,2)*(sy.trace(self.C)**2 - sy.trace(self.C**2))
        if self.isochor:
            I2 = I2*(self.I3())**sy.Rational(-2,3)        
        return I2
    
    def I3(self):
        if self.sym:
            I3 = (self.sym_index_tensor()).det()
        else:
            I3 = self.C.det()
        return I3
        
    def inverse(self):
        if self.sym:
            inv1 = self.sym_index_tensor().inv('LU')
            inver = sy.Matrix([inv1[0,0], inv1[1,1], inv1[2,2], 
                               inv1[1,2], inv1[2,0], inv1[1,0]])
        else:
            inver = self.C.inv('LU')
        return inver
    
    def isochoric_tensor(self):
        return self.I3()**(sy.Rational(-1,3))*self.C
