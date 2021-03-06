# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import logm
import sympy as sy

# Property that is only computed once. All credits (and many thanks) to Victor Couty.
def lazyproperty(f):
  @property
  def wrapper(self,*args,**kwargs):
    if not hasattr(self,'_'+f.__name__):
      setattr(self,'_'+f.__name__,f(self,*args,**kwargs))
    return getattr(self,'_'+f.__name__)
  return wrapper

def mandel_to_index(tensor):
    """
    This function transforms a Mandel tensor (symmetric tensor written in a 
    second rank tensor basis of shape (6,1)) into the Cartesian index notation (3,3)[1]
    
    Parameters
    ----------
    tensor: ndarray of shape (6,n). Along axis 0 : components of the tensor. 
    n >=1  with n the number of tensors to transform.
    
    Returns
    -------
    tens: a (3,3, n) shape array with the n cartesian index tensors.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1,9,20)
    >>> mand = np.array([x,x,1/x**2, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)])
    >>> mand.shape
    ... (6,20)
    >>> cart = mandel_to_index(mand)
    >>> cart.shape
    ... (3,3,20)
    
    [1] M. M. Mehrabadi and S. C. Cowin, Eigentensors of linear anisotropic 
    elastic materials, 
    Q. J. Mech. Appl. Math., vol. 44, no. 2, p. 331, 1991.
    """
    tens = np.array([[tensor[0], tensor[5]/np.sqrt(2), tensor[4]/np.sqrt(2)],
                     [tensor[5]/np.sqrt(2), tensor[1], tensor[3]/np.sqrt(2)],
                     [tensor[4]/np.sqrt(2), tensor[3]/np.sqrt(2), tensor[2]]])
    return tens


def index_to_mandel(tensor):
    """
    This function transforms a Cartesian index symmetric second-order tensor (3,3) 
    into the same tensor in Mandel notation (6,)[1]
    
    Parameters
    ----------
    tensor: a (3,3,n) array of n  cartesian index second order symmetric tensors
    
    Returns
    -------
    tens: a (6, n) array of the mandel tensors build from 
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1,9,20)
    >>> cart = np.array([x,x,1/x**2, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)])
    >>> cart.shape
    ... (3,3,20)
    >>> mand = index_to_mandel(cart)
    >>> mand.shape
    ... (6,20)
    
    References
    ----------
    [1] M. M. Mehrabadi and S. C. Cowin, Eigentensors of linear anisotropic 
    elastic materials, Q. J. Mech. Appl. Math., vol. 44, no. 2, p. 331, 1991.
    """
    tens = np.array([tensor[0,0],
                     tensor[1,1],
                     tensor[2,2],
                     tensor[2,1]*np.sqrt(2),
                     tensor[2,0]*np.sqrt(2),
                     tensor[1,0]*np.sqrt(2)])
    return tens



class Tensors():
    """
    Caution
    ---
    This class is still a WIP
    
    The class returns 2nd order tensors written in the Mandel notation
    
    Parameters:
    -----------
    values: a dict containing the components of the tensor(s). 
    Dict keys are {'F1', 'F2', 'F3', 'F4', 'F5', 'F6'}
    """
    def __init__(self, values):
        self.values = values
        self.F1, self.F2, self.F3, self.F4, self.F5, self.F6 = self.get_components()
        self._mandel = self.write_mandel()
        self.delonnewmandel = ["_indextens", "_invariant", "_inversetens"]
        
    @property
    def mandel(self): 
        return self._mandel
    
    @mandel.setter # La m??thode ci-dessous sera appel??es quand tu attribueras self.mandel ?? une nouvelle valeur
    def mandel(self, value):
        self._mandel = value  # Ne pas oublier de mettre ?? jour notre valeur "cach??e"
        for attr in self.delonnewmandel:
            try:
                delattr(self,attr)
            except AttributeError:
                pass

    def get_components(self):
        """
        The function transforms and stores the dict entries in a sorted list of 
        tensor components
        If key is not specified in the input dict, the value of the components 
        are replaced by 1000
        
        Parameters
        ----------
        values: from init
        
        Returns
        -------
        coords: a list of the tensor components. len(coords) = 6
        """
        defor = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
        coords = [0,0,0,0,0,0]
        for i, element in enumerate(defor):
            try:
                coords[i] = self.values[element]
            except:
                coords[i] = 1000.*np.ones_like(self.values['F1'])
        return coords
    
    def write_mandel(self):
        """
        Returns the 2nd order mandel tensors
        
        Parameters
        ----------
        coords: from init
        
        Returns
        -------
        mand: a 2nd order tensor written in a the 6 dimensional 2nd rank 
        tensor basis. Mandel notation
        """
        if type((self.F1))==np.float: 
            ''' if there's only components for one tensor (ie F1 is of len(1)), 
            the mandel tensor is written in an array of shape (6,1). 
            Adding np.newaxis transforms the usual (6,) in a (6,1)
            '''
            mand = np.array([self.F1, self.F2, self.F3, self.F4*np.sqrt(2), 
                             self.F5*np.sqrt(2), self.F6*np.sqrt(2)
                             ])[:, np.newaxis]
        else:
            mand = np.array([self.F1, self.F2, self.F3, 
                             self.F4*np.sqrt(2), 
                             self.F5*np.sqrt(2), 
                             self.F6*np.sqrt(2)])
        return mand

    @lazyproperty
    def indextens(self):
        """
        returns the 2nd order cartesian index tensor (3x3)
        """
        return mandel_to_index(self.mandel)

    @lazyproperty
    def invariant(self):
        """
        Computes the first and third invariants of a symmetric tensor, 
        using numpy function linalg.det and linalg.trace.
        
        Parameters
        ----------
        self.index: 2nd order tensors written in the Cartesian index notation
        
        Returns
        -------
        det : tensor determinant
        trace: tensor trace
        """
        det = np.array([np.linalg.det(tens) for tens in self.indextens.T]) 
        #mandel tensor is put back to the Cartesian index notation 
        #in ordr to compute the determinant.
        trace =  np.array([np.trace(tens) for tens in self.indextens.T]) 
        # same happens with the trace.
        return det, trace 
    # rajouter les invariants du tenseur, red??finir la m??thode pour s??parer 
    #les invariants
           
    @lazyproperty
    def inversetens(self):
        """
        returns the inverse tensor, making use of the numpy function linalg.inv
        (WIP : should return error when the tensor is not invertible)
        """
        if not self.invariant[0].any():
            return 'error, tensor is not invertible'
        else:
            return index_to_mandel(np.array([np.linalg.inv(tens) 
                                             for tens in self.indextens.T]).T)



class StrainTensor(Tensors):
    """
    This class inherits from the Tensors class. 
    
    Parameters
    ----------
    values: the element of the Cauchy Green Left/Cauchy Green Right tensor. 
    Values helps to build the associated Mandel tensors. 
    Dict keys are {'F1', 'F2', 'F3', 'F4', 'F5', 'F6'}
    
    """
    def __init__(self, values):
        Tensors.__init__(self, values)
        self.delonnewmandel.extend(['_deviat_tensor', 
                                    '_egreenlagrange', 
                                    '_ehencky'])
        
    @lazyproperty
    def deviat_tensor(self):
        """
        returns the deviatoric tensor. This should only be applied to 
        transformation, invertible tensor
        """
        return (self.invariant[0]**(-1/3.))*self.mandel
    
    @lazyproperty
    def egreenlagrange(self):
        """
        defines the Green Lagrange deformation tensors
        
        Parameters
        ----------
        volum: bool, default true. If False, the tensor is computed with the 
        deviatoric tensor, else with the volumic.
        
        Returns
        -------
        E: (6,n) the Green-Lagrange tensors in the the mandel notation
        """
        E = (0.5*((self.mandel).T- np.array([1,1,1,0,0,0]))).T
        Ed = (0.5*((self.deviat_tensor).T- np.array([1,1,1,0,0,0]))).T 
        # same as above but using the deviatoric cauchy green tensor
        return E, Ed
    
    @lazyproperty
    def ehencky(self):
        '''
        Hencky strain [1], computed from Cauchy Green tensor
                
        Parameters
        ----------
        volum: bool, default true. If False, the tensor is computed with the deviatoric Green Cauchy left tensor. Else with the Green Cauchy left.
        
        Returns
        -------
        E: (3,3,n) the Hencky tensors in the the mandel notation
        computation of the Hencky strain with the function logm from the 
        scipy.linalg library. 
        In order to use the numpy function, the Cauchy Green left tensor is 
        transformed from the mandel to the cartesian index notation. 
        [1] Hencky, H. (1928). "??ber die Form des Elastizit??tsgesetzes 
        bei ideal elastischen Stoffen". 
        Zeitschrift f??r technische Physik. 9: 215???220
        '''
        logC = index_to_mandel(np.array([
            0.5*logm(t) for t in mandel_to_index(self.mandel).T]).T)
        logcd = index_to_mandel(np.array([
            0.5*logm(t) for t in mandel_to_index(self.deviat_tensor).T]).T) 
        # same using the deviatoric tensor
        return logC, logcd


class SymbolicTensors():
    def __init__(self, sym = False):
        self.sym = sym
        self.c11, self.c22, self.c33, self.c23, self.c13, self.c12, self.c32, self.c31, self.c21 = sy.symbols('c11, c22, c33, c23, c13, c12, c32, c31, c21')
        self.C = self.make_tensor()

    def make_tensor(self):
        if self.sym:
            c = sy.Matrix([self.c11, self.c22, self.c33, 
                           self.c23, self.c13, self.c12])
        else:
            c = sy.Matrix([[self.c11, self.c12, self.c13],
              [self.c21, self.c22, self.c23],
              [self.c31, self.c32, self.c33]])
        return c
    
    def sym_index_tensor(self):
        C = sy.Matrix([[self.c11, self.c12, self.c13],
              [self.c12, self.c22, self.c23],
              [self.c13, self.c23, self.c33]])
        return C
    
    def I1(self):
        if self.sym:
            trace = sy.trace(self.sym_index_tensor())
        else:
            trace = sy.trace(self.C)
        return trace
    
    def I2(self):
        if self.sym:
            I2 = sy.Rational(1,2)*(sy.trace(self.sym_index_tensor())**2 - sy.trace(self.sym_index_tensor()**2))
        else:
            I2 = sy.Rational(1,2)*(sy.trace(self.C)**2 - sy.trace(self.C**2))
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
            inver = sy.Matrix([inv1[0,0], inv1[1,1], inv1[2,2], inv1[1,2], inv1[2,0], inv1[1,0]])
        else:
            inver = self.C.inv('LU')
        return inver
    
    def deviat_tensor(self):
        return self.I3()**(sy.Rational(-1,3))*self.C
