"""Base class for actuarial math, with common utility functions and constants

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
import math
import numpy as np
import scipy
from typing import Callable, Any, Tuple, List

class Actuarial(object):
    """Define constants and common utility functions

    Constants:
      VARIANCE : select variance as the statistical moment to calculate

      WHOLE : indicates that term of insurance or annuity is Whole Life
    """
    # constants
    VARIANCE = -2
    WHOLE = -999
    _VARIANCE = VARIANCE
    _WHOLE = WHOLE
    _TOL = 1e-6
    _verbose = 0
    _MAXAGE = 130    # default oldest age
    _MINAGE = 0      # default youngest age

    @classmethod
    def methods(cls, echo: bool = True):
        """Helper to partially display docstrings from classes and methods

        Args:
          echo : print if True, else return as text string
        """
        s = "\nclass " + cls.__name__ + " - "
        s += "\n".join([l for l in cls.__doc__.strip().split('\n')]) + "\n"
        if hasattr(cls, '_methods') and cls._methods:
            s += '\n    Methods:\n    --------\n\n'
            for method in cls._methods:
                meth = getattr(cls, method)
                desc = meth.__doc__.strip().split('\n')[0]
                if callable(meth):
                    args = [str(arg) for arg in signature(meth).parameters.keys()
                            if str(arg) not in ['cls', 'self']]
                else:
                    args = []
                s += f"    {method}(" + ", ".join(args) + f"):\n{' '*6}{desc}\n\n"
        return print(s) if echo else s

    #
    # Helpers for numerical computations
    #
    @staticmethod
    def integral(fun: Callable[[float], float],
                 lower: float,
                 upper: float) -> float:
        """Compute integral of the function between lower and upper limits
        
        Args:
          fun : function to integrate
          lower : lower limit
          upper : upper limit
        """
        y = scipy.integrate.quad(fun, lower, upper, full_output=1)
        return y[0]

    @staticmethod
    def derivative(fun: Callable[[float], float], x: float) -> float:
        """Compute derivative of the function at a value
        
        Args:
          fun : function to compute derivative
          x : value to compute derivative at
        """
        return scipy.misc.derivative(fun, x0=x, dx=1)

    @staticmethod
    def solve(fun: Callable[[float], float], target: float, 
              grid: float | Tuple | List, mad: bool = False) -> float:
        """Solve for the root of, or parameter value that minimizes, an equation

        Args:
          fun : function to compute output given input values
          target : target value of function output
          grid : initial guesses or bounds
          root : whether solve root, or minimize absolute difference (False)

        Returns:
          value s.t. output of function fun(value) ~ target
        """
        if mad:   # minimize absolute difference
            f = lambda t: abs(fun(t) - target)
            return scipy.optimize.minimize_scalar(f, grid).x    
        else:     # solve root
            f = lambda x: fun(x) - target
            if isinstance(grid, (list, tuple)):
                grid = min([(abs(f(x)), x)    # guess can be list of guesses
                            for x in np.linspace(min(grid), max(grid), 5)])[1]
            output = scipy.optimize.fsolve(f, [grid], full_output=True)
            fun(output[0][0])  # execute with final answer in case want side effects
            return output[0][0]

    def add_term(self, t: int, n: int) -> int:
        """Add two terms, either term may be Whole Life

        Args:
          t : first term to add
          n : second term to add
        """
        if t == self.WHOLE or n == self.WHOLE:
            return self.WHOLE  # adding any term to WHOLE is still WHOLE
        return t + n

    def max_term(self, x: int, t: int, u: int = 0) -> int:
        """Adjust term if adding term and deferral periods to (x) exceeds maxage

        Args:
          x : age
          t : term of insurance or annuity, after deferral period
          u : term deferred

        Returns:
          value of term t adjusted by deferral and maxage s.t. does not exceed maxage
        """
        if t < 0 or x + t + u > self._MAXAGE:
            return self._MAXAGE - (x + u)
        return t

if __name__ == "__main__":
    actuarial = Actuarial()
    def as_term(t): return "WHOLE_LIFE" if t == Actuarial.WHOLE else t
    
    for a,b in [(3, Actuarial.WHOLE), (3, 2), (3, -1)]:
        print(f"({as_term(a)}) + ({as_term(b)}) =",
              as_term(actuarial.add_term(a, b)))
        
    print(Actuarial.solve(fun=lambda omega: 1/omega, 
                          target=0.05,
                          grid=[1, 100]))
    print(Actuarial.derivative(fun=lambda x: x/50, x=25))