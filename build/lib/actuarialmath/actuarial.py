"""Actuarial math base class

Copyright 2022, Terence Lim

MIT License
"""
from typing import Callable, Dict, Any, Tuple, List, Optional, Union
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.misc import derivative
import numpy as np

class Actuarial:
    """Some helpful utility functions"""
    _help = ['solve', 'integral', 'derivative']

    @classmethod
    def help(cls, echo=True):
        """Helper to pretty print selected docstrings"""
        s = ''
#        s = "\nclass " + cls.__name__ + " - "
        s += "\n".join([l for l in cls.__doc__.strip().split('\n')]) + "\n"
        if hasattr(cls, '_help') and cls._help:
            s += '\n    Methods defined here:\n    ' + '-'*21 + "\n"
            for method in cls._help:
                s += f"\n    {method}(...) - "
                for var in getattr(cls, method).__doc__.strip().split('\n'):
                    s += f"{var}\n"
        return print(s) if echo else s
                           

    @staticmethod
    def ifelse(x, y: Any) -> Any:
        """keep x if it is not None, else swap in y"""
        return y if x is None else x

    #
    # Helpers for numerical computations
    #
    @staticmethod
    def integral(fun: Callable[[float], float],
                 lower: float,
                 upper: float) -> float:
        """Compute integral
        
        Parameters
        ----------
        fun (Callable) : function to integrate
        lower (float) : lower bound
        upper (float) : upper bound
        """
        y = quad(fun, lower, upper, full_output=1)
        return y[0]

    @staticmethod
    def derivative(fun: Callable[[float], float], x: float) -> float:
        """Compute derivative
        
        Parameters
        ----------
        fun (Callable) : function to compute derivative
        x (float) : value to compute derivative at
        """
        return derivative(fun, x0=x, dx=1)

    @classmethod
    def solve(self, fun: Callable[[float], float], target: float, 
              guess: Union[float, Tuple, List], args: Tuple = tuple()) -> float:
        """Solve for the root of an equation

        Parameters
        ----------
        fun (Callable) : function to compute output given input values
        target (float) : target value of function output
        guess (float|List[float,float]) : initial guess, or list of guesses
        args (tuple) : optional arguments required by function f

        Returns
        -------
        root_value s.t. function output fun(root_value) == target
        """
        verbose = self.verbose
        self.verbose = False
        g = lambda x: fun(x, *args) - target
        if isinstance(guess, (list, tuple)):
            guess = min([(abs(g(x)), x)    # guess can be list of guesses
                        for x in np.linspace(min(guess), max(guess), 5)])[1]
        output = fsolve(g, [guess], full_output=True,  args=args)
        f(output[0][0], *args)   # run function one last time with final answer
        self.verbose = verbose
        return output[0][0]

if __name__ == "__main__":
    Actuarial.help()
