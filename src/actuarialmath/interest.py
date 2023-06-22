"""Interest Theory - Applies interest rate formulas

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
import math
import numpy as np
import pandas as pd
from typing import Callable
from actuarialmath import Actuarial

class Interest(Actuarial):
    """Converts interest rates, and computes value of annuity certain
    
    Args:
      i : assumed annual interest rate
      d : or assumed discount rate
      v : or assumed discount factor
      delta : or assumed continuously compounded interest rate
      v_t : or assumed discount rate as a function of time
      i_m : or assumed monthly interest rate
      d_m : or assumed monthly discount rate
      m : m'thly frequency, if i_m or d_m are given

    Examples:
      >>> interest = Interest(v=0.75)
      >>> L = 35 * interest.annuity(t=4, due=False) + 75 * interest.v_t(t=5)
      >>> interest = Interest(v=0.5)
      >>> R = 15 * interest.annuity(t=4, due=False) + 25 * interest.v_t(t=5)
      >>> i2 = Interest.double_force(i=0.05)  # Double the force of interest
      >>> d2 = Interest(i=i2).d               # Convert interest to discount rate
      >>> i = Interest.mthly(i_m=0.05, m=12)  # Convert mthly to annual-pay
      >>> i_m = Interest.mthly(i=i, m=12)     # Convert annual-pay to mthly
    """

    def __init__(self, i: float = -1., delta: float = -1., d: float = -1., 
                 v: float = -1., i_m: float = -1., d_m: float = -1., m: int = 0,
                 v_t: Callable[[float], float] | None = None):
        if i_m >= 0:  # given interest rate mthly compounded
            i = self.mthly(m=m, i_m=i_m)
        if d_m >= 0:  # given discount rate mthly compounded
            d = self.mthly(m=m, d_m=d_m)
        if v_t is None:
            if delta >= 0:     # given continously-compounded rate
                self._i = math.exp(delta) - 1
            elif d >= 0:       # given annual discount rate
                self._i = d / (1 - d)
            elif v >= 0 :      # given annual discount factor
                self._i = (1 / v) - 1
            elif i >= 0:
                self._i = i
            else:              # given annual interest rate
                raise Exception("non-negative interest rate not given")
            self._v_t = lambda t: self._v**t 
            self._v = 1 / (1 + self._i)         # store discount factor
            self._d = self._i / (1 + self._i)    # store discount rate
            self._delta = math.log(1 + self._i) # store continuous rate
        else:   # given discount function
            assert callable(v_t), "v_t must be a callable discount function"
            assert v_t(0) == 1, "v_t(t=0) must equal 1"
            self._v_t = v_t
            #self._i = (1 / v_t(1)) - 1
            self._v = self._d = self._i = self._delta = None

    @property
    def i(self) -> float:
       """effective annual interest rate"""
       return self._i
   
    @property
    def d(self) -> float:
       """discount rate"""
       return self._d
   
    @property
    def delta(self) -> float:
       """continuously compounded interest rate"""
       return self._delta
   
    @property
    def v(self) -> float:
       """discount factor"""
       return self._v
   
    @property
    def v_t(self) -> Callable:
       """discount factor as a function of time"""
       return self._v_t
    

    def annuity(self, t: int = -1, m: int = 1, due: bool = True) -> float:
        """Compute value of the annuity certain factor

        Args:
          t : number of years of payments
          m : m'thly frequency of payments (0 for continuous payments)
          due : whether annuity due (True) or immediate (False)

        Examples:
          >>> print(interest.annuity(t=10, due=False), 2.831059)
        """
        v_t = 0 if t < 0 else self.v**t   # is t finite
        assert m >= 0, "mthly frequency must be non-negative"
        if m == 0:  # if continuous
            return (1 - v_t) / self.delta
        elif due:   # if annuity due
            return (1 - v_t) / self.mthly(m=m, d=self.d)
        else:       # if annuity immediate
            return (1 - v_t) / self.mthly(m=m, i=self.i)

    @staticmethod
    def mthly(m: int = 0, i: float = -1, d: float = -1,
            i_m: float = -1, d_m: float = -1) -> float:
        """Convert to or from m'thly interest rates

        Args:
          m : m'thly frequency
          i : an annual-pay interest rate, to convert to m'thly
          d : or annual-pay discount rate, to convert to m'thly
          i_m : or m'thly interest rate, to convert to annual pay
          d_m : or m'thly discount rate, to convert to annual pay

        Examples:
          >>> i = Interest.mthly(i_m=0.05, m=12)
          >>> print("Convert mthly to annual-pay:", i)
          >>> print("Convert annual-pay to mthly:", Interest.mthly(i=i, m=12))
        """
        assert m >= 0, "mthly frequency must be non-negative"
        if i > 0:
            return m * ((1 + i)**(1 / m) - 1) if m else math.log(1+i)
        elif d > 0:
            return m * (1 - (1 - d)**(1 / m)) if m else -math.log(1-d)
        elif i_m > 0:
            return (1 + (i_m / m))**m - 1
        elif d_m > 0:
            return 1 - (1 - (d_m / m))**m
        else:
            raise Exception("no interest rate given to mthly")

    @staticmethod
    def double_force(i: float = -1., delta: float = -1., d: float = -1., 
                        v: float = -1.):
        """Double the force of interest

        Args:
          i : interest rate to double force of interest
          d : or discount rate
          v : or discount factor
          delta : or continuous rate

        Returns:
          interest rate, of same form as input rate, after doubling force of interest

        Examples:
          >>> print("Double force of interest of i =", Interest.double_force(i=0.05))
          >>> print("Double force of interest of d =", Interest.double_force(d=0.05))
        """
        if delta >= 0:
            return 2 * delta
        elif v >= 0:
            return v**2
        elif d >= 0:
            return 2 * d - (d**2)
        elif i >= 0:
            return 2 * i + (i**2)
        else:
            raise Exception("no interest rate for double_force")

if __name__ == "__main__":
    print("SOA Question 3.10:  (C) 0.86")
    interest = Interest(v=0.75)
    L = 35 * interest.annuity(t=4, due=False) + 75 * interest.v_t(t=5)
    interest = Interest(v=0.5)
    R = 15 * interest.annuity(t=4, due=False) + 25 * interest.v_t(t=5)
    ans = L / (L + R)
    print(ans)
    
    print("Double the force of interest")
    i = 0.05
    i2 = Interest.double_force(i=i)
    print(i2)

    print("Convert interest to discount rate")
    d2 = Interest(i=i2).d
    print(d2)

    print("Convert mthly to annual-pay:")
    i = Interest.mthly(i_m=0.05, m=12)
    print(i)

    print("Convert annual-pay to mthly:")
    i_m = Interest.mthly(i=i, m=12)
    print(i_m)

    
