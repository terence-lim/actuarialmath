"""Interest rates

Copyright 2022, Terence Lim

MIT License
"""
from typing import Callable, Dict, Any, Tuple, List, Optional, Union
import math
import numpy as np
import pandas as pd
from actuarialmath.actuarial import Actuarial

class Interest(Actuarial):
    """Convert and apply interest rates
    
    Parameters
    ----------
    i, d, v, delta (float) : assumed interest rate
    i_m, d_m (float) : or assumed monthly interest rate
    v_t (Callable) : or assumed discount rate as a function of time
    m (int) : m'thly frequency, if i_m or d_m are given
    """
    _help = ['annuity', 'mthly', 'double_force']

    def __init__(self, i: float = -1., delta: float = -1., d: float = -1., 
                v: float = -1., i_m: float = -1., d_m: float = -1., 
                m: int = 0, v_t: Optional[Callable[[float], float]] = None):
        if i_m >= 0:  # given interest rate mthly compounded
            i = self.mthly(m=m, i_m=i_m)
        if d_m >= 0:  # given discount rate mthly compounded
            d = self.mthly(m=m, d_m=d_m)
        if v_t is None:
            if delta >= 0:     # given continously-compounded rate
                self.i = math.exp(delta) - 1
            elif d >= 0:       # given annual discount rate
                self.i = d / (1 - d)
            elif v >= 0 :      # given annual discount factor
                self.i = (1 / v) - 1
            else:              # given annual interest rate
                self.i = max(i, 0)
            self.v_t = lambda t: self.v**t 
        else:   # given discount function
            self.v_t = v_t   
            self.i = (1 / v_t(1)) - 1
        self.v = 1 / (1 + self.i)         # store discount factor
        self.d = self.i / (1 + self.i)    # store discount rate
        self.delta = math.log(1 + self.i) # store continuous rate

    def annuity(self, t: int = -1, m: int = 1, due: bool = True) -> float:
        """Compute value of the annuity certain factor
        t (int) : ending year
        m (int) : m'thly frequency of payments (0 for continuous payments)
        due (bool) : whether annuity due (True) or immediate (False)
        """
        v_t = 0 if t < 0 else self.v**t   # is t finite
        if m == 0:  # if continuous
            return (1 - v_t) / self.delta
        elif due:   # if annuity due
            return (1 - v_t) / self.mthly(m=m, d=self.d)
        else:       # if annuity immediate
            return (1 - v_t) / self.mthly(m=m, i=self.i)

    @staticmethod
    def mthly(m: int = 0, i: float = -1, d: float = -1,
            i_m: float = -1, d_m: float = -1) -> float:
        """Convert to/from m'thly interest rates i, d <-> i_m, d_m
        m (int) : m'thly frequency
        i, d (float): annual-pay interest rate, to convert to m'thly, or
        i_, d_m (float): m'thly interest rate, to convert to annual pay
        """
        assert m >= 0
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
        i, d, v, delta : original interest rate to double force of interest
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
    print(L / (L + R))
    print()
    
    print("Example: double the force of interest with i=0.05")
    i = 0.05
    i2 = Interest.double_force(i=i)
    d2 = Interest.double_force(d=i/(1+i))
    print('i:', round(i2, 6), round(Interest(d=d2).i, 6))
    print('d:', round(d2, 6), round(Interest(i=i2).d, 6))

    Interest.help()
    
