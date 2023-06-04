"""Fractional Ages

Copyright 2022, Terence Lim

MIT License
"""
from actuarialmath.lifetime import Lifetime
import math

class Fractional(Lifetime):
    """Fractional: survival functions and expected lifetimes at fractional ages
    
    - udd (bool) : UDD (True) or constant force of mortality (False) between ages
    """
    _help = ['E_r', 'l_r', 'p_r', 'q_r', 'mu_r', 'f_r', 'e_r', 'e_curtate']

    def __init__(self, udd: bool = False, **kwargs):
        super().__init__(**kwargs)   
        self.udd = udd

        # 
        # extend continuous from integer and fractional survival functions
        #
        def _mu(x: int, s: float) -> float:
            u = math.floor(s)
            return self.mu_r(x, s=u, r=s-u)
        def _l(x: int, s: float) -> float:
            u = math.floor(s)
            return self.l_r(x, s=u, r=s-u)
        def _S(x: int, s, t: float) -> float:
            u = math.floor(t)   # u+r_t_x = u_t_x * 
            return self.p_x(x, s=s, t=u) * self.p_r(x, s=s+u, t=t-u) 
        def _f(x: int, s, t: float) -> float:
            u = math.floor(t)   # f_x(u+r) = u_p_x * f_u(r)
            return self.p_x(x, s=s, t=u) * self.f_r(x, s=s+u, t=t-u)
        self.fractional = dict(mu=_mu, S=_S, f=_f, l=_l)

    #
    # Define fractional survival functions
    # 
    def E_r(self, x: int, s: int = 0, r: float = 0., t: float = 1.) -> float:
        """Pure endowment through fractional age: t_E_[x]+s+r
        - x (int) : age of selection
        - s (int) : years after selection
        - r (float) : fractional year after selection
        - t (float) : limited at fractional year t
        """
        return self.p_r(x, s=s, r=r, t=t) * self.interest.v_t(t)

    def l_r(self, x: int, s: int = 0, r: float = 0.) -> float:
        """Lives at fractional age: l_[x]+s+r
        - x (int) : age of selection
        - s (int) : years after selection
        - r (float) : fractional year after selection
        """
        s += math.floor(r)  # interpolate lives between consecutive integer ages
        r -= math.floor(r)
        if r == 0:
            return self.l_x(x, s=s)
        if r == 1.0:
            return self.l_x(x, s=s+1)
        if self.udd:
            return self.l_x(x, s=s)*(1-r) + self.l_x(x, s=s+1)*r
        else:
            return self.l_x(x, s=s)**(1-r) * self.l_x(x, s=s+1)**r

    def p_r(self, x: int, s: int = 0, r: float = 0., t: float = 1.) -> float:
        """Survival from and through fractional age: t_p_[x]+s+r
        - x (int) : age of selection
        - s (int) : years after selection
        - r (float) : fractional year after selection
        - t (float) : fractional number of years survived
        """
        r_floor = math.floor(r)
        s += r_floor
        r -= r_floor
        if 0. <= r + t <= 1.:
            if self.udd:
                return 1 - self.q_r(x, s=s, r=r, t=t)
            else:          # Constant force shortcut within int age
                return self.p_x(x, s=s)**t   # does not depend on r
        return self.l_r(x, s=s, r=r+t) / self.l_r(x, s=s, r=r)

    def q_r(self, x: int, s: int = 0, r: float = 0., t: float = 1., 
            u: float = 0.) -> float:
        """Deferred mortality within fractional ages: u|t_q_[x]+s+r
        - x (int) : age of selection
        - s (int) : years after selection
        - r (float) : fractional year after selection
        - u (float) : fractional number of years survived, then
        - t (float) : death within next fractional years t
        """
        r_floor = math.floor(r)
        s += r_floor
        r -= r_floor
        if 0 <= r + t + u <= 1:
            if u > 0:     # Dies within u|t_q = dies in t+u but not in u 
                return self.q_r(x, s=s, r=r, t=u+t)-self.q_r(x, s=s, r=r, t=u)
            if self.udd:  # UDD shortcut within integer age
                return (t * self.q_x(x, s=s)) / (1. - r * self.q_x(x, s=s))
            else:
                return 1 - self.p_r(x, s=s, r=r, t=t)
        return ((self.l_r(x, s=s, r=r+u) - self.l_r(x, s=s, r=r+u+t)) 
                / self.l_r(x, s=s, r=r))
        
    def mu_r(self, x: int, s: int = 0, r: float = 0.) -> float:
        """Force of mortality at fractional age: mu_[x]+s+r
        - x (int) : age of selection
        - s (int) : years after selection
        - r (float) : fractional year after selection
        """
        r_floor = math.floor(r)
        s += r_floor
        r -= r_floor
        if self.udd:   # UDD shortcut
            return self.q_x(x, s=s) / (1. - r*self.q_x(x, s=s))
        else:          # Constant force shortcut
            return -math.log(max(0.0001, self.p_x(x, s=s)))

    def f_r(self, x: int, s: int = 0, r: float = 0., t: float = 0.0) -> float:
        """lifetime density function at fractional age: f_[x]+s+r (t)
        - x (int) : age of selection
        - s (int) : years after selection
        - r (float) : fractional year after selection
        - t (float) : death at fractional year t
        """
        if 0. <= r + t <= 1.:   # shortcuts available within integer ages
            if self.udd:           # UDD constant q_x
                return self.q_x(x, s=s)      # does not depent on fractional age
            else:                  # Constant force shortcut
                mu = -math.log(max(0.00001, self.p_x(x, s=s)))
                return math.exp(-mu*t) * mu  # does not depend on fractional age
        else:     # survive to integer ages then extend by fractional
            r_floor = math.floor(r)
            r -= r_floor            # => r < 1 
            s += r_floor            # => x+s+r is unchanged
            t_floor = math.floor(r + t)
            u = t_floor - r         # => u + r is integer
            t = t + r - t_floor     # => t < 1
            return self.p_r(x, s=s, r=r, t=u) * self.f_r(x, s=s+u+r, t=t)

    #
    # Fractional temporary expectations of future lifetime
    #
    def e_r(self, x: int, s: int = 0, t: float = Lifetime.WHOLE) -> float:
        """Expectation of future lifetime through fractional age: e_[x]+s:t
        - x (int) : age of selection
        - s (int) : years after selection
        - t (float) : fractional year limit of expected future lifetime
        """
        if t == 0:
            return 0
        elif t < 0:   # shortcuts for complete expectation 
            if self.udd:  # UDD case
                return self.e_x(x, s=s, t=t, curtate=True) + 0.5
            else:         # Constant Force Case: compute as maxage temporary
                return self.e_r(x, s=s, t=self.max_term(x+s, t))
        elif t <= 1:  # shortcuts within integer age
            if self.udd:  # UDD case
                if t == 1:   # shortcut for UDD 1-year temporary expectation
                    return 1. - self.q_x(x, s=s)*(1/2)
                else:        # UDD formula for fractional temporary expectation
                    return (self.q_r(x, s=s, t=t) * (t/2)
                            + self.p_r(x, s=s, t=t) * t)
            else:         # Constant Force case
                mu = -math.log(max(0.00001, self.p_x(x, s=s))) # constant mu
                return (1. - math.exp(-mu*t)) / mu  # constant force formula
        else:  # apply one-year recursion formula
            return (self.e_x(x, s=s, t=1) 
                    + self.p_x(x, s=s, t=1) * self.e_r(x, s=s+1, t=t-1))

    @staticmethod 
    def e_curtate(e: float = None, e_curtate: float = None) -> float:
        """Convert between curtate and complete lifetime assuming UDD within age
        - e (float) : value of continuous lifetime, or
        - e_curtate (float) : value of curtate lifetime
        """
        if e is not None:
            return e - 0.5
        else:
            return e_curtate + 0.5

if __name__ == "__main__":
    print(Fractional.help())
