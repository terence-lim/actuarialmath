"""Fractional age assumptions- Computes survival and mortality functions between integer ages.

MIT License. Copyright (c) 2022-2023 Terence Lim
"""

import math
from actuarialmath import Lifetime

class Fractional(Lifetime):
    """Compute survival functions at fractional ages and durations

    Args:
      udd : select UDD (True, default) or CFM (False) between integer ages

    Examples:
      >>> print(Fractional.e_approximate(e_complete=15))  # output e_curtate
      >>> print(Fractional.e_approximate(e_curtate=15))   # output e_complete
      >>> x = 45
      >>> life = Fractional(udd=False).set_survival(l=lambda x,t: 50-x-t)
      >>> print(life.q_r(x, r=0.), life.q_r(x, r=0.5), life.q_r(x, r=1.))
      >>> life = Fractional(udd=True).set_survival(l=lambda x,t: 50-x-t)
      >>> print(life.q_r(x, r=0.), life.q_r(x, r=0.5), life.q_r(x, r=1.))
    """

    def __init__(self, udd: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.udd_ = udd

    #
    # Define actuarial forms of survival functions at fractional ages
    # 
    def l_r(self, x: int, s: int = 0, r: float = 0.) -> float:
        """Number of lives at fractional age: l_[x]+s+r

        Args:
          x : age of selection
          s : years after selection
          r : fractional year after selection
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        assert r >= 0, "r must be non-negative"
        s += math.floor(r)  # interpolate lives between consecutive integer ages
        r -= math.floor(r)
        if r == 0:
            return self.l_x(x, s=s)
        if r == 1.0:
            return self.l_x(x, s=s+1)
        if self.udd_:
            return self.l_x(x, s=s)*(1-r) + self.l_x(x, s=s+1)*r
        else:
            return self.l_x(x, s=s)**(1-r) * self.l_x(x, s=s+1)**r

    def p_r(self, x: int, s: int = 0, r: float = 0., t: float = 1.) -> float:
        """Probability of survival from and through fractional age: t_p_[x]+s+r

        Args:
          x : age of selection
          s : years after selection
          r : fractional year after selection
          t : fractional number of years survived

        Examples:
        >>> life = Fractional(udd=False).set_survival(l=lambda x,t: 50-x-t)
        >>> print(life.p_r(47, r=0.), life.p_r(47, r=0.5), life.p_r(47, r=1.))

        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        assert r >= 0, "r must be non-negative"
        assert t >= 0, "t must be non-negative"
        r_floor = math.floor(r)
        s += r_floor
        r -= r_floor
        if 0. <= r + t <= 1.:
            if self.udd_:
                return 1 - self.q_r(x, s=s, r=r, t=t)
            else:          # Constant force shortcut within int age
                return self.p_x(x, s=s)**t   # does not depend on r
        return self.l_r(x, s=s, r=r+t) / self.l_r(x, s=s, r=r)

    def q_r(self, x: int, s: int = 0, r: float = 0., t: float = 1., 
            u: float = 0.) -> float:
        """Deferred mortality rate within fractional ages: u|t_q_[x]+s+r

        Args:
          x : age of selection
          s : years after selection
          r : fractional year after selection
          u : fractional number of years survived, then
          t : death within next fractional years t

        Examples:
          >>> life = Fractional(udd=False).set_survival(l=lambda x,t: 50-x-t)
          >>> print(life.q_r(x, r=0.5))

        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        assert r >= 0, "r must be non-negative"
        assert t >= 0, "t must be non-negative"
        assert t >= 0, "t must be non-negative"
        r_floor = math.floor(r)
        s += r_floor
        r -= r_floor
        if 0 <= r + t + u <= 1:
            if u > 0:      # Die within u|t_q == Die in t+u but not in u 
                return self.q_r(x, s=s, r=r, t=u+t) - self.q_r(x, s=s, r=r, t=u)
            if self.udd_:  # UDD shortcut within integer age
                return (t * self.q_x(x, s=s)) / (1. - r * self.q_x(x, s=s))
            else:
                return 1 - self.p_r(x, s=s, r=r, t=t)
        return ((self.l_r(x, s=s, r=r+u) - self.l_r(x, s=s, r=r+u+t)) 
                / self.l_r(x, s=s, r=r))
        
    def mu_r(self, x: int, s: int = 0, r: float = 0.) -> float:
        """Force of mortality at fractional age: mu_[x]+s+r

        Args:
          x : age of selection
          s : years after selection
          r : fractional year after selection
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        assert r >= 0, "r must be non-negative"
        r_floor = math.floor(r)
        s += r_floor
        r -= r_floor
        if self.udd_:   # UDD shortcut
            return self.q_x(x, s=s) / (1. - r*self.q_x(x, s=s))
        else:          # Constant force shortcut
            return -math.log(max(0.0001, self.p_x(x, s=s)))

    def f_r(self, x: int, s: int = 0, r: float = 0., t: float = 0.0) -> float:
        """mortality function at fractional age: f_[x]+s+r (t)

        Args:
          x : age of selection
          s : years after selection
          r : fractional year after selection
          t : death at fractional year t
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        assert r >= 0, "r must be non-negative"
        assert t >= 0, "t must be non-negative"
        if 0. <= r + t <= 1.:   # shortcuts available within integer ages
            if self.udd_:           # UDD constant q_x
                return self.q_x(x, s=s)      # does not depent on fractional age
            else:                  # Constant force shortcut
                mu = -math.log(max(0.00001, self.p_x(x, s=s)))
                return math.exp(-mu*t) * mu  # does not depend on fractional age
        else:      # survive to integer age then extend by fractional mortality
            r_floor = math.floor(r)
            r -= r_floor            # s.t. r < 1 
            s += r_floor            # while maintaining x+s+r unchanged
            t_floor = math.floor(r + t)
            u = t_floor - r         # s.t. u + r is integer
            t = t + r - t_floor     # s.t. t < 1
            return self.p_r(x, s=s, r=r, t=u) * self.f_r(x, s=s+u+r, t=t)

    #
    # Define fractional age pure endowment function
    #
        
    def E_r(self, x: int, s: int = 0, r: float = 0., t: float = 1.) -> float:
        """Pure endowment at fractional age: t_E_[x]+s+r

        Args:
          x : age of selection
          s : years after selection
          r : fractional year after selection
          t : limited at fractional year t
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        assert r >= 0, "r must be non-negative"
        assert t >= 0, "t must be non-negative"
        return self.p_r(x, s=s, r=r, t=t) * self.interest.v_t(t)

    #
    # Define fractional age expectations of future lifetime
    #
    def e_r(self, x: int, s: int = 0, t: float = Lifetime.WHOLE) -> float:
        """Temporary expected future lifetime at fractional age: e_[x]+s:t

        Args:
          x : age of selection
          s : years after selection
          t : fractional year limit of expected future lifetime
        """ 
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        if t == 0:
            return 0
        elif t < 0:   # shortcuts for complete expectation 
            if self.udd_:  # UDD case
                return self.e_x(x, s=s, t=t, curtate=True) + 0.5
            else:         # Constant Force Case: compute as maxage temporary
                return self.e_r(x, s=s, t=self.max_term(x+s, t))
        elif t <= 1:  # shortcuts within integer age
            if self.udd_:  # UDD case
                if t == 1:   # shortcut for UDD 1-year temporary expectation
                    return 1. - self.q_x(x, s=s)*(1/2)
                else:        # UDD formula for fractional temporary expectation
                    return (self.q_r(x, s=s, t=t) * (t/2)
                            + self.p_r(x, s=s, t=t) * t)
            else:         # Constant Force case
                mu = -math.log(max(0.00001, self.p_x(x, s=s)))  # constant mu
                return (1. - math.exp(-mu*t)) / mu   # constant force formula
        else:  # apply one-year recursion formula
            return (self.e_x(x, s=s, t=1) +
                    (self.p_x(x, s=s, t=1) * self.e_r(x, s=s+1, t=t-1)))

    #
    # Approximation of curtate and complete lifetimes
    #
    @staticmethod 
    def e_approximate(e_complete: float = None, e_curtate: float = None) -> float:
        """Convert between curtate and complete expectations assuming UDD shortcut

        Args:
          e_complete : complete expected lifetime
          e_curtate : or curtate expected lifetime

        Returns:
          approximate complete or curtate expectation assuming UDD
        
        Examples:
          >>> print(Fractional.e_curtate(e_complete=15))
          >>> print(Fractional.e_curtate(e_curtate=15))
        """
        if e_complete is not None:
            assert e_curtate is None, "one of e and e_curtate must be None"
            return e_complete - 0.5
        else:
            return e_curtate + 0.5

if __name__ == "__main__":
    print(Fractional.e_approximate(e_complete=15))  # output e_curtate
    print(Fractional.e_approximate(e_curtate=15))   # output e_complete

    x = 45
    life = Fractional(udd=False).set_survival(l=lambda x,t: 50-x-t)
    print(life.q_r(x, r=0.), life.q_r(x, r=0.5), life.q_r(x, r=1.))
    life = Fractional(udd=True).set_survival(l=lambda x,t: 50-x-t)
    print(life.q_r(x, r=0.), life.q_r(x, r=0.5), life.q_r(x, r=1.))
