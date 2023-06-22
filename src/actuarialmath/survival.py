"""Survival models - Computes survival and mortality functions

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
from typing import Callable, Tuple, Any, List
import math
import numpy as np
from actuarialmath import Life

class Survival(Life):
    """Set and derive basic survival and mortality functions

    Examples:
      >>> B, c = 0.00027, 1.1
      >>> def S(x,s,t): return (math.exp(-B * c**(x+s) * (c**t - 1)/math.log(c)))
      >>> f = Survival().set_survival(S=S).f_x(x=50, t=10)
      >>> def ell(x,s): return (1 - (x+s) / 60)**(1 / 3)
      >>> mu = Survival().set_survival(l=ell).mu_x(35) * 1000
      >>> def ell(x,s): return (1 - ((x+s)/250)) if x+s < 40 else (1 - ((x+s)/100)**2)
      >>> q = Survival().set_survival(l=ell).q_x(30, t=20)
      >>> def fun(k):
      >>>     return Survival().set_survival(l=lambda x,s: 100*(k - (x+s)/2)**(2/3))\
      >>>                      .mu_x(50)
      >>> k = int(Survival.solve(fun, target=1/48, grid=50))
    """
    _RADIX = 100000   # default initial number of lives in life table

    def set_survival(self,
                     S: Callable[[int,float,float], float] | None = None, 
                     f: Callable[[int,float,float], float] | None = None,
                     l: Callable[[int,float], float] | None = None, 
                     mu: Callable[[int,float], float] | None = None,
                     maxage: int = Life._MAXAGE,
                     minage: int = Life._MINAGE) -> "Survival":
        """Construct the basic survival and mortality functions given any one form

        Args:
          S : probability [x]+s survives t years
          f : or mortality of [x]+s after t years 
          l : or number of lives aged (x+t)
          mu : or force of mortality at age (x+t)
          maxage : maximum age
          minage : minimum age
        """
        self._MAXAGE = maxage
        self._MINAGE = minage
        self.S = None   # survival probability: Prob(T_[x]+s > t)
        self.f = None   # lifetime density: f_[x]+s(t) ~ Prob[([x]+s) dies at t]
        self.l = None   # number of lives aged [x]+s: l_[x]+s
        self.mu = None  # force of mortality: mu_(x+t)

        def S_from_l(x: int, s, t: float) -> float:
            """Derive survival probability from number of lives"""
            return (self.l(x, s+t) / self.l(x, s)) if self.l(x, s) else 0.

        def mu_from_l(x: int, t: float) -> float:
            """Derive force of mortality from number of lives"""
            return -self.derivative(lambda s: math.log(self.l(x, s)), t)

        def f_from_l(x: int, s, t: float) -> float:
            """Derive lifetime density function from number of lives"""
            return -self.derivative(lambda t: self.l(x, s+t), t)

        def mu_from_S(x: int, t: float) -> float:
            """Derive force of mortality from survival probability"""
            return -self.derivative(lambda s: math.log(self.S(x, 0, s)), t)

        def f_from_S(x: int, s, t: float) -> float:
            """Derive lifetime density function from survival probability"""
            return -self.derivative(lambda t: self.S(x, s, t), t)

        def S_from_mu(x: int, s, t: float) -> float:
            """Derive survival probability from force of mortality"""
            return math.exp(-self.integral(lambda t: self.mu(x, s+t), 0, t))

        def S_from_f(x: int, s, t: float) -> float:
            """Derive survival probability from lifetime density function"""
            return 1 - self.integral(lambda t: self.f(x, s, t), 0, t)

        def f_from_mu(x: int, s, t: float) -> float:
            """Derive lifetime density function from force of mortality"""
            return self.S(x, s, t) * self.mu(x, s+t)

        def mu_from_f(x: int, t: float) -> float:
            """Derive force of mortality from lifetime density function"""
            self.mu = self.f(x, 0, t) / self.S(x, 0, t)

        # derive and set all forms of basic survival and mortality functions
        if l is not None:
            assert callable(l), "l must be callable"
            self.S = S_from_l
            self.mu = mu_from_l
            self.f = f_from_l
        if S is not None:
            assert callable(S), "S must be callable"
            self.mu = mu_from_S
            self.f = f_from_S
        if mu is not None:
            assert callable(mu), "mu must be callable"
            self.S = S_from_mu
            self.f = f_from_mu
        if f is not None:
            assert callable(f), "f must be callable"
            self.S = S_from_f
            self.mu = mu_from_f
        self.l = l or self.l
        self.S = S or self.S
        self.f = f or self.f
        self.mu = mu or self.mu
        return self


    #
    #  Actuarial forms of survival and mortality functions, at integer ages
    #
    def l_x(self, x: int, s: int = 0) -> float:
        """Number of lives at integer age [x]+s: l_[x]+s

        Args:
          x : age of selection
          s : years after selection
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        if self.l is not None:
            return self.l(x, s)
        return Life.LIFES * self.p_x(x=self._MINAGE, s=0, t=s+x-self._MINAGE)

    def d_x(self, x: int, s: int = 0) -> float:
        """Number of deaths at integer age [x]+s: d_[x]+s

        Args:
          x : age of selection
          s : years after selection
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        return self.l_x(x=x, s=s) - self.l_x(x=x, s=s+1)

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """Probability that [x]+s lives another t years: : t_p_[x]+s

        Args:
          x : age of selection
          s : years after selection
          t : survives at least t years
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        if self.S is not None:
            return self.S(x, s, t)
        raise Exception("No functions implemented to compute survival")

    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        """Probability that [x]+s lives for u, but not t+u years: u|t_q_[x]+s

        Args:
          x : age of selection
          s : years after selection
          u : survives u years, then
          t : dies within next t years
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        return self.p_x(x, s=s, t=u) - self.p_x(x, s=s, t=t+u)

    def f_x(self, x: int, s: int = 0, t: int = 0) -> float:
        """Lifetime density function of [x]+s after t years: f_[x]+s(t)

        Args:
          x : age of selection
          s : years after selection
          t : dies at year t
        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        if self.f is not None:
            return self.f(x, s, t)
        return self.p_x(x, s=s, t=t) * self.mu_x(x, s=s, t=t)

    def mu_x(self, x: int, s: int = 0, t: int = 0) -> float:
        """Force of mortality of [x] at  s+t years: mu_[x](s+t)

        Args:
          x : age of selection
          s : years after selection
          t : force of mortality at year t

        Examples:
          >>> def S(x, s): return (1 - (x+s) / 60)**(1 / 3)
          >>> print(Survival().set_survival(l=S).mu_x(35))

        """
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"
        if self.mu is not None:
            return self.mu(x, s+t)
        return self.f_x(x, s=s, t=t) / self.p_x(x, s=s, t=t)

#    def survival_curve(self, x: int, s: int = 0, stop: int = 0) -> Tuple[List, List]:
#        """Construct curve of survival probabilities at each integer age
#
#        Args:
#          x : age of selection
#          s : years after selection
#          stop : end at time t, inclusive
#
#        Returns:
#          lists of lifetime and survival probability from t, S_x(t) respectively
#        """
#        stop = stop or self._MAXAGE - (x + s)
#        steps = range(stop + 1)
#        return steps, [self.p_x(x=x, s=s, t=t) for t in steps]

if __name__ == "__main__":
    print("SOA Question 2.3: (A) 0.0483")
    B, c = 0.00027, 1.1
    def S(x,s,t): return (math.exp(-B * c**(x+s) * (c**t - 1)/math.log(c)))
    f = Survival().set_survival(S=S).f_x(x=50, t=10)
    print(f)
    
    print("SOA Question 2.6: (C) 13.3")
    def ell(x,s): return (1 - (x+s) / 60)**(1 / 3)
    mu = Survival().set_survival(l=ell).mu_x(35) * 1000
    print(mu)

    print("SOA Question 2.7: (B) 0.1477")
    def ell(x,s): return (1 - ((x+s)/250)) if x+s < 40 else (1 - ((x+s)/100)**2)
    q = Survival().set_survival(l=ell).q_x(30, t=20)
    print(q)

    print("CAS41-F99:12: k = 41")
    def fun(k):
        return Survival().set_survival(l=lambda x,s: 100*(k - (x+s)/2)**(2/3))\
                         .mu_x(50)
    print(Survival.solve(fun, target=1/48, grid=50))

