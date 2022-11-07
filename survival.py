"""Survival and Mortality probability functions

Copyright 2022, Terence Lim

MIT License
"""
from typing import Callable, Tuple, Any, List, Optional, Union
from actuarialmath.life import Life
import math

class Survival(Life):
    """Fundamental Survival Functions"""

    _help = ['l_x', 'p_x', 'q_x', 'f_x', 'mu_x', 'survival_curve']

    def __init__(self, maxage: int = Life.MAXAGE, minage: int = Life.MINAGE,
                 S: Callable[[int, float, float], float] = None, 
                 f: Callable[[int, float, float], float] = None, 
                 l: Callable[[int, float], float] = None, 
                 mu: Callable[[int, float], float] = None, **kwargs):
        super().__init__(**kwargs)
        self.MAXAGE = maxage
        self.MINAGE = minage
        self.set_survival(S=S, f=f, l=l, mu=mu)
        
    def set_survival(self, 
                     S: Optional[Callable[[int, float, float], float]] = None, 
                     f: Optional[Callable[[int, float, float], float]] = None,
                     l: Optional[Callable[[int, float], float]] = None, 
                     mu: Optional[Callable[[int, float], float]] = None) -> "Survival":
        """Derive other fundamental survival functions given any one"""
        self.S = None   # survival probability: S_[x]+s(t) = t_p_[x]+s 
        self.f = None   # mortality pdf: f_[x]+s(t) = Prob[([x]+s) dies at time t]
        self.l = None   # lives aged [x]+s: l_[x]+s
        self.mu = None  # force of mortality: mu_[x]+s

        def S_from_l(x: int, s, t: float) -> float:
            return (self.l(x, s+t) / self.l(x, s)) if self.l(x, s) else 0.

        def mu_from_l(x: int, t: float) -> float:
            return -self.deriv(lambda s: math.log(self.l(x, s)), t)

        def f_from_l(x: int, s, t: float) -> float:
            return -self.deriv(lambda t: self.l(x, s+t), t)

        def mu_from_S(x: int, t: float) -> float:
            return -self.deriv(lambda s: math.log(self.S(x, 0, s)), t)

        def f_from_S(x: int, s, t: float) -> float:
            return -self.deriv(lambda t: self.S(x, s, t), t)

        def S_from_mu(x: int, s, t: float) -> float:
            return math.exp(-self.integrate(lambda t: self.mu(x, s+t), 0, t))

        def S_from_f(x: int, s, t: float) -> float:
            return 1 - self.integrate(lambda t: self.f(x, s, t), 0, t)

        def f_from_mu(x: int, s, t: float) -> float:
            return self.S(x, s, t) * self.mu(x, s+t)

        def mu_from_f(x: int, t: float) -> float:
            self.mu = self.f(x, 0, t) / self.S(x, 0, t)

        if l is not None:
            self.S = S_from_l
            self.mu = mu_from_l
            self.f = f_from_l
        if S is not None:
            self.mu = mu_from_S
            self.f = f_from_S
        if mu is not None:
            self.S = S_from_mu
            self.f = f_from_mu
        if f is not None:   # survive = 1 - cumulative deaths
            self.S = S_from_f
            self.mu = mu_from_f
        self.l = l or self.l
        self.S = S or self.S
        self.f = f or self.f
        self.mu = mu or self.mu
        return self

    #
    #  Define basic integer age survival functions
    #
    def l_x(self, x: int, s: int = 0) -> float:
        """Number of lives age ([x]+s): l_[x]+s"""
        if self.l is not None:
            return self.l(x, s)
        return Life.LIFES * self.p_x(x=self.MINAGE, s=0, t=s+x-self.MINAGE)

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """Probability that (x) lives t years: t_p_x"""
        if self.S is not None:
            return self.S(x, s, t)
        raise Exception("No functions implemented to compute survival")

    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        """Probability that (x) lives for u, but not for t+u: u|t_q_[x]+s"""
        return self.p_x(x, s=s, t=u) - self.p_x(x, s=s, t=t+u)

    def f_x(self, x: int, s: int = 0, t: int = 0) -> float:
        """probability density function of mortality"""
        if self.f is not None:
            return self.f(x, s, t)
        return self.p_x(x, s=s, t=t) * self.mu_x(x, s=s, t=t)

    def mu_x(self, x: int, s: int = 0, t: int = 0) -> float:
        """Force of mortality of (x+t): mu_x+t"""
        if self.mu is not None:
            return self.mu(x, s+t)
        return self.f_x(x, s=s, t=t) / self.p_x(x, s=s, t=t)

    def survival_curve(self) -> Tuple[List[float], List[float]]:
        """Construct curve of survival probabilities at integer ages"""
        x = list(range(self.MINAGE, self.MAXAGE))
        return x, [self.p_x(self.MINAGE, t=s) for s in x]

if __name__ == "__main__":
    print(Survival.help())
    
    print("SOA Question 2.3: (A) 0.0483")
    B, c = 0.00027, 1.1
    life = Survival(S=lambda x,s,t: (math.exp(-B * c**(x+s) 
                                     * (c**t - 1)/math.log(c))))
    print(life.f_x(x=50, t=10))
    print()
    
    print("SOA Question 2.6: (C) 13.3")
    life = Survival(l=lambda x,s: (1 - (x+s) / 60)**(1 / 3))
    print(1000*life.mu_x(35))
    print()

    print("SOA Question 2.7: (B) 0.1477")
    life = Survival(l=lambda x,s: (1 - ((x+s) / 250) if (x+s)<40 
                                  else 1 - ((x+s) / 100)**2))
    print(life.q_x(30, t=20))
    print()

    print("CAS41-F99:12: k = 41")
    fun = (lambda k: 
           Survival(l=lambda x,s: 100*(k - .5*(x+s))**(2/3)).mu_x(50))
    print(int(Survival.solve(fun, target=1/48, guess=50)))
    print()
