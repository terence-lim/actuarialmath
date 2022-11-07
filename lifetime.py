"""Expected future lifetimes

Copyright 2022, Terence Lim

MIT License
"""
from actuarialmath.survival import Survival
import math

class Lifetime(Survival):
    """Expected Future Lifetime"""
    _help = ['e_x']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def e_x(self, x: int, s: int = 0, t: int = Survival.WHOLE, 
          curtate: bool = True, moment: int = 1) -> float:
        """Compute moments of expected future lifetime"""
        assert moment in [1, 2, self.VARIANCE]
        if t == 1 and curtate:      # shortcut for e_x:1
            return self.p_x(x, s=s, t=1)
        t = self.max_term(x+s, t)   # length of term is bounded by max age
        if curtate:
            if moment == 1:
                return sum([self.p_x(x, s=s, t=k) for k in range(1, t+1)]) 
            e2 = sum([(2*k-1) * self.p_x(x, s=s, t=k) for k in range(1, t+1)])
        else:
            if moment == 1:
                return self.integrate(lambda t: self.S(x, s, t), 0., float(t))
            e2 = self.integrate(lambda t: 2 * t * self.S(x, s, t), 0., float(t))

        if moment == self.VARIANCE:  # variance is E[T_x^2] - E[T_x]^2
            return e2 - self.e(x, s=s, t=t, curtate=curtate, moment=1)**2
        return e2   # return second moment

if __name__ == "__main__":
    print(Lifetime.help())
    
    print("SOA Question 2.1: (B) 2.5")
    def fun(omega):  # Solve first for omega, given mu_65 = 1/180
        life = Lifetime(l=lambda x,s: (1 - (x+s)/omega)**0.25)
        return life.mu_x(65)
    omega = int(Lifetime.solve(fun, target=1/180, guess=100))  # solve for omega
    life = Lifetime(l=lambda x,s: (1 - (x+s)/omega)**0.25, maxage=omega)
    print(life.e_x(106))
    print()

    print("SOA Question 2.4: (E) 8.2")
    life = Lifetime(l=lambda x,s: 0. if (x+s) >= 100 else 1 - ((x+s)**2)/10000.)
    print(life.e_x(75, t=10, curtate=False))
    print()

    print("SOA Question 2.8: (C) 0.938")
    def fun(mu):  # Solve first for mu, given start and end proportions
        male = Lifetime(mu=lambda x,s: 1.5 * mu)
        female = Lifetime(mu=lambda x,s: mu)
        return (75 * female.p_x(0, t=20)) / (25 * male.p_x(0, t=20))
    mu = Lifetime.solve(fun, target=85/15, guess=-math.log(0.94))
    life = Lifetime(mu=lambda x,s: mu)
    print(life.p_x(0, t=1))
    print()

