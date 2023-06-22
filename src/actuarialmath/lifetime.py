"""Future lifetimes - Computes expectations and moments of future lifetime

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
import math
from actuarialmath import Survival

class Lifetime(Survival):
    """Computes expected moments of future lifetime

    Examples:
      >>> def l(x, s): return 0. if (x+s) >= 100 else 1 - ((x + s)**2) / 10000.
      >>> e = Lifetime().set_survival(l=l).e_x(75, t=10, curtate=False)
      >>> def fun(omega):  # Solve first for omega, given mu_65 = 1/180
      >>>     return Lifetime().set_survival(l=lambda x,s: (1-(x+s)/omega)**0.25,
      >>>                                    maxage=omega).mu_x(65)
      >>> omega = int(Lifetime.solve(fun, target=1/180, grid=100))  # solve for omega
      >>> e = Lifetime().set_survival(l=lambda x,s: (1 - (x+s)/omega)**0.25, 
      >>>                             maxage=omega).e_x(106)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def e_x(self, x: int, s: int = 0, t: int = Survival.WHOLE, 
            curtate: bool = True, moment: int = 1) -> float:
        """Compute curtate or complete expectations and moments of life

        Args:
          x : age of selection
          s : years after selection
          t : limited at t years
          curtate : whether curtate (True) or complete (False) expectations
          moment : whether to compute first (1) or second (2) moment

        Examples:
          >>> def l(x, s): return 0. if (x+s) >= 100 else 1 - ((x + s)**2) / 10000.
          >>> print(Lifetime().set_survival(l=l).e_x(75, t=10, curtate=False))
        """
        assert moment in [1, 2, self.VARIANCE], "moment must be 1, 2 or -2"
        assert x >= 0, "x must be non-negative"
        assert s >= 0, "s must be non-negative"

        if t == 1 and curtate:      # shortcut for e_x:1
            return self.p_x(x, s=s, t=1)
        
        t = self.max_term(x+s, t)   # length of term must be bounded by max age
        if curtate:
            if moment == 1:
                return sum([self.p_x(x, s=s, t=k)
                            for k in range(1, round(t+1))]) 
            e2 = sum([((2 * k) - 1) * self.p_x(x, s=s, t=k)
                      for k in range(1, round(t+1))])
        else:
            if moment == 1:
                return self.integral(lambda t: self.S(x, s, t), 0., float(t))
            e2 = self.integral(lambda t: 2 * t * self.S(x, s, t), 0., float(t))

        if moment == self.VARIANCE:  # variance is E[T_x^2] - E[T_x]^2
            return e2 - self.e(x, s=s, t=t, curtate=curtate, moment=1)**2
        return e2   # return second moment


if __name__ == "__main__":
    print("SOA Question 2.4: (E) 8.2")
    def l(x, s): return 0. if (x+s) >= 100 else 1 - ((x + s)**2) / 10000.
    e = Lifetime().set_survival(l=l).e_x(75, t=10, curtate=False)
    print(e)

    print("SOA Question 2.1: (B) 2.5")
    def fun(omega):  # Solve first for omega, given mu_65 = 1/180
        return Lifetime().set_survival(l=lambda x,s: (1-(x+s)/omega)**0.25,
                                       maxage=omega).mu_x(65)
    omega = int(Lifetime.solve(fun, target=1/180, grid=100))  # solve for omega
    e = Lifetime().set_survival(l=lambda x,s: (1 - (x+s)/omega)**0.25, 
                                maxage=omega).e_x(106)
    print(e)    
    
