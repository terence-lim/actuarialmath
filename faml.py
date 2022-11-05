"""FAM-L base class

Copyright 2022, Terence Lim

MIT License
"""
from typing import Callable, Dict, Any, Tuple, List
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.special import ndtri
from scipy.stats import norm
import math
import numpy as np
import pandas as pd

class FAML:
    """Base class"""

    VARIANCE = -2   # compute variance instead of non-central moments
    WHOLE = -999    # indicate whole-life, not temporary, contingency
    MAXAGE = 130    # default oldest age
    MINAGE = 0      # default youngest age
    LIFES = 100000  # default initial number of lifes in life table
    verbose = False

    def __init__(self, interest: Dict = dict(i=0)):
        self.set_interest(**interest)
        self._help = ['solve', 'max_term']

    def __str__(self):
        return "\n".join(f"{s}():\n  {getattr(self, s).__doc__}\n" 
                         for s in self._help)

    def set_interest(self, **interest):
        """Initialize interest rate object, given any form of interest rate"""
        self.interest = self.Interest(**interest)     

    #
    # Interest rate store and math
    #
    class Interest:
        """Class for interest rate and math"""
        def __init__(self, i: float = -1., delta: float = -1., d: float = -1., 
                    v: float = -1., i_m: float = -1., d_m: float = -1., 
                    m: int = 0, v_t: Callable[[float], float] | None = None):
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
            """Returns the annuity certain factor"""
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
            """Convert to/from m'thly interest rates i, d <-> i_m, d_m"""
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
            """Double the force of interest"""
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


    #
    # Basic probability formulas
    #
    @staticmethod
    def variance(a, b, var_a, var_b, cov_ab: float) -> float:
        """Compute variance of weighted sum of two r.v."""
        return a**2 * var_a + b**2 * var_b + 2 * a * b * cov_ab

    @staticmethod
    def covariance(a, b, ab: float) -> float:
        """Compute covariance of two r.v."""
        return ab - a * b  # Cov(X,Y) = E[XY] - E[X] E[Y]

    @staticmethod
    def bernoulli(p, a: float = 1, b: float = 0, 
                  variance: bool = False) -> float:
        """Compute mean or variance of bernoulli r.v. with range (a, b)"""
        assert 0 <= p <= 1.
        return (a - b)**2 * p * (1-p) if variance else p * a + (1-p) * b

    @staticmethod
    def binomial(p: float, N: int, variance: bool = False) -> float:
        """Compute mean or variance of binomial r.v."""
        assert 0 <= p <= 1. and N >= 1
        return N * p * (1-p) if variance else N * p

    @staticmethod
    def mixture(p, p1, p2: float, N: int = 1, 
                variance: bool = False) -> float:
        """Mean and variance of mixture of two binomial r.v."""
        assert 0 <= p <= 1 and 0 <= p1 <= 1 and 0 <= p2 <= 1 and N >= 1
        mean1 = FAML.binomial(p1, N)
        mean2 = FAML.binomial(p2, N)
        if variance:
            FAML.bernoulli(p, )
            # var1 = FAML.binomial(p1, N, variance=True)
            # var2 = FAML.binomial(p2, N, variance=True)
            # return (FAML.bernoulli(p, mean1**2 + var1, mean2**2 + var2)
            #         - FAML.bernoulli(p, mean1, mean2)**2)
        else:
            return FAML.bernoulli(p, mean1, mean2)
        
    @staticmethod
    def conditional_variance(p, p1, p2: float, N: int = 1) -> float:
        """Conditional variance formula"""
        assert 0 <= p <= 1 and 0 <= p1 <= 1 and 0 <= p2 <= 1 and N >= 1
        mean1 = FAML.binomial(p1, N)
        mean2 = FAML.binomial(p2, N)
        var1 = FAML.binomial(p1, N, variance=True)
        var2 = FAML.binomial(p2, N, variance=True)
        return (FAML.bernoulli(p, mean1, mean2, variance=True)  # var of mean
                + FAML.bernoulli(p, var1, var2))           # plus mean of var

    @staticmethod
    def portfolio_percentile(mean: float, variance: float,
                             prob: float, N: int = 1) -> float:
        """Percentile of a cumulative probability in the sum of N iid r.v."""
        assert prob < 1.0
        mean *= N
        variance *= N
        return mean + ndtri(prob) * math.sqrt(variance)

    @staticmethod
    def portfolio_cdf(mean: float, variance: float, value, N: int = 1) -> float:
        """CDF Probability of a value in the sum of N iid r.v."""
        mean *= N
        variance *= N
        return norm.cdf(value, loc=mean, scale=math.sqrt(variance))

    @staticmethod
    def frame(data: List[float] = [.8, .85, .9, .95, .975, .99, .995]) -> Any:
        """Display selected values from Normal distribution table"""
        columns = [round(FAML.portfolio_percentile(0, 1, p), 3) for p in data]
        tab = pd.DataFrame.from_dict(data={'Pr(Z<=z)': data}, 
                                     columns=columns, orient='index')\
                                    .rename_axis('z', axis="columns")
        return tab.round(3)

    #
    # Helpers for numerical computations
    #
    @staticmethod
    def integrate(f: Callable[[float], float],
                  lower: float,
                  upper: float) -> float:
        """Wrapper to calculate integral"""
        y = quad(f, lower, upper, full_output=1)
        return y[0]

    @staticmethod
    def deriv(f: Callable[[float], float], x: float) -> float:
        """Wrapper to calculate derivative"""
        return derivative(f, x0=x, dx=1)

    @classmethod
    def solve(self, f: Callable[[float], float], target: int | float, 
              guess: float | Tuple | List, args: Tuple = tuple()) -> float:
        """Wrapper to solve root of equation"""
        verbose = self.verbose
        self.verbose = False
        g = lambda x: f(x, *args) - target
        if isinstance(guess, (list, tuple)):
            guess = min([(abs(g(x)), x) 
                        for x in np.linspace(min(guess), max(guess), 5)])[1]
        output = fsolve(g, [guess], full_output=True,  args=args)
        self.verbose = verbose
        return output[0][0]

    def add_term(self, t: int, n: int) -> int:
        """Add two terms, allowing either term to be whole life"""
        if t == self.WHOLE or n == self.WHOLE:
            return self.WHOLE  # adding any term to WHOLE is still WHOLE
        return t + n

    def max_term(self, x: int, t: int, u: int = 0) -> int:
        """Adjust term if adding term and deferral to (x) exceeds maxage"""
        if t < 0 or x + t + u > self.MAXAGE:
            return self.MAXAGE - (x + u)
        return t

    @staticmethod
    def ifelse(x, y: Any) -> Any:
        """keep x if it is not None, else swap in y"""
        return y if x is None else x


if __name__ == "__main__":

    print(FAML())

    raise Exception
    print("SOA Question 2.2: (D) 400")
    p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
    p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
    print(math.sqrt(FAML.conditional_variance(p=.2, p1=p1, p2=p2, N=100000)))
    print(math.sqrt(FAML.mixture(p=.2, p1=p1, p2=p2, N=100000, variance=True)))
    print()

    print("SOA Question 3.10:  (C) 0.86")
    interest = FAML.Interest(v=0.75)
    L = 35 * interest.annuity(t=4, due=False) + 75 * interest.v_t(t=5)
    interest = FAML.Interest(v=0.5)
    R = 15 * interest.annuity(t=4, due=False) + 25 * interest.v_t(t=5)
    print(L / (L + R))
    print()
    
    print("Example: double the force of interest with i=0.05")
    i = 0.05
    i2 = FAML.Interest.double_force(i=i)
    d2 = FAML.Interest.double_force(d=i/(1+i))
    print('i:', round(i2, 6), round(FAML.Interest(d=d2).i, 6))
    print('d:', round(d2, 6), round(FAML.Interest(i=i2).d, 6))

    print()
    print("Values of z for selected values of Pr(Z<=z)")
    print("-------------------------------------------")
    print(FAML.frame().to_string(float_format=lambda x: f"{x:.3f}"))
    print()
