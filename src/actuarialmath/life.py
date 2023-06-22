"""Life Contingent Risks - Applies probability laws

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
import math
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import norm
from typing import Callable, Dict, Any, Tuple, List
from actuarialmath import Actuarial, Interest

class Life(Actuarial):
    """Compute moments and probabilities

    Examples:
      >>> p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
      >>> p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
      >>> conditional = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
      >>> mixture = math.sqrt(Life.mixture(p=.2, p1=p1, p2=p2, N=100000, variance=True))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_interest(i=0)

    def set_interest(self, **interest) -> "Life":
        """Set interest rate, which can be given in any form

        Args:
          i : assumed annual interest rate
          d : or assumed discount rate
          v : or assumed discount factor
          delta : or assumed contiuously compounded interest rate
          v_t : or assumed discount rate as a function of time
          i_m : or assumed monthly interest rate
          d_m : or assumed monthly discount rate
          m : m'thly frequency, if i_m or d_m are given
        """
        self.interest = Interest(**interest)
        return self

    #
    # Probability theory
    #
    @staticmethod
    def variance(a, b, var_a, var_b, cov_ab: float) -> float:
        """Variance of weighted sum of two r.v.

        Args:
          a : weight on first r.v.
          b : weight on other r.v.
          var_a : variance of first r.v.
          var_b : variance of other r.v.
          cov_ab : covariance of the r.v.'s
        """
        return a**2 * var_a + b**2 * var_b + 2 * a * b * cov_ab

    @staticmethod
    def covariance(a, b, ab: float) -> float:
        """Covariance of two r.v.

        Args:
          a : expected value of first r.v.
          b : expected value of other r.v.
          ab : expected value of product of the two r.v.
        """
        return ab - a * b  # Cov(X,Y) = E[XY] - E[X] E[Y]

    @staticmethod
    def bernoulli(p, a: float = 1, b: float = 0,
                  variance: bool = False) -> float:
        """Mean or variance of bernoulli r.v. with values {a, b}

        Args:
          p : probability of first value
          a : first value
          b : other value
          variance : whether to return variance (True) or mean (False)
        """
        assert 0 <= p <= 1.
        return (a - b)**2 * p * (1-p) if variance else p * a + (1-p) * b

    @staticmethod
    def binomial(p: float, N: int, variance: bool = False) -> float:
        """Mean or variance of binomial r.v.

        Args:
          p : probability of occurence
          N : number of trials
          variance : whether to return variance (True) or mean (False)
        """
        assert 0 <= p <= 1. and N >= 1
        return N * p * (1-p) if variance else N * p

    @staticmethod
    def mixture(p, p1, p2: float, N: int = 1, variance: bool = False) -> float:
        """Mean or variance of binomial mixture

        Args:
          p : probability of selecting first r.v.
          p1 : probability of occurrence if first r.v.
          p2 : probability of occurrence if other r.v.
          N : number of trials
          variance : whether to return variance (True) or mean (False)
        """
        assert 0 <= p <= 1 and 0 <= p1 <= 1 and 0 <= p2 <= 1 and N >= 1
        mean1 = Life.binomial(p1, N)
        mean2 = Life.binomial(p2, N)
        if variance:
            var1 = Life.binomial(p1, N, variance=True)
            var2 = Life.binomial(p2, N, variance=True)
            return (Life.bernoulli(p, mean1**2 + var1, mean2**2 + var2) -
                    Life.bernoulli(p, mean1, mean2)**2)
        else:
            return Life.bernoulli(p, mean1, mean2)
        
    @staticmethod
    def conditional_variance(p, p1, p2: float, N: int = 1) -> float:
        """Conditional variance formula

        Args:
          p : probability of selecting first r.v.
          p1 : probability of occurence for first r.v.
          p2 : probability of occurence for other r.v.
          N : number of trials
        """
        assert 0 <= p <= 1 and 0 <= p1 <= 1 and 0 <= p2 <= 1 and N >= 1
        mean1 = Life.binomial(p1, N)
        mean2 = Life.binomial(p2, N)
        var1 = Life.binomial(p1, N, variance=True)
        var2 = Life.binomial(p2, N, variance=True)
        return (Life.bernoulli(p, mean1, mean2, variance=True)  # var of mean
                + Life.bernoulli(p, var1, var2))           # plus mean of var

    @staticmethod
    def portfolio_percentile(mean: float, variance: float,
                             prob: float, N: int = 1) -> float:
        """Probability percentile of the sum of N iid r.v.'s

        Args:
          mean : mean of each independent obsevation
          variance : variance of each independent observation
          prob : probability threshold
          N : number of observations to sum
        """
        assert prob < 1.0
        mean *= N
        variance *= N
        return mean + ndtri(prob) * math.sqrt(variance)

    @staticmethod
    def portfolio_cdf(mean: float, variance: float, value: float,
                      N: int = 1) -> float:
        """Probability distribution of a value in the sum of N iid r.v.

        Args:
          mean : mean of each independent obsevation
          variance : variance of each independent observation
          value : value to compute probability distribution in the sum
          N : number of observations to sum
        """
        mean *= N
        variance *= N
        return norm.cdf(value, loc=mean, scale=math.sqrt(variance))

    @staticmethod
    def quantiles_frame(quantiles: List[float] = [.8, .85, .9, .95,
                                                  .975, .99, .995]) -> Any:
        """Display selected quantile values from Normal distribution table

        Args:
          quantiles : list of quantiles to display normal distribution values
        """
        columns = [round(Life.portfolio_percentile(0, 1, p), 3) for p in quantiles]
        tab = pd.DataFrame.from_dict(data={'Pr(Z<=z)': quantiles}, 
                                     columns=columns, orient='index')\
                                    .rename_axis('z', axis="columns")
        return tab.round(3)


if __name__ == "__main__":
    print("SOA Question 2.2: (D) 400")
    p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
    p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
    cond = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
    print(cond)   # conditional variance formula
    mix = math.sqrt(Life.mixture(p=.2, p1=p1, p2=p2, N=100000, variance=True))
    print(mix)    # mixture of distributions formula

    print()
    print("Values of z for selected values of Pr(Z<=z)")
    print("-------------------------------------------")
    print(Life.quantiles_frame().to_string(float_format=lambda x: f"{x:.3f}"))
    print()

