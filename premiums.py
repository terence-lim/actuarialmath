"""Premiums

Copyright 2022, Terence Lim

MIT License
"""
from actuarialmath.annuity import Annuity
from typing import Optional

class Premiums(Annuity):
    """Premiums: equivalence principle, net and gross premiums
    """
    _help = ['net_premium', 'gross_premium', 'insurance_equivalence',
             'annuity_equivalence', 'premium_equivalence']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    #
    # Net level premiums for special insurance
    #
    def net_premium(self, x: int, s: int = 0, t: int = Annuity.WHOLE, 
                    u: int = 0, n: int = 0, b: int = 1,
                    endowment: int = 0, discrete: Optional[bool] = True, 
                    return_premium: bool = False, annuity: bool = False,
                    initial_cost: float = 0.) -> float:
        """Net level premium for n-pay, u-deferred t-year term insurance
        - x (int) : age initially insured
        - s (int) : years after selection
        - u (int) : years of deferral
        - n (int) : number of years of premiums paid
        - t (int) : year of death
        - b (int) : benefit amount
        - endowment (int): endowment amount
        - return_premium (bool) : whether premiums without interest refunded at death 
        - annuity (bool) : whether benefit is insurance (False) or deferred annuity
        - discrete (bool) : discrete/annuity due (True) or continuous (False)
        - initial_cost (int) : EPV of any other expenses or benefits
        """
        if annuity:
            A = self.deferred_annuity(x, s=s, b=b, t=t, u=u, 
                                      discrete=discrete or discrete is None)
        else:
            A = self.deferred_insurance(x, s=s, b=b, t=t, u=u, 
                                        discrete=discrete or discrete is None)
            if endowment:
                A += self.E_x(x, s=s, t=t+u) * endowment
        if n == 0:      # if n not specified
            n = u or t  # then set to defer period if any, else same term t
        discrete = discrete or discrete is None  # discrete or semi-discrete
        a = self.temporary_annuity(x, s=s, t=n, discrete=discrete)
        if return_premium: # death benefit include premiums returned w/o interest
            a -= self.increasing_insurance(x, s=s, t=n, discrete=discrete)
        return (A + initial_cost) / a

    #
    # Equivalence principle for WL, Endowment Insurance, and Annuity twins
    #
    def insurance_equivalence(self, premium: float, b: int = 1,
                              discrete: bool = True) -> float:
        """Whole life or endowment insurance factor, given net premium
        - premium (float) : level net premium amount
        - b (int) : benefit amount
        - discrete (bool) : discrete/annuity due (True) or continuous (False)
        """
        d = self.interest.d if discrete else self.interest.delta
        return premium / (d*b + premium)  # from P = b[dA/(1-A)]

    def annuity_equivalence(self, premium: float, b: int = 1,
                            discrete: bool = True) -> float:
        """Whole life or temporary annuity factor, given net premium
        - premium (float) : level net premium amount
        - b (int) : benefit amount
        - discrete (bool) : discrete/annuity due (True) or continuous (False)
        """
        d = self.interest.d if discrete else self.interest.delta
        return b / (d*b + premium)  # from P = b * (1/a - d)

    def premium_equivalence(self, A: Optional[float] = None,
                            a: Optional[float] = None,
                            b: int = 1, discrete: bool = True) -> float:
        """Premium given whole life or temporary/endowment annuity/insurance
        - A (float) : insurance factor
        - a (float) : annuity factor
        - b (int) : insurance benefit amount
        - discrete (bool) : annuity due (True) or continuous (False)
        """
        interest = self.interest.d if discrete else self.interest.delta
        if a is None:   # Annuity not given => use shortcut for insurance
            return b * interest * A / (1 - A)
        elif A is None: # Insurance not given => use shortcut for annuity
            return b * (1/a - interest)
        else:           # Both (special) insurance and annuity factors given
            return b * A / a

    #
    # Gross premiums for special insurance
    #
    def gross_premium(self, a: Optional[float] = None, 
                      A: Optional[float] = None, IA: float = 0, 
                      discrete: bool = True, benefit: float = 1,
                      E: float = 0, endowment: int = 0, 
                      settlement_policy: float = 0.,
                      initial_policy: float = 0.,
                      initial_premium: float = 0.,
                      renewal_policy: float = 0., 
                      renewal_premium: float = 0.) -> float:
        """Gross premium by equivalence principle
        - A (float) : insurance factor
        - a (float) : annuity factor
        - IA (float) : increasing insurance factor for premiums returned w/o interest
        - E (float) : pure endowment factor for endowment benefit
        - benefit (int) : insurance benefit amount
        - endowment (float) : endowment benefit amount
        - settlement_policy (float) : settlement expense per policy
        - initial_policy (float) : initial expense per policy
        - renewal_policy (float) : renewal expense per policy
        - initial_premium (float) : initial premium per $ of gross premium
        - renewal_premium (float) : renewal premium per $ of gross premium
        - discrete (bool) : annuity due (True) or continuous (False)
        """
        if a is None:    # assume WL or Endowment Insurance for twin
            a = self.annuity_twin(A, discrete=discrete)
        elif A is None:  # assume WL or Endowment Insurance for twin
             A = self.insurance_twin(a, discrete=discrete)

        assert endowment == 0 or E > 0   # missing pure endowment if needed
        per_premium = renewal_premium * a + (initial_premium - renewal_premium)
        per_policy = renewal_policy * a + (initial_policy - renewal_policy)
        return (((A*(benefit + settlement_policy) + per_policy) + (E*endowment))
                 / (a - per_premium - IA))  # IA returns premium w/o interest


if __name__ == "__main__":
    import numpy as np
    
    print("SOA Question 6.29  (B) 20.5")
    life = Premiums(interest=dict(i=0.035))
    def fun(a):
        return life.gross_premium(A=life.insurance_twin(a=a),
                                  a=a, benefit=100000, 
                                  initial_policy=200, initial_premium=.5,
                                  renewal_policy=50, renewal_premium=.1)
    print(life.solve(fun, target=1770, guess=[20, 22]))
    print()
    
    print("SOA Question 6.2: (E) 3604")
    life = Premiums()
    A, IA, a = 0.17094, 0.96728, 6.8865
    print(life.gross_premium(a=a, A=A, IA=IA, benefit=100000,
                             initial_premium=0.5, renewal_premium=.05,
                             renewal_policy=200, initial_policy=200))
    print()

    print("SOA Question 6.16: (A) 2408.6")
    life = Premiums(interest=dict(d=0.05))
    A = life.insurance_equivalence(premium=2143, b=100000)
    a = life.annuity_equivalence(premium=2143, b=100000)
    p = life.gross_premium(A=A, a=a, benefit=100000, settlement_policy=0,
                           initial_policy=250, initial_premium=.04+.35,
                           renewal_policy=50, renewal_premium=.04+.02) 
    print(A, a, p)
    print()

    print("SOA Question 6.20:  (B) 459")
    life = Premiums(interest=dict(i=0.04), 
                    l=lambda x,s: dict(zip([75, 76, 77, 78],
                                       np.cumprod([1,.9,.88,.85]))).get(x+s, 0))
    a = life.temporary_annuity(75, t=3)
    IA = life.increasing_insurance(75, t=2)
    A = life.deferred_insurance(75, u=2, t=1)
    print(life.solve(lambda P: P*IA + A*10000 - P*a, target=0, guess=100))
    print()

    print("Other usage")
    life = Premiums(interest=dict(delta=0.06), mu=lambda x,s: 0.04)
    print(life.net_premium(0))

    print(Premiums.help())
