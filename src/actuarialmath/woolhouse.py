"""Woolhouse - 1/mthly insurance and annuities with Woolhouse's approximation

MIT License. Copyright 2022-2023 Terence Lim
"""
import math
from typing import Callable
from actuarialmath import Mthly
from actuarialmath import Annuity

class Woolhouse(Mthly):
    """1/m'thly shortcuts with Woolhouse approximation

    Args:
      m : number of payments per year
      life : original fractional survival and mortality functions
      three_term : whether to include (True) or ignore (False) third term
      approximate_mu : exact (False), approximate (True) or provide function for third term

    Examples:
      >>> life = Recursion().set_interest(i=0.05).set_a(3.4611, x=0)
      >>> woolhouse = Woolhouse(m=4, life=life)
      >>> a2 = woolhouse.whole_life_annuity(x=x)
    """
    
    def __init__(self, m: int, life: Annuity, three_term: bool = False,
                 approximate_mu: Callable[[int, int], float] | bool = True):
        super().__init__(m=m, life=life)
        self.three_term = three_term          # whether to include third term
        self.approximate_mu = approximate_mu  # whether to approximate mu

    def mu_x(self, x: int, s: int = 0) -> float:
        """Computes mu_x or calls approximate_mu for third term

        Args:
          x : age of selection
          s : years after selection
        """
        if self.approximate_mu is True:       # approximate mu
            return -.5 * sum([math.log(self.life.p_x(x, s=s+t)) for t in [0,-1]])
        elif self.approximate_mu is False:
            return self.life.mu_x(x, s=s)     # use exact mu
        else:                                 # apply custom function for mu
            return self.approximate_mu(x, s)

    def whole_life_insurance(self, x: int, s: int = 0, b: int = 1,
                             mu: float | None = 0.) -> float:
        """1/m'thly Woolhouse whole life insurance: A_x
 
        Args:
          x : age of selection
          s : years after selection
          b : amount of benefit
          mu : value of mu at age x+s
        """
        return b * self.insurance_twin(self.whole_life_annuity(x, s=s, mu=mu))

    def term_insurance(self, x: int, s: int = 0, t: int = Annuity.WHOLE,
                       b: int = 1, mu: float | None = 0., 
                       mu1: float | None = None) -> float:
        """1/m'thly Woolhouse term insurance: A_x:t

        Args:
          x : year of selection
          s : years after selection
          t : term of insurance in years
          b : amount of benefit
          mu : value of mu at age x+s
          mu1 : value of mu at age x+s+t
        """
        A = self.whole_life_insurance(x, s=s, b=b, mu=mu)
        if t < 0 or self.life.max_term(x+s, t) < t:
            return A
        E = self.E_x(x, s=s, t=t)
        return A - E * self.whole_life_insurance(x, s=s+t, b=b, mu=mu1)

    def endowment_insurance(self, x: int, s: int = 0, t: int = Annuity.WHOLE,
                            b: int = 1, endowment: int = -1, 
                            mu: float | None = None) -> float:
        """1/m'thly Woolhouse term insurance: A_x:t

        Args:
          x : year of selection
          s : years after selection
          t : term of insurance in years
          b : amount of benefit
          endowment : amount of endowment
          mu : value of mu at age x+s+t
        """
        if endowment < 0:
            endowment = b
        E = self.E_x(x, s=s, t=t)
        A = self.term_insurance(x, s=s, t=t, b=b, mu=mu)
        return A + E * (b if endowment < 0 else endowment)

    def deferred_insurance(self, x: int, s: int = 0, u: int = 0,
                           t: int = Annuity.WHOLE, b: int = 1,
                           mu: float | None = None, 
                           mu1: float | None = None) -> float:
        """1/m'thly Woolhouse deferred insurance as discounted term or WL

        Args:
          x : year of selection
          s : years after selection
          u : number of years deferred
          t : term of insurance in years
          b : amount of benefit
          mu : value of mu at age x+s+u
          mu1 : value of mu at age x+s+u+t
        """
        if self.life.max_term(x+s, u) < u:
            return 0.
        E = self.E_x(x, s=s, t=u)
        return E * self.term_insurance(x, s=s+u, t=t, b=b, mu=mu, mu1=mu1)

    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1,
                           mu: float | None = None) -> float:
        """1/m'thly Woolhouse whole life annuity: a_x

        Args:
          x : year of selection
          s : years after selection
          b : amount of benefit
          mu : value of mu at age x+s
        """
        a = (self.life.whole_life_annuity(x, s=s, discrete=True) -
             (self.m - 1)/(2 * self.m))
        if self.three_term:
            mu = mu or self.mu_x(x, s)
            a -= (mu + self.life.interest.delta)*(self.m**2 - 1)/(12 * self.m**2)
        return a * b

    def temporary_annuity(self, x: int, s: int = 0, t: int = Annuity.WHOLE, 
                          b: int = 1, mu: float | None = None,
                          mu1: float | None = None) -> float:
        """1/m'thly Woolhouse temporary life annuity: a_x

        Args:
          x : year of selection
          s : years after selection
          t : term of annuity in years
          b : amount of benefit
          mu : value of mu at age x+s
          mu1 : value of mu at age x+s+t
        """
        return self.deferred_annuity(x, s=s, t=t, b=b, mu=mu, mu1=mu1) # u=0

    def deferred_annuity(self, x: int, s: int = 0, t: int = Annuity.WHOLE, 
                         u: int = 0, b: int = 1, mu: float | None = None, 
                        mu1: float | None = None) -> float:
        """1/m'thly Woolhouse deferred life annuity: a_x

        Args:
          x : year of selection
          s : years after selection
          u : years of deferral
          t : term of annuity in years
          mu : value of mu at age x+s+u
          mu1 : value of mu at age x+s+u+t
        """
        a_x = self.whole_life_annuity(x, s=s+u, mu=mu)
        a_xt = self.whole_life_annuity(x, s=s+t+u, mu=mu1) if t > 0 else 0
        a = self.E_x(x, s=s, t=u) * (a_x - self.E_x(x, s=s+u, t=t) * a_xt)
        return a * b

if __name__ == "__main__":
    from actuarialmath.sult import SULT
    from actuarialmath.recursion import Recursion
    from actuarialmath.udd import UDD
    from actuarialmath.policyvalues import Contract
    
    print("SOA Question 7.7:  (D) 1110")
    x = 0
    life = Recursion().set_interest(i=0.05).set_A(0.4, x=x+10)
    a = Woolhouse(m=12, life=life).whole_life_annuity(x+10)
    print(a)
    contract = Contract(premium=0, benefit=10000, renewal_policy=100)
    V = life.gross_future_loss(A=0.4, contract=contract.renewals())
    print(V)
    contract = Contract(premium=30*12, renewal_premium=0.05)
    V1 = life.gross_future_loss(a=a, contract=contract.renewals())
    print(V, V1, V+V1)
    print()
     

    print("SOA Question 6.25:  (C) 12330")
    life = SULT()
    woolhouse = Woolhouse(m=12, life=life)
    benefits = woolhouse.deferred_annuity(55, u=10, b=1000 * 12)
    expenses = life.whole_life_annuity(55, b=300)
    payments = life.temporary_annuity(55, t=10)
    print(benefits + expenses, payments)
    def fun(P):
        return life.gross_future_loss(A=benefits + expenses, a=payments,
                                      contract=Contract(premium=P))
    P = life.solve(fun, target=-800, grid=[12110, 12550])
    print(P)
    print()
     

    print("SOA Question 6.15:  (B) 1.002")
    life = Recursion().set_interest(i=0.05).set_a(3.4611, x=0)
    A = life.insurance_twin(3.4611)
    udd = UDD(m=4, life=life)
    a1 = udd.whole_life_annuity(x=x)
    woolhouse = Woolhouse(m=4, life=life)
    a2 = woolhouse.whole_life_annuity(x=x)
    print(life.gross_premium(a=a1, A=A)/life.gross_premium(a=a2, A=A))
    print()
     
    print("SOA Question 5.7:  (C) 17376.7")
    life = Recursion().set_interest(i=0.04)
    life.set_A(0.188, x=35)
    life.set_A(0.498, x=65)
    life.set_p(0.883, x=35, t=30)
    mthly = Woolhouse(m=2, life=life, three_term=False)
    print(mthly.temporary_annuity(35, t=30))
    print(1000 * mthly.temporary_annuity(35, t=30))
    print()
