"""Mthly with Woolhouse approximation

Copyright 2022, Terence Lim

MIT License
"""
from actuarialmath.mthly import Mthly
from actuarialmath.fractional import Fractional
import math
from typing import Callable, Union, Optional

class Woolhouse(Mthly):
    """Woolhouse 1/Mthly Shortcuts"""
    _help = ['mu_x', 'insurance_twin', 'whole_life_insurance', 'term_insurance', 
             'deferred_insurance', 'whole_life_annuity', 'temporary_annuity',
             'deferred_annuity']
    
    def __init__(self, m: int, life: Fractional, three_term: bool = False,
                 approximate_mu: Union[Callable[[int, int], float], bool] = True):
        super().__init__(m=m, life=life)
        self.three_term = three_term          # whether to include third term
        self.approximate_mu = approximate_mu  # whether to approximate mu

    def mu_x(self, x: int, s: int = 0) -> float:
        """Approximate or compute mu_x if not given"""
        if self.approximate_mu is True:     # approximate mu
            return -.5 * sum(math.log(self.life.p_x(x, s=s+t) for t in [0,-1]))
        elif self.approximate_mu is False:
            return super().mu_x(x, s=s)     # use exact mu
        else:                               # apply custom function for mu
            return self.approximate_mu(x, s)

    def insurance_twin(self, a: float) -> float:
        """Return insurance twin of mthly annuity"""
        d = self.life.interest.d
        d_m = self.life.interest.mthly(m=self.m, d=d)        
        return (1 - d_m * a)

    def whole_life_insurance(self, x: int, s: int = 0, b: int = 1,
                             mu: Optional[float] = 0.) -> float:
        """1/Mthly Woolhouse Whole life insurance: A_x"""
        return b * self.insurance_twin(self.whole_life_annuity(x, s=s, mu=mu))

    def term_insurance(self, x: int, s: int = 0, t: int = Fractional.WHOLE,
                       b: int = 1, mu: Optional[float] = 0., 
                       mu1: Optional[float] = None) -> float:
        """1/Mthly Woolhouse Term insurance: A_x:t"""
        A = self.whole_life_insurance(x, s=s, b=b, mu=mu)
        if t < 0 or self.life.max_term(x+s, t) < t:
            return A
        E = self.E_x(x, s=s, t=t)
        return A - E * self.whole_life_insurance(x, s=s+t, b=b, mu=mu1)

    def endowment_insurance(self, x: int, s: int = 0, t: int = Fractional.WHOLE,
                            b: int = 1, endowment: int = -1, 
                            mu: Optional[float] = None) -> float:
        """1/Mthly Woolhouse Term insurance: A_x:t"""
        if endowment < 0:
            endowment = b
        E = self.E_x(x, s=s, t=t)
        A = self.term_insurance(x, s=s, t=t, b=b, mu=mu)
        return A + E * (b if endowment < 0 else endowment)

    def deferred_insurance(self, x: int, s: int = 0, u: int = 0,
                           t: int = Fractional.WHOLE, b: int = 1,
                           mu: Optional[float] = None, 
                           mu1: Optional[float] = None) -> float:
        """1/Mthly Woolhouse Deferred insurance = discounted term or whole life"""
        if self.life.max_term(x+s, u) < u:
            return 0.
        E = self.E_x(x, s=s, t=u)
        return E * self.term_insurance(x, s=s+u, t=t, b=b, mu=mu, mu1=mu1)

    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1,
                           mu: Optional[float] = None) -> float:
        """1/Mthly Woolhouse Whole life annuity: a_x"""
        a = (self.life.whole_life_annuity(x, s=s, discrete=True) 
             - (self.m - 1)/(2 * self.m))
        if self.three_term:
            mu = mu or self.mu_x(x, s)
            a -= (mu + self.interest.delta)*(self.m**2 - 1)/(12 * self.m**2)
        return a * b

    def temporary_annuity(self, x: int, s: int = 0, t: int = Fractional.WHOLE, 
                          b: int = 1, mu: Optional[float] = None,
                          mu1: Optional[float] = None) -> float:
        """1/Mthly Woolhouse Temporary life annuity: a_x"""
        return self.deferred_annuity(x, s=s, t=t, b=b, mu=mu, mu1=mu1) # u=0

    def deferred_annuity(self, x: int, s: int = 0, t: int = Fractional.WHOLE, 
                         u: int = 0, b: int = 1, mu: Optional[float] = None,
                         mu1: Optional[float] = None) -> float:
        """1/Mthly Woolhouse Temporary life annuity: a_x"""
        a_x = self.whole_life_annuity(x, s=s+u, mu=mu)
        a_xt = self.whole_life_annuity(x, s=s+t+u, mu=mu1) if t > 0 else 0
        a = self.E_x(x, s=s, t=u) * (a_x - self.E_x(x, s=s+u, t=t) * a_xt)
        return a * b

if __name__ == "__main__":
    from actuarialmath.sult import SULT
    from actuarialmath.recursion import Recursion
    from actuarialmath.udd import UDD
    print(Woolhouse.help())
    
    print("SOA Question 7.7:  (D) 1110")
    x = 0
    life = Recursion(interest=dict(i=0.05)).set_A(0.4, x=x+10)
    a = Woolhouse(m=12, life=life).whole_life_annuity(x+10)
    print(a)
    policy = life.Policy(premium=0, benefit=10000, renewal_policy=100)
    V = life.gross_future_loss(A=0.4, policy=policy.future)
    print(V)
    policy = life.Policy(premium=30*12, renewal_premium=0.05)
    V1 = life.gross_future_loss(a=a, policy=policy.future)
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
                                      policy=life.Policy(premium=P))
    P = life.solve(fun, target=-800, guess=[12110, 12550])
    print(P)
    print()
     

    print("SOA Question 6.15:  (B) 1.002")
    life = Recursion(interest=dict(i=0.05)).set_a(3.4611, x=0)
    A = life.insurance_twin(3.4611)
    udd = UDD(m=4, life=life)
    a1 = udd.whole_life_annuity(x=x)
    woolhouse = Woolhouse(m=4, life=life)
    a2 = woolhouse.whole_life_annuity(x=x)
    print(life.gross_premium(a=a1, A=A)/life.gross_premium(a=a2, A=A))
    print()
     
    print("SOA Question 5.7:  (C) 17376.7")
    life = Recursion(interest=dict(i=0.04))
    life.set_A(0.188, x=35)
    life.set_A(0.498, x=65)
    life.set_p(0.883, x=35, t=30)
    mthly = Woolhouse(m=2, life=life, three_term=False)
    print(mthly.temporary_annuity(35, t=30))
    print(1000 * mthly.temporary_annuity(35, t=30))
    print()
