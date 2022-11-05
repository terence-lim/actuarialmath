"""Constant Force mortality law shortcuts

Copyright 2022, Terence Lim

MIT License
"""
import math
from scipy.stats import norm
from mathlc.mortalitylaws import MortalityLaws

class ConstantForce(MortalityLaws):
    """Memoryless exponential distribution of deaths"""

    _doc = ['e_x', 'E_x', 'whole_life_insurance', 'temporary_annuity',
            'term_insurance', 'Z_t', 'Y_t']

    def __init__(self, mu: float, udd: bool = False, **kwargs):
        super().__init__(udd=udd, **kwargs)

        def _mu(x: int, s: float) -> float:
            """Constant force of mortality"""
            return mu 

        def _S(x: int, s, t: float) -> float: 
            """Shortcut for survival function with constant force of mortality"""
            return math.exp(-mu * t)

        self.set_survival(mu=_mu, S=_S)
        self.mu_ = mu

    def e_x(self, x: int, s: int = 0, t: int = MortalityLaws.WHOLE, 
           curtate: bool = False, moment: int = 1) -> float:
        """Expected lifetime E[T_x] is memoryless: does not depend on (x)"""
        if not curtate:    # Var[Tx] = 1/mu^2, E[Tx] = 1/mu
            if moment == MortalityLaws.VARIANCE:
                return 1. / self.mu_**2   # shortcut
            elif moment == 1:
                e = 1 / self.mu_  # infinite n case shortcut
                if t >= 0:       # n-limited from recursion e_x = e_x|n + n_p_x e_x
                    e *= (1 - math.exp(-self.mu_ * t))
                return e
        return super().e_x(x, s=s, t=t, curtate=curtate, moment=moment)

    def E_x(self, x: int, s: int = 0, t: int = MortalityLaws.WHOLE, 
            moment: int = 1) -> float:
        """Shortcut for APV of whole life: does not depend on age x"""
        if t == 0:
            return 1.
        if t < 0:
            return 0.
        delta = moment * self.interest.delta   # multiply force of interest
        return math.exp(-(self.mu_ + delta) * t)

    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1,
                           variance: bool = False,
                           discrete: bool = True) -> float:
        """Shortcut for whole life annuity: does not depend on age x"""
        if variance:  # short cut for variance of temporary life annuity
            A1 = self.whole_life_insurance(x, s=s, discrete=discrete)
            A2 = self.whole_life_insurance(x, s=s, moment=2, discrete=discrete)
            return (b**2 * (A2 - A1**2) /
                    (self.interest.d if discrete else self.interest.delta)**2)
        if not discrete:
            return (1. / (self.mu_ + self.interest.delta)) * b
        return super().whole_life_annuity(x, s=s, b=b, discrete=discrete)

    def temporary_annuity(self, x: int, s: int = 0, t: int = MortalityLaws.WHOLE,
                          b: int = 1, variance: bool = False,
                          discrete: bool = True) -> float:
        """Shortcut for temporary life annuity: does not depend on age x"""
        interest = self.interest.d if discrete else self.interest.delta
        if variance:  # short cut for variance of temporary life annuity
            A1 = self.endowment_insurance(x, s=s, t=t, discrete=discrete)
            A2 = self.endowment_insurance(x, s=s, t=t, moment=2,
                                          discrete=discrete)
            return (b**2 * (A2 - A1**2) /
                    (self.interest.d if discrete else self.interest.delta)**2)
        if not discrete:
            a = b * 1/(self.mu_ + self.interest.delta)
            if t < 0:
                return a
            else:
                return a * (1 - math.exp(-(self.mu_ + self.interest.delta)*t))
        return super().temporary_annuity(x, s=s, b=b, t=t, discrete=discrete)

    def whole_life_insurance(self, x: int, s: int = 0, moment: int = 1,
                             b: int = 1, discrete: bool = True) -> float:
        """Shortcut for APV of whole life: does not depend on age x"""
        if moment > 0 and not discrete:
            delta = moment * self.interest.delta   # multiply force of interest
            return (self.mu_ / (self.mu_ + delta))
        return super().whole_life_insurance(x, s=s, moment=moment, b=b,
                                            discrete=discrete)

    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1,
                       moment: int = 1, discrete: bool = True) -> float:
        """Shortcut for APV of term life: does not depend on age x"""
        if moment > 0 and not discrete:
            delta = moment * self.interest.delta   # multiply force of interest
            A = b**moment * self.mu_/(self.mu_ + delta)
            if t < 0:
                return A
            else:
                return A * (1 - math.exp(-(self.mu_ + delta)*t))
        return super().term_insurance(x, s=s, t=t, b=b, moment=moment,
               discrete = discrete)

    def Z_t(self, x: int, prob: float, discrete: bool = True) -> float:
        """Shortcut for T_x (or K_x) given survival probability for insurance"""
        t = -math.log(prob) / self.mu_
        return math.floor(t) if discrete else t    # opposite of annuity        

    def Y_t(self, x: int, prob: float, discrete: bool = True) -> float:
        """Shortcut for T_x (or K_x) given survival probability for annuity"""
        t = -math.log(1 - prob) / self.mu_
        return math.ceil(t) if discrete else t    # opposite of insurance

if __name__ == "__main__":
    print("SOA Question 6.36:  (B) 500")
    life = ConstantForce(mu=0.04, interest=dict(delta=0.08))
    a = life.temporary_annuity(50, t=20, discrete=False)
    A = life.term_insurance(50, t=20, discrete=False)
    print(a,A)
    def fun(R):
       return life.gross_premium(a=a, A=A, initial_premium=R/4500,
                                 renewal_premium=R/4500, benefit=100000)
    R = life.solve(fun, target=4500, guess=[400, 800])
    print(R)
    print()

    print("SOA Question 6.31:  (D) 1330")
    life = ConstantForce(mu=0.01, interest=dict(delta=0.05))
    A = life.term_insurance(35, t=35) + life.E_x(35, t=35) * 0.51791 # A_35
    A = (life.term_insurance(35, t=35, discrete=False) 
            + life.E_x(35, t=35) * 0.51791)    # A_35
    P = life.premium_equivalence(A=A, b=100000, discrete=False)
    print(P)
    print()

    print("SOA Question 6.27:  (D) 10310")
    life = ConstantForce(mu=0.03, interest=dict(delta=0.06))
    x = 0
    payments = (3 * life.temporary_annuity(x, t=20, discrete=False) 
            + life.deferred_annuity(x, u=20, discrete=False))
    benefits = (1000000 * life.term_insurance(x, t=20, discrete=False)
            + 500000 * life.deferred_insurance(x, u=20, discrete=False))
    print(benefits, payments)
    print(life.term_insurance(x, t=20), life.deferred_insurance(x, u=20))
    P = benefits / payments
    print(P)
    print()


    print("SOA Question 5.4:  (A) 213.7")
    life = ConstantForce(mu=0.02, interest=dict(delta=0.01))
    P = 10000 / life.certain_life_annuity(40, u=life.e_x(40, curtate=False), 
                                            discrete=False)
    print()


    print("SOA Question 5.1: (A) 0.705")
    life = ConstantForce(mu=0.01, interest=dict(delta=0.06))
    EY = life.certain_life_annuity(0, u=10, discrete=False)
    print(life.p_x(0, t=life.Y_to_t(EY)))  # 0.705
    print()


    print("Other examples")
    life = ConstantForce(mu=0.03, interest=dict(delta=0.04))
    print(life.whole_life_annuity(20, discrete=False))  # 14.286
    print(life.temporary_annuity(20, t=5, discrete=False))  # 4.219

    life = ConstantForce(mu=0.04, interest=dict(delta=0.07))
    #print(life.T_p(30, 0.7), 1000*life.certain.Z_t(30))  # 8.9169  535.7, , 0.7 ???

    life = ConstantForce(mu=.03, interest=dict(delta=.04))
    print(life.E_x(20, t=5))  # .7047
    print(life.whole_life_insurance(20, discrete=False))  # .4286
    print(life.term_insurance(20, t=5, discrete=False))  # .1266
    print(life.endowment_insurance(20, t=5, discrete=False))  # .8313
    print(life.deferred_insurance(20, u=5, discrete=False))  # .3020

    life1 = ConstantForce(mu=.04, interest=dict(delta=.02))
    life2 = ConstantForce(mu=.05, interest=dict(delta=.02))
    life3 = ConstantForce(mu=.05, interest=dict(delta=.03))

    A1 = life1.term_insurance(0, t=5, discrete=False)
    E1 = life1.E_x(0, t=5)
    A2 = life2.term_insurance(5, t=7, discrete=False)
    E2 = life2.E_x(5, t=7)
    A3 = life3.whole_life_insurance(12, discrete=False) 
    print(A1, E1, A2, E2, A1 + E1 * (A2 + E2 * A3))


    life = ConstantForce(mu=.04, interest=dict(delta=.06))
    A1 = 10 * life.deferred_insurance(0, u=5, discrete=False)
    A2 = 10 * 10 * life.deferred_insurance(0, u=5, moment=2, discrete=False)
    E = 100 * A1
    V = 100 * (A2 - A1**2)
    print(A1, A2, E, V)  # 2.426, 11.233, 242.6, 534.8
    print(E + norm.ppf(0.95) * math.sqrt(V)) # 281
