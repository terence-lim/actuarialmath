"""Mortality Laws: Uniform, Beta, Gompertz, Makeham

Copyright 2022, Terence Lim

MIT License
"""
import math
from mathlc.reserves import Reserves

class MortalityLaws(Reserves):
    """Shortcuts for special mortality laws"""
    _doc = ['l_r', 'p_r', 'q_r', 'mu_r', 'f_r', 'e_r']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._doc = ['l_r', 'p_r', 'q_r', 'mu_r', 'f_r', 'e_r']

    def l_r(self, x: int, s: int = 0, r: float = 0.) -> float:
        """Fractional age lifes given continuous mortality law: l_[x]+s+r"""
        return self.l(x, s+r)

    def p_r(self, x: int, s: int = 0, r: float = 0., t: float = 1.) -> float:
        """Fractional age survival given continuous mortality law"""
        return self.S(x, s+r, t)

    def q_r(self, x: int, s: int = 0, r: float = 0., t: float = 1., 
            u: float = 0.) -> float:
        """Fractional age deferred mortality given continuous mortality law"""
        return self.p_r(x, s=s, r=r, t=u) - self.p_r(x, s=s, r=r, t=t+u)
        
    def mu_r(self, x: int, s: int = 0, r: float = 0.) -> float:
        """Fractional age force of mortality given continuous mortality law"""
        return self.mu(x, s+r)

    def f_r(self, x: int, s: int = 0, r: float = 0., t: float = 0.0) -> float:
        """fractional age mortality pdf given continuous mortality law"""
        return self.f(x, s+r, t)

    def e_r(self, x: int, s: int = 0, t: float = Reserves.WHOLE) -> float:
        """Fractional age future lifetime given continuous mortality law"""
        if t < 0:
            t = self.MAXAGE - (x + s)
        return self.integrate(lambda t: self.S(x, s, t), 0., float(t))

class Beta(MortalityLaws):
    """Is Uniform when alpha = 1"""

    def __init__(self, omega: int, alpha: float, lifes=MortalityLaws.LIFES, 
                 **kwargs):
        """Two parameters: alpha and omega, with mu(x) = alpha/(omega-x)"""

        super().__init__(minage=0, maxage=omega, **kwargs)

        def _mu(x: int, s: float) -> float: 
            return alpha / (omega - (x+s))

        def _l(x: int, s: float) -> float:
            return lifes * (omega - (x+s))**alpha

        def _S(x: int, s,t : float) -> float:
            return ((omega-(x+s+t))/(omega-(x+s)))**alpha

        def _f(x: int, s,t : float) -> float:
            return alpha / (omega - (x+s))
        self.set_survival(mu=_mu, l=_l, S=_S, f=_f)
        self.omega_ = omega
        self.alpha_ = alpha

    def e_r(self, x: int, s: int = 0, t: float = Reserves.WHOLE) -> float:
        """Expectation of future lifetime through fractional age: e_[x]+s:t"""
        e = (self.omega_ - (x+s)) / (self.alpha_ + 1)
        if t > 0 and self.max_term(x+s, t) > t: # temporary expectation
            return e - self.p_r(x, s=s, t=t) * self.e_r(x, s=s+t)
        return e   # complete expectation

    def e_x(self, x: int, s: int = 0, n: int = MortalityLaws.WHOLE, 
          curtate: bool = False, moment: int = 1) -> float:
        """Shortcut formula for complete expectation"""
        if n == 0:
            return 0
        if not curtate:
            if moment == 1:
                return self.e_r(x, s=s, t=n)
            if moment == self.VARIANCE and n < 0: # shortcut for complete variance
                return ((self.omega_ - (x + s))
                        / ((self.alpha_ + 1)**2 * (self.alpha_ + 1)))
        return super().__init__(x=x, s=s, n=n, curtate=curtate, moment=moment)

class Uniform(Beta):
    """Uniform distribution aka DeMoivre's Law"""

    def __init__(self, omega: int, udd: bool = True, **kwargs):
        """One parameter: omega = maxage, with mu(x) = 1/(omega - x)"""
        super().__init__(omega=omega, alpha=1, udd=True, **kwargs)

    def e_x(self, x: int, s: int = 0, t: int = Beta.WHOLE, 
          curtate: bool = False, moment: int = 1) -> float:
        """(P[die within n years] * n/2) plus (P[survive n years] * n)"""
        if moment in [1, self.VARIANCE] and not curtate:
            if t < 0:
                if moment == self.VARIANCE:
                    return (self.omega_ - x)**2 / 12  # complete shortcut
                else:
                    return (self.omega_ - x) / 2
            elif moment == 1:         # temporary expectation shortcut
                t = self.max_term(x+s, t)
                t_p_x = t / (self.omega_ - x)
                return t_p_x * (t / 2) + (1 - t_p_x) * t
        return super().__init__(x=x, s=s, t=t, curtate=curtate, moment=moment)

    def E_x(self, x: int, s: int = 0, t: int = Beta.WHOLE, 
            moment: int = 1) -> float:
        """Shortcut for Pure Endowment"""
        assert moment > 0
        if t == 0:
            return 1.
        if t < 0:
            return 0.
        t = self.max_term(x+s, t)
        t_p_x = (self.omega_ - x - t) / (self.omega_ - x)
        if moment == self.VARIANCE:  # Bernoulli shortcut for variance
            return self.interest.v_t(t)**2 * t_p_x * (1 - t_p_x)
        return (self.interest.v_t(t)**moment * (self.omega_ - x - t) 
                / (self.omega_ - x))

    def whole_life_insurance(self, x: int, s: int = 0, moment: int = 1, 
                             b: int = 1, discrete: bool = True) -> float:
        """Shortcut for whole life insurance"""
        if not discrete:
            if moment == Beta.VARIANCE:
                return (self.whole_life_insurance(x, s=s, b=b, moment=2, 
                                                  discrete=False)
                        - self.whole_life_insurance(x, s=s, b=b, moment=1, 
                                                    discrete=False)**2) * b**2
            return self.term_insurance(x, s=s, t=self.omega_-(x+s), b=b,
                                       moment=moment, discrete=False)
        return super().whole_life_insurance(x, s=s, moment=moment, b=b,
                                            discrete=discrete)

    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                       moment: int = 1, discrete: bool = False) -> float:
        """Shortcut for term insurance"""
        if not discrete and moment in [1, Beta.VARIANCE]:
            if moment == Beta.VARIANCE:
                return (self.term_insurance(x, s=s, b=b, t=t, moment=2, 
                                            discrete=False)
                        - self.term_insurance(x, s=s, b=b, t=t, 
                                            discrete=False)**2) * b**2
            t = self.max_term(x+s, t)                                        
            return (b * (1 - self.interest.v_t(t)) 
                    / (self.interest.delta * (self.omega_ - (x+s))))
        return super().term_insurance(x, s=s, moment=moment, b=b, 
                                      discrete=discrete)

class Makeham(MortalityLaws):
    """includes an element in force of mortality that does not depend on age"""
    def __init__(self, A: float, B: float, c: float, **kwargs):
        """assert c > 1, B > 0, A >= -B"""
        super().__init__(**kwargs)
        self.A_ = A
        self.B_ = B
        self.c_ = c

        def _mu(x, s): 
            return A + B * c**(x+s)
        def _S(x, s, t):
            return math.exp(-A*t - B*c**(x+s) * (c**t - 1)/math.log(c))
        self.set_survival(mu=_mu, S=_S)

class Gompertz(Makeham):
    """As age increases so does force of mortality"""

    def __init__(self, B: float, c: float):
        """Gompertz's Law is Makeham's Law with A = 0"""
        super().__init__(A=0., B=B, c=c)

if __name__ == "__main__":

    print('Beta')
    life = Beta(omega=100, alpha=0.5)
    print(life.q_x(25, t=1, u=10))     # 0.0072
    print(life.e_x(25))                # 50
    print(Beta(omega=60, alpha=1/3).mu_x(35) * 1000)
    print()

    print('Uniform')
    uniform = Uniform(80, interest=dict(delta=0.04))
    print(uniform.whole_life_annuity(20))        # 15.53
    print(uniform.temporary_annuity(20, t=5))   # 4.35
    print(Uniform(161).p_x(70, t=1)) # 0.98901
    print(Uniform(95).e_x(30, t=40, curtate=False)) # 27.692
    print()

    uniform = Uniform(omega=80, interest=dict(delta=0.04))
    print(uniform.E_x(20, t=5))  # .7505
    print(uniform.whole_life_insurance(20, discrete=False))  # .3789
    print(uniform.term_insurance(20, t=5, discrete=False))  # .0755
    print(uniform.endowment_insurance(20, t=5, discrete=False))  # .8260
    print(uniform.deferred_insurance(20, u=5, discrete=False))  # .3033
    print()

    print('Gompertz/Makeham')
    life = Gompertz(B=0.000005, c=1.10)
    p = life.p_x(80, t=10)  # 869.4
    print(life.portfolio_percentile(N=1000, mean=p, variance=p*(1-p), prob=0.99)) 

    print(Gompertz(B=0.00027, c=1.1).f_x(50, t=10)) # 0.04839
    life = Makeham(A=0.00022, B=2.7e-6, c=1.124)
    print(life.mu_x(60) * 0.9803)  # 0.00316
