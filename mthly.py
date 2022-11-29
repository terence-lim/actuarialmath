"""1/Mthly insurance and annuities

Copyright 2022, Terence Lim

MIT License
"""
from typing import Callable
from actuarialmath.premiums import Premiums
from actuarialmath.life import Actuarial
import math
import pandas as pd

class Mthly(Actuarial):
    """1/M'thly insurance and annuities

    - m (int) : number of payments per year
    - life (Premiums) : original survival and life contingent functions
    """
    _help = ['v_m', 'p_m', 'q_m', 'Z_m', 'E_x', 'A_x', 'whole_life_insurance',
             'term_insurance', 'deferred_insurance', 'endowment_insurance',
             'immediate_annuity', 'annuity_twin', 'annuity_variance',
             'whole_life_annuity', 'temporary_annuity', 'deferred_annuity',
             'immediate_annuity']
            
    def __init__(self, m: int, life: Premiums):
        self.life = life
        self.m = max(0, m)
    
    def v_m(self, k: int) -> float:
        """Compute discount rate compounded over k m'thly periods
        - k (int) : number of m'thly periods to compound
        """
        return self.life.interest.v_t(k / self.m)

    def q_m(self, x: int, s_m: int = 0, t_m: int = 1, u_m: int = 0) -> float:
        """Compute deferred mortality over m'thly periods
        - x (int) : year of selection
        - s_m (int) : number of m'thly periods after selection
        - u_m (int) : survive number of m'thly periods , then
        - t_m (int) : dies within number of m'thly periods
        """
        sr = s_m / self.m
        s = math.floor(sr)
        r = sr - s
        q = self.life.q_r(x, s=s, r=r, t=t_m/self.m, u=u_m/self.m)
        return q

    def p_m(self, x: int, s_m: int = 0, t_m: int = 1) -> float:
        """Compute survival probability over m'thly periods
        - x (int) : year of selection
        - s_m (int) : number of m'thly periods after selection
        - t_m (int) : survives number of m'thly periods
        """
        sr = s_m / self.m
        s = math.floor(sr)
        r = sr - s
        return self.life.p_r(x, s=s, r=r, t=t_m/self.m)

    def E_x(self, x: int, s: int = 0, t: int = 1, moment: int = 1,
            endowment: int = 1) -> float:
        """Compute pure endowment factor
        - x (int) : year of selection
        - s (int) : years after selection
        - t (int) : term length in years
        - moment (int) : return first or second moment
        - endowment (int) : endowment amount
        """
        assert moment > 0
        return self.life.E_x(x, s=s, t=t, moment=moment) * endowment**moment

    def Z_m(self, x: int, s: int = 0, t: int = 1, 
            benefit: Callable = lambda x,t: 1, moment: int = 1):
        """Return PV insurance r.v. Z and probability by m'thly period as DataFrame
        - x (int) : year of selection
        - s (int) : years after selection
        - t (int) : year of death
        - benefit (Callable) : amount of benefit by year and age selected
        - moment (int) : return first or second moment
        """
        Z = [(benefit(x+s, k/self.m) * self.v_m(k+1))**moment 
                 for k in range(t * self.m)]
        p = [self.q_m(x, s_m=s*self.m, u_m=k) for k in range (t*self.m)]
        return pd.DataFrame.from_dict(dict(m = range(1, self.m*t + 1),
                                      Z=Z, p=p)).set_index('m')

    def A_x(self, x: int, s: int = 0, t: int = 1, u: int = 0, 
            benefit: Callable = lambda x,t: 1, moment: int = 1) -> float:
        """Compute insurance factor with m'thly benefits
        - x (int) : year of selection
        - s (int) : years after selection
        - u (int) : years deferred
        - t (int) : term of insurance in years
        - benefit (Callable) : amount of benefit by year and age selected
        - moment (int) : return first or second moment
        """
        assert moment in [1, 2]
        t = self.max_term(x+s, t)
        if self.m > 0:
            A = sum([(benefit(x+s, k/self.m) * self.v_m(k+1))**moment 
                     * self.q_m(x, s_m=s*self.m, u_m=k) 
                     for k in range((t+u) * self.m)])
        else:
            Z = lambda t: ((benefit(x+s, t+u) * self.life.v_t(t+u))**moment
                            * self.life.f(x, s, t+u))
            A = self.life.integrate(Z, 0, t)
        return A

    def whole_life_insurance(self, x: int, s: int = 0, moment: int = 1, 
                             b: int = 1) -> float:
        """Whole life insurance: A_x
        - x (int) : age of selection
        - s (int) : years after selection
        - b (int) : amount of benefit
        - moment (int) : compute first or second moment
        """
        assert moment in [1, 2, Premiums.VARIANCE]
        if moment == Premiums.VARIANCE:
            A2 = self.whole_life_insurance(x, s=s, moment=2)
            A1 = self.whole_life_insurance(x, s=s)
            return self.life.insurance_variance(A2=A2, A1=A1, b=b)
        return sum(self.A_x(x, s=s, b=b, moment=moment))


    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                       moment: int = 1) -> float:
        """Term life insurance: A_x:t^1
        - x (int) : year of selection
        - s (int) : years after selection
        - t (int) : term of insurance in years
        - b (int) : amount of benefit
        - moment (int) : return first or second moment
        """
        assert moment in [1, 2, Premiums.VARIANCE]
        if moment == Premiums.VARIANCE:
            A2 = self.term_insurance(x, s=s, t=t, moment=2)
            A1 = self.term_insurance(x, s=s, t=t)
            return self.life.insurance_variance(A2=A2, A1=A1, b=b)
        A = self.whole_life_insurance(x, s=s, b=b, moment=moment)
        if t < 0 or self.life.max_term(x+s, t) < t:
            return A
        E = self.E_x(x, s=s, t=t, moment=moment)
        A -= E * self.whole_life_insurance(x, s=s+t, b=b, moment=moment)
        return A

    def deferred_insurance(self, x: int, s: int = 0, n: int = 0, b: int = 1, 
                           t: int = Premiums.WHOLE, moment: int = 1) -> float:
        """Deferred insurance n|_A_x:t^1 = discounted whole life
        - x (int) : year of selection
        - s (int) : years after selection
        - u (int) : years to defer
        - t (int) : term of insurance in years
        - b (int) : amount of benefit
        - moment (int) : return first or second moment
        """
        if self.life.max_term(x+s, n) < n:
            return 0.
        if moment == self.VARIANCE:
            A2 = self.deferred_insurance(x, s=s, t=t, n=n, moment=2)
            A1 = self.deferred_insurance(x, s=s, t=t, n=n)
            return self.life.insurance_variance(A2=A2, A1=A1, b=b)
        E = self.E_x(x, s=s, t=n, moment=moment)
        A = self.term_insurance(x, s=s+n, t=t, b=b, moment=moment)
        return E * A   # discount insurance by moment*force of interest

    def endowment_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                            endowment: int = -1, moment: int = 1) -> float:
        """Endowment insurance: A_x:t = term insurance + pure endowment
        - x (int) : year of selection
        - s (int) : years after selection
        - t (int) : term of insurance in years
        - b (int) : amount of benefit
        - endowment (int) : amount of endowment
        - moment (int) : return first or second moment
        """
        if moment == self.VARIANCE:
            A2 = self.endowment_insurance(x, s=s, t=t, endowment=endowment, 
                                          b=b, moment=2)
            A1 = self.endowment_insurance(x, s=s, t=t, endowment=endowment, 
                                          b=b)
            return self.life.insurance_variance(A2=A2, A1=A1, b=b)
        E = self.E_x(x, s=s, t=t, moment=moment)
        A = self.term_insurance(x, s=s, t=t, b=b, moment=moment)
        return A + E * (b if endowment < 0 else endowment)**moment

    def annuity_twin(self, A: float) -> float:
        """Return value of annuity twin of m'thly insurance
        - A (float) : amount of m'thly insurance
        """
        return (1-A)/self.life.interest.mthly(m=self.m, d=self.life.interest.d)

    def annuity_variance(self, A2: float, A1: float, b: float = 1) -> float:
        """Variance of m'thly annuity from m'thly insurance moments
        - A2 (float) : double force of interest of m'thly insurance
        - A1 (float) : first moment of m'thly insurance
        - b (float) : amount of benefit
        """
        num = self.life.insurance_variance(A2=A2, A1=A1, b=b)
        den = self.life.interest.mthly(m=self.m, d=self.life.interest.d)
        return num / den**2

    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1, 
                           variance: bool = False) -> float:
        """Whole life m'thly annuity: a_x
        - x (int) : year of selection
        - s (int) : years after selection
        - b (int) : amount of benefit
        - variance (bool) : return first moment (False) or variance (True)
        """
        if variance:  # short cut for variance of whole life
            A1 = self.whole_life_insurance(x, s=s, moment=1)
            A2 = self.whole_life_insurance(x, s=s, moment=2)
            return self.annuity_variance(A2=A2, A1=A1, b=b)
        return b * (1 - self.whole_life_insurance(x, s=s)) / self.d
            
    def temporary_annuity(self, x: int, s: int = 0, t: int = Premiums.WHOLE, 
                          b: int = 1, variance: bool = False) -> float:
        """Temporary m'thly life annuity: a_x:t
        - x (int) : year of selection
        - s (int) : years after selection
        - t (int) : term of annuity in years
        - b (int) : amount of benefit
        - variance (bool) : return first moment (False) or variance (True)
        """
        if variance:  # short cut for variance of temporary life annuity
            A1 = self.term_insurance(x, s=s, t=t)
            A2 = self.term_insurance(x, s=s, t=t, moment=2)
            return self.annuity_variance(A2=A2, A1=A1, b=b)

        # difference of whole life on (x) and deferred whole life on (x+t)
        a = self.whole_life_annuity(x, s=s, b=b)
        if t < 0 or self.max_term(x+s, t) < t:
            return a
        a_t = self.whole_life_annuity(x, s=s+t, b=b)
        return a - (a_t * self.E_x(x, s=s, t=t))

    def deferred_annuity(self, x: int, s: int = 0, u: int = 0, 
                         t: int = Premiums.WHOLE, b: int = 1) -> float:
        """Deferred m'thly life annuity due n|t_a_x =  n+t_a_x - n_a_x
        - x (int) : year of selection
        - s (int) : years after selection
        - u (int) : years of deferral
        - t (int) : term of annuity in years
        - b (int) : amount of benefit
        """
        if self.life.max_term(x+s, u) < u:
            return 0.
        return self.E_x(x, s=s, t=u)*self.temporary_annuity(x, s=s+u, t=t, b=b)

    def immediate_annuity(self, x: int, s: int = 0, t: int = Premiums.WHOLE, 
                          b: int = 1) -> float:
        """Immediate m'thly annuity
        - x (int) : year of selection
        - s (int) : years after selection
        - t (int) : term of annuity in years
        - b (int) : amount of benefit
        """
        a = self.temporary_annuity(x, s=s, t=t)
        if self.m > 0:
            return (a - ((1 - self.E_x(x, s=s, t=t)) / self.m)) * b
        else:
            return a

if __name__ == "__main__":
    from actuarialmath.lifetable import LifeTable

    print("SOA Question 6.4:  (E) 1893.9")
    mthly = Mthly(m=12, life=Premiums(interest=dict(i=0.06)))
    A1, A2 = 0.4075, 0.2105
    mean = mthly.annuity_twin(A1)*15*12
    var = mthly.annuity_variance(A1=A1, A2=A2, b=15 * 12)
    S = Premiums.portfolio_percentile(mean=mean, variance=var, prob=.9, N=200)
    print(S / 200)
    print()
    
    print("SOA Question 4.2:  (D) 0.18")
    life = LifeTable(q={0: .16, 1: .23}, interest=dict(i_m=.18, m=2),
                     udd=False).fill()
    mthly = Mthly(m=2, life=life)
    Z = mthly.Z_m(0, t=2, benefit=lambda x,t: 300000 + t*30000*2)
    print(Z)
    print(Z[Z['Z'] >= 277000].iloc[:, -1].sum())
    print()

    print(Mthly.help())
    
    
