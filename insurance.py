"""Life insurance

Copyright 2022, Terence Lim

MIT License
"""
from typing import Callable, Any, Optional
from mathlc.fractional import Fractional
import math
import matplotlib.pyplot as plt
import numpy as np

class Insurance(Fractional):
    """Life insurance"""
    _doc = ['E_x', 'A_x', 'insurance_variance', 'insurance_twin', 'whole_life_insurance', 
            'term_insurance', 'deferred_insurance', 'endowment_insurance', 
            'increasing_insurance', 'decreasing_insurance', 'Z_t', 'Z_from_t',
            'Z_to_t', 'Z_from_prob','Z_to_prob', 'Z_x','Z_plot']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def E_x(self, x: int, s: int = 0, t: int = 1, endowment: int = 1,
            moment: int = 1) -> float:
        """Pure endowment: t_E_x

        Examples:

        >>> pure_endowment = E_x(x, t)
        >>> pure_endowment_variance = E_x(x, t, moment=Insurance.VARIANCE)
        """
        if t < 0:  # t infinite => EPV(t) = 0
            return 0   
        if t == 0:    # t = 0 => EPV(0) = 1
            return 1
        t = self.max_term(x+s, t)
        t_p_x = self.p_x(x, s=s, t=t)
        if moment == self.VARIANCE:  # Bernoulli shortcut for variance
            return self.interest.v_t(t)**2 * t_p_x * (1 - t_p_x) * endowment**2
        return (endowment * self.interest.v_t(t))**moment * t_p_x

    def A_x(self, x: int, s: int = 0, t: int = Fractional.WHOLE, u: int = 0,
            benefit: Callable = lambda x,t: 1., endowment: float = 0.,
            moment: int = 1, discrete: bool = True) -> float:
        """Numerically compute APV of insurance from survival functions"""
        assert moment >= 1
        if t >=0 and endowment > 0:
            E = self.E_x(x, s=s, t=t+u, moment=moment) * endowment
        else:
            E = 0
        t = self.max_term(x+s+u, t=t)
        if discrete:
            A = sum([(benefit(x+s, k+1)*self.interest.v_t(k+1))**moment 
                     * self.q_x(x, s=s, u=k) for k in range(u, t+u)])
        else:   # use continous first principles
            Z = lambda t: ((benefit(x+s, t+u) * self.interest.v_t(t+u))**moment 
                           * self.f(x, s, t+u))
            A = self.integrate(Z, 0, t)
        return A + E

    @staticmethod
    def insurance_variance(A2: float, A1: float, b: float = 1) -> float:
        """Compute variance of insurance given its two moments and benefit"""
        return b**2 * (A2 - A1**2)

    def insurance_twin(self, a: float, moment: int = 1, 
                       discrete: bool = True) -> float:
        """Returns WL or Endowment Insurance twin from annuity"""
        assert moment in [1]
        return 1 - a*(self.interest.d if discrete else self.interest.delta)
  
    def whole_life_insurance(self, x: int, s: int = 0, moment: int = 1, 
                             b: int = 1, discrete: bool = True) -> float:
        """Whole life insurance: A_x"""
        if moment == self.VARIANCE:
            A2 = self.whole_life_insurance(x, s=s, moment=2, discrete=discrete)
            A1 = self.whole_life_insurance(x, s=s, discrete=discrete)**2
            return self.insurance_variance(A2=A2, A1=A1, b=b)
        return self.A_x(x, s=s, t=self.WHOLE, benefit=lambda x,t: b, 
                        moment=moment, discrete=discrete)

    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                       moment: int = 1, discrete: bool = True) -> float:
        """Term life insurance: A_x:t^1"""
        if moment == self.VARIANCE:
            A2 = self.term_insurance(x, s=s, t=t, moment=2, discrete=discrete)
            A2 = self.term_insurance(x, s=s, t=t, discrete=discrete)**2
            return self.insurance_variance(A2=A2, A1=A1, b=b)
        A = self.whole_life_insurance(x, s=s, b=b, moment=moment, 
                                      discrete=discrete)
        if t < 0 or self.max_term(x+s, t=t) < t:
            return A
        E = self.E_x(x, s=s, t=t, moment=moment)
        A -= E * self.whole_life_insurance(x, s=s+t, b=b, moment=moment,
                                            discrete=discrete)
        return A

    def deferred_insurance(self, x: int, s: int = 0, u: int = 0, 
                           t: int = Fractional.WHOLE, b: int = 1, 
                           moment: int = 1, discrete: bool = True) -> float:
        """Deferred insurance n|_A_x:t^1 = discounted term or whole life"""
        if self.max_term(x+s, u) < u:
            return 0.        
        if moment == self.VARIANCE:
            A2 = self.deferred_insurance(x, s=s, t=t, u=u, moment=2, 
                                         discrete=discrete)
            A1 = self.deferred_insurance(x, s=s, t=t, u=u, discrete=discrete)
            return self.insurance_variance(A2=A2, A1=A1, b=b)
        E = self.E_x(x, s=s, t=u, moment=moment)
        A = self.term_insurance(x, s=s+u, t=t, b=b, moment=moment, 
                                discrete=discrete)
        return E * A  # discount insurance by moment*force of interest

    def endowment_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                            endowment: int = -1, moment: int = 1, 
                            discrete: bool = True) -> float:
        """Endowment insurance: A_x^1:t = term insurance + pure endowment"""
        if moment == self.VARIANCE:
            A2 = self.endowment_insurance(x, s=s, t=t, endowment=endowment, 
                                          b=b, moment=2, discrete=discrete)
            A1 = self.endowment_insurance(x, s=s, t=t, endowment=endowment,
                                          b=b, discrete=discrete)
            return self.insurance_variance(A2=A2, A1=A1, b=b)
        E = self.E_x(x, s=s, t=t, moment=moment)
        A = self.term_insurance(x, s=s, t=t, b=b, moment=moment, discrete=discrete)
        return A + E * (b if endowment < 0 else endowment)**moment

    def increasing_insurance(self, x: int, s: int = 0, t: int = Fractional.WHOLE, 
                             b: int = 1, discrete: bool = True) -> float:
        """Increasing life insurance: (IA)_x"""
        return self.A_x(x, s=s, t=t, benefit=lambda x,t: t * b, 
                        discrete=discrete)

    def decreasing_insurance(self, x, s: int = 0, t: int = 1, b: int = 1,
                             discrete: bool = True) -> float:
        """Decreasing life insurance: (DA)_x"""
        assert t > 0  # decreasing must be term insurance
        A = self.term_insurance(x, t=t, b=b, discrete=discrete)
        n = t + int(discrete)   #  (DA)_x:n + (IA)_x:n = (n+1) A^1_x:n
        return A*n - self.increasing_insurance(x, s=s, t=t, b=b, 
                                               discrete=discrete)

    #
    # Insurance random variable: Y(t)
    #
    def Z_t(self, x: int, prob: float, discrete: bool = True) -> float:
        """T_x given percentile of the r.v. Z: PV of WL or Term insurance"""
        assert prob < 1.0
        t = Insurance.solve(lambda t: self.S(x, 0, t), target=prob, guess=50)
        return math.floor(t) if discrete else t    # opposite of annuity

    def Z_from_t(self, t: float, discrete: bool = True) -> float:
        """PV of insurance payment Z(t), given T_x (or K_x if discrete)"""
        return self.interest.v_t((math.floor(t) + 1) if discrete else t)

    def Z_to_t(self, Z: float) -> float:
        """T_x s.t. PV of insurance payment is Z"""
        #t = Insurance.solve(lambda t: self.Z_from_t(t) - Z, self.MAXAGE/2)
        t = math.log(Z) / math.log(self.interest.v)
        return t

    def Z_from_prob(self, x: int, prob: float, discrete: bool = True) -> float:
        """Percentile of insurance PV r.v. Z, given probability"""
        t = self.Z_t(45, prob)          # opposite of annuity!
        return self.Z_from_t(t, discrete=discrete)  # z is WL or Term Insurance

    def Z_to_prob(self, x: int, Z: float) -> float:
        """Cumulative density of insurance PV r.v. Z, given percentile value"""
        t = self.Z_to_t(Z) 
        return self.S(x, 0, t)      # z is WL or Term Insurance

    def Z_x(self, x, s: int = 0, t: int = 1, discrete: bool = True):
        """APV of year t insurance death benefit"""
        assert t > 0
        if discrete:
            u = math.ceil(t) - 1
            return self.q_x(x, s=s, u=u) * self.interest.v_t(u+1)
        else:
            return self.f_r(x, s=s, t=t) * self.interest.v_t(t)

    def Z_plot(self, x: int, benefit: Callable = lambda x,k: 1,
               T: Optional[float] = None, discrete: bool = True,
               min_t: Optional[int] = None, max_t: Optional[int] = None,
               ax: Any = None, color='r', curve=(), verbose=True) -> float:
        """Plot PV of insurance r.v. Z vs T"""
        min_t = self.MINAGE if min_t is None else min_t
        max_t = self.MAXAGE if max_t is None else max_t
        t = np.arange(min_t, max_t+1)
        y = [benefit(x, k) * self.Z_from_t(k, discrete=discrete) for k in t]
        K = 'K' if discrete else 'T'
        z = 0
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if curve:
            ax2 = ax.twinx()
            ax2.bar(curve[0], curve[1], color='g', alpha=.2, width=1, align='edge')
            if verbose:
                ax2.set_ylabel('Survival Probability', color='g')
        if T is not None:
            ax.step(t, y, ':.', c=color, where='pre' if discrete else 'post')
            xmin, xmax = ax.get_xlim()
            p = self.S(x, 0, T)     # opposite of annuity!
            z = self.Z_from_t(T, discrete=discrete) * benefit(x, T)
            ymin, ymax = ax.get_ylim()
            yjig = (ymax - ymin) / 50
            ax.vlines(T, ymin, z, colors='g', linestyles=':')
            ax.text(T + ((max_t-min_t)/50), ymin, f"{K}={T:.2f}", c='g')
            ax.text(T, z + yjig, f"Z*={z:.2f}", c=color)
            ax.plot(T, z, c=color, marker='o')
            if curve:
                ax2.hlines(p, T, xmax, colors='g', linestyles=':')
                ymin, ymax = ax2.get_ylim()
                yjig = (ymax - ymin) / 50
                ax2.text(xmax, p-yjig, f"Prob={p:.3f}", c='g', 
                         va='top', ha='right')
                ax2.plot(T, p, c='g', marker='o')
            else:
                ax.hlines(z, T, xmax, colors='g', linestyles=':')
                ax.text(xmax, z-yjig, f"Prob={p:.3f}", c='g', 
                        va='top', ha='right')

            if verbose:
                ax.set_title(f"Percentile of Z: Pr[${K}_x$ >= {K}(Z*)] < {p:.3}")
        else:
            ax.bar(t, y, width=1, alpha=0.5, color=color)
            if verbose:
                ax.set_title(f"PV of benefit payment Z(T)")
        if verbose:
            ax.set_ylabel(f"Z(T)", color=color)
            ax.set_xlabel(f"T")
        return z

if __name__ == "__main__":

    print("SOA Question 6.33:  (B) 0.13")
    life = Insurance(mu=lambda x,t: 0.02*t, interest=dict(i=0.03))
    x = 0
    print(life.p_x(x, t=15))
    var = life.E_x(x, t=15, moment=life.VARIANCE, endowment=10000)
    print(var)
    p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
    print(p)
    print()

    print("SOA Question 4.18  (A) 81873 ")
    life = Insurance(interest=dict(delta=0.05), 
                     maxage=10,
                     f=lambda x,s,t: .1 if t < 2 else .4*t**(-2))
    benefit = lambda x,t: 0 if t < 2 else 100000
    prob = 0.9 - life.q_x(0, t=2)
    x, y = life.survival_curve()
    T = life.Z_t(0, prob=prob)
    life.Z_plot(0, T=T, benefit=benefit, discrete=False, curve=(x,y))
    print(life.Z_from_t(T) * benefit(0, T))
    print()

    print("SOA Question 4.10:  (D)")
    life = Insurance(interest=dict(i=0.01), S=lambda x,s,t: 1, maxage=40)
    def fun(x, t):
        if 10 <= t <= 20: return life.interest.v_t(t)
        elif 20 < t <= 30: return 2 * life.interest.v_t(t)
        else: return 0
    def A(x, t):  # Z_x+k (t-k)
        return life.interest.v_t(t - x) * (t > x)
    x = 0
    benefits=[lambda x,t: (life.E_x(x, t=10) * A(x+10, t)
                             + life.E_x(x, t=20)* A(x+20, t)
                             - life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (A(x, t)
                             + life.E_x(x, t=20) * A(x+20, t)
                             - 2 * life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (life.E_x(x, t=10) * A(x, t)
                             + life.E_x(x, t=20) * A(x+20, t)
                             - 2 * life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (life.E_x(x, t=10) * A(x+10, t)
                             + life.E_x(x, t=20) * A(x+20, t)
                             - 2 * life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (life.E_x(x, t=10)
                             * (A(x+10, t)
                                + life.E_x(x+10, t=10) * A(x+20, t)
                                - life.E_x(x+20, t=10) * A(x+30, t)))]
    fig, ax = plt.subplots(3, 2)
    ax = ax.ravel()
    for i, b in enumerate([fun] + benefits):
        life.Z_plot(0, benefit=b, ax=ax[i], verbose=False, color=f"C{i+1}")
        ax[i].legend(["(" + "abcde"[i-1] + ")" if i else "Z"])
    z = [sum(abs(b(0, t) - fun(0, t)) for t in range(40)) for b in benefits]
    print("ABCDE"[np.argmin(z)])
    print()

    print("SOA Question 4.12:  (C) 167")
    cov = Insurance.covariance(a=1.65, b=10.75, ab=0)  # E[Z1 Z2] = 0 nonoverlapping
    print(Insurance.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov))
    print()

    print("SOA Question 4.11:  (A) 143385")
    A1 = 528/1000   # E[Z1]  term insurance
    C1 = 0.209      # E[pure_endowment]
    C2 = 0.136      # E[pure_endowment^2]
    def fun(A2):
        B1 = A1 + C1   # endowment = term + pure_endowment
        B2 = A2 + C2   # double force of interest
        return Insurance.insurance_variance(A2=B2, A1=B1)
    A2 = Insurance.solve(fun, target=15000/(1000*1000), guess=[143400, 279300])
    print(Insurance.insurance_variance(A2=A2, A1=A1, b=1000))
    print()

    print("SOA Question 4.15  (E) 0.0833 ")
    life = Insurance(mu=lambda *x: 0.04,
                     interest=dict(delta=0.06))
    benefit = lambda x,t: math.exp(0.02*t)
    A1 = life.A_x(0, benefit=benefit, discrete=False)
    A2 = life.A_x(0, moment=2, benefit=benefit, discrete=False)
    print(A2 - A1**2)
    print()

    print("SOA Question 4.4  (A) 0.036")
    life = Insurance(f=lambda *x: 0.025, 
                     maxage=40+40,
                     interest=dict(v_t=lambda t: (1 + .2*t)**(-2)))
    benefit = lambda x,t: 1 + .2 * t
    A1 = life.A_x(40, benefit=benefit, discrete=False)
    A2 = life.A_x(40, moment=2, benefit=benefit, discrete=False)
    print(A2 - A1**2)
    print()

    # Example: plot Z vs T
    life = Insurance(interest=dict(delta=0.06), mu=lambda *x: 0.04)
    prob = 0.8
    x = 0
    discrete = False
    t = life.Z_t(0, prob, discrete=discrete)
    Z = life.Z_from_prob(x, prob=prob, discrete=discrete)
    print(t, life.Z_to_t(Z))
    print(Z, life.Z_from_t(t, discrete=discrete))
    print(prob, life.Z_to_prob(x, Z=Z))
    life.Z_plot(0, T=t, discrete=discrete)

    print("Other examples of usage")
    life = Insurance(interest=dict(delta=0.06), mu=lambda *x: 0.04)
    benefit = lambda x,t: math.exp(0.02 * t)
    A1 = life.A_x(0, benefit=benefit)
    A2 = life.A_x(0, moment=2, benefit=benefit)
    print(A1, A2, A2 - A1**2)  # 0.0833

    life = Insurance(interest=dict(delta=0.05), mu=lambda x,s: 0.03)    
    benefit = lambda x,t: math.exp(0.04 * t)
    print(life.A_x(0, benefit=benefit))  #0.75
    print(life.A_x(0, moment=2, benefit=benefit)) #0.60

    life = Insurance(interest=dict(delta=0.08), 
                     maxage=25,
                     S=lambda x,s,t: 1 - (0.02*t + 0.0008*(t**2)))
    print(life.A_x(0)*10000)  #3647
    print()
