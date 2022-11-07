"""Policy Values

Copyright 2022, Terence Lim

MIT License
"""
import math
import numpy as np
import scipy
from actuarialmath.premiums import Premiums
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class PolicyValues(Premiums):
    """Policy Values"""
    _help = ['net_future_loss', 'net_variance_loss', 'net_policy_variance', 
             'gross_future_loss', 'gross_policy_variance', 'gross_policy_value',
             'L_from_t', 'L_to_t', 'L_from_prob', 'L_to_prob', 'L_plot']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Policy:
        """Class to store policy expenses and benefits"""
        def __init__(self, 
                     premium: float = 1,
                     initial_policy: float = 0, 
                     initial_premium: float = 0, 
                     renewal_policy: float = 0, 
                     renewal_premium: float = 0,
                     settlement_policy: float = 0,
                     benefit: float = 1, 
                     endowment: float = 0,
                     T: int = Premiums.WHOLE,
                     discrete: bool = True):
            self.premium=premium
            self.benefit=benefit
            self.discrete=discrete
            self.initial_policy=initial_policy
            self.initial_premium=initial_premium
            self.renewal_policy=renewal_policy
            self.renewal_premium=renewal_premium
            self.settlement_policy=settlement_policy
            self.endowment=endowment
            self.T = T
        
        def set(self, **terms) -> Any:
            for key, value in terms.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            return self

        @property
        def premium_terms(self) -> Dict:    # terms required by gross_premiums 
            """Return dict of terms required for calculating premiums"""
            return dict(benefit=self.benefit,
                        initial_policy=self.initial_policy,
                        initial_premium=self.initial_premium,
                        renewal_policy=self.renewal_policy,
                        renewal_premium=self.renewal_premium,
                        settlement_policy=self.settlement_policy)

        @property
        def future(self) -> Any:
            """New policy object with initial terms set to renewal, for t > 0"""
            return PolicyValues.Policy(benefit=self.benefit,
                                       renewal_policy=self.renewal_policy,
                                       renewal_premium=self.renewal_premium,
                                       initial_policy=self.renewal_policy,
                                       initial_premium=self.renewal_premium,
                                       settlement_policy=self.settlement_policy,
                                       discrete=self.discrete,
                                       premium=self.premium,
                                       endowment=self.endowment, T=self.T)

        @property
        def renewal_profit(self) -> float:
            """Renewal premium, less renewal per premium and policy expenses"""
            return ((self.premium * (1 - self.renewal_premium)) 
                    - self.renewal_policy)

        @property
        def initial_cost(self) -> float:
            """Initial per premium and per policy expense, less premium"""
            return ((self.initial_policy - self.renewal_policy)
                    + (self.premium * (self.initial_premium 
                                       - self.renewal_premium)))

        @property
        def claims_cost(self) -> float:
            """Total claims costs = death benefit + settlement expense"""
            return self.benefit + self.settlement_policy

    #
    # Net Future Loss shortcuts for WL and Endowment Insurance 
    #
    def net_future_loss(self, A: float, A1: float, b: int = 1) -> float:
        """Assume WL or Endowment Ins for shortcuts since P from equivalence"""
        return b * (A1 - A) / (1 - A)

    def net_variance_loss(self, A1: float, A2: float, A: float = 0, 
                         b: int = 1) -> float:
        """Helper for variance of net loss shortcuts of WL or Endowment Ins loss"""
        if not A:
            A = A1   # assume t = 0 => A = A1
        return b**2 * (A2 - A1**2) / (1 - A)**2

    def net_policy_variance(self, x, s: int = 0, t: int = 0, b: int = 1, 
                            n: int = Premiums.WHOLE, endowment: int = 0, 
                            discrete: bool = True) -> float:
        """Shortcuts for variance of future loss for WL or Endow Ins"""
        if n < 0:             # Whole Life
            A2 = self.whole_life_insurance(x, s=s+t, moment=2,
                                            discrete=discrete)
            A1 = self.whole_life_insurance(x, s=s+t, discrete=discrete)
            A = self.whole_life_insurance(x, s=s, discrete=discrete)
        elif endowment == b:   # Endowment Insurance 
            n = self.max_term(x+s+t, n)
            A2 = self.endowment_insurance(x, s=s+t, t=n-t, moment=2,
                                            discrete=discrete)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, discrete=discrete)
            A = self.endowment_insurance(x, s=s, t=n, discrete=discrete)
        else:
            raise Exception("Variances for WL and Endowment Ins only")
        return self.net_variance_loss(A=A, A1=A1, A2=A2, b=b)

    #
    # Net Policy Value for special insurance
    #
    def net_policy_value(self, x: int, s: int = 0, t: int = 0, b: int = 1, 
                         n: int = Premiums.WHOLE, endowment: int = 0, 
                         discrete: bool = True) -> float:
        """Net policy values where premiums from equivalence: E[L_t]"""
        if n < 0:             # Shortcut available for Whole Life
            A1 = self.whole_life_insurance(x, s=s+t, discrete=discrete)
            A = self.whole_life_insurance(x, s=s, discrete=discrete)
        elif endowment == b:  # Shortcut available for Endowment Insurance
            n = self.max_term(x+s+t, n)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, 
                                            discrete=discrete)
            A = self.endowment_insurance(x, s=s, t=n, discrete=discrete)
        else:  # Special Term or (unequal) Endowment insurance
            n = self.max_term(x+s+t, n)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, discrete=discrete,
                                            b=b, endowment=endowment)
            a1 = self.temporary_annuity(x, s=s+t, t=n-t, b=b, 
                                        discrete=discrete)
            A = self.endowment_insurance(x, s=s, t=n, discrete=discrete,
                                            b=b, endowment=endowment)
            a = self.temporary_annuity(x, s=s, t=n, b=b, discrete=discrete)
            return A1 - a1 * (A / a)
        return self.net_future_loss(A=A, A1=A1, b=b) # Shortcut for WL or Endowment

    #
    # Gross Future Loss shortcuts for WL and Endowment Insurance 
    #
    def gross_future_loss(self, A: Optional[float] = None, 
                          a: Optional[float] = None, 
                          policy: Policy = Policy()) -> float:
        """Shortcut for WL or Endowment Ins gross future loss"""
        if a is None:    # assume WL or Endowment Insurance for twin annuity
            a = self.annuity_twin(A, discrete=policy.discrete)
        elif A is None:  # assume WL or Endowment Insurance for twin annuity
             A = self.insurance_twin(a, discrete=policy.discrete)
        return ((A * policy.claims_cost + policy.initial_cost)
                - (a * policy.renewal_profit))

    def gross_variance_loss(self, A1: float, A2: float = 0,
                            policy: Policy = Policy()) -> float:
        """Helper for variance of gross loss shortcuts for WL or Endowment Ins"""
        interest = self.interest.d if policy.discrete else self.interest.delta
        return (((policy.renewal_profit / interest) + policy.benefit 
                    + policy.settlement_policy)**2 * (A2 - A1**2))

    def gross_policy_variance(self, x: int, s: int = 0, t: int = 0,
                           n: int = Premiums.WHOLE,
                           policy: Policy = Policy()) -> float:
        """Shortcut for gross policy value of WL and Endowment Insurance"""
        if n < 0:  # WL
            A2 = self.whole_life_insurance(x, s=s+t, moment=2,
                                            discrete=policy.discrete)
            A1 = self.whole_life_insurance(x, s=s+t, 
                                            discrete=policy.discrete)
        elif policy.endowment == policy.claims_cost:  # Endowment
            n = self.max_term(x+s+t, n)
            A2 = self.endowment_insurance(x, s=s+t, t=n-t, moment=2, 
                                            discrete=policy.discrete)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, 
                                            discrete=policy.discrete)
        else:
            raise Exception("Variance for WL or Endowment Ins only")
        return self.gross_variance_loss(A1=A1, A2=A2, policy=policy)

    #
    # Gross Policy Value for special insurance
    #
    def gross_policy_value(self, x: int, s: int = 0, t: int = 0,
                           n: int = Premiums.WHOLE,
                           policy: Policy = Policy()) -> float:
        """Gross policy values for insurance: t_V = E[L_t]"""
        if n < 0:    # Whole life shortcut
            A = self.whole_life_insurance(x, s=s+t, 
                                          discrete=policy.discrete)
        elif policy.endowment == policy.claims_cost:  # Endowment Ins shortcut
            n = self.max_term(x+s+t, n)
            A = self.endowment_insurance(x, s=s+t, t=n-t, 
                                            discrete=policy.discrete)
        else:  # Special term insurance
            n = self.max_term(x+s+t, n)
            A = self.term_insurance(x, s=s+t, t=n-t, discrete=policy.discrete)
            a = self.temporary_annuity(x, s=s+t, t=n-t, discrete=policy.discrete)
            endowment = 0
            if policy.endowment:  # endowment not equal to claims cost
                endowment = self.E_x(x, s=s+t, t=n-t) * policy.endowment
            initial_cost = 0 if t else policy.initial_cost
            return (A * policy.claims_cost + initial_cost + endowment
                    - (a * policy.renewal_profit))
        policy = policy.future if t else policy  # if t>0, ignore init expense
        return self.gross_future_loss(A=A, policy=policy)

    #
    # Future Loss random variable: L(t)
    #
    def L_from_t(self, t: float, policy: Policy = Policy()) -> float:
        """PV of Loss L(t), given T_x (or K_x if discrete)"""
        k = math.floor(t) if policy.discrete else t  # if endowment paid
        if policy.T > 0 and k >= policy.T:
            t = policy.T
            endowment = policy.endowment * self.Z_from_t(t)
        else:
            endowment = 0
        return ((policy.claims_cost 
                 * self.Z_from_t(t, discrete=policy.discrete))
                + policy.initial_cost 
                + endowment
                - (policy.renewal_profit 
                   * self.Y_from_t(t, discrete=policy.discrete)))

    def L_to_t(self, L: float, policy: Policy = Policy()) -> float:
        """T_x s.t. PV of loss is Z"""
#        t = PolicyValues.solve(lambda t: self.L_from_t(t, policy) - L, 10)
        t = scipy.optimize.minimize_scalar(lambda t: abs(self.L_from_t(t, policy) - L), 
                                           bounds=(0, self.MAXAGE)).x
        return t

    def L_from_prob(self, x: int, prob: float, 
                   policy: Policy = Policy()) -> float:
        """Percentile of loss PV r.v. L, given probability"""
        t = self.Z_t(x, prob, discrete=policy.discrete)
        return self.L_from_t(t, policy)

    def L_to_prob(self, x: int, L: float, policy: Policy = Policy()) -> float:
        """Cumulative density of loss PV r.v. L, given percentile value"""
        t = self.L_to_t(L, policy)
        return self.S(x, 0, t)

    def L_plot(self, x: int, T: Optional[float] = None, policy: Policy = Policy(),
               min_t: int = 0, max_t: Optional[int] = None,
               ax: Any = None, color='r', curve=(), verbose=True) -> float:
        """Plot loss r.v. L vs T"""        
        max_t = self.MAXAGE - x if max_t is None else max_t
        t = np.arange(min_t, max_t+1)
        y = [self.L_from_t(k, policy=policy) for k in t]
        K = 'K' if policy.discrete else 'T'
        z = 0
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if curve:
            ax2 = ax.twinx()
            ax2.bar(curve[0], curve[1], color='g', alpha=.2, width=1, align='edge')
            if verbose:
                ax2.set_ylabel('Survival Probability', color='g')
        if T is not None:
            ax.plot(t, y, '.', c=color)
            xmin, xmax = ax.get_xlim()
            p = self.S(x, 0, T)     # like annuity
            z = self.L_from_t(T, policy=policy)
            ymin, ymax = ax.get_ylim()
            yjig = (ymax - ymin) / 50
            ax.vlines(T, ymin, z, colors='g', linestyles=':')
            ax.plot(T, z, c=color, marker='o')
            ax.text(T + ((max_t-min_t)/50), ymin, f"{K}={T:.2f}", c='g')
            ax.text(T, z + yjig, f"Z*={z:.2f}", c=color)
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
                ax.set_title(f"Percentile of L: Pr[${K}_x$ >= {K}(Z*)] > {p:.3}")
        else:
            ax.plot(t, y, '.', color=color)
            if verbose:
                ax.set_title(f"PV of Loss L(T)")
        if verbose:
            ax.set_ylabel(f"L(T)", color=color)
            ax.set_xlabel(f"T")
        return z

if __name__ == "__main__":
    from actuarialmath.sult import SULT
    print(PolicyValues.help())
    
    print("SOA Question 6.24:  (E) 0.30")
    life = PolicyValues(interest=dict(delta=0.07))
    x, A1 = 0, 0.30   # Policy for first insurance
    P = life.premium_equivalence(A=A1, discrete=False)  # Need its premium
    policy = life.Policy(premium=P, discrete=False)
    def fun(A2):  # Solve for A2, given Var(Loss)
        return life.gross_variance_loss(A1=A1, A2=A2, policy=policy)
    A2 = life.solve(fun, target=0.18, guess=0.18)
    print()
    
    policy = life.Policy(premium=0.06, discrete=False) # Solve second insurance
    variance = life.gross_variance_loss(A1=A1, A2=A2, policy=policy)
    print(variance)
    print()

    print("SOA Question 6.30:  (A) 900")
    life = PolicyValues(interest=dict(i=0.04))
    policy = life.Policy(premium=2.338, benefit=100, initial_premium=.1,
                         renewal_premium=0.05)
    var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
                                   A2=0.17, policy=policy)
    print(var)
    print()

    print("SOA Question 7.32:  (B) 1.4")
    life = PolicyValues(interest=dict(i=0.06))
    policy = life.Policy(benefit=1, premium=0.1)
    def fun(A2):
        return life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    A2 = life.solve(fun, target=0.455, guess=0.455)
    policy = life.Policy(benefit=2, premium=0.16)
    var = life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    print(var)
    print()

    print("SOA Question 6.12:  (E) 88900")
    life = PolicyValues(interest=dict(i=0.06))
    a = 12
    A = life.insurance_twin(a)
    policy = life.Policy(benefit=1000, settlement_policy=20, 
                         initial_policy=10, initial_premium=0.75, 
                         renewal_policy=2, renewal_premium=0.1)
    policy.premium = life.gross_premium(A=A, a=a, **policy.premium_terms)
    print(A, policy.premium)
    L = life.gross_variance_loss(A1=A, A2=0.14, policy=policy)
    print(L)
    print()

    print("Plot Example -- SOA Question 6.6:  (B) 0.79")
    life = SULT()
    P = life.net_premium(62, b=10000)
    policy = life.Policy(premium=1.03*P, renewal_policy=5,
                         initial_policy=5, initial_premium=0.05, benefit=10000)
    L = life.gross_policy_value(62, policy=policy)
    var = life.gross_policy_variance(62, policy=policy)
    prob = life.portfolio_cdf(mean=L, variance=var, value=40000, N=600)
    print(prob, 0.79)
    life.L_plot(62, policy=policy)
    print()

    print("Plot Example -- SOA QUestion 7.6:  (E) -25.4")
    life = SULT()
    P = life.net_premium(45, b=2000)
    policy = life.Policy(benefit=2000, initial_premium=.25, renewal_premium=.05,
                         initial_policy=2*1.5 + 30, renewal_policy=2*.5 + 10)
    G = life.gross_premium(a=life.whole_life_annuity(45), **policy.premium_terms)
    gross = life.gross_policy_value(45, t=10, policy=policy.set(premium=G))
    net = life.net_policy_value(45, t=10, b=2000)
    V = gross - net
    print(V, -25.4)
    T = life.L_to_t(G, policy=policy)
    print(G)
    life.L_plot(45, T=int(T), policy=policy)

    plt.show()
