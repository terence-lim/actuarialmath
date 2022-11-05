"""Recursive, Interim and Modified Reserves

Copyright 2022, Terence Lim

MIT License
"""
import math
import numpy as np
from mathlc.policyvalues import PolicyValues
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional

class Reserves(PolicyValues):
    """Recursive, Interim and Modified Reserves"""
    _doc = ['set_reserves', 'fill_reserves', 'V_plot', 't_V_forward', 
            't_V_backward', 't_V', 'r_V_forward', 'r_V_backward',
            'FPT_premium', 'FPT_policy_value']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reserves = {'V': {}}
        self.T = 0
    #
    # Set up reserves table for recursion
    #
    def set_reserves(self, T: int = 0, endowment: int = 0,
                     V: Optional[Dict] = None) -> "Reserves":
        # To check that V[0] = 0 if equivalence, and V[MAXAGE] consistent
        if T:
            self.T = T
        if V:
            self._reserves['V'].update(V)
            self.T = max(len(V) - 1, self.T)
        self._reserves['V'][0] = 0  # initial reserve is 0 by equivalence
        self._reserves['V'][self.T] = endowment  # n_V is 0 or endowment
        return self

    def fill_reserves(self, x: int, reserve_benefit: bool = False,
            policy: PolicyValues.Policy = PolicyValues.Policy(),  
            max_iter: int = 4):
        for _ in range(max_iter):
            for t in range(self.T + 1):
                if self._reserves['V'].get(t, None) is not None:
                    continue
                if t == policy.T:
                    v = self.t_V(x=x, t=t, premium=0, benefit=lambda t: 0, 
                                 per_policy = -policy.endowment)
                elif t == 1:
                    v = self.t_V(x=x, t=t, premium=policy.premium, 
                                 benefit=lambda t: policy.benefit,
                                 reserve_benefit=reserve_benefit,
                                 per_premium=policy.initial_premium, 
                                 per_policy=policy.initial_policy)
                elif t == 0:
                    v = 0
                else:
                    v = self.t_V(x=x, t=t, premium=policy.premium, 
                                 benefit=lambda t: policy.benefit,
                                 reserve_benefit=reserve_benefit,
                                 per_premium=policy.renewal_premium, 
                                 per_policy=policy.renewal_policy)
                if v is not None:
                    self._reserves['V'][t] = v

    def V_plot(self, verbose: bool = True, color: str = 'r'):
        fig, ax = plt.subplots(1, 1)
        y = [self._reserves['V'].get(t, None) for t in range(self.T + 1)]
        ax.plot(list(range(self.T + 1)), y, '.', color=color)
        if verbose:
            ax.set_title(f"Policy Value t_V")
            ax.set_ylabel(f"t_V", color=color)
            ax.set_xlabel(f"T")


    #
    # Reserves recursion
    #
    def t_V_forward(self, x: int, t: int = 0, premium: float = 0, 
                   benefit: Callable = lambda t: 1, 
                   per_premium: float = 0, per_policy: float = 0,
                   reserve_benefit: bool = False) -> Optional[float]:
        """Forward recursion (allows for optional reserve benefit"""
        if t+1 not in self._reserves['V']:
            return None
        V = self._reserves['V'][t+1]
        b = benefit(t+1) + V * reserve_benefit   # total death benefit 
        if V == b:   # special case if death benefit == forward reserve
            V = b
        else:
            if V:
                V *= self.p_x(x+t)
            if b:
                V += self.q_x(x+t) * b
        V = V * self.interest.v - (premium*(1 - per_premium) - per_policy)
        return V

    def t_V_backward(self, x: int, t: int = 0, premium: float = 0,
                    benefit: Callable = lambda t: 1, 
                    per_premium: float = 0, per_policy: float = 0,
                    reserve_benefit: bool = False) -> Optional[float]:
        """Backward recursion (allows for optional reserve benefit)"""
        if t-1 not in self._reserves['V']:
            return None
        V = self._reserves['V'][t-1]
        V = (V + premium*(1 - per_premium) - per_policy) / self.interest.v
        if benefit(t):
            V -= self.q_x(x+t-1) * benefit(t)
        if not reserve_benefit:
            V /= self.p_x(x+t-1)
        return V

    def t_V(self, x: int, t: int = 0,
            premium: float = 0, benefit: Callable = lambda t: 1, 
            reserve_benefit: bool = False,
            per_premium: float = 0, per_policy: float = 0) -> float:
        """Try to solve time-t Reserve by forward or backward recursion"""
        if t in self._reserves['V']:   # already solved in reserves table
            return self._reserves['V'][t]
        V = self.t_V_backward(x=x, t=t, premium=premium, 
                              benefit=benefit,
                              reserve_benefit=reserve_benefit, 
                              per_premium=per_premium, 
                              per_policy=per_policy)
        if V is not None:
            return V
        V = self.t_V_forward(x=x, t=t, premium=premium, 
                             benefit=benefit,
                             reserve_benefit=reserve_benefit, 
                             per_premium=per_premium, 
                             per_policy=per_policy)
        if V is not None:
            return V

    #
    # Interim reserves
    #
    def r_V_forward(self, x: int, s: int = 0, r: float = 0,
                   premium: float = 0, benefit: int = 1) -> Optional[float]:
        """Forward recursion for interim reserves"""
        s = int(s + r)
        r = r - math.floor(r)
        if s+1 not in self._reserves['V']:        # forward recursion
            return None
        V = self._reserves['V'][s+1]
        if V:
            V *= self.p_r(x, s=s, r=r, t=1-r)
        if benefit:
            V += self.q_r(x, s=s, r=r, t=1-r) * benefit
        V = V * self.interest.v_t(1-r) - premium
        return V

    def r_V_backward(self, x: int, s: int = 0, r: float = 0,
                    premium: float = 0, benefit: int = 1) -> Optional[float]:
        """Backward recursion for interim reserves"""
        s = int(s + r)
        r = r - math.floor(r)
        if s not in self._reserves['V']:
            return None
        V = self._reserves['V'][s]
        V = (V + premium) / self.interest.v_t(r)
        if benefit:
            V -= self.q_r(x, s=s, t=r) * benefit * self.interest.v_t(1-r)
        V /= self.p_r(x, s=s, t=r)
        return V

    #
    # Full Preliminary Term (FPT) modified reserves
    #
    def FPT_premium(self, x: int, s: int = 0, n: int = PolicyValues.WHOLE, 
                    b: int = 1, first: bool = False) -> float:
        """Initial or renewal Full Preliminary Term premiums"""
        if first:
            return self.net_premium(x, s=s, b=b, t=1)
        else:
            return self.net_premium(x, s=s+1, b=b, t=self.add_term(n, -1))

    def FPT_policy_value(self, x: int, s: int = 0, t: int = 0, b: int = 1,
                         n: int = PolicyValues.WHOLE,
                         endowment: int = 0, discrete: bool = True) -> float:
        """Compute Full Preliminary Term policy value at time t"""
        if t in [0, 1]:  # FPT is 0 at t = 0 or 1
            return 0
        else:
            return self.net_policy_value(x, s=s+1, t=t-1, n=self.add_term(n,-1),
                                         b=b, endowment=endowment,
                                         discrete=discrete)


if __name__ == "__main__":
    from sult import SULT

    print("SOA Question 7.31:  (E) 0.310")
    x = 0
    life = Reserves().set_reserves(T=3)
    print(life._reserves)
    G = 368.05
    def fun(P):  # solve net premium from expense reserve equation
        return life.t_V(x=x, t=2, premium=G-P, benefit=lambda t: 0, 
                        per_policy=5 + .08*G)
    P = life.solve(fun, target=-23.64, guess=[.29, .31]) / 1000
    print(P)
    print()
    
    print("SOA Question 7.13: (A) 180")
    life = SULT()
    V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
    print(V)
    print()
    

    # print("Plot example: TODO from 6.12 -- this needs more work!!!")
    # life = PolicyValues(interest=dict(i=0.06))
    # a = 12
    # A = life.insurance_twin(a)
    # policy = life.Policy(benefit=1000, settlement_policy=20, 
    #                      initial_policy=10, initial_premium=0.75, 
    #                      renewal_policy=2, renewal_premium=0.1)
    # policy.premium = life.gross_premium(A=A, a=a, **policy.premium_terms)
    # life = Reserves(interest=dict(delta=0.06), mu=lambda x,s: 0.04)
    # life.set_reserves(T=100)
    # life.fill_reserves(x=0, policy=policy)
    # life.V_plot()
    # plt.show()
