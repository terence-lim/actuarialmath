"""Reserves - Computes recursive, interim and modified reserves

MIT License. Copyright 2022-2023 Terence Lim
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any
from actuarialmath import PolicyValues, Contract

class Reserves(PolicyValues):
    """Compute recursive, interim and modified reserves

    Examples:
      >>> x = 0
      >>> life = Reserves().set_reserves(T=3)
      >>> G = 368.05
      >>> def fun(P):  # solve net premium from expense reserve equation
      >>>     return life.t_V(x=x, t=2, premium=G-P, benefit=lambda t: 0,
      >>>                     per_policy=5+.08*G)
      >>> P = life.solve(fun, target=-23.64, grid=[.29, .31]) / 1000
      >>> life = SULT()
      >>> x, T, b = 50, 20, 500000    # $500K 20-year term insurance for (50)
      >>> P = life.net_premium(x=x, t=T, b=b)
      >>> life.set_reserves(T=T).fill_reserves(x=x, contract=Contract(premium=P, benefit=b))
      >>> life.V_plot(title=f"Reserves for ${b} {T}-year term insurance issued to ({x})")
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reserves = {'V': {}}
        self.T = 0

    #
    # Set up reserves table for recursion
    #
    def set_reserves(self, T: int = 0, endowment: int | float= 0,
                     V: Dict[int, float] | None = None) -> "Reserves":
        """Set values of the reserves table and the endowment benefit amount

        Args:
          T : max term of policy
          V : reserve values, keyed by time t
          endowment : endowment benefit amount
        """
        if T:
            self.T = T
        if V:
            self._reserves['V'].update(V)
            self.T = max(len(V) - 1, self.T)
        self._reserves['V'][0] = 0  # initial reserve is 0 by equivalence
        self._reserves['V'][self.T] = endowment  # n_V is 0 or endowment
        return self

    def fill_reserves(self, x: int, s: int = 0, reserve_benefit: bool = False,
                      contract: Contract | None = None) -> "Reserves":
        """Iteratively fill in missing values in reserves table

        Args:
          x : age selected
          s : starting from s years after selection
          reserve_benefit : whether benefit includes value of reserves
          contract : policy contract terms and expenses
        """
        contract = contract or Contract()        
        for _ in range(2):
            for t in range(self.T + 1):
                if self._reserves['V'].get(t, None) is not None:
                    continue
                if t == contract.T:
                    v = self.t_V(x=x, s=s, t=t, premium=0, benefit=lambda t: 0, 
                                 per_policy = -contract.endowment)
                elif t == 1:
                    v = self.t_V(x=x, s=s, t=t, premium=contract.premium, 
                                 benefit=lambda t: contract.benefit,
                                 reserve_benefit=reserve_benefit,
                                 per_premium=contract.initial_premium, 
                                 per_policy=contract.initial_policy)
                elif t == 0:
                    v = 0
                else:
                    v = self.t_V(x=x, s=s, t=t, premium=contract.premium, 
                                 benefit=lambda t: contract.benefit,
                                 reserve_benefit=reserve_benefit,
                                 per_premium=contract.renewal_premium, 
                                 per_policy=contract.renewal_policy)
                if v is not None:
                    self._reserves['V'][t] = v
        return self

    def V_plot(self, ax: Any = None, color: str = 'r', title: str = ''):
        """Plot values from reserves tables

        Args:
          title : title to display
          color : color to plot curve
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        y = [self._reserves['V'].get(t, None) for t in range(self.T + 1)]
        ax.plot(list(range(self.T + 1)), y, '.', color=color)
        ax.set_title(title)
        ax.set_ylabel(f"$_tV$", color=color)
        ax.set_xlabel(f"t")

    def V_t(self):
        """Returns reserves table as a DataFrame"""
        return pd.DataFrame(self._reserves)\
                 .rename_axis('t')\
                 .rename(columns={'V':'V_t'})

    #
    # Reserves recursion
    #
    def t_V_backward(self, x: int, s: int = 0, t: int = 0, premium: float = 0, 
                     benefit: Callable = lambda t: 1, 
                     per_premium: float = 0, per_policy: float = 0,
                     reserve_benefit: bool = False) -> float | None:
        """Backward recursion (with optional reserve benefit)

        Args:
          x : age selected
          s : starting s years after selection
          t : year of reserve to solve
          benefit : benefit amount at t+1
          premium : amount of premium paid just after t
          per_premium : expense per $ premium
          per_policy : expense per policy
          reserve_benefit : whether reserve value at t+1 included in benefit
        """
        if t+1 not in self._reserves['V']:
            return None
        V = self._reserves['V'][t+1]
        b = benefit(t+1) + V * reserve_benefit   # total death benefit 
        if V == b:   # special case if death benefit == forward reserve
            V = b
        else:
            if V:
                V *= self.p_x(x=x, s=s+t)
            if b:
                V += self.q_x(x=x, s=s+t) * b
        V = V * self.interest.v - (premium*(1 - per_premium) - per_policy)
        return V

    def t_V_forward(self, x: int, s: int = 0, t: int = 0, premium: float = 0,
                    benefit: Callable = lambda t: 1, 
                    per_premium: float = 0, per_policy: float = 0,
                    reserve_benefit: bool = False) -> float | None:
        """Forward recursion (with optional reserve benefit)

        Args:
          x : age selected
          s : starting s years after selection
          t : year of reserve to solve
          benefit : benefit amount at t
          premium : amount of premium paid just after t-1
          per_premium : expense per $ premium
          per_policy : expense per policy
          reserve_benefit : whether reserve value at t included in benefit
        """
        if t-1 not in self._reserves['V']:
            return None
        V = self._reserves['V'][t-1]
        V = (V + premium*(1 - per_premium) - per_policy) / self.interest.v
        if benefit(t):
            V -= self.q_x(x=x, s=s+t-1) * benefit(t)
        if not reserve_benefit:
            V /= self.p_x(x=x, s=s+t-1)
        return V

    def t_V(self, x: int, s: int = 0, t: int = 0,
            premium: float = 0, benefit: Callable = lambda t: 1, 
            reserve_benefit: bool = False,
            per_premium: float = 0, per_policy: float = 0) -> float | None:
        """Solve year-t reserves by forward or backward recursion

        Args:
          x : age selected
          s : starting s years after selection
          t : year of reserve to solve
          benefit : benefit amount
          premium : amount of premium
          per_premium : expense per $ premium
          per_policy : expense per policy
          reserve_benefit : whether reserve value included in benefit
        """
        if t in self._reserves['V']:   # already solved in reserves table
            return self._reserves['V'][t]
        V = self.t_V_forward(x=x, s=s, t=t, premium=premium, 
                             benefit=benefit,
                             reserve_benefit=reserve_benefit, 
                             per_premium=per_premium, 
                             per_policy=per_policy)
        if V is not None:
            return V
        V = self.t_V_backward(x=x, s=s, t=t, premium=premium, 
                              benefit=benefit,
                              reserve_benefit=reserve_benefit, 
                              per_premium=per_premium, 
                              per_policy=per_policy)
        if V is not None:
            return V

    #
    # Interim reserves
    #
    def r_V_backward(self, x: int, s: int = 0, r: float = 0,
                   benefit: int = 1) -> float | None:
        """Backward recursion for interim reserves

        Args:
          x : age of selection
          s : years after selection
          r : solve for interim reserve at fractional year x+s+r
          benefit : benefit amount in year x+s+1
        """
        s = int(s + r)
        r = r - math.floor(r)
        if s+1 not in self._reserves['V']:        # forward recursion
            return None
        V = self._reserves['V'][s+1]
        if V:
            V *= self.p_r(x, s=s, r=r, t=1-r)
        if benefit:
            V += self.q_r(x, s=s, r=r, t=1-r) * benefit
        V = V * self.interest.v_t(1-r)
        return V

    def r_V_forward(self, x: int, s: int = 0, r: float = 0,
                    premium: float = 0, benefit: int = 1) -> float | None:
        """Forward recursion for interim reserves

        Args:
          x : age of selection
          s : years after selection
          r : solve for interim reserve at fractional year x+s+r
          benefit : benefit amount in year x+s+1
          premium : premium amount just after year x+s
        """
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
        """Initial or renewal Full Preliminary Term premiums

        Args:
          x : age of selection
          s : years after selection
          n : term of insurance
          b : benefit amount in year x+s+1
          first : calculate year 1 (True) or year 2+ (False) FPT premium
        """
        if first:
            return self.net_premium(x, s=s, b=b, t=1)
        else:
            return self.net_premium(x, s=s+1, b=b, t=self.add_term(n, -1))

    def FPT_policy_value(self, x: int, s: int = 0, t: int = 0, b: int = 1,
                         n: int = PolicyValues.WHOLE,
                         endowment: int = 0, discrete: bool = True) -> float:
        """Compute Full Preliminary Term policy value at time t

        Args:
          x : age of selection
          s : years after selection
          n : term of insurance
          t : year of policy value to calculate
          b : benefit amount in year x+s+1
          endowment : endowment amount
          discrete : fully discrete (True) or continuous (False) insurance
        """
        if t in [0, 1]:  # FPT is 0 at t = 0 or 1
            return 0
        else:
            return self.net_policy_value(x, s=s+1, t=t-1, n=self.add_term(n,-1),
                                         b=b, endowment=endowment,
                                         discrete=discrete)


if __name__ == "__main__":
    from actuarialmath.sult import SULT
    from actuarialmath.policyvalues import Contract
    life = SULT()
    x, T, b = 50, 20, 500000    # $500K 20-year term insurance for (50)
    P = life.net_premium(x=x, t=T, b=b)
    life.set_reserves(T=T)\
        .fill_reserves(x=x, contract=Contract(premium=P, benefit=b))
    life.V_plot(title=f"Reserves for ${b} {T}-year term insurance issued to ({x})")

if False:
    print("SOA Question 7.31:  (E) 0.310")
    x = 0
    life = Reserves().set_reserves(T=3)
    print(life._reserves)
    G = 368.05
    def fun(P):  # solve net premium from expense reserve equation
        return life.t_V(x=x, t=2, premium=G-P, benefit=lambda t: 0,
                        per_policy=5+.08*G)
    P = life.solve(fun, target=-23.64, grid=[.29, .31]) / 1000
    print(P)
    print()

    from actuarialmath.sult import SULT
    print("SOA Question 7.13: (A) 180")
    life = SULT()
    V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
    print(V)
    print()
