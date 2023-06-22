"""Policy Values - Computes present value of future losses and reserves

MIT License. Copyright 2022-2023 Terence Lim
"""
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import Dict, Any
from actuarialmath import Premiums, Actuarial

class Contract(Actuarial):
    """Set and retrieve policy contract terms

    Args:
      premium : level premium amount
      benefit : insurance death benefit amount
      settlement_policy : settlement expense per policy
      endowment : endowment benefit amount
      initial_policy : first year total expense per policy
      initial_premium : first year total premium per $ of gross premium
      renewal_policy : renewal expense per policy
      renewal_premium : renewal premium per $ of gross premium
      discrete : annuity due (True) or continuous (False)
      T : term of insurance
      discrete : annuity due (True) or continuous (False)        

    Examples:
      >>> life = PolicyValues().set_interest(i=0.06)
      >>> a = 12
      >>> A = life.insurance_twin(a)
      >>> contract = Contract(benefit=1000, settlement_policy=20, 
      >>>                     initial_policy=10, initial_premium=0.75, 
      >>>                     renewal_policy=2, renewal_premium=0.1)
      >>> contract.premium = life.gross_premium(A=A, a=a, **contract.premium_terms)
      >>> print(A, contract.premium)
      >>> L = life.gross_variance_loss(A1=A, A2=0.14, contract=contract)
    """
    
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
    
    def set_contract(self, **terms) -> Any:
        """Update any existing policy contract terms

        Args:
          **kwargs : one or more contract policy terms, and its value to update
        """
        for key, value in terms.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    @property
    def premium_terms(self) -> Dict:    # terms required by gross_premiums 
        """Getter returns dict of terms required for calculating gross premiums"""
        return dict(benefit=self.benefit,
                    initial_policy=self.initial_policy,
                    initial_premium=self.initial_premium,
                    renewal_policy=self.renewal_policy,
                    renewal_premium=self.renewal_premium,
                    settlement_policy=self.settlement_policy)

    def renewals(self, t: int = 0) -> "Contract":
        """Returns contract object with initial terms set to renewal terms

        Args:
          t : number of years after initial
        """
        return Contract(benefit=self.benefit,
                        renewal_policy=self.renewal_policy,
                        renewal_premium=self.renewal_premium,
                        initial_policy=self.renewal_policy,
                        initial_premium=self.renewal_premium,
                        settlement_policy=self.settlement_policy,
                        discrete=self.discrete,
                        premium=self.premium,
                        endowment=self.endowment,
                        T=self.T - t)

    @property
    def renewal_profit(self) -> float:
        """Getter returns renewal dollar profit (premium less renewal expenses)"""
        # premium less renewal per premium and expense"""
        return ((self.premium * (1 - self.renewal_premium)) - self.renewal_policy)

    @property
    def initial_cost(self) -> float:
        """Getter returns total initial cost (excludes renewal expenses)"""
        return ((self.initial_policy - self.renewal_policy) +
                (self.premium * (self.initial_premium - self.renewal_premium)))

    @property
    def claims_cost(self) -> float:
        """Getter returns total claims cost (death benefit + settlement expense)"""
        return self.benefit + self.settlement_policy

class PolicyValues(Premiums):
    """Compute net and gross future losses and policy values

    Examples:
      >>> life = PolicyValues().set_interest(i=0.04)
      >>> contract = Contract(premium=2.338, benefit=100, initial_premium=.1, 
      >>>                     renewal_premium=0.05)
      >>> var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
      >>>                                A2=0.17, contract=contract)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #
    # Net Future Loss shortcuts for WL and Endowment Insurance 
    #
    def net_future_loss(self, A: float, A1: float, b: int = 1) -> float:
        """Shortcuts for WL or Endowment Insurance net loss

        Args:
          A : insurance factor at age (x)
          A1 : insurance factor at t years after x
          b : benefit amount
        """
        return b * (A1 - A) / (1 - A)

    def net_variance_loss(self, A1: float, A2: float, A: float = 0, 
                         b: int = 1) -> float:
        """Shortcuts for variance of net loss of WL or Endowment Insurance

        Args:
          A : insurance factor at age (x)
          A1 : first moment of insurance factor at t years after x
          A2 : insurance factor at double force of interest t years after x
          b : benefit amount
        """
        if not A:
            A = A1   # assume t = 0 => A = A1
        return b**2 * (A2 - A1**2) / (1 - A)**2

    def net_policy_variance(self, x, s: int = 0, t: int = 0, b: int = 1, 
                            n: int = Premiums.WHOLE, endowment: int = 0, 
                            discrete: bool = True) -> float:
        """Variance of future loss for WL or Endowment Ins assuming equivalence

        Args:
          x : age of selection
          s : years after selection
          t : term of life annuity in years
          n : number of years premiums paid
          b : benefit amount
          endowment : endowment amount
          discrete : annuity due (True) or continuous (False)
        """
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
    # Net Policy Value for special and WL/endowment insurance
    #
    def net_policy_value(self, x: int, s: int = 0, t: int = 0, b: int = 1, 
                         n: int = Premiums.WHOLE, endowment: int = 0, 
                         discrete: bool = True) -> float:
        """Net policy value assuming premiums from equivalence: E[L_t]

        Args:
          x : age initially insured
          s : years after selection
          n : number of years of premiums paid
          t : year of death
          b : benefit amount
          endowment : endowment amount
          discrete : discrete/annuity due (True) or continuous (False)
        """
        if n < 0:             # Shortcut available for Whole Life
            A1 = self.whole_life_insurance(x, s=s+t, discrete=discrete)
            A = self.whole_life_insurance(x, s=s, discrete=discrete)
        elif endowment == b:  # Shortcut available for (equal) Endowment Insurance
            n = self.max_term(x+s+t, n)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, 
                                            discrete=discrete)
            A = self.endowment_insurance(x, s=s, t=n, discrete=discrete)
        else:   # Special Term or (unequal) Endowment insurance has no shortcut
            n = self.max_term(x+s+t, n)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, discrete=discrete,
                                            b=b, endowment=endowment)
            a1 = self.temporary_annuity(x, s=s+t, t=n-t, b=b, 
                                        discrete=discrete)
            A = self.endowment_insurance(x, s=s, t=n, discrete=discrete,
                                            b=b, endowment=endowment)
            a = self.temporary_annuity(x, s=s, t=n, b=b, discrete=discrete)
            return A1 - a1 * (A / a)
        return self.net_future_loss(A=A, A1=A1, b=b)  # apply shortcut

    #
    # Gross Future Loss shortcuts for WL and Endowment Insurance 
    #
    def gross_future_loss(self, A: float | None = None, 
                          a: float | None = None, 
                          contract: Contract | None = None) -> float:
        """Shortcut for WL or Endowment Insurance gross future loss

        Args:
          A : insurance factor at age (x)
          a : annuity factor at age (x)
          contract : policy contract terms and expenses
        """
        contract = contract or Contract()
        if a is None:    # assume WL or Endowment Insurance for twin annuity
            a = self.annuity_twin(A, discrete=contract.discrete)
        elif A is None:  # assume WL or Endowment Insurance for twin annuity
             A = self.insurance_twin(a, discrete=contract.discrete)
        return ((A * contract.claims_cost + contract.initial_cost) -
                (a * contract.renewal_profit))

    def gross_variance_loss(self, A1: float, A2: float = 0,
                            contract: Contract | None = None) -> float:
        """Shortcuts for variance of gross loss for WL or endowment insurance

        Args:
          A1 : insurance factor
          A2 : insurance factor at double the force of interest
          policy : policy terms and expenses
        """
        contract = contract or Contract()
        interest = self.interest.d if contract.discrete else self.interest.delta
        return (((contract.renewal_profit / interest) + contract.benefit +
                 contract.settlement_policy)**2 * (A2 - A1**2))

    def gross_policy_variance(self, x: int, s: int = 0, t: int = 0,
                              n: int = Premiums.WHOLE,
                              contract: Contract | None = None) -> float:
        """Variance of gross policy value for WL and Endowment Insurance

        Args:
          x : age initially insured
          s : years after selection
          n : number of years of premiums paid
          t : year of death
          contract : policy contract terms and expenses
        """
        contract = contract or Contract()
        if n < 0:  # WL
            A2 = self.whole_life_insurance(x, s=s+t, moment=2,
                                            discrete=contract.discrete)
            A1 = self.whole_life_insurance(x, s=s+t, 
                                            discrete=contract.discrete)
        elif contract.endowment == contract.claims_cost:  # Endowment
            n = self.max_term(x+s+t, n)
            A2 = self.endowment_insurance(x, s=s+t, t=n-t, moment=2, 
                                            discrete=contract.discrete)
            A1 = self.endowment_insurance(x, s=s+t, t=n-t, 
                                            discrete=contract.discrete)
        else:
            raise Exception("Variance for WL or Endowment Ins only")
        return self.gross_variance_loss(A1=A1, A2=A2, contract=contract)

    #
    # Gross Policy Value for special insurance
    #
    def gross_policy_value(self, x: int, s: int = 0, t: int = 0,
                           n: int = Premiums.WHOLE,
                           contract: Contract | None = None) -> float:
        """Gross policy values for insurance: t_V = E[L_t]

        Args:
          x : age initially insured
          s : years after selection
          t : number of years of premiums paid
          n : term of insurance
          contract : policy contract terms
        """
        contract = contract or Contract()
        if n < 0:    # Whole life shortcut
            A = self.whole_life_insurance(x, s=s+t, 
                                          discrete=contract.discrete)
        elif contract.endowment == contract.claims_cost:  # Endowment Ins shortcut
            n = self.max_term(x+s+t, n)
            A = self.endowment_insurance(x, s=s+t, t=n-t, 
                                            discrete=contract.discrete)
        else:  # Special term insurance
            n = self.max_term(x+s+t, n)
            A = self.term_insurance(x, s=s+t, t=n-t, discrete=contract.discrete)
            a = self.temporary_annuity(x, s=s+t, t=n-t,
                                       discrete=contract.discrete)
            endowment = 0
            if contract.endowment:  # endowment not equal to claims cost
                endowment = self.E_x(x, s=s+t, t=n-t) * contract.endowment
            initial_cost = 0 if t else contract.initial_cost
            return (A * contract.claims_cost + initial_cost + endowment -
                    (a * contract.renewal_profit))
        contract = contract.renewals(t) if t else contract # ignore initial if t>0
        return self.gross_future_loss(A=A, contract=contract)

    #
    # Future Loss random variable: L(T_x)
    #
    def L_from_t(self, t: float, contract: Contract | None = None) -> float:
        """PV of Loss L(t) at time of death t = T_x (or K_x if discrete)

        Args:
          t : year of death
          contract : policy contract
        """
        contract = contract or Contract()
        k = math.floor(t) if contract.discrete else t  # if endowment paid
        if contract.T > 0 and k >= contract.T:
            t = contract.T
            endowment = contract.endowment * self.Z_from_t(t)
        else:
            endowment = 0
        return ((contract.claims_cost
                 * self.Z_from_t(t, discrete=contract.discrete))
                + contract.initial_cost
                + endowment
                - (contract.renewal_profit
                   * self.Y_from_t(t, discrete=contract.discrete)))

    def L_to_t(self, L: float, contract: Contract | None = None) -> float:
        """Compute time of death T_x s.t. PV future loss is L

        Args:
          L : PV of future loss
          contract : policy contract terms and expenses
        """
        contract = contract or Contract()
        return Contract.solve(lambda t: self.L_from_t(t, contract),
                              target=L, grid=(0, self._MAXAGE), mad=True)

    def L_from_prob(self, x: int, prob: float, 
                    contract: Contract | None = None) -> float:
        """Compute PV of future loss at given percentile prob

        Args:
          x : age
          prob : probability threshold
          contract : policy contract
        """
        t = self.Z_t(x, prob, discrete=contract.discrete)
        return self.L_from_t(t, contract)

    def L_to_prob(self, x: int, L: float,
                  contract: Contract = Contract()) -> float:
        """Compute percentile of L on the PV of future loss curve"

        Args:
          x : age selected
          L : PV of future loss
          contract : policy contract terms and expenses
        """
        t = self.L_to_t(L, contract)
        return self.S(x, 0, t)

    def L_plot(self, x: int, s: int = 0, stop: int = 0,
               T: float | None = None,
               contract: Contract = Contract(),
               ax: Any = None,
               title: str | None = None,
               color='r') -> float:
        """Plot PV of future loss r.v. L vs time of death T_x
        
        Args:
          x : age selected
          s : years after selection
          stop : time to end plot
          contract : policy contract terms and expenses
          T : point in time to indicate probability and loss values
          title : title of plot 
          color : color to plot curve
        """        
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        K = 'K' if contract.discrete else 'T'
        stop = stop or self._MAXAGE - (x + s)
        step = 1 if contract.discrete else stop / 1000.
        steps = np.arange(0, stop + step, step)

        # plot PV loss values
        y = [self.L_from_t(t, contract=contract) for t in steps]
        ax.plot(steps, y, '.', c=color)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        yjig = (ymax - ymin) / 50
        xjig = (xmax - xmin) / 50
        if T is None:
            ax.plot(steps, y, ".", color=color)
            ax.hlines(0, xmin, xmax, colors='g', linestyles=':')
            z = None
        else:
            # plot indicate(T*)
            z = self.L_from_t(T, contract=contract)
            ax.plot(T, z, c=color, marker='o')
            ax.text(T, z + yjig, f"L*={z:.2f}", c=color)

            # indicate given time of death T*
            ax.vlines(T, ymin, z, colors='g', linestyles=':')
            ax.text(T + xjig, ymin, f"{K}={T:.2f}", c='g')

            # indicate corresponding S(T*)
            p = self.S(x, s, T)
            ax.hlines(z, T, xmax, colors='g', linestyles=':')
            ax.text(xmax, z-yjig, f"Prob={p:.3f}", c='g', va='top', ha='right')
            #ax.set_title(f"Pr[${K}_x$ >= {K}(Z*)] > {p:.3}")
        ax.set_title(f"PV future loss r.v. $L({K}_{{{x if x else 'x'}}})$"
                     if title is None else title)
        ax.set_ylabel(f"$L({K}_x)$", color=color)
        ax.set_xlabel(f"${K}_x$")
        plt.tight_layout()
        return z

if __name__ == "__main__":
    from actuarialmath.sult import SULT
    from actuarialmath.policyvalues import Contract
    life = SULT()
    x = 20
    P = life.net_premium(x=x)
    contract = Contract(premium=P, discrete=True)
    T = life.L_to_t(L=0, contract=contract)  # breakeven T
    print(T)
    life.L_plot(x=x, T=T, contract=contract)

    print("SOA Question 6.24:  (E) 0.30")
    life = PolicyValues().set_interest(delta=0.07)
    x, A1 = 0, 0.30   # Policy for first insurance
    P = life.premium_equivalence(A=A1, discrete=False)  # Need its premium
    contract = Contract(premium=P, discrete=False)
    def fun(A2):  # Solve for A2, given Var(Loss)
        return life.gross_variance_loss(A1=A1, A2=A2, contract=contract)
    A2 = life.solve(fun, target=0.18, grid=0.18)
    contract = Contract(premium=0.06, discrete=False)     # Solve second insurance
    variance = life.gross_variance_loss(A1=A1, A2=A2, contract=contract)
    print(variance)
    print()

    print("SOA Question 6.30:  (A) 900")
    life = PolicyValues().set_interest(i=0.04)
    contract = Contract(premium=2.338, benefit=100, initial_premium=.1, renewal_premium=0.05)
    var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
                                   A2=0.17, contract=contract)
    print(var)
    print()

    print("SOA Question 7.32:  (B) 1.4")
    life = PolicyValues().set_interest(i=0.06)
    contract = Contract(benefit=1, premium=0.1)
    def fun(A2):
        return life.gross_variance_loss(A1=0, A2=A2, contract=contract)
    A2 = life.solve(fun, target=0.455, grid=0.455)
    contract = Contract(benefit=2, premium=0.16)
    var = life.gross_variance_loss(A1=0, A2=A2, contract=contract)
    print(var)
    print()

    print("SOA Question 6.12:  (E) 88900")
    life = PolicyValues().set_interest(i=0.06)
    a = 12
    A = life.insurance_twin(a)
    contract = Contract(benefit=1000, settlement_policy=20, 
                        initial_policy=10, initial_premium=0.75, 
                        renewal_policy=2, renewal_premium=0.1)
    contract.premium = life.gross_premium(A=A, a=a, **contract.premium_terms)
    print(A, contract.premium)
    L = life.gross_variance_loss(A1=A, A2=0.14, contract=contract)
    print(L)
    print()

    from actuarialmath.sult import SULT
    print("SOA Question 6.6:  (B) 0.79")
    life = SULT()
    P = life.net_premium(62, b=10000)
    contract = Contract(premium=1.03*P, renewal_policy=5,
                        initial_policy=5, initial_premium=0.05, benefit=10000)
    L = life.gross_policy_value(62, contract=contract)
    var = life.gross_policy_variance(62, contract=contract)
    prob = life.portfolio_cdf(mean=L, variance=var, value=40000, N=600)
    print(prob, 0.79)
    life.L_plot(62, contract=contract)
    print()

    print("SOA Question 7.6:  (E) -25.4")
    from actuarialmath.sult import SULT
    life = SULT()
    P = life.net_premium(45, b=2000)
    contract = Contract(benefit=2000, initial_premium=.25, renewal_premium=.05,
                        initial_policy=2*1.5 + 30, renewal_policy=2*.5 + 10)
    G = life.gross_premium(a=life.whole_life_annuity(45),
                           **contract.premium_terms)
    gross = life.gross_policy_value(45, t=10,
                                    contract=contract.set_contract(premium=G))
    net = life.net_policy_value(45, t=10, b=2000)
    V = gross - net
    print(V, -25.4)
    T = life.L_to_t(G, contract=contract)
    print(G)

    
