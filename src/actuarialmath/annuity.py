"""Life annuities - Computes present values of annuity benefits

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
from typing import Callable, Any
import math
import matplotlib.pyplot as plt
import numpy as np
from actuarialmath.insurance import Insurance

class Annuity(Insurance):
    """Compute present values and relationships of life annuities"""

    def a_x(self, x: int, s: int = 0, t: int = Insurance.WHOLE, u: int = 0,
            benefit: Callable = lambda x,t: 1, discrete: bool = True) -> float:
        """Numerically compute EPV of annuities from survival functions

        Args:
          x : age of selection
          s : years after selection
          u : year deferred
          t : term of insurance
          benefit : benefit as a function of age and year
          discrete : annuity due (True) or continuous (False)
        """
        t = self.max_term(x+s+u, t=t)
        if discrete:
            a = sum([benefit(x+s, k) * self.interest.v_t(k) 
                     * self.p_x(x, s=s, t=k) for k in range(u, t+u)])
        else:   # use continous first principles
            Y = lambda t: (benefit(x+s, t+u) * self.interest.v_t(t+u) 
                           * self.S(x, 0, t=t+u))
            a = self.integral(Y, 0, t)
        return a


    def immediate_annuity(self, x: int, s: int = 0, t: int = Insurance.WHOLE, 
                          b: int = 1, variance=False) -> float:
        """Compute EPV of immediate life annuity

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : benefit amount
          variance : return EPV (False) or variance (True)
        """
        if variance:
            return self.temporary_annuity(x, s=s, t=self.add_term(t, 1), b=b,
                                          discrete=True, variance=True)
        return (self.temporary_annuity(x, s=s, t=t, discrete=True) 
                - 1 + self.E_x(x, s=s, t=t)) * b
    
    def annuity_twin(self, A: float, discrete: bool = True) -> float:
        """Returns annuity from its WL or Endowment Insurance twin"

        Args:
          A : cost of insurance
          discrete : discrete/annuity due (True) or continous (False)
        """
        interest = (self.interest.d if discrete else self.interest.delta)
        return ((1-A) / interest) if interest else 0 # undefined for 0 interest

    def insurance_twin(self, a: float, moment: int = 1, 
                       discrete: bool = True) -> float:
        """Returns WL or Endowment Insurance twin from annuity

        Args:
          a : cost of annuity
          discrete : discrete/annuity due (True) or continous (False)
        """
        assert moment in [1]
        return 1 - a*(self.interest.d if discrete else self.interest.delta)
  
    def annuity_variance(self, A2: float, A1: float, b: float = 1.,
                         discrete: bool = True) -> float:
        """Compute variance from WL or endowment insurance twin

        Args:
          A2 : second moment of insurance factor
          A1 : first moment of insurance factor
          b : annuity benefit amount
          discrete : discrete/annuity due (True) or continous (False)
        """
        return (b**2 * self.insurance_variance(A2=A2, A1=A1)
                / (self.interest.d if discrete else self.interest.delta)**2)
        
    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1, 
                           variance: bool = False, 
                           discrete: bool = True) -> float:
        """Whole life annuity: a_x

        Args:
          x : age of selection
          s : years after selection
          b : annuity benefit amount
          variance : return EPV (True) or variance (False)
          discrete : annuity due (True) or continuous (False)
        """
        interest = self.interest.d if discrete else self.interest.delta
        if variance:  # short cut for variance of whole life
            A1 = self.whole_life_insurance(x, s=s, moment=1, discrete=discrete)
            A2 = self.whole_life_insurance(x, s=s, moment=2, discrete=discrete)
            return self.annuity_variance(A2=A2, A1=A1, discrete=discrete, b=b)
        A = self.whole_life_insurance(x, s=s, discrete=discrete)
        return b * (1 - A) / interest
            
    def temporary_annuity(self, x: int, s: int = 0, t: int = Insurance.WHOLE, 
                          b: int = 1, variance: bool = False, 
                          discrete: bool = True) -> float:
        """Temporary life annuity: a_x:t

        Args:
          x : age of selection
          s : years after selection
          t : term of annuity in years
          b (int) : annuity benefit amount
          variance : return EPV (True) or variance (False)
          discrete : annuity due (True) or continuous (False)
        """
        if variance:  # short cut for variance of temporary life annuity
            A1 = self.endowment_insurance(x, s=s, t=t, discrete=discrete)
            A2 = self.endowment_insurance(x, s=s, t=t, moment=2, 
                                          discrete=discrete)
            return self.annuity_variance(A2=A2, A1=A1, discrete=discrete, b=b)

        # difference of whole life on (x) and deferred whole life on (x+t)
        a = self.whole_life_annuity(x, s=s, b=b, discrete=discrete)
        if t < 0 or self.max_term(x+s, t) < t:
            return a
        a_t = self.whole_life_annuity(x, s=s+t, b=b, discrete=discrete)
        return a - (a_t * self.E_x(x, s=s, t=t))

    def deferred_annuity(self, x: int, s: int = 0, u: int = 0, 
                         t: int = Insurance.WHOLE, b: int = 1, 
                         discrete: bool = True) -> float:
        """Deferred life annuity n|t_a_x =  n+t_a_x - n_a_x

        Args:
          x : age of selection
          s : years after selection
          u : years deferred
          t : term of annuity in years
          b : annuity benefit amount
          discrete : annuity due (True) or continuous (False)
        """
        a = self.temporary_annuity(x, s=s+u, t=t, b=b, discrete=discrete)
        return self.E_x(x, s=s, t=u) * a

    def certain_life_annuity(self, x: int, s: int = 0, u: int = 0, 
                             t: int = Insurance.WHOLE, b: int = 1, 
                             discrete: bool = True) -> float:
        """Certain and life annuity = certain + deferred

        Args:
          x : age of selection
          s : years after selection
          u : years of certain annuity
          t : term of life annuity in years
          b : annuity benefit amount
          discrete : annuity due (True) or continuous (False)
        """
        u = self.max_term(x+s, u)
        if u < 0:
            return 0.
        a = self.deferred_annuity(x, s=s, u=u, t=t, b=b, discrete=discrete)
        return self.interest.annuity(u, m=int(discrete)) + a

    def increasing_annuity(self, x: int, s: int = 0, t: int = Insurance.WHOLE, 
                           b: int = 1, discrete: bool = True) -> float:
        """Increasing annuity

        Args:
          x : age of selection
          s : years after selection
          t : term of life annuity in years
          b : benefit amount at end of first year
          discrete : annuity due (True) or continuous (False)
        """
        t = self.max_term(x+s, t=t)
        benefit = lambda x, s: b * (s + 1) # increasing benefit
        return self.a_x(x, s=s, benefit=benefit, t=t, discrete=discrete)

    def decreasing_annuity(self, x: int, s: int = 0, t: int = 0, 
                           b: int = 1, discrete: bool = True) -> float:
        """Identity (Da)_x:n + (Ia)_x:n = (n+1) a_x:n temporary annuity

        Args:
          x : age of selection
          s : years after selection
          t : term of life annuity in years
          b : benefit amount at end of first year
          discrete : annuity due (True) or continuous (False)
        """
        assert t >= 0   # decreasing must be till term
        t = self.max_term(x+s, t=t)
        a = self.temporary_annuity(x, s=s, t=t, discrete=discrete)
        n = t + int(discrete)
        return b * (a*n - self.increasing_annuity(x, s=s, t=t, discrete=discrete))

    #
    # Annuity random variable: Y(t)
    #
    def Y_t(self, x: int, prob: float, discrete: bool = True) -> float:
        """T_x given percentile of the r.v. Y = PV of WL or Temporary Annuity

        Args:
          x : age of selection
          prob : desired probability threshold
          discrete : annuity due (True) or continuous (False)
        """
        assert prob < 1.0
        t = Insurance.solve(lambda t: self.S(x, 0, t), target=1-prob, grid=25)
        return math.floor(t) if discrete else t   # opposite of insurance

    def Y_from_t(self, t: float, discrete: bool = True) -> float:
        """PV of insurance payment Y(t), given T_x (or K_x if discrete)

        Args:
          t : year of death
          discrete : annuity due (True) or continuous (False)
        """
        if discrete:
            return (1 - self.interest.v_t(math.floor(t) + 1)) / self.interest.d
        else:
            return (1 - self.interest.v_t(t)) / self.interest.delta

    def Y_to_t(self, Y: float) -> float:
        """T_x  s.t. PV of annuity payments is Y

        Args:
          Y : Present value of benefits paid
        """
        t = math.log(1 - self.interest.delta * Y) / math.log(self.interest.v)
        return t

    def Y_from_prob(self, x: int, prob: float, discrete: bool = True) -> float:
        """Percentile of annuity PV r.v. Y, given probability

        Args:
          x : age initially insured
          prob : desired probability threshold
          discrete : annuity due (True) or continuous (False)
        """
        t = self.Y_t(x, prob, discrete=discrete)
        return self.Y_from_t(t=t, discrete=discrete)

    def Y_to_prob(self, x: int, Y: float) -> float:
        """Cumulative density of insurance PV r.v. Y, given percentile value

        Args:
          x : age initially insured
          Y : present value of benefits paid
        """
        t = self.Y_to_t(Y)
        return 1 - self.S(x, 0, t)   # opposite of Insurance

    def Y_x(self, x, s: int = 0, t: int = 1, discrete: bool = True) -> float:
        """EPV of year t annuity benefit for life aged [x]+s: b_x[s]+s(t)

        Args:
          x : age initially insured
          s : years after selection
          t : year of benefit
          discrete : annuity due (True) or continuous (False)
        """
        assert t >= 0
        if discrete:
            return (self.interest.v_t(math.floor(t)) 
                    * self.p_x(x, s=s, t=math.floor(t)))
        else:
            return (self.interest.v_t(t) * self.p_r(x, s=s, t=t))

    def Y_plot(self, x: int, s: int = 0, stop : int = 0,
               benefit: Callable = lambda x,k: 1,
               T: float | None = None,
               discrete: bool = True,
               ax: Any = None,
               title: str | None = None,               
               color='r') -> float:
        """Plot PV of annuity r.v. Y vs T

        Args:
          x : age of selection
          s : years after selection
          stop : time to end plot
          benefit : benefit as a function of selection age and time
          discrete : discrete or continuous insurance
          ax : figure object to plot in
          color : color to plot
          title : title of plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        K = 'K' if discrete else 'T'
        stop = stop or self._MAXAGE - (x + s)
        step = 1 if discrete else stop / 1000.
        steps = np.arange(0, stop + step, step)

        # plot benefit values
        y = [benefit(x, s+t) * self.Y_from_t(t, discrete=discrete) for t in steps]
        if T is None:
            ax.bar(steps, y, width=stepsize, alpha=0.5, color=color)
            Y = None
        else:
            ax.step(steps, y, ':', c=color, where='pre' if discrete else 'post')

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            yjig = (ymax - ymin) / 50
            xjig = (xmax - xmin) / 50
            
            # indicate PV benefit Y(T*)
            Y = self.Y_from_t(T, discrete=discrete) * benefit(self._MINAGE, T)
            ax.plot(T, Y, c=color, marker='o')
            ax.text(T + xjig, Y + 1*yjig, f"Y*={Y:.2f}", c=color)

            # indicate given T* time of death
            ax.vlines(T, ymin, Y, colors='g', linestyles=':')
            ax.text(T + xjig, ymin, f"{K}={T:.2f}", c='g')

            # indicate corresponding probability S(T*)
            p = 1 - self.S(x, 0, T)     # note this is opposite of insurance
            ax.hlines(Y, T, xmax, colors='g', linestyles=':')
            ax.text(xmax, Y - yjig, f"Prob={p:.3f}", c='g', va='top', ha='right')
        ax.set_title(f"PV annuity r.v. $Y({K}_{{{x if x else 'x'}}})$"
                     if title is None else title)
        ax.set_ylabel(f"$Y({K}_x)$", color=color)
        ax.set_xlabel(f"${K}_x$")
        plt.tight_layout()
        return Y

if __name__ == "__main__":
    from actuarialmath.sult import SULT
    life = SULT()
    life.Y_plot(x=65, T=life.Y_t(x=65, prob=0.5))

    
    life = Annuity().set_interest(delta=0.06)\
                    .set_survival(mu=lambda *x: 0.04)
    prob = 0.8
    x = 0
    discrete = False
    t = life.Y_t(0, prob, discrete=discrete)
    Y = life.Y_from_prob(x, prob=prob, discrete=discrete)
    print(t, life.Y_to_t(Y))
    print(Y, life.Y_from_t(t, discrete=discrete))
    print(prob, life.Y_to_prob(x, Y=Y))
    life.Y_plot(x=0, T=t, discrete=discrete)

    print("SOA Question 5.6:  (D) 1200")
    life = Annuity().set_interest(i=0.05)
    var = life.annuity_variance(A2=0.22, A1=0.45)
    mean = life.annuity_twin(A=0.45)
    print(life.portfolio_percentile(mean=mean, variance=var, prob=.95, N=100))
    print()
    
    print("Other usage")
    mu = 0.04
    delta = 0.06
    life = Annuity().set_interest(delta=delta)\
                    .set_survival(mu=lambda *x: mu)
    print(life.temporary_annuity(50, t=20, b=10000, discrete=False))
    print(life.endowment_insurance(50, t=20, b=10000, discrete=False))
    print(life.E_x(50, t=20))
    print(life.whole_life_annuity(50, b=10000, discrete=False))
    print(life.whole_life_annuity(70, b=10000, discrete=False))

    mu = 0.07
    delta = 0.02
    life = Annuity().set_interest(delta=delta)\
                    .set_survival(mu=lambda *x: mu)
    print(life.whole_life_annuity(0, discrete=False) * 30)   # 333.33
    print(life.temporary_annuity(0, t=10, discrete=False) * 30)  # 197.81
    print(life.interest.annuity(5, m=0))  # 4.7581
    print(life.deferred_annuity(0, u=5, discrete=False)) # 7.0848
    print(life.certain_life_annuity(0, u=5, discrete=False))  # 11.842

    mu = 0.02
    delta = 0.05
    life = Annuity().set_interest(delta=delta)\
                    .set_survival(mu=lambda *x: mu)
    print(life.decreasing_annuity(0, t=5, discrete=False))  # 6.94