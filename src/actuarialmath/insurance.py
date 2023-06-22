"""Life insurance - Computes present values of insurance benefits

MIT License. Copyright (c) 2022-2023 Terence Lim
"""
from typing import Callable, Any, Tuple, List
import math
import matplotlib.pyplot as plt
import numpy as np
from actuarialmath import Fractional

class Insurance(Fractional):
    """Compute expected present values of life insurance


    Examples:
      >>> life = Insurance().set_interest(delta=0.06).set_survival(mu=lambda *x: 0.04)
      >>> life.whole_life_insurance(x=0)
      >>> life.term_insurance(x=0, t=30)
      >>> life.deferred_insurance(x=0, u=10, t=20)
      >>> life.endowment_insurance(x=0, t=10)
      >>> life.increasing_insurance(x=0, t=10)
      >>> life.decreasing_insurance(x=0, t=10)
      >>> prob, x, discrete = 0.8, 20, True
      >>> t = life.Z_t(x, prob, discrete=discrete)
      >>> Z = life.Z_from_prob(x, prob=prob, discrete=discrete)
      >>> print(t, life.Z_to_t(Z))
      >>> print(Z, life.Z_from_t(t, discrete=discrete))
      >>> print(prob, life.Z_to_prob(x, Z=Z))
      >>> life.Z_plot(x, T=t, discrete=discrete)
    """

    def E_x(self, x: int, s: int = 0, t: int = 1, endowment: int = 1,
            moment: int = 1) -> float:
        """Pure endowment: t_E_x

        Args:
          x : age of selection
          s : years after selection
          t : term of pure endowment
          endowment : amount of pure endowment
          moment : compute first or second moment

        Examples:
          >>> life = Insurance().set_survival(mu=lambda x,t: 0.02*t).set_interest(i=0.03)
          >>> var = life.E_x(0, t=15, moment=life.VARIANCE, endowment=10000)
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
        """Numerically compute EPV of insurance from basic survival functions

        Args:
          x : age of selection
          s : years after selection
          u : year deferred
          t : term of insurance
          benefit : benefit as a function of age and year
          endowment : amount of endowment for endowment insurance
          moment : compute first or second moment
          discrete : benefit paid yearend (True) or moment of death (False)

        Examples:
          >>> life = Insurance().set_interest(delta=0.05).set_survival(mu=lambda x,s: 0.03)
          >>> benefit = lambda x,t: math.exp(0.04 * t)
          >>> A = life.A_x(0, benefit=benefit)
          >>> print(A)   # 0.75
          >>> A2 = life.A_x(0, moment=2, benefit=benefit)
          >>> print(A2)  #0.60
        """
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
            A = self.integral(Z, 0, t)
        return A + E

    @staticmethod
    def insurance_variance(A2: float, A1: float, b: float = 1) -> float:
        """Compute variance of insurance given moments and benefit

        Args:
          A2 : second moment of insurance r.v.
          A1 : first moment of insurance r.v.
          b : benefit amount
        """
        return b**2 * max(0, A2 - A1**2)

    def whole_life_insurance(self, x: int, s: int = 0, moment: int = 1, 
                             b: int = 1, discrete: bool = True) -> float:
        """Whole life insurance: A_x

        Args:
          x : age of selection
          s : years after selection
          b : amount of benefit
          moment : compute first or second moment
          discrete : benefit paid year-end (True) or moment of death (False)

        Examples:
          >>> life.whole_life_insurance(x=0)
        """
        if moment == self.VARIANCE:
            A2 = self.whole_life_insurance(x, s=s, moment=2, discrete=discrete)
            A1 = self.whole_life_insurance(x, s=s, discrete=discrete)**2
            return self.insurance_variance(A2=A2, A1=A1, b=b)
        return self.A_x(x, s=s, t=self.WHOLE, benefit=lambda x,t: b, 
                        moment=moment, discrete=discrete)

    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                       moment: int = 1, discrete: bool = True) -> float:
        """Term life insurance: A_x:t^1

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit
          moment : compute first or second moment
          discrete : benefit paid year-end (True) or moment of death (False)

        Examples:
          >>> life.term_insurance(x=0, t=30)
        """
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
        """Deferred insurance n|_A_x:t^1 = discounted term or whole life

        Args:
          x : age of selection
          s : years after selection
          u : year deferred
          t : term of insurance
          b : amount of benefit
          moment : compute first or second moment
          discrete : benefit paid year-end (True) or moment of death (False)

        Examples:
          >>> life.deferred_insurance(x=0, u=10, t=20)
        """
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
        """Endowment insurance: A_x^1:t = term insurance + pure endowment

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit
          endowment : amount of endowment paid at end of term if survive
          moment : compute first or second moment
          discrete : benefit paid year-end (True) or moment of death (False)

        Examples:
          >>> life.endowment_insurance(x=0, t=10)
        """
        if moment == self.VARIANCE:
            A2 = self.endowment_insurance(x, s=s, t=t, endowment=endowment, 
                                          b=b, moment=2, discrete=discrete)
            A1 = self.endowment_insurance(x, s=s, t=t, endowment=endowment,
                                          b=b, discrete=discrete)
            return self.insurance_variance(A2=A2, A1=A1, b=b)
        E = self.E_x(x, s=s, t=t, moment=moment)
        A = self.term_insurance(x, s=s, t=t, b=b, moment=moment,
                                discrete=discrete)
        return A + E * (b if endowment < 0 else endowment)**moment

    def increasing_insurance(self, x: int, s: int = 0, t: int =
                             Fractional.WHOLE, 
                             b: int = 1, discrete: bool = True) -> float:
        """Increasing life insurance: (IA)_x

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit in first year
          discrete : benefit paid year-end (True) or moment of death (False)

        Examples:
          >>> life.increasing_insurance(x=0, t=10)
        """
        return self.A_x(x, s=s, t=t, benefit=lambda x,t: t * b, 
                        discrete=discrete)

    def decreasing_insurance(self, x, s: int = 0, t: int = 1, b: int = 1,
                             discrete: bool = True) -> float:
        """Decreasing life insurance: (DA)_x

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit in first year
          discrete : benefit paid year-end (True) or moment of death (False)

        Examples:
          >>> life.decreasing_insurance(x=0, t=10)
        """
        assert t > 0  # decreasing must be term insurance
        A = self.term_insurance(x, t=t, b=b, discrete=discrete)
        n = t + int(discrete)   #  (DA)_x:n + (IA)_x:n = (n+1) A^1_x:n
        return A*n - self.increasing_insurance(x, s=s, t=t, b=b, 
                                               discrete=discrete)

    #
    # Insurance random variable: Z(t)
    #
    def Z_t(self, x: int, prob: float, discrete: bool = True) -> float:
        """T_x given percentile of the PV of WL or Term insurance, i.e. r.v. Z(t)

        Args:
          x : age initially insured
          prob : desired probability threshold
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        assert prob < 1.0
        t = self.solve(lambda t: self.S(x, 0, t), target=prob, grid=50)
        return math.floor(t) if discrete else t    # opposite of annuity

    def Z_from_t(self, t: float, discrete: bool = True) -> float:
        """PV of insurance payment Z(t), given T_x (or K_x if discrete)

        Args:
          t : year of death
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        return self.interest.v_t((math.floor(t) + 1) if discrete else t)

    def Z_to_t(self, Z: float) -> float:
        """T_x s.t. PV of insurance payment is Z

        Args:
          Z : Present value of benefit paid
        """
        #t = Insurance.solve(lambda t: self.Z_from_t(t), Z, self._MAXAGE/2)
        t = math.log(Z) / math.log(self.interest.v)
        return t

    def Z_from_prob(self, x: int, prob: float, discrete: bool = True) -> float:
        """Percentile of insurance PV r.v. Z, given probability

        Args:
          x : age initially insured
          prob : threshold for probability of survival
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        t = self.Z_t(45, prob, discrete=discrete)   # opposite of annuity!
        return self.Z_from_t(t, discrete=discrete)  # z is WL or Term Insurance

    def Z_to_prob(self, x: int, Z: float) -> float:
        """Cumulative density of insurance PV r.v. Z, given percentile value

        Args:
          x : age initially insured
          Z : present value of benefit paid
        """
        t = self.Z_to_t(Z) 
        return self.S(x, 0, t)      # z is WL or Term Insurance

    def Z_x(self, x, s: int = 0, t: int = 1, discrete: bool = True):
        """EPV of year t insurance death benefit for life aged [x]+s: b_x[s]+s(t)

        Args:
          x : age of selection
          s : years after selection
          t : year of benefit
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        assert t > 0
        if discrete:
            u = math.ceil(t) - 1
            return self.q_x(x, s=s, u=u) * self.interest.v_t(u+1)
        else:
            return self.f_r(x, s=s, t=t) * self.interest.v_t(t)

    def Z_plot(self, x: int, s: int = 0, stop : int = 0,
               benefit: Callable = lambda x,k: 1,
               T: float | None = None,
               discrete: bool = True,
               ax: Any = None,
               title: str | None = None,
               color: str ='r')-> float | None:
        """Plot of PV of insurance r.v. Z vs t

        Args:
          x : age of selection
          s : years after selection
          stop : time to end plot
          benefit : benefit as a function of selection age and time
          discrete  discrete or continuous insurance
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
        
        # plot PV benefit values
        z = [benefit(x, s+t) * self.Z_from_t(t, discrete=discrete) for t in steps]
        if T is None:
            ax.bar(steps, z, width=step, alpha=0.5, color=color)
            Z = None
        else:
            # plot Z(t)
            ax.step(steps, z, ':', c=color, where='pre' if discrete else 'post')
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            yjig = (ymax - ymin) / 50
            xjig = (xmax - xmin) / 50
            
            # indicate PV of benefit Z(T*)
            Z = self.Z_from_t(T, discrete=discrete) * benefit(self._MINAGE, T)
            ax.plot(T, Z, c=color, marker='o')
            ax.text(T, Z + yjig, f"Z*={Z:.2f}", c=color)

            # indicate given time of death T*
            ax.vlines(T, ymin, Z, colors='g', linestyles=':')
            ax.text(T + xjig, ymin, f"${K}_x$={T:.2f}", c='g')

            # indicate corresponding S(T*)
            p = self.S(x, 0, T)     # S(t): note that is opposite of annuity
            ax.hlines(Z, T, xmax, colors='g', linestyles=':')
            ax.text(xmax, Z - yjig, f"Prob={p:.3f}", c='g', va='top', ha='right')
        ax.set_title(f"PV insurance r.v. $Z({K}_{{{x if x else 'x'}}})$"
                     if title is None else title)
        ax.set_ylabel(f"$Z({K}_x)$", color=color)
        ax.set_xlabel(f"${K}_x$")
        plt.tight_layout()
        return Z

    def Z_curve(self, x: int, s: int = 0, stop: int = 0,
                benefit: Callable = lambda x, k: 1,
                T: float | None = None,
                discrete: bool = True,
                title: str | None = None,
                ax: Any = None):
        """Plot PV of insurance r.v. Z(t) and survival probability vs time t

        Args:
          x : age of selection
          s : years after selection
          stop : end at time t, inclusive
          benefit : benefit as a function of selection age and time
          discrete : discrete or continuous insurance
          T : point in time to indicate benefit and survival probability
          ax : figure object to plot in
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)        
        K = 'K' if discrete else 'T'
        stop = stop or self._MAXAGE - (x + s)
        step = 1 # if discrete else stop/1000.
        steps = range(stop + step)
        p = [self.p_x(x=x, s=s, t=t) for t in steps]

        # plot survival probabilities in secondary axis
        bx = ax.twinx()
        bx.bar(steps, p, color='g', alpha=.2, width=step, align='edge')
        bx.set_ylabel(f"$S({K})$", color='g')
        bx.tick_params(axis='y', colors='g')

        # plot benefit values in primary axis
        z = [benefit(x, s+t) * self.Z_from_t(t, discrete=discrete) for t in steps]
        ax.step(steps, z, ':', c='r', where='pre' if discrete else 'post')
        ax.set_ylabel(f"Z({K})", color='r')
        ax.tick_params(axis='y', colors='r')

        if T is not None:   # plot PV benefit value and survival prob at given T
            Z = self.Z_from_t(T, discrete=discrete) * benefit(self._MINAGE, T)
            label1, = ax.plot(T, Z, c='r', marker='o',
                              label=f"Z({K}*={T:.2f}): {Z:.2f}")
            ax.legend(handles=[label1], loc='center left')
            
            prob = self.S(x, 0, T)      # note: is opposite of annuity
            label2, = bx.plot(T, prob, c='g', marker='o',
                               label=f"Pr[{K}>{K}*]<={prob:.3f}")
            bx.legend(handles=[label2], loc='center right')
        ax.set_title(title if title is not None else
                     f"PV benefit $Z({K})$ and survival probability $S({K})$")
        ax.set_xlabel(f"${K}$")
        plt.tight_layout()

    
if __name__ == "__main__":

    from actuarialmath.sult import SULT
    life = SULT()
    life.Z_curve(x=20, stop=80, T=life.Z_t(x=20, prob=0.5),
                 title="PV insurance benefit if (x) survives median lifetime")
    
    print("SOA Question 4.18  (A) 81873 ")
    def f(x,s,t): return 0.1 if t < 2 else 0.4*t**(-2)
    life = Insurance().set_interest(delta=0.05)\
                      .set_survival(f=f, maxage=10)
    def benefit(x,t): return 0 if t < 2 else 100000
    prob = 0.9 - life.q_x(x=0, t=2)
    T = life.Z_t(x=0, prob=prob)
    life.Z_curve(x=0, T=T, benefit=benefit, discrete=False)
    Z = life.Z_from_t(T) * benefit(0, T)
    print(Z)
    
    # Example: plot Z vs T
    life = Insurance().set_interest(delta=0.06)\
                      .set_survival(mu=lambda *x: 0.04)
    prob = 0.8
    x = 20
    discrete = True
    t = life.Z_t(x, prob, discrete=discrete)
    Z = life.Z_from_prob(x, prob=prob, discrete=discrete)
    print(t, life.Z_to_t(Z))
    print(Z, life.Z_from_t(t, discrete=discrete))
    print(prob, life.Z_to_prob(x, Z=Z))
    life.Z_plot(x, T=t, discrete=discrete)
    plt.show()
    

    print("SOA Question 4.10:  (D)")
    x = 0
    life = Insurance().set_interest(i=0.0)\
                      .set_survival(S=lambda x,s,t: 1, maxage=x+40)
    def expected(x, t):  # true E[Z]
        if 10 <= t <= 20: return life.interest.v_t(t)
        elif 20 < t <= 30: return 2 * life.interest.v_t(t)
        else: return 0
        
    def A(x, t):  # Z_x+k (t-k)
        return life.interest.v_t(t - x) * (t > x)
    
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
    z = [sum(abs(b(x, t) - expected(x, t)) for t in range(40)) for b in benefits]
    print("ABCDE"[np.argmin(z)])
    
    fig, ax = plt.subplots(3, 2)
    ax = ax.ravel()
    for i, b in enumerate([expected] + benefits):
        life.Z_plot(x, benefit=b, ax=ax[i], color=f"C{i+1}", title=' ')
        ax[i].legend(["(" + "abcde"[i-1] + ")" if i else "Z"])        

    print("SOA Question 6.33:  (B) 0.13")
    life = Insurance().set_survival(mu=lambda x,t: 0.02*t).set_interest(i=0.03)
    x = 0
    var = life.E_x(x, t=15, moment=life.VARIANCE, endowment=10000)
    p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
    print(p)


    print("SOA Question 4.12:  (C) 167")
    cov = Insurance.covariance(a=1.65, b=10.75, ab=0) # Z1 and Z2 nonoverlapping
    var = Insurance.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov)
    print(var)

    print("SOA Question 4.11:  (A) 143385")
    A1 = 528/1000   # E[Z1]  term insurance
    C1 = 0.209      # E[pure_endowment]
    C2 = 0.136      # E[pure_endowment^2]
    def fun(A2):
        B1 = A1 + C1   # endowment = term + pure_endowment
        B2 = A2 + C2   # double force of interest
        return Insurance.insurance_variance(A2=B2, A1=B1)
    A2 = Insurance.solve(fun, target=15000/(1000*1000), grid=[143400, 279300])
    var = Insurance.insurance_variance(A2=A2, A1=A1, b=1000)
    print(var)

    print("SOA Question 4.15  (E) 0.0833 ")
    life = Insurance().set_survival(mu=lambda *x: 0.04)\
                      .set_interest(delta=0.06)
    benefit = lambda x,t: math.exp(0.02*t)
    A1 = life.A_x(0, benefit=benefit, discrete=False)
    A2 = life.A_x(0, moment=2, benefit=benefit, discrete=False)
    var = A2 - A1**2
    print(var)

    print("SOA Question 4.4  (A) 0.036")
    x = 40
    life = Insurance().set_survival(f=lambda *x: 0.025, maxage=x+40)\
                      .set_interest(v_t=lambda t: (1 + .2*t)**(-2))
    benefit = lambda x,t: 1 + .2 * t
    A1 = life.A_x(x, benefit=benefit, discrete=False)
    A2 = life.A_x(x, moment=2, benefit=benefit, discrete=False)
    var = A2 - A1**2
    print(var)

    print("Other examples of usage")
    life = Insurance().set_interest(delta=0.06)\
                      .set_survival(mu=lambda *x: 0.04)
    benefit = lambda x,t: math.exp(0.02 * t)
    A1 = life.A_x(0, benefit=benefit)
    A2 = life.A_x(0, moment=2, benefit=benefit)
    var = A2 - A1**2 
    print(var)  # 0.0833

    life = Insurance().set_interest(delta=0.05)\
                      .set_survival(mu=lambda x,s: 0.03)
    benefit = lambda x,t: math.exp(0.04 * t)
    A = life.A_x(0, benefit=benefit)
    print(A)   # 0.75
    A2 = life.A_x(0, moment=2, benefit=benefit)
    print(A2)  #0.60
