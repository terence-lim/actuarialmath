"""SULT - Loads and uses a standard ultimate life table

MIT License. Copyright 2022-2023 Terence Lim
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, Callable
from actuarialmath import LifeTable

# Makeham's Law parameters from SOA’s Excel Workbook for FAM-L Tables
_A, _B, _c = 0.00022, 0.0000027, 1.124
def _faml_sult(x, t: float) -> float:
    return math.exp(-_A*t - (_B*_c**x*(_c**t - 1)) / math.log(_c))

class SULT(LifeTable):
    """Generates and uses a standard ultimate life table

    Args:
      i : interest rate
      radix : initial number of lives
      minage : minimum age
      maxage : maximum age
      S : survival function, default is Makeham with SOA FAM-L parameters

    Examples:
      >>> sult = SULT()
      >>> a = sult.temporary_annuity(70, t=10)
      >>> A = sult.deferred_annuity(70, u=10)
      >>> P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
      >>>                        renewal_premium=0.05)
    """


    def __init__(self, i: float = 0.05, radix: int = 100000, 
                 S: Callable[[float, float], float] = _faml_sult,
                 minage: int = 20, maxage: int = 130, **kwargs):
        """Construct SULT"""
        super().__init__(**kwargs)
        l = {t+minage: radix * S(minage, t) for t in range(1+maxage-minage)}
        self.set_interest(i=i).set_table(l=l, minage=minage, maxage=maxage)

    def __getitem__(self, col: str) -> Dict[int, float]:
        """Returns a column of the sult table

        Args:
          col : name of life table column to return
        """
        funs = {'q': (self.q_x, dict()),
                'p': (self.p_x, dict()),                
                'a': (self.whole_life_annuity, dict()),
                'A': (self.whole_life_insurance, dict())}
        assert col[0].lower() in funs, f"must be one of {list(funs.keys())}"
        f, args = funs[col[0]]
        return {x: f(x, **args) for x in range(self._MINAGE, self._MAXAGE)}

    def frame(self, minage: int = 20, maxage: int = 100):
        """Derive FAM-L exam table columns of SULT as a DataFrame

        Args:
          minage : first age to display row
          maxage : large age to display row
        """
        # specify methods and arguments for computing columns of FAM-L exam table
        funs = {'q_x': (self.q_x, dict()),
                'a_x': (self.whole_life_annuity, dict()),
                'A_x': (self.whole_life_insurance, dict()),
                '2A_x': (self.whole_life_insurance, dict(moment=2)),
                'a_x:10': (self.temporary_annuity, dict(t=10)),
                'A_x:10': (self.endowment_insurance, dict(t=10)),
                'a_x:20': (self.temporary_annuity, dict(t=20)),
                'A_x:20': (self.endowment_insurance, dict(t=20)),
                '5_E_x': (self.E_x, dict(t=5)),
                '10_E_x': (self.E_x, dict(t=10)),
                '20_E_x': (self.E_x, dict(t=20))}
        t = {col: {x: f(x, **args) for x in range(self._MINAGE, self._MAXAGE)}
             for col, (f, args) in funs.items()}
        tab = pd.DataFrame(dict(l_x=self._table['l'])).sort_index()
        tab = tab.join(pd.DataFrame.from_dict(t).set_index(tab.index[:-1]))
        for digits, col in zip([1, 6, 4, 5, 5, 4, 5, 4, 5, 5, 5, 5], tab.columns):
             tab[col] = tab[col].map(f"{{:.{digits}f}}".format)
        return tab.loc[minage:maxage]

if __name__ == "__main__":
    print("SOA Question 6.52:  (D) 50.80")
    sult = SULT()
    a = sult.temporary_annuity(45, t=10)
    other_cost = 10 * sult.deferred_annuity(45, u=10)
    P = sult.gross_premium(a=a, A=0, benefit=0, 
                           initial_premium=1.05, renewal_premium=0.05,
                           initial_policy=100 + other_cost, renewal_policy=20)
    print(a, P)
    print()
    

    print("SOA Question 6.47:  (D) 66400")
    sult = SULT()
    a = sult.temporary_annuity(70, t=10)
    A = sult.deferred_annuity(70, u=10)
    P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
                           renewal_premium=0.05)
    print(P)
    print()
    

    print("SOA Question 6.43:  (C) 170")
    sult = SULT()
    a = sult.temporary_annuity(30, t=5)
    A = sult.term_insurance(30, t=10)
    other_expenses = 4 * sult.deferred_annuity(30, u=5, t=5)
    P = sult.gross_premium(a=a, A=A, benefit=200000, initial_premium=0.35,
                           initial_policy=8 + other_expenses, renewal_policy=4,
                           renewal_premium=0.15)
    print(P)
    print()

    print("SOA Question 6.39:  (A) 29")
    sult = SULT()
    P40 = sult.premium_equivalence(sult.whole_life_insurance(40), b=1000)
    P80 = sult.premium_equivalence(sult.whole_life_insurance(80), b=1000)
    p40 = sult.p_x(40, t=10)
    p80 = sult.p_x(80, t=10)
    P = (P40 * p40 + P80 * p80) / (p80 + p40)
    print(P)
    print()
    

    print("SOA Question 6.37:  (D) 820")
    sult = SULT()
    benefits = sult.whole_life_insurance(35, b=50000 + 100)
    expenses = sult.immediate_annuity(35, b=100)
    a = sult.temporary_annuity(35, t=10)
    print(benefits, expenses, a)
    print((benefits + expenses) / a)
    print()
    

    print("SOA Question 6.35:  (D) 530")
    sult = SULT()
    A = sult.whole_life_insurance(35, b=100000)
    a = sult.whole_life_annuity(35)
    print(sult.gross_premium(a=a, A=A, initial_premium=.19, renewal_premium=.04))
    print()
    

    print("SOA Question 5.8: (C) 0.92118")
    sult = SULT()
    a = sult.certain_life_annuity(55, u=5)
    print(sult.p_x(55, t=math.floor(a)))
    print()
    

    print("SOA Question 5.3:  (C) 6.239")
    sult = SULT()
    t = 10.5
    print(t * sult.E_r(40, t=t))
    print()
    

    print("SOA Question 4.17:  (A) 1126.7")
    sult = SULT()
    median = sult.Z_t(48, prob=0.5, discrete=False)
    benefit = lambda x,t: 5000 if t < median else 10000
    print(sult.A_x(48, benefit=benefit))
    print()
    

    print("SOA Question 4.14:  (E) 390000    ")
    sult = SULT()
    p = sult.p_x(60, t=85-60)
    mean = sult.bernoulli(p)
    var = sult.bernoulli(p, variance=True)
    F = sult.portfolio_percentile(mean=mean, variance=var, prob=.86, N=400)
    print(F * 5000 * sult.interest.v_t(85-60))
    print()
    
    from actuarialmath.interest import Interest
    print("SOA Question 4.5:  (C) 35200")
    sult = SULT(udd=True).set_interest(delta=0.05)
    Z = 100000 * sult.Z_from_prob(45, prob=0.95, discrete=False)
    print(Z)

    print("SOA Question 3.9:  (E) 3850")
    sult = SULT()
    p1 = sult.p_x(20, t=25)
    p2 = sult.p_x(45, t=25)
    mean = sult.bernoulli(p1) * 2000 + sult.bernoulli(p2) * 2000
    var = (sult.bernoulli(p1, variance=True) * 2000 
           + sult.bernoulli(p2, variance=True) * 2000)
    print(sult.portfolio_percentile(mean=mean, variance=var, prob=.99))
    print()
    

    print("SOA Question 3.8:  (B) 1505")
    sult = SULT()
    p1 = sult.p_x(35, t=40)
    p2 = sult.p_x(45, t=40)
    mean = sult.bernoulli(p1) * 1000 + sult.bernoulli(p2) * 1000
    var = (sult.bernoulli(p1, variance=True) * 1000 
           + sult.bernoulli(p2, variance=True) * 1000)
    print(sult.portfolio_percentile(mean=mean, variance=var, prob=.95))
    print()
    

    print("SOA Question 3.4:  (B) 815")
    sult = SULT()
    mean = sult.p_x(25, t=95-25)
    var = sult.bernoulli(mean, variance=True)
    print(sult.portfolio_percentile(N=4000, mean=mean, variance=var, prob=.1))
    print()
    
    print("Standard Ultimate Life Table at i=0.05")
    print(sult.frame())
    print()
