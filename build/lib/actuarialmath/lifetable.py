"""Life Tables

Copyright 2022, Terence Lim

MIT License
"""
from typing import Dict, Optional
from actuarialmath.reserves import Reserves
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

class LifeTable(Reserves):
    """LifeTable: life tables

    - minage (int) : minimum age
    - maxage (int) : maximum age
    - udd (bool) : Fractional age UDD (True) or constant force of mortality (False) 
    - l (Dict[int, float]) : lives at start of year x, or
    - d (Dict[int, float]) : deaths in year x, or
    - p (Dict[int, float]) : probabilities that (x) survives one year, or
    - q (Dict[int, float]) : probabilities that (x) dies in one year
    """
    _help = ['fill',  '__getitem__', '__setitem__', 'frame',
             'l_x', 'd_x', 'p_x', 'q_x', 'f_x', 'mu_x', 'e_x', 'E_x']

    def __init__(self, minage: int = -1, maxage: int = -1, 
                 l: Optional[Dict[int, float]] = None,
                 d: Optional[Dict[int, float]] = None,
                 p: Optional[Dict[int, float]] = None, 
                 q: Optional[Dict[int, float]] = None,
                 udd: bool = True, **kwargs):
        l = self.ifelse(l, {})   # lives aged x
        d = self.ifelse(d, {})   # deaths aged x
        p = self.ifelse(p, {})   # probability (x) survives one year
        q = self.ifelse(q, {})   # probability (x) dies within on year
        super().__init__(udd=udd, **kwargs)
        self.set_survival(**self.fractional)  # UDD survival within integer ages

        self._table = {'l':l, 'd':d, 'q':q, 'p':p}  # columns for life table
        for x, p_x in p.items():  # infer and store q given p
            self._table['q'][x] = 1 - p_x

        first_age = True
        for _, row in self._table.items():  # infer max and min ages
            ages = row.keys()
            if ages:
                if first_age:
                    self.MAXAGE = max(ages) + 1
                    self.MINAGE = min(ages)
                    first_age = False
                if maxage < 0 and max(ages) + 1 > self.MAXAGE:
                    self.MAXAGE = max(ages) + 1
                if minage < 0 and min(ages) < self.MINAGE:
                    self.MINAGE = min(ages) 
        if minage >= 0:
            self.MINAGE = minage
        if maxage >= 0:
            self.MAXAGE = maxage

    def __getitem__(self, col: str) -> Dict[int, float]:
        """Return a column of the life table
        - col (str) : name of table column to return
        """
        fn = {'q': (self.q_x, dict()),
              'p': (self.p_x, dict()),
              'a': (self.whole_life_annuity, dict()),
              'A': (self.whole_life_insurance, dict()),
              '2A': (self.whole_life_insurance, dict(moment=2)),
              'a10': (self.temporary_annuity, dict(t=10)),
              'A10': (self.endowment_insurance, dict(t=10)),
              'a20': (self.temporary_annuity, dict(t=20)),
              'A20': (self.endowment_insurance, dict(t=20)),
              '5E': (self.E_x, dict(t=5)),
              '10E': (self.E_x, dict(t=10)),
              '20E': (self.E_x, dict(t=20))}
        return {x: fn[col][0](x, **fn[col][1]) 
                for x in range(self.MINAGE, self.MAXAGE)}

    def __setitem__(self, col: str, row: Dict[int, float]) -> None:
        """Sets a column of the life table
        col (str) : name of table column to set
        row (Dict[int, float]) : values to set in table column
        """
        for age, value in row.items():
            self._table[col][age] = float(value)
            self.MINAGE = age if self.MINAGE < 0 else min(self.MINAGE, age)
            self.MAXAGE = max(self.MAXAGE, age)

    def fill(self, lives: int = Reserves.LIVES, max_iter: int = 4,
             verbose: bool = False) -> "Lifetable":
        """Fill in missing lives and mortality (does not check consistency)
        - lives (int) : initial number of lives
        - max_iter (int) : number of iterations to fill
        - verbose (bool) : level of verbosity
        """

        def q_x(x: int) -> Optional[float]:
            """Try to compute one-year mortality rate for (x): 1_q_x"""
            if x in self._table['q']:
                return self._table['q'][x]
            if x in self._table['d'] and x in self._table['l']:
                return self._table['d'][x] / self._table['l'][x]
            return None

        def p_x(x: int) -> Optional[float]:
            """Try to compute one-year survival for (x): 1_q_x"""
            if x in self._table['p']:
                return self._table['p'][x]
            if x in self._table['q']:
                return 1 - self._table['q'][x]
            return None

        def l_x(x: int) -> Optional[float]:
            """Try to compute number of lives aged x: l_x"""
            if x in self._table['l']:
                return self._table['l'][x]
            if x+1 in self._table['l'] and x in self._table['q']:
                return self._table['l'][x+1] / (1 - self._table['q'][x])
            if x-1 in self._table['l'] and x-1 in self._table['q']:
                return self._table['l'][x-1] * (1 - self._table['q'][x-1])
            if x in self._table['d'] and x in self._table['q']:
                return self._table['d'][x] / self._table['q'][x]
            return None

        def d_x(x: int) -> Optional[float]:
            """Try to compute number of deaths in one year for (x): d_x"""
            if x in self._table['d']:
                return self._table['d'][x]
            if x+1 in self._table['l'] and x in self._table['l']:
                return self._table['l'][x] - self._table['l'][x+1]
            else:
                return None

        # loop a few times to impute life table values
        funs = {'l': l_x, 'd': d_x, 'q': q_x, 'p': p_x}
        updated = 0
        for loop in range(max_iter):
            prev = updated - 1
            while updated != prev:
                prev = updated
                for col, fun in funs.items():
                    for x in range(self.MINAGE, self.MAXAGE + 1):
                        if x not in self._table[col]:
                            value = fun(x)
                            if verbose:
                                print(loop, updated, col, x, value)
                            if value is not None:
                                self._table[col][x] = round(value, 7)
                                updated += 1
            if not self._table['l']: # assume starting number of lives if necc
                self._table['l'][self.MINAGE] = lives
        return self

    def mu_x(self, x: int, s: int = 0, t: int = 0) -> float:
        """Compute mu_x from p_x in life table
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : death within next t years
        """
        return -math.log(max(0.00001, self.p_x(x, s=s+t, t=1)))

    def l_x(self, x: int, s: int = 0) -> float:
        """Lookup l_x from life table
        - x (int) : age of selection
        - s (int) : years after selection
        """
        if x+s in self._table['l']:
            return self._table['l'][x+s]
        else:
            return 0

    def d_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """Compute deaths as lives at x_t divided by lives at x
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : death within next t years
        """
        if x+s+t <= self.MAXAGE:
            return self.l_x(x, s=s) - self.l_x(x, s=s+t)
        else:
            return 0.

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """t_p_x = lives beginning year x+t divided lives beginning year x
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : death within next t years
        """
        denom = self.l_x(x, s=s)
        if denom and x+s+t <= self.MAXAGE:
            return self.l_x(x, s=s+t) / denom
        else:
            return 0.   # in the long term, we are all dead

    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        """Deferred mortality: u|t_q_x = (l[x+u] - l[x+u+t]) / l[x]
        - x (int) : age of selection
        - s (int) : years after selection
        - u (int) : survive u years, then...
        - t (int) : death within next t years
        """
        denom = self.l_x(x, s=s)
        if denom and x+s+t <= self.MAXAGE:
            return self.d_x(x, s=s+u, t=t) / self.l_x(x, s=s)
        else:
            return 1     # the only certainty in life

    def e_x(self, x: int, s: int = 0, n: int = Reserves.WHOLE) -> float:
        """Expected curtate lifetime from sum of lives in table
        - x (int) : age of selection
        - s (int) : years after selection
        - n (int) : future lifetime limited at n years
        """
        # E[K_x] = sum([self.p(x, k+1) for k in range(n)])
        n = min(self.MAXAGE - x, n)
        e = sum([self.l(x+1+s) for s in range(n)]) # since s_p_x = l_x+s / l_x
        return e

    def E_x(self, x: int, s: int = 0, t: int = 1, moment: int = 1) -> float:
        """Pure Endowment from life table and interest rate
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : survives t years
        - moment (int) : return first (1) or second (2) moment or variance (-2)
        """
        if t == 0:
            return 1.
        if t < 0:
            return 0.
        t = self.max_term(x+s, t)
        p = self.l_x(x, s=s+t) / self.l_x(x, s=s)
        if moment == self.VARIANCE:
            return self.interest.v_t(t)**moment * p * (1 - p)
        if moment == 1:
            return self.interest.v_t(t) * p
        # SULT shortcut: t_E_x(moment=2) = t_E_x(moment=1) * v**t
        return self.interest.v_t(t)**(moment-1) * self.E_x(x, s=s, t=t)

    def frame(self) -> pd.DataFrame:
        """Return life table values in a DataFrame
        """
        return pd.DataFrame.from_dict(self._table).sort_index(axis=0)


if __name__ == "__main__":
    print("SOA Question 6.53:  (D) 720")
    x = 0
    life = LifeTable(interest=dict(i=0.08), q={x:.1, x+1:.1, x+2:.1}).fill()
    A = life.term_insurance(x, t=3)
    P = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
    print(A, P)
    print(life.frame())
    print()
    
    print("SOA Question 6.41:  (B) 1417")
    x = 0
    life = LifeTable(interest=dict(i=0.05), q={x:.01, x+1:.02}).fill()
    P = 1416.93
    a = 1 + life.E_x(x, t=1) * 1.01
    A = (life.deferred_insurance(x, u=0, t=1) 
         + 1.01 * life.deferred_insurance(x, u=1, t=1))
    print(a, A)
    P = 100000 * A / a
    print(P)
    print(life.frame())
    print()
    

    print("SOA Question 3.11:  (B) 0.03")
    life = LifeTable(q={50//2: .02, 52//2: .04}, udd=True).fill()
    print(life.q_r(50//2, t=2.5/2))
    print(life.frame())
    print()
    

    print("SOA Question 3.5:  (E) 106")
    l = [99999, 88888, 77777, 66666, 55555, 44444, 33333, 22222]
    a = LifeTable(l={age:l for age,l in zip(range(60, 68), l)}, udd=True)\
        .q_r(60, u=3.4, t=2.5)
    b = LifeTable(l={age:l for age,l in zip(range(60, 68), l)}, udd=False)\
        .q_r(60, u=3.4, t=2.5)
    print(100000 * (a - b))
    print()
    

    print("SOA Question 3.14:  (C) 0.345")
    life = LifeTable(l={90: 1000, 93: 825},
                     d={97: 72},
                     p={96: .2},
                     q={95: .4, 97: 1}, udd=True).fill()
    print(life.q_r(90, u=93-90, t=95.5-93))
    print(life.frame())
    print()
    
    print("Other usage examples")
    l = [110, 100, 92, 74, 58, 38, 24, 10, 0]
    table = LifeTable(l={age:l for age,l in zip(range(79, 88), l)}, 
                      interest=dict(i=0.06), maxage=87)
    print(table.mu_x(80))
#    print(table.temporary_annuity(80, t=4, m=4, due=True)) # 2.7457
#    print(table.whole_life_annuity(80, m=4, due=True, woolhouse=True)) # 3.1778
#    print(table.whole_life_annuity(80, m=4, due=False, woolhouse=True)) # 2.9278
    print(table.temporary_annuity(80, t=4)) # 2.7457
    print(table.whole_life_annuity(80)) # 3.1778
    print('*', table.whole_life_annuity(80, discrete=False)) # 2.9278

    l = [100, 90, 70, 50, 40, 20, 0]
    table = LifeTable(l={age:l for age,l in zip(range(70, 77), l)}, 
                      interest=dict(i=0.08), maxage=76)
    print(table.A_x(70),
          table.A_x(70, moment=2))  # .75848, .58486
    print(table.endowment_insurance(70, t=3)) # .81974
    print('*', table.endowment_insurance(70, t=3, discrete=False)) # .81974

    print(table.E_x(70, t=3)) # .39692
    print('*', table.term_insurance(70, t=3, discrete=False)) # .43953
    print('*', table.endowment_insurance(70, t=3, discrete=False)) # .83644
    print(table.E_x(70, t=3, moment=2)) # .31503
    print('*', table.term_insurance(70, t=3, moment=2, discrete=False)) # .38786
    print('*', table.endowment_insurance(70, t=3, moment=2, discrete=False)) # .70294


    l = [1000, 990, 975, 955, 925, 890, 840]
    table = LifeTable(l={age:l for age,l in zip(range(70, 77), l)}, 
                      interest=dict(i=0.08), maxage=76)
    print(table.increasing_annuity(70, t=4, discrete=True))
    print(table.decreasing_annuity(71, t=5, discrete=False))

    print('*', table.endowment_insurance(70, t=3, discrete=False)) # .7976

    l = [100, 90, 70, 50, 40, 20, 0]
    table = LifeTable(l={age:l for age,l in zip(range(70, 77), l)}, 
                      interest=dict(i=0.08), maxage=76)
    print(1e6*table.whole_life_annuity(70, variance=True)) #1743784

    print(LifeTable.help())
    
    
