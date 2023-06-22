"""Life Tables - Loads and calculates life tables

MIT License. Copyright 2022-2023 Terence Lim
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from actuarialmath import Reserves

class LifeTable(Reserves):
    """Calculate life table, and iteratively fill in missing values

    Args:
      udd : assume UDD or constant force of mortality for fractional ages
      verbose : whether to echo update steps

    Notes:
      4 types of columns can be loaded and calculated in the life table:

      - 'q' : probability (x) dies in one year
      - 'l' : number of lives aged x
      - 'd' : number of deaths of age x
      - 'p' : probability (x) survives at least one year

    Examples:
      >>> life = LifeTable(udd=True).set_table(l={90: 1000, 93: 825},
      >>>                                      d={97: 72},
      >>>                                      p={96: .2},
      >>>                                      q={95: .4, 97: 1})
      >>> print(life.q_r(90, u=93-90, t=95.5-93))
      >>> print(life.frame())
    """

    def __init__(self, udd: bool = True, verbose: bool = False, **kwargs):
        super().__init__(udd=udd, **kwargs)
        self._verbose = verbose
        self._table = {'l':{}, 'd':{}, 'q':{}, 'p':{}}  # columns in life table

        # Set basic survival functions by interpolating lifetable integer ages
        def _mu(x: int, s: float) -> float:
            u = math.floor(s)
            return self.mu_r(x, s=u, r=s-u)
        
        def _l(x: int, s: float) -> float:
            u = math.floor(s)
            return self.l_r(x, s=u, r=s-u)
        
        def _S(x: int, s, t: float) -> float:
            u = math.floor(t)   # u+r_t_x = u_t_x * 
            return self.p_x(x, s=s, t=u) * self.p_r(x, s=s+u, t=t-u)
        
        def _f(x: int, s, t: float) -> float:
            u = math.floor(t)   # f_x(u+r) = u_p_x * f_u(r)
            return self.p_x(x, s=s, t=u) * self.f_r(x, s=s+u, t=t-u)

        self.set_survival(mu=_mu, S=_S, f=_f, l=_l, minage=-1, maxage=-1)


    def set_table(self, fill: bool = True, minage: int = -1, maxage: int = -1,
                  l: Dict[int, float] | None = None,
                  d: Dict[int, float] | None = None,
                  p: Dict[int, float] | None = None, 
                  q: Dict[int, float] | None = None) -> "LifeTable":
        """Update life table

        Args:
          l : lives at start of year x, or
          d : deaths in year x, or
          p : probabilities that (x) survives one year, or
          q : probabilities that (x) dies in one year
          fill : whether to automatically fill table cells (default is True)
          minage : minimum age in table
          maxage : maximum age in table
        """

        inputs = {k:v for k,v in {'l':l, 'd':d, 'q':q, 'p':p}.items() if v}

        # infer min and max ages from inputs
        if minage < 0:
            minage = min([min(v) for v in inputs.values()])
            if self._MINAGE < 0 or minage < self._MINAGE:
                self._MINAGE = minage
        else:
            self._MINAGE = minage
        if maxage < 0:
            maxage = max([max(v) for v in inputs.values()])
            if self._MAXAGE < 0:
                self._MAXAGE = maxage + 1
            if maxage > self._MAXAGE:
                self._MAXAGE = maxage
        else:
            self._MAXAGE = maxage

        # update table from inputs
        for label, col in inputs.items():
            self._table[label].update(col)

        # derive and fill table values
        if fill:
            self.fill_table()
        return self

    def fill_table(self, radix: int = Reserves._RADIX) -> "LifeTable":
        """Iteratively fill in missing table cells (does not check consistency)

        Args:
          radix : initial number of lives
        """

        def q_x(x: int) -> float | None:
            """Helper to try compute one-year mortality rate for (x): 1_q_x"""
            if x in self._table['q']:
                return self._table['q'][x]
            if x in self._table['p']:
                return 1 - self._table['p'][x]
            if x in self._table['d'] and x in self._table['l']:
                return self._table['d'][x] / self._table['l'][x]
            return None

        def p_x(x: int) -> float | None:
            """Helper to try compute one-year survival for (x): 1_q_x"""
            if x in self._table['p']:
                return self._table['p'][x]
            if x in self._table['q']:
                return 1 - self._table['q'][x]
            return None

        def l_x(x: int) -> float | None:
            """Helper to try compute number of lives aged x: l_x"""
            if x in self._table['l']:
                return self._table['l'][x]
            if x+1 in self._table['l'] and x in self._table['q']:
                return self._table['l'][x+1] / (1 - self._table['q'][x])
            if x-1 in self._table['l'] and x-1 in self._table['q']:
                return self._table['l'][x-1] * (1 - self._table['q'][x-1])
            if x in self._table['d'] and x in self._table['q']:
                return self._table['d'][x] / self._table['q'][x]
            return None

        def d_x(x: int) -> float | None:
            """Helper to try compute number of deaths in one year for (x): d_x"""
            if x in self._table['d']:
                return self._table['d'][x]
            if x+1 in self._table['l'] and x in self._table['l']:
                return self._table['l'][x] - self._table['l'][x+1]
            else:
                return None

        # Iterate a few times to impute life table values
        funs = {'l': l_x, 'd': d_x, 'q': q_x, 'p': p_x}
        updated = 0
        for loop in range(2):   # loop second time if radix needed
            prev = updated - 1
            while updated != prev:        # continue while changes  
                prev = updated
                for col, fun in funs.items():  # loop columns
                    for x in range(self._MINAGE, self._MAXAGE + 1):  # loop ages
                        if x not in self._table[col]:
                            value = fun(x)
                            if value is not None:   # update value 
                                self._table[col][x] = round(value, 7)
                                updated += 1        # increment counter of changes
                                if self._verbose:
                                    print(f"{updated} {col}(x={x}) = {value}")
            if not self._table['l']:  # assume starting number of lives if necc
                self._table['l'][self._MINAGE] = radix
        return self

    def mu_x(self, x: int, s: int = 0, t: int = 0) -> float:
        """Compute mu_x from p_x in life table

        Args:
          x : age of selection
          s : years after selection
          t : death within next t years
        """
        return -math.log(max(0.00001, self.p_x(x, s=s+t, t=1)))

    def l_x(self, x: int, s: int = 0) -> float:
        """Lookup l_x from life table

        Args:
          x : age of selection
          s : years after selection
        """
        if x+s in self._table['l']:
            return self._table['l'][x+s]
        else:
            return 0

    def d_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """Compute deaths as lives at x_t divided by lives at x

        Args:
          x : age of selection
          s : years after selection
          t : death within next t years
        """
        if x+s+t <= self._MAXAGE:
            return self.l_x(x, s=s) - self.l_x(x, s=s+t)
        else:
            return 0.

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """t_p_x = lives beginning year x+t divided lives beginning year x

        Args:
          x : age of selection
          s : years after selection
          t : death within next t years
        """
        denom = self.l_x(x, s=s)
        if denom and x+s+t <= self._MAXAGE:
            return self.l_x(x, s=s+t) / denom
        else:
            return 0.   # in the long term, we are all dead

    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        """Deferred mortality: u|t_q_x = (l[x+u] - l[x+u+t]) / l[x]

        Args:
          x : age of selection
          s : years after selection
          u : survive u years, then...
          t : death within next t years
        """
        denom = self.l_x(x, s=s)
        if denom and x+s+t <= self._MAXAGE:
            return self.d_x(x, s=s+u, t=t) / self.l_x(x, s=s)
        else:
            return 1     # the only certainty in life

    def e_x(self, x: int, s: int = 0, n: int = Reserves.WHOLE,
            curtate: bool = True, moment: int = 1) -> float:
        """Expected curtate lifetime from sum of lives in table

        Args:
          x : age of selection
          s : years after selection
          n : future lifetime limited at n years
          curtate : whether curtate (True) or complete (False) expectations
          moment : whether to compute first (1) or second (2) moment

        """
        if moment == 1:
            # E[K_x] = sum([self.p(x, k+1) for k in range(n)])
            n = min(self._MAXAGE - x, n) if n > 0 else self._MAXAGE
            # approximate complete by UDD between integer age recursion
            e = sum([(1 - curtate)*(self.l(x, s=s+t) - self.l(x, s=s+t+1))*0.5
                     + self.l(x, s=s+t+1) for t in range(n)])  # s_p_x = l_x+s/l_x
            return e / self.l(x, s=0)
        else:
            return super().e_x(x=x, s=s, n=n, curtate=curtate, moment=moment)

    def E_x(self, x: int, s: int = 0, t: int = 1, moment: int = 1) -> float:
        """Pure Endowment from life table and interest rate

        Args:
          x : age of selection
          s : years after selection
          t : survives t years
          moment : return first (1) or second (2) moment or variance (-2)
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

    def __getitem__(self, col: str) -> Dict[int, float]:
        """Returns a column of the life table

        Args:
          col : name of life table column to return
        """
        assert col[0] in self._table, f"must be one of {list(self._table.keys())}"
        return self._table[col[0]]

    def frame(self) -> pd.DataFrame:
        """Return life table columns and values in a DataFrame"""
        return pd.DataFrame.from_dict(self._table).sort_index(axis=0)


if __name__ == "__main__":
    print("SOA Question 6.53:  (D) 720")
    x = 0
    life = LifeTable().set_interest(i=0.08)\
                      .set_table(q={x: 0.1, x+1: 0.1, x+2: 0.1})
    A = life.term_insurance(x, t=3)
    G = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
    print(A, G)
    print(life.frame())
    print()

    print("SOA Question 6.41:  (B) 1417")
    x = 0
    life = LifeTable().set_interest(i=0.05)\
                      .set_table(q={x:.01, x+1:.02})
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
    life = LifeTable(udd=True).set_table(q={50//2: .02, 52//2: .04})
    print(life.q_r(50//2, t=2.5/2))
    print(life.frame())
    print()
    

    print("SOA Question 3.5:  (E) 106")
    l = {60 + x: l * 11111 for x,l in enumerate([9, 8, 7, 6, 5, 4, 3, 2])}
    a, b = (LifeTable(udd=udd).set_table(l=l).q_r(60, u=3.4, t=2.5)
            for udd in [True, False])
    print(100000 * (a - b))
    print()
    

    print("SOA Question 3.14:  (C) 0.345")
    life = LifeTable(udd=True).set_table(l={90: 1000, 93: 825},
                                         d={97: 72},
                                         p={96: .2},
                                         q={95: .4, 97: 1})
    print(life.q_r(90, u=93-90, t=95.5-93))
    print(life.frame())
    print()
