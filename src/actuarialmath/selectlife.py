"""Select and Ultimate Life Table -- Loads and calculates select life tables

MIT License. Copyright 2022-2023 Terence Lim
"""
from typing import Dict, List
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from actuarialmath import LifeTable

class SelectLife(LifeTable):
    """Calculate select life table, and iteratively fill in missing values

    Args:
      periods : number of select period years
      verbose : whether to echo update steps

    Notes:
      6 types of columns can be loaded and calculated in the select table:

      - 'q' : probability [x]+s dies in one year
      - 'l' : number of lives aged [x]+s
      - 'd' : number of deaths of age [x]+s
      - 'A' : whole life insurance
      - 'a' : whole life annuity
      - 'e' : expected future curtate lifetime of [x]+s

    Examples:
      >>> life = SelectLife().set_interest(i=0.04).set_table(q={65: [.08, .10, .12, .14],
      >>>                                                       66: [.09, .11, .13, .15],
      >>>                                                       67: [.10, .12, .14, .16],
      >>>                                                       68: [.11, .13, .15, .17],
      >>>                                                       69: [.12, .14, .16, .18]})
      >>> print(life.deferred_insurance(65, t=2, u=2, b=2000))
    """

    def __init__(self, periods: int = 0, udd: bool = True,
                 verbose: bool = False, **kwargs):
        super().__init__(udd=udd, **kwargs)
        self._select = {'l':{}, 'd':{}, 'q':{}, 'e':{}, 'A':{}, 'a':{}}
        self.periods_ = periods
        self._verbose = verbose

    def __getitem__(self, table: str) -> Dict[int, float]:
        """Returns values from a select and ultimate table

        Args:
          table :  may be {'l', 'q', 'e', 'd', 'a', 'A'}
        """
        return self._select[table[0]]

    def set_table(self, fill: bool = True,
                  l: Dict[int, List[float]] | None = None,
                  d: Dict[int, List[float]] | None = None, 
                  q: Dict[int, List[float]] | None = None,
                  A: Dict[int, List[float]] | None = None,
                  a: Dict[int, List[float]] | None = None, 
                  e: Dict[int, List[float]] | None = None) -> "SelectLife":
        """Update from table, every age has row for all select durations

        Args:
          q : probability [x]+s dies in one year
          l : number of lives aged [x]+s
          d : number of deaths of [x]+s
          A : whole life insurance, or
          a : whole life annuity, or 
          e : expected future lifetime of [x]+s
        
        Examples:
          >>> life = SelectLife().set_table(l={55: [10000, 9493, 8533, 7664],
          >>>                                  56: [8547, 8028, 6889, 5630],
          >>>                                  57: [7011, 6443, 5395, 3904],
          >>>                                  58: [5853, 4846, 3548, 2210]},
          >>>                               e={57: [None, None, None, 1]})
          >>> print(life.e_r(58, s=2))
        """
        periods = self.periods_    # infer number of select years, and age range
        minage = self._MINAGE 
        maxage = self._MAXAGE
        
        for lbl, table in [('A',A), ('a',a), ('q',q), ('d',d), ('l',l), ('e',e)]:
            self._select[lbl] = {}
            if table:     # update given table cells
                for age, row in table.items():
                    periods = max(len(row) - 1, periods)
                    minage = age if minage < 0 else min(minage, age)
                    maxage = age if maxage < 0 else max(maxage, age)
                    self._select[lbl][age] = {k:v for k,v in enumerate(row)}

        self.periods_ = periods    # update number of select years, and age range
        self._MINAGE = minage
        self._MAXAGE = maxage

        if fill:         # iteratively fill missing table values
            self.fill_table()
        return self

    def set_select(self, s: int, age_selected: bool, fill: bool = False,
                   l: Dict[int, float] | None = None,
                   d: Dict[int, float] | None = None,
                   q: Dict[int, float] | None = None,
                   A: Dict[int, float] | None = None,
                   a: Dict[int, float] | None = None,
                   e: Dict[int, float] | None = None) -> "SelectLife":
        """Update a table column, for a particular duration s in the select period

        Args:
          s : column to populate - n is ultimate, 0..n-1 is year after select
          age_selected : is indexed by age selected or actual (False, default)
          q : probabilities [x]+s dies in next year, by age
          l : number of lives aged [x]+s, by age
          d : number of deaths of [x]+s, by age
          A : whole life insurance of [x]+s, by age
          a : whole life annuity of [x]+s, by age
          e : expected future lifetime of [x]+s, by age
        """
        def ifelse(x, y): return y if x is None else x
        
        s = self.periods_ if s < 0 else s
        tables = {
            'l': ifelse(l, {}),
            'q': ifelse(q, {}),
            'd': ifelse(e, {}),
            'e': ifelse(e, {}),
            'a': ifelse(a, {}),
            'A': ifelse(A, {})
        }
        for table, item in tables.items():
            if item:
                if table not in self._select:
                    self._select[table] = {}
                for age in item.keys():
                    x = age - s * (1 - age_selected)          # get true age index
                    if x not in self._select[table]:
                        self._select[table][x] = {}
                    self._select[table][x][s] = item[age]

                    if self._MINAGE < 0 or x < self._MINAGE:  # infer min age
                        self._MINAGE = x
                    if age > self._MAXAGE:                    # infer max age
                        self._MAXAGE = x
        if fill:
            self.fill_table()
        return self

    def _get_sel(self, x: int, s: int, table: str) -> float | None:
        """Helper to read right across, and down if neccesary (when s > n)"""
        if s > self.periods_:
            x += (s - self.periods_)
            s = self.periods_
        if (x in self._select[table] and s in self._select[table][x] 
            and self._select[table][x][s] is not None):
            return self._select[table][x][s]  # in select

    def _isin_sel(self, x: int, s: int, table: str) -> bool:
        """Helper to check if value not missing"""
        return self._get_sel(x, s, table) is not None

    def fill_table(self, radix: int = LifeTable._RADIX) -> "SelectLife":
        """Fills in missing table values (does not check for consistency)

        Args:
          radix : initial number of lives
        """
        
        def A_x(x: int, s: int) -> float | None:
            """Helper to apply backward and forward recursion for insurance A_x"""
            if self._isin_sel(x, s, 'A'):
                return self._get_sel(x, s, 'A')

            _x, _s = (x, s+1) if s < self.periods_ else (x+1, s)  # right or down
            if self._isin_sel(x, s, 'q') and self._isin_sel(_x, _s, 'A'):
                q = self._get_sel(x, s, 'q')  # A_x = (v q) + (v p A_x+1)
                return self.interest.v * (q + (1-q)*self._get_sel(_x, _s, 'A'))

            backwards = [(x, s-1)]   # to move backwards along row
            if s >= self.periods_:      # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # A_x+1 = (A_x - qv) / (p v)
                if self._isin_sel(_x, _s, 'q') and self._isin_sel(_x, _s, 'A'):
                    q = self._get_sel(_x, _s, 'q')
                    return ((self._get_sel(_x, _s, 'A') - q * self.interest.v)
                            / (self.interest.v * (1 - q)))

        def a_x(x: int, s: int) -> float | None:
            """Helper to apply backward and forward recursion for annuity a_x"""
            if self._isin_sel(x, s, 'a'):
                return self._get_sel(x, s, 'a')

            _x, _s = (x, s+1) if s < self.periods_ else (x+1, s)  # right or down
            if self._isin_sel(x, s, 'q') and self._isin_sel(_x, _s, 'a'):
                p = 1 - self._get_sel(x, s, 'q')  # a_x = 1 +  (v p a_x+1)
                return 1 + self.interest.v * p * self._get_sel(_x, _s, 'a')

            backwards = [(x, s-1)]  # to move backwards along row
            if s >= self.periods_:      # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # a_x+1 = (a_x - 1) / (p v)
                if self._isin_sel(_x, _s, 'q') and self._isin_sel(_x, _s, 'a'):
                    p = 1 - self._get_sel(_x, _s, 'q')
                    return (self._get_sel(_x, _s, 'a') - 1)/(p*self.interest.v)


        def l_x(x: int, s: int) -> float | None:
            """Helper to solve for number of lives aged [x]+s: l_[x]+s"""
            if self._isin_sel(x, s, 'l'):
                return self._get_sel(x, s, 'l')
            if self._isin_sel(x, s-1, 'l') and self._isin_sel(x, s-1, 'q'):
                return self._get_sel(x, s-1, 'l')*(1 - self._get_sel(x, s-1, 'q'))
            if self._isin_sel(x, s+1, 'l') and self._isin_sel(x, s, 'q'):
                return self._get_sel(x, s+1, 'l') / (1 - self._get_sel(x, s, 'q'))
            if self._isin_sel(x, s, 'd') and self._isin_sel(x, s, 'q'):
                return self._get_sel(x, s, 'd') / self._get_sel(x, s, 'q')

            backwards = [(x, s-1)]  # to move backwards along row
            if s >= self.periods_:     # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # l_x+1 = l_x * p_x
                if self._isin_sel(_x, _s, 'l') and self._isin_sel(_x, _s, 'q'):
                    p = 1 - self._get_sel(_x, _s, 'q')
                    return self._get_sel(_x, _s, 'l') * p

        def d_x(x: int, s: int) -> float | None:
            """Helper to solve for deaths at [x]+s: l_[x]+s - l_[x]+s+1"""
            if self._isin_sel(x, s, 'd'):
                return self._get_sel(x, s, 'd')
            if self._isin_sel(x, s+1, 'l') and self._isin_sel(x, s, 'l'):
                return self._get_sel(x, s, 'l') - self._get_sel(x, s+1, 'l')

        def q_x(x: int, s: int) -> float | None:
            """Helper to solve for one-year mortality [x]+s: q_[x]+s"""
            if self._isin_sel(x, s, 'q'):
                return self._get_sel(x, s, 'q')
            if self._isin_sel(x, s, 'd') and self._isin_sel(x, s, 'l'):
                return self._get_sel(x, s, 'd') / self._get_sel(x, s, 'l')
            if self._isin_sel(x, s, 'e') and self._isin_sel(x, s+1, 'e'):
                return 1 - (self._get_sel(x, s, 'e') 
                            / (1 + self._get_sel(x, s+1, 'e')))

        def e_x(x: int, s: int) -> float | None:
            """Helper to solve for expected kurtate lifetime: e_[x]+s"""
            if self._isin_sel(x, s, 'e'):
                return self._get_sel(x, s, 'e')

            _x, _s = (x, s+1) if s < self.periods_ else (x+1, s) # right or down
            if self._isin_sel(x, s, 'q') and self._isin_sel(_x, _s, 'e'):
                return ((1 - self._get_sel(x, s, 'q'))
                        * (1 + self._get_sel(_x, _s, 'e')))  # e_x = p(1 + e_x+1)

            backwards = [(x, s-1)]  # to move backwards along row
            if s >= self.periods_:     # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # e_x+1 = e_x/p_x - 1
                if self._isin_sel(_x, _s, 'e') and self._isin_sel(_x, _s, 'q'):
                    return (self._get_sel(_x, _s, 'e') 
                            / (1 - self._get_sel(_x, _s, 'q'))) - 1

        # Iterate a few times to impute select table values
        funs = {'l': l_x, 'A': A_x, 'a': a_x, 'q': q_x, 'd': d_x, 'e': e_x}
        curr = 0   # number of updates filled in
        for loop in range(2):   # loop second time if need to assume radix
            prev = curr - 1
            while curr != prev:        # continue while changes made
                prev = curr
                for tab, fun in funs.items():   # for each table
                    if tab not in self._select:
                        self._select[tab] = {}
                    for x in range(self._MINAGE, self._MAXAGE + 1):  # for each age
                        if x not in self._select[tab]:
                            self._select[tab][x] = {}
                        for s in range(self.periods_+1): # each year after select
                            if not self._isin_sel(x, s, tab):
                                val = fun(x, s)
                                if val is not None:
                                    self._select[tab][x][s] = val
                                    curr += 1
                                    if self._verbose:
                                        print(f"{curr} {tab}(x={x}, s={s}) = {val}")
            if 0 not in self._select['l'][self._MINAGE]: # arbitrary initial lives
                self._select['l'][self._MINAGE][0] = radix
        return self

    def A_x(self, x: int, s: int = 0, moment: int = 1, discrete: bool = True,
            **kwargs) -> float:
        """Returns insurance value computed from select table

        Args:
          x : age of selection
          s : years after selection
        """
        assert moment == 1 and discrete, "Must be discrete insurance"
        if self._isin_sel(x, s, 'A'):
            return self._get_sel(x, s, 'A')
        else:
            return super().A_x(x=x, s=s, moment=1, discrete=True, **kwargs)

    def a_x(self, x: int, s: int = 0, moment: int = 1, discrete: bool = True,
            **kwargs) -> float:
        """Returns annuity value computed from select table

        Args:
          x : age of selection
          s : years after selection
        """
        assert moment == 1 and discrete, "Must be discrete annuity"
        if self._isin_sel(x, s, 'a'):
            return self._get_sel(x, s, 'a')
        else:
            return super().A_x(x=x, s=s, moment=1, discrete=True, **kwargs)

    def l_x(self, x: int, s: int = 0) -> float:
        """Returns number of lives aged [x]+s computed from select table

        Args:
          x : age of selection
          s : years after selection
        """
        return self._get_sel(x, s, 'l')

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """t_p_[x]+s by chain rule: prod(1_p_[x]+s+y) for y in range(t)

        Args:
          x : age of selection
          s : years after selection
          t : survives t years
        """
        return np.prod([1 - self._get_sel(x, s+y, 'q') for y in range(t)])

    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        """t|u_q_[x]+s = [x]+s survives u years, does not survive next t

        Args:
          x : age of selection
          s : years after selection
          u : survives u years, then
          t : dies within next t years
        """
        return (1. - self.p_x(x, s=s + u, t=t)) * self.p_x(x, s=s, t=u)

    def e_x(self, x: int, s: int = 0, t: int = LifeTable.WHOLE,
            curtate: int = True) -> float:
        """Returns expected life time computed from select table

        Args:
          x : age of selection
          s : years after selection
          t : limit of expected future lifetime
        """
        assert curtate, "Must be curtate lifetimes"
        if (self._isin_sel(x, s, 'e')):
            return self._get_sel(x, s, 'e')
        e = sum([self.p_x(x, s=s, t=k+1) for k in range(max(1, t))])
        return e

    def frame(self, table: str = 'l') -> pd.DataFrame:
        """Returns select and ultimate table values as a DataFrame

        Args:
          table : table to return, one of ['A', 'a', 'q', 'd', 'e', 'l']

        Examples:
          >>> table={21: [.00120, .00150, .00170, .00180],
          >>>        22: [.00125, .00155, .00175, .00185],
          >>>        23: [.00130, .00160, .00180, .00195]}
          >>> life = SelectLife(verbose=True).set_table(q=table)
          >>> print(life.frame('l').round(1))
          >>> print(life.frame('q').round(6))
        """
        return pd.DataFrame.from_dict(self._select[table[0]], orient='index')\
                           .sort_index(axis=0)\
                           .sort_index(axis=1)\
                           .rename_axis('Age')\
                           .rename_axis(table[0] + '_[x]+s:', axis=1)

if __name__ == "__main__":
    print("SOA Question 3.2:  (D) 14.7")
    e_curtate = SelectLife.e_approximate(e_complete=15)
    life = SelectLife(udd=True).set_table(l={65: [1000, None,],
                                             66: [955, None]},
                                          e={65: [e_curtate, None]},
                                          d={65: [40, None,],
                                             66: [45, None]})
    print(life.e_r(66))
    print(life.frame('e'))
    print()
    

    print("SOA Question 4.16:  (D) .1116")
    q = [.045, .050, .055, .060]
    q_ = {50+x: [0.7 * q[x] if x < 4 else None, 
                 0.8 * q[x+1] if x+1 < 4 else None, 
                 q[x+2] if x+2 < 4 else None] 
          for x in range(4)}
    life = SelectLife(verbose=True).set_table(q=q_).set_interest(i=.04)
    print(life.term_insurance(50, t=3))
    print()
    
    print("SOA Question 4.13:  (C) 350 ")
    life = SelectLife().set_interest(i=0.04)\
                           .set_table(q={65: [.08, .10, .12, .14],
                                         66: [.09, .11, .13, .15],
                                         67: [.10, .12, .14, .16],
                                         68: [.11, .13, .15, .17],
                                         69: [.12, .14, .16, .18]})
    print(life.deferred_insurance(65, t=2, u=2, b=2000))
    print()
    
    print("SOA Question 3.13:  (B) 1.6")
    life = SelectLife().set_table(l={55: [10000, 9493, 8533, 7664],
                                     56: [8547, 8028, 6889, 5630],
                                     57: [7011, 6443, 5395, 3904],
                                     58: [5853, 4846, 3548, 2210]},
                                  e={57: [None, None, None, 1]})
    print(life.e_r(58, s=2))
    print()


    print("SOA Question 3.12: (C) 0.055 ")
    life = SelectLife(udd=False).set_table(l={60: [10000, 9600, 8640, 7771],
                                              61: [8654, 8135, 6996, 5737],
                                              62: [7119, 6549, 5501, 4016],
                                              63: [5760, 4954, 3765, 2410]})
    print(life.q_r(60, s=1, t=3.5) - life.q_r(61, s=0, t=3.5))
    print()
    

    print("SOA Question 3.7: (b) 16.4")
    life = SelectLife().set_table(q={50: [.0050, .0063, .0080],
                                     51: [.0060, .0073, .0090],
                                     52: [.0070, .0083, .0100],
                                     53: [.0080, .0093, .0110]})
    print(1000*life.q_r(50, s=0, r=0.4, t=2.5))
    print()
    


    print("SOA Question 3.6:  (D) 5.85")
    life = SelectLife().set_table(q={60: [.09, .11, .13, .15],
                                     61: [.1, .12, .14, .16],
                                     62: [.11, .13, .15, .17],
                                     63: [.12, .14, .16, .18],
                                     64: [.13, .15, .17, .19]},
                                  e={61: [None, None, None, 5.1]})
    print(life.e_x(61))
    print()
    

    print("SOA Question 3.3:  (E) 1074")
    life = SelectLife().set_table(l={50: [99, 96, 93],
                                     51: [97, 93, 89],
                                     52: [93, 88, 83],
                                     53: [90, 84, 78]})
    print(10000*life.q_r(51, s=0, r=0.5, t=2.2))
    print()
    

    print("SOA Question 3.1:  (B) 117")
    life = SelectLife().set_table(l={60: [80000, 79000, 77000, 74000],
                                     61: [78000, 76000, 73000, 70000],
                                     62: [75000, 72000, 69000, 67000],
                                     63: [71000, 68000, 66000, 65000]})
    print(1000*life.q_r(60, s=0, r=0.75, t=3, u=2))
    print()
    

    print("Other usage examples")
    table={21: [.00120, .00150, .00170, .00180],
           22: [.00125, .00155, .00175, .00185],
           23: [.00130, .00160, .00180, .00195]}
    life = SelectLife(verbose=True).set_table(q=table)
    print(life.frame('l').round(1))
    print('--------------')
    print(life.frame('q').round(6))
    print('==============')
    print(life.p_x(21, 1, 4))  #0.99317
    
    
