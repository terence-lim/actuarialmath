"""Select and Ultimate Life Table

Copyright 2022, Terence Lim

MIT License
"""
from typing import Optional, Dict
import math
from mathlc.sult import SULT
import matplotlib.pyplot as plt
import pandas as pd

class Select(SULT):
    """Implement select and ultimate mortality life table

    Select: when mortality depends on the age when a person is selected
    - newly selected policyholder is in the best health condition possible
    - the selection process wears off

    Ultimate: after several years, selection has no effect on mortality.
    """
    _doc = ['fill', 'l_x', 'p_x', 'q_x', 'e_x', 'A_x', 'a_x']

    def __init__(self, n: int = 0, minage: int = 9999, maxage: int = 0, 
                 A: Optional[Dict] = None, a: Optional[Dict] = None, 
                 q: Optional[Dict] = None, d: Optional[Dict] = None, 
                 l: Optional[Dict] = None, e: Optional[Dict] = None, **kwargs):
        """if not specified, infers ages and select-period from initial input"""
        super().__init__(**kwargs)

        self._select = {}
        l = self.ifelse(l, {})
        d = self.ifelse(d, {})
        q = self.ifelse(q, {})
        e = self.ifelse(e, {})
        A = self.ifelse(A, {})
        a = self.ifelse(a, {})
        for lbl, col in [('A', A),('a', a),('q', q),('d', d),('l', l),('e', e)]:
            self._select[lbl] = {}
            if col:
                for age, row in col.items():
                    minage = min(minage, age)
                    maxage = max(maxage, age)
                    self._select[lbl][age] = {}
                    for ncol, val in enumerate(row):
                        self._select[lbl][age][ncol] = val
                        n = max(ncol, n)

        self.n = n   # n-year select table
        if maxage > 0:
            self.MINAGE = minage
            self.MAXAGE = maxage
            for age in range(minage, maxage+1):
                for lbl in self._select.keys():
                    if age not in self._select[lbl]:
                        self._select[lbl][age] = {}

    def __getitem__(self, col: str) -> Dict[int, float]:
        """Returns values for one of {'l', 'q', 'e', 'a', 'A'}"""
        return self._select[col]

    def set_select(self, column: int, select_age: bool,
                   q: Dict = {}, l: Dict = {}, a: Dict = {}, A: Dict = {}, 
                   e: Dict = {}) -> "Select":
        """Populate columns of table by year after selection
        - select_age: by age selected (True) or actual age (False)
        """
        column = self.n if column < 0 else column
        cols = {'l': l, 'q': q, 'e': e, 'a': a, 'A': A}
        for col, item in cols.items():
            if item:
                if col not in self._select:
                    self._select[col] = {}
                for age in range(self.MINAGE, self.MAXAGE+1):
                    item_age = age + column * (1 - select_age)  
                    if item_age in item:   # if input age is not select age (x) 
                        if age not in self._select[col]:
                            self._select[col][age] = {}
                        self._select[col][age][column] = item[item_age]
        return self

    def get_sel(self, x: int, s: int, col: str) -> Optional[float]:
        """Helper to read right across, and down if necc (when s > n)"""
        if s > self.n:
            x += (s - self.n)
            s = self.n
        if (x in self._select[col] and s in self._select[col][x] 
            and self._select[col][x][s] is not None):
            return self._select[col][x][s]  # in select

    def isin_sel(self, x: int, s: int, col: str) -> bool:
        """Helper to check if non-missing value"""
        return self.get_sel(x, s, col) is not None

    def fill(self, lifes: int = SULT.LIFES, max_iter: int = 4,
             verbose: bool = False) -> "Select":
        """Fills in missing mortality values. Does not check for consistency"""

        def A_x(x: int, s: int) -> Optional[float]:
            """Apply backward and forward recursion to solve insurance A_x"""
            if self.isin_sel(x, s, 'A'):
                return self.get_sel(x, s, 'A')

            _x, _s = (x, s+1) if s < self.n else (x+1, s)   # to move right or down
            if self.isin_sel(x, s, 'q') and self.isin_sel(_x, _s, 'A'):
                q = self.get_sel(x, s, 'q')  # A_x = (v q) + (v p A_x+1)
                return self.interest.v * (q + (1-q)*self.get_sel(_x, _s, 'A'))

            backwards = [(x, s-1)]  # move backwards along row
            if s >= self.n:         # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # A_x+1 = (A_x - qv) / (p v)
                if self.isin_sel(_x, _s, 'q') and self.isin_sel(_x, _s, 'A'):
                    q = self.get_sel(_x, _s, 'q')
                    return ((self.get_sel(_x, _s, 'A') - q * self.interest.v)
                            / (self.interest.v * (1 - q)))

        def a_x(x: int, s: int) -> Optional[float]:
            """Apply backward and forward recursion to solve annuity a_x"""
            if self.isin_sel(x, s, 'a'):
                return self.get_sel(x, s, 'a')

            _x, _s = (x, s+1) if s < self.n else (x+1, s)   # to move right or down
            if self.isin_sel(x, s, 'q') and self.isin_sel(_x, _s, 'a'):
                p = 1 - self.get_sel(x, s, 'q')  # a_x = 1 +  (v p a_x+1)
                return 1 + self.interest.v * p * self.get_sel(_x, _s, 'a')

            backwards = [(x, s-1)]  # move backwards along row
            if s >= self.n:         # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # a_x+1 = (a_x - 1) / (p v)
                if self.isin_sel(_x, _s, 'q') and self.isin_sel(_x, _s, 'a'):
                    p = 1 - self.get_sel(_x, _s, 'q')
                    return (self.get_sel(_x, _s, 'a') - 1)/(p*self.interest.v)


        def l_x(x: int, s: int) -> Optional[float]:
            """Solve for number of lives aged [x]+s: l_[x]+s"""
            if self.isin_sel(x, s, 'l'):
                return self.get_sel(x, s, 'l')
            if self.isin_sel(x, s-1, 'l') and self.isin_sel(x, s-1, 'q'):
                return self.get_sel(x, s-1, 'l')*(1 - self.get_sel(x, s-1, 'q'))
            if self.isin_sel(x, s+1, 'l') and self.isin_sel(x, s, 'q'):
                return self.get_sel(x, s+1, 'l') / (1 - self.get_sel(x, s, 'q'))
            if self.isin_sel(x, s, 'd') and self.isin_sel(x, s, 'q'):
                return self.get_sel(x, s, 'd') / self.get_sel(x, s, 'q')

            backwards = [(x, s-1)]  # move backwards along row
            if s >= self.n:         # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # l_x+1 = l_x * p_x
                if self.isin_sel(_x, _s, 'l') and self.isin_sel(_x, _s, 'q'):
                    p = 1 - self.get_sel(_x, _s, 'q')
                    return self.get_sel(_x, _s, 'l') * p

        def d_x(x: int, s: int) -> Optional[float]:
            """Solve for deaths over one year of [x]+s: l_[x]+s - l_[x]+s+1"""
            if self.isin_sel(x, s, 'd'):
                return self.get_sel(x, s, 'd')
            if self.isin_sel(x, s+1, 'l') and self.isin_sel(x, s, 'l'):
                return self.get_sel(x, s, 'l') - self.get_sel(x, s+1, 'l')

        def q_x(x: int, s: int) -> Optional[float]:
            """Solve for one-year mortality [x]+s: q_[x]+s"""
            if self.isin_sel(x, s, 'q'):
                return self.get_sel(x, s, 'q')
            if self.isin_sel(x, s, 'd') and self.isin_sel(x, s, 'l'):
                return self.get_sel(x, s, 'd') / self.get_sel(x, s, 'l')
            if self.isin_sel(x, s, 'e') and self.isin_sel(x, s+1, 'e'):
                return 1 - (self.get_sel(x, s, 'e') 
                            / (1 + self.get_sel(x, s+1, 'e')))

        def e_x(x: int, s: int) -> Optional[float]:
            """Solve for expected kurtate lifetime: e_[x]+s"""
            if self.isin_sel(x, s, 'e'):
                return self.get_sel(x, s, 'e')

            _x, _s = (x, s+1) if s < self.n else (x+1, s)  # to move right or down
            if self.isin_sel(x, s, 'q') and self.isin_sel(_x, _s, 'e'):
                return ((1 - self.get_sel(x, s, 'q'))
                        * (1 + self.get_sel(_x, _s, 'e')))  # e_x = p(1 + e_x+1)

            backwards = [(x, s-1)]  # move backwards along row
            if s >= self.n:         # if in ultimate, can also move up column
                backwards += [(x-1, s)]
            for _x, _s in backwards:   # e_x+1 = e_x/p_x - 1
                if self.isin_sel(_x, _s, 'e') and self.isin_sel(_x, _s, 'q'):
                    return (self.get_sel(_x, _s, 'e') 
                            / (1 - self.get_sel(_x, _s, 'q'))) - 1

        # Loop a few times to impute select table values
        funs = {'l': l_x, 'A': A_x, 'a': a_x, 'q': q_x, 'd': d_x, 'e': e_x}
        updated = 0
        for loop in range(max_iter):
            prev = updated - 1
            while updated != prev:
                prev = updated
                for col, fun in funs.items():
                    if col not in self._select:
                        self._select[col] = {}
                    for x in range(self.MINAGE, self.MAXAGE + 1):
                        if x not in self._select[col]:
                            self._select[col][x] = {}
                        for s in range(self.n + 1):
                            if not self.isin_sel(x, s, col):
                                value = fun(x, s)
                                if verbose:
                                    print(loop, updated, col, x, s, value)
                                if value is not None:
                                    self._select[col][x][s] = value
                                    updated += 1
            if 0 not in self._select['l'][self.MINAGE]: # arbitrary initial lifes
                self._select['l'][self.MINAGE][0] = lifes
        return self

    def A_x(self, x: int, s: int = 0, t: int = SULT.WHOLE, benefit=None, 
            moment: int = 1, discrete: bool = True) -> float:
        """Returns insurance value computed from select table"""
        assert moment == 1 and discrete
        if self.isin_sel(x, s, 'A'):
            return self.get_sel(x, s, 'A')
        else:
            return super().A_x(x=x, s=s, t=t, benefit=benefit, moment=moment,
                               discrete=discrete)

    def a_x(self, x: int, s: int = 0, t: int = SULT.WHOLE, benefit=None, 
            moment: int = 1, discrete: bool = True) -> float:
        """Returns annuity value computed from select table"""
        assert moment == 1 and discrete
        if self.isin_sel(x, s, 'a'):
            return self.get_sel(x, s, 'a')
        else:
            return super().A_x(x=x, s=s, t=t, benefit=benefit, moment=moment,
                               discrete=discrete)

    def l_x(self, x: int, s: int = 0) -> float:
        """Returns number of lifes computed from select table"""
        return self.get_sel(x, s, 'l')

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """t_p_[x]+s by chain rule: prod(1_p_[x]+s+y) for y in range(t)"""
        return math.prod([1 - self.get_sel(x, s+y, 'q') for y in range(t)])

    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        """t|u_q_[x]+s = [x]+s survives u years, does not survive next t"""
        return (1. - self.p_x(x, s=s + u, t=t)) * self.p_x(x, s=s, t=u)

    def e_x(self, x: int, s: int = 0, t: int = SULT.WHOLE,
            curtate: int = True) -> float:
        """Returns curtate expected life time computed from select table"""
        if (self.isin_sel(x, s, 'e')):
                return self.get_sel(x, s, 'e')
        e = sum([self.p_x(x, s=s, t=k+1) for k in range(t)])
        return e

    def frame(self, col: str = 'l') -> pd.DataFrame:
        """Returns select table values as a DataFrame"""
        return pd.DataFrame.from_dict(self._select[col], orient='index')\
            .sort_index(axis=0).sort_index(axis=1).rename_axis(col)

if __name__ == "__main__":

    print("SOA Question 3.2:  (D) 14.7")
    e_curtate = Select.e_curtate(e=15)
    life = Select(l={65: [1000, None,],
                     66: [955, None]},
                  e={65: [e_curtate, None]},
                  d={65: [40, None,],
                     66: [45, None]}, udd=True).fill()
    print(life.e_r(66))
    print(life.frame('e'))
    print()
    

    print("SOA Question 4.16:  (D) .1116")
    q = [.045, .050, .055, .060]
    q_ = {50+x: [0.7 * q[x] if x < 4 else None, 
                 0.8 * q[x+1] if x+1 < 4 else None, 
                 q[x+2] if x+2 < 4 else None] 
          for x in range(4)}
    life = Select(q=q_, interest=dict(i=.04)).fill()
    print(life.term_insurance(50, t=3))
    print()
    

    print("SOA Question 4.13:  (C) 350 ")
    life = Select(q={65: [.08, .10, .12, .14],
                     66: [.09, .11, .13, .15],
                     67: [.10, .12, .14, .16],
                     68: [.11, .13, .15, .17],
                     69: [.12, .14, .16, .18]}, interest=dict(i=.04)).fill()
    print(life.deferred_insurance(65, t=2, u=2, b=2000))
    print()
    

    print("SOA Question 3.13:  (B) 1.6")
    life = Select(l={55: [10000, 9493, 8533, 7664],
                     56: [8547, 8028, 6889, 5630],
                     57: [7011, 6443, 5395, 3904],
                     58: [5853, 4846, 3548, 2210]},
                  e={57: [None, None, None, 1]}).fill()
    print(life.e_r(58, s=2))
    print()
    

    print("SOA Question 3.12: (C) 0.055 ")
    life = Select(l={60: [10000, 9600, 8640, 7771],
                     61: [8654, 8135, 6996, 5737],
                     62: [7119, 6549, 5501, 4016],
                     63: [5760, 4954, 3765, 2410]}, udd=False).fill()
    print(life.q_r(60, s=1, t=3.5) - life.q_r(61, s=0, t=3.5))

    print()
    

    print("SOA Question 3.7: (b) 16.4")
    life = Select(q={50: [.0050, .0063, .0080],
                     51: [.0060, .0073, .0090],
                     52: [.0070, .0083, .0100],
                     53: [.0080, .0093, .0110]}).fill()
    print(1000*life.q_r(50, s=0, r=0.4, t=2.5))
    print()
    


    print("SOA Question 3.6:  (D) 15.85")
    life = Select(q={60: [.09, .11, .13, .15],
                     61: [.1, .12, .14, .16],
                     62: [.11, .13, .15, .17],
                     63: [.12, .14, .16, .18],
                     64: [.13, .15, .17, .19]},
                  e={61: [None, None, None, 5.1]}).fill()
    print(life.e_x(61))
    print()
    

    print("SOA Question 3.3:  (E) 1074")
    life = Select(l={50: [99, 96, 93],
                     51: [97, 93, 89],
                     52: [93, 88, 83],
                     53: [90, 84, 78]})
    print(10000*life.q_r(51, s=0, r=0.5, t=2.2))

    print()
    

    print("SOA Question 3.1:  (B) 117")
    life = Select(l={60: [80000, 79000, 77000, 74000],
                     61: [78000, 76000, 73000, 70000],
                     62: [75000, 72000, 69000, 67000],
                     63: [71000, 68000, 66000, 65000]})
    print(1000*life.q_r(60, s=0, r=0.75, t=3, u=2))
    print()
    

    print("Other usage examples")
    life = Select(minage=20, maxage=30, n=3)
    life.set_select(column=3, select_age=False, q=SULT()['q']).fill()
    print(life._select)
    print(life.frame('l'))
    print('--------------')
    print(life.frame('q'))
    print('==============')

    life = Select(q={21: [0.00120, 0.00150, 0.00170, 0.00180],
                       22: [0.00125, 0.00155, 0.00175, 0.00185],
                       23: [0.00130, 0.00160, 0.00180, 0.00195]}).fill()
    print(life.frame('l'))
    print('--------------')
    print(life.frame('q'))
    print('==============')
    print(life.p_x(21, 1, 4))  #0.99317
