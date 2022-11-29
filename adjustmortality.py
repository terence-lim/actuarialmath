"""Adjust Mortality

Copyright 2022, Terence Lim

MIT License
"""
from actuarialmath.survival import Survival
from actuarialmath.life import Actuarial
from typing import Dict
import math

class Adjust(Actuarial):
    """Adjust: adjusts mortality by extra risk
    
    - life (Survival) : original survival and mortality functions
    """
    ADD_FORCE = 1
    MULTIPLY_FORCE = 2
    ADD_AGE = 3
    MULTIPLY_RATE = 4
    _help = ['__getitem__', '__call__', 'q_x', 'p_x']

    def __init__(self, life: Survival):
        self.life = life

    def __call__(self, extra: float, adjust: int = 0) -> "Adjust":
        """Specify type and amount of mortality adjustment to apply
        - extra (float) : amount to adjust by
        - adjust (int) : one of {ADD_FORCE, MULTIPLY_FORCE, ADD_AGE, MULTIPLY_RATE}
        """
        self.extra = extra
        self.adjust = adjust
        return self

    def __getitem__(self, col: str) -> Dict[int, float]:
        """Return adjusted survival or mortality values, as dict keyed by age
        - col (str) : one of {'q', 'p'}
        """
        fn = {'q': self.q_x, 'p': self.p_x}.get(col)
        return {x: fn(x) for x in range(self.life.MINAGE, self.life.MAXAGE+1)}

    def p_x(self, x: int, s: int = 0) -> float:
        """Return p_[x]+s after adding or multiplying force of mortality
        - x (int) : age of selection
        - s (int) : years after selection
        """
        if self.adjust in [self.MULTIPLY_RATE]:
            return 1 - self.q_x(x, s=s)
        if self.adjust in [self.ADD_AGE]:
            return self.life.p_x(x + self.extra, s=s)    
        p = self.life.p_x(x, s=s)
        if self.adjust == self.MULTIPLY_FORCE:
            p = p**self.extra
        if self.adjust == self.ADD_FORCE:
            p *= math.exp(-self.extra)
        return p

    def q_x(self, x: int, s: int = 0) -> float:
        """Return q_[x]+s after adding age rating or multipliying mortality rate
        - x (int) : age of selection
        - s (int) : years after selection
        """
        if self.adjust in [self.ADD_FORCE, self.MULTIPLY_FORCE]:
            return 1 - self.p_x(x, s=s)
        if self.adjust in [self.ADD_AGE]:
            return self.life.q_x(x + self.extra, s=s)
        if self.adjust in [self.MULTIPLY_RATE]:
            return self.extra * self.life.q_x(x, s=s)

if __name__ == "__main__":
    from actuarialmath.selectlife import Select
    from actuarialmath.sult import SULT
    
    print("SOA Question 5.5: (A) 1699.6")
    life = SULT()
    adjust = Adjust(life=life)
    q = adjust(extra=0.05, adjust=Adjust.ADD_FORCE)['q']
    select = Select(n=1)\
             .set_select(column=0, select_age=True, q=q)\
             .set_select(column=1, select_age=False, a=life['a']).fill()
    print(100*select['a'][45][0])
    print()

    print("SOA Question 4.19:  (B) 59050")
    life = SULT()
    adjust = Adjust(life=life)
    q = adjust(extra=0.8, adjust=Adjust.MULTIPLY_RATE)['q']
    select = Select(n=1)\
             .set_select(column=0, select_age=True, q=q)\
             .set_select(column=1, select_age=False, q=life['q']).fill()
    print(100000*select.whole_life_insurance(80, s=0))
    print()
    
    print("Other usage examples")
    life = SULT()
    adjust = Adjust(life=life)(extra=0.05, adjust=Adjust.ADD_FORCE)
    print(life.p_x(45), adjust.p_x(45))

    print(Adjust.help())
    
