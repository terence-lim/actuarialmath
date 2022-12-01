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
    
    - life (Survival) : original survival and mortality rates
    - adjust (int) : {ADD_FORCE, MULTIPLY_FORCE, ADD_AGE, MULTIPLY_RATE}
    - extra (float) : amount of extra risk to adjust by
    """
    ADD_FORCE = 1
    MULTIPLY_FORCE = 2
    ADD_AGE = 3
    MULTIPLY_RATE = 4
    _help = ['q_x', 'p_x', 'q', 'p']

    def __init__(self, life: Survival, adjust: int = 0, extra: float = 0.) -> "Adjust":
        """Specify type and amount of mortality adjustment to apply"""
        self.life = life
        self.extra = extra
        self.adjust = adjust

    @property
    def q(self) -> Dict[int, float]:
        """Adjusted mortality rates q_x, as dict keyed by age
        """
        return {x: self.q_x(x) for x in range(self.life.MINAGE, self.life.MAXAGE+1)}
    
    @property
    def p(self) -> Dict[int, float]:
        """Adjusted survival probabilities p_x, as dict keyed by age
        """
        return {x: self.p_x(x) for x in range(self.life.MINAGE, self.life.MAXAGE+1)}

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
    adjust = Adjust(life=life, extra=0.05, adjust=Adjust.ADD_FORCE)
    select = Select(n=1)\
             .set_select(column=0, select_age=True, q=adjust.q)\
             .set_select(column=1, select_age=False, a=life['a']).fill()
    print(100*select['a'][45][0])
    print()

    print("SOA Question 4.19:  (B) 59050")
    life = SULT()
    adjust = Adjust(life=life, extra=0.8, adjust=Adjust.MULTIPLY_RATE)
    select = Select(n=1)\
             .set_select(column=0, select_age=True, q=adjust.q)\
             .set_select(column=1, select_age=False, q=life['q']).fill()
    print(100000*select.whole_life_insurance(80, s=0))
    print()
    
    print("Other usage examples")
    life = SULT()
    adjust = Adjust(life=life, extra=0.05, adjust=Adjust.ADD_FORCE)
    print(life.p_x(45), adjust.p_x(45))

    print(Adjust.help())
    
