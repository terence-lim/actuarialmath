"""Extra Risk - Adjusts force of mortality, age rating or mortality rate

MIT License. Copyright 2022-2023 Terence Lim
"""
from typing import Dict
import math
from actuarialmath import Survival
from actuarialmath import Actuarial

class ExtraRisk(Actuarial):
    """Adjust mortality by extra risk

    Args:
      life : original survival and mortality rates
      extra : amount of extra risk to adjust
      risk : adjust by {"ADD_FORCE" "MULTIPLY_FORCE" "ADD_AGE" "MULTIPLY_RATE"}

    Examples:
      >>> life = SULT()
      >>> extra = ExtraRisk(life=life, extra=2, risk="MULTIPLY_FORCE")
      >>> print(life.p_x(45), extra.p_x(45))
    """
    risks = ["ADD_FORCE", "MULTIPLY_FORCE", "ADD_AGE", "MULTIPLY_RATE"]

    def __init__(self, life: Survival, risk: str = "",
                 extra: float = 0.) -> "ExtraRisk":
        """Specify type and amount of mortality adjustment to apply"""
        assert not risk or risk in self.risks, "risk must be one of " + str(risks)
        assert extra >= 0, "amount of extra risk must be non-negative"
        self.life = life
        self.extra_ = extra
        self.risk_ = risk

    def __getitem__(self, col: str) -> Dict[int, float]:
        """Returns survival function values adjusted by extra risk

        Args:
          col : {'p', 'q'} for one-year survival or mortality function values

        Returns:
          dict of age and survival function values adjusted by extract risk

        Examples:
          >>> life = SULT()
          >>> extra = ExtraRisk(life=life, extra=0.05, risk="ADD_FORCE")
          >>> select = SelectLife(periods=1).set_select(s=0, age_selected=True, q=extra['q'])
        """
        f = {'q': self.q_x, 'p': self.p_x}[col[0]]
        return {x: f(x) for x in range(self.life._MINAGE, self.life._MAXAGE+1)}

    def p_x(self, x: int, s: int = 0) -> float:
        """Return p_[x]+s after adding or multiplying force of mortality
    
        Args:
          x : age of selection
          s : years after selection
        """
        if self.risk_ in ["MULTIPLY_RATE"]:
            return 1 - self.q_x(x, s=s)
        if self.risk_ in ["ADD_AGE"]:
            return self.life.p_x(x + self.extra_, s=s)    
        p = self.life.p_x(x, s=s)
        if self.risk_ in ["MULTIPLY_FORCE"]:
            p = p**self.extra_
        if self.risk_ in ["ADD_FORCE"]:
            p *= math.exp(-self.extra_)
        return p

    def q_x(self, x: int, s: int = 0) -> float:
        """Return q_[x]+s after adding age rating or multipliying mortality rate
    
        Args:
          x : age of selection
          s : years after selection
        """
        if self.risk_ in ["ADD_FORCE", "MULTIPLY_FORCE"]:
            return 1 - self.p_x(x, s=s)
        if self.risk_ in ["ADD_AGE"]:
            return self.life.q_x(x + self.extra_, s=s)
        if self.risk_ in ["MULTIPLY_RATE"]:
            return self.extra_ * self.life.q_x(x, s=s)

if __name__ == "__main__":
    from actuarialmath.selectlife import SelectLife
    from actuarialmath.sult import SULT
    
    print("SOA Question 5.5: (A) 1699.6")
    life = SULT()
    extra = ExtraRisk(life=life, extra=0.05, risk="ADD_FORCE")
    select = SelectLife(periods=1)\
        .set_interest(i=.05)\
        .set_select(s=0, age_selected=True, q=extra['q'])\
        .set_select(s=1, age_selected=False, a=life['a'])\
        .fill_table()
    print(100*select['a'][45][0])
    print()

    print("SOA Question 4.19:  (B) 59050")
    life = SULT()
    extra = ExtraRisk(life=life, extra=0.8, risk="MULTIPLY_RATE")
    select = SelectLife(periods=1)\
        .set_interest(i=.05)\
        .set_select(s=0, age_selected=True, q=extra['q'])\
        .set_select(s=1, age_selected=False, q=life['q'])\
        .fill_table()
    print(100000*select.whole_life_insurance(80, s=0))
    print()
    
    print("Other usage examples")
    life = SULT()
    extra = ExtraRisk(life=life, extra=2, risk="MULTIPLY_FORCE")
    print(life.p_x(45), extra.p_x(45))
