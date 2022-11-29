"""Recursion Formulas and Identity Relationships

Copyright 2022, Terence Lim

MIT License
"""
from actuarialmath.reserves import Reserves
from typing import Tuple, Optional

class Recursion(Reserves):
    """Recursion: apply recursion and alternate formulas

    - depth (int) : maximum depth of recursions
    - verbose (bool) : level of verbosity
    """
    _help = ['set_q', 'set_p', 'set_e', 'set_E', 'set_A', 'set_IA', 'set_DA',
             'set_a']
    def __init__(self, depth: int = 6, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.db = {}
        Recursion._depth = depth
        Recursion.verbose = verbose

    #
    # helpers to record and display recursion message history
    #
    class _H:
        """Class to hold message history"""
        def __init__(self, title: str = ""):
            self.title = title   # title to identify this recursion stack
            self.history = []
            self.depth = Recursion._depth
        
        def __call__(self, depth, *args):
            """Append next message to history"""
            # while len(self._history) and self._history[-1][0] < depth:
            # self._history.pop(-1)
            self.history.append([abs(depth), " ".join([str(a) for a in args])])

        def __str__(self, verbose=True):
            """Display message history"""
            _str = ""
            if verbose and Recursion.verbose:
                if self.history:
                    _str += self.title + '\n'
                self.history.reverse()
                for depth, msg in self.history:
                    _str += ("   " * (1+self.depth-abs(depth))) + f" {msg}\n"
            return _str

    #
    # helpers to store given input values
    #
    def db_key(self, **kwargs) -> Tuple:
        """Generate a unique key for the item"""
        return tuple(sorted(kwargs.items()))

    def db_put(self, key: Tuple, val: Optional[float]) -> "Recursion":
        """Store the item's key and value; or remove if value is None"""
        if val is None and key in self.db:
            self.db.pop(key)
        else:
            self.db[key] = val
        return self

    def db_print(self):
        """Display the stored keys and values"""
        for k in sorted(self.db.keys()):
            print(k, self.db[k])

    #
    # Formulas for Mortality: u|t_q_x
    #
    def get_q(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> Optional[float]:
        """Get mortality rate from key-value store"""
        key = self.db_key(_key='q', x=x+s, u=u, t=t)
        return self.db.get(key, None)

    def set_q(self, val: float, x: int, s: int = 0, t: int = 1, 
              u: int = 0) -> "Recursion":
        """Set mortality rate u|t_q_[x]+s given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - u (int) : survive u years, then...
        - t (int) : death within next t years        
        """
        return self.db_put(self.db_key(_key='q', x=x+s, u=u, t=t), val)

    def _q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0, 
             depth: int = 1) -> float:
        """Helper to compute mortality from recursive and alternate formulas"""
        found = self.get_q(x, s=s, t=t, u=u)
        if found is not None:
            return found
        if t == 0:
            return 0
        if t < 0:
            return 1
        if u > 0:
            pu = self._p_x(x, s=s, t=u, depth=1-abs(depth))
            qt = self.get_q(x, s=s+u, t=t)
            if pu is not None and qt is not None:
                self._h(depth, f"defer mortality: u|t_q_x = u_p_x * t_q_x+u")
                return pu * qt        # (1) u_p_x * t_q_x+u
            qu = self.get_q(x, s=s, t=u)
            qt = self.get_q(x, s=s, t=u+t)
            if qu is not None and qt is not None:
                self._h(depth, f"limit mortality: u|t_q_x = u+t_q_x - u_q_x")
                return qt - qu        # (2) u+t_q_x - u_q_x
        if depth <= 0:
            return None
        pu = self._p_x(x, s=s, t=u, depth=depth-1)
        pt = self._p_x(x, s=s, t=u+t, depth=depth-1)
        if pu is not None and pt is not None:
            msg = f"complement survival: {u}|{t}_q_{x+s} = u_p_x - u+t_p_x"
            self._h(depth, msg)
            return pu - pt        # (3) u_p_x - u+t_p_x


    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        self._h = self._H(f"[ Mortality: {(str(u)+'|') if u else ''}"
                                + f"{t}_q_{x+s} ]")
        """Compute mortality rate with recursion helper"""
        q = self._q_x(x, s=s, t=t, u=u, depth=self._depth)
        if q is not None:
            print(self._h, end='')
        return q

    #
    # Formulas for Survival: t_p_x
    #
    def get_p(self, x: int, s: int = 0, t: int = 1) -> Optional[float]:
        """Get survival probability from key-value store"""
        if t == 0:
            return 1
        if t < 0:
            return 0
        key = self.db_key(_key='p', x=x+s, t=t)
        return self.db.get(key, None)

    def set_p(self, val: float, x: int, s: int = 0, t: int = 1) -> "Recursion":
        """Set survival probability t_p_[x]+s given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : survives next t years
        """
        return self.db_put(self.db_key(_key='p', x=x+s, t=t), val)

    def _p_x(self, x: int, s: int = 0, t: int = 1, depth: int = 1) -> float:
        """Helper to compute survival from recursive and alternate formulas"""
        found = self.get_p(x, s=s, t=t)
        if found is not None:
            return found
        found = self.get_q(x, s=s, t=t)  
        if found is not None:
            return 1 - found  # (1) complement of q_x
        if depth <= 0:
            return None
        if t > 1:    # (2) chain rule: p_x(t) = p_x * p_x+1(t-1)
            found = self._p_x(x, s=s+1, t=t-1, depth=depth-1)
            p = self._p_x(x, s=s, t=1, depth=1-abs(depth))
            if found is not None and p is not None:
                msg = (f"survival recursion {t}_p_{x+s} = "
                        + f"p_{x+s} * {t-1}_p_{x+s+1}")
                self._h(depth, msg)
                return found * p
        #
        # TODO: back out from Insurance and Annuity recursions?
        #

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """Compute survival probability with recursion helper"""
        self._h = self._H(f"[ Survival: {t}_p_{x+s} ]")
        p = self._p_x(x, s=s, t=t, depth=self._depth)
        if p is not None:
            print(self._h, end='')
        return p

    #
    # Formulas for Expected Future Lifetime: e_x
    #
    def get_e(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
              curtate: bool = False, moment: int = 1) -> Optional[float]:
        """Get expected future lifetime from key-value store"""
        key = self.db_key(_key='e', x=x+s, t=t, curtate=curtate, moment=moment)
        return self.db.get(key, None)

    def set_e(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE, 
              curtate: bool = False, moment: int = 1) -> "Recursion":
        """Set expected future lifetime e_[x]+s:t given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : limit of expected future lifetime
        - curtate (bool) : curtate (True) or complete expectation (False)
        - moment (int) : first or second moment of expected future lifetime
        """
        return self.db_put(self.db_key(_key='e', x=x+s, t=t, moment=moment,
                                       curtate=curtate), val)

    def _e_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
            curtate: bool = False, moment: int = 1, 
            depth: int = 1) -> Optional[float]:
        """Helper to compute from recursive and alternate formulas"""
        found = self.get_e(x, s=s, t=t, curtate=curtate, moment=moment)
        if found is not None:
            return found
        if depth <= 0:
            return None
        if moment == 1:
            if t > 0:  
                p_t = self._p_x(x, s=s, t=t)
                if t == 1 and curtate:
                    self._h(depth, f"shortcut 1-year curtate e_{x+s}:1")
                    return p_t   # (1) if curtate and t=1: e_x:1 = p_x 
                if t > 1:
                    e = self._e_x(x, s=s, t=Reserves.WHOLE, curtate=curtate, 
                                  moment=1, depth=depth-1)
                    e_t = self._e_x(x, s=s+t, t=Reserves.WHOLE, curtate=curtate,
                                    moment=1, depth=depth-1)
                    if e is not None and e_t is not None and p_t is not None:
                        self._h(depth, f"temporary e_{x+s}:{t}: e_x - p_x e_x+t")
                        return e - p_t*e_t  # (2) temporary = e_x - t_p_x e_x+t
            e = self._e_x(x, s=s-1, t=self.add_term(t, 1), curtate=curtate,
                          moment=1, depth=depth-1)
            e1 = self._e_x(x, s=s-1, t=1, curtate=curtate, moment=1,
                          depth=depth-1)
            p = self._p_x(x, s=s-1)
            if e is not None and e1 is not None and p is not None:
                _t = "" if t < 0 else ":" + str(t)
                msg = f"forward e_{x+s}{_t} = e_{x+s}:1 + p_{x+s} e_{x+s+1}"
                self._h(depth, msg)
                return (e - e1) / p # (3) forward: (e_x-1 - e_x-1:1) / p_x-1
            e = self._e_x(x, s=s, t=1, curtate=curtate,
                          moment=1, depth=depth-1)
            e_t = self._e_x(x, s=s+1, t=self.add_term(t, -1), curtate=curtate,
                            moment=1, depth=depth-1)
            p = self._p_x(x, s=s)
            if e is not None and e_t is not None and p is not None:
                self._h(depth, f"backward: e_x:1 + p_x e_x+1:{t}")
                return e + p * e_t # (4) backward: e_x:1 + p_x e_x+1:t-1


    def e_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
            curtate: bool = False, moment: int = 1) -> float:
        """Compute expected future lifetime with recursion helper"""
        self._h = self._H(f"[ Lifetime: e_{x+s}{(':'+str(t)) if t >=0 else ''} ]")
        e = self._e_x(x, s=s, t=t, curtate=curtate, moment=moment,
                      depth=self._depth)
        if e is not None:
            print(self._h, end='')
            return e

    #
    # Formulas for Pure Endowment: t_E_x
    #
    def get_E(self, x: int, s: int = 0, t: int = 1, 
              endowment: int = 1, moment: int = 1) -> Optional[float]:
        """Get pure endowment from key-value store"""
        key = self.db_key(_key='E', x=x+s, t=t, moment=moment)
        val = self.db.get(key, None)
        if val is not None:  
            return val * endowment   # stored with benefit=1

    def set_E(self, val: float, x: int, s: int = 0, t: int = 1, 
              endowment: int = 1, moment: int = 1) -> "Recursion":
        """Set pure endowment t_E_[x]+s given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : death within next t years
        - endowment (int) : endowment value
        - moment (int) : first or second moment of pure endowment
        """
        val /= endowment   # store with benefit=1
        return self.db_put(self.db_key(_key='E', x=x+s, t=t, moment=moment), val)

    def _E_x(self, x: int, s: int = 0, t: int = 1, endowment: int = 1, 
             moment: int = 1, depth: int = 1) -> float:
        """Helper to compute from recursive and alternate formulas"""
        found = self.get_E(x, s=s, t=t, endowment=endowment, moment=moment)
        if found is not None:
            return found
        if t < 0:     # t infinite => EPV(t) = 0
            return 0
        if t == 0:    # t = 0 => EPV(0) = endowment**moment
            return endowment**moment
        if moment > 1:
            E = self._E_x(x, s=s, endowment=endowment, depth=depth)
            if found:  # (1) Shortcut: 2E_x = v E_x
                return E * self.interest.v**(moment-1) 
        p = self._p_x(x, s=s, t=t, depth=1-abs(depth))
        if p is not None:   # (2) E_x = p_x * v
            msg = f"pure endowment {t}_E_{x+s} = {t}_p_{x+s} * v^{t}"
            self._h(self._depth, msg)
            return p * (endowment * self.interest.v_t(t))**moment
        if depth <= 0:
            return None

        At = self._A_x(x, s=s, t=t, moment=moment, b=endowment, endowment=0,
                       depth=1-abs(depth))
        A = self._A_x(x, s=s, t=t, b=endowment, endowment=endowment, 
                      moment=moment, depth=1-abs(depth))        
        if A is not None and At is not None:
            self._h(depth, f"endowment - term insurance = {t}_E_{x+s}")
            return A - At  # (3) endowment insurance - term (helpful SULT)

        E = self._E_x(x, s=s, moment=moment, depth=1-abs(depth))
        Et = self._E_x(x, s=s+1, t=t-1, moment=moment, depth=depth-1)
        if E is not None and Et is not None:
            msg = f"chain Rule: {t}_E_{x+s} = E_{x+s} * {t-1}_E_{x+s+1}"
            self._h(depth, msg)
            return E * Et * endowment**moment # (4) chain rule

    def E_x(self, x: int, s: int = 0, t: int = 1, 
            endowment: int = 1, moment: int = 1) -> float:
        """Compute pure endowment with recursion helper"""
        self._h = self._H(f"[ Pure Endowment: {t}_E_{x+s} ]")
        if moment == self.VARIANCE:  # Bernoulli shortcut for variance
            found = self.get_E(x, s=s, t=t, endowment=endowment, moment=moment)
            if found is not None:
                return found
            t_p_x = self.p_x(x, s=s, t=t)
            return (endowment * self.interest.v_t(t))**2 * t_p_x * (1-t_p_x)
        found = self._E_x(x, s=s, t=t, endowment=endowment, moment=moment,
                         depth=self._depth)
        if found is not None:
            print(self._h, end='')
            return found

    #
    # Formulas for Increasing Insurance: IA_x:t
    #
    def get_IA(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, discrete: bool = True) -> Optional[float]:
        """Get increasing insurance from key-value store"""
        key = self.db_key(_key='IA', x=x+s, t=t, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * b   # stored with benefit=1

    def set_IA(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, discrete: bool = True) -> "Recursion":
        """Set increasing insurance IA_[x]+s:t given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : term of increasing insurance
        - b (int) : benefit after year 1
        - discrete (bool) : discrete or continuous increasing insurance
        """
        val /= b   # store with benefit=1
        return self.db_put(self.db_key(_key='IA', x=x+s, t=t, 
                                       discrete=discrete), val)

    def _IA_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, b: int = 1,
              discrete: bool = True, depth: int = 1) -> Optional[float]:
        """Helper to compute from recursive and alternate formulas"""
        if t == 0:
            return 0
        found = self.get_IA(x=x, s=s, t=t, b=b, discrete=discrete)
        if found is not None:
            return found
        if depth <= 0:
            return None

        if t > 0:  # decreasing must be term insurance
            A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=depth-1)
            n = t + int(discrete)
            DA = self._DA_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=depth-1)
            if A is not None and DA is not None:
                self._h(depth, f"identity IA_{x+s}:{t}: ({n})A - DA")
                return A * n - DA  # (1) Identity with term and decreasing

        A = self._A_x(x=x, s=s, t=1, b=b, discrete=discrete, depth=depth-1)
        IA = self._IA_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, depth=depth-1)
        p = self._p_x(x, s=s, t=t, depth=1-abs(depth))
        if A is not None and IA is not None and p is not None:
            self._h(depth, f"backward IA_{x+s}:{t}: A + IA_{x+s+1}:{t-1}")
            return A + p * self.interest.v * IA  # (2) backward recursion

    def increasing_insurance(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                             b: int = 1, discrete: bool = True) -> float:
        """Compute increasing insurance with recursive helper"""
        self._h = self._H(f"[ Increasing Insurance: IA_{x+s}"
                          + f"{(':'+str(t)) if t >=0 else ''} ]")
        IA = self._IA_x(x, s=s, b=b, t=t, discrete=discrete, depth=self._depth)
        if IA is not None:
            print(self._h, end='')
            return IA
        IA = super().increasing_insurance(x, s=s, b=b, t=t, discrete=discrete)
        if IA is not None:
            print(self._h, end='')
            return IA

    #
    # Formulas for Decreasing insurance: DA_x:t
    #
    def get_DA(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, discrete: bool = True) -> Optional[float]:
        """Get decreasing insurance from key-value store"""
        key = self.db_key(_key='DA', x=x+s, t=t, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * b   # stored with benefit=1

    def set_DA(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, discrete: bool = True) -> "Recursion":
        """Set decreasing insurance DA_[x]+s:t given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - t (int) : term of decreasing insurance
        - b (int) : benefit after year 1
        - discrete (bool) : discrete or continuous decreasing insurance
        """
        val /= b     # store with benefit=1
        return self.db_put(self.db_key(_key='DA', x=x+s, t=t, 
                                       discrete=discrete), val)

    def _DA_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, b: int = 1,
              discrete: bool = True, depth: int = 1) -> Optional[float]:
        """Helper to compute from recursive and alternate formulas"""
        found = self.get_DA(x=x, s=s, t=t, discrete=discrete)
        if found is not None:
            return found
        if t == 0:
            return 0
        if depth <= 0:
            return None
        if t < 0:  # decreasing must be term insurance
            return None
    
        A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=depth-1)
        n = t + int(discrete)
        IA = self._DA_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=depth-1)
        if A is not None and IA is not None:
            self._h(depth, f"identity DA_{x+s}:{t}: ({n})A - IA")
            return A * n - IA  # (1) identity with term and decreasing

        DA = self._IA_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, depth=depth-1)
        p = self._p_x(x, s=s, depth=1-abs(depth))
        if DA is not None and p is not None:
            msg = f"backward DA_{x+s}:{t}: v(t q_{x+s} + p_{x+s} DA_{x+s+1}:{t-1})"
            self._h(depth, msg)
            return self.interest.v * ((1-p)*t + p*DA)  # (2) backward recursion

    def decreasing_insurance(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                             b: int = 1, discrete: bool = True) -> float:
        """Compute decreasing insurance from recursive helper"""
        self._h = self._H(f"[ Decreasing Insurance: "
                          + f"DA_{x+s}{(':'+str(t)) if t >=0 else ''} ]")
        if t == 0:
            return 0
        A = self._DA_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=self._depth)
        if A is not None:
            print(self._h, end='')
            return A
        A = super().decreasing_insurance(x, s=s, b=b, t=t, discrete=discrete)
        if A is not None:
            print(self._h, end='')
            return A

    #
    # Formulas for Insurance: A_x:t
    #
    def get_A(self, x: int, s: int = 0, u: int = 0, t: int = Reserves.WHOLE,
              b: int = 1, moment: int = 1, endowment: int = 0, 
              discrete: bool = True) -> Optional[float]:
        """Get insurance from key-value store"""
        if endowment < 0:   # endowment insurance with equal benefits
            endowment = b
        if t == 0:          # terminal value of insurance
            return endowment
        if b == 0:          # normalize insurance factor by benefit amount
            scale = endowment
            endowment = 1
        else:
            scale = b
            endowment = 1 if b == endowment else endowment / b
        key = self.db_key(_key='A', x=x+s, u=u, t=t, moment=moment, 
                          endowment=endowment, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * scale   # stored with benefit=1


    def set_A(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
              u: int = 0, b: int = 1, moment: int = 1, endowment: int = 0, 
              discrete: bool = True) -> "Recursion":
        """Set insurance u|_A_[x]+s:t given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - u (int) : defer u years
        - t (int) : term of insurance
        - endowment (int) : endowment amount
        - discrete (bool) : discrete (True) or continuous (False) insurance
        - moment (int) : first or second moment of insurance
        """
        if endowment < 0:      # endowment insurance with equal death and endow
            endowment = b
        if endowment == 0:     # normalize insurance factor by benefit amount
            val /= b
        elif b == 0:
            val /= endowment   # store with benefit=1
        else:
            if b != 1:
                val /= b
                endowment /= b
        return self.db_put(self.db_key(_key='A', x=x+s, t=t, u=u,
                                       moment=moment, endowment=endowment, 
                                       discrete=discrete), val)

    def _A_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, u: int = 0,
             b: int = 1, discrete: bool = True, endowment: int = 0,
             moment: int = 1, depth: int = 1) -> Optional[float]:
        """Helper to compute from recursive and alternate formulas"""
        if endowment == b and t == 1 and discrete: # 1-year endow ins
            return (self.interest.v_t(1) * endowment)**moment
        found = self.get_A(x=x, s=s, t=t, b=b, u=u, discrete=discrete, 
                           moment=moment, endowment=endowment)
        if found is not None:
            return found
        if depth <= 0:
            return None

        if u > 0:  # (1) deferred insurance  
            A = self._A_x(x=x, s=s+1, t=t, b=b, u=u-1, discrete=discrete, 
                              moment=moment, endowment=endowment, depth=depth-1)
            E = self._E_x(x, s=s, t=1, moment=moment, depth=depth-1)
            if A is not None and p is not None:  # (1a) backward E_x * A
                msg = f"backward deferred {u}_A_{x+s}: {u}_E * A_{x+s+u}"
                self._h(depth, msg)
                return E * A

            A = self._A_x(x, s=s-1, t=t, b=b, u=u+1, discrete=discrete, 
                          moment=moment, endowment=endowment, depth=depth-1)
            E = self._E_x(x, s=s-1, t=1, moment=moment, depth=depth-1)
            if A is not None and E is not None: # (1b) forward recursion
                msg = f"forward deferred {u}_A_{x+s}: {u+1}A_{x+s-1} / E"
                self._h(depth, msg)
                return A / E
            return None

        if endowment > 0: # (2a) endowment = term + E_x * endowment
            A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, 
                          moment=moment, depth=1-abs(depth))
            E_x = self._E_x(x=x, s=s, t=t, moment=moment, 
                            endowment=endowment, depth=1-abs(depth))
            if A is not None and E_x is not None:
                self._h(depth, f"term + pure insurance = A_{x+s}:{t}")
                return A + E_x
        else:             # (2b) term = endowment insurance - E_x * endowment
            A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, 
                          moment=moment, endowment=b, depth=1-abs(depth))
            E_x = self._E_x(x=x, s=s, t=t, moment=moment, endowment=b, 
                            depth=1-abs(depth))
            if A is not None and E_x is not None:
                msg = f"endowment insurance - pure endowment = A_{x+s}^1:{t}"
                self._h(depth, msg)
                return A - E_x

        if not discrete:  # recursions for discrete insurance
            return None
        if t == 1:  # special cases for discrete one-year insurance
            if endowment == b:  # (3a) discrete one-year endowment insurance
                self._h(depth, f"shortcut 1-year endowment insurance = 1")
                return (self.interest.v * endowment)**moment
            p = self._p_x(x, s=s, t=1)
            if p is not None:  # (3b) one-year discrete insurance
                self._h(depth, f"discrete 1-year insurance: A_{x+s}:1 = qv")
                return (self.interest.v**moment 
                        * ((1 - p) * b**moment + p * endowment**moment))
    
        A = self._A_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, 
                      discrete=discrete, moment=moment,
                      endowment=endowment, depth=depth-1)
        p = self._p_x(x, s=s, t=1, depth=1-abs(depth)) # (4) backward recursion
        if A is not None and p is not None:
            self._h(depth, f"backward: A_{x+s} = qv + pvA_{x+s+1}")
            return self.interest.v_t(1)**moment * ((1 - p)*b**moment + p*A)

        A = self._A_x(x=x, s=s-1, t=self.add_term(t, 1), b=b, u=u, 
                      discrete=discrete, moment=moment, 
                      endowment=endowment, depth=depth-1)
        p = self._p_x(x, s=s-1, t=1, depth=1-abs(depth))
        if A is not None and p is not None:  # (5) forward recursion
            self._h(depth, f"forward: A_{x+s} = (A_{x+s-1}/v - q) / p")
            return (A/self.interest.v_t(1)**moment - (1-p)*b**moment) / p

    def whole_life_insurance(self, x: int, s: int = 0, b: int = 1, 
                             discrete: bool = True, moment: int = 1) -> float:
        """Compute whole life insurance with recursion helper: A_x"""
        self._h = self._H(f"[ Whole Life Insurance: A_{x+s} ]")
        found = self._A_x(x, s=s, b=b, moment=moment, discrete=discrete,
                          depth=self._depth)
        if found is not None:
            print(self._h, end='')
            return found
        if moment == 1 and self.interest.i > 0:  # (1) twin annuity
            a = self._a_x(x, s=s, b=b, discrete=discrete, depth=self._depth)
            if a is not None:
                print(self._h, end='')
                return self.insurance_twin(a=a, discrete=discrete)
        A = super().whole_life_insurance(x, s=s, b=b, discrete=discrete,
                                         moment=moment)
        if A is not None:
            print(self._h, end='')
            return A

    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                       moment: int = 1, discrete: bool = True) -> float:
        """Compute term life insurance with recursion helper: A_x:t^1"""
        self._h = self._H(f"[ Term Insurance: "
                          + f"A_{x+s}{('^1:'+str(t)) if t >=0 else ''} ]")
        found = self._A_x(x, s=s, b=b, t=t, moment=moment, discrete=discrete,
                          depth=self._depth)
        if found is not None:
            print(self._h, end='')
            return found
        A = super().term_insurance(x, s=s, b=b, t=t, discrete=discrete,
                                   moment=moment)
        if A is not None:
            print(self._h, end='')
            return A

    def deferred_insurance(self, x: int, s: int = 0, b: int = 1, u: int = 0, 
                           t: int = Reserves.WHOLE, moment: int = 1, 
                           discrete: bool = True) -> float:
        """Compute deferred life insurance with recursion helper: u|A_x:t^1"""
        self._h = self._H(f"[ Deferred Insurance: {(str(u)+'_') if u else ''}"
                          + f"A_{x+s}{('^1:'+str(t)) if t >=0 else ''} ]")
        A = self.get_A(x=x, s=s, t=t, b=b, u=u, discrete=discrete, 
                       moment=moment)
        if A is not None:
            print(self._h, end='')
            return A
        A = super().deferred_insurance(x, s=s, b=b, t=t, u=u, 
                                       discrete=discrete, moment=moment)
        if A is not None:
            print(self._h, end='')
            return A
    
    def endowment_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1,
                            endowment: int = -1, moment: int = 1,
                            discrete: bool = True) -> float:
        """Compute endowment insurance with recursion helper: u|A_x:t"""
        self._h = self._H(f"[ Endowment Insurance: A_{x+s}"
                          + f"{(':'+str(t)) if t >=0 else ''} ]")
        assert t >= 0
        if endowment < 0:
            endowment = b
        found = self._A_x(x, s=s, b=b, t=t, moment=moment, discrete=discrete,
                          endowment=endowment, depth=self._depth)
        if found is not None:
            print(self._h, end='')
            return found
        if moment == 1 and endowment == b and self.interest.i > 0:
            a = self._a_x(x, s=s, b=b, t=t, moment=moment, discrete=discrete,
                              depth=self._depth)
            if a is not None:   # twin insurance
                print(self._h, end='')
                return self.insurance_twin(a=a, discrete=discrete)
        A = super().endowment_insurance(x, s=s, b=b, t=t, discrete=discrete,
                                        moment=moment, endowment=endowment)
        if A is not None:
            print(self._h, end='')
            return A

    #
    # Formulas for Annuties: a_x:t
    #
    def get_a(self, x: int, s: int = 0, u: int = 0, t: int = Reserves.WHOLE,
              b: int = 1, variance: bool = False, 
              discrete: bool = True) -> Optional[float]:
        """Get annuity from key-value store"""
        key = self.db_key(_key='a', x=x+s, u=u, t=t, 
                          variance=variance, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * b    # stored with benefit=1

    def set_a(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
              u: int = 0, b: int = 1, variance: bool = False, 
              discrete: bool = True) -> "Recursion":
        """Set annuity u|_a_[x]+s:t given value
        - val (float) : value to set
        - x (int) : age of selection
        - s (int) : years after selection
        - u (int) : defer u years
        - t (int) : term of annuity
        - b (int) : benefit amount
        - discrete (bool) : whether annuity due (True) or continuous (False)
        - variance (bool) : whether first moment (False) or variance (True)
        """
        val /= b    # store with benefit=1
        return self.db_put(self.db_key(_key='a', x=x+s, t=t, u=u, 
                           variance=variance, discrete=discrete), val)

    def _a_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
             u: int = 0, b: int = 1, discrete: bool = True, 
             variance: bool = False, depth: int = 1) -> Optional[float]:
        """Helper to compute from recursive and alternate formulas"""
        if t == 1 and discrete:
            self._h(depth, "1-year discrete annuity: a_x:1 = 1")
            return b
        if t == 0:
            return 0
        found = self.get_a(x=x, s=s, t=t, b=b, u=u, discrete=discrete,
                           variance=variance)
        if found is not None:
            return found
        if depth <= 0:
            return None
        if variance:
            return None

        if u > 0:  # (1) deferred annuity
            found = self._a_x(x=x, s=s+1, t=t, b=b, u=u-1, discrete=discrete, 
                              variance=variance, depth=depth-1)
            E = self._E_x(x, s=s, t=1, depth=depth-1)
            if found is not None and E is not None:
                msg = f"backward {u}_a_{x+s} = {u}_E * a_{x+s+u}"
                self._h(depth, msg)
                return E * found  # (1a) backward recusion

            found = self._a_x(x=x, s=s-1, t=t, b=b, u=u, discrete=discrete, 
                              depth=depth-1)
            E = self._E_x(x, s=s-1, t=1, depth=depth-1)
            if found is not None and E is not None:  # (1b) forward
                msg = f"forward: {u}_a_{x+s} = {u-1}_a_{x+s-1}/E_{x+s-1}"
                self._h(depth, msg)
                return found / E

        else:  # (2) Temporary and whole annuity
            found = self._a_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, u=u, 
                              discrete=discrete, 
                              variance=variance, depth=depth-1)
            E = self._E_x(x, s=s, t=1, depth=depth-1)
            if found is not None and E is not None:  # (2a) backward
                msg = (f"backward: a_{x+s}{'' if t < 0 else (':'+str(t))} = 1 + "
                       + f"E_{x+s} a_{x+s+1}{'' if t < 0 else (':'+str(t-1))}")
                _t = "" if t < 0 else f":{t-1}"
                self._h(depth, msg)
                return b + E * found

            found = self._a_x(x=x, s=s-1, t=self.add_term(t, 1), b=b, u=u, 
                              discrete=discrete, depth=depth-1)
            E = self._E_x(x, s=s-1, t=1, depth=depth-1)
            if found is not None and E is not None:  # (2b) forward
                _t = "" if t < 0 else f":{t-1}"
                self._h(depth, f"forward: a_{x+s}{_t} = (a_{x+s-1} - 1)/E")
                return (found - b) / E

    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1, 
                           variance: bool = False,
                           discrete: bool = True) -> float:
        """Compute whole life annuity with recursion first: a_x"""
        self._h = self._H(f"[ Whole Life Annuity: a_{x+s} ]")
        found = self._a_x(x, s=s, b=b, variance=variance, 
                          discrete=discrete, depth=self._depth)
        if found is not None:
            print(self._h, end='')
            return found
        if not variance and self.interest.i > 0:  # (1) twin insurance shortcut
            A = self._A_x(x, s=s, b=b, discrete=discrete, depth=self._depth)
            if A is not None:
                print(self._h, end='')
                return self.annuity_twin(A=A, discrete=discrete)
        a = super().whole_life_annuity(x, s=s, b=b, discrete=discrete,
                                       variance=variance)
        if a is not None:
            print(self._h, end='') 
            return a

    def temporary_annuity(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                          b: int = 1, variance:bool = False,
                          discrete: bool = True) -> float:
        """Compute temporary annuity with recursion first: a_x:t"""
        self._h = self._H(f"[ Temporary Annuity: "
                          + f"a_{x+s}{(':'+str(t)) if t >=0 else ''} ]")
        found = self._a_x(x, s=s, b=b, t=t, variance=variance, 
                          discrete=discrete, depth=self._depth)
        if found is not None:
            print(self._h, end='')
            return found
        if not variance and self.interest.i > 0: # (1) twin insurance shortcut
            A = self._A_x(x, s=s, b=b, t=t, endowment=b, discrete=discrete, 
                          depth=self._depth)
            if A is not None:
                self._h(self._depth, "Annuity twin: a = (1 - A) / d")
                print(self._h, end='')
                return self.annuity_twin(A=A, discrete=discrete)
        a = super().temporary_annuity(x, s=s, b=b, t=t, discrete=discrete,
                                      variance=variance)
        if a is not None:
            print(self._h, end='')
            return a

    def deferred_annuity(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                         u: int = 0, b: int = 1, discrete: bool = True) -> float:
        """Compute deferred annuity with recursion first: u|a_x:t"""
        self._h = self._H(f"[ Deferred Annuity: {(str(u)+'_') if u else ''}"
                          + f"a_{x+s}{(':'+str(t)) if t >=0 else ''} ]")
        a = self._a_x(x, s=s, b=b, t=t, u=u, 
                          discrete=discrete, depth=self._depth)
        if a is not None:
            print(self._h, end='')
            return a
        a = self._a_x(x, s=s, b=b, t=self.add_term(u, t),  
                      discrete=discrete, depth=self._depth)
        a_t = self._a_x(x, s=s, b=b, t=u,  
                        discrete=discrete, depth=self._depth)
        if a is not None and a_t is not None:
            print(self._h, end='')
            return a - a_t
        return super().deferred_annuity(x, s=s, b=b, t=t, discrete=discrete)


if __name__ == "__main__":
    from actuarialmath.constantforce import ConstantForce
    from actuarialmath.policyvalues import Policy
    
    print("SOA Question 6.48:  (A) 3195")
    life = Recursion(interest=dict(i=0.06), depth=5)
    x = 0
    life.set_p(0.95, x=x, t=5)
    life.set_q(0.02, x=x+5)
    life.set_q(0.03, x=x+6)
    life.set_q(0.04, x=x+7)
    a = 1 + life.E_x(x, t=5)
    A = life.deferred_insurance(x, u=5, t=3)
    P = life.gross_premium(A=A, a=a, benefit=100000)
    print(P)
    print()
    
    print("SOA Question 6.40: (C) 116 ")
    # - standard formula discounts/accumulates by too much (i should be smaller)
    x = 0
    life = Recursion(interest=dict(i=0.06)).set_a(7, x=x+1).set_q(0.05, x=x)
    a = life.whole_life_annuity(x)
    A = 110 * a / 1000
    print(a, A)
    life = Recursion(interest=dict(i=0.06)).set_A(A, x=x).set_q(0.05, x=x)
    A1 = life.whole_life_insurance(x+1)
    P = life.gross_premium(A=A1 / 1.03, a=7) * 1000
    print(P)
    print()
    
    print("SOA Question 6.17:  (A) -30000")
    x = 0
    life = ConstantForce(mu=0.1, interest=dict(i=0.08))
    A = life.endowment_insurance(x, t=2, b=100000, endowment=30000)
    a = life.temporary_annuity(x, t=2)
    P = life.gross_premium(a=a, A=A)
    print(A, a, P)

    life1 = Recursion(interest=dict(i=0.08))\
            .set_q(life.q_x(x, t=1) * 1.5, x=x, t=1)\
            .set_q(life.q_x(x+1, t=1) * 1.5, x=x+1, t=1)
    policy = Policy(premium=P * 2, benefit=100000, endowment=30000)
    L = life1.gross_policy_value(x, t=0, n=2, policy=policy)
    print(L)
    print()

    print(Recursion.help())
    
