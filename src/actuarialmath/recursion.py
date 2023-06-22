"""Recursion - Applies recursion, shortcut and actuarial formulas

MIT License. Copyright 2022-2023 Terence Lim
"""
from typing import Tuple
import matplotlib.pyplot as plt
from actuarialmath import Reserves

class Recursion(Reserves):
    """Solve by appling recursive, shortcut and actuarial formulas repeatedly

    Args:
      depth : maximum depth of recursions (default is 3)
      verbose : whether to echo recursion steps (True, default)

    Notes:
      7 types of information can be loaded and calculated in recursions:

      - 'q' : (deferred) probability (x) dies in t years
      - 'p' : probability (x) survives t years
      - 'e' : (temporary) expected future lifetime, and moments
      - 'A' : deferred, term, endowment or whole life insurance, and moments
      - 'IA' : decreasing life insurance of t years
      - 'DA' : increasing life insurance of t years
      - 'a' : deferred, temporary or whole life annuity of t years, and moments

    Examples:
      >>> x = 0
      >>> life = Recursion().set_interest(i=0.06).set_a(7, x=x+1).set_q(0.05, x=x)
      >>> a = life.whole_life_annuity(x)
      >>> A = 110 * a / 1000
      >>> print(a, A)
      >>> life = Recursion().set_interest(i=0.06).set_A(A, x=x).set_q(0.05, x=x)
      >>> A1 = life.whole_life_insurance(x+1)
      >>> P = life.gross_premium(A=A1 / 1.03, a=7) * 1000
    """

    def __init__(self, depth: int = 5, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.db = {}
        self.maxdepth = depth
        _Blog._verbose = verbose

    #
    # helpers to store given input values
    #
    def _db_key(self, *args, **kwargs) -> Tuple:
        """Generate a unique key representing values of given arguments"""
        assert args and kwargs
        return tuple(list(args) + sorted(kwargs.items()))

    def _db_put(self, key: Tuple, value: float | None) -> "Recursion":
        """Store the item's key and value; or remove if value is None
        Args:
          key : key of the item
          value : value to store for item
        """
        if value is None and key in self.db:
            self.db.pop(key)
        else:
            self.db[key] = value
        return self

    def _db_print(self):
        """Display the stored keys and values"""
        for k in sorted(self.db.keys()):
            print(k, self.db[k])

    #
    # Formulas for Mortality: u|t_q_x
    #
    def _get_q(self, x: int, s: int = 0, t: int = 1,
               u: int = 0) -> float | None:
        """Get mortality rate from key-value store

        Args:
          x : age of selection
          s : years after selection
          u : survive u years, then...
          t : death within next t years        
        """
        key = self._db_key('q', x=x+s, u=u, t=t)
        return self.db.get(key, None)

    def set_q(self, val: float, x: int, s: int = 0, t: int = 1, 
              u: int = 0) -> "Recursion":
        """Set mortality rate u|t_q_[x+s] to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          u : survive u years, then...
          t : death within next t years        
        """
        return self._db_put(self._db_key('q', x=x+s, u=u, t=t), val)

    def _q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0, 
             depth: int = 1) -> float:
        """Helper to compute mortality from recursive and alternate formulas"""
        found = self._get_q(x, s=s, t=t, u=u)
        if found is not None:
            return found
        if t == 0:
            return 0
        if t < 0:
            return 1
        if u > 0:
            pu = self._p_x(x, s=s, t=u, depth=depth-1) #depth-1)
            qt = self._get_q(x, s=s+u, t=t)
            if pu is not None and qt is not None:
                self.blog(_Blog.q(x=x, s=s, t=t, u=u), '=',
                          _Blog.p(x=x, s=s, t=u), '*', _Blog.q(x=x, s=s+u, t=t),
                          depth=depth, rule="defer mortality")
                return pu * qt        # (1) u_p_x * t_q_x+u
            qu = self._get_q(x, s=s, t=u)
            qt = self._get_q(x, s=s, t=u+t)
            if qu is not None and qt is not None:
                self.blog(_Blog.q(x=x, s=s, t=t, u=u), '=',
                          _Blog.q(x=x, s=s, t=t+u), '-', _Blog.q(x=x, s=s, t=u),
                          depth=depth, rule="limit mortality")
                return qt - qu        # (2) u+t_q_x - u_q_x
        if depth <= 0:
            return None
        pu = self._p_x(x, s=s, t=u, depth=depth-1)
        pt = self._p_x(x, s=s, t=u+t, depth=depth-1)
        if pu is not None and pt is not None:
            self.blog(_Blog.q(x=x, s=s, t=t, u=u), '=',
                      _Blog.p(x=x, s=s, t=u), '-', _Blog.p(x=x, s=s, t=t+u),
                      depth=depth, rule="complement survival")
            return pu - pt            # (3) u_p_x - u+t_p_x


    def q_x(self, x: int, s: int = 0, t: int = 1, u: int = 0) -> float:
        self.blog = _Blog("Mortality",
                         _Blog.q(x=x, s=s, t=t, u=u),
                         levels=self.maxdepth)
        """Compute mortality rate by calling recursion helper"""
        q = self._q_x(x, s=s, t=t, u=u, depth=self.maxdepth)
        if q is not None:
            self.blog.write()
        return q

    #
    # Formulas for Survival: t_p_x
    #
    def _get_p(self, x: int, s: int = 0, t: int = 1) -> float | None:
        """Get survival probability from key-value store

        Args:
          x : age of selection
          s : years after selection
          t : survives next t years
        """
        if t == 0:
            return 1
        if t < 0:
            return 0
        key = self._db_key('p', x=x+s, t=t)
        return self.db.get(key, None)

    def set_p(self, val: float, x: int, s: int = 0, t: int = 1) -> "Recursion":
        """Set survival probability t_p_[x+s] to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          t : survives next t years
        """
        return self._db_put(self._db_key('p', x=x+s, t=t), val)

    def _p_x(self, x: int, s: int = 0, t: int = 1, depth: int = 1) -> float:
        """Helper to compute survival from recursive and alternate formulas"""
        found = self._get_p(x, s=s, t=t)
        if found is not None:
            return found
        found = self._get_q(x, s=s, t=t)  
        if found is not None:
            self.blog(_Blog.p(x=x, s=s, t=t), '= 1 -', _Blog.q(x=x, s=s, t=t),
                      depth=depth, rule='complement of mortality')
            return 1 - found  # (1) complement of q_x
        if depth <= 0:
            return None
        
        # (2a) inverse chain rule: p_x(t) = p_x-1(t+1) / p_x-1 
        found = self._p_x(x, s=s-1, t=t+1, depth=depth-1)
        p = self._p_x(x, s=s-1, t=1, depth=depth-1)
        if found is not None and p is not None:
            self.blog(_Blog.p(x=x, s=s, t=t), '=',
                      _Blog.p(x=x, s=s-1, t=t+1), '/', _Blog.p(x=x, s=s-1, t=1),
                      depth=depth, rule="survival chain rule")
            return found / p
        
        # (2b) inverse chain rule: p_x(t) = p_x(t+1) / p_x+t 
        found = self._p_x(x, s=s, t=t+1, depth=depth-1)
        p = self._p_x(x, s=s+t, t=1, depth=depth-1)
        if found is not None and p is not None:
            self.blog(_Blog.p(x=x, s=s, t=t), '=',
                      _Blog.p(x=x, s=s, t=t+1), '/', _Blog.p(x=x, s=s+t, t=1),
                      depth=depth, rule="survival chain rule")
            return found / p
        if t > 1:
            # (3a) chain rule: p_x(t) = p_x * p_x+1(t-1)
            found = self._p_x(x, s=s+1, t=t-1, depth=depth-1)
            p = self._p_x(x, s=s, t=1, depth=depth-1)
            if found is not None and p is not None:
                self.blog(_Blog.p(x=x, s=s, t=t), '=',
                          _Blog.p(x=x, s=s+1, t=t-1), '*', _Blog.p(x=x, s=s, t=1),
                          depth=depth, rule="survival chain rule")
                return found * p

            # (3b) chain rule: p_x(t) = p_x+t-1 * p_x(t-1)
            found = self._p_x(x, s=s, t=t-1, depth=depth-1)
            p = self._p_x(x, s=s+t-1, t=1, depth=depth-1)
            if found is not None and p is not None:
                self.blog(_Blog.p(x=x, s=s, t=t), '=',
                          _Blog.p(x=x, s=s, t=t-1), '*', _Blog.p(x=x, s=s+t-1, t=1),
                          depth=depth, rule="survival chain rule")
                return found * p

        if t == 1:
            E = self._E_x(x, s=s, t=1, depth=depth-1)
            if E is not None:
                self.blog(_Blog.p(x=x, s=s, t=1), '=',
                          _Blog.E(x=x, s=s, t=1), "/v",
                          depth=depth, rule="one-year endowment")
                return E / self.interest.v

            # (4a) annuity recursion: p_x = [a_x(t) - 1] / [v a_x+1(t-1)
            for _t in [self.WHOLE, 2, 3]:  # consider only WL and 2-term
                a = self._a_x(x, s=s, t=_t, depth=depth-1)
                a1 = self._a_x(x, s=s+1, t=self.add_term(_t, -1), depth=depth-1)
                if a is not None and a1 is not None:
                    self.blog(_Blog.p(x=x, s=s, t=1), '= [',
                              _Blog.a(x=x, s=s, t=_t), '- 1 ] / [ v *',
                              _Blog.a(x=x, s=s+1, t=_t-1), ']',
                              depth=depth, rule="annuity recursion")
                    return (a - 1) / (self.interest.v * a1)

            
            # (4b) insurance recursion: p_x = [v - A_x(t)] / [v (1 - A_x+1(t-1))]
            for _t in [self.WHOLE, 2, 3]:  # consider only WL and 2-term
                A = self._A_x(x, s=s, t=_t, depth=depth-1)
                A1 = self._A_x(x, s=s+1, t=self.add_term(_t, -1), depth=depth-1)
                if A is not None and A1 is not None:
                    self.blog(_Blog.p(x=x, s=s, t=1), '= [ v -',
                              _Blog.A(x=x, s=s, t=_t), '] / [v * [ 1 -',
                              _Blog.A(x=x, s=s+1, t=_t-1), ']]',
                              depth=depth, rule="insurance recursion")
                    return (self.interest.v - A)/(self.interest.v * (1 - A1))

    def p_x(self, x: int, s: int = 0, t: int = 1) -> float:
        """Compute survival probability by calling recursion helper

        Args:
          x : age of selection
          s : years after selection
          t : survives at least t years
        """
        self.blog = _Blog("Survival",
                         _Blog.p(x=x, s=s, t=t),
                         levels=self.maxdepth)
        p = self._p_x(x, s=s, t=t, depth=self.maxdepth)
        if p is not None:
            self.blog.write()
        return p

    #
    # Formulas for Expected Future Lifetime: e_x
    #
    def _get_e(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
               curtate: bool = False, moment: int = 1) -> float | None:
        """Get expected future lifetime from key-value store

        Args:
          x : age of selection
          s : years after selection
          t : limit of expected future lifetime
          curtate : curtate (True) or complete expectation (False)
          moment : first or second moment of expected future lifetime
        """
        key = self._db_key('e', x=x+s, t=t, curtate=curtate, moment=moment)
        return self.db.get(key, None)

    def set_e(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE, 
              curtate: bool = False, moment: int = 1) -> "Recursion":
        """Set expected future lifetime e_[x+s]:t to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          t : limit of expected future lifetime
          curtate : curtate (True) or complete expectation (False)
          moment : first or second moment of expected future lifetime
        """
        return self._db_put(self._db_key('e', x=x+s, t=t, moment=moment,
                                         curtate=curtate), val)

    def _e_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
            curtate: bool = False, moment: int = 1, 
            depth: int = 1) -> float | None:
        """Helper to compute from recursive and alternate formulas"""
        found = self._get_e(x, s=s, t=t, curtate=curtate, moment=moment)
        if found is not None:
            return found
        if depth <= 0:
            return None
        if moment == 1:
            if t > 0:  
                p_t = self._p_x(x, s=s, t=t)
                if t == 1 and curtate:
                    self.blog(_Blog.e(x=x, s=s, t=1, curtate=curtate), '=',
                              _Blog.p(x=x, s=s, t=1),
                              #f"e_{x+s}:1",
                              depth=depth, rule='1-year curtate shortcut')
                    return p_t   # (1) if curtate and t=1: e_x:1 = p_x 
                if t > 1:
                    e = self._e_x(x, s=s, t=Reserves.WHOLE, curtate=curtate, 
                                  moment=1, depth=depth-1)
                    e_t = self._e_x(x, s=s+t, t=Reserves.WHOLE, curtate=curtate,
                                    moment=1, depth=depth-1)
                    if e is not None and e_t is not None and p_t is not None:
                        self.blog(_Blog.e(x=x, s=s, t=t), '=', _Blog.e(x=x, s=s), '-',
                                  _Blog.p(x=x, s=s), '*', _Blog.e(x=x, s=s+t),
                                  #"e_{x+s}:{t}: e_x - p_x e_x+t",
                                  depth=depth, rule="temporary lifetime")
                        return e - p_t*e_t  # (2) temporary = e_x - t_p_x e_x+t
            e = self._e_x(x, s=s-1, t=self.add_term(t, 1), curtate=curtate,
                          moment=1, depth=depth-1)
            e1 = self._e_x(x, s=s-1, t=1, curtate=curtate, moment=1,
                          depth=depth-1)
            p = self._p_x(x, s=s-1)
            if e is not None and e1 is not None and p is not None:
                #_t = "" if t < 0 else ":" + str(t)
                #msg = f"forward e_{x+s}{_t} = e_{x+s}:1 + p_{x+s} e_{x+s+1}"
                self.blog(_Blog.e(x=x, s=s, t=t), '=', _Blog.e(x=x, s=s, t=t), '+',
                          _Blog.p(x=x, s=s), '*', _Blog.e(x=x, s=s+t),
                          depth=depth, rule='lifetime forward recursion')
                return (e - e1) / p # (3) forward: (e_x-1 - e_x-1:1) / p_x-1
            e = self._e_x(x, s=s, t=1, curtate=curtate,
                          moment=1, depth=depth-1)
            e_t = self._e_x(x, s=s+1, t=self.add_term(t, -1), curtate=curtate,
                            moment=1, depth=depth-1)
            p = self._p_x(x, s=s)
            if e is not None and e_t is not None and p is not None:
                self.blog(_Blog.e(x=x, s=s, t=t), '=', _Blog.e(x=x ,s=s, t=1), '+',
                          _Blog.p(x=x, s=s), '*', _Blog.e(x=x, s=s+1, t=t-1),
                          #f"backward: e_x:1 + p_x e_x+1:{t}",
                          depth=depth, rule='lifetime backward recursion')
                return e + p * e_t # (4) backward: e_x:1 + p_x e_x+1:t-1


    def e_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
            curtate: bool = False, moment: int = 1) -> float:
        """Compute expected future lifetime by calling recursion helper

        Args:
          x : age of selection
          s : years after selection
          t : limited at t years
          curtate : whether curtate (True) or complete (False) lifetime
          moment : whether to compute first (1) or second (2) moment
        """
        self.blog = _Blog("Lifetime",
                         _Blog.e(x=x, s=s, t=t, moment=moment, curtate=curtate),
                         levels=self.maxdepth)
        e = self._e_x(x, s=s, t=t, curtate=curtate, moment=moment,
                      depth=self.maxdepth)
        if e is not None:
            self.blog.write()
            return e

    #
    # Formulas for Pure Endowment: t_E_x
    #
    def _get_E(self, x: int, s: int = 0, t: int = 1, 
               endowment: int = 1, moment: int = 1) -> float | None:
        """Get pure endowment from key-value store

        Args:
          x : age of selection
          s : years after selection
          t : death within next t years
          endowment : endowment value
          moment : first or second moment of pure endowment
        """
        key = self._db_key('E', x=x+s, t=t, moment=moment)
        val = self.db.get(key, None)
        if val is not None:
            return val * endowment   # stored with benefit=1

    def set_E(self, val: float, x: int, s: int = 0, t: int = 1, 
              endowment: int = 1, moment: int = 1) -> "Recursion":
        """Set pure endowment t_E_[x+s] to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          t : death within next t years
          endowment : endowment value
          moment : first or second moment of pure endowment
        """
        val /= endowment   # store with benefit=1
        return self._db_put(self._db_key('E', x=x+s, t=t, moment=moment), val)

    def _E_x(self, x: int, s: int = 0, t: int = 1, endowment: int = 1, 
             moment: int = 1, depth: int = 1) -> float:
        """Helper to compute from recursive and alternate formulas"""
        E = self._get_E(x, s=s, t=t, endowment=endowment, moment=moment)
        if E is not None:
            return E
        if t < 0:     # t infinite => EPV(t) = 0
            return 0
        if t == 0:    # t = 0 => EPV(0) = endowment**moment
            return endowment**moment
        if moment > 1:
            E = self._E_x(x, s=s, endowment=endowment, depth=depth)
            if E:  # (1) Shortcut: 2E_x = v E_x
                self.blog(_Blog.E(x=x, s=s, t=t, moment=moment),
                          f"= v(t={t})"+_Blog.m(moment), '*', _Blog.E(x=x, s=s, t=t),
                          depth=depth, rule='moments of pure endowment')
                return E * self.interest.v**(moment-1) 
        p = self._p_x(x, s=s, t=t, depth=depth-1)  # depth-1)
        if p is not None:   # (2) E_x = p_x * v
            #msg = f"pure endowment {t}_E_{x+s} = {t}_p_{x+s} * v^{t}"
            self.blog(_Blog.E(x=x, s=s, t=t), '=', _Blog.p(x=x, s=s, t=t),
                      f"* v(t={t})"+_Blog.m(moment),
                      depth=depth, rule='pure endowment')
            return p * (endowment * self.interest.v_t(t))**moment
        if depth <= 0:
            return None

        At = self._A_x(x, s=s, t=t, moment=moment, b=endowment, endowment=0,
                       depth=depth-1)  #depth-1)
        A = self._A_x(x, s=s, t=t, b=endowment, endowment=endowment, 
                      moment=moment, depth=depth-1)        
        if A is not None and At is not None:
            self.blog(_Blog.E(x=x, s=s, t=t), '=',
                      _Blog.A(x=x, s=s, t=t, endowment=endowment), '-',
                      _Blog.A(x=x, s=s, t=t, endowment=0),
                      #f"endowment - term insurance = {t}_E_{x+s}",
                      depth=depth, rule='endowment insurance minus term')
            return A - At  # (3) endowment insurance - term (helpful SULT)

        E = self._E_x(x, s=s, moment=moment, depth=depth-1)
        Et = self._E_x(x, s=s+1, t=t-1, moment=moment, depth=depth-1)
        if E is not None and Et is not None:
            msg = f"chain Rule: {t}_E_{x+s} = E_{x+s} * {t-1}_E_{x+s+1}"
            self.blog(_Blog.E(x=x, s=s, t=t, moment=moment), '=',
                      _Blog.E(x=x, s=s, t=1, moment=moment), '*',
                      _Blog.E(x=x, s=s+1, t=t-1, moment=moment), '*',
                      _Blog.m(moment, endow=endowment),
                      depth=depth, rule='pure endowment chain rule')
            return E * Et * endowment**moment # (4) chain rule

    def E_x(self, x: int, s: int = 0, t: int = 1, 
            endowment: int = 1, moment: int = 1) -> float:
        """Compute pure endowment by calling recursion helper

        Args:
          x : age of selection
          s : years after selection
          t : term of pure endowment
          endowment : amount of pure endowment
          moment : compute first or second moment
        """
        self.blog = _Blog("Pure Endowment",
                         _Blog.E(x=x, s=s, t=t, moment=moment, endowment=endowment),
                         levels=self.maxdepth)
        if moment == self.VARIANCE:  # Bernoulli shortcut for variance
            found = self._get_E(x, s=s, t=t, endowment=endowment, moment=moment)
            if found is not None:
                return found
            t_p_x = self.p_x(x, s=s, t=t)
            return (endowment * self.interest.v_t(t))**2 * t_p_x * (1-t_p_x)
        found = self._E_x(x, s=s, t=t, endowment=endowment, moment=moment,
                          depth=self.maxdepth)
        if found is not None:
            self.blog.write()
            return found

    #
    # Formulas for Increasing Insurance: IA_x:t
    #
    def _get_IA(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                b: int = 1, discrete: bool = True) -> float | None:
        """Get increasing insurance from key-value store

        Args:
          x : age of selection
          s : years after selection
          t : term of increasing insurance
          b : benefit after year 1
          discrete : discrete or continuous increasing insurance
        """
        key = self._db_key('IA', x=x+s, t=t, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * b   # stored with benefit=1

    def set_IA(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, discrete: bool = True) -> "Recursion":
        """Set increasing insurance IA_[x+s]:t to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          t : term of increasing insurance
          b : benefit after year 1
          discrete : discrete or continuous increasing insurance
        """
        val /= b   # store with benefit=1
        return self._db_put(self._db_key('IA', x=x+s, t=t, 
                                         discrete=discrete), val)

    def _IA_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, b: int = 1,
              discrete: bool = True, depth: int = 1) -> float | None:
        """Helper to compute from recursive and alternate formulas"""
        if t == 0:
            return 0
        found = self._get_IA(x=x, s=s, t=t, b=b, discrete=discrete)
        if found is not None:
            return found
        if depth <= 0:
            return None

        if t > 0:  # decreasing must be term insurance
            A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=depth-1)
            n = t + int(discrete)
            DA = self._DA_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=depth-1)
            if A is not None and DA is not None:
                self.blog(_Blog.IA(x=x, s=s, t=t), f'= {n}',
                          _Blog.A(x=x, s=s, t=t), '-', _Blog.DA(x=x, s=s, t=t),
                    #f"identity IA_{x+s}:{t}: ({n})A - DA",
                          depth=depth, rule='varying insurance identity')
                return A * n - DA  # (1) Identity with term and decreasing

        A = self._A_x(x=x, s=s, t=1, b=b, discrete=discrete, depth=depth-1)
        IA = self._IA_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, depth=depth-1)
        p = self._p_x(x, s=s, t=1, depth=depth-1)   # FIXED t=1
        if A is not None and IA is not None and p is not None:
            self.blog(_Blog.IA(x=x, s=s, t=t), '=', _Blog.A(x=x, s=s, t=t), '+',
                      _Blog.p(x=x, s=s), f"* v(t={t}) *", _Blog.IA(x=x, s=s+1, t=t-1),
                      #f"backward IA_{x+s}:{t}: A + IA_{x+s+1}:{t-1}",
                      depth=depth, rule='backward recursion')
            return A + p * self.interest.v * IA  # (2) backward recursion

    def increasing_insurance(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                             b: int = 1, discrete: bool = True) -> float:
        """Compute increasing insurance with recursive helper

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit in first year
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        self.blog = _Blog("Increasing Insurance",
                         _Blog.IA(x=x, s=s, t=t, b=b, discrete=discrete),
                         levels=self.maxdepth)
        IA = self._IA_x(x, s=s, b=b, t=t, discrete=discrete, depth=self.maxdepth)
        if IA is not None:
            self.blog.write()
            return IA
        IA = super().increasing_insurance(x, s=s, b=b, t=t, discrete=discrete)
        if IA is not None:
            self.blog.write()
            return IA

    #
    # Formulas for Decreasing insurance: DA_x:t
    #
    def _get_DA(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                b: int = 1, discrete: bool = True) -> float | None:
        """Get decreasing insurance from key-value store

        Args:
          x : age of selection
          s : years after selection
          t : term of decreasing insurance
          b : benefit after year 1
          discrete : discrete or continuous decreasing insurance
        """
        key = self._db_key('DA', x=x+s, t=t, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * b   # stored with benefit=1

    def set_DA(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, discrete: bool = True) -> "Recursion":
        """Set decreasing insurance DA_[x+s]:t to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          t : term of decreasing insurance
          b : benefit after year 1
          discrete : discrete or continuous decreasing insurance
        """
        val /= b     # store with benefit=1
        return self._db_put(self._db_key('DA', x=x+s, t=t, 
                                         discrete=discrete), val)

    def _DA_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, b: int = 1,
              discrete: bool = True, depth: int = 1) -> float | None:
        """Helper to compute from recursive and alternate formulas"""
        found = self._get_DA(x=x, s=s, t=t, discrete=discrete)
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
            self.blog(_Blog.DA(x=x, s=s, t=t), f'= {n}',
                      _Blog.A(x=x, s=s, t=t), '-', _Blog.IA(x=x, s=s, t=t),
                      depth=depth, rule='varying insurance identity')
            return A * n - IA  # (1) identity with term and decreasing

        DA = self._IA_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, depth=depth-1)
        p = self._p_x(x, s=s, depth=depth-1)
        if DA is not None and p is not None:
            #msg = f"backward DA_{x+s}:{t}: v(t q_{x+s} + p_{x+s} DA_{x+s+1}:{t-1})"
            self.blog(_Blog.DA(x=x, s=s, t=t), f"= v(t={t}) * t *", _Blog.q(x=x, s=s),
                      '+', _Blog.p(x=x, s=s), '*', _Blog.DA(x=x, s=s+1, t=t-1),
                      depth=depth, rule='backward recursion')
            return self.interest.v * ((1-p)*t + p*DA)  # (2) backward recursion

    def decreasing_insurance(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                             b: int = 1, discrete: bool = True) -> float:
        """Compute decreasing insurance by calling recursive helper first

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit in first year
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        self.blog = _Blog("Increasing Insurance",
                         _Blog.DA(x=x, s=s, t=t, b=b, discrete=discrete),
                         levels=self.maxdepth)
        if t == 0:
            return 0
        A = self._DA_x(x=x, s=s, t=t, b=b, discrete=discrete, depth=self.maxdepth)
        if A is not None:
            self.blog.write()
            return A
        A = super().decreasing_insurance(x, s=s, b=b, t=t, discrete=discrete)
        if A is not None:
            self.blog.write()
            return A

    #
    # Formulas for Insurance: A_x:t
    #
    def _get_A(self, x: int, s: int = 0, u: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, moment: int = 1, endowment: int = 0, 
               discrete: bool = True) -> float | None:
        """Get insurance from key-value store

        Args:
          x : age of selection
          s : years after selection
          u : defer u years
          t : term of insurance
          endowment : endowment amount
          discrete : discrete (True) or continuous (False) insurance
          moment : first or second moment of insurance
        """
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
        key = self._db_key('A', x=x+s, u=u, t=t, moment=moment, 
                           endowment=endowment, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * scale   # stored with benefit=1


    def set_A(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
              u: int = 0, b: int = 1, moment: int = 1, endowment: int = 0, 
              discrete: bool = True) -> "Recursion":
        """Set insurance u|_A_[x+s]:t to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          u : defer u years
          t : term of insurance
          endowment : endowment amount
          discrete : discrete (True) or continuous (False) insurance
          moment : first or second moment of insurance
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
        return self._db_put(self._db_key('A', x=x+s, t=t, u=u,
                                         moment=moment, endowment=endowment, 
                                         discrete=discrete), val)

    def _A_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, u: int = 0,
             b: int = 1, discrete: bool = True, endowment: int = 0,
             moment: int = 1, depth: int = 1) -> float | None:
        """Helper to compute from recursive and alternate formulas"""
        
        if endowment == b and t == 1 and discrete: # 1-year endow ins
            return (self.interest.v_t(1) * endowment)**moment
        found = self._get_A(x=x, s=s, t=t, b=b, u=u, discrete=discrete, 
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
                #msg = f"backward deferred {u}_A_{x+s}: {u}_E * A_{x+s+u}"
                self.blog(_Blog.A(x=x, s=s, t=t, u=u, b=b, moment=moment), '=',
                          _Blog.E(x=x, s=s, moment=moment), '*',
                          _Blog.A(x=x, s=s+1, t=t, u=u-1, moment=moment),
                          depth=depth, rule='backward recursion')
                return E * A

            A = self._A_x(x, s=s-1, t=t, b=b, u=u+1, discrete=discrete, 
                          moment=moment, endowment=endowment, depth=depth-1)
            E = self._E_x(x, s=s-1, t=1, moment=moment, depth=depth-1)
            if A is not None and E is not None: # (1b) forward recursion
                msg = f"forward deferred {u}_A_{x+s}: {u+1}A_{x+s-1} / E"
                self.blog(_Blog.A(x=x, s=s, t=t, u=u, b=b, moment=moment), '=',
                          _Blog.A(x=x, s=s-1, t=t, u=u+1, moment=moment), '/',
                          _Blog.E(x=x, s=s-1, moment=moment),
                          depth=depth, rule='forward recursion')
                return A / E
            return None

        if endowment > 0: # (2a) endowment = term + E_x * endowment
            A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, 
                          moment=moment, depth=depth-1)
            E_x = self._E_x(x=x, s=s, t=t, moment=moment, 
                            endowment=endowment, depth=depth-1)
            if A is not None and E_x is not None:
                # f"term + pure insurance = A_{x+s}:{t}",
                self.blog(_Blog.A(x=x, s=s, t=t, endowment=endowment, moment=moment),
                          '=', _Blog.A(x=x, s=s, t=t, moment=moment), '+',
                          _Blog.E(x=x, s=s, t=t, endowment=endowment, moment=moment),
                          depth=depth, rule='term plus pure endowment')
                return A + E_x
        else:             # (2b) term = endowment insurance - E_x * endowment
            A = self._A_x(x=x, s=s, t=t, b=b, discrete=discrete, 
                          moment=moment, endowment=b, depth=depth-1)
            E_x = self._E_x(x=x, s=s, t=t, moment=moment, endowment=b, 
                            depth=depth-1)
            if A is not None and E_x is not None:
                #msg = f"endowment insurance - pure endowment = A_{x+s}^1:{t}"
                self.blog(_Blog.A(x=x, s=s, t=t, moment=moment), '=',
                          _Blog.A(x=x, s=s, t=t, moment=moment, endowment=b), '-',
                          _Blog.E(x=x, s=s, t=t, endowment=b, moment=moment),
                          depth=depth, rule='endowment insurance - pure')
                return A - E_x

        if not discrete:  # recursions for discrete insurance
            return None
        if t == 1:  # special cases for discrete one-year insurance
            if endowment == b:  # (3a) discrete one-year endowment insurance
                self.blog(_Blog.A(x=x, s=s, t=1, endowment=endowment,
                                 moment=moment), f"= v"+_Blog.m(moment),
                          "*", _Blog.m(moment, endow=endowment),
                          depth=depth, rule='one-year endowment insurance')
                return (self.interest.v * endowment)**moment
            p = self._p_x(x, s=s, t=1)
            if p is not None:  # (3b) one-year discrete insurance
                self.blog(_Blog.A(x=x, s=s, t=t, moment=moment, endowment=endowment),
                          f"= v"+_Blog.m(moment), "*",
                          _Blog.q(x=x, s=s), f"*", "v"*_Blog.m(moment), "+",
                          _Blog.p(x=x, s=s), f"*".
                          _Blog.m(moment, endow=endowment),
                          #f"discrete 1-year insurance: A_{x+s}:1 = qv",
                          depth=depth, rule='one-year discrete insurance')
                return (self.interest.v**moment 
                        * ((1 - p) * b**moment + p * endowment**moment))
    
        A = self._A_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, 
                      discrete=discrete, moment=moment,
                      endowment=endowment, depth=depth-1)
        p = self._p_x(x, s=s, t=1, depth=depth-1) # (4) backward recursion
        if A is not None and p is not None:
            self.blog(_Blog.A(x=x, s=s, t=t, b=b, moment=moment),
                      f"= v"+_Blog.m(moment), "* [",
                      _Blog.q(x=x,s=s), f"* b"+_Blog.m(moment), "+",
                      _Blog.p(x=x, s=s), '*',
                      _Blog.A(x=x, s=s+1, t=t-1, b=b, moment=moment), ']',
                      #f"backward: A_{x+s} = qv + pvA_{x+s+1}",
                      depth=depth, rule='backward recursion')
            return self.interest.v_t(1)**moment * ((1 - p)*b**moment + p*A)

        A = self._A_x(x=x, s=s-1, t=self.add_term(t, 1), b=b, u=u, 
                      discrete=discrete, moment=moment, 
                      endowment=endowment, depth=depth-1)
        p = self._p_x(x, s=s-1, t=1, depth=depth-1)
        if A is not None and p is not None:  # (5) forward recursion
            self.blog(_Blog.A(x=x, s=s, t=t, b=b, moment=moment), '= [',
                      _Blog.A(x=x, s=s-1, t=t+1, b=b, moment=moment),
                      f"/v"+_Blog.m(moment), "-", _Blog.q(x=x,s=s),
                      f"* b"+_Blog.m(moment), "] /", _Blog.p(x=x, s=s),
                      #f"forward: A_{x+s} = (A_{x+s-1}/v - q) / p",
                      depth=depth, rule='forward recursion')
            return (A/self.interest.v_t(1)**moment - (1-p)*b**moment) / p

    def whole_life_insurance(self, x: int, s: int = 0, b: int = 1, 
                             discrete: bool = True, moment: int = 1) -> float:
        """Compute whole life insurance A_x by calling recursion helper and twin

        Args:
          x : age of selection
          s : years after selection
          b : amount of benefit
          moment : compute first or second moment
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        self.blog = _Blog("Whole Life Insurance",
                         _Blog.A(x=x, s=s, t=Reserves.WHOLE, b=b, u=0,
                                endowment=0, moment=moment, discrete=discrete),
                         levels=self.maxdepth)
        found = self._A_x(x, s=s, b=b, moment=moment, discrete=discrete,
                          depth=self.maxdepth)
        if found is not None:
            self.blog.write()
            return found
        if moment == 1 and self.interest.i > 0:  # (1) twin annuity
            a = self._a_x(x, s=s, b=b, discrete=discrete, depth=self.maxdepth)
            if a is not None:
                self.blog("A_x = 1 - d * a_x",
                          depth=self.maxdepth, rule='annuity twin')
                self.blog.write()
                return self.insurance_twin(a=a, discrete=discrete)
        A = super().whole_life_insurance(x, s=s, b=b, discrete=discrete,
                                         moment=moment)
        if A is not None:
            self.blog.write()
            return A

    def term_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1, 
                       moment: int = 1, discrete: bool = True) -> float:
        """Compute term life insurance A_x:t^1 by calling recursion helper: 

        Args:
          x : age of selection
          s : years after selection
          t : term of insurance
          b : amount of benefit
          moment : compute first or second moment
          discrete : benefit paid year-end (True) or moment of death (False)
        """
        self.blog = _Blog("Term Insurance",
                         _Blog.A(x=x, s=s, t=t, b=b, u=0, discrete=discrete,
                                 endowment=0, moment=moment),
                         levels=self.maxdepth)
        found = self._A_x(x, s=s, b=b, t=t, moment=moment, discrete=discrete,
                          depth=self.maxdepth)
        if found is not None:
            self.blog.write()
            return found
        A = super().term_insurance(x, s=s, b=b, t=t, discrete=discrete,
                                   moment=moment)
        if A is not None:
            self.blog.write()
            return A

    def deferred_insurance(self, x: int, s: int = 0, b: int = 1, u: int = 0, 
                           t: int = Reserves.WHOLE, moment: int = 1, 
                           discrete: bool = True) -> float:
        """Compute deferred life insurance u|A_x:t^1 by calling recursion helper: """
        self.blog = _Blog("Deferred Insurance",
                         _Blog.A(x=x, s=s, t=t, b=b, u=u, discrete=discrete,
                                endowment=0, moment=moment),
                         levels=self.maxdepth)
        A = self._get_A(x=x, s=s, t=t, b=b, u=u, discrete=discrete, 
                        moment=moment)
        if A is not None:
            self.blog.write()
            return A
        A = super().deferred_insurance(x, s=s, b=b, t=t, u=u, 
                                       discrete=discrete, moment=moment)
        if A is not None:
            self.blog.write()
            return A
    
    def endowment_insurance(self, x: int, s: int = 0, t: int = 1, b: int = 1,
                            endowment: int = -1, moment: int = 1,
                            discrete: bool = True) -> float:
        """Compute endowment insurance u|A_x:t by calling recursion helper: """
        self.blog = _Blog("Endowment Insurance",
                         _Blog.A(x=x, s=s, t=t, b=b, u=0, discrete=discrete,
                                endowment=endowment, moment=moment),
                         levels=self.maxdepth)
        assert t >= 0
        if endowment < 0:
            endowment = b
        found = self._A_x(x, s=s, b=b, t=t, moment=moment, discrete=discrete,
                          endowment=endowment, depth=self.maxdepth)
        if found is not None:
            self.blog.write()
            return found
        if moment == 1 and endowment == b and self.interest.i > 0:
            a = self._a_x(x, s=s, b=b, t=t, moment=moment, discrete=discrete,
                          depth=self.maxdepth)
            if a is not None:   # twin insurance
                self.blog.write()
                return self.insurance_twin(a=a, discrete=discrete)
        A = super().endowment_insurance(x, s=s, b=b, t=t, discrete=discrete,
                                        moment=moment, endowment=endowment)
        if A is not None:
            self.blog.write()
            return A

    #
    # Formulas for Annuties: a_x:t
    #
    def _get_a(self, x: int, s: int = 0, u: int = 0, t: int = Reserves.WHOLE,
               b: int = 1, variance: bool = False, 
               discrete: bool = True) -> float | None:
        """Get annuity from key-value store

        Args:
          x : age of selection
          s : years after selection
          u : defer u years
          t : term of annuity
          b : benefit amount
          discrete : whether annuity due (True) or continuous (False)
          variance : whether first moment (False) or variance (True)
        """
        key = self._db_key('a', x=x+s, u=u, t=t, 
                           variance=variance, discrete=discrete)
        val = self.db.get(key, None)
        if val is not None:
            return val * b    # stored with benefit=1

    def set_a(self, val: float, x: int, s: int = 0, t: int = Reserves.WHOLE,
              u: int = 0, b: int = 1, variance: bool = False, 
              discrete: bool = True) -> "Recursion":
        """Set annuity u|_a_[x+s]:t to given value

        Args:
          val : value to set
          x : age of selection
          s : years after selection
          u : defer u years
          t : term of annuity
          b : benefit amount
          discrete : whether annuity due (True) or continuous (False)
          variance : whether first moment (False) or variance (True)
        """
        val /= b    # store with benefit=1
        return self._db_put(self._db_key('a', x=x+s, t=t, u=u, 
                                         variance=variance, discrete=discrete), val)

    def _a_x(self, x: int, s: int = 0, t: int = Reserves.WHOLE, 
             u: int = 0, b: int = 1, discrete: bool = True, 
             variance: bool = False, depth: int = 1) -> float | None:
        """Helper to compute from recursive and alternate formulas"""
        if t == 1 and not u and discrete:
            self.blog(_Blog.a(x=x, s=s, t=1, discrete=discrete), '= 1',
                      depth=depth, rule='one-year discrete annuity')
            return b 
        if t == 0:
            return 0
        found = self._get_a(x=x, s=s, t=t, b=b, u=u, discrete=discrete,
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
                #msg = f"backward {u}_a_{x+s} = {u}_E * a_{x+s+u}"
                self.blog(_Blog.a(x=x, s=s, u=u, t=t), '=',
                          _Blog.a(x=x, s=s+1, t=t, u=u-1), '/', _Blog.E(x=x, s=s),
                          depth=depth, rule='backward deferred annuity')
                return E * found  # (1a) backward recusion

            found = self._a_x(x=x, s=s-1, t=t, b=b, u=u+1, discrete=discrete, 
                              depth=depth-1)         # FIXED u=u+1
            E = self._E_x(x, s=s-1, t=1, depth=depth-1)
            if found is not None and E is not None:  # (1b) forward
                #msg = f"forward: {u}_a_{x+s} = {u+1}_a_{x+s-1}/E_{x+s-1}"
                self.blog(_Blog.a(x=x, s=s, u=u, t=t), '=',
                          _Blog.a(x=x, s=s-1, t=t, u=u+1), '/', _Blog.E(x=x, s=s-1),
                          depth=depth, rule='forward deferred annuity')
                return found / E

        else:  # (2) Temporary and whole annuity
            found = self._a_x(x=x, s=s+1, t=self.add_term(t, -1), b=b, u=u, 
                              discrete=discrete, 
                              variance=variance, depth=depth-1)
            E = self._E_x(x, s=s, t=1, depth=depth-1)
            if found is not None and E is not None:  # (2a) backward
                #msg = (f"backward: a_{x+s}{'' if t < 0 else (':'+str(t))} = 1 + "
                #       + f"E_{x+s} a_{x+s+1}{'' if t < 0 else (':'+str(t-1))}")
                #_t = "" if t < 0 else f":{t-1}"
                self.blog(_Blog.a(x=x, s=s, t=t), '= 1 +',
                          _Blog.E(x=x, s=s, t=1), '*', _Blog.a(x=x, s=s+1, t=t-1),
                          depth=depth, rule='backward recursion')
                return b + E * found

            found = self._a_x(x=x, s=s-1, t=self.add_term(t, 1), b=b, u=u, 
                              discrete=discrete, depth=depth-1)
            E = self._E_x(x, s=s-1, t=1, depth=depth-1)
            if found is not None and E is not None:  # (2b) forward
                _t = "" if t < 0 else f":{t-1}"
                self.blog(_Blog.a(x=x, s=s, t=t), '= [',
                          _Blog.a(x=x, s=s-1), '- 1 ] /',
                          _Blog.E(x=x, s=s-1, t=1),
                    #f"forward: a_{x+s}{_t} = (a_{x+s-1} - 1)/E",
                          depth=depth, rule='forward recursion')
                return (found - b) / E

    def whole_life_annuity(self, x: int, s: int = 0, b: int = 1, 
                           variance: bool = False,
                           discrete: bool = True) -> float:
        """Compute whole life annuity a_x by calling recursion then twin first

        Args:
          x : age of selection
          s : years after selection
          b : annuity benefit amount
          variance (bool): return EPV (True) or variance (False)
          discrete : annuity due (True) or continuous (False)
        """
        self.blog = _Blog("Whole Life Annuity",
                         _Blog.a(x=x, s=s, t=Reserves.WHOLE, b=b, u=0,
                                discrete=discrete, variance=False),
                         levels=self.maxdepth)
        found = self._a_x(x, s=s, b=b, variance=variance, 
                          discrete=discrete, depth=self.maxdepth)
        if found is not None:
            self.blog.write()
            return found
        if not variance and self.interest.i > 0:  # (1) twin insurance shortcut
            A = self._A_x(x, s=s, b=b, discrete=discrete, depth=self.maxdepth)
            if A is not None:
                self.blog("a_x = (1-A_x) / d",
                          depth=self.maxdepth, rule='insurance twin')
                self.blog.write()
                return self.annuity_twin(A=A, discrete=discrete)
        a = super().whole_life_annuity(x, s=s, b=b, discrete=discrete,
                                       variance=variance)
        if a is not None:
            self.blog.write() 
            return a

    def temporary_annuity(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                          b: int = 1, variance: bool = False,
                          discrete: bool = True) -> float:
        """Compute temporary annuity a_x:t by calling recursion then twin first

        Args:
          x : age of selection
          s : years after selection
          t : term of annuity in years
          b : annuity benefit amount
          variance (bool): return EPV (True) or variance (False)
          discrete : annuity due (True) or continuous (False)
        """
        self.blog = _Blog("Temporary Annuity",
                         _Blog.a(x=x, s=s, t=t, b=b, u=0,
                                discrete=discrete, variance=variance),
                         levels=self.maxdepth)
        found = self._a_x(x, s=s, b=b, t=t, variance=variance, 
                          discrete=discrete, depth=self.maxdepth)
        if found is not None:
            self.blog.write()
            return found
        if not variance and self.interest.i > 0: # (1) twin insurance shortcut
            A = self._A_x(x, s=s, b=b, t=t, endowment=b, discrete=discrete, 
                          depth=self.maxdepth)
            if A is not None:
                self.blog(_Blog.a(x=x, s=s, t=t), '= [ 1 -',
                          _Blog.A(x=x, s=s, t=t), f"] / d(t={t})",
                          #"Annuity twin: a = (1 - A) / d",
                          depth=self.maxdepth, rule='annuity twin')
                self.blog.write()
                return self.annuity_twin(A=A, discrete=discrete)
        a = super().temporary_annuity(x, s=s, b=b, t=t, discrete=discrete,
                                      variance=variance)
        if a is not None:
            self.blog.write()
            return a

    def deferred_annuity(self, x: int, s: int = 0, t: int = Reserves.WHOLE,
                         u: int = 0, b: int = 1, discrete: bool = True) -> float:
        """Compute deferred annuity u|a_x:t by calling recursion first

        Args:
          x : age of selection
          s : years after selection
          u : years deferred
          t : term of annuity in years
          b : annuity benefit amount
          discrete : annuity due (True) or continuous (False)
        """
        self.blog = _Blog("Deferred Annuity",
                         _Blog.a(x=x, s=s, t=t, b=b, u=u,
                                discrete=discrete, variance=False),
                         levels=self.maxdepth)
        a = self._a_x(x, s=s, b=b, t=t, u=u, 
                          discrete=discrete, depth=self.maxdepth)
        if a is not None:
            self.blog.write()
            return a
        a = self._a_x(x, s=s, b=b, t=self.add_term(u, t),  
                      discrete=discrete, depth=self.maxdepth)
        a_t = self._a_x(x, s=s, b=b, t=u,  
                        discrete=discrete, depth=self.maxdepth)
        if a is not None and a_t is not None:
            self.blog.write()
            return a - a_t
        return super().deferred_annuity(x, s=s, b=b, t=t, discrete=discrete)


class _Blog:
    """Helper to track and display recursion progress"""
    _verbose : bool = True
    
    def __init__(self, *args, levels: int = 3, width: int = 80):
        self.title = ' *' + " ".join(args) + " <--"  # to identify this stack
        self.levels = levels                         # maximum levels to indent
        self.width = width - 3                       # display line width
        self._history = []
        self._rules = []
        self._depths = []

    def __call__(self, *args, depth: int | None = None, rule: str = ''):
        """Append next message to history"""
        assert depth is not None and rule, "depth and rule must be specified"
        msg = " ".join([a for a in args]) 
        if msg not in self._history:
            self._history.insert(0, msg)
            self._depths.insert(0, depth)
            self._rules.insert(0, rule)

    def __len__(self) -> int:
        return int(_Blog._verbose and bool(self._history))

    def __str__(self):
        """Display message history"""
        if not len(self):
            return ''
        _str = self.title + '\n'
        for msg, depth, rule in zip(self._history, self._depths, self._rules):
            left = ' '*(3+max(self.levels-abs(depth), 0))
            right = ' '*max(5, self.width - len(msg) - len(left) - len(rule))
            _str += left + msg + right + '~' + rule + '\n'
        return _str

    def write(self, end=''):
        print(self, end=end)

    @staticmethod
    def q(x: int, s: int = 0, t: int = 1, u: int = 0) -> str:
        """Return string representation of mortality u|t_q_x term"""
        out = []
        if t != 1:
            out.append(f"t={t}")
        if u > 0:
            out.append(f"defer={u}")
        return f"q_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)

    @staticmethod
    def p(x: int, s: int = 0, t: int = 1) -> str:
        """Return string representation of survival t_p_x term"""
        out = []
        if t != 1:
            out.append(f"t={t}")
        return f"p_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)


    @staticmethod
    def e(x: int, s: int = 0, t: int = Reserves.WHOLE, curtate: bool = False,
          moment: int = 1) -> str:
        """Return string representation of expected future lifetime e_x term"""
        out = []
        if t >= 0:
            out.append(f"t={t}")
        if moment != 1:
            out.append(f"mom={moment}")
        out.append('curtate' if curtate else 'complete')
        return f"e_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)

    @staticmethod
    def E(x: int, s: int = 0, t: int = 1, endowment: int = 1,
          moment: int = 1) -> str:
        """Return string representation of pure endowment t_E_x term"""
        out = []
        out.append(f"t={t if t >= 0 else 'WL'}")  # term or whole life
        if moment != 1:
            out.append(f"mom={moment}")
        if endowment != 1:
            out.append(f"endow={endowment}")
        return f"E_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)
    
    @staticmethod
    def IA(x: int, s: int = 0, t: int = Reserves.WHOLE, b: int = 1,
           discrete: bool = True) -> str:
        """Return string representation of increasing insurance IA_x term"""
        out = []
        out.append(f"t={t if t >= 0 else 'WL'}")  # term or whole life
        if b != 1:
            out.append(f"b={b}")
        return f"IA_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)

    @staticmethod
    def DA(x: int, s: int = 0, t: int = Reserves.WHOLE, b: int = 1,
           discrete: bool = True) -> str:
        """Return string representation of decreasing insurance DA_x term"""
        out = []
        out.append(f"t={t if t >= 0 else 'WL'}")  # term or whole life
        if b != 1:
            out.append(f"b={b}")
        return f"DA_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)

    @staticmethod
    def A(x: int, s: int = 0, t: int = Reserves.WHOLE, u: int = 0, b: int = 1,
          moment: int = 1, endowment: int = 0, discrete: bool = True) -> str:
        """Return string representation of insurance A_x term"""
        out = []
        out.append(f"t={t if t >= 0 else 'WL'}")  # term or whole life
        if u != 0:
            out.append(f"u={u}")
        if b != 1:
            out.append(f"b={b}")
        if endowment != 0:
            out.append(f"endow={endowment}")
        if moment != 1:
            out.append(f"mom={moment}")
        return f"A_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)

    @staticmethod
    def a(x: int, s: int = 0, t: int = Reserves.WHOLE, u: int = 0, b: int = 1,
          variance: bool = False, discrete: bool = True) -> str:
        """Return string representation of annuity a_x term"""
        out = []
        out.append(f"t={t if t >= 0 else 'WL'}")  # term or whole life
        if u != 0:
            out.append(f"u={u}")
        if b != 1:
            out.append(f"b={b}")
        if variance:
            out.append('var')
        return f"a_{x+s}" + "("*bool(out) + ",".join(out) + ")"*bool(out)

    @staticmethod
    def m(moment: int, **kwargs) -> str:
        """Return string representation of moment exponent"""
        out = f"^{moment}"*(moment != 1)
        if not kwargs:
            return out
        args = [k for k,v in kwargs.items() if v]
        return ("(" + "*".join(args) + ")")+out if args else "0"


class _Blogtex(_Blog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fig = plt.figure()
        ax = fig.add_subplot(frame_on=False)
        fig.subplots_adjust(top=0.85)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        # Set titles for the figure and the subplot respectively
        fig.suptitle(title, fontsize=fontsize+4, fontweight='bold')

        # Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
        ax.axis([0, xmax, 0, ymax])

        
    def write(self,
              text = [],
              title='',
              skip=1,
              ymax=90,
              xmax=10,
              fontsize=16,
              fontname=None):
        """Draws lines of text in a blank images, so that can be savefig'd to jpg
        Args:
          text (list of str): to draw line by line
          title (str): title to draw in bold at top
          skip (int): number of blank lines after title
          ymax (int): number of vertical lines in image
          xmax (int): width of x-axis
          fontsize (int): fontsize of normal text
          fontname (str): e.g. 'monospace' or 'serif'
        """
        #ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        #        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

        for num, line in enumerate(text):
            ax.text(x=1, y=ymax - (skip + num), s=line, fontsize=fontsize,
                    fontname=fontname)
        
if __name__ == "__main__":
    from actuarialmath.constantforce import ConstantForce
    from actuarialmath.policyvalues import Contract

    print("SOA Question 6.10: (D) 0.91")
    x = 0
    life = Recursion().set_interest(i=0.06)\
                      .set_p(0.975, x=x)\
                      .set_a(152.85/56.05, x=x, t=3)\
                      .set_A(152.85, x=x, t=3, b=1000)
    print(life.p_x(x=x+2))
    #isclose(0.91, p, question="Q6.10")

    raise Exception
    
    print("AMLCR2 Exercise 2.6")
    x = 0
    life = Recursion(depth=3).set_interest(i=0.06)\
                             .set_p(0.99, x=x)\
                             .set_p(0.985, x=x+1)\
                             .set_p(0.95, x=x+1, t=3)\
                             .set_q(0.02, x=x+3)

    print(life.p_x(x=x+3))  # 0.98
    print(life.p_x(x=x, t=2))  # 0.97515
    print(life.p_x(x=x+1, t=2))  # 0.96939
    print(life.p_x(x=x, t=3))  # 0.95969
    print(life.q_x(x=x, t=2, u=1))  # 0.03031

    print("SOA Question 6.48:  (A) 3195")
    life = Recursion().set_interest(i=0.06)
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
    life = Recursion().set_interest(i=0.06).set_a(7, x=x+1).set_q(0.05, x=x)
    a = life.whole_life_annuity(x)
    A = 110 * a / 1000
    print(a, A)
    life = Recursion().set_interest(i=0.06).set_A(A, x=x).set_q(0.05, x=x)
    A1 = life.whole_life_insurance(x+1)
    P = life.gross_premium(A=A1 / 1.03, a=7) * 1000
    print(P)
    print()
    
    print("SOA Question 6.17:  (A) -30000")
    x = 0
    life = ConstantForce(mu=0.1).set_interest(i=0.08)
    A = life.endowment_insurance(x, t=2, b=100000, endowment=30000)
    a = life.temporary_annuity(x, t=2)
    P = life.gross_premium(a=a, A=A)
    print(A, a, P)

    life1 = Recursion().set_interest(i=0.08)\
                       .set_q(life.q_x(x, t=1) * 1.5, x=x, t=1)\
                       .set_q(life.q_x(x+1, t=1) * 1.5, x=x+1, t=1)
    contract = Contract(premium=P * 2, benefit=100000, endowment=30000)
    L = life1.gross_policy_value(x, t=0, n=2, contract=contract)
    print(L)
    print()

    life = Recursion(verbose=False).set_interest(i=0.05).set_E(0.95, 0,t=1)
    E = life.E_x(0, t=1, moment=2)
    print(E)
