"""Solutions to SOA FAM-L sample questions

Copyright 2022, Terence Lim

MIT License
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
from actuarialmath.life import Life
from actuarialmath.survival import Survival
from actuarialmath.lifetime import Lifetime
from actuarialmath.insurance import Insurance
from actuarialmath.annuity import Annuity
from actuarialmath.premiums import Premiums
from actuarialmath.policyvalues import PolicyValues
from actuarialmath.reserves import Reserves
from actuarialmath.recursion import Recursion
from actuarialmath.lifetable import LifeTable
from actuarialmath.sult import SULT
from actuarialmath.selectlife import Select
from actuarialmath.constantforce import ConstantForce
from actuarialmath.adjustmortality import Adjust
from actuarialmath.mthly import Mthly
from actuarialmath.udd import UDD
from actuarialmath.woolhouse import Woolhouse

class SOA:
    """To keep score of solutions correct"""
    def __init__(self, terminate: bool = False):
        self.score = {}
        self.terminate = terminate

    def __call__(self, solution: Union[float, str], answer: Union[float, str], 
                 msg: str, rel_tol: float = 0.01):
        """Grade this question, and accumulate score"""
        section, question = str(msg).split('.')
        msg = "SOA Question " + str(msg) + ":"
        if isinstance(solution, str) or isinstance(answer, str):
            correct = (solution == answer)
        else:
            correct = math.isclose(solution, answer, rel_tol=rel_tol)
        print(msg, '[', solution, ']', answer, '*' * 10 * (1-correct))
        if section not in self.score:
            self.score[section] = {}
        self.score[section][question] = correct
        if self.terminate:
            assert correct, msg

    def summary(self):
        """Return final score and by section"""
        score = {int(k): [len(v), sum(v.values())] for k, v in self.score.items()}
        out = pd.DataFrame.from_dict(score, orient='index', 
                                     columns=['num', 'correct'])
        out.loc[0] = [sum(out['num']), sum(out['correct'])]
        return out.sort_index()

soa = SOA(terminate=False)

if __name__ == "__main__":

    ## SOA Question 2.1: (B) 2.5
    def fun(omega):  # Solve first for omega, given mu_65 = 1/180
        return Lifetime(l=lambda x,s: (1 - (x+s)/omega)**0.25).mu_x(65)
    omega = int(Lifetime.solve(fun, target=1/180, guess=(106, 126)))
    life = Lifetime(l=lambda x,s: (1 - (x+s)/omega)**0.25, maxage=omega)
    soa(2.5, life.e_x(106, curtate=True), 2.1)

    ## SOA Question 2.2: (D) 400
    p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
    p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
    v = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
    soa(400, v, 2.2)

    ## SOA Question 2.3: (A) 0.0483
    B, c = 0.00027, 1.1
    life = Survival(S=lambda x,s,t: (math.exp(-B * c**(x+s) 
                                     * (c**t - 1)/math.log(c))))
    soa(0.0483, life.f_x(x=50, t=10), 2.3)

    ## SOA Question 2.4: (E) 8.2
    life = Lifetime(l=lambda x,s: 0. if (x+s) >= 100 else 1 - ((x+s)**2)/10000.)
    soa(8.2, life.e_x(75, t=10, curtate=False), 2.4)

    ## SOA Question 2.5:  (B) 37.1
    life = Recursion().set_e(25, x=60, curtate=True)
    life.set_q(0.2, x=40, t=20).set_q(0.003, x=40)
    def fun(e):  # solve e_40 from e_40:20 = e_40 - 20_p_40 e_60
        return life.set_e(e, x=40, curtate=True).e_x(x=40, t=20, curtate=True)
    life.set_e(life.solve(fun, target=18, guess=[36, 41]), x=40, curtate=True)
    soa(37.1, life.e_x(41, curtate=True), 2.5)

    ## SOA Question 2.6: (C) 13.3
    life = Survival(l=lambda x,s: (1 - (x+s)/60)**(1/3))
    soa(13.3, 1000*life.mu_x(35), 2.6)

    ## SOA Question 2.7: (B) 0.1477
    life = Survival(l=lambda x,s: 
                      (1-((x+s)/250) if (x+s)<40 else 1-((x+s)/100)**2))
    soa(0.1477, life.q_x(30, t=20), 2.7)

    ## SOA Question 2.8: (C) 0.94
    def fun(p):  # Solve first for mu, given start and end proportions
        mu = -math.log(p)
        male = Lifetime(mu=lambda x,s: 1.5 * mu)
        female = Lifetime(mu=lambda x,s: mu)
        return (75 * female.p_x(0, t=20)) / (25 * male.p_x(0, t=20))
    soa(0.94, Lifetime.solve(fun, target=85/15, guess=[0.89, 0.99]), 2.8)

    ## SOA Question 3.1:  (B) 117
    life = Select(l={60: [80000, 79000, 77000, 74000],
                     61: [78000, 76000, 73000, 70000],
                     62: [75000, 72000, 69000, 67000],
                     63: [71000, 68000, 66000, 65000]})
    soa(117, 1000*life.q_r(60, s=0, r=0.75, t=3, u=2), 3.1)

    ## SOA Question 3.2:  (D) 14.7
    e_curtate = Select.e_curtate(e=15)
    life = Select(l={65: [1000, None,],
                     66: [955, None]},
                  e={65: [e_curtate, None]},
                  d={65: [40, None,],
                     66: [45, None]}, udd=True).fill()
    soa(14.7, life.e_r(66), 3.2)

    ## SOA Question 3.3:  (E) 1074
    life = Select(l={50: [99, 96, 93],
                     51: [97, 93, 89],
                     52: [93, 88, 83],
                     53: [90, 84, 78]})
    soa(1074, 10000*life.q_r(51, s=0, r=0.5, t=2.2), 3.3)

    ## SOA Question 3.4:  (B) 815
    sult = SULT()
    mean = sult.p_x(25, t=95-25)
    var = sult.bernoulli(mean, variance=True)
    p = sult.portfolio_percentile(N=4000, mean=mean, variance=var, prob=0.1)
    soa(815, p, 3.4)

    ## SOA Question 3.5:  (E) 106
    l = [99999, 88888, 77777, 66666, 55555, 44444, 33333, 22222]
    a = LifeTable(l={age:l for age,l in zip(range(60, 68), l)}, udd=True)\
        .q_r(60, u=3.4, t=2.5)
    b = LifeTable(l={age:l for age,l in zip(range(60, 68), l)}, udd=False)\
        .q_r(60, u=3.4, t=2.5)
    soa(106, 100000 * (a - b), 3.5)

    ## SOA Question 3.6:  (D) 15.85
    life = Select(q={60: [.09, .11, .13, .15],
                     61: [.1, .12, .14, .16],
                     62: [.11, .13, .15, .17],
                     63: [.12, .14, .16, .18],
                     64: [.13, .15, .17, .19]},
                  e={61: [None, None, None, 5.1]}).fill()
    soa(5.85, life.e_x(61), 3.6)

    ## SOA Question 3.7: (b) 16.4
    life = Select(q={50: [.0050, .0063, .0080],
                     51: [.0060, .0073, .0090],
                     52: [.0070, .0083, .0100],
                     53: [.0080, .0093, .0110]}).fill()
    soa(16.4, 1000*life.q_r(50, s=0, r=0.4, t=2.5), 3.7)

    ## SOA Question 3.8:  (B) 1505
    sult = SULT()
    p1 = sult.p_x(35, t=40)
    p2 = sult.p_x(45, t=40)
    mean = sult.bernoulli(p1) * 1000 + sult.bernoulli(p2) * 1000
    var = (sult.bernoulli(p1, variance=True) * 1000 
           + sult.bernoulli(p2, variance=True) * 1000)
    soa(1505, sult.portfolio_percentile(mean=mean, variance=var, prob=.95), 3.8)

    ## SOA Question 3.9:  (E) 3850
    sult = SULT()
    p1 = sult.p_x(20, t=25)
    p2 = sult.p_x(45, t=25)
    mean = sult.bernoulli(p1) * 2000 + sult.bernoulli(p2) * 2000
    var = (sult.bernoulli(p1, variance=True) * 2000 
           + sult.bernoulli(p2, variance=True) * 2000)
    soa(3850, sult.portfolio_percentile(mean=mean, variance=var, prob=.99), 3.9)

    ## SOA Question 3.10:  (C) 0.86
    interest = Life.Interest(v=0.75)
    L = 35*interest.annuity(t=4, due=False) + 75*interest.v_t(t=5)
    interest = Life.Interest(v=0.5)
    R = 15*interest.annuity(t=4, due=False) + 25*interest.v_t(t=5)
    soa(0.86, L / (L + R), "3.10")

    ## SOA Question 3.11:  (B) 0.03
    life = LifeTable(q={50//2: .02, 52//2: .04}, udd=True).fill()
    soa(0.03, life.q_r(50//2, t=2.5/2), 3.11)

    ## SOA Question 3.12: (C) 0.055 
    life = Select(l={60: [10000, 9600, 8640, 7771],
                     61: [8654, 8135, 6996, 5737],
                     62: [7119, 6549, 5501, 4016],
                     63: [5760, 4954, 3765, 2410]}, udd=False).fill()
    soa(0.055, life.q_r(60, s=1, t=3.5)-life.q_r(61, s=0, t=3.5), 3.12)


    ## SOA Question 3.13:  (B) 1.6
    life = Select(l={55: [10000, 9493, 8533, 7664],
                     56: [8547, 8028, 6889, 5630],
                     57: [7011, 6443, 5395, 3904],
                     58: [5853, 4846, 3548, 2210]},
                  e={57: [None, None, None, 1]}).fill()
    soa(1.6, life.e_r(58, s=2), 3.13)

    ## SOA Question 3.14:  (C) 0.345
    life = LifeTable(l={90: 1000, 93: 825},
                     d={97: 72},
                     p={96: .2},
                     q={95: .4, 97: 1}, udd=True).fill()
    soa(0.345, life.q_r(90, u=93-90, t=95.5 - 93), 3.14)

    ## SOA Question 4.1:  (A) 0.27212
    life = Recursion(interest=dict(i=0.03))
    life.set_A(0.36987, x=40).set_A(0.62567, x=60)
    life.set_E(0.51276, x=40, t=20).set_E(0.17878, x=60, t=20)
    Z2 = 0.24954
    A = (2 * life.term_insurance(40, t=20) 
         + life.deferred_insurance(40, u=20))
    soa(0.27212, math.sqrt(life.insurance_variance(A2=Z2, A1=A)), 4.1)

    ## SOA Question 4.2:  (D) 0.18
    life = LifeTable(q={0: .16, 1: .23}, 
                     interest=dict(i_m=.18, m=2),
                     udd=False).fill()
    mthly = Mthly(m=2, life=life)
    Z = mthly.Z_m(0, t=2, benefit=lambda x,t: 300000 + t*30000*2)
    soa(0.18, Z[Z['Z'] >= 277000].iloc[:, -1].sum(), 4.2)

    ## SOA Question 4.3: (D) 0.878  -- multi recursion on endowment insurance
    life = Recursion(interest=dict(i=0.05)).set_q(0.01, x=60)
    def fun(q):   # solve for q_61
        return life.set_q(q, x=61).endowment_insurance(60, t=3)
    q = life.solve(fun, target=0.86545, guess=0.01)
    life.set_q(q, x=61).set_interest(i=0.045)
    A = life.endowment_insurance(60, t=3)
    soa(0.878, A, "4.3")

    ## SOA Question 4.4  (A) 0.036
    life = Insurance(f=lambda *x: 0.025, 
                     maxage=40+40,
                     interest=dict(v_t=lambda t: (1 + .2*t)**(-2)))
    benefit = lambda x,t: 1 + .2 * t
    A1 = life.A_x(40, benefit=benefit, discrete=False)
    A2 = life.A_x(40, moment=2, benefit=benefit, discrete=False)
    soa(0.036, life.insurance_variance(A2=A2, A1=A1), 4.4)

    ## SOA Question 4.5:  (C) 35200
    sult = SULT(interest=dict(delta=0.05))
    Z = 100000 * sult.Z_from_prob(45, 0.95)
    soa(35200, Z, 4.5)

    ## SOA Question 4.6:  (B) 29.85
    sult = SULT()
    life = LifeTable(interest=dict(i=0.05),
                     q={70+k: .95**k * sult.q_x(70+k) for k in range(3)}).fill()
    A = life.term_insurance(70, t=3, b=1000)
    soa(29.85, A, 4.6)

    ## SOA Question 4.7:  (B) 0.06
    def fun(i):
        life = Recursion(interest=dict(i=i), verbose=False).set_p(0.57, x=0, t=25)
        return 0.1*life.E_x(0, t=25) - life.E_x(0, t=25, moment=life.VARIANCE)
    soa(0.06, Recursion.solve(fun, target=0, guess=[0.058, 0.066]), 4.7)

    ## SOA Question 4.8  (C) 191
    v_t = lambda t: 1.04**(-t) if t < 1 else 1.04**(-1) * 1.05**(-t+1)
    life = SULT(interest=dict(v_t=v_t))
    soa(191, life.whole_life_insurance(50, b=1000), 4.8)


    ## SOA Question 4.9:  (D) 0.5
    life = Recursion().set_A(0.39, x=35, t=15, endowment=1)\
                      .set_A(0.25, x=35, t=15)
    E = life.E_x(35, t=15)
    life = Recursion().set_A(0.32, x=35)\
                      .set_E(E, x=35, t=15)
    def fun(A):
        return life.set_A(A, x=50).term_insurance(35, t=15)
    A = life.solve(fun, target=0.25, guess=[0.35, 0.55])
    soa(0.5, A, 4.9)

    ## SOA Question 4.10:  (D)
    life = Insurance(interest=dict(i=0.01), S=lambda x,s,t: 1, maxage=40)
    def fun(x, t):
        if 10 <= t <= 20: return life.interest.v_t(t)
        elif 20 < t <= 30: return 2 * life.interest.v_t(t)
        else: return 0
    def A(x, t):  # Z_x+k (t-k)
        return life.interest.v_t(t - x) * (t > x)
    x = 0
    benefits=[lambda x,t: (life.E_x(x, t=10) * A(x+10, t)
                             + life.E_x(x, t=20)* A(x+20, t) 
                             - life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (A(x, t)  
                             + life.E_x(x, t=20) * A(x+20, t) 
                             - 2 * life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (life.E_x(x, t=10) * A(x, t) 
                             + life.E_x(x, t=20) * A(x+20, t) 
                             - 2 * life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (life.E_x(x, t=10) * A(x+10, t) 
                             + life.E_x(x, t=20) * A(x+20, t) 
                             - 2 * life.E_x(x, t=30) * A(x+30, t)),
              lambda x,t: (life.E_x(x, t=10)
                             * (A(x+10, t) 
                                + life.E_x(x+10, t=10) * A(x+20, t) 
                                - life.E_x(x+20, t=10) * A(x+30, t)))]
    fig, ax = plt.subplots(3, 2)
    ax = ax.ravel()
    for i, b in enumerate([fun] + benefits):
        life.Z_plot(0, benefit=b, ax=ax[i], verbose=False, color=f"C{i+1}")
        ax[i].legend(["(" + "abcde"[i-1] + ")" if i else "Z"])
    z = [sum(abs(b(0, t) - fun(0, t)) for t in range(40)) for b in benefits]
    soa('D', "ABCDE"[np.argmin(z)], '4.10')

    ## SOA Question 4.11:  (A) 143385
    A1 = 528/1000   # E[Z1]  term insurance
    C1 = 0.209      # E[pure_endowment]
    C2 = 0.136      # E[pure_endowment^2]
    B1 = A1 + C1    # endowment = term + pure_endowment
    def fun(A2):
        B2 = A2 + C2   # double force of interest
        return Insurance.insurance_variance(A2=B2, A1=B1)  
    A2 = Insurance.solve(fun, target=15000/(1000*1000), guess=[143400, 279300])
    soa(143385, Insurance.insurance_variance(A2=A2, A1=A1, b=1000), 4.11)

    ## SOA Question 4.12:  (C) 167 
    cov = Life.covariance(a=1.65, b=10.75, ab=0)  # E[Z1 Z2] = 0 nonoverlapping
    soa(167, Life.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov), 4.12)
    
    ## SOA Question 4.13:  (C) 350 
    life = Select(q={65: [.08, .10, .12, .14],
                     66: [.09, .11, .13, .15],
                     67: [.10, .12, .14, .16],
                     68: [.11, .13, .15, .17],
                     69: [.12, .14, .16, .18]}, interest=dict(i=.04)).fill()
    soa(350, life.deferred_insurance(65, t=2, u=2, b=2000), 4.13)

    ## SOA Question 4.14:  (E) 390000
    sult = SULT()
    p = sult.p_x(60, t=85-60)
    mean = sult.bernoulli(p)
    var = sult.bernoulli(p, variance=True)
    F = sult.portfolio_percentile(mean=mean, variance=var, prob=.86, N=400)
    soa(390000, F * 5000 * sult.interest.v_t(85-60), 4.14)

    ## SOA Question 4.15  (E) 0.0833 
    life = Insurance(mu=lambda *x: 0.04, interest=dict(delta=0.06))
    benefit = lambda x,t: math.exp(0.02*t)
    A1 = life.A_x(0, benefit=benefit, discrete=False)
    A2 = life.A_x(0, moment=2, benefit=benefit, discrete=False)
    soa(0.0833, life.insurance_variance(A2=A2, A1=A1), 4.15)

    ## SOA Question 4.16:  (D) 0.11
    q = [.045, .050, .055, .060]
    q_ = {50+x: [0.7 * q[x] if x < len(q) else None, 
                 0.8 * q[x+1] if x+1 < len(q) else None, 
                 q[x+2] if x+2 < len(q) else None] 
          for x in range(4)}
    life = Select(q=q_, interest=dict(i=.04)).fill()
    soa(0.1116, life.term_insurance(50, t=3), 4.16)


    ## SOA Question 4.17:  (A) 1126.7
    sult = SULT()
    median = sult.Z_t(48, prob=0.5, discrete=False)
    benefit = lambda x,t: 5000 if t < median else 10000
    A = sult.A_x(48, benefit=benefit)
    soa(1130, A, 4.17)

    ## SOA Question 4.18  (A) 81873 
    life = Insurance(interest=dict(delta=0.05), 
                     maxage=10,
                     f=lambda x,s,t: .1 if t < 2 else .4*t**(-2))
    benefit = lambda x,t: 0 if t < 2 else 100000
    prob = 0.9 - life.q_x(0, t=2)
    x, y = life.survival_curve()
    T = life.Z_t(0, prob=prob)
    life.Z_plot(0, T=T, benefit=benefit, discrete=False, curve=(x,y))
    Z = life.Z_from_t(T) * benefit(0, T)
    #plt.show()
    soa(81873, Z, 4.18)

    ## SOA Question 4.19:  (B) 59050
    life = SULT()
    adjust = Adjust(life=life)
    q = adjust(extra=0.8, adjust=Adjust.MULTIPLY_RATE)['q']
    select = Select(n=1)\
             .set_select(column=0, select_age=True, q=q)\
             .set_select(column=1, select_age=False, q=life['q']).fill()
    soa(59050, 100000*select.whole_life_insurance(80, s=0), 4.19)

    ## SOA Question 5.1: (A) 0.705
    life = ConstantForce(mu=0.01, interest=dict(delta=0.06))
    EY = life.certain_life_annuity(0, u=10, discrete=False)
    a = life.p_x(0, t=life.Y_to_t(EY))
    soa(0.705, a, 5.1)  # 0.705

    ## SOA Question 5.2:  (B) 9.64
    x, n = 0, 10
    life = Recursion(interest=dict(i=0.05))
    life.set_A(0.3, x).set_A(0.4, x+n).set_E(0.35, x, t=n)
    a = life.immediate_annuity(x, t=n)
    soa(9.64, a, 5.2)

    ## SOA Question 5.3:  (C) 6.239
    sult = SULT()
    t = 10.5
    soa(6.239, t * sult.E_r(40, t=t), 5.3)

    ## SOA Question 5.4:  (A) 213.7
    life = ConstantForce(mu=0.02, interest=dict(delta=0.01))
    P = 10000 / life.certain_life_annuity(40, u=life.e_x(40, curtate=False), 
                                          discrete=False)
    soa(213.7, P, 5.4) # 213.7

    ## SOA Question 5.5: (A) 1699.6
    life = SULT()
    adjust = Adjust(life=life)
    q = adjust(extra=0.05, adjust=Adjust.ADD_FORCE)['q']
    select = Select(n=1)\
             .set_select(column=0, select_age=True, q=q)\
             .set_select(column=1, select_age=False, a=life['a']).fill()
    soa(1700, 100*select['a'][45][0], 5.5)

    ## SOA Question 5.6:  (D) 1200
    life = Annuity(interest=dict(i=0.05))
    var = life.annuity_variance(A2=0.22, A1=0.45)
    mean = life.annuity_twin(A=0.45)
    soa(1200, life.portfolio_percentile(mean, var, prob=.95, N=100), 5.6)

    ## SOA Question 5.7:  (C) 
    life = Recursion(interest=dict(i=0.04))
    life.set_A(0.188, x=35).set_A(0.498, x=65).set_p(0.883, x=35, t=30)
    mthly = Woolhouse(m=2, life=life, three_term=False)
    soa(17376.7, 1000 * mthly.temporary_annuity(35, t=30), 5.7)

    ## SOA Question 5.8: (C) 0.92118
    sult = SULT()
    a = sult.certain_life_annuity(55, u=5)
    soa(0.92118, sult.p_x(55, t=math.floor(a)), 5.8)

    ## SOA Question 5.9:  (C) 0.015
    x, p = 0, 0.9  # set arbitrary p_x = 0.9
    life1 = Recursion().set_a(21.854, x=x).set_p(p, x=x)
    life2 = Recursion().set_a(22.167, x=x)
    def fun(k):
        life2.set_p((1 + k) * p, x=x)
        return life1.whole_life_annuity(x+1) - life2.whole_life_annuity(x+1)
    soa(0.015, life2.solve(fun, target=0, guess=[0.005, 0.025]), 5.9)
    
    ## SOA Question 6.1: (D) 35.36
    life = SULT(interest=dict(i=0.03))
    soa(35.36, life.net_premium(80, t=2, b=1000, return_premium=True), 6.1)


    ## SOA Question 6.2: (E) 3604
    life = Premiums()
    A, IA, a = 0.17094, 0.96728, 6.8865
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=100000,
                           initial_premium=0.5, renewal_premium=.05,
                           renewal_policy=200, initial_policy=200)
    soa(3604, P, 6.2)

    ## SOA Question 6.3:  (C) 0.390
    life = SULT()
    P = life.net_premium(45, u=20, annuity=True)
    t = life.Y_to_t(life.whole_life_annuity(65))
    p = 1 - life.p_x(65, t=math.floor(t) - 1)
    soa(0.39, p, 6.3)

    ## SOA Question 6.4:  (E) 1890
    mthly = Mthly(m=12, life=Reserves(interest=dict(i=0.06)))
    A1, A2 = 0.4075, 0.2105
    mean = mthly.annuity_twin(A1)*15*12
    var = mthly.annuity_variance(A1=A1, A2=A2, b=15 * 12)
    S = Reserves.portfolio_percentile(mean=mean, variance=var, prob=.9, N=200)
    soa(1890, S / 200, 6.4)

    ## SOA Question 6.5:  (D) 33
    life = SULT()
    P = life.net_premium(30, b=1000)
    def fun(k):
        return (life.Y_x(30, t=k) * P
                - life.Z_x(30, t=k) * 1000)
    soa(33, min([k for k in range(20, 40) if fun(k) < 0]), 6.5)

    ## SOA Question 6.6:  (B) 0.79
    life = SULT()
    P = life.net_premium(62, b=10000)
    policy = life.Policy(premium=1.03*P, renewal_policy=5,
                         initial_policy=5, initial_premium=0.05, benefit=10000)
    L = life.gross_policy_value(62, policy=policy)
    var = life.gross_policy_variance(62, policy=policy)
    prob = life.portfolio_cdf(mean=L, variance=var, value=40000, N=600)
    soa(.79, prob, 6.6)

    ## SOA Question 6.7:  (C) 2880
    life=SULT()
    a = life.temporary_annuity(40, t=20) 
    A = life.E_x(40, t=20)
    IA = a - life.interest.annuity(t=20) * life.p_x(40, t=20)
    soa(2880, life.gross_premium(a=a, A=A, IA=IA, benefit=100000), 6.7)

    ## SOA Question 6.8:  (B) 9.5
    life = SULT()
    initial_cost = (50 + 10 * life.deferred_annuity(60, u=1, t=9)
                    + 5 * life.deferred_annuity(60, u=10, t=10))
    soa(9.5, life.net_premium(60, initial_cost=initial_cost), 6.8)

    ## SOA Question 6.9:  (D) 647
    life = SULT()
    a = life.temporary_annuity(50, t=10)
    A = life.term_insurance(50, t=20)
    initial_cost = 25 * life.deferred_annuity(50, u=10, t=10)
    P = life.gross_premium(a=a, A=A, benefit=100000,
                           initial_premium=0.42, renewal_premium=0.12,
                           initial_policy=75 + initial_cost, renewal_policy=25)
    soa(647, P, 6.9)

    ## SOA Question 6.10:  (D) 0.91
    x = 0
    life = Recursion(interest=dict(i=0.06)).set_p(0.975, x=x)
    a = 152.85/56.05  # solve a_x:3, given net premium and benefit APV

    def fun(p):   # solve p_x+2, given a_x:3
        return life.set_p(p, x=x+1).temporary_annuity(x, t=3)
    life.set_p(life.solve(fun, target=a, guess=0.975), x=x+1)

    def fun(p):   # finally solve p_x+3, given A_x:3
        return life.set_p(p, x=x+2).term_insurance(x=x, t=3, b=1000)
    p = life.solve(fun, target=152.85, guess=0.975)
    soa(0.91, p, "6.10")

    ## SOA Question 6.11:  (C) 0.041
    life = Recursion(interest=dict(i=0.04))
    life.set_A(0.39788, 51)
    life.set_q(0.0048, 50)
    A = life.whole_life_insurance(50)
    P = life.gross_premium(A=A, a=life.annuity_twin(A=A))
    life.set_q(0.048, 50)
    A = life.whole_life_insurance(50)
    soa(0.041, A - life.annuity_twin(A) * P, 6.11)

    ## SOA Question 6.12:  (E) 88900
    life = PolicyValues(interest=dict(i=0.06))
    a = 12
    A = life.insurance_twin(a)
    policy = life.Policy(benefit=1000, settlement_policy=20,
                         initial_policy=10, initial_premium=0.75, 
                         renewal_policy=2, renewal_premium=0.1)
    policy.premium = life.gross_premium(A=A, a=a, **policy.premium_terms)
    L = life.gross_variance_loss(A1=A, A2=0.14, policy=policy)
    soa(88900, L, 6.12)

    ## SOA Question 6.13:  (D) -400
    life = SULT(interest=dict(i=0.05))
    A = life.whole_life_insurance(45)
    policy = life.Policy(benefit=10000, initial_premium=.8, renewal_premium=.1)
    def fun(P):   # Solve for premium, given Loss(t=0) = 4953
        return life.L_from_t(t=10.5, policy=policy.set(premium=P))
    policy.premium = life.solve(fun, target=4953, guess=100)
    L = life.gross_policy_value(45, policy=policy)
    soa(-400, L, 6.13)

    ## SOA Question 6.14  (D) 1150
    life = SULT(interest=dict(i=0.05))
    a = (life.temporary_annuity(40, t=10)
         + 0.5 * life.deferred_annuity(40, u=10, t=10))
    A = life.whole_life_insurance(40)
    P = life.gross_premium(a=a, A=A, benefit=100000)
    soa(1150, P, 6.14)

    ## SOA Question 6.15:  (B) 1.002
    life = Recursion(interest=dict(i=0.05)).set_a(3.4611, x=0)
    A = life.insurance_twin(3.4611)
    udd = UDD(m=4, life=life)
    a1 = udd.whole_life_annuity(x=x)
    woolhouse = Woolhouse(m=4, life=life)
    a2 = woolhouse.whole_life_annuity(x=x)
    P = life.gross_premium(a=a1, A=A)/life.gross_premium(a=a2, A=A)
    soa(1.002, P, 6.15)

    ## SOA Question 6.16: (A) 2408.6
    life = Premiums(interest=dict(d=0.05))
    A = life.insurance_equivalence(premium=2143, b=100000)
    a = life.annuity_equivalence(premium=2143, b=100000)
    p = life.gross_premium(A=A, a=a, benefit=100000, settlement_policy=0,
                           initial_policy=250, initial_premium=0.04 + 0.35,
                           renewal_policy=50, renewal_premium=0.04 + 0.02) 
    soa(2410, p, 6.16)

    ## SOA Question 6.17:  (A) -30000
    x = 0
    life = ConstantForce(mu=0.1, interest=dict(i=0.08))
    A = life.endowment_insurance(x, t=2, b=100000, endowment=30000)
    a = life.temporary_annuity(x, t=2)
    P = life.gross_premium(a=a, A=A)
    life1 = Recursion(interest=dict(i=0.08))
    life1.set_q(life.q_x(x, t=1) * 1.5, x=x, t=1)
    life1.set_q(life.q_x(x+1, t=1) * 1.5, x=x+1, t=1)
    policy = life1.Policy(premium=P*2, benefit=100000, endowment=30000)
    L = life1.gross_policy_value(x, t=0, n=2, policy=policy)
    soa(-30000, L, 6.17)

    ## SOA Question 6.18:  (D) 166400
    life = SULT(interest=dict(i=0.05))
    def fun(P):
        A = (life.term_insurance(40, t=20, b=P)
             + life.deferred_annuity(40, u=20, b=30000))
        return life.gross_premium(a=1, A=A) - P
    P = life.solve(fun, target=0, guess=[162000, 168800])
    soa(166400, P, 6.18)

    ## SOA Question 6.19:  (B) 0.033
    life = SULT()
    policy = life.Policy(initial_policy=.2, renewal_policy=.01)
    a = life.whole_life_annuity(50)
    A = life.whole_life_insurance(50)
    policy.premium = life.gross_premium(A=A, a=a, **policy.premium_terms)
    L = life.gross_policy_variance(50, policy=policy)
    soa(0.033, L, 6.19)

    ## SOA Question 6.20:  (B) 459
    life = LifeTable(interest=dict(i=0.04),
                     p={75: 0.9, 76: 0.88, 77: 0.85}).fill()
    a = life.temporary_annuity(75, t=3)
    IA = life.increasing_insurance(75, t=2)
    A = life.deferred_insurance(75, u=2, t=1)
    def fun(P):
        return life.gross_premium(a=a, A=P*IA + A*10000) - P
    soa(459, life.solve(fun, target=0, guess=[449, 489]), "6.20")

    ## SOA Question 6.21:  (C) 100
    life = Recursion(interest=dict(d=0.04))
    life.set_A(0.7, x=75, t=15, endowment=1)
    life.set_E(0.11, x=75, t=15)
    def fun(P):
        P = float(P)
        return (P * life.temporary_annuity(75, t=15)
                - life.endowment_insurance(75, t=15, b=1000, endowment=15*P))
    P = life.solve(fun, target=0, guess=(80, 120))
    soa(100, P, 6.21)

    ## SOA Question 6.22:  (C) 102
    life=SULT(udd=True)
    a = UDD(m=12, life=life).temporary_annuity(45, t=20)
    A = UDD(m=0, life=life).whole_life_insurance(45)
    P = life.gross_premium(A=A, a=a, benefit=100000) / 12
    soa(102, P, 6.22)

    ## SOA Question 6.23:  (D) 44.7
    x = 0
    life = Recursion().set_a(15.3926, x=x)\
                      .set_a(10.1329, x=x, t=15)\
                      .set_a(14.0145, x=x, t=30)
    def fun(P):
        per_policy = 30 + (30 * life.whole_life_annuity(x))
        per_premium = (0.6 + 0.1 * life.temporary_annuity(x, t=15)
                        + 0.1 * life.temporary_annuity(x, t=30))
        a = life.temporary_annuity(x, t=30)
        return (P * a) - (per_policy + per_premium * P)
    P = life.solve(fun, target=0, guess=[30.3, 49.5])
    soa(44.7, P, 6.23)

    ## SOA Question 6.24:  (E) 0.30
    life = PolicyValues(interest=dict(delta=0.07))
    x, A1 = 0, 0.30   # Policy for first insurance
    P = life.premium_equivalence(A=A1, discrete=False)  # Need its premium
    policy = life.Policy(premium=P, discrete=False)
    def fun(A2):  # Solve for A2, given Var(Loss)
        return life.gross_variance_loss(A1=A1, A2=A2, policy=policy)
    A2 = life.solve(fun, target=0.18, guess=0.18)

    policy = life.Policy(premium=0.06, discrete=False) # Solve second insurance
    variance = life.gross_variance_loss(A1=A1, A2=A2, policy=policy)
    soa(0.304, variance, 6.24)

    ## SOA Question 6.25:  (C) 12330
    life = SULT()
    woolhouse = Woolhouse(m=12, life=life)
    benefits = woolhouse.deferred_annuity(55, u=10, b=1000 * 12)
    expenses = life.whole_life_annuity(55, b=300)
    payments = life.temporary_annuity(55, t=10)
    def fun(P):
        return life.gross_future_loss(A=benefits + expenses, a=payments,
                                      policy=life.Policy(premium=P))
    P = life.solve(fun, target=-800, guess=[12110, 12550])
    soa(12330, P, 6.25)

    ## SOA Question 6.26  (D) 180
    life = SULT(interest=dict(i=0.05))
    def fun(P):
        return P - life.net_premium(90, b=1000, initial_cost=P)
    P = life.solve(fun, target=0, guess=[150, 190])
    soa(180, P, 6.26)

    ## SOA Question 6.27:  (D) 10310
    life = ConstantForce(mu=0.03, interest=dict(delta=0.06))
    x = 0
    payments = (3 * life.temporary_annuity(x, t=20, discrete=False) 
                + life.deferred_annuity(x, u=20, discrete=False))
    benefits = (1000000 * life.term_insurance(x, t=20, discrete=False)
                + 500000 * life.deferred_insurance(x, u=20, discrete=False))
    P = benefits / payments
    soa(10310, P, 6.27)
    
    ## SOA Question 6.28  (B) 36
    life = SULT(interest=dict(i=0.05))
    a = life.temporary_annuity(40, t=5)
    A = life.whole_life_insurance(40)
    P = life.gross_premium(a=a, A=A, benefit=1000, 
                           initial_policy=10, renewal_premium=.05,
                           renewal_policy=5, initial_premium=.2)
    soa(36, P, 6.28)

    ## SOA Question 6.29  (B) 20.5
    life = Premiums(interest=dict(i=0.035))
    def fun(a):
        return life.gross_premium(A=life.insurance_twin(a=a), a=a, 
                                  initial_policy=200, initial_premium=.5,
                                  renewal_policy=50, renewal_premium=.1,
                                  benefit=100000)
    a = life.solve(fun, target=1770, guess=[20, 22])
    soa(20.5, a, 6.29)

    ## SOA Question 6.30:  (A) 900
    life = PolicyValues(interest=dict(i=0.04))
    policy = life.Policy(premium=2.338, benefit=100, initial_premium=.1,
                         renewal_premium=0.05)
    var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
                                   A2=0.17, policy=policy)
    soa(900, var, "6.30")
    
    ## SOA Question 6.31:  (D) 1330
    life = ConstantForce(mu=0.01, interest=dict(delta=0.05))
    A = (life.term_insurance(35, t=35, discrete=False) 
         + life.E_x(35, t=35) * 0.51791)    # A_35
    P = life.premium_equivalence(A=A, b=100000, discrete=False)
    soa(1330, P, 6.31)

    ## SOA Question 6.32:  (C) 550
    x = 0
    life = Recursion(interest=dict(i=0.05)).set_a(9.19, x=x)
    benefits = UDD(m=0, life=life).whole_life_insurance(x)
    payments = UDD(m=12, life=life).whole_life_annuity(x)
    P = life.gross_premium(a=payments, A=benefits, benefit=100000)/12
    soa(550, P, 6.32)

    ## SOA Question 6.33:  (B) 0.13
    life = Insurance(mu=lambda x,t: 0.02*t, interest=dict(i=0.03))
    x = 0
    var = life.E_x(x, t=15, moment=life.VARIANCE, endowment=10000)
    p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
    soa(0.13, p, 6.33, rel_tol=0.02)

    ## SOA Question 6.34:  (A) 23300
    life = SULT()
    def fun(benefit):
        A = life.whole_life_insurance(61)
        a = life.whole_life_annuity(61)
        return life.gross_premium(A=A, a=a, benefit=benefit, 
                                  initial_premium=0.15, renewal_premium=0.03)
    b = life.solve(fun, target=500, guess=[23300, 23700])
    soa(23300, b, 6.34)

    ## SOA Question 6.35:  (D) 530
    sult = SULT()
    A = sult.whole_life_insurance(35, b=100000)
    a = sult.whole_life_annuity(35)
    P = sult.gross_premium(a=a, A=A, initial_premium=.19, renewal_premium=.04)
    soa(530, P, 6.35)

    ## SOA Question 6.36:  (B) 500
    life = ConstantForce(mu=0.04, interest=dict(delta=0.08))
    a = life.temporary_annuity(50, t=20, discrete=False)
    A = life.term_insurance(50, t=20, discrete=False)
    def fun(R):
        return life.gross_premium(a=a, A=A, initial_premium=R/4500,
                                  renewal_premium=R/4500, benefit=100000)
    R = life.solve(fun, target=4500, guess=[400, 800])
    soa(500, R, 6.36)

    ## SOA Question 6.37:  (D) 820
    sult = SULT()
    benefits = sult.whole_life_insurance(35, b=50000 + 100)
    expenses = sult.immediate_annuity(35, b=100)
    a = sult.temporary_annuity(35, t=10)
    P = (benefits + expenses) / a
    soa(820, P, 6.37)

    ## SOA Question 6.38:  (B) 11.3
    x, n = 0, 10
    life = Recursion(interest=dict(i=0.05))
    life.set_A(0.192, x=x, t=n, endowment=1, discrete=False)
    life.set_E(0.172, x=x, t=n)
    a = life.temporary_annuity(x, t=n, discrete=False)

    def fun(a):   # solve for discrete annuity, given continuous
        life = Recursion(interest=dict(i=0.05), verbose=False)
        life.set_a(a, x=x, t=n).set_E(0.172, x=x, t=n)
        return UDD(m=0, life=life).temporary_annuity(x, t=n)
    a = life.solve(fun, target=a, guess=a)  # discrete annuity
    P = life.gross_premium(a=a, A=0.192, benefit=1000)
    soa(11.3, P, 6.38)

    ## SOA Question 6.39:  (A) 29
    sult = SULT()
    P40 = sult.premium_equivalence(sult.whole_life_insurance(40), b=1000)
    P80 = sult.premium_equivalence(sult.whole_life_insurance(80), b=1000)
    p40 = sult.p_x(40, t=10)
    p80 = sult.p_x(80, t=10)
    P = (P40 * p40 + P80 * p80) / (p80 + p40)
    soa(29, P, 6.39)

    ## SOA Question 6.40: (C) 116 
    # - standard formula discounts/accumulates by too much (i should be smaller)
    x = 0
    life = Recursion(interest=dict(i=0.06)).set_a(7, x=x+1).set_q(0.05, x=x)
    a = life.whole_life_annuity(x)
    A = 110 * a / 1000
    life = Recursion(interest=dict(i=0.06)).set_A(A, x=x).set_q(0.05, x=x)
    A1 = life.whole_life_insurance(x+1)
    P = life.gross_premium(A=A1 / 1.03, a=7) * 1000
    soa(116, P, "6.40")

    ## SOA Question 6.41:  (B) 1417
    x = 0
    life = LifeTable(interest=dict(i=0.05), q={x:.01, x+1:.02}).fill()
    a = 1 + life.E_x(x, t=1) * 1.01
    A = (life.deferred_insurance(x, u=0, t=1) 
         + 1.01 * life.deferred_insurance(x, u=1, t=1))
    P = 100000 * A / a
    soa(1417, P, 6.41)

    ## SOA Question 6.42:  (D) 0.113
    x = 0
    life = ConstantForce(interest=dict(delta=0.06), mu=0.06)
    policy = life.Policy(discrete=True, premium=315.8, 
                         T=3, endowment=1000, benefit=1000)
    L = [life.L_from_t(t, policy=policy) for t in range(3)]    # L(t)
    Q = [life.q_x(x, u=u, t=1) for u in range(3)]        # prob(die in year t)
    Q[-1] = 1 - sum(Q[:-1])  # follows SOA Solution incorrect treat endowment!
    p = sum([q for (q, l) in zip (Q, L) if l > 0])
    soa(0.113, p, 6.42)

    ## SOA Question 6.43:  (C) 170
    sult = SULT()
    a = sult.temporary_annuity(30, t=5)
    A = sult.term_insurance(30, t=10)
    other_expenses = 4 * sult.deferred_annuity(30, u=5, t=5)
    P = sult.gross_premium(a=a, A=A, benefit=200000, initial_premium=0.35,
                           initial_policy=8 + other_expenses, renewal_policy=4,
                           renewal_premium=0.15)
    soa(170, P, 6.43)

    ## SOA Question 6.44:  (D) 2.18
    life = Recursion(interest=dict(i=0.05)).set_IA(0.15, x=50, t=10)
    life.set_a(17, x=50).set_a(15, x=60).set_E(0.6, x=50, t=10)
    A = life.deferred_insurance(50, u=10)
    IA = life.increasing_insurance(50, t=10)
    a = life.temporary_annuity(50, t=10)
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=100)
    soa(2.2, P, 6.44)

    ## SOA Question 6.45:  (E) 690
    life = SULT(udd=True)
    policy = life.Policy(benefit=100000, premium=560, discrete=False)
    p = life.L_from_prob(35, prob=0.75, policy=policy)
    soa(690, p, 6.45)

    ## SOA Question 6.46:  (E) 208
    life = Recursion(interest=dict(i=0.05)).set_IA(0.51213, x=55, t=10)
    life.set_a(12.2758, x=55).set_a(7.4575, x=55, t=10)
    A = life.deferred_annuity(55, u=10)
    IA = life.increasing_insurance(55, t=10)
    a = life.temporary_annuity(55, t=10)
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=300)
    soa(208, P, 6.46)

    ## SOA Question 6.47:  (D) 66400
    sult = SULT()
    a = sult.temporary_annuity(70, t=10)
    A = sult.deferred_annuity(70, u=10)
    P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
                           renewal_premium=0.05)
    soa(66400, P, 6.47)

    ## SOA Question 6.48:  (A) 3195 -- example of deep insurance recursion
    x = 0
    life = Recursion(interest=dict(i=0.06), depth=5).set_p(.95, x=x, t=5)
    life.set_q(.02, x=x+5).set_q(.03, x=x+6).set_q(.04, x=x+7)
    a = 1 + life.E_x(x, t=5)
    A = life.deferred_insurance(x, u=5, t=3)
    P = life.gross_premium(A=A, a=a, benefit=100000)
    soa(3195, P, 6.48)

    ## SOA Question 6.49:  (C) 86
    sult = SULT(udd=True)
    a = UDD(m=12, life=sult).temporary_annuity(40, t=20)
    A = sult.whole_life_insurance(40, discrete=False)
    P = sult.gross_premium(a=a, A=A, benefit=100000, initial_policy=200,
                           renewal_premium=0.04, initial_premium=0.04) / 12
    soa(86, P, 6.49)

    ## SOA Question 6.50:  (A) -47000
    life = SULT()
    P = life.premium_equivalence(a=life.whole_life_annuity(35), b=1000) 
    a = life.deferred_annuity(35, u=1, t=1)
    A = life.term_insurance(35, t=1, b=1000)
    cash = (A - a * P) * 10000 / life.interest.v
    soa(-47000, cash, "6.50")

    ## SOA Question 6.51:  (D) 34700
    life = Recursion()
    life.set_DA(0.4891, x=62, t=10)
    life.set_A(0.0910, x=62, t=10)
    life.set_a(12.2758, x=62)
    life.set_a(7.4574, x=62, t=10)
    IA = life.increasing_insurance(62, t=10)
    A = life.deferred_annuity(62, u=10)
    a = life.temporary_annuity(62, t=10)
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=50000)
    soa(34700, P, 6.51)

    ## SOA Question 6.52:  (D) 50.80 -- hint: set face value benefits to 0
    sult = SULT()
    a = sult.temporary_annuity(45, t=10)
    other_cost = 10 * sult.deferred_annuity(45, u=10)
    P = sult.gross_premium(a=a, A=0, benefit=0,    # set face value H = 0
                           initial_premium=1.05, renewal_premium=0.05,
                           initial_policy=100 + other_cost, renewal_policy=20)
    soa(50.8, P, 6.52)

    ## SOA Question 6.53:  (D) 720
    x = 0
    life = LifeTable(interest=dict(i=0.08), q={x:.1, x+1:.1, x+2:.1}).fill()
    A = life.term_insurance(x, t=3)
    P = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
    soa(720, P, 6.53)

    ## SOA Question 6.54:  (A) 25440
    life = SULT()
    s = math.sqrt(life.net_policy_variance(45, b=200000))
    soa(25440, s, 6.54)

    ## SOA Question 7.1:  (C) 11150
    life = SULT()
    x, n, t = 40, 20, 10
    A = (life.whole_life_insurance(x+t, b=50000)
         + life.deferred_insurance(x+t, u=n-t, b=50000))
    a = life.temporary_annuity(x+t, t=n-t, b=875)
    L = life.gross_future_loss(A=A, a=a)
    soa(11150, L, 7.1)

    ## SOA Question 7.2:  (C) 1152
    x = 0
    life = Recursion(interest=dict(i=.1)).set_q(0.15, x=x).set_q(0.165, x=x+1)
    life.set_reserves(T=2, endowment=2000)

    def fun(P):  # solve P s.t. V is equal backwards and forwards
        policy = dict(t=1, premium=P, 
                      benefit=lambda t: 2000, reserve_benefit=True)
        return life.t_V_backward(x, **policy) - life.t_V_forward(x, **policy)
    P = life.solve(fun, target=0, guess=[1070, 1230])
    soa(1152, P, 7.2)

    ## SOA Question 7.3:  (E) 730
    x = 0  # x=0 is (90) and interpret every 3 months as t=1 year
    life = LifeTable(interest=dict(i=0.08/4), 
                     l={0:1000, 1:898, 2:800, 3:706}).fill()
    life.set_reserves(T=8, V={3: 753.72})
    life.set_reserves(V={2: life.t_V_forward(x=0, t=2, premium=60*0.9, 
                                             benefit=lambda t: 1000)})
    V = life.t_V_forward(x=0, t=1, premium=0, benefit=lambda t: 1000)
    soa(730, V, 7.3)


    ## SOA Question 7.4:  (B) -74 -- split benefits into two policies
    life = SULT()
    P = life.gross_premium(a=life.whole_life_annuity(40),
                           A=life.whole_life_insurance(40),
                           initial_policy=100, renewal_policy=10,
                           benefit=1000)
    P += life.gross_premium(a=life.whole_life_annuity(40),
                            A=life.deferred_insurance(40, u=11),
                            benefit=4000)   # for deferred portion
    policy = life.Policy(benefit=1000, premium=1.02*P, 
                         renewal_policy=10, initial_policy=100)
    V = life.gross_policy_value(x=40, t=1, policy=policy)
    policy = life.Policy(benefit=4000, premium=0)  
    A = life.deferred_insurance(41, u=10)
    V += life.gross_future_loss(A=A, a=0, policy=policy) # for deferred portion
    soa(-74, V, 7.4)

    ## SOA Question 7.5:  (E) 1900
    x = 0
    life = Recursion(interest=dict(i=0.03), udd=True).set_q(0.04561, x=x+4)
    life.set_reserves(T=3, V={4: 1405.08})
    V = life.r_V_backward(x, s=4, r=0.5, benefit=10000, premium=647.46)
    soa(1900, V, 7.5)

    ## SOA Question 7.6:  (E) -25.4
    life = SULT()
    P = life.net_premium(45, b=2000)
    policy = life.Policy(benefit=2000, initial_premium=.25, renewal_premium=.05,
                         initial_policy=2*1.5 + 30, renewal_policy=2*.5 + 10)
    G = life.gross_premium(a=life.whole_life_annuity(45), **policy.premium_terms)
    gross = life.gross_policy_value(45, t=10, policy=policy.set(premium=G))
    net = life.net_policy_value(45, t=10, b=2000)
    V = gross - net

    soa(-25.4, V, 7.6)    

    ## SOA Question 7.7:  (D) 1110
    x = 0
    life = Recursion(interest=dict(i=0.05)).set_A(0.4, x=x+10)
    a = Woolhouse(m=12, life=life).whole_life_annuity(x+10)
    policy = life.Policy(premium=0, benefit=10000, renewal_policy=100)
    V = life.gross_future_loss(A=0.4, policy=policy.future)
    policy = life.Policy(premium=30*12, renewal_premium=0.05)
    V += life.gross_future_loss(a=a, policy=policy.future)
    soa(1110, V, 7.7)

    ## SOA Question 7.8:  (C) 29.85
    sult = SULT()
    x = 70
    q = {x: [sult.q_x(x+k)*(.7 + .1*k) for k in range(3)] + [sult.q_x(x+3)]}
    life = Recursion(interest=dict(i=.05)).set_q(sult.q_x(70)*.7, x=x)\
                                          .set_reserves(T=3)
    V = life.t_V(x=70, t=1, premium=35.168, benefit=lambda t: 1000)
    soa(29.85, V, 7.8)

    ## SOA Question 7.9:  (A) 38100
    sult = SULT(udd=True)
    x, n, t = 45, 20, 10
    a = UDD(m=12, life=sult).temporary_annuity(x+10, t=n-10)
    A = UDD(m=0, life=sult).endowment_insurance(x+10, t=n-10)
    policy = sult.Policy(premium=253*12, endowment=100000, benefit=100000)
    V = sult.gross_future_loss(A=A, a=a, policy=policy)
    soa(38100, V, 7.9)

    ## SOA Question 7.10: (C) -970
    life = SULT()
    G = 977.6
    P = life.net_premium(45, b=100000)
    policy = life.Policy(benefit=0, premium=G-P, renewal_policy=.02*G + 50)
    V = life.gross_policy_value(45, t=5, policy=policy)
    soa(-970, V, "7.10")

    ## SOA Question 7.11:  (B) 1460
    life=Recursion(interest=dict(i=0.05)).set_a(13.4205, x=55)
    policy=life.Policy(benefit=10000)
    def fun(P):
        return life.L_from_t(t=10, policy=policy.set(premium=P))
    P = life.solve(fun, target=4450, guess=400)
    V = life.gross_policy_value(45, t=10, policy=policy.set(premium=P))
    soa(1460, V, 7.11)

    ## SOA Question 7.12:  (E) 4.09
    benefit = lambda k: 26 - k
    x = 44
    life = Recursion(interest=dict(i=0.04)).set_q(0.15, x=55)
    life.set_reserves(T=25, endowment=1, V={11: 5.})
    def fun(P):  # solve for net premium, from final year recursion
        return life.t_V(x=x, t=24, premium=P, benefit=benefit)
    P = life.solve(fun, target=0.6, guess=0.5)    # solved net premium
    V = life.t_V(x, t=12, premium=P, benefit=benefit)  # recursion formula
    soa(4.09, V, 7.12)


    ## SOA Question 7.13: (A) 180
    life = SULT()
    V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
    soa(180, V, 7.13)

    ## SOA Question 7.14:  (A) 2200
    x = 45
    life = Recursion(interest=dict(i=0.05)).set_q(0.009, x=50)
    life.set_reserves(T=10, V={5: 5500})
    def fun(P):  # solve for net premium, from year 6 reserve
        return life.t_V(x=x, t=6, premium=P*0.96 - 50, 
                        benefit=lambda t: 100000 + 200)
    P = life.solve(fun, target=7100, guess=[2200, 2400])
    soa(2200, P, 7.14)

    ## SOA Question 7.15:  (E) 50.91
    x = 0
    life = Recursion(udd=True, interest=dict(i=0.05)).set_q(0.1, x=x+15)
    life.set_reserves(T=3, V={16: 49.78})
    V = life.r_V_forward(x, s=15, r=0.6, benefit=100)
    soa(50.91, V, 7.15)

    ## SOA Question 7.16:  (D) 380
    life = Select(interest=dict(v=.95), A={86: [683/1000]},
                  q={80+k: [.01*(k+1)] for k in range(6)}).fill()
    x, t, n = 80, 3, 5
    A = life.whole_life_insurance(x+t)
    a = life.temporary_annuity(x+t, t=n-t)
    V = life.gross_future_loss(A=A, a=a, 
                               policy=life.Policy(benefit=1000, premium=130))
    soa(380, V, 7.16)

    ## SOA Question 7.17:  (D) 1.018
    x = 0
    life = Recursion(interest=dict(v=math.sqrt(0.90703)))
    life.set_q(0.02067, x=x+10)
    life.set_A(0.52536, x=x+11)
    life.set_A(0.30783, x=x+11, moment=2)
    A1 = life.whole_life_insurance(x+10)
    A2 = life.whole_life_insurance(x+10, moment=2)
    ratio = (life.insurance_variance(A2=A2, A1=A1) 
             / life.insurance_variance(A2=0.30783, A1=0.52536))
    soa(1.018, ratio, 7.17)

    ## SOA Question 7.18:  (A) 17.1
    x = 10
    life = Recursion(interest=dict(i=0.04)).set_q(0.009, x=x)
    def fun(a):
        return life.set_a(a, x=x).net_policy_value(x, t=1)
    a = life.solve(fun, target=0.012, guess=[17.1, 19.1])
    soa(17.1, a, 7.18)

    ## SOA Question 7.19:  (D) 720
    life = SULT()
    policy = life.Policy(benefit=100000, initial_policy=300, initial_premium=.5,
                         renewal_premium=.1)
    P = life.gross_premium(A=life.whole_life_insurance(40), 
                           **policy.premium_terms)
    A = life.whole_life_insurance(41)
    a = life.immediate_annuity(41)   # after premium and expenses are paid
    V = life.gross_future_loss(A=A, a=a, policy=policy.set(premium=P).future)
    soa(720, V, 7.19)

    ## SOA Question 7.20: (E) -277.23
    life = SULT()
    S = life.FPT_policy_value(35, t=1, b=1000)  # is 0 for FPT at t=0,1
    policy = life.Policy(benefit=1000, initial_premium=.3, initial_policy=300,
                         renewal_premium=.04, renewal_policy=30)
    P = life.gross_premium(A=life.whole_life_insurance(35), 
                           **policy.premium_terms)
    R = life.gross_policy_value(35, t=1, policy=policy.set(premium=P))
    soa(-277.23, R - S, "7.20")

    ## SOA Question 7.21:  (D) 11866
    life = SULT()
    x, t, u = 55, 9, 10
    P = life.gross_premium(IA=0.14743, a=life.temporary_annuity(x, t=u),
                           A=life.deferred_annuity(x, u=u), benefit=1000)
    policy = life.Policy(initial_policy=life.term_insurance(x+t, t=1, b=10*P),
                         premium=P, benefit=1000)
    a = life.temporary_annuity(x+t, t=u-t)
    A = life.deferred_annuity(x+t, u=u-t)
    V = life.gross_future_loss(A=A, a=a, policy=policy)
    soa(11866, V, 7.21)

    ## SOA Question 7.22:  (C) 46.24
    life = PolicyValues(interest=dict(i=0.06))
    policy = life.Policy(benefit=8, premium=1.250)
    def fun(A2):
        return life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    A2 = life.solve(fun, target=20.55, guess=20.55/8**2)
    policy = life.Policy(benefit=12, premium=1.875)
    var = life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    soa(46.2, var, 7.22)

    ## SOA Question 7.23:  (D) 233
    life = Recursion(interest=dict(i=0.04)).set_p(0.995, x=25)
    A = life.term_insurance(25, t=1, b=10000)
    def fun(beta):  # value of premiums in first 20 years must be equal
        return beta * 11.087 + (A - beta) 
    beta = life.solve(fun, target=216 * 11.087, guess=[140, 260])
    soa(233, beta, 7.23)

    ## SOA Question 7.24:  (C) 680
    life = SULT()
    P = life.premium_equivalence(A=life.whole_life_insurance(50), b=1000000)
    soa(680, 11800 - P, 7.24)

    ## SOA Question 7.25:  (B) 3947.37
    life = Select(interest=dict(i=.04), A={55: [.23, .24, .25],
                                           56: [.25, .26, .27],
                                           57: [.27, .28, .29],
                                           58: [.20, .30, .31]})
    V = life.FPT_policy_value(55, t=3, b=100000)
    soa(3950, V, 7.25)

    ## SOA Question 7.26:  (D) 28540 -- backward-forward reserve recursion
    x = 0
    life = Recursion(interest=dict(i=.05)).set_p(0.85, x=x).set_p(0.85, x=x+1)
    life.set_reserves(T=2, endowment=50000)
    benefit = lambda k: k*25000
    def fun(P):  # solve P s.t. V is equal backwards and forwards
        policy = dict(t=1, premium=P, benefit=benefit, reserve_benefit=True)
        return life.t_V_backward(x, **policy) - life.t_V_forward(x, **policy)
    P = life.solve(fun, target=0, guess=[27650, 28730])
    soa(28540, P, 7.26)

    ## SOA Question 7.27:  (B) 213
    x = 0
    life = Recursion(interest=dict(i=0.03)).set_q(0.008, x=x)
    life.set_reserves(V={0: 0})
    def fun(G):  # Solve gross premium from expense reserves equation
        return life.t_V(x=x, t=1, premium=G-187, benefit=lambda t: 0, 
                                 per_policy=10 + 0.25*G)
    G = life.solve(fun, target=-38.70, guess=[200, 252])
    soa(213, G, 7.27)

    ## SOA Question 7.28:  (D) 24.3
    life = SULT()
    PW = life.net_premium(65, b=1000)   # 20_V=0 => P+W is net premium for A_65
    P = life.net_premium(45, t=20, b=1000)  # => P is net premium for A_45:20
    soa(24.3, PW - P, 7.28)

    ## SOA Question 7.29:  (E) 2270
    x = 0
    life = Recursion(interest=dict(i=0.04)).set_a(14.8, x=x)\
                                           .set_a(11.4, x=x+10)
    def fun(B):   # Solve for benefit B given net 10_V = 2290
        return life.net_policy_value(x, t=10, b=B)
    B = life.solve(fun, target=2290, guess=2290*10)
    policy = life.Policy(initial_policy=30, renewal_policy=5, benefit=B)
    G = life.gross_premium(a=life.whole_life_annuity(x), **policy.premium_terms)
    V = life.gross_policy_value(x, t=10, policy=policy.set(premium=G))
    soa(2270, V, 7.29)

    ## SOA Question 7.30:  (E) 9035
    policy = SULT.Policy(premium=0, benefit=10000)  # premiums=0 after t=10
    L = SULT().gross_policy_value(35, policy=policy)
    V = SULT(interest=dict(i=0)).gross_policy_value(35, policy=policy) # 10000
    soa(9035, V-L, "7.30")

    ## SOA Question 7.31:  (E) 0.310
    x = 0
    life = Reserves().set_reserves(T=3)
    G = 368.05
    def fun(P):  # solve net premium from expense reserve equation
        return life.t_V(x=x, t=2, premium=G-P, benefit=lambda t: 0, 
                                 per_policy=5 + .08*G)
    P = life.solve(fun, target=-23.64, guess=[.29, .31]) / 1000
    soa(0.310, P, 7.31)

    ## SOA Question 7.32:  (B) 1.4
    life = PolicyValues(interest=dict(i=0.06))
    policy = life.Policy(benefit=1, premium=0.1)
    def fun(A2):
        return life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    A2 = life.solve(fun, target=0.455, guess=0.455)
    policy = life.Policy(benefit=2, premium=0.16)
    variance = life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    soa(1.39, variance, 7.32)

    print(soa.summary())

