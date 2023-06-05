# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3.10.5 ('env3.10')
#     language: python
#     name: python3
# ---

# + [markdown] id="11569414"
# # Sample Solutions and Hints
#
# __actuarialmath -- Life Contingent Risks with Python__
#
# This package implements fundamental methods for modeling life contingent risks, and closely follows traditional topics covered in actuarial exams and standard texts such as the "Fundamentals of Actuarial Math - Long-term" exam syllabus by the Society of Actuaries, and "Actuarial Mathematics for Life Contingent Risks" by Dickson, Hardy and Waters.  These code chunks demonstrate how to solve each of the sample FAM-L exam questions released by the SOA.
#
# Sources:
#
# - actuarialmath [github repo](https://github.com/terence-lim/actuarialmath.git)
#
# - [online tutorial](https://terence-lim.github.io/actuarialmath-tutorial/) or [pdf copy](https://terence-lim.github.io/notes/actuarialmath-tutorial.pdf/)
#
# - SOA FAM-L Sample Solutions: [copy retrieved Aug 2022](https://terence-lim.github.io/notes/2022-10-exam-fam-l-sol.pdf)
#
# - SOA FAM-L Sample Questions: [copy retrieved Aug 2022](https://terence-lim.github.io/notes/2022-10-exam-fam-l-quest.pdf)
#

# + colab={"base_uri": "https://localhost:8080/"} id="i9j4jVPE-Fpk" outputId="5ffeb283-f49e-4be6-cdb4-50c727350d88"
# # ! pip install actuarialmath

# + id="a0729fd1"
"""Solutions code and hints for SOA FAM-L sample questions

MIT License.  Copyright 2022-2023, Terence Lim
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from actuarialmath.interest import Interest
from actuarialmath.life import Life
from actuarialmath.survival import Survival
from actuarialmath.lifetime import Lifetime
from actuarialmath.fractional import Fractional
from actuarialmath.insurance import Insurance
from actuarialmath.annuity import Annuity
from actuarialmath.premiums import Premiums
from actuarialmath.policyvalues import PolicyValues, Contract
from actuarialmath.reserves import Reserves
from actuarialmath.recursion import Recursion
from actuarialmath.lifetable import LifeTable
from actuarialmath.sult import SULT
from actuarialmath.selectlife import SelectLife
from actuarialmath.mortalitylaws import MortalityLaws, Beta, Uniform, Makeham, Gompertz
from actuarialmath.constantforce import ConstantForce
from actuarialmath.extrarisk import ExtraRisk
from actuarialmath.mthly import Mthly
from actuarialmath.udd import UDD
from actuarialmath.woolhouse import Woolhouse

# + [markdown] id="c2b4cbf3"
# __Helper to compare computed answers to expected solutions__

# + id="903e972b"
class IsClose:
    """Helper class for testing and reporting if two values are close"""
    def __init__(self, rel_tol : float = 0.01, score : bool = False,
                 verbose: bool = False):
        self.den = self.num = 0
        self.score = score      # whether to count INCORRECTs instead of assert
        self.verbose = verbose  # whether to run silently
        self.incorrect = []     # to keep list of messages for INCORRECT
        self.tol = rel_tol

    def __call__(self, solution, answer, question="", rel_tol=None):
        """Compare solution to answer within relative tolerance

        Args:
          solution (str | numeric) : gold label
          answer (str | numeric) : computed answer
          question (str) : label to associate with this test
          rel_tol (float) : relative tolerance to be considered close
        """
        if isinstance(solution, str):
            isclose = (solution == answer)
        else:
            isclose = math.isclose(solution, answer, rel_tol=rel_tol or self.tol)
        self.den += 1
        self.num += isclose
        msg = f"{question} {solution}: {answer}"
        if self.verbose:
            print("-----", msg, "[OK]" if isclose else "[INCORRECT]", "-----")
        if not self.score:
            assert isclose, msg
        if not isclose:
            self.incorrect.append(msg)
        return isclose

    def __str__(self):
        """Display cumulative score and errors"""
        return f"Passed: {self.num}/{self.den}\n" + "\n".join(self.incorrect)
isclose = IsClose(0.01, score=False, verbose=True)

# + [markdown] id="831d29e6"
# ## 1 Tables
#
#
# These tables are provided in the FAM-L exam
# - Interest Functions at i=0.05
# - Normal Distribution Table
# - Standard Ultimate Life Table
#
# but you actually do not need them here!

# + colab={"base_uri": "https://localhost:8080/", "height": 224} id="a6dc0057" outputId="6bedde0d-a830-4a3b-e6f5-da8508c35a21"
print("Interest Functions at i=0.05")
UDD.interest_frame()

# + colab={"base_uri": "https://localhost:8080/"} id="b3de4194" outputId="f79d8eb3-8e4a-4f82-eea1-74fb88c980b7"
print("Values of z for selected values of Pr(Z<=z)")
print(Life.quantiles_frame().to_string(float_format=lambda x: f"{x:.3f}"))

# + colab={"base_uri": "https://localhost:8080/", "height": 441} id="1c334975" outputId="e82e6c6a-822f-4ffa-c8a3-75895fdfd0b7"
print("Standard Ultimate Life Table at i=0.05")
SULT().frame()


# + [markdown] id="f7eb3c8d"
# ## 2 Survival models

# + [markdown] id="c41c8385"
# SOA Question 2.1: (B) 2.5
# - derive formula for $\mu$ from given survival function
# - solve for $\omega$ given $\mu_{65}$
# - calculate $e$ by summing survival probabilities
#

# + colab={"base_uri": "https://localhost:8080/"} id="59f81e7f" outputId="c819294e-6baa-403d-c29f-f81a055fda34"
life = Lifetime()
def mu_from_l(omega):   # first solve for omega, given mu_65 = 1/180            
    return life.set_survival(l=lambda x,s: (1 - (x+s)/omega)**0.25).mu_x(65)
omega = int(life.solve(mu_from_l, target=1/180, grid=100))
e = life.set_survival(l=lambda x,s:(1 - (x + s)/omega)**.25, maxage=omega)\
        .e_x(106)       # then solve expected lifetime from omega              
isclose(2.5, e, question="Q2.1")

# + [markdown] id="f2310896"
# SOA Question 2.2: (D) 400
# - calculate survival probabilities for the two scenarios
# - apply conditional variance formula (or mixed distribution)

# + colab={"base_uri": "https://localhost:8080/"} id="1f07bbbd" outputId="b91211ff-271c-4c32-f5ac-ebb1b6b1c071"
p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
std = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
isclose(400, std, question="Q2.2")

# + [markdown] id="246aaf2d"
# SOA Question 2.3: (A) 0.0483
# 1. Derive formula for $f$ given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="e36facdd" outputId="dd66846d-4fae-4377-faa8-af496b5689ba"
B, c = 0.00027, 1.1
S = lambda x,s,t: math.exp(-B * c**(x+s) * (c**t - 1)/math.log(c))
life = Survival().set_survival(S=S)
f = life.f_x(x=50, t=10)
isclose(0.0483, f, question="Q2.3")

# + [markdown] id="166f7d31"
# SOA Question 2.4: (E) 8.2
# - derive survival probability function $_tp_x$ given $_tq_0$
# - compute $\overset{\circ}{e}$ by integration
#

# + colab={"base_uri": "https://localhost:8080/"} id="a6412173" outputId="4fd51cd2-d372-4ac0-d7b5-d83963746e1c"
def l(x, s): return 0. if (x+s) >= 100 else 1 - ((x + s)**2) / 10000.
e = Lifetime().set_survival(l=l).e_x(75, t=10, curtate=False)
isclose(8.2, e, question="Q2.4")

# + [markdown] id="b73ac219"
# SOA Question 2.5:  (B) 37.1
# - solve for $e_{40}$ from limited lifetime formula
# - compute $e_{41}$ using backward recursion

# + colab={"base_uri": "https://localhost:8080/"} id="7485cb2a" outputId="32aac3dc-adca-4f03-c812-13a7cce815e9"
life = Recursion(verbose=False).set_e(25, x=60, curtate=True)\
                               .set_q(0.2, x=40, t=20)\
                               .set_q(0.003, x=40)
def fun(e):   # solve e_40 from e_40:20 = e_40 - 20_p_40 e_60
    return life.set_e(e, x=40, curtate=True)\
               .e_x(x=40, t=20, curtate=True)
e40 = life.solve(fun, target=18, grid=[36, 41])
life.verbose=True
fun(e40)
e41 = life.e_x(41, curtate=True)
isclose(37.1, e41, question="Q2.5")

# + [markdown] id="b626c732"
# SOA Question 2.6: (C) 13.3
# - derive force of mortality function $\mu$ from given survival function
#

# + colab={"base_uri": "https://localhost:8080/"} id="b4a5e978" outputId="7c2a114c-5048-4d02-c2e7-5800eb3d632c"
life = Survival().set_survival(l=lambda x,s: (1 - (x+s)/60)**(1/3))
mu = 1000 * life.mu_x(35)
isclose(13.3, mu, question="Q2.6")

# + [markdown] id="3406e6fd"
# SOA Question 2.7: (B) 0.1477
# - calculate from given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="1cbb1f35" outputId="aa3c294a-428d-4a24-8ebb-2f55b1ee937f"
l = lambda x,s: (1-((x+s)/250) if (x+s)<40 else 1-((x+s)/100)**2)
q = Survival().set_survival(l=l).q_x(30, t=20)
isclose(0.1477, q, question="Q2.7")

# + [markdown] id="59f19984"
# SOA Question 2.8: (C) 0.94
# - relate $p_{male}$ and $p_{female}$ through the common term $\mu$ and the given proportions
#

# + colab={"base_uri": "https://localhost:8080/"} id="d9433c52" outputId="84e65aad-9d80-4af3-e417-1e7fbfcb4d80"
def fun(mu):  # Solve first for mu, given ratio of start and end proportions
    male = Survival().set_survival(mu=lambda x,s: 1.5 * mu)
    female = Survival().set_survival(mu=lambda x,s: mu)
    return (75 * female.p_x(0, t=20)) / (25 * male.p_x(0, t=20))
mu = Survival.solve(fun, target=85/15, grid=[0.89, 0.99])
p = Survival().set_survival(mu=lambda x,s: mu).p_x(0, t=1)
isclose(0.94, p, question="Q2.8")

# + [markdown] id="42e1818d"
# ## 3 Life tables and selection

# + [markdown] id="8158646b"
# SOA Question 3.1:  (B) 117
# - interpolate with constant force of maturity
#

# + colab={"base_uri": "https://localhost:8080/"} id="07539d91" outputId="88aa2e68-27b1-470a-8c28-a39e47d42521"
life = SelectLife().set_table(l={60: [80000, 79000, 77000, 74000],
                                 61: [78000, 76000, 73000, 70000],
                                 62: [75000, 72000, 69000, 67000],
                                 63: [71000, 68000, 66000, 65000]})
q = 1000 * life.q_r(60, s=0, r=0.75, t=3, u=2)
isclose(117, q, question="Q3.1")

# + [markdown] id="17e5b9f4"
# SOA Question 3.2:  (D) 14.7
# - UDD $\Rightarrow \overset{\circ}{e}_{x} = e_x + 0.5$
# - fill select table using curtate expectations 
#

# + colab={"base_uri": "https://localhost:8080/"} id="b3c05afd" outputId="4c9c4a7f-71b6-4585-ab64-5c6d40347584"
e_curtate = Fractional.e_approximate(e_complete=15)
life = SelectLife(udd=True).set_table(l={65: [1000, None,],
                                         66: [955, None]},
                                      e={65: [e_curtate, None]},
                                      d={65: [40, None,],
                                         66: [45, None]})
e = life.e_r(66)
isclose(14.7, e, question="Q3.2")

# + [markdown] id="fb02d76f"
# SOA Question 3.3:  (E) 1074
# - interpolate lives between integer ages with UDD

# + colab={"base_uri": "https://localhost:8080/"} id="9576af60" outputId="a1c2617e-a3b7-40f4-c977-9dac5b2bfa40"
life = SelectLife().set_table(l={50: [99, 96, 93],
                                 51: [97, 93, 89],
                                 52: [93, 88, 83],
                                 53: [90, 84, 78]})
q = 10000 * life.q_r(51, s=0, r=0.5, t=2.2)
isclose(1074, q, question="Q3.3")

# + [markdown] id="2247b56f"
# SOA Question 3.4:  (B) 815
# - compute portfolio percentile with N=4000, and mean and variance  from binomial distribution

# + colab={"base_uri": "https://localhost:8080/"} id="fb29aeca" outputId="73eb5435-6450-4a03-8267-59193145e5f0"
sult = SULT()
mean = sult.p_x(25, t=95-25)
var = sult.bernoulli(mean, variance=True)
pct = sult.portfolio_percentile(N=4000, mean=mean, variance=var, prob=0.1)
isclose(815, pct, question="Q3.4")

# + [markdown] id="a989a344"
# SOA Question 3.5:  (E) 106
# - compute mortality rates by interpolating lives between integer ages, with UDD and constant force of mortality assumptions

# + colab={"base_uri": "https://localhost:8080/"} id="4a01bc25" outputId="ddcc8fc2-dbdb-4e5d-89c8-ea1965e08512"
l = [99999, 88888, 77777, 66666, 55555, 44444, 33333, 22222]
a = LifeTable(udd=True).set_table(l={age:l for age,l in zip(range(60, 68), l)})\
                       .q_r(60, u=3.4, t=2.5)
b = LifeTable(udd=False).set_table(l={age:l for age,l in zip(range(60, 68), l)})\
                        .q_r(60, u=3.4, t=2.5)
isclose(106, 100000 * (a - b), question="Q3.5")

# + [markdown] id="cc6f9e8f"
# SOA Question 3.6:  (D) 15.85
# - apply recursion formulas for curtate expectation
#

# + colab={"base_uri": "https://localhost:8080/"} id="ab3dfad5" outputId="8f050db5-93d6-4245-f27a-66392d065220"
e = SelectLife().set_table(q={60: [.09, .11, .13, .15],
                              61: [.1, .12, .14, .16],
                              62: [.11, .13, .15, .17],
                              63: [.12, .14, .16, .18],
                              64: [.13, .15, .17, .19]},
                           e={61: [None, None, None, 5.1]})\
                .e_x(61)
isclose(5.85, e, question="Q3.6")

# + [markdown] id="6e9d1f8c"
# SOA Question 3.7: (b) 16.4
# - use deferred mortality formula
# - use chain rule for survival probabilities,
# - interpolate between integer ages with constant force of mortality
#

# + colab={"base_uri": "https://localhost:8080/"} id="b745cec2" outputId="98929f53-6809-4c03-dfb3-b788d6d7d8de"
life = SelectLife().set_table(q={50: [.0050, .0063, .0080],
                                 51: [.0060, .0073, .0090],
                                 52: [.0070, .0083, .0100],
                                 53: [.0080, .0093, .0110]})
q = 1000 * life.q_r(50, s=0, r=0.4, t=2.5)
isclose(16.4, q, question="Q3.7")

# + [markdown] id="b66e6e89"
# SOA Question 3.8:  (B) 1505
# - compute portfolio means and variances from sum of 2000 independent members' means and variances of survival.
#

# + colab={"base_uri": "https://localhost:8080/"} id="907c1755" outputId="09dadcaa-d77f-4a77-86fe-29a3db9fc85b"
sult = SULT()
p1 = sult.p_x(35, t=40)
p2 = sult.p_x(45, t=40)
mean = sult.bernoulli(p1) * 1000 + sult.bernoulli(p2) * 1000
var = (sult.bernoulli(p1, variance=True) * 1000 
       + sult.bernoulli(p2, variance=True) * 1000)
pct = sult.portfolio_percentile(mean=mean, variance=var, prob=.95)
isclose(1505, pct, question="Q3.8")

# + [markdown] id="71438d84"
# SOA Question 3.9:  (E) 3850
# - compute portfolio means and variances as sum of 4000 independent members' means and variances (of survival)
# - retrieve normal percentile
#

# + colab={"base_uri": "https://localhost:8080/"} id="5e87a932" outputId="004b7caf-8522-40ee-b49d-e3c3b86a0f60"
sult = SULT()
p1 = sult.p_x(20, t=25)
p2 = sult.p_x(45, t=25)
mean = sult.bernoulli(p1) * 2000 + sult.bernoulli(p2) * 2000
var = (sult.bernoulli(p1, variance=True) * 2000 
       + sult.bernoulli(p2, variance=True) * 2000)
pct = sult.portfolio_percentile(mean=mean, variance=var, prob=.99)
isclose(3850, pct, question="Q3.9")

# + [markdown] id="5430d382"
# SOA Question 3.10:  (C) 0.86
# - reformulate the problem by reversing time: survival to year 6 is calculated in reverse as discounting by the same number of years. 
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="cfd72881" outputId="1507d135-1dea-4ed7-9a1a-403c683ad856"
interest = Interest(v=0.75)
L = 35*interest.annuity(t=4, due=False) + 75*interest.v_t(t=5)
interest = Interest(v=0.5)
R = 15*interest.annuity(t=4, due=False) + 25*interest.v_t(t=5)
isclose(0.86, L / (L + R), question="Q3.10")

# + [markdown] id="a51f9f7a"
# SOA Question 3.11:  (B) 0.03
# - calculate mortality rate by interpolating lives assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="ced9708b" outputId="687b3111-f37d-4e01-9823-c3958ca5a49a"
life = LifeTable(udd=True).set_table(q={50//2: .02, 52//2: .04})
q = life.q_r(50//2, t=2.5/2)
isclose(0.03, q, question="Q3.11")

# + [markdown] id="6dae2d07"
# SOA Question 3.12: (C) 0.055 
# - compute survival probability by interpolating lives assuming constant force
#

# + colab={"base_uri": "https://localhost:8080/"} id="7e5ce19d" outputId="e5513e9b-ac5e-4024-e9a7-b6831bb27020"
life = SelectLife(udd=False).set_table(l={60: [10000, 9600, 8640, 7771],
                                          61: [8654, 8135, 6996, 5737],
                                          62: [7119, 6549, 5501, 4016],
                                          63: [5760, 4954, 3765, 2410]})
q = life.q_r(60, s=1, t=3.5) - life.q_r(61, s=0, t=3.5)               
isclose(0.055, q, question="Q3.12")

# + [markdown] id="459b6c3d"
# SOA Question 3.13:  (B) 1.6
# - compute curtate expectations using recursion formulas
# - convert to complete expectation assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="8db335e1" outputId="9ce4b4f4-85e5-45f4-b1ad-63e3b7c13491"
life = SelectLife().set_table(l={55: [10000, 9493, 8533, 7664],
                                 56: [8547, 8028, 6889, 5630],
                                 57: [7011, 6443, 5395, 3904],
                                 58: [5853, 4846, 3548, 2210]},
                              e={57: [None, None, None, 1]})
e = life.e_r(58, s=2)
isclose(1.6, e, question="Q3.13")

# + [markdown] id="b784697d"
# SOA Question 3.14:  (C) 0.345
# - compute mortality by interpolating lives between integer ages assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="39107fed" outputId="7bdf2ff4-5105-408f-84dd-6d890ca4ca67"
life = LifeTable(udd=True).set_table(l={90: 1000, 93: 825},
                                     d={97: 72},
                                     p={96: .2},
                                     q={95: .4, 97: 1})
q = life.q_r(90, u=93-90, t=95.5 - 93)
isclose(0.345, q, question="Q3.14")

# + [markdown] id="0c0888ae"
# ## 4 Insurance benefits

# + [markdown] id="52a66927"
# SOA Question 4.1:  (A) 0.27212
# - solve EPV as sum of term and deferred insurance
# - compute variance as difference of second moment and first moment squared
#

# + colab={"base_uri": "https://localhost:8080/"} id="7a9f699d" outputId="6018551e-08ce-490a-b38a-d5c1edca7ae6"
life = Recursion().set_interest(i=0.03)
life.set_A(0.36987, x=40).set_A(0.62567, x=60)
life.set_E(0.51276, x=40, t=20).set_E(0.17878, x=60, t=20)
Z2 = 0.24954
A = (2 * life.term_insurance(40, t=20) + life.deferred_insurance(40, u=20))
std = math.sqrt(life.insurance_variance(A2=Z2, A1=A))
isclose(0.27212, std, question="Q4.1")

# + [markdown] id="0255b59a"
# SOA Question 4.2:  (D) 0.18
# - calculate Z(t) and deferred mortality for each half-yearly t
# - sum the deferred mortality probabilities for periods when PV > 277000 
#

# + colab={"base_uri": "https://localhost:8080/"} id="2145a29e" outputId="f68f055b-c4bf-462d-d679-26e17465379f"
life = LifeTable(udd=False).set_table(q={0: .16, 1: .23})\
                           .set_interest(i_m=.18, m=2)
mthly = Mthly(m=2, life=life)
Z = mthly.Z_m(0, t=2, benefit=lambda x,t: 300000 + t*30000*2)
p = Z[Z['Z'] >= 277000]['q'].sum()
isclose(0.18, p, question="Q4.2")

# + [markdown] id="f749bbb5"
# SOA Question 4.3: (D) 0.878
# - solve $q_{61}$ from endowment insurance EPV formula
# - solve $A_{60:\overline{3|}}$ with new $i=0.045$ as EPV of endowment insurance benefits.
#

# + colab={"base_uri": "https://localhost:8080/"} id="db579f3b" outputId="19a03ee3-dd68-42f4-dc46-221ca7063dca"
life = Recursion(verbose=False).set_interest(i=0.05).set_q(0.01, x=60)
def fun(q):   # solve for q_61
    return life.set_q(q, x=61).endowment_insurance(60, t=3)
life.solve(fun, target=0.86545, grid=0.01)
A = life.set_interest(i=0.045).endowment_insurance(60, t=3)
isclose(0.878, A, question="Q4.3")

# + [markdown] id="de2d0427"
# SOA Question 4.4  (A) 0.036
# - integrate to find EPV of $Z$ and $Z^2$
# - variance is difference of second moment and first moment squared
#

# + colab={"base_uri": "https://localhost:8080/"} id="3fa24393" outputId="04aecb56-aacb-4462-d973-b670a4007ac2"
x = 40
life = Insurance().set_survival(f=lambda *x: 0.025, maxage=x+40)\
                  .set_interest(v_t=lambda t: (1 + .2*t)**(-2))
def benefit(x,t): return 1 + .2 * t
A1 = life.A_x(x, benefit=benefit, discrete=False)
A2 = life.A_x(x, moment=2, benefit=benefit, discrete=False)
var = A2 - A1**2
isclose(0.036, var, question="Q4.4")

# + [markdown] id="2bb789fa"
# SOA Question 4.5:  (C) 35200
# - interpolate between integer ages with UDD, and find lifetime that mortality rate exceeded
# - compute PV of death benefit paid at that time.
#

# + colab={"base_uri": "https://localhost:8080/"} id="3c9d0b1e" outputId="acbc851b-6764-4f4a-8bc6-24993db5166c"
sult = SULT(udd=True).set_interest(delta=0.05)
Z = 100000 * sult.Z_from_prob(45, 0.95, discrete=False)
isclose(35200, Z, question="Q4.5")

# + [markdown] id="1792b7aa"
# SOA Question 4.6:  (B) 29.85
# - calculate adjusted mortality rates
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="f31ee601" outputId="a630fe03-52f5-435e-8d76-836832867aca"
sult = SULT()
life = LifeTable().set_interest(i=0.05)\
                  .set_table(q={70+k: .95**k * sult.q_x(70+k) for k in range(3)})
A = life.term_insurance(70, t=3, b=1000)
isclose(29.85, A, question="Q4.6")


# + [markdown] id="230429ad"
# SOA Question 4.7:  (B) 0.06
# - use Bernoulli shortcut formula for variance of pure endowment Z 
# - solve for $i$, since $p$ is given.

# + colab={"base_uri": "https://localhost:8080/"} id="f38c4ab6" outputId="1963f0c7-ffc6-41c5-bbbf-d67fd0891b39"
def fun(i):
    life = Recursion(verbose=False).set_interest(i=i)\
                                   .set_p(0.57, x=0, t=25)
    return 0.1*life.E_x(0, t=25) - life.E_x(0, t=25, moment=life._VARIANCE)
i = Recursion.solve(fun, target=0, grid=[0.058, 0.066])
isclose(0.06, i, question="Q4.7")

# + [markdown] id="ccb0f3ff"
# SOA Question 4.8  (C) 191
#
# - use insurance recursion with special interest rate $i=0.04$ in first year.
#

# + colab={"base_uri": "https://localhost:8080/"} id="f3ad0bbe" outputId="efc4f3d3-2609-4252-995f-57ab578991fd"
def v_t(t): return 1.04**(-t) if t < 1 else 1.04**(-1) * 1.05**(-t+1)
A = SULT().set_interest(v_t=v_t).whole_life_insurance(50, b=1000)
isclose(191, A, question="Q4.8")

# + [markdown] id="4408c9ef"
# SOA Question 4.9:  (D) 0.5
# - use whole-life, term and endowment insurance relationships.
#

# + colab={"base_uri": "https://localhost:8080/"} id="0ab006d1" outputId="6186bf3e-bd92-4ab3-9054-2648b798a006"
E = Recursion().set_A(0.39, x=35, t=15, endowment=1)\
               .set_A(0.25, x=35, t=15)\
               .E_x(35, t=15)
life = Recursion().set_A(0.32, x=35)\
                  .set_E(E, x=35, t=15)
def fun(A): return life.set_A(A, x=50).term_insurance(35, t=15)
A = life.solve(fun, target=0.25, grid=[0.35, 0.55])
isclose(0.5, A, question="Q4.9")

# + [markdown] id="f46ca953"
# SOA Question 4.10:  (D)
# - draw and compared benefit diagrams
#

# + colab={"base_uri": "https://localhost:8080/", "height": 521} id="14fca3d8" outputId="b81eff4e-af91-4bf7-d399-5139fb35bb5c"
life = Insurance().set_interest(i=0.0).set_survival(S=lambda x,s,t: 1, maxage=40)
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
    life.Z_plot(0, benefit=b, ax=ax[i], color=f"C{i+1}", title='')
    ax[i].legend(["(" + "abcde"[i-1] + ")" if i else "Z"])
z = [sum(abs(b(0, t) - fun(0, t)) for t in range(40)) for b in benefits]
ans = "ABCDE"[np.argmin(z)]
isclose('D', ans, question="Q4.10")

# + [markdown] id="8dae2f7c"
# SOA Question 4.11:  (A) 143385
# - compute endowment insurance = term insurance + pure endowment 
# - apply formula of variance as the difference of second moment and first moment squared.
#

# + colab={"base_uri": "https://localhost:8080/"} id="a3858d16" outputId="c63b4fda-e60f-40f6-a4d3-b09e0c74548e"
A1 = 528/1000   # E[Z1]  term insurance
C1 = 0.209      # E[pure_endowment]
C2 = 0.136      # E[pure_endowment^2]
B1 = A1 + C1    # endowment = term + pure_endowment
def fun(A2):
    B2 = A2 + C2   # double force of interest
    return Insurance.insurance_variance(A2=B2, A1=B1)  
A2 = Insurance.solve(fun, target=15000/(1000*1000), grid=[143400, 279300])
var = Insurance.insurance_variance(A2=A2, A1=A1, b=1000)
isclose(143385, var, question="Q4.11")

# + [markdown] id="4a79514c"
# SOA Question 4.12:  (C) 167 
# - since $Z_1,~Z_2$ are non-overlapping, $E[Z_1~ Z_2] = 0$ for computing $Cov(Z_1, Z_2)$
# - whole life is sum of term and deferred, hence equals variance of components plus twice their covariance
#

# + colab={"base_uri": "https://localhost:8080/"} id="b34e726c" outputId="fec05279-1701-4d88-93ae-33562b90cb5c"
cov = Life.covariance(a=1.65, b=10.75, ab=0)  # E[Z1 Z2] = 0 nonoverlapping
var = Life.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov)
isclose(167, var, question="Q4.12")

# + [markdown] id="ae69b52f"
# SOA Question 4.13:  (C) 350 
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="a838d9f1" outputId="8ca77d7d-f3d6-4486-ce84-07d6a7a25796"
life = SelectLife().set_table(q={65: [.08, .10, .12, .14],
                                 66: [.09, .11, .13, .15],
                                 67: [.10, .12, .14, .16],
                                 68: [.11, .13, .15, .17],
                                 69: [.12, .14, .16, .18]})\
                   .set_interest(i=.04)
A = life.deferred_insurance(65, t=2, u=2, b=2000)
isclose(350, A, question="Q4.13")

# + [markdown] id="3f3b547c"
# SOA Question 4.14:  (E) 390000
# - discount (by interest rate $i=0.05$) the value at the portfolio percentile, of the sum of 400 bernoulli r.v. with survival probability $_{25}p_{60}$
#

# + colab={"base_uri": "https://localhost:8080/"} id="f0e506f8" outputId="146c93cc-c3a1-4537-cba6-4351ae2f019c"
sult = SULT()
p = sult.p_x(60, t=85-60)
mean = sult.bernoulli(p)
var = sult.bernoulli(p, variance=True)
F = sult.portfolio_percentile(mean=mean, variance=var, prob=.86, N=400)
F *= 5000 * sult.interest.v_t(85-60)
isclose(390000, F, question="Q4.14")

# + [markdown] id="c736d0c5"
# SOA Question 4.15  (E) 0.0833 
# - this special benefit function has effect of reducing actuarial discount rate to use in constant force of mortality shortcut formulas
#

# + colab={"base_uri": "https://localhost:8080/"} id="4b7eee09" outputId="488f89c2-79a0-4ef7-96cc-198e738d32a0"
life = Insurance().set_survival(mu=lambda *x: 0.04).set_interest(delta=0.06)
benefit = lambda x,t: math.exp(0.02*t)
A1 = life.A_x(0, benefit=benefit, discrete=False)
A2 = life.A_x(0, moment=2, benefit=benefit, discrete=False)
var = life.insurance_variance(A2=A2, A1=A1)
isclose(0.0833, var, question="Q4.15")

# + [markdown] id="79f63326"
# SOA Question 4.16:  (D) 0.11
# - compute EPV of future benefits with adjusted mortality rates

# + colab={"base_uri": "https://localhost:8080/"} id="3c74f0e6" outputId="e58bd9a6-1a51-4f71-ca4f-056901d3e435"
q = [.045, .050, .055, .060]
q = {50 + x: [q[x] * 0.7 if x < len(q) else None, 
              q[x+1] * 0.8 if x + 1 < len(q) else None, 
              q[x+2] if x + 2 < len(q) else None] 
     for x in range(4)}
life = SelectLife().set_table(q=q).set_interest(i=.04)
A = life.term_insurance(50, t=3)
isclose(0.1116, A, question="Q4.16")

# + [markdown] id="2ab65168"
# SOA Question 4.17:  (A) 1126.7
# - find future lifetime with 50\% survival probability
# - compute EPV of special whole life as sum of term and deferred insurance, that have different benefit amounts before and after median lifetime.

# + colab={"base_uri": "https://localhost:8080/"} id="330ac8db" outputId="995c4c92-efc6-4d62-e056-97e9104550ea"
sult = SULT()
median = sult.Z_t(48, prob=0.5, discrete=False)
def benefit(x,t): return 5000 if t < median else 10000
A = sult.A_x(48, benefit=benefit)
isclose(1130, A, question="Q4.17")

# + [markdown] id="258c80e6"
# SOA Question 4.18  (A) 81873 
# - find values of limits such that integral of lifetime density function equals required survival probability
#

# + colab={"base_uri": "https://localhost:8080/"} id="53795941" outputId="d622625f-e404-4a8d-cc02-dd7ed2d602e6"
def f(x,s,t): return 0.1 if t < 2 else 0.4*t**(-2)
life = Insurance().set_interest(delta=0.05)\
                  .set_survival(f=f, maxage=10)
def benefit(x,t): return 0 if t < 2 else 100000
prob = 0.9 - life.q_x(0, t=2)
T = life.Z_t(0, prob=prob)
Z = life.Z_from_t(T) * benefit(0, T)
isclose(81873, Z, question="Q4.18")

# + [markdown] id="04492903"
# SOA Question 4.19:  (B) 59050
# - calculate adjusted mortality for the one-year select period
# - compute whole life insurance using backward recursion formula
#

# + colab={"base_uri": "https://localhost:8080/"} id="13a8420d" outputId="198dd641-b474-4487-92b4-76f37d376b04"
life = SULT()
q = ExtraRisk(life=life, extra=0.8, risk="MULTIPLY_RATE")['q']
select = SelectLife(periods=1).set_select(s=0, age_selected=True, q=q)\
                              .set_select(s=1, age_selected=False, q=life['q'])\
                              .set_interest(i=.05)\
                              .fill_table()
A = 100000 * select.whole_life_insurance(80, s=0)
isclose(59050, A, question="Q4.19")

# + [markdown] id="c52b272d"
# ## 5 Annuities

# + [markdown] id="4e448f05"
# SOA Question 5.1: (A) 0.705
# - sum of annuity certain and deferred life annuity with constant force of mortality shortcut
# - use equation for PV annuity r.v. Y to infer lifetime
# - compute survival probability from constant force of mortality function.
#

# + colab={"base_uri": "https://localhost:8080/"} id="18b1a0c0" outputId="089a36f0-e632-46fc-a0f3-3834faada710"
life = ConstantForce(mu=0.01).set_interest(delta=0.06)
EY = life.certain_life_annuity(0, u=10, discrete=False)
p = life.p_x(0, t=life.Y_to_t(EY))
isclose(0.705, p, question="Q5.1")  # 0.705

# + [markdown] id="f90b71c6"
# SOA Question 5.2:  (B) 9.64
# - compute term life as difference of whole life and deferred insurance
# - compute twin annuity-due, and adjust to an immediate annuity. 

# + colab={"base_uri": "https://localhost:8080/"} id="206b600b" outputId="dcd5e778-d641-49c7-ce9a-1a427a0cf4f8"
x, n = 0, 10
a = Recursion().set_interest(i=0.05)\
               .set_A(0.3, x)\
               .set_A(0.4, x+n)\
               .set_E(0.35, x, t=n)\
               .immediate_annuity(x, t=n)
isclose(9.64, a, question="Q5.2")

# + [markdown] id="439db468"
# SOA Question 5.3:  (C) 6.239
# - Differential reduces to the the EPV of the benefit payment at the upper time limit.
#

# + colab={"base_uri": "https://localhost:8080/"} id="eeca16c1" outputId="6b109a8b-0194-421b-e7fd-23ea222bde92"
t = 10.5
E = t * SULT().E_r(40, t=t)
isclose(6.239, E, question="Q5.3")

# + [markdown] id="cd3027da"
# SOA Question 5.4:  (A) 213.7
# - compute certain and life annuity factor as the sum of a certain annuity and a deferred life annuity.
# - solve for amount of annual benefit that equals given EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="297311f0" outputId="37039fca-a943-424f-9786-6e827cbd042e"
life = ConstantForce(mu=0.02).set_interest(delta=0.01)
u = life.e_x(40, curtate=False)
P = 10000 / life.certain_life_annuity(40, u=u, discrete=False)
isclose(213.7, P, question="Q5.4") # 213.7

# + [markdown] id="46f357cd"
# SOA Question 5.5: (A) 1699.6
# - adjust mortality rate for the extra risk
# - compute annuity by backward recursion.
#

# + colab={"base_uri": "https://localhost:8080/"} id="9737dc8d" outputId="ecae4e5a-65be-4408-9b24-607b3c3dce0c"
life = SULT()   # start with SULT life table
q = ExtraRisk(life=life, extra=0.05, risk="ADD_FORCE")['q']
select = SelectLife(periods=1).set_select(s=0, age_selected=True, q=q)\
                              .set_select(s=1, age_selected=False, a=life['a'])\
                              .set_interest(i=0.05)\
                              .fill_table()
a = 100 * select['a'][45][0]
isclose(1700, a, question="Q5.5")

# + [markdown] id="3387fd23"
# SOA Question 5.6:  (D) 1200
# - compute mean and variance of EPV of whole life annuity from whole life insurance twin and variance identities. 
# - portfolio percentile of the sum of $N=100$ life annuity payments

# + colab={"base_uri": "https://localhost:8080/"} id="8445b834" outputId="5c7fd1d0-5a71-4949-a0bf-e66a02129970"
life = Annuity().set_interest(i=0.05)
var = life.annuity_variance(A2=0.22, A1=0.45)
mean = life.annuity_twin(A=0.45)
fund = life.portfolio_percentile(mean, var, prob=.95, N=100)
isclose(1200, fund, question="Q5.6")

# + [markdown] id="b7c08c39"
# SOA Question 5.7:  (C) 
# - compute endowment insurance from relationships of whole life, temporary and deferred insurances.
# - compute temporary annuity from insurance twin
# - apply Woolhouse approximation

# + colab={"base_uri": "https://localhost:8080/"} id="93c40a7c" outputId="b98a00cd-5c4d-4df1-c7e2-05c49341b169"
life = Recursion().set_interest(i=0.04)\
                  .set_A(0.188, x=35)\
                  .set_A(0.498, x=65)\
                  .set_p(0.883, x=35, t=30)
mthly = Woolhouse(m=2, life=life, three_term=False)
a = 1000 * mthly.temporary_annuity(35, t=30)
isclose(17376.7, a, question="Q5.7")

# + [markdown] id="0851fa7c"
# SOA Question 5.8: (C) 0.92118
# - calculate EPV of certain and life annuity.
# - find survival probability of lifetime s.t. sum of annual payments exceeds EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="3db058df" outputId="c43253f6-a9e5-42d3-86b9-388c9ffb4339"
sult = SULT()
a = sult.certain_life_annuity(55, u=5)
p = sult.p_x(55, t=math.floor(a))
isclose(0.92118, p, question="Q5.8")

# + [markdown] id="ad7d5d47"
# SOA Question 5.9:  (C) 0.015
# - express both EPV's expressed as forward recursions
# - solve for unknown constant $k$.
#

# + colab={"base_uri": "https://localhost:8080/"} id="1937f550" outputId="c19cd806-89c9-4194-b1f1-0e750fe1ba0a"
x, p = 0, 0.9  # set arbitrary p_x = 0.9
a = Recursion().set_a(21.854, x=x)\
               .set_p(p, x=x)\
               .whole_life_annuity(x+1)
life = Recursion(verbose=False).set_a(22.167, x=x)
def fun(k): return a - life.set_p((1 + k) * p, x=x).whole_life_annuity(x + 1)
k = life.solve(fun, target=0, grid=[0.005, 0.025])
isclose(0.015, k, question="Q5.9")

# + [markdown] id="2bc86ecf"
# ## 6 Premium Calculation

# + [markdown] id="c4aafcee"
# SOA Question 6.1: (D) 35.36
# - calculate IA factor for return of premiums without interest
# - solve net premium such that EPV benefits = EPV premium

# + colab={"base_uri": "https://localhost:8080/"} id="68d68c2e" outputId="4dd66c03-1353-4778-8971-a2ffd8106001"
P = SULT().set_interest(i=0.03)\
          .net_premium(80, t=2, b=1000, return_premium=True)
isclose(35.36, P, question="Q6.1")

# + [markdown] id="8a9f7924"
# SOA Question 6.2: (E) 3604
# - EPV return of premiums without interest = Premium $\times$ IA factor
# - solve for gross premiums such that EPV premiums = EPV benefits and expenses

# + colab={"base_uri": "https://localhost:8080/"} id="cde906a7" outputId="94851ba5-0840-49bc-88f2-2a362ecf458a"
life = Premiums()
A, IA, a = 0.17094, 0.96728, 6.8865
P = life.gross_premium(a=a, A=A, IA=IA, benefit=100000,
                       initial_premium=0.5, renewal_premium=.05,
                       renewal_policy=200, initial_policy=200)
isclose(3604, P, question="Q6.2")

# + [markdown] id="c4fc553b"
# SOA Question 6.3:  (C) 0.390
# - solve lifetime $t$ such that PV annuity certain = PV whole life annuity at age 65
# - calculate mortality rate through the year before curtate lifetime   
#

# + colab={"base_uri": "https://localhost:8080/"} id="1d438209" outputId="35bcc829-f499-4b25-8269-6fc53cd61006"
life = SULT()
t = life.Y_to_t(life.whole_life_annuity(65))
q = 1 - life.p_x(65, t=math.floor(t) - 1)
isclose(0.39, q, question="Q6.3")

# + [markdown] id="8afc2a87"
# SOA Question 6.4:  (E) 1890
#

# + colab={"base_uri": "https://localhost:8080/"} id="5b9948fb" outputId="ede1b846-6d99-43c2-b8fb-8d84639d05ef"
mthly = Mthly(m=12, life=Annuity().set_interest(i=0.06))
A1, A2 = 0.4075, 0.2105
mean = mthly.annuity_twin(A1) * 15 * 12
var = mthly.annuity_variance(A1=A1, A2=A2, b=15 * 12)
S = Annuity.portfolio_percentile(mean=mean, variance=var, prob=.9, N=200) / 200
isclose(1890, S, question="Q6.4")

# + [markdown] id="fd4150b6"
# SOA Question 6.5:  (D) 33
#

# + colab={"base_uri": "https://localhost:8080/"} id="bda89a9a" outputId="063f4003-ee0e-481d-b298-291dc56bf89f"
life = SULT()
P = life.net_premium(30, b=1000)
def gain(k): return life.Y_x(30, t=k) * P - life.Z_x(30, t=k) * 1000
k = min([k for k in range(20, 40) if gain(k) < 0])
isclose(33, k, question="Q6.5")

# + [markdown] id="bba959b2"
# SOA Question 6.6:  (B) 0.79
#

# + colab={"base_uri": "https://localhost:8080/"} id="2a248f2c" outputId="39918b18-0f8f-4783-967c-ab0193b10283"
life = SULT()
P = life.net_premium(62, b=10000)
contract = Contract(premium=1.03*P,
                    renewal_policy=5,
                    initial_policy=5,
                    initial_premium=0.05,
                    benefit=10000)
L = life.gross_policy_value(62, contract=contract)
var = life.gross_policy_variance(62, contract=contract)
prob = life.portfolio_cdf(mean=L, variance=var, value=40000, N=600)
isclose(.79, prob, question="Q6.6")

# + [markdown] id="efd51de5"
# SOA Question 6.7:  (C) 2880
#

# + colab={"base_uri": "https://localhost:8080/"} id="56437e4c" outputId="a738dad9-a925-495e-980e-478171e3f7de"
life = SULT()
a = life.temporary_annuity(40, t=20) 
A = life.E_x(40, t=20)
IA = a - life.interest.annuity(t=20) * life.p_x(40, t=20)
G = life.gross_premium(a=a, A=A, IA=IA, benefit=100000)
isclose(2880, G, question="Q6.7")

# + [markdown] id="af651363"
# SOA Question 6.8:  (B) 9.5
#
# - calculate EPV of expenses as deferred life annuities
# - solve for level premium
#

# + colab={"base_uri": "https://localhost:8080/"} id="e90a196f" outputId="cb9e0ce6-0789-418b-b150-7e0d3d9f1377"
life = SULT()
initial_cost = (50 + 10 * life.deferred_annuity(60, u=1, t=9)
                + 5 * life.deferred_annuity(60, u=10, t=10))
P = life.net_premium(60, initial_cost=initial_cost)
isclose(9.5, P, question="Q6.8")

# + [markdown] id="cc58d89d"
# SOA Question 6.9:  (D) 647
#

# + colab={"base_uri": "https://localhost:8080/"} id="a5ff35d6" outputId="e7696dfd-5c6a-4aef-e0b2-a29803de55d2"
life = SULT()
a = life.temporary_annuity(50, t=10)
A = life.term_insurance(50, t=20)
initial_cost = 25 * life.deferred_annuity(50, u=10, t=10)
P = life.gross_premium(a=a, A=A, benefit=100000,
                       initial_premium=0.42, renewal_premium=0.12,
                       initial_policy=75 + initial_cost, renewal_policy=25)
isclose(647, P, question="Q6.9")

# + [markdown] id="196e0607"
# SOA Question 6.10:  (D) 0.91
#

# + colab={"base_uri": "https://localhost:8080/"} id="a6ea62e1" outputId="bdde781d-23f6-4c97-ab70-efdb3cd5e5bd"
x = 0
life = Recursion(verbose=False).set_interest(i=0.06).set_p(0.975, x=x)
a = 152.85/56.05
life.set_a(a, x=x, t=3)
p1 = life.p_x(x=x+1)                                                  
life.set_p(p1, x=x+1)
def fun(p): 
    return life.set_p(p, x=x+2).term_insurance(x=x, t=3, b=1000)
p = life.solve(fun, target=152.85, grid=0.975)  # finally solve p_x+3, given A_x:3
isclose(0.91, p, question="Q6.10")

# + [markdown] id="1a93e76e"
# SOA Question 6.11:  (C) 0.041
#

# + colab={"base_uri": "https://localhost:8080/"} id="84bc4d87" outputId="1de2ce6f-f4c9-443a-a601-597f4587130b"
life = Recursion().set_interest(i=0.04)
A = life.set_A(0.39788, 51)\
        .set_q(0.0048, 50)\
        .whole_life_insurance(50)
P = life.gross_premium(A=A, a=life.annuity_twin(A=A))
A = life.set_q(0.048, 50).whole_life_insurance(50)
loss = A - life.annuity_twin(A) * P
isclose(0.041, loss, question="Q6.11")

# + [markdown] id="2d504c4a"
# SOA Question 6.12:  (E) 88900
#

# + colab={"base_uri": "https://localhost:8080/"} id="761a8575" outputId="71728a43-b669-4099-a492-23389687fb83"
life = PolicyValues().set_interest(i=0.06)
a = 12
A = life.insurance_twin(a)
contract = Contract(benefit=1000, settlement_policy=20,
                        initial_policy=10, initial_premium=0.75, 
                        renewal_policy=2, renewal_premium=0.1)
contract.premium = life.gross_premium(A=A, a=a, **contract.premium_terms)
L = life.gross_variance_loss(A1=A, A2=0.14, contract=contract)
isclose(88900, L, question="Q6.12")

# + [markdown] id="eabcd0f2"
# SOA Question 6.13:  (D) -400
#

# + colab={"base_uri": "https://localhost:8080/", "height": 522} id="3d187b82" outputId="38762b89-933e-4efb-c590-825bf0b3fe61"
life = SULT().set_interest(i=0.05)
A = life.whole_life_insurance(45)
contract = Contract(benefit=10000, initial_premium=.8, renewal_premium=.1)
def fun(P):   # Solve for premium, given Loss(t=0) = 4953
    return life.L_from_t(t=10.5, contract=contract.set_contract(premium=P))
contract.set_contract(premium=life.solve(fun, target=4953, grid=100))
L = life.gross_policy_value(45, contract=contract)
life.L_plot(x=45, T=10.5, contract=contract)
isclose(-400, L, question="Q6.13")

# + [markdown] id="73a7c727"
# SOA Question 6.14  (D) 1150
#

# + colab={"base_uri": "https://localhost:8080/"} id="d6f0c625" outputId="1ccad21d-8f41-4296-f844-f2bb28f48d17"
life = SULT().set_interest(i=0.05)
a = life.temporary_annuity(40, t=10) + 0.5*life.deferred_annuity(40, u=10, t=10)
A = life.whole_life_insurance(40)
P = life.gross_premium(a=a, A=A, benefit=100000)
isclose(1150, P, question="Q6.14")

# + [markdown] id="ba7ed0a0"
# SOA Question 6.15:  (B) 1.002
#

# + colab={"base_uri": "https://localhost:8080/"} id="3b081e5c" outputId="3a25d717-83e5-43b3-fad2-6fe3a2779dcc"
life = Recursion().set_interest(i=0.05).set_a(3.4611, x=0)
A = life.insurance_twin(3.4611)
udd = UDD(m=4, life=life)
a1 = udd.whole_life_annuity(x=x)
woolhouse = Woolhouse(m=4, life=life)
a2 = woolhouse.whole_life_annuity(x=x)
P = life.gross_premium(a=a1, A=A)/life.gross_premium(a=a2, A=A)
isclose(1.002, P, question="Q6.15")

# + [markdown] id="d328bce8"
# SOA Question 6.16: (A) 2408.6
#

# + colab={"base_uri": "https://localhost:8080/"} id="b4867776" outputId="b667545f-60fc-4406-c494-7f6b89fbffc1"
life = Premiums().set_interest(d=0.05)
A = life.insurance_equivalence(premium=2143, b=100000)
a = life.annuity_equivalence(premium=2143, b=100000)
p = life.gross_premium(A=A, a=a, benefit=100000, settlement_policy=0,
                       initial_policy=250, initial_premium=0.04 + 0.35,
                       renewal_policy=50, renewal_premium=0.04 + 0.02) 
isclose(2410, p, question="Q6.16")

# + [markdown] id="f8b3364c"
# SOA Question 6.17:  (A) -30000
#

# + colab={"base_uri": "https://localhost:8080/"} id="e84e6eb4" outputId="12fb5684-9a7d-43be-fc7a-bd351711df05"
x = 0
life = ConstantForce(mu=0.1).set_interest(i=0.08)
A = life.endowment_insurance(x, t=2, b=100000, endowment=30000)
a = life.temporary_annuity(x, t=2)
P = life.gross_premium(a=a, A=A)
life1 = Recursion().set_interest(i=0.08)\
                   .set_q(life.q_x(x, t=1) * 1.5, x=x, t=1)\
                   .set_q(life.q_x(x+1, t=1) * 1.5, x=x+1, t=1)
contract = Contract(premium=P*2, benefit=100000, endowment=30000)
L = life1.gross_policy_value(x, t=0, n=2, contract=contract)
isclose(-30000, L, question="Q6.17")

# + [markdown] id="b8298896"
# SOA Question 6.18:  (D) 166400
#

# + colab={"base_uri": "https://localhost:8080/"} id="0f94e213" outputId="fc40007d-a337-40f4-8756-fd23736efed6"
life = SULT().set_interest(i=0.05)
def fun(P):
    A = (life.term_insurance(40, t=20, b=P)
         + life.deferred_annuity(40, u=20, b=30000))
    return life.gross_premium(a=1, A=A) - P
P = life.solve(fun, target=0, grid=[162000, 168800])
isclose(166400, P, question="Q6.18")

# + [markdown] id="023fb15f"
# SOA Question 6.19:  (B) 0.033
#

# + colab={"base_uri": "https://localhost:8080/"} id="0ad57ec7" outputId="1dfcb6d7-67c7-4e64-af1a-8afb7d741aa8"
life = SULT()
contract = Contract(initial_policy=.2, renewal_policy=.01)
a = life.whole_life_annuity(50)
A = life.whole_life_insurance(50)
contract.premium = life.gross_premium(A=A, a=a, **contract.premium_terms)
L = life.gross_policy_variance(50, contract=contract)
isclose(0.033, L, question="Q6.19")

# + [markdown] id="63fbc144"
# SOA Question 6.20:  (B) 459
#

# + colab={"base_uri": "https://localhost:8080/"} id="d1afe338" outputId="5eb69abc-ca16-4491-e78b-e8329ed2cfa6"
life = LifeTable().set_interest(i=.04).set_table(p={75: .9, 76: .88, 77: .85})
a = life.temporary_annuity(75, t=3)
IA = life.increasing_insurance(75, t=2)
A = life.deferred_insurance(75, u=2, t=1)
def fun(P): return life.gross_premium(a=a, A=P*IA + A*10000) - P
P = life.solve(fun, target=0, grid=[449, 489])
isclose(459, P, question="Q6.20")

# + [markdown] id="b44723ae"
# SOA Question 6.21:  (C) 100
#

# + colab={"base_uri": "https://localhost:8080/"} id="7c07aea5" outputId="e1f70069-9d46-48a0-e5f5-45a0cda9402e"
life = Recursion(verbose=False).set_interest(d=0.04)
life.set_A(0.7, x=75, t=15, endowment=1)
life.set_E(0.11, x=75, t=15)
def fun(P):
    return (P * life.temporary_annuity(75, t=15) -
            life.endowment_insurance(75, t=15, b=1000, endowment=15*float(P)))
P = life.solve(fun, target=0, grid=(80, 120))
isclose(100, P, question="Q6.21")

# + [markdown] id="e1792e95"
# SOA Question 6.22:  (C) 102
#

# + colab={"base_uri": "https://localhost:8080/"} id="e154a4ce" outputId="9a6f9e8d-81b8-4709-d905-361426bd1d41"
life=SULT(udd=True)
a = UDD(m=12, life=life).temporary_annuity(45, t=20)
A = UDD(m=0, life=life).whole_life_insurance(45)
P = life.gross_premium(A=A, a=a, benefit=100000) / 12
isclose(102, P, question="Q6.22")

# + [markdown] id="1f2bd9fa"
# SOA Question 6.23:  (D) 44.7
#

# + colab={"base_uri": "https://localhost:8080/"} id="4721d51b" outputId="808fe871-3388-467d-871d-3a170a187d57"
x = 0
life = Recursion().set_a(15.3926, x=x)\
                  .set_a(10.1329, x=x, t=15)\
                  .set_a(14.0145, x=x, t=30)
def fun(P):
    per_policy = 30 + (30 * life.whole_life_annuity(x))
    per_premium = (0.6 + 0.1*life.temporary_annuity(x, t=15)
                    + 0.1*life.temporary_annuity(x, t=30))
    a = life.temporary_annuity(x, t=30)
    return (P * a) - (per_policy + per_premium * P)
P = life.solve(fun, target=0, grid=[30.3, 49.5])
isclose(44.7, P, question="Q6.23")



# + [markdown] id="266bc6a3"
# SOA Question 6.24:  (E) 0.30
#

# + colab={"base_uri": "https://localhost:8080/"} id="092a752e" outputId="03dad73c-b481-4795-9c2c-f5dee0466679"
life = PolicyValues().set_interest(delta=0.07)
x, A1 = 0, 0.30   # Policy for first insurance
P = life.premium_equivalence(A=A1, discrete=False)  # Need its premium
contract = Contract(premium=P, discrete=False)
def fun(A2):  # Solve for A2, given Var(Loss)
    return life.gross_variance_loss(A1=A1, A2=A2, contract=contract)
A2 = life.solve(fun, target=0.18, grid=0.18)

contract = Contract(premium=0.06, discrete=False) # Solve second insurance
var = life.gross_variance_loss(A1=A1, A2=A2, contract=contract)
isclose(0.304, var, question="Q6.24")

# + [markdown] id="28b083e2"
# SOA Question 6.25:  (C) 12330
#

# + colab={"base_uri": "https://localhost:8080/"} id="b27d7264" outputId="58b53abb-8300-46e9-e78b-d1e014225d5a"
life = SULT()
woolhouse = Woolhouse(m=12, life=life)
benefits = woolhouse.deferred_annuity(55, u=10, b=1000 * 12)
expenses = life.whole_life_annuity(55, b=300)
payments = life.temporary_annuity(55, t=10)
def fun(P):
    return life.gross_future_loss(A=benefits + expenses, a=payments,
                                  contract=Contract(premium=P))
P = life.solve(fun, target=-800, grid=[12110, 12550])
isclose(12330, P, question="Q6.25")

# + [markdown] id="71edd123"
# SOA Question 6.26  (D) 180
#

# + colab={"base_uri": "https://localhost:8080/"} id="e0bc9ac7" outputId="de43a1b4-6ac6-4dd3-aaea-d0e1fa147a53"
life = SULT().set_interest(i=0.05)
def fun(P): 
    return P - life.net_premium(90, b=1000, initial_cost=P)
P = life.solve(fun, target=0, grid=[150, 190])
isclose(180, P, question="Q6.26")

# + [markdown] id="984c9535"
# SOA Question 6.27:  (D) 10310
#

# + colab={"base_uri": "https://localhost:8080/"} id="f807f50d" outputId="c82c5ade-31c1-4534-ed78-41d6a22f8df2"
life = ConstantForce(mu=0.03).set_interest(delta=0.06)
x = 0
payments = (3 * life.temporary_annuity(x, t=20, discrete=False) 
            + life.deferred_annuity(x, u=20, discrete=False))
benefits = (1000000 * life.term_insurance(x, t=20, discrete=False)
            + 500000 * life.deferred_insurance(x, u=20, discrete=False))
P = benefits / payments
isclose(10310, P, question="Q6.27")

# + [markdown] id="a83ac2e9"
# SOA Question 6.28  (B) 36
#

# + colab={"base_uri": "https://localhost:8080/"} id="e4d655be" outputId="dfeb9940-d56a-474a-c5e4-697263b5479a"
life = SULT().set_interest(i=0.05)
a = life.temporary_annuity(40, t=5)
A = life.whole_life_insurance(40)
P = life.gross_premium(a=a, A=A, benefit=1000, 
                       initial_policy=10, renewal_premium=.05,
                       renewal_policy=5, initial_premium=.2)
isclose(36, P, question="Q6.28")

# + [markdown] id="48cd4a00"
# SOA Question 6.29  (B) 20.5
#

# + colab={"base_uri": "https://localhost:8080/"} id="19f0454d" outputId="9c0b73e6-dbe5-434a-e572-4b748c860c34"
life = Premiums().set_interest(i=0.035)
def fun(a):
    return life.gross_premium(A=life.insurance_twin(a=a), a=a, 
                              initial_policy=200, initial_premium=.5,
                              renewal_policy=50, renewal_premium=.1,
                              benefit=100000)
a = life.solve(fun, target=1770, grid=[20, 22])
isclose(20.5, a, question="Q6.29")

# + [markdown] id="65adf914"
# SOA Question 6.30:  (A) 900
#

# + colab={"base_uri": "https://localhost:8080/"} id="a29edf61" outputId="0b9e70a0-ad3d-4762-80fe-ac125bf7cc79"
life = PolicyValues().set_interest(i=0.04)
contract = Contract(premium=2.338,
                    benefit=100,
                    initial_premium=.1,
                    renewal_premium=0.05)
var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
                               A2=0.17, contract=contract)
isclose(900, var, question="Q6.30")

# + [markdown] id="13f8f705"
# SOA Question 6.31:  (D) 1330
#

# + colab={"base_uri": "https://localhost:8080/"} id="2dfd7470" outputId="ece90548-3b6f-410d-8bac-cbe280802406"
life = ConstantForce(mu=0.01).set_interest(delta=0.05)
A = (life.term_insurance(35, t=35, discrete=False) 
     + life.E_x(35, t=35)*0.51791)     # A_35
P = life.premium_equivalence(A=A, b=100000, discrete=False)
isclose(1330, P, question="Q6.31")

# + [markdown] id="9876aca3"
# SOA Question 6.32:  (C) 550
#

# + colab={"base_uri": "https://localhost:8080/"} id="9775a2e0" outputId="44256da0-0cf7-4f79-cb7e-810502e59e2a"
x = 0
life = Recursion().set_interest(i=0.05).set_a(9.19, x=x)
benefits = UDD(m=0, life=life).whole_life_insurance(x)
payments = UDD(m=12, life=life).whole_life_annuity(x)
P = life.gross_premium(a=payments, A=benefits, benefit=100000)/12
isclose(550, P, question="Q6.32")

# + [markdown] id="3765e3c2"
# SOA Question 6.33:  (B) 0.13
#

# + colab={"base_uri": "https://localhost:8080/"} id="5410107c" outputId="1e3d44f7-e75a-4af4-c78f-fa8f24b1b2df"
life = Insurance().set_survival(mu=lambda x,t: 0.02*t).set_interest(i=0.03)
x = 0
var = life.E_x(x, t=15, moment=life._VARIANCE, endowment=10000)
p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
isclose(0.13, p, question="Q6.33", rel_tol=0.02)

# + [markdown] id="d47dfed4"
# SOA Question 6.34:  (A) 23300
#

# + colab={"base_uri": "https://localhost:8080/"} id="a5ef2c99" outputId="c39292a6-196f-4b8f-c46a-411bf091e8c4"
life = SULT()
def fun(benefit):
    A = life.whole_life_insurance(61)
    a = life.whole_life_annuity(61)
    return life.gross_premium(A=A, a=a, benefit=benefit, 
                              initial_premium=0.15, renewal_premium=0.03)
b = life.solve(fun, target=500, grid=[23300, 23700])
isclose(23300, b, question="Q6.34")

# + [markdown] id="e1084dfc"
# SOA Question 6.35:  (D) 530
#

# + colab={"base_uri": "https://localhost:8080/"} id="2079db39" outputId="2b849509-9216-4e55-a946-059651a0445a"
sult = SULT()
A = sult.whole_life_insurance(35, b=100000)
a = sult.whole_life_annuity(35)
P = sult.gross_premium(a=a, A=A, initial_premium=.19, renewal_premium=.04)
isclose(530, P, question="Q6.35")

# + [markdown] id="c0b919c2"
# SOA Question 6.36:  (B) 500
#

# + colab={"base_uri": "https://localhost:8080/"} id="f849ca70" outputId="acefc03f-7fe6-42c0-d88b-58b3bc2ea079"
life = ConstantForce(mu=0.04).set_interest(delta=0.08)
a = life.temporary_annuity(50, t=20, discrete=False)
A = life.term_insurance(50, t=20, discrete=False)
def fun(R):
    return life.gross_premium(a=a, A=A, initial_premium=R/4500,
                              renewal_premium=R/4500, benefit=100000)
R = life.solve(fun, target=4500, grid=[400, 800])
isclose(500, R, question="Q6.36")

# + [markdown] id="cc4c7356"
# SOA Question 6.37:  (D) 820
#

# + colab={"base_uri": "https://localhost:8080/"} id="fa96592b" outputId="862aa876-8377-4152-df87-ecff6d6d7d48"
sult = SULT()
benefits = sult.whole_life_insurance(35, b=50000 + 100)
expenses = sult.immediate_annuity(35, b=100)
a = sult.temporary_annuity(35, t=10)
P = (benefits + expenses) / a
isclose(820, P, question="Q6.37")

# + [markdown] id="07e0a134"
# SOA Question 6.38:  (B) 11.3
#

# + colab={"base_uri": "https://localhost:8080/"} id="017b6427" outputId="3c1dd14b-4a99-4004-aaa5-da390f510296"
x, n = 0, 10
life = Recursion().set_interest(i=0.05)\
                  .set_A(0.192, x=x, t=n, endowment=1, discrete=False)\
                  .set_E(0.172, x=x, t=n)
a = life.temporary_annuity(x, t=n, discrete=False)

def fun(a):   # solve for discrete annuity, given continuous
    life = Recursion(verbose=False).set_interest(i=0.05)
    life.set_a(a, x=x, t=n).set_E(0.172, x=x, t=n)
    return UDD(m=0, life=life).temporary_annuity(x, t=n)
a = life.solve(fun, target=a, grid=a)  # discrete annuity
P = life.gross_premium(a=a, A=0.192, benefit=1000)
isclose(11.3, P, question="Q6.38")

# + [markdown] id="801638b7"
# SOA Question 6.39:  (A) 29
#

# + colab={"base_uri": "https://localhost:8080/"} id="4f13b0f0" outputId="30f40642-4f6f-4d81-d7d3-c6a001b62f90"
sult = SULT()
P40 = sult.premium_equivalence(sult.whole_life_insurance(40), b=1000)
P80 = sult.premium_equivalence(sult.whole_life_insurance(80), b=1000)
p40 = sult.p_x(40, t=10)
p80 = sult.p_x(80, t=10)
P = (P40 * p40 + P80 * p80) / (p80 + p40)
isclose(29, P, question="Q6.39")

# + [markdown] id="2654e85e"
# SOA Question 6.40: (C) 116 
#

# + colab={"base_uri": "https://localhost:8080/"} id="464ce774" outputId="be812870-950e-4bea-ff21-8e65524581bf"
# - standard formula discounts/accumulates by too much (i should be smaller)
x = 0
life = Recursion().set_interest(i=0.06).set_a(7, x=x+1).set_q(0.05, x=x)
a = life.whole_life_annuity(x)
A = 110 * a / 1000
life = Recursion().set_interest(i=0.06).set_A(A, x=x).set_q(0.05, x=x)
A1 = life.whole_life_insurance(x+1)
P = life.gross_premium(A=A1 / 1.03, a=7) * 1000
isclose(116, P, question="Q6.40")

# + [markdown] id="fc621da9"
# SOA Question 6.41:  (B) 1417
#

# + colab={"base_uri": "https://localhost:8080/"} id="a76e5f76" outputId="e8d0f896-709d-4b0d-f5e9-210247d3da2a"
x = 0
life = LifeTable().set_interest(i=0.05).set_table(q={x:.01, x+1:.02})
a = 1 + life.E_x(x, t=1) * 1.01
A = life.deferred_insurance(x, u=0, t=1) + 1.01*life.deferred_insurance(x, u=1, t=1)
P = 100000 * A / a
isclose(1417, P, question="Q6.41")

# + [markdown] id="88c94cf0"
# SOA Question 6.42:  (D) 0.113
#

# + colab={"base_uri": "https://localhost:8080/"} id="b7ba7db5" outputId="4408ec33-2dbd-48e2-e7b2-2c516f60913e"
x = 0
life = ConstantForce(mu=0.06).set_interest(delta=0.06)
contract = Contract(discrete=True, premium=315.8, 
                    T=3, endowment=1000, benefit=1000)
L = [life.L_from_t(t, contract=contract) for t in range(3)]    # L(t)
Q = [life.q_x(x, u=u, t=1) for u in range(3)]              # prob(die in year t)
Q[-1] = 1 - sum(Q[:-1])   # follows SOA Solution: incorrectly treats endowment!
p = sum([q for (q, l) in zip (Q, L) if l > 0])
isclose(0.113, p, question="Q6.42")

# + [markdown] id="61779531"
# SOA Question 6.43:  (C) 170
# - although 10-year term, premiums only paid first first years: separately calculate the EPV of per-policy maintenance expenses in years 6-10 and treat as additional initial expense

# + colab={"base_uri": "https://localhost:8080/"} id="acfc0713" outputId="a5496b98-f5bd-4cc8-bb0a-617ec76172ed"
sult = SULT()
a = sult.temporary_annuity(30, t=5)
A = sult.term_insurance(30, t=10)
other_expenses = 4 * sult.deferred_annuity(30, u=5, t=5)
P = sult.gross_premium(a=a, A=A, benefit=200000, initial_premium=0.35,
                       initial_policy=8 + other_expenses, renewal_policy=4,
                       renewal_premium=0.15)
isclose(170, P, question="Q6.43")

# + [markdown] id="9c97699e"
# SOA Question 6.44:  (D) 2.18
#

# + colab={"base_uri": "https://localhost:8080/"} id="11930092" outputId="46a932d8-25dc-41fd-ec51-a8a0da03415e"
life = Recursion().set_interest(i=0.05)\
                  .set_IA(0.15, x=50, t=10)\
                  .set_a(17, x=50)\
                  .set_a(15, x=60)\
                  .set_E(0.6, x=50, t=10)
A = life.deferred_insurance(50, u=10)
IA = life.increasing_insurance(50, t=10)
a = life.temporary_annuity(50, t=10)
P = life.gross_premium(a=a, A=A, IA=IA, benefit=100)
isclose(2.2, P, question="Q6.44")

# + [markdown] id="96ebbff1"
# SOA Question 6.45:  (E) 690
#

# + colab={"base_uri": "https://localhost:8080/", "height": 522} id="e434ceb8" outputId="9aa96ca6-9440-403a-8767-15bbc6d4d7d3"
life = SULT(udd=True)
contract = Contract(benefit=100000, premium=560, discrete=False)
L = life.L_from_prob(x=35, prob=0.75, contract=contract)
life.L_plot(x=35, contract=contract, 
            T=life.L_to_t(L=L, contract=contract))
isclose(690, L, question="Q6.45")

# + [markdown] id="96fbe650"
# SOA Question 6.46:  (E) 208
#

# + colab={"base_uri": "https://localhost:8080/"} id="4467d9db" outputId="cd2c079c-d56c-4af2-bdae-731b704b1823"
life = Recursion().set_interest(i=0.05)\
                  .set_IA(0.51213, x=55, t=10)\
                  .set_a(12.2758, x=55)\
                  .set_a(7.4575, x=55, t=10)
A = life.deferred_annuity(55, u=10)
IA = life.increasing_insurance(55, t=10)
a = life.temporary_annuity(55, t=10)
P = life.gross_premium(a=a, A=A, IA=IA, benefit=300)
isclose(208, P, question="Q6.46")

# + [markdown] id="6eb80e22"
# SOA Question 6.47:  (D) 66400
#

# + colab={"base_uri": "https://localhost:8080/"} id="5f701e65" outputId="d1668bf6-9e2c-4df5-fd25-c335ccb38691"
sult = SULT()
a = sult.temporary_annuity(70, t=10)
A = sult.deferred_annuity(70, u=10)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
                        renewal_premium=0.05)
isclose(66400, P, question="Q6.47")

# + [markdown] id="7ed3e46c"
# SOA Question 6.48:  (A) 3195 -- example of deep insurance recursion
#

# + colab={"base_uri": "https://localhost:8080/"} id="022f6301" outputId="65b37ae4-3226-4fd2-86d7-f00c2da24741"
x = 0
life = Recursion().set_interest(i=0.06)\
                  .set_p(.95, x=x, t=5)\
                  .set_q(.02, x=x+5)\
                  .set_q(.03, x=x+6)\
                  .set_q(.04, x=x+7)
a = 1 + life.E_x(x, t=5)
A = life.deferred_insurance(x, u=5, t=3)
P = life.gross_premium(A=A, a=a, benefit=100000)
isclose(3195, P, question="Q6.48")

# + [markdown] id="3d130096"
# SOA Question 6.49:  (C) 86
#

# + colab={"base_uri": "https://localhost:8080/"} id="c0fe3957" outputId="65d1e3ee-4ea1-481b-dcda-0893062d0062"
sult = SULT(udd=True)
a = UDD(m=12, life=sult).temporary_annuity(40, t=20)
A = sult.whole_life_insurance(40, discrete=False)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_policy=200,
                       renewal_premium=0.04, initial_premium=0.04) / 12
isclose(86, P, question="Q6.49")

# + [markdown] id="c442a990"
# SOA Question 6.50:  (A) -47000
#

# + colab={"base_uri": "https://localhost:8080/"} id="f5713876" outputId="ffb15886-90be-4415-84b6-86d766501831"
life = SULT()
P = life.premium_equivalence(a=life.whole_life_annuity(35), b=1000) 
a = life.deferred_annuity(35, u=1, t=1)
A = life.term_insurance(35, t=1, b=1000)
cash = (A - a * P) * 10000 / life.interest.v
isclose(-47000, cash, question="Q6.50")

# + [markdown] id="8aa16e1d"
# SOA Question 6.51:  (D) 34700
#

# + colab={"base_uri": "https://localhost:8080/"} id="7def4285" outputId="5cf3947b-003e-4184-a91a-f04e07f4ffde"
life = Recursion().set_DA(0.4891, x=62, t=10)\
                   .set_A(0.0910, x=62, t=10)\
                   .set_a(12.2758, x=62)\
                   .set_a(7.4574, x=62, t=10)
IA = life.increasing_insurance(62, t=10)
A = life.deferred_annuity(62, u=10)
a = life.temporary_annuity(62, t=10)
P = life.gross_premium(a=a, A=A, IA=IA, benefit=50000)
isclose(34700, P, question="Q6.51")

# + [markdown] id="8e1e6a29"
# SOA Question 6.52:  (D) 50.80
#
# - set face value benefits to 0
#

# + colab={"base_uri": "https://localhost:8080/"} id="4529d7f9" outputId="56fa7d8f-1ea2-4846-bd57-3403881533cd"
sult = SULT()
a = sult.temporary_annuity(45, t=10)
other_cost = 10 * sult.deferred_annuity(45, u=10)
P = sult.gross_premium(a=a, A=0, benefit=0,    # set face value H = 0
                       initial_premium=1.05, renewal_premium=0.05,
                       initial_policy=100 + other_cost, renewal_policy=20)
isclose(50.8, P, question="Q6.52")

# + [markdown] id="1d1fd427"
# SOA Question 6.53:  (D) 720
#

# + colab={"base_uri": "https://localhost:8080/"} id="a9d23ae6" outputId="922de4af-7566-466b-ec91-6198c6bb637e"
x = 0
life = LifeTable().set_interest(i=0.08).set_table(q={x:.1, x+1:.1, x+2:.1})
A = life.term_insurance(x, t=3)
P = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
isclose(720, P, question="Q6.53")

# + [markdown] id="41e939f0"
# SOA Question 6.54:  (A) 25440
#

# + colab={"base_uri": "https://localhost:8080/"} id="2ea3fc85" outputId="adda1b20-f8bd-47e7-b5d3-9ff7ecc9a822"
life = SULT()
std = math.sqrt(life.net_policy_variance(45, b=200000))
isclose(25440, std, question="Q6.54")

# + [markdown] id="04a31b19"
# ## 7 Policy Values

# + [markdown] id="b265fc75"
# SOA Question 7.1:  (C) 11150
#

# + colab={"base_uri": "https://localhost:8080/"} id="628c0418" outputId="fae81c70-792e-4d22-e18f-5311ae6f253e"
life = SULT()
x, n, t = 40, 20, 10
A = (life.whole_life_insurance(x+t, b=50000)
     + life.deferred_insurance(x+t, u=n-t, b=50000))
a = life.temporary_annuity(x+t, t=n-t, b=875)
L = life.gross_future_loss(A=A, a=a)
isclose(11150, L, question="Q7.1")

# + [markdown] id="40760dac"
# SOA Question 7.2:  (C) 1152
#

# + colab={"base_uri": "https://localhost:8080/"} id="76adf1c8" outputId="e534f0d7-cd11-48e9-e281-bfbd7a62867c"
x = 0
life = Recursion(verbose=False).set_interest(i=.1)\
                               .set_q(0.15, x=x)\
                               .set_q(0.165, x=x+1)\
                               .set_reserves(T=2, endowment=2000)

def fun(P):  # solve P s.t. V is equal backwards and forwards
    policy = dict(t=1, premium=P, benefit=lambda t: 2000, reserve_benefit=True)
    return life.t_V_backward(x, **policy) - life.t_V_forward(x, **policy)
P = life.solve(fun, target=0, grid=[1070, 1230])
isclose(1152, P, question="Q7.2")

# + [markdown] id="9603ae62"
# SOA Question 7.3:  (E) 730
#

# + colab={"base_uri": "https://localhost:8080/"} id="985efb46" outputId="03907ef7-7024-4b6b-c230-74c0933a8b7f"
x = 0  # x=0 is (90) and interpret every 3 months as t=1 year
life = LifeTable().set_interest(i=0.08/4)\
                  .set_table(l={0:1000, 1:898, 2:800, 3:706})\
                  .set_reserves(T=8, V={3: 753.72})
V = life.t_V_backward(x=0, t=2, premium=60*0.9, benefit=lambda t: 1000)
V = life.set_reserves(V={2: V})\
        .t_V_backward(x=0, t=1, premium=0, benefit=lambda t: 1000)
isclose(730, V, question="Q7.3")

# + [markdown] id="070ec29c"
# SOA Question 7.4:  (B) -74 -- split benefits into two policies
#

# + colab={"base_uri": "https://localhost:8080/"} id="c4940dc2" outputId="898f272d-07fd-4c6a-b8ea-ac967943af90"
life = SULT()
P = life.gross_premium(a=life.whole_life_annuity(40),
                       A=life.whole_life_insurance(40),
                       initial_policy=100, renewal_policy=10,
                       benefit=1000)
P += life.gross_premium(a=life.whole_life_annuity(40),
                        A=life.deferred_insurance(40, u=11),
                        benefit=4000)   # for deferred portion
contract = Contract(benefit=1000, premium=1.02*P, 
                    renewal_policy=10, initial_policy=100)
V = life.gross_policy_value(x=40, t=1, contract=contract)
contract = Contract(benefit=4000, premium=0)  
A = life.deferred_insurance(41, u=10)
V += life.gross_future_loss(A=A, a=0, contract=contract)   # for deferred portion
isclose(-74, V, question="Q7.4")

# + [markdown] id="30a5ba21"
# SOA Question 7.5:  (E) 1900
#

# + colab={"base_uri": "https://localhost:8080/"} id="0605610d" outputId="3a3c9157-ede3-44a9-a606-e0d87110cb6e"
x = 0
life = Recursion(udd=True).set_interest(i=0.03)\
                          .set_q(0.04561, x=x+4)\
                          .set_reserves(T=3, V={4: 1405.08})
V = life.r_V_forward(x, s=4, r=0.5, benefit=10000, premium=647.46)
isclose(1900, V, question="Q7.5")

# + [markdown] id="984679f6"
# Answer 7.6:  (E) -25.4
#

# + colab={"base_uri": "https://localhost:8080/"} id="bf5b013c" outputId="104fd3ad-cf7c-4dfc-d10f-2d1f3c839632"
life = SULT()
P = life.net_premium(45, b=2000)
contract = Contract(benefit=2000, initial_premium=.25, renewal_premium=.05,
                    initial_policy=2*1.5 + 30, renewal_policy=2*.5 + 10)
G = life.gross_premium(a=life.whole_life_annuity(45), **contract.premium_terms)
gross = life.gross_policy_value(45, t=10, contract=contract.set_contract(premium=G))
net = life.net_policy_value(45, t=10, b=2000)
V = gross - net
isclose(-25.4, V, question="Q7.6")    

# + [markdown] id="57e85e76"
# SOA Question 7.7:  (D) 1110
#

# + colab={"base_uri": "https://localhost:8080/"} id="a5d4d205" outputId="a7e9ca46-c491-4c47-aeb8-59fd71beaca1"
x = 0
life = Recursion().set_interest(i=0.05).set_A(0.4, x=x+10)
a = Woolhouse(m=12, life=life).whole_life_annuity(x+10)
contract = Contract(premium=0, benefit=10000, renewal_policy=100)
V = life.gross_future_loss(A=0.4, contract=contract.renewals())
contract = Contract(premium=30*12, renewal_premium=0.05)
V += life.gross_future_loss(a=a, contract=contract.renewals())
isclose(1110, V, question="Q7.7")

# + [markdown] id="78784280"
# SOA Question 7.8:  (C) 29.85
#

# + colab={"base_uri": "https://localhost:8080/"} id="9311988d" outputId="d45ec115-7020-496b-87d4-aa363023b6a3"
sult = SULT()
x = 70
q = {x: [sult.q_x(x+k)*(.7 + .1*k) for k in range(3)] + [sult.q_x(x+3)]}
life = Recursion().set_interest(i=.05)\
                  .set_q(sult.q_x(70)*.7, x=x)\
                  .set_reserves(T=3)
V = life.t_V(x=70, t=1, premium=35.168, benefit=lambda t: 1000)
isclose(29.85, V, question="Q7.8")

# + [markdown] id="13417b4b"
# SOA Question 7.9:  (A) 38100
#

# + colab={"base_uri": "https://localhost:8080/"} id="95ee4b66" outputId="dd837400-a4e3-41d5-bfdd-e635a3aa9741"
sult = SULT(udd=True)
x, n, t = 45, 20, 10
a = UDD(m=12, life=sult).temporary_annuity(x+10, t=n-10)
A = UDD(m=0, life=sult).endowment_insurance(x+10, t=n-10)
contract = Contract(premium=253*12, endowment=100000, benefit=100000)
V = sult.gross_future_loss(A=A, a=a, contract=contract)
isclose(38100, V, question="Q7.9")

# + [markdown] id="4f341dc3"
# SOA Question 7.10: (C) -970
#

# + colab={"base_uri": "https://localhost:8080/"} id="83268a47" outputId="087ff455-6a2a-4531-e9f4-5724907ccd11"
life = SULT()
G = 977.6
P = life.net_premium(45, b=100000)
contract = Contract(benefit=0, premium=G-P, renewal_policy=.02*G + 50)
V = life.gross_policy_value(45, t=5, contract=contract)
isclose(-970, V, question="Q7.10")

# + [markdown] id="55157c76"
# SOA Question 7.11:  (B) 1460
#

# + colab={"base_uri": "https://localhost:8080/"} id="2e0c21b0" outputId="0ac7906f-f946-4206-e6c7-b7c4e0dd1de0"
life = Recursion().set_interest(i=0.05).set_a(13.4205, x=55)
contract = Contract(benefit=10000)
def fun(P):
    return life.L_from_t(t=10, contract=contract.set_contract(premium=P))
P = life.solve(fun, target=4450, grid=400)
V = life.gross_policy_value(45, t=10, contract=contract.set_contract(premium=P))
isclose(1460, V, question="Q7.11")

# + [markdown] id="6c52d5c3"
# SOA Question 7.12:  (E) 4.09
#

# + colab={"base_uri": "https://localhost:8080/"} id="7993f016" outputId="da0ed101-aefe-455b-f74a-faf47704096c"
benefit = lambda k: 26 - k
x = 44
life = Recursion().set_interest(i=0.04)\
                  .set_q(0.15, x=55)\
                  .set_reserves(T=25, endowment=1, V={11: 5.})
def fun(P):  # solve for net premium, from final year recursion
    return life.t_V(x=x, t=24, premium=P, benefit=benefit)
P = life.solve(fun, target=0.6, grid=0.5)    # solved net premium
V = life.t_V(x, t=12, premium=P, benefit=benefit)  # recursion formula
isclose(4.09, V, question="Q7.12")


# + [markdown] id="4ea704b8"
# Answer 7.13: (A) 180
#

# + colab={"base_uri": "https://localhost:8080/"} id="0b8778cc" outputId="e35a1cda-135e-4090-fb42-8b9bccae70c3"
life = SULT()
V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
isclose(180, V, question="Q7.13")

# + [markdown] id="58f053bd"
# SOA Question 7.14:  (A) 2200
#

# + colab={"base_uri": "https://localhost:8080/"} id="e0bf9ee2" outputId="be9495c6-4ae7-4bdc-f6f6-7c9a5ac46aca"
x = 45
life = Recursion(verbose=False).set_interest(i=0.05)\
                               .set_q(0.009, x=50)\
                               .set_reserves(T=10, V={5: 5500})
def fun(P):  # solve for net premium,
    return life.t_V(x=x, t=6, premium=P*0.96 - 50, benefit=lambda t: 100000+200)
P = life.solve(fun, target=7100, grid=[2200, 2400])
isclose(2200, P, question="Q7.14")

# + [markdown] id="caab2d47"
# SOA Question 7.15:  (E) 50.91
#

# + colab={"base_uri": "https://localhost:8080/"} id="bbed6a97" outputId="c299c988-9c74-4131-a9e9-d867a391f133"
x = 0
V = Recursion(udd=True).set_interest(i=0.05)\
                       .set_q(0.1, x=x+15)\
                       .set_reserves(T=3, V={16: 49.78})\
                       .r_V_backward(x, s=15, r=0.6, benefit=100)
isclose(50.91, V, question="Q7.15")

# + [markdown] id="cf793972"
# SOA Question 7.16:  (D) 380
#

# + colab={"base_uri": "https://localhost:8080/"} id="b3317669" outputId="4aa2dc04-6b96-454b-e0d1-d755f6ff255a"
life = SelectLife().set_interest(v=.95)\
                   .set_table(A={86: [683/1000]},
                              q={80+k: [.01*(k+1)] for k in range(6)})
x, t, n = 80, 3, 5
A = life.whole_life_insurance(x+t)
a = life.temporary_annuity(x+t, t=n-t)
V = life.gross_future_loss(A=A, a=a, contract=Contract(benefit=1000, premium=130))
isclose(380, V, question="Q7.16")

# + [markdown] id="d16fa0a1"
# SOA Question 7.17:  (D) 1.018
#

# + colab={"base_uri": "https://localhost:8080/"} id="e26e27a3" outputId="b19942d7-97f0-4178-bf9f-8b066e5a34af"
x = 0
life = Recursion().set_interest(v=math.sqrt(0.90703))\
                  .set_q(0.02067, x=x+10)\
                  .set_A(0.52536, x=x+11)\
                  .set_A(0.30783, x=x+11, moment=2)
A1 = life.whole_life_insurance(x+10)
A2 = life.whole_life_insurance(x+10, moment=2)
ratio = (life.insurance_variance(A2=A2, A1=A1) 
         / life.insurance_variance(A2=0.30783, A1=0.52536))
isclose(1.018, ratio, question="Q7.17")

# + [markdown] id="1330efbd"
# SOA Question 7.18:  (A) 17.1
#

# + colab={"base_uri": "https://localhost:8080/"} id="789aef65" outputId="e6478f8a-d6a1-4acd-e8b2-ebfbafa28707"
x = 10
life = Recursion(verbose=False).set_interest(i=0.04).set_q(0.009, x=x)
def fun(a):
    return life.set_a(a, x=x).net_policy_value(x, t=1)
a = life.solve(fun, target=0.012, grid=[17.1, 19.1])
isclose(17.1, a, question="Q7.18")

# + [markdown] id="bcd7d9ae"
# SOA Question 7.19:  (D) 720
#

# + colab={"base_uri": "https://localhost:8080/"} id="9372ed3c" outputId="42965479-7979-4447-dc29-bf03de51b6df"
life = SULT()
contract = Contract(benefit=100000,
                    initial_policy=300,
                    initial_premium=.5,
                    renewal_premium=.1)
P = life.gross_premium(A=life.whole_life_insurance(40), **contract.premium_terms)
A = life.whole_life_insurance(41)
a = life.immediate_annuity(41)   # after premium and expenses are paid
V = life.gross_future_loss(A=A,
                           a=a,
                           contract=contract.set_contract(premium=P).renewals())
isclose(720, V, question="Q7.19")

# + [markdown] id="9aa726ed"
# SOA Question 7.20: (E) -277.23
#

# + colab={"base_uri": "https://localhost:8080/"} id="ab37f0e4" outputId="5013d932-047e-487e-d4f1-631a5a88fd43"
life = SULT()
S = life.FPT_policy_value(35, t=1, b=1000)  # is 0 for FPT at t=0,1
contract = Contract(benefit=1000,
                    initial_premium=.3,
                    initial_policy=300,
                    renewal_premium=.04,
                    renewal_policy=30)
G = life.gross_premium(A=life.whole_life_insurance(35), **contract.premium_terms)
R = life.gross_policy_value(35, t=1, contract=contract.set_contract(premium=G))
isclose(-277.23, R - S, question="Q7.20")

# + [markdown] id="bc06c100"
# SOA Question 7.21:  (D) 11866
#

# + colab={"base_uri": "https://localhost:8080/"} id="25c496da" outputId="3e6a4fde-0c01-4271-c650-7fa403d12360"
life = SULT()
x, t, u = 55, 9, 10
P = life.gross_premium(IA=0.14743,
                       a=life.temporary_annuity(x, t=u),
                       A=life.deferred_annuity(x, u=u),
                       benefit=1000)
contract = Contract(initial_policy=life.term_insurance(x+t, t=1, b=10*P),
                    premium=P,
                    benefit=1000)
a = life.temporary_annuity(x+t, t=u-t)
A = life.deferred_annuity(x+t, u=u-t)
V = life.gross_future_loss(A=A, a=a, contract=contract)
isclose(11866, V, question="Q7.21")

# + [markdown] id="d2110715"
# SOA Question 7.22:  (C) 46.24
#

# + colab={"base_uri": "https://localhost:8080/"} id="6a2abdfc" outputId="558ce4ab-0494-4213-c3bd-313376c4c084"
life = PolicyValues().set_interest(i=0.06)
contract = Contract(benefit=8, premium=1.250)
def fun(A2): 
    return life.gross_variance_loss(A1=0, A2=A2, contract=contract)
A2 = life.solve(fun, target=20.55, grid=20.55/8**2)
contract = Contract(benefit=12, premium=1.875)
var = life.gross_variance_loss(A1=0, A2=A2, contract=contract)
isclose(46.2, var, question="Q7.22")

# + [markdown] id="18e5b35a"
# SOA Question 7.23:  (D) 233
#

# + colab={"base_uri": "https://localhost:8080/"} id="c4c42da2" outputId="9c4a3528-bc2e-4d78-faa6-bb969b329954"
life = Recursion().set_interest(i=0.04).set_p(0.995, x=25)
A = life.term_insurance(25, t=1, b=10000)
def fun(beta):  # value of premiums in first 20 years must be equal
    return beta * 11.087 + (A - beta) 
beta = life.solve(fun, target=216 * 11.087, grid=[140, 260])
isclose(233, beta, question="Q7.23")

# + [markdown] id="4b82caf4"
# SOA Question 7.24:  (C) 680
#

# + colab={"base_uri": "https://localhost:8080/"} id="75d8d20b" outputId="0ef6af97-ec26-4bea-b3f1-292fe0111594"
life = SULT()
P = life.premium_equivalence(A=life.whole_life_insurance(50), b=1000000)
isclose(680, 11800 - P, question="Q7.24")

# + [markdown] id="8410e40c"
# SOA Question 7.25:  (B) 3947.37
#

# + colab={"base_uri": "https://localhost:8080/"} id="9928c491" outputId="47426ebe-a93d-4132-a0df-70e4d37123e0"
life = SelectLife().set_interest(i=.04)\
                   .set_table(A={55: [.23, .24, .25],
                                 56: [.25, .26, .27],
                                 57: [.27, .28, .29],
                                 58: [.20, .30, .31]})
V = life.FPT_policy_value(55, t=3, b=100000)
isclose(3950, V, question="Q7.25")

# + [markdown] id="9e538fc9"
# SOA Question 7.26:  (D) 28540 
# - backward = forward reserve recursion
#

# + colab={"base_uri": "https://localhost:8080/"} id="9fc808b8" outputId="80809054-f790-4a82-cb9c-da13bacf6b38"
x = 0
life = Recursion(verbose=False).set_interest(i=.05)\
                               .set_p(0.85, x=x)\
                               .set_p(0.85, x=x+1)\
                               .set_reserves(T=2, endowment=50000)
def benefit(k): return k * 25000
def fun(P):  # solve P s.t. V is equal backwards and forwards
    policy = dict(t=1, premium=P, benefit=benefit, reserve_benefit=True)
    return life.t_V_backward(x, **policy) - life.t_V_forward(x, **policy)
P = life.solve(fun, target=0, grid=[27650, 28730])
isclose(28540, P, question="Q7.26")

# + [markdown] id="45908bc1"
# SOA Question 7.27:  (B) 213
#

# + colab={"base_uri": "https://localhost:8080/"} id="7ad342a1" outputId="b073d19a-6c5f-48c0-c28f-33c26bc1863f"
x = 0
life = Recursion(verbose=False).set_interest(i=0.03)\
                               .set_q(0.008, x=x)\
                               .set_reserves(V={0: 0})
def fun(G):  # Solve gross premium from expense reserves equation
    return life.t_V(x=x, t=1, premium=G - 187, benefit=lambda t: 0,
                    per_policy=10 + 0.25*G)
G = life.solve(fun, target=-38.70, grid=[200, 252])
isclose(213, G, question="Q7.27")

# + [markdown] id="a691f65c"
# SOA Question 7.28:  (D) 24.3
#

# + colab={"base_uri": "https://localhost:8080/"} id="99412e64" outputId="b502f1c6-ab37-4091-ad2f-0049e9baa206"
life = SULT()
PW = life.net_premium(65, b=1000)   # 20_V=0 => P+W is net premium for A_65
P = life.net_premium(45, t=20, b=1000)  # => P is net premium for A_45:20
isclose(24.3, PW - P, question="Q7.28")

# + [markdown] id="04bd97d2"
# SOA Question 7.29:  (E) 2270
#

# + colab={"base_uri": "https://localhost:8080/"} id="e301c605" outputId="5a4cb4e6-ee9c-4150-eee9-80cde3ee5c0f"
x = 0
life = Recursion(verbose=False).set_interest(i=0.04)\
                               .set_a(14.8, x=x)\
                               .set_a(11.4, x=x+10)
def fun(B): 
    return life.net_policy_value(x, t=10, b=B)
B = life.solve(fun, target=2290, grid=2290*10)  # Solve benefit B given net 10_V
contract = Contract(initial_policy=30, renewal_policy=5, benefit=B)
G = life.gross_premium(a=life.whole_life_annuity(x), **contract.premium_terms)
V = life.gross_policy_value(x, t=10, contract=contract.set_contract(premium=G))
isclose(2270, V, question="Q7.29")

# + [markdown] id="72f5987a"
# SOA Question 7.30:  (E) 9035
#

# + colab={"base_uri": "https://localhost:8080/"} id="cdd25b58" outputId="58819880-ca58-435a-864d-d36cb36110f3"
contract = Contract(premium=0, benefit=10000)  # premiums=0 after t=10
L = SULT().gross_policy_value(35, contract=contract)
V = SULT().set_interest(i=0).gross_policy_value(35, contract=contract) # 10000
isclose(9035, V - L, question="Q7.30")

# + [markdown] id="df03679b"
# SOA Question 7.31:  (E) 0.310
#

# + colab={"base_uri": "https://localhost:8080/"} id="0d9cd985" outputId="931f37ce-1f0d-4708-985d-ce043a6bc674"
x = 0
life = Reserves().set_reserves(T=3)
G = 368.05
def fun(P):  # solve net premium expense reserve equation
    return life.t_V(x, t=2, premium=G-P, benefit=lambda t:0, per_policy=5+0.08*G)
P = life.solve(fun, target=-23.64, grid=[.29, .31]) / 1000
isclose(0.310, P, question="Q7.31")

# + [markdown] id="164f3498"
# SOA Question 7.32:  (B) 1.4
#

# + colab={"base_uri": "https://localhost:8080/"} id="cf0604c6" outputId="cfcab2f1-3c3d-406b-fa14-93c8ea955c84"
life = PolicyValues().set_interest(i=0.06)
contract = Contract(benefit=1, premium=0.1)
def fun(A2): 
    return life.gross_variance_loss(A1=0, A2=A2, contract=contract)
A2 = life.solve(fun, target=0.455, grid=0.455)
contract = Contract(benefit=2, premium=0.16)
var = life.gross_variance_loss(A1=0, A2=A2, contract=contract)
isclose(1.39, var, question="Q7.32")

# + [markdown] id="1ebd4d3a"
# __Final Score__

# + colab={"base_uri": "https://localhost:8080/"} id="40395bce" outputId="0b51bc19-93aa-408e-cbe3-0361c6996c82"
from datetime import datetime
print(datetime.now())
print(isclose)
