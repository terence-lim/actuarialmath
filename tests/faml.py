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
# - SOA FAM-L Sample Solutions: [copy retrieved Aug 2022](https://terence-lim.github.io/notes/2022-10-exam-fam-l-sol.pdf)
#
# - SOA FAM-L Sample Questions: [copy retrieved Aug 2022](https://terence-lim.github.io/notes/2022-10-exam-fam-l-quest.pdf)
#
# - [Online tutorial](https://terence-lim.github.io/actuarialmath-tutorial/), or [download pdf](https://terence-lim.github.io/notes/actuarialmath-tutorial.pdf)
#
# - [Code documentation](https://terence-lim.github.io/actuarialmath-docs)
#
# - [Github repo](https://github.com/terence-lim/actuarialmath.git) and [issues](https://github.com/terence-lim/actuarialmath/issues)
#

# + colab={"base_uri": "https://localhost:8080/"} id="i9j4jVPE-Fpk" outputId="c2addd37-6cbe-49bf-f4d3-7252d6d3f0c5"
# #! pip install actuarialmath

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

# + colab={"base_uri": "https://localhost:8080/", "height": 224} id="a6dc0057" outputId="cf6a6493-e98c-4d90-d592-ee02878967e3"
print("Interest Functions at i=0.05")
UDD.interest_frame()

# + colab={"base_uri": "https://localhost:8080/"} id="b3de4194" outputId="5a4bf48e-7df5-4533-caeb-8b6380be0a3d"
print("Values of z for selected values of Pr(Z<=z)")
print(Life.quantiles_frame().to_string(float_format=lambda x: f"{x:.3f}"))

# + colab={"base_uri": "https://localhost:8080/", "height": 441} id="1c334975" outputId="80293d1b-3ba2-4b84-f22d-ee789fb81c53"
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

# + colab={"base_uri": "https://localhost:8080/"} id="59f81e7f" outputId="173c7ae2-f55e-497b-be8d-4bd4903b106d"
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

# + colab={"base_uri": "https://localhost:8080/"} id="1f07bbbd" outputId="16cffcd5-d289-481d-f5ce-222bc25e01e6"
p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
std = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
isclose(400, std, question="Q2.2")

# + [markdown] id="246aaf2d"
# SOA Question 2.3: (A) 0.0483
# 1. Derive formula for $f$ given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="e36facdd" outputId="ab177cfc-a9f8-469c-9cd2-ab51d27f8760"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a6412173" outputId="832f29a4-ef2a-4847-afa7-5ae4612c63c8"
def l(x, s): return 0. if (x+s) >= 100 else 1 - ((x + s)**2) / 10000.
e = Lifetime().set_survival(l=l).e_x(75, t=10, curtate=False)
isclose(8.2, e, question="Q2.4")

# + [markdown] id="b73ac219"
# SOA Question 2.5:  (B) 37.1
# - solve for $e_{40}$ from limited lifetime formula
# - compute $e_{41}$ using backward recursion

# + colab={"base_uri": "https://localhost:8080/"} id="7485cb2a" outputId="5006156c-27ea-43b6-925b-080690daafcd"
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

# + colab={"base_uri": "https://localhost:8080/"} id="b4a5e978" outputId="d60772a4-c868-4a2e-c6fd-c040f5ef9aee"
life = Survival().set_survival(l=lambda x,s: (1 - (x+s)/60)**(1/3))
mu = 1000 * life.mu_x(35)
isclose(13.3, mu, question="Q2.6")

# + [markdown] id="3406e6fd"
# SOA Question 2.7: (B) 0.1477
# - calculate from given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="1cbb1f35" outputId="e893abe6-5aa6-40a9-9882-b29f17e6390a"
l = lambda x,s: (1-((x+s)/250) if (x+s)<40 else 1-((x+s)/100)**2)
q = Survival().set_survival(l=l).q_x(30, t=20)
isclose(0.1477, q, question="Q2.7")

# + [markdown] id="59f19984"
# SOA Question 2.8: (C) 0.94
# - relate $p_{male}$ and $p_{female}$ through the common term $\mu$ and the given proportions
#

# + colab={"base_uri": "https://localhost:8080/"} id="d9433c52" outputId="1dfc6877-2ff0-4eed-c32a-7ccf0343bbc3"
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

# + colab={"base_uri": "https://localhost:8080/"} id="07539d91" outputId="8c07f8e7-c98d-4509-8172-11baab58f8d1"
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

# + colab={"base_uri": "https://localhost:8080/"} id="b3c05afd" outputId="70ff503b-9a20-4a33-ef3f-587cadc266da"
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

# + colab={"base_uri": "https://localhost:8080/"} id="9576af60" outputId="831e950a-adcb-4f56-d188-8a8cdabe10b0"
life = SelectLife().set_table(l={50: [99, 96, 93],
                                 51: [97, 93, 89],
                                 52: [93, 88, 83],
                                 53: [90, 84, 78]})
q = 10000 * life.q_r(51, s=0, r=0.5, t=2.2)
isclose(1074, q, question="Q3.3")

# + [markdown] id="2247b56f"
# SOA Question 3.4:  (B) 815
# - compute portfolio percentile with N=4000, and mean and variance  from binomial distribution

# + colab={"base_uri": "https://localhost:8080/"} id="fb29aeca" outputId="396f6731-4b3f-4b22-baa4-edcd2be1fbf0"
sult = SULT()
mean = sult.p_x(25, t=95-25)
var = sult.bernoulli(mean, variance=True)
pct = sult.portfolio_percentile(N=4000, mean=mean, variance=var, prob=0.1)
isclose(815, pct, question="Q3.4")

# + [markdown] id="a989a344"
# SOA Question 3.5:  (E) 106
# - compute mortality rates by interpolating lives between integer ages, with UDD and constant force of mortality assumptions

# + colab={"base_uri": "https://localhost:8080/"} id="4a01bc25" outputId="9143281b-4a6a-4cd7-875b-3722fbdd93ce"
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

# + colab={"base_uri": "https://localhost:8080/"} id="ab3dfad5" outputId="2edfb71c-fffc-4c1a-a7e7-e6801f705994"
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

# + colab={"base_uri": "https://localhost:8080/"} id="b745cec2" outputId="b3c0c1d7-4285-46f7-c0e5-4a1d8e460c35"
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

# + colab={"base_uri": "https://localhost:8080/"} id="907c1755" outputId="4f090570-86f1-480a-a0ca-e746c339cbf6"
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

# + colab={"base_uri": "https://localhost:8080/"} id="5e87a932" outputId="0b5f2dfe-83be-4596-a370-38071802f5e1"
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

# + colab={"base_uri": "https://localhost:8080/"} id="cfd72881" outputId="fdd32eb4-aef6-4af2-fe44-f838509d6ee3"
interest = Interest(v=0.75)
L = 35*interest.annuity(t=4, due=False) + 75*interest.v_t(t=5)
interest = Interest(v=0.5)
R = 15*interest.annuity(t=4, due=False) + 25*interest.v_t(t=5)
isclose(0.86, L / (L + R), question="Q3.10")

# + [markdown] id="a51f9f7a"
# SOA Question 3.11:  (B) 0.03
# - calculate mortality rate by interpolating lives assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="ced9708b" outputId="fb9127a8-b675-4dca-c6e8-5468849c5b0b"
life = LifeTable(udd=True).set_table(q={50//2: .02, 52//2: .04})
q = life.q_r(50//2, t=2.5/2)
isclose(0.03, q, question="Q3.11")

# + [markdown] id="6dae2d07"
# SOA Question 3.12: (C) 0.055 
# - compute survival probability by interpolating lives assuming constant force
#

# + colab={"base_uri": "https://localhost:8080/"} id="7e5ce19d" outputId="a32962b7-6708-4d7a-95de-e80fc4fd32d1"
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

# + colab={"base_uri": "https://localhost:8080/"} id="8db335e1" outputId="636f1fff-646a-436a-adf1-80c8f765cd43"
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

# + colab={"base_uri": "https://localhost:8080/"} id="39107fed" outputId="ae6b9e58-86f6-4d7a-9496-a3c70eec2a95"
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

# + colab={"base_uri": "https://localhost:8080/"} id="7a9f699d" outputId="732e4674-e462-40b5-9f05-ca2815343f1c"
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

# + colab={"base_uri": "https://localhost:8080/"} id="2145a29e" outputId="4c9e905a-cc91-4850-8db0-854989e25476"
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

# + colab={"base_uri": "https://localhost:8080/"} id="db579f3b" outputId="e6e2e006-1b78-45b6-b270-435ad567034c"
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

# + colab={"base_uri": "https://localhost:8080/"} id="3fa24393" outputId="e3094da8-2a2d-43a6-a4e3-06c01620ff5e"
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

# + colab={"base_uri": "https://localhost:8080/"} id="3c9d0b1e" outputId="9ce6ae62-ce6a-4afb-d1bf-abe00bb38caf"
sult = SULT(udd=True).set_interest(delta=0.05)
Z = 100000 * sult.Z_from_prob(45, 0.95, discrete=False)
isclose(35200, Z, question="Q4.5")

# + [markdown] id="1792b7aa"
# SOA Question 4.6:  (B) 29.85
# - calculate adjusted mortality rates
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="f31ee601" outputId="ea7759a3-8d35-44f8-8015-34afc05162e1"
sult = SULT()
life = LifeTable().set_interest(i=0.05)\
                  .set_table(q={70+k: .95**k * sult.q_x(70+k) for k in range(3)})
A = life.term_insurance(70, t=3, b=1000)
isclose(29.85, A, question="Q4.6")


# + [markdown] id="230429ad"
# SOA Question 4.7:  (B) 0.06
# - use Bernoulli shortcut formula for variance of pure endowment Z 
# - solve for $i$, since $p$ is given.

# + colab={"base_uri": "https://localhost:8080/"} id="f38c4ab6" outputId="f9f9dca4-f476-41fa-c282-5ac5700d99c2"
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

# + colab={"base_uri": "https://localhost:8080/"} id="f3ad0bbe" outputId="ab3a4680-849c-4997-edb9-521ef8bc0dde"
def v_t(t): return 1.04**(-t) if t < 1 else 1.04**(-1) * 1.05**(-t+1)
A = SULT().set_interest(v_t=v_t).whole_life_insurance(50, b=1000)
isclose(191, A, question="Q4.8")

# + [markdown] id="4408c9ef"
# SOA Question 4.9:  (D) 0.5
# - use whole-life, term and endowment insurance relationships.
#

# + colab={"base_uri": "https://localhost:8080/"} id="0ab006d1" outputId="39b2b025-14e7-43c3-9b26-cdb734ec6915"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 521} id="14fca3d8" outputId="c579febc-dca7-42ba-e755-1cd352498c71"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a3858d16" outputId="fc9a0cbe-7fec-4a83-f49c-00f5a65bcd6d"
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

# + colab={"base_uri": "https://localhost:8080/"} id="b34e726c" outputId="87270bd2-36a4-4448-bd22-9b8c48ed8d6b"
cov = Life.covariance(a=1.65, b=10.75, ab=0)  # E[Z1 Z2] = 0 nonoverlapping
var = Life.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov)
isclose(167, var, question="Q4.12")

# + [markdown] id="ae69b52f"
# SOA Question 4.13:  (C) 350 
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="a838d9f1" outputId="204c8d16-e1c3-45ec-9708-ad40003b5623"
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

# + colab={"base_uri": "https://localhost:8080/"} id="f0e506f8" outputId="30f216b8-e2c3-4c9b-8620-ffcc76f35315"
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

# + colab={"base_uri": "https://localhost:8080/"} id="4b7eee09" outputId="3d7744a6-cd64-4689-ac3f-7bf9fb26b828"
life = Insurance().set_survival(mu=lambda *x: 0.04).set_interest(delta=0.06)
benefit = lambda x,t: math.exp(0.02*t)
A1 = life.A_x(0, benefit=benefit, discrete=False)
A2 = life.A_x(0, moment=2, benefit=benefit, discrete=False)
var = life.insurance_variance(A2=A2, A1=A1)
isclose(0.0833, var, question="Q4.15")

# + [markdown] id="79f63326"
# SOA Question 4.16:  (D) 0.11
# - compute EPV of future benefits with adjusted mortality rates

# + colab={"base_uri": "https://localhost:8080/"} id="3c74f0e6" outputId="98b5bae1-3f4b-483b-e9f8-f32b21a6af9c"
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

# + colab={"base_uri": "https://localhost:8080/"} id="330ac8db" outputId="746e4217-1b91-4477-e42e-a4a95f371c1f"
sult = SULT()
median = sult.Z_t(48, prob=0.5, discrete=False)
def benefit(x,t): return 5000 if t < median else 10000
A = sult.A_x(48, benefit=benefit)
isclose(1130, A, question="Q4.17")

# + [markdown] id="258c80e6"
# SOA Question 4.18  (A) 81873 
# - find values of limits such that integral of lifetime density function equals required survival probability
#

# + colab={"base_uri": "https://localhost:8080/"} id="53795941" outputId="db564de7-cefa-498e-fff6-356c069639f9"
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

# + colab={"base_uri": "https://localhost:8080/"} id="13a8420d" outputId="9d7a38b4-8a74-4a3a-bdb1-d78a03df4ea2"
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

# + colab={"base_uri": "https://localhost:8080/"} id="18b1a0c0" outputId="683baef8-8a0a-4d77-a92e-84854e8023f3"
life = ConstantForce(mu=0.01).set_interest(delta=0.06)
EY = life.certain_life_annuity(0, u=10, discrete=False)
p = life.p_x(0, t=life.Y_to_t(EY))
isclose(0.705, p, question="Q5.1")  # 0.705

# + [markdown] id="f90b71c6"
# SOA Question 5.2:  (B) 9.64
# - compute term life as difference of whole life and deferred insurance
# - compute twin annuity-due, and adjust to an immediate annuity. 

# + colab={"base_uri": "https://localhost:8080/"} id="206b600b" outputId="a25eb40f-ab1b-48d5-f262-152c276545e4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="eeca16c1" outputId="55f511e0-6919-4a9a-c905-4d1354bb3660"
t = 10.5
E = t * SULT().E_r(40, t=t)
isclose(6.239, E, question="Q5.3")

# + [markdown] id="cd3027da"
# SOA Question 5.4:  (A) 213.7
# - compute certain and life annuity factor as the sum of a certain annuity and a deferred life annuity.
# - solve for amount of annual benefit that equals given EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="297311f0" outputId="9a5628e5-3106-4e66-cc41-a64b7bee4650"
life = ConstantForce(mu=0.02).set_interest(delta=0.01)
u = life.e_x(40, curtate=False)
P = 10000 / life.certain_life_annuity(40, u=u, discrete=False)
isclose(213.7, P, question="Q5.4") # 213.7

# + [markdown] id="46f357cd"
# SOA Question 5.5: (A) 1699.6
# - adjust mortality rate for the extra risk
# - compute annuity by backward recursion.
#

# + colab={"base_uri": "https://localhost:8080/"} id="9737dc8d" outputId="d54cad73-6bd0-42ec-cf97-7ea88d67b27a"
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

# + colab={"base_uri": "https://localhost:8080/"} id="8445b834" outputId="44946fe9-270f-405e-bad3-4d526a8c9c0e"
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

# + colab={"base_uri": "https://localhost:8080/"} id="93c40a7c" outputId="03592256-abe1-433d-bbc0-a77e52f95aeb"
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

# + colab={"base_uri": "https://localhost:8080/"} id="3db058df" outputId="19eee192-a275-446c-dbf3-29c12abd710b"
sult = SULT()
a = sult.certain_life_annuity(55, u=5)
p = sult.p_x(55, t=math.floor(a))
isclose(0.92118, p, question="Q5.8")

# + [markdown] id="ad7d5d47"
# SOA Question 5.9:  (C) 0.015
# - express both EPV's expressed as forward recursions
# - solve for unknown constant $k$.
#

# + colab={"base_uri": "https://localhost:8080/"} id="1937f550" outputId="88e2a1e6-2dc5-4257-830a-5cb532619065"
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

# + colab={"base_uri": "https://localhost:8080/"} id="68d68c2e" outputId="0ca740e0-370e-40f3-f91c-f1267c58d20b"
P = SULT().set_interest(i=0.03)\
          .net_premium(80, t=2, b=1000, return_premium=True)
isclose(35.36, P, question="Q6.1")

# + [markdown] id="8a9f7924"
# SOA Question 6.2: (E) 3604
# - EPV return of premiums without interest = Premium $\times$ IA factor
# - solve for gross premiums such that EPV premiums = EPV benefits and expenses

# + colab={"base_uri": "https://localhost:8080/"} id="cde906a7" outputId="912154a5-db2c-4b2e-a968-d47cce321df9"
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

# + colab={"base_uri": "https://localhost:8080/"} id="1d438209" outputId="7e2e7ab7-eb87-4736-eeab-808846b22e23"
life = SULT()
t = life.Y_to_t(life.whole_life_annuity(65))
q = 1 - life.p_x(65, t=math.floor(t) - 1)
isclose(0.39, q, question="Q6.3")

# + [markdown] id="8afc2a87"
# SOA Question 6.4:  (E) 1890
#

# + colab={"base_uri": "https://localhost:8080/"} id="5b9948fb" outputId="28337f0d-0910-46c6-b4e4-bc5a8745bf24"
mthly = Mthly(m=12, life=Annuity().set_interest(i=0.06))
A1, A2 = 0.4075, 0.2105
mean = mthly.annuity_twin(A1) * 15 * 12
var = mthly.annuity_variance(A1=A1, A2=A2, b=15 * 12)
S = Annuity.portfolio_percentile(mean=mean, variance=var, prob=.9, N=200) / 200
isclose(1890, S, question="Q6.4")

# + [markdown] id="fd4150b6"
# SOA Question 6.5:  (D) 33
#

# + colab={"base_uri": "https://localhost:8080/"} id="bda89a9a" outputId="aee7da6d-8a7d-4dae-c28a-6bbcee937c29"
life = SULT()
P = life.net_premium(30, b=1000)
def gain(k): return life.Y_x(30, t=k) * P - life.Z_x(30, t=k) * 1000
k = min([k for k in range(20, 40) if gain(k) < 0])
isclose(33, k, question="Q6.5")

# + [markdown] id="bba959b2"
# SOA Question 6.6:  (B) 0.79
#

# + colab={"base_uri": "https://localhost:8080/"} id="2a248f2c" outputId="efe699a3-4dc6-46ab-b64c-3fe2313c4287"
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

# + colab={"base_uri": "https://localhost:8080/"} id="56437e4c" outputId="1ed51001-e7f6-4570-e42b-de1a87146e6b"
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

# + colab={"base_uri": "https://localhost:8080/"} id="e90a196f" outputId="0c3d1045-e4c9-4fa5-b6c8-b1c2b3fef0f3"
life = SULT()
initial_cost = (50 + 10 * life.deferred_annuity(60, u=1, t=9)
                + 5 * life.deferred_annuity(60, u=10, t=10))
P = life.net_premium(60, initial_cost=initial_cost)
isclose(9.5, P, question="Q6.8")

# + [markdown] id="cc58d89d"
# SOA Question 6.9:  (D) 647
#

# + colab={"base_uri": "https://localhost:8080/"} id="a5ff35d6" outputId="782cff9a-7380-4f74-b717-6e2424cefeb4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a6ea62e1" outputId="eeae8f35-92a8-48f3-c9e8-c2db9782fd03"
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

# + colab={"base_uri": "https://localhost:8080/"} id="84bc4d87" outputId="edb9e2d9-b4a6-44e5-e67c-3cdb92d6038f"
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

# + colab={"base_uri": "https://localhost:8080/"} id="761a8575" outputId="22f5fa83-302c-40df-d9e7-998f33f92f40"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 522} id="3d187b82" outputId="38492855-adc1-4869-a8aa-49d374209142"
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

# + colab={"base_uri": "https://localhost:8080/"} id="d6f0c625" outputId="9eabb789-da5b-4b87-b927-fba149cc4bef"
life = SULT().set_interest(i=0.05)
a = life.temporary_annuity(40, t=10) + 0.5*life.deferred_annuity(40, u=10, t=10)
A = life.whole_life_insurance(40)
P = life.gross_premium(a=a, A=A, benefit=100000)
isclose(1150, P, question="Q6.14")

# + [markdown] id="ba7ed0a0"
# SOA Question 6.15:  (B) 1.002
#

# + colab={"base_uri": "https://localhost:8080/"} id="3b081e5c" outputId="a79a0eb2-6fba-4231-b190-2849d1d38048"
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

# + colab={"base_uri": "https://localhost:8080/"} id="b4867776" outputId="779e50ec-a612-444b-f55b-26999d0519f7"
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

# + colab={"base_uri": "https://localhost:8080/"} id="e84e6eb4" outputId="eb611de2-ba4d-4af7-a7c2-0d8cba7e6a72"
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

# + colab={"base_uri": "https://localhost:8080/"} id="0f94e213" outputId="27ac6099-fb6a-4596-fcb5-a9950bb33147"
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

# + colab={"base_uri": "https://localhost:8080/"} id="0ad57ec7" outputId="a1a792ee-c851-402a-f74d-6f66b8214628"
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

# + colab={"base_uri": "https://localhost:8080/"} id="d1afe338" outputId="64d34903-0e3d-44ec-a78c-3e2aee45c4e9"
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

# + colab={"base_uri": "https://localhost:8080/"} id="7c07aea5" outputId="be0ca60f-a228-4798-896f-2e188e46a096"
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

# + colab={"base_uri": "https://localhost:8080/"} id="e154a4ce" outputId="9dd469c1-ed38-4c9d-d8d3-e3df7ab5f861"
life=SULT(udd=True)
a = UDD(m=12, life=life).temporary_annuity(45, t=20)
A = UDD(m=0, life=life).whole_life_insurance(45)
P = life.gross_premium(A=A, a=a, benefit=100000) / 12
isclose(102, P, question="Q6.22")

# + [markdown] id="1f2bd9fa"
# SOA Question 6.23:  (D) 44.7
#

# + colab={"base_uri": "https://localhost:8080/"} id="4721d51b" outputId="bf0875ae-d05a-4626-f950-8a268fd382b7"
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

# + colab={"base_uri": "https://localhost:8080/"} id="092a752e" outputId="83923f6a-66fa-48f2-93d8-e8b6339e26ec"
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

# + colab={"base_uri": "https://localhost:8080/"} id="b27d7264" outputId="b52aeb37-78e1-4955-8830-31a7b953b95e"
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

# + colab={"base_uri": "https://localhost:8080/"} id="e0bc9ac7" outputId="8ac47789-329d-4c0b-b0d3-9248ca8d4fd5"
life = SULT().set_interest(i=0.05)
def fun(P): 
    return P - life.net_premium(90, b=1000, initial_cost=P)
P = life.solve(fun, target=0, grid=[150, 190])
isclose(180, P, question="Q6.26")

# + [markdown] id="984c9535"
# SOA Question 6.27:  (D) 10310
#

# + colab={"base_uri": "https://localhost:8080/"} id="f807f50d" outputId="060bc90e-da1a-4dcd-f9b7-8131e5510a12"
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

# + colab={"base_uri": "https://localhost:8080/"} id="e4d655be" outputId="58a434ce-1ce3-4960-b480-e3915d372b3d"
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

# + colab={"base_uri": "https://localhost:8080/"} id="19f0454d" outputId="f521c125-af53-49c9-8532-72423df35344"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a29edf61" outputId="5857d98a-4162-4354-edfb-4bf100d9f4b0"
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

# + colab={"base_uri": "https://localhost:8080/"} id="2dfd7470" outputId="da072e23-ddf1-4a4b-8c17-ae2c11ae0069"
life = ConstantForce(mu=0.01).set_interest(delta=0.05)
A = (life.term_insurance(35, t=35, discrete=False) 
     + life.E_x(35, t=35)*0.51791)     # A_35
P = life.premium_equivalence(A=A, b=100000, discrete=False)
isclose(1330, P, question="Q6.31")

# + [markdown] id="9876aca3"
# SOA Question 6.32:  (C) 550
#

# + colab={"base_uri": "https://localhost:8080/"} id="9775a2e0" outputId="c93d4c69-b676-4b51-b093-82b48840c969"
x = 0
life = Recursion().set_interest(i=0.05).set_a(9.19, x=x)
benefits = UDD(m=0, life=life).whole_life_insurance(x)
payments = UDD(m=12, life=life).whole_life_annuity(x)
P = life.gross_premium(a=payments, A=benefits, benefit=100000)/12
isclose(550, P, question="Q6.32")

# + [markdown] id="3765e3c2"
# SOA Question 6.33:  (B) 0.13
#

# + colab={"base_uri": "https://localhost:8080/"} id="5410107c" outputId="a40a6c88-d79e-4ea5-f08e-6b236286457d"
life = Insurance().set_survival(mu=lambda x,t: 0.02*t).set_interest(i=0.03)
x = 0
var = life.E_x(x, t=15, moment=life._VARIANCE, endowment=10000)
p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
isclose(0.13, p, question="Q6.33", rel_tol=0.02)

# + [markdown] id="d47dfed4"
# SOA Question 6.34:  (A) 23300
#

# + colab={"base_uri": "https://localhost:8080/"} id="a5ef2c99" outputId="cf9126a1-80a5-494a-d12a-2acf83e69aca"
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

# + colab={"base_uri": "https://localhost:8080/"} id="2079db39" outputId="26c60fcf-c5e0-4ece-dfdf-6138ebc4886e"
sult = SULT()
A = sult.whole_life_insurance(35, b=100000)
a = sult.whole_life_annuity(35)
P = sult.gross_premium(a=a, A=A, initial_premium=.19, renewal_premium=.04)
isclose(530, P, question="Q6.35")

# + [markdown] id="c0b919c2"
# SOA Question 6.36:  (B) 500
#

# + colab={"base_uri": "https://localhost:8080/"} id="f849ca70" outputId="b1200f3d-fe92-4dc4-8ac9-42b6194072f6"
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

# + colab={"base_uri": "https://localhost:8080/"} id="fa96592b" outputId="21e8c04c-d45f-4a90-9dc1-a718d7a908ec"
sult = SULT()
benefits = sult.whole_life_insurance(35, b=50000 + 100)
expenses = sult.immediate_annuity(35, b=100)
a = sult.temporary_annuity(35, t=10)
P = (benefits + expenses) / a
isclose(820, P, question="Q6.37")

# + [markdown] id="07e0a134"
# SOA Question 6.38:  (B) 11.3
#

# + colab={"base_uri": "https://localhost:8080/"} id="017b6427" outputId="537f2fb4-781e-4db9-fa08-5c5bfddb0ef1"
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

# + colab={"base_uri": "https://localhost:8080/"} id="4f13b0f0" outputId="3bc9b157-e829-42ae-ebf9-2eaa8e0196e9"
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

# + colab={"base_uri": "https://localhost:8080/"} id="464ce774" outputId="6960e111-7723-4996-de30-1621183a2618"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a76e5f76" outputId="a7e2d0ff-e1d9-4685-9f04-32b9557003bc"
x = 0
life = LifeTable().set_interest(i=0.05).set_table(q={x:.01, x+1:.02})
a = 1 + life.E_x(x, t=1) * 1.01
A = life.deferred_insurance(x, u=0, t=1) + 1.01*life.deferred_insurance(x, u=1, t=1)
P = 100000 * A / a
isclose(1417, P, question="Q6.41")

# + [markdown] id="88c94cf0"
# SOA Question 6.42:  (D) 0.113
#

# + colab={"base_uri": "https://localhost:8080/"} id="b7ba7db5" outputId="486cac14-847e-415b-f58b-512cd3e4c7a6"
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

# + colab={"base_uri": "https://localhost:8080/"} id="acfc0713" outputId="5b2db075-b4a6-4c73-f692-3ac4fe7136d2"
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

# + colab={"base_uri": "https://localhost:8080/"} id="11930092" outputId="1bb43750-0704-4c99-81f5-a4462e81cc4d"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 522} id="e434ceb8" outputId="36a56753-5525-44d3-8341-84f3645c71e4"
life = SULT(udd=True)
contract = Contract(benefit=100000, premium=560, discrete=False)
L = life.L_from_prob(x=35, prob=0.75, contract=contract)
life.L_plot(x=35, contract=contract, 
            T=life.L_to_t(L=L, contract=contract))
isclose(690, L, question="Q6.45")

# + [markdown] id="96fbe650"
# SOA Question 6.46:  (E) 208
#

# + colab={"base_uri": "https://localhost:8080/"} id="4467d9db" outputId="1e064427-1d40-4524-eb59-e5765a4cd8da"
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

# + colab={"base_uri": "https://localhost:8080/"} id="5f701e65" outputId="87db7fc3-48dc-45b0-db79-6de75a5444c2"
sult = SULT()
a = sult.temporary_annuity(70, t=10)
A = sult.deferred_annuity(70, u=10)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
                        renewal_premium=0.05)
isclose(66400, P, question="Q6.47")

# + [markdown] id="7ed3e46c"
# SOA Question 6.48:  (A) 3195 -- example of deep insurance recursion
#

# + colab={"base_uri": "https://localhost:8080/"} id="022f6301" outputId="68a325b9-2d97-476d-bfc2-e0c36019dcb4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="c0fe3957" outputId="6593ce49-458a-4489-e5a9-edfde9e285f6"
sult = SULT(udd=True)
a = UDD(m=12, life=sult).temporary_annuity(40, t=20)
A = sult.whole_life_insurance(40, discrete=False)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_policy=200,
                       renewal_premium=0.04, initial_premium=0.04) / 12
isclose(86, P, question="Q6.49")

# + [markdown] id="c442a990"
# SOA Question 6.50:  (A) -47000
#

# + colab={"base_uri": "https://localhost:8080/"} id="f5713876" outputId="026de178-685d-4d57-f4d4-c90fe5b5a27b"
life = SULT()
P = life.premium_equivalence(a=life.whole_life_annuity(35), b=1000) 
a = life.deferred_annuity(35, u=1, t=1)
A = life.term_insurance(35, t=1, b=1000)
cash = (A - a * P) * 10000 / life.interest.v
isclose(-47000, cash, question="Q6.50")

# + [markdown] id="8aa16e1d"
# SOA Question 6.51:  (D) 34700
#

# + colab={"base_uri": "https://localhost:8080/"} id="7def4285" outputId="ca87ff06-ba3f-49ab-dd99-d56113c7197d"
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

# + colab={"base_uri": "https://localhost:8080/"} id="4529d7f9" outputId="990c0666-fe37-49aa-c328-51104bc2da6b"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a9d23ae6" outputId="831df386-d243-4a38-8135-aff0e29af063"
x = 0
life = LifeTable().set_interest(i=0.08).set_table(q={x:.1, x+1:.1, x+2:.1})
A = life.term_insurance(x, t=3)
P = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
isclose(720, P, question="Q6.53")

# + [markdown] id="41e939f0"
# SOA Question 6.54:  (A) 25440
#

# + colab={"base_uri": "https://localhost:8080/"} id="2ea3fc85" outputId="d1d9bcd6-0894-4605-dd25-3f4a44d69c89"
life = SULT()
std = math.sqrt(life.net_policy_variance(45, b=200000))
isclose(25440, std, question="Q6.54")

# + [markdown] id="04a31b19"
# ## 7 Policy Values

# + [markdown] id="b265fc75"
# SOA Question 7.1:  (C) 11150
#

# + colab={"base_uri": "https://localhost:8080/"} id="628c0418" outputId="e3e2ec59-0e2b-420a-9277-44542270cafa"
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

# + colab={"base_uri": "https://localhost:8080/"} id="76adf1c8" outputId="1a57e6cf-fa97-421f-eb4b-e007ddbdecb1"
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

# + colab={"base_uri": "https://localhost:8080/"} id="985efb46" outputId="75c17657-64a3-4c9c-9fd8-f5220c65ce9a"
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

# + colab={"base_uri": "https://localhost:8080/"} id="c4940dc2" outputId="d4555e21-59c9-4d52-f36d-a5660de18442"
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

# + colab={"base_uri": "https://localhost:8080/"} id="0605610d" outputId="6b36385d-3cc9-44e5-98a3-c0ce1da4f6b9"
x = 0
life = Recursion(udd=True).set_interest(i=0.03)\
                          .set_q(0.04561, x=x+4)\
                          .set_reserves(T=3, V={4: 1405.08})
V = life.r_V_forward(x, s=4, r=0.5, benefit=10000, premium=647.46)
isclose(1900, V, question="Q7.5")

# + [markdown] id="984679f6"
# Answer 7.6:  (E) -25.4
#

# + colab={"base_uri": "https://localhost:8080/"} id="bf5b013c" outputId="0987c107-86be-46e3-fef4-5fbb0a401ee4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="a5d4d205" outputId="ae989774-7e55-4eb0-8db9-23be781de321"
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

# + colab={"base_uri": "https://localhost:8080/"} id="9311988d" outputId="76397b86-167e-4661-b3d1-ca91370e6fec"
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

# + colab={"base_uri": "https://localhost:8080/"} id="95ee4b66" outputId="a39768fc-33af-451c-e8f8-740877c107b4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="83268a47" outputId="7de54cc6-30cb-4944-ec28-ab998b7aa9c0"
life = SULT()
G = 977.6
P = life.net_premium(45, b=100000)
contract = Contract(benefit=0, premium=G-P, renewal_policy=.02*G + 50)
V = life.gross_policy_value(45, t=5, contract=contract)
isclose(-970, V, question="Q7.10")

# + [markdown] id="55157c76"
# SOA Question 7.11:  (B) 1460
#

# + colab={"base_uri": "https://localhost:8080/"} id="2e0c21b0" outputId="55159429-6e46-48e7-92d6-57a7bad68cc4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="7993f016" outputId="9767525f-ceda-4743-c1a3-5f098fb73e35"
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

# + colab={"base_uri": "https://localhost:8080/"} id="0b8778cc" outputId="e86b40bf-745e-4d08-a5ae-bc70e33c4c37"
life = SULT()
V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
isclose(180, V, question="Q7.13")

# + [markdown] id="58f053bd"
# SOA Question 7.14:  (A) 2200
#

# + colab={"base_uri": "https://localhost:8080/"} id="e0bf9ee2" outputId="91220904-60db-453c-9ea0-3931de0b55b2"
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

# + colab={"base_uri": "https://localhost:8080/"} id="bbed6a97" outputId="5ceafecc-598b-465b-b1b5-a509c56afca5"
x = 0
V = Recursion(udd=True).set_interest(i=0.05)\
                       .set_q(0.1, x=x+15)\
                       .set_reserves(T=3, V={16: 49.78})\
                       .r_V_backward(x, s=15, r=0.6, benefit=100)
isclose(50.91, V, question="Q7.15")

# + [markdown] id="cf793972"
# SOA Question 7.16:  (D) 380
#

# + colab={"base_uri": "https://localhost:8080/"} id="b3317669" outputId="9a993a84-58eb-4ef6-a4f6-8bd6cfd33a23"
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

# + colab={"base_uri": "https://localhost:8080/"} id="e26e27a3" outputId="9ca91101-2582-49c4-ca50-8c34c8e7e7d4"
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

# + colab={"base_uri": "https://localhost:8080/"} id="789aef65" outputId="08fc06d5-23b0-47b8-cab4-464ce5900496"
x = 10
life = Recursion(verbose=False).set_interest(i=0.04).set_q(0.009, x=x)
def fun(a):
    return life.set_a(a, x=x).net_policy_value(x, t=1)
a = life.solve(fun, target=0.012, grid=[17.1, 19.1])
isclose(17.1, a, question="Q7.18")

# + [markdown] id="bcd7d9ae"
# SOA Question 7.19:  (D) 720
#

# + colab={"base_uri": "https://localhost:8080/"} id="9372ed3c" outputId="e7299adc-0587-49ac-f6be-ad28a7cd12ef"
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

# + colab={"base_uri": "https://localhost:8080/"} id="ab37f0e4" outputId="5b19e726-20b7-4479-8712-64f957268bc8"
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

# + colab={"base_uri": "https://localhost:8080/"} id="25c496da" outputId="7bbf37b7-2194-48e5-8741-0f00ca75ed76"
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

# + colab={"base_uri": "https://localhost:8080/"} id="6a2abdfc" outputId="eaec365e-118b-42b1-a2c3-9da83962e4e8"
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

# + colab={"base_uri": "https://localhost:8080/"} id="c4c42da2" outputId="e4ea3f12-649d-4e2f-b614-dcb99e226f4d"
life = Recursion().set_interest(i=0.04).set_p(0.995, x=25)
A = life.term_insurance(25, t=1, b=10000)
def fun(beta):  # value of premiums in first 20 years must be equal
    return beta * 11.087 + (A - beta) 
beta = life.solve(fun, target=216 * 11.087, grid=[140, 260])
isclose(233, beta, question="Q7.23")

# + [markdown] id="4b82caf4"
# SOA Question 7.24:  (C) 680
#

# + colab={"base_uri": "https://localhost:8080/"} id="75d8d20b" outputId="4a1e843e-c342-4b5b-b9fa-a8b11d772431"
life = SULT()
P = life.premium_equivalence(A=life.whole_life_insurance(50), b=1000000)
isclose(680, 11800 - P, question="Q7.24")

# + [markdown] id="8410e40c"
# SOA Question 7.25:  (B) 3947.37
#

# + colab={"base_uri": "https://localhost:8080/"} id="9928c491" outputId="7b6ece99-0b78-4860-c706-3ff13a520657"
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

# + colab={"base_uri": "https://localhost:8080/"} id="9fc808b8" outputId="83fb08e0-06da-4cb8-92a4-05f9ed1fc035"
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

# + colab={"base_uri": "https://localhost:8080/"} id="7ad342a1" outputId="f223b0c4-cda8-4739-bc6c-4fc51c84a85d"
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

# + colab={"base_uri": "https://localhost:8080/"} id="99412e64" outputId="87fc6efa-2bb9-4bc5-f91d-b74731102985"
life = SULT()
PW = life.net_premium(65, b=1000)   # 20_V=0 => P+W is net premium for A_65
P = life.net_premium(45, t=20, b=1000)  # => P is net premium for A_45:20
isclose(24.3, PW - P, question="Q7.28")

# + [markdown] id="04bd97d2"
# SOA Question 7.29:  (E) 2270
#

# + colab={"base_uri": "https://localhost:8080/"} id="e301c605" outputId="6a27f280-1530-4b9f-dda9-4b26dd1366c8"
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

# + colab={"base_uri": "https://localhost:8080/"} id="cdd25b58" outputId="27b419e7-2e45-4c15-b867-a7c22fbb8c08"
contract = Contract(premium=0, benefit=10000)  # premiums=0 after t=10
L = SULT().gross_policy_value(35, contract=contract)
V = SULT().set_interest(i=0).gross_policy_value(35, contract=contract) # 10000
isclose(9035, V - L, question="Q7.30")

# + [markdown] id="df03679b"
# SOA Question 7.31:  (E) 0.310
#

# + colab={"base_uri": "https://localhost:8080/"} id="0d9cd985" outputId="0756a3a7-c334-4159-804c-f71f5a395664"
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

# + colab={"base_uri": "https://localhost:8080/"} id="cf0604c6" outputId="0a51b4dd-cf4a-4ef1-d517-a576b01fa30b"
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

# + colab={"base_uri": "https://localhost:8080/"} id="40395bce" outputId="392524a9-6e25-4857-a99d-c517a8a21905"
from datetime import datetime
print(datetime.now())
print(isclose)
