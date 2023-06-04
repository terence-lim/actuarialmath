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

# + [markdown] id="j-xP1JxMKaiC"
# # Sample Solutions and Hints

# + id="qjZAYOOm9vGS" vscode={"languageId": "python"}
"""Solutions code for SOA FAM-L sample questions

Copyright 2022-2023, Terence Lim

MIT License
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
from actuarialmath.constantforce import ConstantForce
from actuarialmath.extrarisk import ExtraRisk
from actuarialmath.mthly import Mthly
from actuarialmath.udd import UDD
from actuarialmath.woolhouse import Woolhouse

# + [markdown] id="YNWb-AbrUimB"
# __Helper to compare computed answers to expected solutions__

# + id="3PKYpYkr9vGU" vscode={"languageId": "python"}
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

# + [markdown] id="Y_syVwbN9vGV"
# ## 1 Tables
#
#
# These tables are provided in the FAM-L exam
# - Interest Functions at i=0.05
# - Normal Distribution Table
# - Standard Ultimate Life Table
#
# but you actually do not need them here!

# + colab={"base_uri": "https://localhost:8080/", "height": 224} id="4nJ6b-Gq9vGV" outputId="ddab64ab-54ea-4947-c7da-f5c06e11e399" vscode={"languageId": "python"}
print("Interest Functions at i=0.05")
UDD.interest_frame()

# + colab={"base_uri": "https://localhost:8080/"} id="eI07yLAj9vGW" outputId="2d8a0ca2-7af3-4f0d-d3fc-a6aab7df1a5b" vscode={"languageId": "python"}
print("Values of z for selected values of Pr(Z<=z)")
print(Life.quantiles_frame().to_string(float_format=lambda x: f"{x:.3f}"))

# + colab={"base_uri": "https://localhost:8080/", "height": 441} id="MUqzz5s79vGX" outputId="b1d3bf7c-d324-4df9-dc09-18dbcda5c34a" vscode={"languageId": "python"}
print("Standard Ultimate Life Table at i=0.05")
SULT().frame()


# + [markdown] id="EtWXmyOv9vGX"
# ## 2 Survival models

# + [markdown] id="apYDkhZd9vGX"
# SOA Question 2.1: (B) 2.5
# - derive formula for $\mu$ from given survival function
# - solve for $\omega$ given $\mu_{65}$
# - calculate $e$ by summing survival probabilities
#

# + colab={"base_uri": "https://localhost:8080/"} id="IvNoq-Ky9vGX" outputId="9fb08695-47cf-4a99-81af-d4d3cbf02d42" vscode={"languageId": "python"}
life = Lifetime()
def mu_from_l(omega):   # first solve for omega, given mu_65 = 1/180            
    return life.set_survival(l=lambda x,s: (1 - (x+s)/omega)**0.25).mu_x(65)
omega = int(life.solve(mu_from_l, target=1/180, grid=100))
e = life.set_survival(l=lambda x,s:(1 - (x + s)/omega)**.25, maxage=omega)\
        .e_x(106)       # then solve expected lifetime from omega              
isclose(2.5, e, question="Q2.1")

# + [markdown] id="56F6N4eH9vGY"
# SOA Question 2.2: (D) 400
# - calculate survival probabilities for the two scenarios
# - apply conditional variance formula (or mixed distribution)

# + colab={"base_uri": "https://localhost:8080/"} id="3Ot-ELpG9vGY" outputId="1aa0ca33-9697-4011-e347-7ec7b7f0cff7" vscode={"languageId": "python"}
p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
std = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
isclose(400, std, question="Q2.2")

# + [markdown] id="u00epu5W9vGY"
# SOA Question 2.3: (A) 0.0483
# 1. Derive formula for $f$ given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="SzkD8YvY9vGY" outputId="a5271002-1dd6-4b4f-d416-6a4d1837cc86" vscode={"languageId": "python"}
B, c = 0.00027, 1.1
S = lambda x,s,t: math.exp(-B * c**(x+s) * (c**t - 1)/math.log(c))
life = Survival().set_survival(S=S)
f = life.f_x(x=50, t=10)
isclose(0.0483, f, question="Q2.3")

# + [markdown] id="tLF2sqau9vGZ"
# SOA Question 2.4: (E) 8.2
# - derive survival probability function $_tp_x$ given $_tq_0$
# - compute $\overset{\circ}{e}$ by integration
#

# + colab={"base_uri": "https://localhost:8080/"} id="OwnBc1a09vGZ" outputId="5e9b08e0-a8c7-4935-9678-fca85c26be5f" vscode={"languageId": "python"}
def l(x, s): return 0. if (x+s) >= 100 else 1 - ((x + s)**2) / 10000.
e = Lifetime().set_survival(l=l).e_x(75, t=10, curtate=False)
isclose(8.2, e, question="Q2.4")

# + [markdown] id="OlHhfW0y9vGZ"
# SOA Question 2.5:  (B) 37.1
# - solve for $e_{40}$ from limited lifetime formula
# - compute $e_{41}$ using backward recursion

# + colab={"base_uri": "https://localhost:8080/"} id="Xsz2NQ0S9vGZ" outputId="7a9324f7-5ba5-4327-d922-a3d280fd3158" vscode={"languageId": "python"}
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

# + [markdown] id="kKmy8mEM9vGa"
# SOA Question 2.6: (C) 13.3
# - derive force of mortality function $\mu$ from given survival function
#

# + colab={"base_uri": "https://localhost:8080/"} id="_m2YW3rR9vGa" outputId="6841d764-f5f3-434c-c5c5-ccaa241510dc" vscode={"languageId": "python"}
life = Survival().set_survival(l=lambda x,s: (1 - (x+s)/60)**(1/3))
mu = 1000 * life.mu_x(35)
isclose(13.3, mu, question="Q2.6")

# + [markdown] id="b6ENXbNF9vGa"
# SOA Question 2.7: (B) 0.1477
# - calculate from given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="PQvNwS069vGa" outputId="9482dbc9-96c8-4cde-f595-4e0236e8f962" vscode={"languageId": "python"}
l = lambda x,s: (1-((x+s)/250) if (x+s)<40 else 1-((x+s)/100)**2)
q = Survival().set_survival(l=l).q_x(30, t=20)
isclose(0.1477, q, question="Q2.7")

# + [markdown] id="5ciwwj969vGb"
# SOA Question 2.8: (C) 0.94
# - relate $p_{male}$ and $p_{female}$ through the common term $\mu$ and the given proportions
#

# + colab={"base_uri": "https://localhost:8080/"} id="-8592SF99vGb" outputId="a01b8527-99c5-446a-da0a-3c56544d5602" vscode={"languageId": "python"}
def fun(mu):  # Solve first for mu, given ratio of start and end proportions
    male = Survival().set_survival(mu=lambda x,s: 1.5 * mu)
    female = Survival().set_survival(mu=lambda x,s: mu)
    return (75 * female.p_x(0, t=20)) / (25 * male.p_x(0, t=20))
mu = Survival.solve(fun, target=85/15, grid=[0.89, 0.99])
p = Survival().set_survival(mu=lambda x,s: mu).p_x(0, t=1)
isclose(0.94, p, question="Q2.8")

# + [markdown] id="-NtYFLJe9vGb"
# ## 3 Life tables and selection

# + [markdown] id="ITFsT3UN9vGb"
# SOA Question 3.1:  (B) 117
# - interpolate with constant force of maturity
#

# + colab={"base_uri": "https://localhost:8080/"} id="QH6OUCeM9vGc" outputId="867b9603-bcdb-4973-da71-0a09624324e3" vscode={"languageId": "python"}
life = SelectLife().set_table(l={60: [80000, 79000, 77000, 74000],
                                 61: [78000, 76000, 73000, 70000],
                                 62: [75000, 72000, 69000, 67000],
                                 63: [71000, 68000, 66000, 65000]})
q = 1000 * life.q_r(60, s=0, r=0.75, t=3, u=2)
isclose(117, q, question="Q3.1")

# + [markdown] id="tBMXphAz9vGc"
# SOA Question 3.2:  (D) 14.7
# - UDD $\Rightarrow \overset{\circ}{e}_{x} = e_x + 0.5$
# - fill select table using curtate expectations 
#

# + colab={"base_uri": "https://localhost:8080/"} id="0ytZ_yjK9vGc" outputId="a57bfdcf-92c6-497e-b7a1-a3cbbe370a4c" vscode={"languageId": "python"}
e_curtate = Fractional.e_approximate(e_complete=15)
life = SelectLife(udd=True).set_table(l={65: [1000, None,],
                                         66: [955, None]},
                                      e={65: [e_curtate, None]},
                                      d={65: [40, None,],
                                         66: [45, None]})
e = life.e_r(66)
isclose(14.7, e, question="Q3.2")

# + [markdown] id="mYqBhG_O9vGc"
# SOA Question 3.3:  (E) 1074
# - interpolate lives between integer ages with UDD

# + colab={"base_uri": "https://localhost:8080/"} id="RKyrAGA49vGc" outputId="8c1e364d-e663-4c29-8660-1dda9c86a3e1" vscode={"languageId": "python"}
life = SelectLife().set_table(l={50: [99, 96, 93],
                                 51: [97, 93, 89],
                                 52: [93, 88, 83],
                                 53: [90, 84, 78]})
q = 10000 * life.q_r(51, s=0, r=0.5, t=2.2)
isclose(1074, q, question="Q3.3")

# + [markdown] id="zs6-mPxu9vGd"
# SOA Question 3.4:  (B) 815
# - compute portfolio percentile with N=4000, and mean and variance  from binomial distribution

# + colab={"base_uri": "https://localhost:8080/"} id="ayBjo7eX9vGd" outputId="9712b5fc-cc4f-4819-9ccd-b9302aa08437" vscode={"languageId": "python"}
sult = SULT()
mean = sult.p_x(25, t=95-25)
var = sult.bernoulli(mean, variance=True)
pct = sult.portfolio_percentile(N=4000, mean=mean, variance=var, prob=0.1)
isclose(815, pct, question="Q3.4")

# + [markdown] id="PcHjyYvt9vGd"
# SOA Question 3.5:  (E) 106
# - compute mortality rates by interpolating lives between integer ages, with UDD and constant force of mortality assumptions

# + colab={"base_uri": "https://localhost:8080/"} id="yasU-2fB9vGd" outputId="37f0b235-abf1-4c05-8d0b-65c73d29bc74" vscode={"languageId": "python"}
l = [99999, 88888, 77777, 66666, 55555, 44444, 33333, 22222]
a = LifeTable(udd=True).set_table(l={age:l for age,l in zip(range(60, 68), l)})\
                       .q_r(60, u=3.4, t=2.5)
b = LifeTable(udd=False).set_table(l={age:l for age,l in zip(range(60, 68), l)})\
                        .q_r(60, u=3.4, t=2.5)
isclose(106, 100000 * (a - b), question="Q3.5")

# + [markdown] id="8sjosZUB9vGd"
# SOA Question 3.6:  (D) 15.85
# - apply recursion formulas for curtate expectation
#

# + colab={"base_uri": "https://localhost:8080/"} id="97N6Zr_A9vGe" outputId="56629dd6-6d08-4597-8eac-7f37f7bf27e5" vscode={"languageId": "python"}
e = SelectLife().set_table(q={60: [.09, .11, .13, .15],
                              61: [.1, .12, .14, .16],
                              62: [.11, .13, .15, .17],
                              63: [.12, .14, .16, .18],
                              64: [.13, .15, .17, .19]},
                           e={61: [None, None, None, 5.1]})\
                .e_x(61)
isclose(5.85, e, question="Q3.6")

# + [markdown] id="YI1-y6aZ9vGe"
# SOA Question 3.7: (b) 16.4
# - use deferred mortality formula
# - use chain rule for survival probabilities,
# - interpolate between integer ages with constant force of mortality
#

# + colab={"base_uri": "https://localhost:8080/"} id="UbANhNs89vGe" outputId="f39f76bf-ea99-41fc-9d66-eaf88c5c9703" vscode={"languageId": "python"}
life = SelectLife().set_table(q={50: [.0050, .0063, .0080],
                                 51: [.0060, .0073, .0090],
                                 52: [.0070, .0083, .0100],
                                 53: [.0080, .0093, .0110]})
q = 1000 * life.q_r(50, s=0, r=0.4, t=2.5)
isclose(16.4, q, question="Q3.7")

# + [markdown] id="RLBoPZc49vGe"
# SOA Question 3.8:  (B) 1505
# - compute portfolio means and variances from sum of 2000 independent members' means and variances of survival.
#

# + colab={"base_uri": "https://localhost:8080/"} id="gCl-zHJt9vGe" outputId="8d4761b4-7c19-4d0d-8b53-aaf7c379d76e" vscode={"languageId": "python"}
sult = SULT()
p1 = sult.p_x(35, t=40)
p2 = sult.p_x(45, t=40)
mean = sult.bernoulli(p1) * 1000 + sult.bernoulli(p2) * 1000
var = (sult.bernoulli(p1, variance=True) * 1000 
       + sult.bernoulli(p2, variance=True) * 1000)
pct = sult.portfolio_percentile(mean=mean, variance=var, prob=.95)
isclose(1505, pct, question="Q3.8")

# + [markdown] id="vFdt6XLC9vGf"
# SOA Question 3.9:  (E) 3850
# - compute portfolio means and variances as sum of 4000 independent members' means and variances (of survival)
# - retrieve normal percentile
#

# + colab={"base_uri": "https://localhost:8080/"} id="Gsl5hbLG9vGf" outputId="f923df97-fd2c-4fdc-f9d8-0338af0584dd" vscode={"languageId": "python"}
sult = SULT()
p1 = sult.p_x(20, t=25)
p2 = sult.p_x(45, t=25)
mean = sult.bernoulli(p1) * 2000 + sult.bernoulli(p2) * 2000
var = (sult.bernoulli(p1, variance=True) * 2000 
       + sult.bernoulli(p2, variance=True) * 2000)
pct = sult.portfolio_percentile(mean=mean, variance=var, prob=.99)
isclose(3850, pct, question="Q3.9")

# + [markdown] id="ZvOJY-Om9vGf"
# SOA Question 3.10:  (C) 0.86
# - reformulate the problem by reversing time: survival to year 6 is calculated in reverse as discounting by the same number of years. 
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="DuRXTRuj9vGf" outputId="1d04e74d-5e3d-4b8c-dd2b-dc396c02faa3" vscode={"languageId": "python"}
interest = Interest(v=0.75)
L = 35*interest.annuity(t=4, due=False) + 75*interest.v_t(t=5)
interest = Interest(v=0.5)
R = 15*interest.annuity(t=4, due=False) + 25*interest.v_t(t=5)
isclose(0.86, L / (L + R), question="Q3.10")

# + [markdown] id="j1u9d7mY9vGf"
# SOA Question 3.11:  (B) 0.03
# - calculate mortality rate by interpolating lives assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="3a01O1mI9vGf" outputId="881c1038-acc7-499d-8bdf-32278952314b" vscode={"languageId": "python"}
life = LifeTable(udd=True).set_table(q={50//2: .02, 52//2: .04})
q = life.q_r(50//2, t=2.5/2)
isclose(0.03, q, question="Q3.11")

# + [markdown] id="8__Xih3p9vGg"
# SOA Question 3.12: (C) 0.055 
# - compute survival probability by interpolating lives assuming constant force
#

# + colab={"base_uri": "https://localhost:8080/"} id="UFMwKnsv9vGg" outputId="cc49bfd6-af4a-4545-fd36-00d9659eadf5" vscode={"languageId": "python"}
life = SelectLife(udd=False).set_table(l={60: [10000, 9600, 8640, 7771],
                                          61: [8654, 8135, 6996, 5737],
                                          62: [7119, 6549, 5501, 4016],
                                          63: [5760, 4954, 3765, 2410]})
q = life.q_r(60, s=1, t=3.5) - life.q_r(61, s=0, t=3.5)               
isclose(0.055, q, question="Q3.12")

# + [markdown] id="_jC3gF9i9vGg"
# SOA Question 3.13:  (B) 1.6
# - compute curtate expectations using recursion formulas
# - convert to complete expectation assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="zvLIlGMT9vGg" outputId="9771a171-0863-40b0-93b4-5a8cec29a569" vscode={"languageId": "python"}
life = SelectLife().set_table(l={55: [10000, 9493, 8533, 7664],
                                 56: [8547, 8028, 6889, 5630],
                                 57: [7011, 6443, 5395, 3904],
                                 58: [5853, 4846, 3548, 2210]},
                              e={57: [None, None, None, 1]})
e = life.e_r(58, s=2)
isclose(1.6, e, question="Q3.13")

# + [markdown] id="jGkN4MxF9vGh"
# SOA Question 3.14:  (C) 0.345
# - compute mortality by interpolating lives between integer ages assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="5VIidCUH9vGh" outputId="5fab7f27-b99e-4797-8272-9ff8f33bd730" vscode={"languageId": "python"}
life = LifeTable(udd=True).set_table(l={90: 1000, 93: 825},
                                     d={97: 72},
                                     p={96: .2},
                                     q={95: .4, 97: 1})
q = life.q_r(90, u=93-90, t=95.5 - 93)
isclose(0.345, q, question="Q3.14")

# + [markdown] id="xu628P689vGh"
# ## 4 Insurance benefits

# + [markdown] id="7SFaQ8Vf9vGh"
# SOA Question 4.1:  (A) 0.27212
# - solve EPV as sum of term and deferred insurance
# - compute variance as difference of second moment and first moment squared
#

# + colab={"base_uri": "https://localhost:8080/"} id="OyKmE7sr9vGh" outputId="c4cac0f0-573b-46a9-e237-c52e1356820f" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.03)
life.set_A(0.36987, x=40).set_A(0.62567, x=60)
life.set_E(0.51276, x=40, t=20).set_E(0.17878, x=60, t=20)
Z2 = 0.24954
A = (2 * life.term_insurance(40, t=20) + life.deferred_insurance(40, u=20))
std = math.sqrt(life.insurance_variance(A2=Z2, A1=A))
isclose(0.27212, std, question="Q4.1")

# + [markdown] id="PMizQeAd9vGh"
# SOA Question 4.2:  (D) 0.18
# - calculate Z(t) and deferred mortality for each half-yearly t
# - sum the deferred mortality probabilities for periods when PV > 277000 
#

# + colab={"base_uri": "https://localhost:8080/"} id="y5YHGW479vGi" outputId="5e8415d8-0054-4aeb-b6f3-1ac3ec090f19" vscode={"languageId": "python"}
life = LifeTable(udd=False).set_table(q={0: .16, 1: .23})\
                           .set_interest(i_m=.18, m=2)
mthly = Mthly(m=2, life=life)
Z = mthly.Z_m(0, t=2, benefit=lambda x,t: 300000 + t*30000*2)
p = Z[Z['Z'] >= 277000]['q'].sum()
isclose(0.18, p, question="Q4.2")

# + [markdown] id="-uc-Hq4O9vGi"
# SOA Question 4.3: (D) 0.878
# - solve $q_{61}$ from endowment insurance EPV formula
# - solve $A_{60:\overline{3|}}$ with new $i=0.045$ as EPV of endowment insurance benefits.
#

# + colab={"base_uri": "https://localhost:8080/"} id="8F-OgIlf9vGi" outputId="98185107-8620-4ffa-c6e5-3328d2992d23" vscode={"languageId": "python"}
life = Recursion(verbose=False).set_interest(i=0.05).set_q(0.01, x=60)
def fun(q):   # solve for q_61
    return life.set_q(q, x=61).endowment_insurance(60, t=3)
life.solve(fun, target=0.86545, grid=0.01)
A = life.set_interest(i=0.045).endowment_insurance(60, t=3)
isclose(0.878, A, question="Q4.3")

# + [markdown] id="o-gdUvGK9vGi"
# SOA Question 4.4  (A) 0.036
# - integrate to find EPV of $Z$ and $Z^2$
# - variance is difference of second moment and first moment squared
#

# + colab={"base_uri": "https://localhost:8080/"} id="L3VMj3PE9vGi" outputId="dfe180d2-92c7-4f4c-ca42-58ff8b4d6bdf" vscode={"languageId": "python"}
x = 40
life = Insurance().set_survival(f=lambda *x: 0.025, maxage=x+40)\
                  .set_interest(v_t=lambda t: (1 + .2*t)**(-2))
def benefit(x,t): return 1 + .2 * t
A1 = life.A_x(x, benefit=benefit, discrete=False)
A2 = life.A_x(x, moment=2, benefit=benefit, discrete=False)
var = A2 - A1**2
isclose(0.036, var, question="Q4.4")

# + [markdown] id="SgPDXery9vGj"
# SOA Question 4.5:  (C) 35200
# - interpolate between integer ages with UDD, and find lifetime that mortality rate exceeded
# - compute PV of death benefit paid at that time.
#

# + colab={"base_uri": "https://localhost:8080/"} id="SBuV1-di9vGj" outputId="94f3651a-90ae-4ccb-e381-a6c3efc47050" vscode={"languageId": "python"}
sult = SULT(udd=True).set_interest(delta=0.05)
Z = 100000 * sult.Z_from_prob(45, 0.95, discrete=False)
isclose(35200, Z, question="Q4.5")

# + [markdown] id="pD76ZxaT9vGj"
# SOA Question 4.6:  (B) 29.85
# - calculate adjusted mortality rates
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="leNwlHUj9vGj" outputId="7f3b04b9-5bf1-4e01-c52b-512857f4cc50" vscode={"languageId": "python"}
sult = SULT()
life = LifeTable().set_interest(i=0.05)\
                  .set_table(q={70+k: .95**k * sult.q_x(70+k) for k in range(3)})
A = life.term_insurance(70, t=3, b=1000)
isclose(29.85, A, question="Q4.6")


# + [markdown] id="zTFV6rvp9vGj"
# SOA Question 4.7:  (B) 0.06
# - use Bernoulli shortcut formula for variance of pure endowment Z 
# - solve for $i$, since $p$ is given.

# + colab={"base_uri": "https://localhost:8080/"} id="Q45Wwz8W9vGk" outputId="23b0d69f-03d6-4474-ae28-5cb6286a20b4" vscode={"languageId": "python"}
def fun(i):
    life = Recursion(verbose=False).set_interest(i=i)\
                                   .set_p(0.57, x=0, t=25)
    return 0.1*life.E_x(0, t=25) - life.E_x(0, t=25, moment=life._VARIANCE)
i = Recursion.solve(fun, target=0, grid=[0.058, 0.066])
isclose(0.06, i, question="Q4.7")

# + [markdown] id="fvqumlvt9vGk"
# SOA Question 4.8  (C) 191
#
# - use insurance recursion with special interest rate $i=0.04$ in first year.
#

# + colab={"base_uri": "https://localhost:8080/"} id="a1kCG1qL9vGk" outputId="21a64320-13db-443d-f733-3429100ef8c2" vscode={"languageId": "python"}
def v_t(t): return 1.04**(-t) if t < 1 else 1.04**(-1) * 1.05**(-t+1)
A = SULT().set_interest(v_t=v_t).whole_life_insurance(50, b=1000)
isclose(191, A, question="Q4.8")

# + [markdown] id="w08yNX6S9vGk"
# SOA Question 4.9:  (D) 0.5
# - use whole-life, term and endowment insurance relationships.
#

# + colab={"base_uri": "https://localhost:8080/"} id="oElA_bNq9vGk" outputId="c8a4c7e1-9eb6-4546-c91f-61d79a2ef857" vscode={"languageId": "python"}
E = Recursion().set_A(0.39, x=35, t=15, endowment=1)\
               .set_A(0.25, x=35, t=15)\
               .E_x(35, t=15)
life = Recursion().set_A(0.32, x=35)\
                  .set_E(E, x=35, t=15)
def fun(A): return life.set_A(A, x=50).term_insurance(35, t=15)
A = life.solve(fun, target=0.25, grid=[0.35, 0.55])
isclose(0.5, A, question="Q4.9")

# + [markdown] id="o-YWaRHc9vGl"
# SOA Question 4.10:  (D)
# - draw and compared benefit diagrams
#

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="GPiuEBqc9vGl" outputId="2ef1a7d0-1adb-49bf-f7d5-95006182659e" vscode={"languageId": "python"}
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

# + [markdown] id="6ZsrFQqm9vGl"
# SOA Question 4.11:  (A) 143385
# - compute endowment insurance = term insurance + pure endowment 
# - apply formula of variance as the difference of second moment and first moment squared.
#

# + colab={"base_uri": "https://localhost:8080/"} id="zFQaNDuK9vGl" outputId="a4a0b471-073c-4635-b63d-d42efc6f40bd" vscode={"languageId": "python"}
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

# + [markdown] id="Zz-8ysEx9vGl"
# SOA Question 4.12:  (C) 167 
# - since $Z_1,~Z_2$ are non-overlapping, $E[Z_1~ Z_2] = 0$ for computing $Cov(Z_1, Z_2)$
# - whole life is sum of term and deferred, hence equals variance of components plus twice their covariance
#

# + colab={"base_uri": "https://localhost:8080/"} id="4TdupEj_9vGm" outputId="9bbe3441-b9c6-4c3e-d949-a2e92abc8725" vscode={"languageId": "python"}
cov = Life.covariance(a=1.65, b=10.75, ab=0)  # E[Z1 Z2] = 0 nonoverlapping
var = Life.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov)
isclose(167, var, question="Q4.12")

# + [markdown] id="ErkNkhz49vGm"
# SOA Question 4.13:  (C) 350 
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="EFwyjXOa9vGm" outputId="a275717e-efdb-44a7-c24b-3f47bbb08185" vscode={"languageId": "python"}
life = SelectLife().set_table(q={65: [.08, .10, .12, .14],
                                 66: [.09, .11, .13, .15],
                                 67: [.10, .12, .14, .16],
                                 68: [.11, .13, .15, .17],
                                 69: [.12, .14, .16, .18]})\
                   .set_interest(i=.04)
A = life.deferred_insurance(65, t=2, u=2, b=2000)
isclose(350, A, question="Q4.13")

# + [markdown] id="eZf5HCyv9vGm"
# SOA Question 4.14:  (E) 390000
# - discount (by interest rate $i=0.05$) the value at the portfolio percentile, of the sum of 400 bernoulli r.v. with survival probability $_{25}p_{60}$
#

# + colab={"base_uri": "https://localhost:8080/"} id="DUqv602M9vGm" outputId="e452a3ef-7319-4fea-8645-d73e23527485" vscode={"languageId": "python"}
sult = SULT()
p = sult.p_x(60, t=85-60)
mean = sult.bernoulli(p)
var = sult.bernoulli(p, variance=True)
F = sult.portfolio_percentile(mean=mean, variance=var, prob=.86, N=400)
F *= 5000 * sult.interest.v_t(85-60)
isclose(390000, F, question="Q4.14")

# + [markdown] id="lNmrHU8c9vGm"
# SOA Question 4.15  (E) 0.0833 
# - this special benefit function has effect of reducing actuarial discount rate to use in constant force of mortality shortcut formulas
#

# + colab={"base_uri": "https://localhost:8080/"} id="LYpdSDwG9vGn" outputId="1b409a7d-ce84-42a0-dd99-05e4c7812980" vscode={"languageId": "python"}
life = Insurance().set_survival(mu=lambda *x: 0.04).set_interest(delta=0.06)
benefit = lambda x,t: math.exp(0.02*t)
A1 = life.A_x(0, benefit=benefit, discrete=False)
A2 = life.A_x(0, moment=2, benefit=benefit, discrete=False)
var = life.insurance_variance(A2=A2, A1=A1)
isclose(0.0833, var, question="Q4.15")

# + [markdown] id="Mc72yVrV9vGn"
# SOA Question 4.16:  (D) 0.11
# - compute EPV of future benefits with adjusted mortality rates

# + colab={"base_uri": "https://localhost:8080/"} id="rlvtqXRz9vGn" outputId="ff80a79a-ff5e-4954-a213-3e2108cabc84" vscode={"languageId": "python"}
q = [.045, .050, .055, .060]
q = {50 + x: [q[x] * 0.7 if x < len(q) else None, 
              q[x+1] * 0.8 if x + 1 < len(q) else None, 
              q[x+2] if x + 2 < len(q) else None] 
     for x in range(4)}
life = SelectLife().set_table(q=q).set_interest(i=.04)
A = life.term_insurance(50, t=3)
isclose(0.1116, A, question="Q4.16")

# + [markdown] id="INvNvhLP9vGn"
# SOA Question 4.17:  (A) 1126.7
# - find future lifetime with 50\% survival probability
# - compute EPV of special whole life as sum of term and deferred insurance, that have different benefit amounts before and after median lifetime.

# + colab={"base_uri": "https://localhost:8080/"} id="5IzEgZjD9vGn" outputId="2b7afa40-e195-41ae-9a28-a4f4463ec285" vscode={"languageId": "python"}
sult = SULT()
median = sult.Z_t(48, prob=0.5, discrete=False)
def benefit(x,t): return 5000 if t < median else 10000
A = sult.A_x(48, benefit=benefit)
isclose(1130, A, question="Q4.17")

# + [markdown] id="14_c7W0X9vGn"
# SOA Question 4.18  (A) 81873 
# - find values of limits such that integral of lifetime density function equals required survival probability
#

# + colab={"base_uri": "https://localhost:8080/", "height": 314} id="mV-7K5th9vGn" outputId="0abe02dc-a935-48d2-90af-eebe3be7211f" vscode={"languageId": "python"}
def f(x,s,t): return 0.1 if t < 2 else 0.4*t**(-2)
life = Insurance().set_interest(delta=0.05)\
                  .set_survival(f=f, maxage=10)
def benefit(x,t): return 0 if t < 2 else 100000
prob = 0.9 - life.q_x(0, t=2)
T = life.Z_t(0, prob=prob)
Z = life.Z_from_t(T) * benefit(0, T)
isclose(81873, Z, question="Q4.18")

# + [markdown] id="7XZ1FdCx9vGo"
# SOA Question 4.19:  (B) 59050
# - calculate adjusted mortality for the one-year select period
# - compute whole life insurance using backward recursion formula
#

# + colab={"base_uri": "https://localhost:8080/"} id="EqjNsyIk9vGo" outputId="053f9219-729d-47f0-b462-fc26c3f0d420" vscode={"languageId": "python"}
life = SULT()
q = ExtraRisk(life=life, extra=0.8, risk="MULTIPLY_RATE")['q']
select = SelectLife(periods=1).set_select(s=0, age_selected=True, q=q)\
                              .set_select(s=1, age_selected=False, q=life['q'])\
                              .set_interest(i=.05)\
                              .fill_table()
A = 100000 * select.whole_life_insurance(80, s=0)
isclose(59050, A, question="Q4.19")

# + [markdown] id="EfWl21389vGo"
# ## 5 Annuities

# + [markdown] id="wanGyzF39vGo"
# SOA Question 5.1: (A) 0.705
# - sum of annuity certain and deferred life annuity with constant force of mortality shortcut
# - use equation for PV annuity r.v. Y to infer lifetime
# - compute survival probability from constant force of mortality function.
#

# + colab={"base_uri": "https://localhost:8080/"} id="DDhYQPYN9vGo" outputId="ff9962e7-a9e5-4cbe-f483-7e7b680807fa" vscode={"languageId": "python"}
life = ConstantForce(mu=0.01).set_interest(delta=0.06)
EY = life.certain_life_annuity(0, u=10, discrete=False)
p = life.p_x(0, t=life.Y_to_t(EY))
isclose(0.705, p, question="Q5.1")  # 0.705

# + [markdown] id="NhB04ckb9vGo"
# SOA Question 5.2:  (B) 9.64
# - compute term life as difference of whole life and deferred insurance
# - compute twin annuity-due, and adjust to an immediate annuity. 

# + colab={"base_uri": "https://localhost:8080/"} id="a1eZLQRA9vGp" outputId="aa0376b0-ca73-4c8b-873a-a7ec900c1365" vscode={"languageId": "python"}
x, n = 0, 10
a = Recursion().set_interest(i=0.05)\
               .set_A(0.3, x)\
               .set_A(0.4, x+n)\
               .set_E(0.35, x, t=n)\
               .immediate_annuity(x, t=n)
isclose(9.64, a, question="Q5.2")

# + [markdown] id="1bajTwJZ9vGp"
# SOA Question 5.3:  (C) 6.239
# - Differential reduces to the the EPV of the benefit payment at the upper time limit.
#

# + colab={"base_uri": "https://localhost:8080/"} id="9gZ6XJt_9vGp" outputId="ad4c1f19-70fa-4dcb-dc12-ea3460726e9f" vscode={"languageId": "python"}
t = 10.5
E = t * SULT().E_r(40, t=t)
isclose(6.239, E, question="Q5.3")

# + [markdown] id="4UlYXhJv9vGp"
# SOA Question 5.4:  (A) 213.7
# - compute certain and life annuity factor as the sum of a certain annuity and a deferred life annuity.
# - solve for amount of annual benefit that equals given EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="CfXai-Ra9vGp" outputId="18325841-5f2a-4fa2-abd6-4aa0b4ba091b" vscode={"languageId": "python"}
life = ConstantForce(mu=0.02).set_interest(delta=0.01)
u = life.e_x(40, curtate=False)
P = 10000 / life.certain_life_annuity(40, u=u, discrete=False)
isclose(213.7, P, question="Q5.4") # 213.7

# + [markdown] id="OeL1Jvl79vGp"
# SOA Question 5.5: (A) 1699.6
# - adjust mortality rate for the extra risk
# - compute annuity by backward recursion.
#

# + colab={"base_uri": "https://localhost:8080/"} id="s0KqIsfs9vGp" outputId="8a79f52e-8133-4e1d-d967-627aae7c6530" vscode={"languageId": "python"}
life = SULT()   # start with SULT life table
q = ExtraRisk(life=life, extra=0.05, risk="ADD_FORCE")['q']
select = SelectLife(periods=1).set_select(s=0, age_selected=True, q=q)\
                              .set_select(s=1, age_selected=False, a=life['a'])\
                              .set_interest(i=0.05)\
                              .fill_table()
a = 100 * select['a'][45][0]
isclose(1700, a, question="Q5.5")

# + [markdown] id="PY9BK9NQ9vGq"
# SOA Question 5.6:  (D) 1200
# - compute mean and variance of EPV of whole life annuity from whole life insurance twin and variance identities. 
# - portfolio percentile of the sum of $N=100$ life annuity payments

# + colab={"base_uri": "https://localhost:8080/"} id="p5mvdVAg9vGq" outputId="909f5e0e-a5ee-4a76-b470-233415eb6626" vscode={"languageId": "python"}
life = Annuity().set_interest(i=0.05)
var = life.annuity_variance(A2=0.22, A1=0.45)
mean = life.annuity_twin(A=0.45)
fund = life.portfolio_percentile(mean, var, prob=.95, N=100)
isclose(1200, fund, question="Q5.6")

# + [markdown] id="uG4P5eWY9vGq"
# SOA Question 5.7:  (C) 
# - compute endowment insurance from relationships of whole life, temporary and deferred insurances.
# - compute temporary annuity from insurance twin
# - apply Woolhouse approximation

# + colab={"base_uri": "https://localhost:8080/"} id="NEo5b0uY9vGq" outputId="dff2ba1a-80ed-488d-bc86-32bcfae9bc16" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.04)\
                  .set_A(0.188, x=35)\
                  .set_A(0.498, x=65)\
                  .set_p(0.883, x=35, t=30)
mthly = Woolhouse(m=2, life=life, three_term=False)
a = 1000 * mthly.temporary_annuity(35, t=30)
isclose(17376.7, a, question="Q5.7")

# + [markdown] id="ZasxaEEQ9vGr"
# SOA Question 5.8: (C) 0.92118
# - calculate EPV of certain and life annuity.
# - find survival probability of lifetime s.t. sum of annual payments exceeds EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="eoJrcxfS9vGr" outputId="f42d8fea-2f24-4756-8bbb-d592e0c2231a" vscode={"languageId": "python"}
sult = SULT()
a = sult.certain_life_annuity(55, u=5)
p = sult.p_x(55, t=math.floor(a))
isclose(0.92118, p, question="Q5.8")

# + [markdown] id="Di8lim4M9vGr"
# SOA Question 5.9:  (C) 0.015
# - express both EPV's expressed as forward recursions
# - solve for unknown constant $k$.
#

# + colab={"base_uri": "https://localhost:8080/"} id="8tOnpRj-9vGr" outputId="d37bf7fe-987c-460d-809c-0e8c62415046" vscode={"languageId": "python"}
x, p = 0, 0.9  # set arbitrary p_x = 0.9
a = Recursion().set_a(21.854, x=x)\
               .set_p(p, x=x)\
               .whole_life_annuity(x+1)
life = Recursion(verbose=False).set_a(22.167, x=x)
def fun(k): return a - life.set_p((1 + k) * p, x=x).whole_life_annuity(x + 1)
k = life.solve(fun, target=0, grid=[0.005, 0.025])
isclose(0.015, k, question="Q5.9")

# + [markdown] id="XyxNk-Ai9vGr"
# ## 6 Premium Calculation

# + [markdown] id="MMk89pUe9vGr"
# SOA Question 6.1: (D) 35.36
# - calculate IA factor for return of premiums without interest
# - solve net premium such that EPV benefits = EPV premium

# + colab={"base_uri": "https://localhost:8080/"} id="sPt-YT1N9vGs" outputId="34065885-fa3f-4a25-cd0f-68629af96f7c" vscode={"languageId": "python"}
P = SULT().set_interest(i=0.03)\
          .net_premium(80, t=2, b=1000, return_premium=True)
isclose(35.36, P, question="Q6.1")

# + [markdown] id="RKxWJOZ99vGs"
# SOA Question 6.2: (E) 3604
# - EPV return of premiums without interest = Premium $\times$ IA factor
# - solve for gross premiums such that EPV premiums = EPV benefits and expenses

# + colab={"base_uri": "https://localhost:8080/"} id="XpBG7-PW9vGs" outputId="3c1634bc-f3a3-4679-e95a-33406865c513" vscode={"languageId": "python"}
life = Premiums()
A, IA, a = 0.17094, 0.96728, 6.8865
P = life.gross_premium(a=a, A=A, IA=IA, benefit=100000,
                       initial_premium=0.5, renewal_premium=.05,
                       renewal_policy=200, initial_policy=200)
isclose(3604, P, question="Q6.2")

# + [markdown] id="Rb2mjaAR9vGs"
# SOA Question 6.3:  (C) 0.390
# - solve lifetime $t$ such that PV annuity certain = PV whole life annuity at age 65
# - calculate mortality rate through the year before curtate lifetime   
#

# + colab={"base_uri": "https://localhost:8080/"} id="zLIlMfiE9vGs" outputId="efc84152-c288-49da-f6ef-65d1bf733cc7" vscode={"languageId": "python"}
life = SULT()
t = life.Y_to_t(life.whole_life_annuity(65))
q = 1 - life.p_x(65, t=math.floor(t) - 1)
isclose(0.39, q, question="Q6.3")

# + [markdown] id="E9HlINxZ9vGt"
# SOA Question 6.4:  (E) 1890
#

# + colab={"base_uri": "https://localhost:8080/"} id="5XQBHhMb9vGt" outputId="afe20274-9b92-4be1-ed01-034cbf786f7e" vscode={"languageId": "python"}
mthly = Mthly(m=12, life=Annuity().set_interest(i=0.06))
A1, A2 = 0.4075, 0.2105
mean = mthly.annuity_twin(A1) * 15 * 12
var = mthly.annuity_variance(A1=A1, A2=A2, b=15 * 12)
S = Annuity.portfolio_percentile(mean=mean, variance=var, prob=.9, N=200) / 200
isclose(1890, S, question="Q6.4")

# + [markdown] id="nD3_eiZI9vGt"
# SOA Question 6.5:  (D) 33
#

# + colab={"base_uri": "https://localhost:8080/"} id="1sNfYIO79vGt" outputId="776183c8-5c99-4eba-ed6f-fe5627bbfce1" vscode={"languageId": "python"}
life = SULT()
P = life.net_premium(30, b=1000)
def gain(k): return life.Y_x(30, t=k) * P - life.Z_x(30, t=k) * 1000
k = min([k for k in range(20, 40) if gain(k) < 0])
isclose(33, k, question="Q6.5")

# + [markdown] id="7hH6SXGZ9vGu"
# SOA Question 6.6:  (B) 0.79
#

# + colab={"base_uri": "https://localhost:8080/"} id="oDi-k3c69vGu" outputId="51ef2d50-b251-45fc-f953-988a9adfb17f" vscode={"languageId": "python"}
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

# + [markdown] id="jTVCT7529vGu"
# SOA Question 6.7:  (C) 2880
#

# + colab={"base_uri": "https://localhost:8080/"} id="VsCbviTa9vGu" outputId="a05f7a76-cb47-4cd7-fba7-6e8a1bc24ea3" vscode={"languageId": "python"}
life = SULT()
a = life.temporary_annuity(40, t=20) 
A = life.E_x(40, t=20)
IA = a - life.interest.annuity(t=20) * life.p_x(40, t=20)
G = life.gross_premium(a=a, A=A, IA=IA, benefit=100000)
isclose(2880, G, question="Q6.7")

# + [markdown] id="iKUNQKe69vGu"
# SOA Question 6.8:  (B) 9.5
#
# - calculate EPV of expenses as deferred life annuities
# - solve for level premium
#

# + colab={"base_uri": "https://localhost:8080/"} id="tlQ3auqs9vGu" outputId="eebb05b4-f5cc-4d30-e013-0605b4d5393f" vscode={"languageId": "python"}
life = SULT()
initial_cost = (50 + 10 * life.deferred_annuity(60, u=1, t=9)
                + 5 * life.deferred_annuity(60, u=10, t=10))
P = life.net_premium(60, initial_cost=initial_cost)
isclose(9.5, P, question="Q6.8")

# + [markdown] id="8JlSpUV89vGv"
# SOA Question 6.9:  (D) 647
#

# + colab={"base_uri": "https://localhost:8080/"} id="LI4rbm-a9vGv" outputId="c6229f5d-238b-4919-b039-242ef4ebdeae" vscode={"languageId": "python"}
life = SULT()
a = life.temporary_annuity(50, t=10)
A = life.term_insurance(50, t=20)
initial_cost = 25 * life.deferred_annuity(50, u=10, t=10)
P = life.gross_premium(a=a, A=A, benefit=100000,
                       initial_premium=0.42, renewal_premium=0.12,
                       initial_policy=75 + initial_cost, renewal_policy=25)
isclose(647, P, question="Q6.9")

# + [markdown] id="wC--Jm4z9vGv"
# SOA Question 6.10:  (D) 0.91
#

# + colab={"base_uri": "https://localhost:8080/"} id="Q_CLIxYw9vGv" outputId="56b7d524-1719-4bbc-e619-6a7a327e4cd0" vscode={"languageId": "python"}
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

# + [markdown] id="Lle5ExS79vGv"
# SOA Question 6.11:  (C) 0.041
#

# + colab={"base_uri": "https://localhost:8080/"} id="_MkpvlkU9vGv" outputId="39922f59-657c-4e4e-c884-0b25f06e03d5" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.04)
A = life.set_A(0.39788, 51)\
        .set_q(0.0048, 50)\
        .whole_life_insurance(50)
P = life.gross_premium(A=A, a=life.annuity_twin(A=A))
A = life.set_q(0.048, 50).whole_life_insurance(50)
loss = A - life.annuity_twin(A) * P
isclose(0.041, loss, question="Q6.11")

# + [markdown] id="NdKUGnPz9vGw"
# SOA Question 6.12:  (E) 88900
#

# + colab={"base_uri": "https://localhost:8080/"} id="8BvYef7L9vGw" outputId="2b6e3344-2dab-4b57-cab6-bb7b99ad3ecf" vscode={"languageId": "python"}
life = PolicyValues().set_interest(i=0.06)
a = 12
A = life.insurance_twin(a)
contract = Contract(benefit=1000, settlement_policy=20,
                        initial_policy=10, initial_premium=0.75, 
                        renewal_policy=2, renewal_premium=0.1)
contract.premium = life.gross_premium(A=A, a=a, **contract.premium_terms)
L = life.gross_variance_loss(A1=A, A2=0.14, contract=contract)
isclose(88900, L, question="Q6.12")

# + [markdown] id="icuPfe0I9vGw"
# SOA Question 6.13:  (D) -400
#

# + colab={"base_uri": "https://localhost:8080/"} id="DOSmNBdZ9vGw" outputId="188263d8-d084-4220-8754-70bf4d82453d" vscode={"languageId": "python"}
life = SULT().set_interest(i=0.05)
A = life.whole_life_insurance(45)
contract = Contract(benefit=10000, initial_premium=.8, renewal_premium=.1)
def fun(P):   # Solve for premium, given Loss(t=0) = 4953
    return life.L_from_t(t=10.5, contract=contract.set_contract(premium=P))
contract.set_contract(premium=life.solve(fun, target=4953, grid=100))
L = life.gross_policy_value(45, contract=contract)
life.L_plot(x=45, T=10.5, contract=contract)
isclose(-400, L, question="Q6.13")

# + [markdown] id="LDdSgirA9vGw"
# SOA Question 6.14  (D) 1150
#

# + colab={"base_uri": "https://localhost:8080/"} id="L14v2XwM9vGw" outputId="c4bf71e8-8487-49ae-ff8e-1f6663a89ce0" vscode={"languageId": "python"}
life = SULT().set_interest(i=0.05)
a = life.temporary_annuity(40, t=10) + 0.5*life.deferred_annuity(40, u=10, t=10)
A = life.whole_life_insurance(40)
P = life.gross_premium(a=a, A=A, benefit=100000)
isclose(1150, P, question="Q6.14")

# + [markdown] id="Qdk7fKTe9vGx"
# SOA Question 6.15:  (B) 1.002
#

# + colab={"base_uri": "https://localhost:8080/"} id="NQhP6eiu9vGx" outputId="2624d277-11df-4dd0-f03c-7e762436b4cc" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.05).set_a(3.4611, x=0)
A = life.insurance_twin(3.4611)
udd = UDD(m=4, life=life)
a1 = udd.whole_life_annuity(x=x)
woolhouse = Woolhouse(m=4, life=life)
a2 = woolhouse.whole_life_annuity(x=x)
P = life.gross_premium(a=a1, A=A)/life.gross_premium(a=a2, A=A)
isclose(1.002, P, question="Q6.15")

# + [markdown] id="yqA_2NZg9vGx"
# SOA Question 6.16: (A) 2408.6
#

# + colab={"base_uri": "https://localhost:8080/"} id="hJD9EitR9vGx" outputId="059909aa-5c73-4aa1-b800-d2b46cbd3bd8" vscode={"languageId": "python"}
life = Premiums().set_interest(d=0.05)
A = life.insurance_equivalence(premium=2143, b=100000)
a = life.annuity_equivalence(premium=2143, b=100000)
p = life.gross_premium(A=A, a=a, benefit=100000, settlement_policy=0,
                       initial_policy=250, initial_premium=0.04 + 0.35,
                       renewal_policy=50, renewal_premium=0.04 + 0.02) 
isclose(2410, p, question="Q6.16")

# + [markdown] id="37LjDzIH9vGx"
# SOA Question 6.17:  (A) -30000
#

# + colab={"base_uri": "https://localhost:8080/"} id="nG01nwmd9vGy" outputId="aa8e8771-ce2a-47c1-e8ea-f24b16026025" vscode={"languageId": "python"}
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

# + [markdown] id="QoVCw8Pa9vGy"
# SOA Question 6.18:  (D) 166400
#

# + colab={"base_uri": "https://localhost:8080/"} id="rMBOIppO9vGy" outputId="18d3b8d7-d382-45d5-8b74-2b44e940820c" vscode={"languageId": "python"}
life = SULT().set_interest(i=0.05)
def fun(P):
    A = (life.term_insurance(40, t=20, b=P)
         + life.deferred_annuity(40, u=20, b=30000))
    return life.gross_premium(a=1, A=A) - P
P = life.solve(fun, target=0, grid=[162000, 168800])
isclose(166400, P, question="Q6.18")

# + [markdown] id="ZcMqA89S9vGy"
# SOA Question 6.19:  (B) 0.033
#

# + colab={"base_uri": "https://localhost:8080/"} id="TFWmVLHq9vGy" outputId="db3a29a2-789f-468f-8724-1522278b9238" vscode={"languageId": "python"}
life = SULT()
contract = Contract(initial_policy=.2, renewal_policy=.01)
a = life.whole_life_annuity(50)
A = life.whole_life_insurance(50)
contract.premium = life.gross_premium(A=A, a=a, **contract.premium_terms)
L = life.gross_policy_variance(50, contract=contract)
isclose(0.033, L, question="Q6.19")

# + [markdown] id="00sgYJBt9vGy"
# SOA Question 6.20:  (B) 459
#

# + colab={"base_uri": "https://localhost:8080/"} id="b8EAxOB89vGz" outputId="6f208979-92ae-4a29-cd3b-f876d31b8ae5" vscode={"languageId": "python"}
life = LifeTable().set_interest(i=.04).set_table(p={75: .9, 76: .88, 77: .85})
a = life.temporary_annuity(75, t=3)
IA = life.increasing_insurance(75, t=2)
A = life.deferred_insurance(75, u=2, t=1)
def fun(P): return life.gross_premium(a=a, A=P*IA + A*10000) - P
P = life.solve(fun, target=0, grid=[449, 489])
isclose(459, P, question="Q6.20")

# + [markdown] id="jaapeEum9vGz"
# SOA Question 6.21:  (C) 100
#

# + colab={"base_uri": "https://localhost:8080/"} id="wIqhDwJ59vGz" outputId="a42bb458-c8b2-4129-d37a-d0ce1128e6cf" vscode={"languageId": "python"}
life = Recursion(verbose=False).set_interest(d=0.04)
life.set_A(0.7, x=75, t=15, endowment=1)
life.set_E(0.11, x=75, t=15)
def fun(P):
    return (P * life.temporary_annuity(75, t=15) -
            life.endowment_insurance(75, t=15, b=1000, endowment=15*float(P)))
P = life.solve(fun, target=0, grid=(80, 120))
isclose(100, P, question="Q6.21")

# + [markdown] id="OKTQ_TC89vG0"
# SOA Question 6.22:  (C) 102
#

# + colab={"base_uri": "https://localhost:8080/"} id="7aF-4oV_9vG0" outputId="93a294cc-accc-431e-a007-7de77464faca" vscode={"languageId": "python"}
life=SULT(udd=True)
a = UDD(m=12, life=life).temporary_annuity(45, t=20)
A = UDD(m=0, life=life).whole_life_insurance(45)
P = life.gross_premium(A=A, a=a, benefit=100000) / 12
isclose(102, P, question="Q6.22")

# + [markdown] id="O7cr0lF59vG0"
# SOA Question 6.23:  (D) 44.7
#

# + colab={"base_uri": "https://localhost:8080/"} id="dmOB0Lym9vG0" outputId="d212b3d4-83d8-476f-93b5-caccd46cf596" vscode={"languageId": "python"}
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



# + [markdown] id="EqkONBmS9vG1"
# SOA Question 6.24:  (E) 0.30
#

# + colab={"base_uri": "https://localhost:8080/"} id="0eRjXMcj9vG1" outputId="c38cb91b-3399-44e3-bfde-6dc4200fa7d2" vscode={"languageId": "python"}
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

# + [markdown] id="eaJoqsnZ9vG1"
# SOA Question 6.25:  (C) 12330
#

# + colab={"base_uri": "https://localhost:8080/"} id="kMLAbBRx9vG2" outputId="07bc4248-da30-4dad-a6fe-f43bac64e7ee" vscode={"languageId": "python"}
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

# + [markdown] id="zCwTCG3I9vG2"
# SOA Question 6.26  (D) 180
#

# + colab={"base_uri": "https://localhost:8080/"} id="h6vKorks9vG2" outputId="71d4cc05-97da-4a20-c6f4-b54fefa1aed5" vscode={"languageId": "python"}
life = SULT().set_interest(i=0.05)
def fun(P): 
    return P - life.net_premium(90, b=1000, initial_cost=P)
P = life.solve(fun, target=0, grid=[150, 190])
isclose(180, P, question="Q6.26")

# + [markdown] id="oRIet2xy9vG2"
# SOA Question 6.27:  (D) 10310
#

# + colab={"base_uri": "https://localhost:8080/"} id="OQYGAYM09vG3" outputId="a3d84e61-5b31-4e60-98bd-34ad5058794e" vscode={"languageId": "python"}
life = ConstantForce(mu=0.03).set_interest(delta=0.06)
x = 0
payments = (3 * life.temporary_annuity(x, t=20, discrete=False) 
            + life.deferred_annuity(x, u=20, discrete=False))
benefits = (1000000 * life.term_insurance(x, t=20, discrete=False)
            + 500000 * life.deferred_insurance(x, u=20, discrete=False))
P = benefits / payments
isclose(10310, P, question="Q6.27")

# + [markdown] id="ZJfJJdaL9vG3"
# SOA Question 6.28  (B) 36
#

# + colab={"base_uri": "https://localhost:8080/"} id="z8hMa1ai9vG3" outputId="8412fce8-9fcb-4c80-c403-db998365dbbb" vscode={"languageId": "python"}
life = SULT().set_interest(i=0.05)
a = life.temporary_annuity(40, t=5)
A = life.whole_life_insurance(40)
P = life.gross_premium(a=a, A=A, benefit=1000, 
                       initial_policy=10, renewal_premium=.05,
                       renewal_policy=5, initial_premium=.2)
isclose(36, P, question="Q6.28")

# + [markdown] id="cncqT-ZL9vG3"
# SOA Question 6.29  (B) 20.5
#

# + colab={"base_uri": "https://localhost:8080/"} id="m43vanTb9vG4" outputId="396959f7-5749-4505-e376-e174c76ae59a" vscode={"languageId": "python"}
life = Premiums().set_interest(i=0.035)
def fun(a):
    return life.gross_premium(A=life.insurance_twin(a=a), a=a, 
                              initial_policy=200, initial_premium=.5,
                              renewal_policy=50, renewal_premium=.1,
                              benefit=100000)
a = life.solve(fun, target=1770, grid=[20, 22])
isclose(20.5, a, question="Q6.29")

# + [markdown] id="iZNST4689vG4"
# SOA Question 6.30:  (A) 900
#

# + colab={"base_uri": "https://localhost:8080/"} id="flPaelSq9vG4" outputId="42ada4ef-cf82-4788-8262-0e1296c5cce3" vscode={"languageId": "python"}
life = PolicyValues().set_interest(i=0.04)
contract = Contract(premium=2.338,
                    benefit=100,
                    initial_premium=.1,
                    renewal_premium=0.05)
var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
                               A2=0.17, contract=contract)
isclose(900, var, question="Q6.30")

# + [markdown] id="c1jbfgLr9vG5"
# SOA Question 6.31:  (D) 1330
#

# + colab={"base_uri": "https://localhost:8080/"} id="qkoIZjvw9vG5" outputId="10124aed-9f2c-4d99-f20b-872f03399d02" vscode={"languageId": "python"}
life = ConstantForce(mu=0.01).set_interest(delta=0.05)
A = (life.term_insurance(35, t=35, discrete=False) 
     + life.E_x(35, t=35)*0.51791)     # A_35
P = life.premium_equivalence(A=A, b=100000, discrete=False)
isclose(1330, P, question="Q6.31")

# + [markdown] id="AioBGkV_9vG5"
# SOA Question 6.32:  (C) 550
#

# + colab={"base_uri": "https://localhost:8080/"} id="cNRyDIDq9vG5" outputId="85c85dd5-80a6-44cf-dc7c-79ff56f60d63" vscode={"languageId": "python"}
x = 0
life = Recursion().set_interest(i=0.05).set_a(9.19, x=x)
benefits = UDD(m=0, life=life).whole_life_insurance(x)
payments = UDD(m=12, life=life).whole_life_annuity(x)
P = life.gross_premium(a=payments, A=benefits, benefit=100000)/12
isclose(550, P, question="Q6.32")

# + [markdown] id="ebOOmO3r9vG6"
# SOA Question 6.33:  (B) 0.13
#

# + colab={"base_uri": "https://localhost:8080/"} id="5XzvzCyY9vG6" outputId="abebd8c3-f299-4b24-b606-92190999176a" vscode={"languageId": "python"}
life = Insurance().set_survival(mu=lambda x,t: 0.02*t).set_interest(i=0.03)
x = 0
var = life.E_x(x, t=15, moment=life._VARIANCE, endowment=10000)
p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
isclose(0.13, p, question="Q6.33", rel_tol=0.02)

# + [markdown] id="gbAdkAYQ9vG6"
# SOA Question 6.34:  (A) 23300
#

# + colab={"base_uri": "https://localhost:8080/"} id="CPcYynHS9vG6" outputId="cee17f88-0a53-453c-9782-d165e218bcd1" vscode={"languageId": "python"}
life = SULT()
def fun(benefit):
    A = life.whole_life_insurance(61)
    a = life.whole_life_annuity(61)
    return life.gross_premium(A=A, a=a, benefit=benefit, 
                              initial_premium=0.15, renewal_premium=0.03)
b = life.solve(fun, target=500, grid=[23300, 23700])
isclose(23300, b, question="Q6.34")

# + [markdown] id="NBbvFO2x9vG6"
# SOA Question 6.35:  (D) 530
#

# + colab={"base_uri": "https://localhost:8080/"} id="NOghD5vh9vG6" outputId="5858517b-c68f-4501-8452-b52123613f94" vscode={"languageId": "python"}
sult = SULT()
A = sult.whole_life_insurance(35, b=100000)
a = sult.whole_life_annuity(35)
P = sult.gross_premium(a=a, A=A, initial_premium=.19, renewal_premium=.04)
isclose(530, P, question="Q6.35")

# + [markdown] id="bQfiVt4j9vG7"
# SOA Question 6.36:  (B) 500
#

# + colab={"base_uri": "https://localhost:8080/"} id="hy4bStS19vG7" outputId="a7977e5b-09e2-4059-f4a8-00afc76cc0a6" vscode={"languageId": "python"}
life = ConstantForce(mu=0.04).set_interest(delta=0.08)
a = life.temporary_annuity(50, t=20, discrete=False)
A = life.term_insurance(50, t=20, discrete=False)
def fun(R):
    return life.gross_premium(a=a, A=A, initial_premium=R/4500,
                              renewal_premium=R/4500, benefit=100000)
R = life.solve(fun, target=4500, grid=[400, 800])
isclose(500, R, question="Q6.36")

# + [markdown] id="GrV4g-Bi9vG7"
# SOA Question 6.37:  (D) 820
#

# + colab={"base_uri": "https://localhost:8080/"} id="3i6sGOvF9vG7" outputId="07455b7c-6515-4bc8-a5cd-404bf7e4111c" vscode={"languageId": "python"}
sult = SULT()
benefits = sult.whole_life_insurance(35, b=50000 + 100)
expenses = sult.immediate_annuity(35, b=100)
a = sult.temporary_annuity(35, t=10)
P = (benefits + expenses) / a
isclose(820, P, question="Q6.37")

# + [markdown] id="b6JJiBNb9vG7"
# SOA Question 6.38:  (B) 11.3
#

# + colab={"base_uri": "https://localhost:8080/"} id="zmkkGWyj9vG7" outputId="306298ad-76e4-41df-fa89-831e92610d36" vscode={"languageId": "python"}
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

# + [markdown] id="AKm7vNe59vG7"
# SOA Question 6.39:  (A) 29
#

# + colab={"base_uri": "https://localhost:8080/"} id="dggrBcE39vG8" outputId="2a83808e-d3f5-44ac-e05f-491090e33c1e" vscode={"languageId": "python"}
sult = SULT()
P40 = sult.premium_equivalence(sult.whole_life_insurance(40), b=1000)
P80 = sult.premium_equivalence(sult.whole_life_insurance(80), b=1000)
p40 = sult.p_x(40, t=10)
p80 = sult.p_x(80, t=10)
P = (P40 * p40 + P80 * p80) / (p80 + p40)
isclose(29, P, question="Q6.39")

# + [markdown] id="9LPFohDX9vG8"
# SOA Question 6.40: (C) 116 
#

# + colab={"base_uri": "https://localhost:8080/"} id="mOMHVhhC9vG8" outputId="2afbe4ed-770d-42f8-d38f-1957de31fa30" vscode={"languageId": "python"}
# - standard formula discounts/accumulates by too much (i should be smaller)
x = 0
life = Recursion().set_interest(i=0.06).set_a(7, x=x+1).set_q(0.05, x=x)
a = life.whole_life_annuity(x)
A = 110 * a / 1000
life = Recursion().set_interest(i=0.06).set_A(A, x=x).set_q(0.05, x=x)
A1 = life.whole_life_insurance(x+1)
P = life.gross_premium(A=A1 / 1.03, a=7) * 1000
isclose(116, P, question="Q6.40")

# + [markdown] id="yHCVsRUL9vG8"
# SOA Question 6.41:  (B) 1417
#

# + colab={"base_uri": "https://localhost:8080/"} id="BxOzHHAQ9vG8" outputId="0cea690a-d09a-4ea5-d02b-43c09855489e" vscode={"languageId": "python"}
x = 0
life = LifeTable().set_interest(i=0.05).set_table(q={x:.01, x+1:.02})
a = 1 + life.E_x(x, t=1) * 1.01
A = life.deferred_insurance(x, u=0, t=1) + 1.01*life.deferred_insurance(x, u=1, t=1)
P = 100000 * A / a
isclose(1417, P, question="Q6.41")

# + [markdown] id="yuX0IRFd9vG8"
# SOA Question 6.42:  (D) 0.113
#

# + colab={"base_uri": "https://localhost:8080/"} id="K5COKbI89vG8" outputId="3fcd936e-3fc4-4651-f656-a66c1d7cbd11" vscode={"languageId": "python"}
x = 0
life = ConstantForce(mu=0.06).set_interest(delta=0.06)
contract = Contract(discrete=True, premium=315.8, 
                    T=3, endowment=1000, benefit=1000)
L = [life.L_from_t(t, contract=contract) for t in range(3)]    # L(t)
Q = [life.q_x(x, u=u, t=1) for u in range(3)]              # prob(die in year t)
Q[-1] = 1 - sum(Q[:-1])   # follows SOA Solution: incorrectly treats endowment!
p = sum([q for (q, l) in zip (Q, L) if l > 0])
isclose(0.113, p, question="Q6.42")

# + [markdown] id="nil8k4en9vG9"
# SOA Question 6.43:  (C) 170
# - although 10-year term, premiums only paid first first years: separately calculate the EPV of per-policy maintenance expenses in years 6-10 and treat as additional initial expense

# + colab={"base_uri": "https://localhost:8080/"} id="PVLeppAQ9vG9" outputId="d7fd2ab4-f099-445b-c7e0-5ade3948f8de" vscode={"languageId": "python"}
sult = SULT()
a = sult.temporary_annuity(30, t=5)
A = sult.term_insurance(30, t=10)
other_expenses = 4 * sult.deferred_annuity(30, u=5, t=5)
P = sult.gross_premium(a=a, A=A, benefit=200000, initial_premium=0.35,
                       initial_policy=8 + other_expenses, renewal_policy=4,
                       renewal_premium=0.15)
isclose(170, P, question="Q6.43")

# + [markdown] id="QpZvt3Oi9vG9"
# SOA Question 6.44:  (D) 2.18
#

# + colab={"base_uri": "https://localhost:8080/"} id="WEVs1_u59vG9" outputId="dad29bb8-ee6a-4f7f-d6d0-6f6b376ad0ad" vscode={"languageId": "python"}
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

# + [markdown] id="SzfeMzZf9vG9"
# SOA Question 6.45:  (E) 690
#

# + colab={"base_uri": "https://localhost:8080/"} id="oY2QEtEq9vG9" outputId="faa5aaaf-729b-4a33-b7c1-c9d738be255c" vscode={"languageId": "python"}
life = SULT(udd=True)
contract = Contract(benefit=100000, premium=560, discrete=False)
L = life.L_from_prob(x=35, prob=0.75, contract=contract)
life.L_plot(x=35, contract=contract, 
            T=life.L_to_t(L=L, contract=contract))
isclose(690, L, question="Q6.45")

# + [markdown] id="XNAOezVA9vG9"
# SOA Question 6.46:  (E) 208
#

# + colab={"base_uri": "https://localhost:8080/"} id="nHX7OoYc9vG-" outputId="3ce41d25-18bd-4535-e366-03a4ca327e97" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.05)\
                  .set_IA(0.51213, x=55, t=10)\
                  .set_a(12.2758, x=55)\
                  .set_a(7.4575, x=55, t=10)
A = life.deferred_annuity(55, u=10)
IA = life.increasing_insurance(55, t=10)
a = life.temporary_annuity(55, t=10)
P = life.gross_premium(a=a, A=A, IA=IA, benefit=300)
isclose(208, P, question="Q6.46")

# + [markdown] id="6T1ptC7F9vG-"
# SOA Question 6.47:  (D) 66400
#

# + colab={"base_uri": "https://localhost:8080/"} id="t4yWKlaN9vG-" outputId="fbc32f31-6fd3-42b2-b27a-686d12d0b64d" vscode={"languageId": "python"}
sult = SULT()
a = sult.temporary_annuity(70, t=10)
A = sult.deferred_annuity(70, u=10)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
                        renewal_premium=0.05)
isclose(66400, P, question="Q6.47")

# + [markdown] id="eGst1Fhk9vG-"
# SOA Question 6.48:  (A) 3195 -- example of deep insurance recursion
#

# + colab={"base_uri": "https://localhost:8080/"} id="-tKN7-LZ9vG-" outputId="8d75d5dd-5ef1-4ac6-a8cd-5fe695303f55" vscode={"languageId": "python"}
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

# + [markdown] id="1RhvEUTP9vG-"
# SOA Question 6.49:  (C) 86
#

# + colab={"base_uri": "https://localhost:8080/"} id="Qeoh1oRG9vG-" outputId="b4d0b0a4-2791-4fbe-d7af-606e11c01019" vscode={"languageId": "python"}
sult = SULT(udd=True)
a = UDD(m=12, life=sult).temporary_annuity(40, t=20)
A = sult.whole_life_insurance(40, discrete=False)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_policy=200,
                       renewal_premium=0.04, initial_premium=0.04) / 12
isclose(86, P, question="Q6.49")

# + [markdown] id="8DVvN3Id9vG_"
# SOA Question 6.50:  (A) -47000
#

# + colab={"base_uri": "https://localhost:8080/"} id="7KAGpNAy9vG_" outputId="00db8358-ba34-4d74-f569-ad611a722c98" vscode={"languageId": "python"}
life = SULT()
P = life.premium_equivalence(a=life.whole_life_annuity(35), b=1000) 
a = life.deferred_annuity(35, u=1, t=1)
A = life.term_insurance(35, t=1, b=1000)
cash = (A - a * P) * 10000 / life.interest.v
isclose(-47000, cash, question="Q6.50")

# + [markdown] id="E6bZCbjh9vG_"
# SOA Question 6.51:  (D) 34700
#

# + colab={"base_uri": "https://localhost:8080/"} id="Dm5CUxWk9vG_" outputId="e50ffe52-aa47-4add-da2a-597ce716a86e" vscode={"languageId": "python"}
life = Recursion().set_DA(0.4891, x=62, t=10)\
                   .set_A(0.0910, x=62, t=10)\
                   .set_a(12.2758, x=62)\
                   .set_a(7.4574, x=62, t=10)
IA = life.increasing_insurance(62, t=10)
A = life.deferred_annuity(62, u=10)
a = life.temporary_annuity(62, t=10)
P = life.gross_premium(a=a, A=A, IA=IA, benefit=50000)
isclose(34700, P, question="Q6.51")

# + [markdown] id="ynsTsjDv9vG_"
# SOA Question 6.52:  (D) 50.80
#
# - set face value benefits to 0
#

# + colab={"base_uri": "https://localhost:8080/"} id="1kFZWjmX9vG_" outputId="cd014cf1-b0bb-46bd-9d33-51371ab79fa0" vscode={"languageId": "python"}
sult = SULT()
a = sult.temporary_annuity(45, t=10)
other_cost = 10 * sult.deferred_annuity(45, u=10)
P = sult.gross_premium(a=a, A=0, benefit=0,    # set face value H = 0
                       initial_premium=1.05, renewal_premium=0.05,
                       initial_policy=100 + other_cost, renewal_policy=20)
isclose(50.8, P, question="Q6.52")

# + [markdown] id="xdzRRvLM9vG_"
# SOA Question 6.53:  (D) 720
#

# + colab={"base_uri": "https://localhost:8080/"} id="6j4PzepV9vG_" outputId="0af60136-bc12-4d93-c4cb-2468a509bdd9" vscode={"languageId": "python"}
x = 0
life = LifeTable().set_interest(i=0.08).set_table(q={x:.1, x+1:.1, x+2:.1})
A = life.term_insurance(x, t=3)
P = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
isclose(720, P, question="Q6.53")

# + [markdown] id="_b71u50p9vG_"
# SOA Question 6.54:  (A) 25440
#

# + colab={"base_uri": "https://localhost:8080/"} id="V1bKVxcB9vHA" outputId="2f2afa6b-9973-4557-eed0-572f1baa06b1" vscode={"languageId": "python"}
life = SULT()
std = math.sqrt(life.net_policy_variance(45, b=200000))
isclose(25440, std, question="Q6.54")

# + [markdown] id="FcLQHamA9vHA"
# ## 7 Policy Values

# + [markdown] id="NUe-Xt7I9vHA"
# SOA Question 7.1:  (C) 11150
#

# + colab={"base_uri": "https://localhost:8080/"} id="vMvwBXiR9vHA" outputId="9d92d13f-b1ef-4956-c31b-4489d3b23b15" vscode={"languageId": "python"}
life = SULT()
x, n, t = 40, 20, 10
A = (life.whole_life_insurance(x+t, b=50000)
     + life.deferred_insurance(x+t, u=n-t, b=50000))
a = life.temporary_annuity(x+t, t=n-t, b=875)
L = life.gross_future_loss(A=A, a=a)
isclose(11150, L, question="Q7.1")

# + [markdown] id="4e3PnDH69vHA"
# SOA Question 7.2:  (C) 1152
#

# + colab={"base_uri": "https://localhost:8080/"} id="qsjRTV-39vHA" outputId="26228983-0417-4827-e2e5-999fcf5e5e40" vscode={"languageId": "python"}
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

# + [markdown] id="g1HwU2Tc9vHA"
# SOA Question 7.3:  (E) 730
#

# + colab={"base_uri": "https://localhost:8080/"} id="MTRFj4rH9vHA" outputId="3ddc8993-8863-45ef-dd5a-5555ef67c050" vscode={"languageId": "python"}
x = 0  # x=0 is (90) and interpret every 3 months as t=1 year
life = LifeTable().set_interest(i=0.08/4)\
                  .set_table(l={0:1000, 1:898, 2:800, 3:706})\
                  .set_reserves(T=8, V={3: 753.72})
V = life.t_V_backward(x=0, t=2, premium=60*0.9, benefit=lambda t: 1000)
V = life.set_reserves(V={2: V})\
        .t_V_backward(x=0, t=1, premium=0, benefit=lambda t: 1000)
isclose(730, V, question="Q7.3")

# + [markdown] id="1P6MEnku9vHA"
# SOA Question 7.4:  (B) -74 -- split benefits into two policies
#

# + colab={"base_uri": "https://localhost:8080/"} id="zSN7PaJQ9vHB" outputId="b67feec0-e542-4e06-e14e-d1168efe8bb7" vscode={"languageId": "python"}
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

# + [markdown] id="z7MDI1zR9vHB"
# SOA Question 7.5:  (E) 1900
#

# + colab={"base_uri": "https://localhost:8080/"} id="6J4ZHYSP9vHB" outputId="520ad047-bca1-4ae6-8b1f-a008130f5f1c" vscode={"languageId": "python"}
x = 0
life = Recursion(udd=True).set_interest(i=0.03)\
                          .set_q(0.04561, x=x+4)\
                          .set_reserves(T=3, V={4: 1405.08})
V = life.r_V_forward(x, s=4, r=0.5, benefit=10000, premium=647.46)
isclose(1900, V, question="Q7.5")

# + [markdown] id="xbkufOv79vHB"
# Answer 7.6:  (E) -25.4
#

# + colab={"base_uri": "https://localhost:8080/"} id="cYj-b4fa9vHB" outputId="3e358891-4a08-4ba8-c531-9a04df50437b" vscode={"languageId": "python"}
life = SULT()
P = life.net_premium(45, b=2000)
contract = Contract(benefit=2000, initial_premium=.25, renewal_premium=.05,
                    initial_policy=2*1.5 + 30, renewal_policy=2*.5 + 10)
G = life.gross_premium(a=life.whole_life_annuity(45), **contract.premium_terms)
gross = life.gross_policy_value(45, t=10, contract=contract.set_contract(premium=G))
net = life.net_policy_value(45, t=10, b=2000)
V = gross - net
isclose(-25.4, V, question="Q7.6")    

# + [markdown] id="Lzkdk8h_9vHB"
# SOA Question 7.7:  (D) 1110
#

# + colab={"base_uri": "https://localhost:8080/"} id="8DWmpEO-9vHC" outputId="68910e36-ee4b-41dd-e3d0-58488fcdcb23" vscode={"languageId": "python"}
x = 0
life = Recursion().set_interest(i=0.05).set_A(0.4, x=x+10)
a = Woolhouse(m=12, life=life).whole_life_annuity(x+10)
contract = Contract(premium=0, benefit=10000, renewal_policy=100)
V = life.gross_future_loss(A=0.4, contract=contract.renewals())
contract = Contract(premium=30*12, renewal_premium=0.05)
V += life.gross_future_loss(a=a, contract=contract.renewals())
isclose(1110, V, question="Q7.7")

# + [markdown] id="3p_yFwql9vHC"
# SOA Question 7.8:  (C) 29.85
#

# + colab={"base_uri": "https://localhost:8080/"} id="XDyGtjjM9vHC" outputId="237da86b-1095-4454-d21d-865db12fbdaa" vscode={"languageId": "python"}
sult = SULT()
x = 70
q = {x: [sult.q_x(x+k)*(.7 + .1*k) for k in range(3)] + [sult.q_x(x+3)]}
life = Recursion().set_interest(i=.05)\
                  .set_q(sult.q_x(70)*.7, x=x)\
                  .set_reserves(T=3)
V = life.t_V(x=70, t=1, premium=35.168, benefit=lambda t: 1000)
isclose(29.85, V, question="Q7.8")

# + [markdown] id="lqQbeLzb9vHC"
# SOA Question 7.9:  (A) 38100
#

# + colab={"base_uri": "https://localhost:8080/"} id="TGBvaPng9vHC" outputId="7cfeb3a9-90b6-45d7-a20e-b76671c65459" vscode={"languageId": "python"}
sult = SULT(udd=True)
x, n, t = 45, 20, 10
a = UDD(m=12, life=sult).temporary_annuity(x+10, t=n-10)
A = UDD(m=0, life=sult).endowment_insurance(x+10, t=n-10)
contract = Contract(premium=253*12, endowment=100000, benefit=100000)
V = sult.gross_future_loss(A=A, a=a, contract=contract)
isclose(38100, V, question="Q7.9")

# + [markdown] id="7Jsgepay9vHC"
# SOA Question 7.10: (C) -970
#

# + colab={"base_uri": "https://localhost:8080/"} id="F4vYPfeB9vHC" outputId="aa975cd0-6f50-4f07-b1f5-17f56337be4f" vscode={"languageId": "python"}
life = SULT()
G = 977.6
P = life.net_premium(45, b=100000)
contract = Contract(benefit=0, premium=G-P, renewal_policy=.02*G + 50)
V = life.gross_policy_value(45, t=5, contract=contract)
isclose(-970, V, question="Q7.10")

# + [markdown] id="bOqwHr8s9vHC"
# SOA Question 7.11:  (B) 1460
#

# + colab={"base_uri": "https://localhost:8080/"} id="Z7uS5tEJ9vHD" outputId="392f97b8-d513-46d3-a619-fe9f6871202b" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.05).set_a(13.4205, x=55)
contract = Contract(benefit=10000)
def fun(P):
    return life.L_from_t(t=10, contract=contract.set_contract(premium=P))
P = life.solve(fun, target=4450, grid=400)
V = life.gross_policy_value(45, t=10, contract=contract.set_contract(premium=P))
isclose(1460, V, question="Q7.11")

# + [markdown] id="Np_rYr_f9vHD"
# SOA Question 7.12:  (E) 4.09
#

# + colab={"base_uri": "https://localhost:8080/"} id="6HkiLWWl9vHD" outputId="35eefda9-da8f-4151-a77f-8d9d16e0334b" vscode={"languageId": "python"}
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


# + [markdown] id="CFY9zV8a9vHD"
# Answer 7.13: (A) 180
#

# + colab={"base_uri": "https://localhost:8080/"} id="jmjMCJiW9vHD" outputId="d5bb3025-d82c-49b6-d5ac-3e43387ddc28" vscode={"languageId": "python"}
life = SULT()
V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
isclose(180, V, question="Q7.13")

# + [markdown] id="XbZZpb2j9vHE"
# SOA Question 7.14:  (A) 2200
#

# + colab={"base_uri": "https://localhost:8080/"} id="3iG68-Fj9vHE" outputId="bf211f10-dbcd-4351-f4ea-9686dd807fe5" vscode={"languageId": "python"}
x = 45
life = Recursion(verbose=False).set_interest(i=0.05)\
                               .set_q(0.009, x=50)\
                               .set_reserves(T=10, V={5: 5500})
def fun(P):  # solve for net premium,
    return life.t_V(x=x, t=6, premium=P*0.96 - 50, benefit=lambda t: 100000+200)
P = life.solve(fun, target=7100, grid=[2200, 2400])
isclose(2200, P, question="Q7.14")

# + [markdown] id="bo9wPVQ69vHE"
# SOA Question 7.15:  (E) 50.91
#

# + colab={"base_uri": "https://localhost:8080/"} id="d-yrZ69M9vHE" outputId="80a66cd1-339b-4e99-bf16-7d72d7695279" vscode={"languageId": "python"}
x = 0
V = Recursion(udd=True).set_interest(i=0.05)\
                       .set_q(0.1, x=x+15)\
                       .set_reserves(T=3, V={16: 49.78})\
                       .r_V_backward(x, s=15, r=0.6, benefit=100)
isclose(50.91, V, question="Q7.15")

# + [markdown] id="SYc6EWHD9vHE"
# SOA Question 7.16:  (D) 380
#

# + colab={"base_uri": "https://localhost:8080/"} id="_VmNfOCZ9vHE" outputId="9c0abb7f-5319-4b43-88e4-14ab923567db" vscode={"languageId": "python"}
life = SelectLife().set_interest(v=.95)\
                   .set_table(A={86: [683/1000]},
                              q={80+k: [.01*(k+1)] for k in range(6)})
x, t, n = 80, 3, 5
A = life.whole_life_insurance(x+t)
a = life.temporary_annuity(x+t, t=n-t)
V = life.gross_future_loss(A=A, a=a, contract=Contract(benefit=1000, premium=130))
isclose(380, V, question="Q7.16")

# + [markdown] id="_gQKarnK9vHE"
# SOA Question 7.17:  (D) 1.018
#

# + colab={"base_uri": "https://localhost:8080/"} id="g9TH_wKZ9vHE" outputId="08bfc646-273c-409e-af08-d0460426f596" vscode={"languageId": "python"}
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

# + [markdown] id="K_HFgCFi9vHF"
# SOA Question 7.18:  (A) 17.1
#

# + colab={"base_uri": "https://localhost:8080/"} id="e6E0EVQS9vHF" outputId="3bd66ac7-874c-4505-a7f9-a858ed87d57a" vscode={"languageId": "python"}
x = 10
life = Recursion(verbose=False).set_interest(i=0.04).set_q(0.009, x=x)
def fun(a):
    return life.set_a(a, x=x).net_policy_value(x, t=1)
a = life.solve(fun, target=0.012, grid=[17.1, 19.1])
isclose(17.1, a, question="Q7.18")

# + [markdown] id="gpYYsKR-9vHF"
# SOA Question 7.19:  (D) 720
#

# + colab={"base_uri": "https://localhost:8080/"} id="lJqmu8K_9vHF" outputId="6202fe14-5cb5-41fa-9f9b-56fe939a7df6" vscode={"languageId": "python"}
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

# + [markdown] id="NAudCSbC9vHF"
# SOA Question 7.20: (E) -277.23
#

# + colab={"base_uri": "https://localhost:8080/"} id="6C_Fv2nY9vHF" outputId="255df0bf-a5c6-4a98-a01f-0f67fe9c0e17" vscode={"languageId": "python"}
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

# + [markdown] id="w1-8wjt39vHF"
# SOA Question 7.21:  (D) 11866
#

# + colab={"base_uri": "https://localhost:8080/"} id="9pd77ATV9vHG" outputId="fbe5a5e8-9f25-4777-8aff-4e69c7afdef1" vscode={"languageId": "python"}
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

# + [markdown] id="Ryka7Xlg9vHG"
# SOA Question 7.22:  (C) 46.24
#

# + colab={"base_uri": "https://localhost:8080/"} id="ZRkFT6-y9vHG" outputId="fb097c5f-b686-4d47-b2c3-5de174c4b4c4" vscode={"languageId": "python"}
life = PolicyValues().set_interest(i=0.06)
contract = Contract(benefit=8, premium=1.250)
def fun(A2): 
    return life.gross_variance_loss(A1=0, A2=A2, contract=contract)
A2 = life.solve(fun, target=20.55, grid=20.55/8**2)
contract = Contract(benefit=12, premium=1.875)
var = life.gross_variance_loss(A1=0, A2=A2, contract=contract)
isclose(46.2, var, question="Q7.22")

# + [markdown] id="UpC304SQ9vHG"
# SOA Question 7.23:  (D) 233
#

# + colab={"base_uri": "https://localhost:8080/"} id="E_CH-AUx9vHG" outputId="7c0a5615-d260-4f31-c324-58e816edbcd3" vscode={"languageId": "python"}
life = Recursion().set_interest(i=0.04).set_p(0.995, x=25)
A = life.term_insurance(25, t=1, b=10000)
def fun(beta):  # value of premiums in first 20 years must be equal
    return beta * 11.087 + (A - beta) 
beta = life.solve(fun, target=216 * 11.087, grid=[140, 260])
isclose(233, beta, question="Q7.23")

# + [markdown] id="JRR1eV1T9vHG"
# SOA Question 7.24:  (C) 680
#

# + colab={"base_uri": "https://localhost:8080/"} id="0BoPW3Nc9vHG" outputId="253aa30e-06d5-43b6-ec99-33c278c9ec30" vscode={"languageId": "python"}
life = SULT()
P = life.premium_equivalence(A=life.whole_life_insurance(50), b=1000000)
isclose(680, 11800 - P, question="Q7.24")

# + [markdown] id="XOY94lYn9vHH"
# SOA Question 7.25:  (B) 3947.37
#

# + colab={"base_uri": "https://localhost:8080/"} id="YaigL6D19vHH" outputId="142c0a7c-93a0-43bd-ad6d-b023d6de4f21" vscode={"languageId": "python"}
life = SelectLife().set_interest(i=.04)\
                   .set_table(A={55: [.23, .24, .25],
                                 56: [.25, .26, .27],
                                 57: [.27, .28, .29],
                                 58: [.20, .30, .31]})
V = life.FPT_policy_value(55, t=3, b=100000)
isclose(3950, V, question="Q7.25")

# + [markdown] id="HNP0rbDR9vHH"
# SOA Question 7.26:  (D) 28540 
# - backward = forward reserve recursion
#

# + colab={"base_uri": "https://localhost:8080/"} id="MhM3zoMi9vHH" outputId="5481db08-af2d-49d5-9ffd-577ab622c895" vscode={"languageId": "python"}
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

# + [markdown] id="PFl85-0a9vHH"
# SOA Question 7.27:  (B) 213
#

# + colab={"base_uri": "https://localhost:8080/"} id="yCAwFPMj9vHH" outputId="816541ba-9d18-4bdd-cac2-2a67cf27f782" vscode={"languageId": "python"}
x = 0
life = Recursion(verbose=False).set_interest(i=0.03)\
                               .set_q(0.008, x=x)\
                               .set_reserves(V={0: 0})
def fun(G):  # Solve gross premium from expense reserves equation
    return life.t_V(x=x, t=1, premium=G - 187, benefit=lambda t: 0,
                    per_policy=10 + 0.25*G)
G = life.solve(fun, target=-38.70, grid=[200, 252])
isclose(213, G, question="Q7.27")

# + [markdown] id="vDjDeAqK9vHH"
# SOA Question 7.28:  (D) 24.3
#

# + colab={"base_uri": "https://localhost:8080/"} id="HCYiiACj9vHI" outputId="eca57324-5c6b-43ef-d80b-830bb2cd78b9" vscode={"languageId": "python"}
life = SULT()
PW = life.net_premium(65, b=1000)   # 20_V=0 => P+W is net premium for A_65
P = life.net_premium(45, t=20, b=1000)  # => P is net premium for A_45:20
isclose(24.3, PW - P, question="Q7.28")

# + [markdown] id="IAXcmheg9vHI"
# SOA Question 7.29:  (E) 2270
#

# + colab={"base_uri": "https://localhost:8080/"} id="OaeXFQiC9vHI" outputId="84e5db77-64aa-4a24-9a85-ded47bab1323" vscode={"languageId": "python"}
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

# + [markdown] id="-2DakbOB9vHI"
# SOA Question 7.30:  (E) 9035
#

# + colab={"base_uri": "https://localhost:8080/"} id="KPWe4zrI9vHI" outputId="82a7247c-d72d-449d-988b-cb608ee0588c" vscode={"languageId": "python"}
contract = Contract(premium=0, benefit=10000)  # premiums=0 after t=10
L = SULT().gross_policy_value(35, contract=contract)
V = SULT().set_interest(i=0).gross_policy_value(35, contract=contract) # 10000
isclose(9035, V - L, question="Q7.30")

# + [markdown] id="phC98jCG9vHI"
# SOA Question 7.31:  (E) 0.310
#

# + colab={"base_uri": "https://localhost:8080/"} id="qij4IpoX9vHJ" outputId="18ae88e2-f58d-4049-d6fd-d234ae7887e6" vscode={"languageId": "python"}
x = 0
life = Reserves().set_reserves(T=3)
G = 368.05
def fun(P):  # solve net premium expense reserve equation
    return life.t_V(x, t=2, premium=G-P, benefit=lambda t:0, per_policy=5+0.08*G)
P = life.solve(fun, target=-23.64, grid=[.29, .31]) / 1000
isclose(0.310, P, question="Q7.31")

# + [markdown] id="XCZRvoRC9vHJ"
# SOA Question 7.32:  (B) 1.4
#

# + colab={"base_uri": "https://localhost:8080/"} id="Z5eq8PF89vHM" outputId="e3127bff-ab3f-4d48-f3be-5eebe7f4e07c" vscode={"languageId": "python"}
life = PolicyValues().set_interest(i=0.06)
contract = Contract(benefit=1, premium=0.1)
def fun(A2): 
    return life.gross_variance_loss(A1=0, A2=A2, contract=contract)
A2 = life.solve(fun, target=0.455, grid=0.455)
contract = Contract(benefit=2, premium=0.16)
var = life.gross_variance_loss(A1=0, A2=A2, contract=contract)
isclose(1.39, var, question="Q7.32")

# + [markdown] id="vYHwoQiOgjV6"
# __Final Score__

# + colab={"base_uri": "https://localhost:8080/", "height": 286} id="lWMtOVQd9vHM" outputId="48943e0e-0798-4fa7-da22-0b58e7aa0267" vscode={"languageId": "python"}
from datetime import datetime
print(datetime.now())
print(isclose)
