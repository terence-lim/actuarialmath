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
# - [Online User Guide](https://terence-lim.github.io/actuarialmath-guide/), or [download pdf](https://terence-lim.github.io/notes/actuarialmath-guide.pdf)
#
# - [API reference](https://terence-lim.github.io/actuarialmath-docs)
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
from actuarialmath import Interest
from actuarialmath import Life
from actuarialmath import Survival
from actuarialmath import Lifetime
from actuarialmath import Fractional
from actuarialmath import Insurance
from actuarialmath import Annuity
from actuarialmath import Premiums
from actuarialmath import PolicyValues, Contract
from actuarialmath import Reserves
from actuarialmath import Recursion
from actuarialmath import LifeTable
from actuarialmath import SULT
from actuarialmath import SelectLife
from actuarialmath import MortalityLaws, Beta, Uniform, Makeham, Gompertz
from actuarialmath import ConstantForce
from actuarialmath import ExtraRisk
from actuarialmath import Mthly
from actuarialmath import UDD
from actuarialmath import Woolhouse

# + [markdown] id="c2b4cbf3"
# __Helper to compare computed answers to expected solutions__

# + id="903e972b"
import time
class IsClose:
    """Helper class for testing and reporting if two values are close"""
    def __init__(self, rel_tol : float = 0.01, score : bool = False,
                 verbose: bool = False):
        self.den = self.num = 0
        self.score = score      # whether to count INCORRECTs instead of assert
        self.verbose = verbose  # whether to run silently
        self.incorrect = []     # to keep list of messages for INCORRECT
        self.tol = rel_tol
        self.tic = time.time()

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
        return f"Elapsed: {time.time()-self.tic:.1f} secs\n" \
               + f"Passed:  {self.num}/{self.den}\n" + "\n".join(self.incorrect)
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
# __SOA Question 2.1__ : (B) 2.5
#
# You are given:
#
# 1. $S_0(t) = \left(1 - \frac{t}{\omega} \right)^{\frac{1}{4}}, \quad 0 \le t \le \omega$
#
# 2. $\mu_{65} = \frac{1}{180} $
#
# Calculate $e_{106}$, the curtate expectation of life at age 106.
#
# *hints:*
#
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
# __SOA Question 2.2__ : (D) 400
#
# Scientists are searching for a vaccine for a disease. You are given:
# 1. 100,000 lives age x are exposed to the disease
# 2. Future lifetimes are independent, except that the vaccine, if available, will be given to all at the end of year 1
# 3. The probability that the vaccine will be available is 0.2
# 4. For each life during year 1, $q_x$ = 0.02
# 5. For each life during year 2, $q_{x+1}$ = 0.01 if the vaccine has been given and $q_{x+1}$ = 0.02 if it has not been given
#
# Calculate the standard deviation of the number of survivors at the end of year 2.
#
# *hints:*
#
#
# - calculate survival probabilities for the two scenarios
# - apply conditional variance formula (or mixed distribution)

# + colab={"base_uri": "https://localhost:8080/"} id="1f07bbbd" outputId="16cffcd5-d289-481d-f5ce-222bc25e01e6"
p1 = (1. - 0.02) * (1. - 0.01)  # 2_p_x if vaccine given
p2 = (1. - 0.02) * (1. - 0.02)  # 2_p_x if vaccine not given
std = math.sqrt(Life.conditional_variance(p=.2, p1=p1, p2=p2, N=100000))
isclose(400, std, question="Q2.2")

# + [markdown] id="246aaf2d"
# __SOA Question 2.3__ : (A) 0.0483
#
# You are given that mortality follows Gompertz Law with B = 0.00027 and c = 1.1.
#
# Calculate $f_{50}(10)$.
#
#
# *hints:*
#
#
# - Derive formula for $f$ given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="e36facdd" outputId="ab177cfc-a9f8-469c-9cd2-ab51d27f8760"
B, c = 0.00027, 1.1
S = lambda x,s,t: math.exp(-B * c**(x+s) * (c**t - 1)/math.log(c))
life = Survival().set_survival(S=S)
f = life.f_x(x=50, t=10)
isclose(0.0483, f, question="Q2.3")

# + [markdown] id="166f7d31"
# __SOA Question 2.4__ : (E) 8.2
#
#
# You are given $_tq_0 = \frac{t^2}{10,000} \quad 0 < t < 100$. Calculate 
# $\overset{\circ}{e}_{75:\overline{10|}}$.
#
#
# *hints:*
#
#
# - derive survival probability function $_tp_x$ given $_tq_0$
# - compute $\overset{\circ}{e}$ by integration
#

# + colab={"base_uri": "https://localhost:8080/"} id="a6412173" outputId="832f29a4-ef2a-4847-afa7-5ae4612c63c8"
def q(t) : return (t**2)/10000. if t < 100 else 1.
e = Lifetime().set_survival(l=lambda x,s: 1 - q(x+s)).e_x(75, t=10, curtate=False)
isclose(8.2, e, question="Q2.4")

# + [markdown] id="b73ac219"
# __SOA Question 2.5__ :  (B) 37.1
#
# You are given the following:
# 1. $e_{40:20} = 18$
# 2. $e_{60} = 25$
# 3. $_{20}q_{40} = 0.2$
# 4. $q_{40} = 0.003$
#
# Calculate $e_{41}$.
#
# *hints:*
#
#
# - solve for $e_{40}$ from limited lifetime formula
# - compute $e_{41}$ using forward recursion

# + colab={"base_uri": "https://localhost:8080/"} id="7485cb2a" outputId="5006156c-27ea-43b6-925b-080690daafcd"
life = Recursion(verbose=True).set_e(25, x=60, curtate=True)\
                              .set_q(0.2, x=40, t=20)\
                              .set_q(0.003, x=40)\
                              .set_e(18, x=40, t=20, curtate=True)
e = life.e_x(41, curtate=True)
isclose(37.1, e, question="Q2.5")

# + [markdown] id="b626c732"
# __SOA Question 2.6__ : (C) 13.3
#
# You are given the survival function:
#
# $S_0(x) = \left( 1 − \frac{x}{60} \right)^{\frac{1}{3}}, \quad 0 \le x \le 60$
#
# Calculate $1000 \mu_{35}$.
#
# *hints:*
#
#
# - derive force of mortality function $\mu$ from given survival function
#

# + colab={"base_uri": "https://localhost:8080/"} id="b4a5e978" outputId="d60772a4-c868-4a2e-c6fd-c040f5ef9aee"
life = Survival().set_survival(l=lambda x,s: (1 - (x+s)/60)**(1/3))
mu = 1000 * life.mu_x(35)
isclose(13.3, mu, question="Q2.6")

# + [markdown] id="3406e6fd"
# __SOA Question 2.7__ : (B) 0.1477
#
# You are given the following survival function of a newborn:
#
# $$
# \begin{align*}
# S_0(x) & = 1 - \frac{x}{250}, \quad 0 \le x < 40 \\
# & = 1 - \left( \frac{x}{100} \right)^2, \quad 40 \le x \le 100
# \end{align*}
# $$
#
#
# Calculate the probability that (30) dies within the next 20 years.
#
# *hints:*
#
#
# - calculate from given survival function

# + colab={"base_uri": "https://localhost:8080/"} id="1cbb1f35" outputId="e893abe6-5aa6-40a9-9882-b29f17e6390a"
l = lambda x,s: (1-((x+s)/250) if (x+s)<40 else 1-((x+s)/100)**2)
q = Survival().set_survival(l=l).q_x(30, t=20)
isclose(0.1477, q, question="Q2.7")

# + [markdown] id="59f19984"
# __SOA Question 2.8__ : (C) 0.94
#
# In a population initially consisting of 75% females and 25% males, you are given:
#
# 1. For a female, the force of mortality is constant and equals $\mu$
# 2. For a male, the force of mortality is constant and equals 1.5 $\mu$
# 3. At the end of 20 years, the population is expected to consist of 85% females and 15% males
#
# Calculate the probability that a female survives one year.
#
# *hints:*
#
#
# - relate $p_{male}$ and $p_{female}$ through the common term $\mu$ and the given proportions
#

# + colab={"base_uri": "https://localhost:8080/"} id="d9433c52" outputId="1dfc6877-2ff0-4eed-c32a-7ccf0343bbc3"
def fun(mu):  # Solve first for mu, given ratio of start and end proportions
    male = Survival().set_survival(mu=lambda x,s: 1.5 * mu)
    female = Survival().set_survival(mu=lambda x,s: mu)
    return (75 * female.p_x(0, t=20)) / (25 * male.p_x(0, t=20))
mu = Survival.solve(fun, target=85/15, grid=0.5)
p = Survival().set_survival(mu=lambda x,s: mu).p_x(0, t=1)
isclose(0.94, p, question="Q2.8")

# + [markdown] id="42e1818d"
# ## 3 Life tables and selection

# + [markdown] id="8158646b"
# __SOA Question 3.1__ :  (B) 117
#
# You are given:
#
# 1. An excerpt from a select and ultimate life table with a select period of 3 years:
#
# | $x$ | $\ell_{[ x ]}$ | $\ell_{[x]+1}$ | $\ell_{[x]+2}$ | $\ell_{x+3}$ | $x+3$ |
# |---|---|---|---|---|---|
# | 60 | 80,000 | 79,000 | 77,000 | 74,000 | 63 |
# | 61 | 78,000 | 76,000 | 73,000 | 70,000 | 64 |
# | 62 | 75,000 | 72,000 | 69,000 | 67,000 | 65 |
# | 63 | 71,000 | 68,000 | 66,000 | 65,000 | 66 |
#
# 2. Deaths follow a constant force of mortality over each year of age
#
# Calculate $1000~ _{23}q_{[60] + 0.75}$.
#
# *hints:*
#
#
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
# __SOA Question 3.2__ :  (D) 14.7
#
# You are given:
#
# 1. The following extract from a mortality table with a one-year select period:
#
# | $x$ | $l_{[x]}$ | $d_{[x]}$ | $l_{x+1}$ | $x + 1$ |
# |---|---|---|---|---|
# | 65 | 1000 | 40 | − | 66 |
# | 66 | 955 | 45 | − | 67 |
#
# 2. Deaths are uniformly distributed over each year of age
#
# $\overset{\circ}{e}_{[65]} = 15.0$
#
# Calculate $\overset{\circ}{e}_{[66]}$.
#
# *hints:*
#
#
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
e = life.e_r(x=66)
isclose(14.7, e, question="Q3.2")

# + [markdown] id="fb02d76f"
# __SOA Question 3.3__ :  (E) 1074
#
# You are given:
#
# 1. An excerpt from a select and ultimate life table with a select period of 2 years:
#
# | $x$ | $\ell_{[ x ]}$ | $\ell_{[ x ] + 1}$ | $\ell_{x + 2}$ | $x + 2$ |
# |---|---|---|---|---|
# | 50 | 99,000 | 96,000 | 93,000 | 52 |
# | 51 | 97,000 | 93,000 | 89,000 | 53 |
# | 52 | 93,000 | 88,000 | 83,000 | 54 |
# | 53 | 90,000 | 84,000 | 78,000 | 55 |
#
# 2. Deaths are uniformly distributed over each year of age
#
# Calculate $10,000 ~ _{2.2}q_{[51]+0.5}$.
#
# *hints:*
#
#
# - interpolate lives between integer ages with UDD

# + colab={"base_uri": "https://localhost:8080/"} id="9576af60" outputId="831e950a-adcb-4f56-d188-8a8cdabe10b0"
life = SelectLife().set_table(l={50: [99, 96, 93],
                                 51: [97, 93, 89],
                                 52: [93, 88, 83],
                                 53: [90, 84, 78]})
q = 10000 * life.q_r(51, s=0, r=0.5, t=2.2)
isclose(1074, q, question="Q3.3")

# + [markdown] id="2247b56f"
# __SOA Question 3.4__ :  (B) 815
#
# The SULT Club has 4000 members all age 25 with independent future lifetimes. The
# mortality for each member follows the Standard Ultimate Life Table.
#
# Calculate the largest integer N, using the normal approximation, such that the probability that there are at least N survivors at age 95 is at least 90%.
#
#
# *hints:*
#
#
# - compute portfolio percentile with N=4000, and mean and variance  from binomial distribution

# + colab={"base_uri": "https://localhost:8080/"} id="fb29aeca" outputId="396f6731-4b3f-4b22-baa4-edcd2be1fbf0"
sult = SULT()
mean = sult.p_x(25, t=95-25)
var = sult.bernoulli(mean, variance=True)
pct = sult.portfolio_percentile(N=4000, mean=mean, variance=var, prob=0.1)
isclose(815, pct, question="Q3.4")

# + [markdown] id="a989a344"
# __SOA Question 3.5__ :  (E) 106
#
#
# You are given:
#
# | $x$ | 60 | 61 | 62 | 63 |64 | 65 | 66 | 67 |
# |---|---|---|---|---|---|---|---|---|
# | $l_x$ | 99,999 | 88,888 |77,777 | 66,666 | 55,555 | 44,444 | 33,333 | 22,222|
#
# $a =~ _{3.4|2.5}q_{60}$ assuming a uniform distribution of deaths over each year of age
#
# $b =~ _{3.4|2.5}q_{60}$ assuming a constant force of mortality over each year of age
#
# Calculate $100,000( a − b )$
#
# *hints:*
#
#
# - compute mortality rates by interpolating lives between integer ages, with UDD and constant force of mortality assumptions

# + colab={"base_uri": "https://localhost:8080/"} id="4a01bc25" outputId="9143281b-4a6a-4cd7-875b-3722fbdd93ce"
l = [99999, 88888, 77777, 66666, 55555, 44444, 33333, 22222]
a = LifeTable(udd=True).set_table(l={age:l for age,l in zip(range(60, 68), l)})\
                       .q_r(60, u=3.4, t=2.5)
b = LifeTable(udd=False).set_table(l={age:l for age,l in zip(range(60, 68), l)})\
                        .q_r(60, u=3.4, t=2.5)
isclose(106, 100000 * (a - b), question="Q3.5")

# + [markdown] id="cc6f9e8f"
# __SOA Question 3.6__ :  (D) 15.85
#
# You are given the following extract from a table with a 3-year select period:
#
# | $x$ | $q_{[x]}$ | $q_{[x]+1}$ | $q_{[x]+2}$ | $q_{x+3}$ | $x+3$ |
# |---|---|---|---|---|---|
# | 60 | 0.09 | 0.11 | 0.13 | 0.15 | 63 |
# | 61 | 0.10 | 0.12 | 0.14 | 0.16 | 64 |
# | 62 | 0.11 | 0.13 | 0.15 | 0.17 | 65 |
# | 63 | 0.12 | 0.14 | 0.16 | 0.18 | 66 |
# | 64 | 0.13 | 0.15 | 0.17 | 0.19 | 67 |
#
# $e_{64} = 5.10$
#
# Calculate $e_{[61]}$.
#
# *hints:*
#
#
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
# __SOA Question 3.7__ : (b) 16.4
#
# For a mortality table with a select period of two years, you are given:
#
# | $x$ | $q_{[x]}$ | $q_{[x]+1}$ | $q_{x+2}$ | $x+2$ |
# |---|---|---|---|---|
# | 50 | 0.0050 | 0.0063 | 0.0080 | 52 |
# | 51 | 0.0060 | 0.0073 | 0.0090 | 53 |
# | 52 | 0.0070 | 0.0083 | 0.0100 | 54 |
# | 53 | 0.0080 | 0.0093 | 0.0110 | 55 |
#
# The force of mortality is constant between integral ages.
#
# Calculate $1000 ~_{2.5}q_{[50]+0.4}$.
#
# *hints:*
#
#
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
# __SOA Question 3.8__ :  (B) 1505
#
# A club is established with 2000 members, 1000 of exact age 35 and 1000 of exact age 45. You are given:
# 1. Mortality follows the Standard Ultimate Life Table
# 2. Future lifetimes are independent
# 3. N is the random variable for the number of members still alive 40 years after the club is established
#
# Using the normal approximation, without the continuity correction, calculate the smallest $n$ such that $Pr( N \ge n ) \le 0.05$.
#
# *hints:*
#
#
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
# __SOA Question 3.9__ :  (E) 3850
#
# A father-son club has 4000 members, 2000 of which are age 20 and the other 2000 are age 45. In 25 years, the members of the club intend to hold a reunion.
#
# You are given:
# 1. All lives have independent future lifetimes.
# 2. Mortality follows the Standard Ultimate Life Table.
#
# Using the normal approximation, without the continuity correction, calculate the 99th percentile of the number of surviving members at the time of the reunion.
#
# *hints:*
#
#
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
# __SOA Question 3.10__ :  (C) 0.86
#
# A group of 100 people start a Scissor Usage Support Group. The
# rate at which members
# enter and leave the group is dependent on whether they are right-handed or left-handed.
#
# You are given the following:
#
# 1. The initial membership is made up of 75% left-handed members (L)
#  and 25% right-handed members (R)
#
# 2. After the group initially forms, 35 new (L) and 15 new (R) join
#  the group at the
#  start of each subsequent year
#
# 3. Members leave the group only at the end of each year
#
# 4. $q_L$ = 0.25 for all years
#
# 5. $q_R$ = 0.50 for all years
#
#  Calculate the proportion of the Scissor Usage Support Group's expected
#  membership that is left-handed at the start of the group's 6th year, before any new members join for that year.
#
# *hints:*
#
#
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
# __SOA Question 3.11__ :  (B) 0.03
#
# For the country of Bienna, you are given:
# 1. Bienna publishes mortality rates in biennial form, that is, mortality rates are of the form: $_2q_{2x},$ for $x = 0,1, 2,...$
#
# 2. Deaths are assumed to be uniformly distributed between ages $2x$ and $2x + 2$, for $x = 0,1, 2,...$
# 3. $_2q_{50} = 0.02$
# 4. $_2q_{52} = 0.04$
#
# Calculate the probability that (50) dies during the next 2.5 years.
#
# *hints:*
#
#
# - calculate mortality rate by interpolating lives assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="ced9708b" outputId="fb9127a8-b675-4dca-c6e8-5468849c5b0b"
life = LifeTable(udd=True).set_table(q={50//2: .02, 52//2: .04})
q = life.q_r(50//2, t=2.5/2)
isclose(0.03, q, question="Q3.11")

# + [markdown] id="6dae2d07"
# __SOA Question 3.12__ : (C) 0.055 
#
# X and Y are both age 61. X has just purchased a whole life insurance policy. Y purchased a whole life insurance policy one year ago.
#
# Both X and Y are subject to the following 3-year select and ultimate table:
#
# | $x$ | $\ell_{[x]}$ | $\ell_{[x]+1}$ | $\ell_{[x] + 2}$ | $\ell_{x+3}$ | $x+3$ |
# |---|---|---|---|---|---|
# | 60 | 10,000 | 9,600 | 8,640 | 7,771 | 63 |
# | 61 | 8,654 | 8,135 | 6,996 | 5,737 | 64 |
# | 62 | 7,119 | 6,549 | 5,501 | 4,016 | 65 |
# | 63 | 5,760 | 4,954 | 3,765 | 2,410 | 66 |
#
# The force of mortality is constant over each year of age.
#
# Calculate the difference in the probability of survival to age 64.5 between X and Y.
#
# *hints:*
#
#
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
# __SOA Question 3.13__ :  (B) 1.6
#
# A life is subject to the following 3-year select and ultimate table:
#
# | $[x]$ | $\ell_{[x]}$ | $\ell_{[x]+1}$ | $\ell_{[x]+2}$ | $\ell_{x+3}$ | $x+3$ |
# |---|---|---|---|---|---
# | 55 | 10,000 | 9,493 | 8,533 | 7,664 | 58 |
# | 56 | 8,547 | 8,028 | 6,889 | 5,630 | 59 |
# | 57 | 7,011 | 6,443 | 5,395 | 3,904 | 60 |
# | 58 | 5,853 | 4,846 | 3,548 | 2,210 | 61 |
#
# You are also given:
# 1. $e_{60} = 1$
# 2. Deaths are uniformly distributed over each year of age
#
# Calculate $\overset{\circ}{e}_{[58]+2}$ .
#
# *hints:*
#
#
# - compute curtate expectations using recursion formulas
# - convert to complete expectation assuming UDD
#

# + colab={"base_uri": "https://localhost:8080/"} id="8db335e1" outputId="636f1fff-646a-436a-adf1-80c8f765cd43"
life = SelectLife().set_table(l={55: [10000, 9493, 8533, 7664],
                                 56: [8547, 8028, 6889, 5630],
                                 57: [7011, 6443, 5395, 3904],
                                 58: [5853, 4846, 3548, 2210]},
                              e={57: [None, None, None, 1]})
e = life.e_r(x=58, s=2)
isclose(1.6, e, question="Q3.13")

# + [markdown] id="b784697d"
# __SOA Question 3.14__ :  (C) 0.345
#
# You are given the following information from a life table:
#
# | x | $l_x$ | $d_x$ | $p_x$ | $q_x$ |
# |---|---|---|---|---|
# | 95 | − | − | − | 0.40 |
# | 96 | − | − | 0.20 | − |
# | 97 | − | 72 | − | 1.00 |
#
# You are also given:
# 1. $l_{90} = 1000$ and $l_{93} = 825$
# 2. Deaths are uniformly distributed over each year of age.
#
# Calculate the probability that (90) dies between ages 93 and 95.5.
#
# *hints:*
#
#
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
# __SOA Question 4.1__ :  (A) 0.27212
#
# For a special whole life insurance policy issued on (40), you
# are given:
#
# 1. Death benefits are payable at the end of the year of death
#
# 2. The amount of benefit is 2 if death occurs within the first 20
#  years and is 1 thereafter
#
# 3. *Z* is the present value random variable for the payments
#  under this insurance
#
# 4. *i* = 0.03
#
# 5.
#
# | x | $A_x$ | $_{20}E_x$ |
# |---|---|---|
# | 40 | 0.36987 | 0.51276 |
# | 60 | 0.62567 | 0.17878 |
#
#
# 6. $E[Z^2] =0.24954$
#
# Calculate the standard deviation of *Z*.
#
#
# *hints:*
#
#
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
# __SOA Question 4.2__ :  (D) 0.18
#
# or a special 2-year term insurance policy on (*x*), you are
# given:
#
# 1. Death benefits are payable at the end of the half-year of death
#
# 2. The amount of the death benefit is 300,000 for the first
#  half-year and increases by 30,000 per half-year thereafter
#
# 3. $q_x$ = 0.16 and $q_{x+1}$ = 0.23
#
# 4. $i^{(2)}$ = 0.18
#
# 5. Deaths are assumed to follow a constant force of mortality
#  between integral ages
#
# 6. *Z* is the present value random variable for this insurance
#
# Calculate Pr( *Z* \> 277,000) .
#
# *hints:*
#
#
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
# __SOA Question 4.3__ : (D) 0.878
#
# You are given:
#
# 1. $q_{60} = 0.01$
# 2. Using $i = 0.05, ~ A_{60:\overline{3|}} = 0.86545$
# 3. Using $i = 0.045$ calculate $A_{60:\overline{3|}}$
#
# *hints:*
#
#
# - solve $q_{61}$ from endowment insurance EPV formula
# - solve $A_{60:\overline{3|}}$ with new $i=0.045$ as EPV of endowment insurance benefits.
#

# + colab={"base_uri": "https://localhost:8080/"} id="db579f3b" outputId="e6e2e006-1b78-45b6-b270-435ad567034c"
life = Recursion(verbose=True).set_interest(i=0.05)\
                              .set_q(0.01, x=60)\
                              .set_A(0.86545, x=60, t=3, endowment=1)
q = life.q_x(x=61)
A = Recursion(verbose=True).set_interest(i=0.045)\
                           .set_q(0.01, x=60)\
                           .set_q(q, x=61)\
                           .endowment_insurance(60, t=3)
isclose(0.878, A, question="Q4.3")

# + [markdown] id="de2d0427"
# __SOA Question 4.4__ : (A) 0.036
#
# For a special increasing whole life insurance on (40), payable at the moment of death, you are given :
# 1. The death benefit at time t is $b_t = 1 + 0.2 t, \quad t \ge 0$
# 2. The interest discount factor at time t is $v(t) = (1 + 0.2 t ) − 2, \quad t \ge 0$
# 3. $_tp_{40} ~ \mu_{40+t} = 0.025~\text{if} ~ 0 \le t < 40$, otherwise $0$
# 4. Z is the present value random variable for this insurance
#
# Calculate Var(Z).
#
# *hints:*
#
#
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
# __SOA Question 4.5__ :  (C) 35200
#
# For a 30-year term life insurance of 100,000 on (45), you are given:
# 1. The death benefit is payable at the moment of death
# 2. Mortality follows the Standard Ultimate Life Table
# 3. $\delta = 0.05$
# 4. Deaths are uniformly distributed over each year of age
#
# Calculate the 95th percentile of the present value of benefits random variable for this insurance
#
# *hints:*
#
#
# - interpolate between integer ages with UDD, and find lifetime that mortality rate exceeded
# - compute PV of death benefit paid at that time.
#

# + colab={"base_uri": "https://localhost:8080/"} id="3c9d0b1e" outputId="9ce6ae62-ce6a-4afb-d1bf-abe00bb38caf"
sult = SULT(udd=True).set_interest(delta=0.05)
Z = 100000 * sult.Z_from_prob(45, prob=0.95, discrete=False)
isclose(35200, Z, question="Q4.5")

# + [markdown] id="1792b7aa"
# __SOA Question 4.6__ :  (B) 29.85
#
# For a 3-year term insurance of 1000 on (70), you are given:
# 1. $q^{SULT}_{70+k}$ is the mortality rate from the Standard Ultimate Life Table, for k = 0,1,2
#
# 2. $q_{70 + k}$ is the mortality rate used to price this insurance, for k = 0,1, 2
#
# 3. $q_{70 + k} = (0.95)^k q_{70+k}^{SULT}$, for k = 0,1, 2
#
# 4. *i* = 0.05
#
# Calculate the single net premium.
#
# *hints:*
#
#
# - calculate adjusted mortality rates
# - compute term insurance as EPV of benefits

# + colab={"base_uri": "https://localhost:8080/"} id="f31ee601" outputId="ea7759a3-8d35-44f8-8015-34afc05162e1"
sult = SULT()
life = LifeTable().set_interest(i=0.05)\
                  .set_table(q={70+k: .95**k * sult.q_x(70+k) for k in range(3)})
A = life.term_insurance(70, t=3, b=1000)
isclose(29.85, A, question="Q4.6")


# + [markdown] id="230429ad"
# __SOA Question 4.7__ :  (B) 0.06
#
# For a 25-year pure endowment of 1 on (*x*), you are given:
#
# 1. *Z* is the present value random variable at issue of the benefit
#  payment
#
# 2. *Var (Z)* = 0.10 *E[Z]*
#
# 3. $_{25}p_x = 0.57$
#
#  Calculate the annual effective interest rate.
#
# *hints:*
#
# - use Bernoulli shortcut formula for variance of pure endowment Z 
# - solve for $i$, since $p$ is given.

# + colab={"base_uri": "https://localhost:8080/"} id="f38c4ab6" outputId="f9f9dca4-f476-41fa-c282-5ac5700d99c2"
def fun(i):
    life = Recursion(verbose=False).set_interest(i=i)\
                                   .set_p(0.57, x=0, t=25)
    return 0.1*life.E_x(0, t=25) - life.E_x(0, t=25, moment=life.VARIANCE)
i = Recursion.solve(fun, target=0, grid=[0.058, 0.066])
isclose(0.06, i, question="Q4.7")

# + [markdown] id="ccb0f3ff"
# __SOA Question 4.8__ :  (C) 191
#
# For a whole life insurance of 1000 on (50), you are given :
#
# 1. The death benefit is payable at the end of the year of death
#
# 2. Mortality follows the Standard Ultimate Life Table
#
# 3. *i* = 0.04 in the first year, and *i* = 0.05 in subsequent
#  years
#
#  Calculate the actuarial present value of this insurance.
#
# *hints:*
#
# - use insurance recursion with special interest rate $i=0.04$ in first year.
#

# + colab={"base_uri": "https://localhost:8080/"} id="f3ad0bbe" outputId="ab3a4680-849c-4997-edb9-521ef8bc0dde"
def v_t(t): return 1.04**(-t) if t < 1 else 1.04**(-1) * 1.05**(-t+1)
A = SULT().set_interest(v_t=v_t).whole_life_insurance(50, b=1000)
isclose(191, A, question="Q4.8")

# + [markdown] id="4408c9ef"
# __SOA Question 4.9__ :  (D) 0.5
#
# You are given:
#
# 1. $A_{35:\overline{15|}} = 0.39$
# 2. $A^1_{35:\overline{15|}} = 0.25$
# 4. $A_{35} = 0.32$
#
# Calculate $A_{50}$.
#
# *hints:*
#
#
# - solve $_{15}E_{35}$ from endowment insurance minus term insurance
#
# - solve implicitly from whole life as term plus deferred insurance
#

# + colab={"base_uri": "https://localhost:8080/"} id="0ab006d1" outputId="39b2b025-14e7-43c3-9b26-cdb734ec6915"
E = Recursion().set_A(0.39, x=35, t=15, endowment=1)\
               .set_A(0.25, x=35, t=15)\
               .E_x(35, t=15)
life = Recursion(verbose=False).set_A(0.32, x=35)\
                               .set_E(E, x=35, t=15)
def fun(A): return life.set_A(A, x=50).term_insurance(35, t=15)
A = life.solve(fun, target=0.25, grid=[0.35, 0.55])
isclose(0.5, A, question="Q4.9")

# + [markdown] id="f46ca953"
# __SOA Question 4.10__ :  (D)
#
# The present value random variable for an insurance policy on (x) is expressed as:
# $$\begin{align*}
# Z & =0, \quad \textrm{if } T_x \le 10\\
# & =v^T, \quad \textrm{if } 10 < T_x \le 20\\
# & =2v^T, \quad \textrm{if } 20 < T_x \le 30\\
# & =0, \quad \textrm{thereafter}
# \end{align*}$$
#
# Determine which of the following is a correct expression for $E[Z]$.
#
#
# (A) $_{10|}\overline{A}_x + _{20|}\overline{A}_x - _{30|}\overline{A}_x$
#
# (B) $\overline{A}_x + _{20}E_x \overline{A}_{x+20} - 2~_{30}E_x \overline{A}_{x +30}$
#
# (C) $_{10}E_x \overline{A}_x + _{20}E_x \overline{A}_{x+20} - 2 ~_{30}E_x \overline{A}_{x +30}$
#
# (D) $_{10}E_x \overline{A}_{x+10} + _{20}E_x \overline{A}_{x+20} - 2~ _{30}E_x \overline{A}_{x+30}$
#
# (E) $_{10}E_x [\overline{A}_{x} + _{10}E_{x+10} + \overline{A}_{x+20} - _{10}E_{x+20} + \overline{A}_{x+30}]$
#
# *hints:*
#
# - draw and compare benefit diagrams
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
# __SOA Question 4.11__ :  (A) 143385
#
# You are given:
#
# 1. $Z_1$ is the present value random variable for an n-year term insurance of 1000
# issued to (x)
# 2. $Z_2$ is the present value random variable for an n-year endowment insurance of
# 1000 issued to (x)
# 3. For both $Z_1$ and $Z_2$ the death benefit is payable at the end of the year of death
# 4. $E [ Z_1 ] = 528$
# 5. $Var ( Z_2 ) = 15,000$
# 6. $A^{~~~~1}_{x:{\overline{n|}}} = 0.209$
# 7. $^2A^{~~~~1}_{x:{\overline{n|}}} = 0.136$
#
# Calculate $Var(Z_1)$.
#
# *hints:*
#
#
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
# __SOA Question 4.12__ :  (C) 167 
#
# For three fully discrete insurance products on the same (x), you are given:
# - $Z_1$ is the present value random variable for a 20-year term insurance of 50
# - $Z_2$ is the present value random variable for a 20-year deferred whole life
# insurance of 100
# - $Z_3$ is the present value random variable for a whole life insurance of 100.
# - $E[Z_1] = 1.65$ and $E[Z_2] = 10.75$
# - $Var(Z_1) = 46.75$ and $Var(Z_2) = 50.78$
#
# Calculate $Var(Z_3)$.
#
# *hints:*
#
#
# - since $Z_1,~Z_2$ are non-overlapping, $E[Z_1~ Z_2] = 0$ for computing $Cov(Z_1, Z_2)$
# - whole life is sum of term and deferred, hence equals variance of components plus twice their covariance
#

# + colab={"base_uri": "https://localhost:8080/"} id="b34e726c" outputId="87270bd2-36a4-4448-bd22-9b8c48ed8d6b"
cov = Life.covariance(a=1.65, b=10.75, ab=0)  # E[Z1 Z2] = 0 nonoverlapping
var = Life.variance(a=2, b=1, var_a=46.75, var_b=50.78, cov_ab=cov)
isclose(167, var, question="Q4.12")

# + [markdown] id="ae69b52f"
# __SOA Question 4.13__ :  (C) 350 
#
# For a 2-year deferred, 2-year term insurance of 2000 on [65], you are given:
#
# 1. The following select and ultimate mortality table with a 3-year select period:
#
# | $x$ | $q_{[x]}$ | $q_{[x]+1}$ | $q_{[x]+2}$ | $q_{x+3}$ | $x+3$ |
# |---|---|---|---|---|---
# | 65 | 0.08 | 0.10 | 0.12 | 0.14 | 68 |
# | 66 | 0.09 | 0.11 | 0.13 | 0.15 | 69 |
# | 67 | 0.10 | 0.12 | 0.14 | 0.16 | 70 |
# | 68 | 0.11 | 0.13 | 0.15 | 0.17 | 71 |
# | 69 | 0.12 | 0.14 | 0.16 | 0.18 | 72 |
#
# 2. $i = 0.04$
# 3. The death benefit is payable at the end of the year of death
#
# Calculate the actuarial present value of this insurance.
#
# *hints:*
#
#
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
# __SOA Question 4.14__ :  (E) 390000
#
# A fund is established for the benefit of 400 workers all age 60 with independent future lifetimes. When they reach age 85, the fund will be dissolved and distributed to the survivors.
#
# The fund will earn interest at a rate of 5% per year.
#
# The initial fund balance, $F$, is determined so that the probability that the fund will pay at least 5000 to each survivor is 86%, using the normal approximation.
#
# Mortality follows the Standard Ultimate Life Table.
#
# Calculate $F$.
#
# *hints:*
#
#
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
# __SOA Question 4.15__ :  (E) 0.0833 
#
# For a special whole life insurance on (x), you are given :
# - Death benefits are payable at the moment of death
# - The death benefit at time $t$ is $b_t = e^{0.02t}$, for $t \ge 0$
# - $\mu_{x+t} = 0.04$, for $t \ge 0$
# - $\delta = 0.06$
# - Z is the present value at issue random variable for this insurance.
#
# Calculate $Var(Z)$.
#
# *hints:*
#
#
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
# __SOA Question 4.16__ :  (D) 0.11
#
# You are given the following extract of ultimate mortality rates from a two-year select and ultimate mortality table:
#
# |$x$ | $q_x$ |
# |---|---|
# | 50 | 0.045 |
# | 51 | 0.050 |
# | 52 | 0.055 |
# | 53 | 0.060 |
#
# The select mortality rates satisfy the following:
# 1. $q_{[x]} = 0.7 q_x$
# 2. $q_{[x]+1} = 0.8 q_{x + 1}$
#
# You are also given that $i = 0.04$.
#
# Calculate $A^1_{[50]:\overline{3|}}$.
#
# *hints:*
#
#
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
# __SOA Question 4.17__ :  (A) 1126.7
#
# For a special whole life policy on (48), you are given:
#
# 1. The policy pays 5000 if the insured’s death is before the median curtate future
# lifetime at issue and 10,000 if death is after the median curtate future lifetime at issue
# 2. Mortality follows the Standard Ultimate Life Table
# 3. Death benefits are paid at the end of the year of death
# 4. i = 0.05
#
# Calculate the actuarial present value of benefits for this policy.
#
# *hints:*
#
#
# - find future lifetime with 50\% survival probability
# - compute EPV of special whole life as sum of term and deferred insurance, that have different benefit amounts before and after median lifetime.

# + colab={"base_uri": "https://localhost:8080/"} id="330ac8db" outputId="746e4217-1b91-4477-e42e-a4a95f371c1f"
sult = SULT()
median = sult.Z_t(48, prob=0.5, discrete=False)
def benefit(x,t): return 5000 if t < median else 10000
A = sult.A_x(48, benefit=benefit)
isclose(1130, A, question="Q4.17")

# + [markdown] id="258c80e6"
# __SOA Question 4.18__ :  (A) 81873 
#
# You are given that T, the time to first failure of an industrial robot, has a density f(t) given by
#
# $$
# \begin{align*}
# f(t) &= 0.1, \quad 0 \le t < 2\\
#  &= 0.4t^{-2}, \quad t \le t < 10
# \end{align*}
# $$
#
# with $f(t)$ undetermined on $[10, \infty)$.
#
# Consider a supplemental warranty on this robot that pays 100,000 at the time T of its first failure if $2 \le T \le 10$ , with no benefits payable otherwise.
# You are also given that $\delta = 5\%$. Calculate the 90th percentile of the present value of the future benefits under this warranty.
#
# *hints:*
#
#
# - find values of limits such that integral of lifetime density function equals required survival probability
#

# + colab={"base_uri": "https://localhost:8080/"} id="53795941" outputId="db564de7-cefa-498e-fff6-356c069639f9"
def f(x,s,t): return 0.1 if t < 2 else 0.4*t**(-2)
life = Insurance().set_interest(delta=0.05)\
                  .set_survival(f=f, maxage=10)
def benefit(x,t): return 0 if t < 2 else 100000
prob = 0.9 - life.q_x(0, t=2)
T = life.Z_t(0, prob=prob)
Z = life.Z_from_t(T, discrete=False) * benefit(0, T)
isclose(81873, Z, question="Q4.18")

# + [markdown] id="04492903"
# __SOA Question 4.19__ :  (B) 59050
#
# (80) purchases a whole life insurance policy of 100,000. You are given:
# 1. The policy is priced with a select period of one year
# 2. The select mortality rate equals 80% of the mortality rate from the Standard
# Ultimate Life Table
# 3. Ultimate mortality follows the Standard Ultimate Life Table
# 4. $i = 0.05$
#
# Calculate the actuarial present value of the death benefits for this insurance
#
# *hints:*
#
#
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
# __SOA Question 5.1__ : (A) 0.705
#
# You are given:
# 1. $\delta_t = 0.06, \quad t \ge 0$
# 2. $\mu_x(t) = 0.01, \quad t \ge 0$
# 3. $Y$ is the present value random variable for a continuous annuity of 1 per year,
# payable for the lifetime of (x) with 10 years certain
#
# Calculate $Pr( Y > E[Y])$.
#
# *hints:*
#
#
# - sum annuity certain and deferred life annuity with constant force of mortality shortcut
# - apply equation for PV annuity r.v. Y to infer lifetime
# - compute survival probability from constant force of mortality function.
#

# + colab={"base_uri": "https://localhost:8080/"} id="18b1a0c0" outputId="683baef8-8a0a-4d77-a92e-84854e8023f3"
life = ConstantForce(mu=0.01).set_interest(delta=0.06)
EY = life.certain_life_annuity(0, u=10, discrete=False)
p = life.p_x(0, t=life.Y_to_t(EY))
isclose(0.705, p, question="Q5.1")  # 0.705

# + [markdown] id="f90b71c6"
# __SOA Question 5.2__ :  (B) 9.64
#
# You are given:
#
# 1. $A_x = 0.30$
# 2. $A_{x + n} = 0.40$
# 3. $A^{~~~~1}_{x:\overline{n|}} = 0.35$
# 4. *i* = 0.05
#    
# Calculate $a_{x:\overline{n|}}$.
#
# *hints:*
#
#
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
# __SOA Question 5.3__ :  (C) 6.239
#
# You are given:
#
# - Mortality follows the Standard Ultimate Life Table
# - Deaths are uniformly distributed over each year of age
# - i = 0.05
#
# Calculate
# $\frac{d}{dt}(\overline{I}\overline{a})_{40:\overline{t|}}$ at $t = 10.5$.
#
# *hints:*
#
# - Differential reduces to be the EPV of the benefit payment at the upper time limit.
#

# + colab={"base_uri": "https://localhost:8080/"} id="eeca16c1" outputId="55f511e0-6919-4a9a-c905-4d1354bb3660"
t = 10.5
E = t * SULT().E_r(40, t=t)
isclose(6.239, E, question="Q5.3")

# + [markdown] id="cd3027da"
# __SOA Question 5.4__ :  (A) 213.7
#
# (40) wins the SOA lottery and will receive both:
# - A deferred life annuity of K per year, payable continuously, starting at age
# $40 + \overset{\circ}{e}_{40}$ and
# - An annuity certain of K per year, payable continuously, for $\overset{\circ}{e}_{40}$ years
#
# You are given:
# 1. $\mu = 0.02$
# 2. $\delta = 0.01$
# 3. The actuarial present value of the payments is 10,000
#
# Calculate K.
#
# *hints:*
#
#
# - compute certain and life annuity factor as the sum of a certain annuity and a deferred life annuity.
# - solve for amount of annual benefit that equals given EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="297311f0" outputId="9a5628e5-3106-4e66-cc41-a64b7bee4650"
life = ConstantForce(mu=0.02).set_interest(delta=0.01)
u = life.e_x(40, curtate=False)
P = 10000 / life.certain_life_annuity(40, u=u, discrete=False)
isclose(213.7, P, question="Q5.4") # 213.7

# + [markdown] id="46f357cd"
# __SOA Question 5.5__ : (A) 1699.6
#
# For an annuity-due that pays 100 at the beginning of each year that (45) is alive, you are given:
# 1. Mortality for standard lives follows the Standard Ultimate Life Table
# 2. The force of mortality for standard lives age 45 + t is represented as $\mu_{45+t}^{SULT}$
# 3. The force of mortality for substandard lives age 45 + t, $\mu_{45+t}^{S}$, is defined as:
#
# $$\begin{align*}
# \mu_{45+t}^{S} &= \mu_{45+t}^{SULT} + 0.05, \quad 0 \le t < 1\\
# &= \mu_{45+t}^{SULT}, \quad t \ge 1
# \end{align*}$$
# 4. $i = 0.05$
#
# Calculate the actuarial present value of this annuity for a substandard life age 45.
#
# *hints:*
#
#
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
# __SOA Question 5.6__ :  (D) 1200
#
#
# For a group of 100 lives age x with independent future lifetimes, you are given:
# 1. Each life is to be paid 1 at the beginning of each year, if alive
# 2. $A_x = 0.45$
# 3. $^2A_x = 0.22$
# 4. $i = 0.05$
# 5. $Y$ is the present value random variable of the aggregate payments.
#
# Using the normal approximation to $Y$, calculate the initial size of the fund needed to be 95% certain of being able to make the payments for these life annuities.
#
#
# *hints:*
#
#
# - compute mean and variance of EPV of whole life annuity from whole life insurance twin and variance identities. 
# - portfolio percentile of the sum of $N=100$ life annuity payments

# + colab={"base_uri": "https://localhost:8080/"} id="8445b834" outputId="44946fe9-270f-405e-bad3-4d526a8c9c0e"
life = Annuity().set_interest(i=0.05)
var = life.annuity_variance(A2=0.22, A1=0.45)
mean = life.annuity_twin(A=0.45)
fund = life.portfolio_percentile(mean, var, prob=.95, N=100)
isclose(1200, fund, question="Q5.6")

# + [markdown] id="b7c08c39"
# __SOA Question 5.7__ :  (C) 
#
# You are given:
# 1. $A_{35} = 0.188$
# 2. $A_{65} = 0.498$
# 3. $_{30}p_{35} = 0.883$
# 4. $i = 0.04$
#
# Calculate $1000 \ddot{a}^{(2)}_{35:\overline{30|}}$ using the two-term Woolhouse approximation.
#
# *hints:*
#
#
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
# __SOA Question 5.8__ : (C) 0.92118
#
# For an annual whole life annuity-due of 1 with a 5-year certain period on (55), you are given:
# 1. Mortality follows the Standard Ultimate Life Table
# 2. i = 0.05
#
# Calculate the probability that the sum of the undiscounted payments actually made under this annuity will exceed the expected present value, at issue, of the annuity.
#
# *hints:*
#
#
# - calculate EPV of certain and life annuity.
# - find survival probability of lifetime s.t. sum of annual payments exceeds EPV
#

# + colab={"base_uri": "https://localhost:8080/"} id="3db058df" outputId="19eee192-a275-446c-dbf3-29c12abd710b"
sult = SULT()
a = sult.certain_life_annuity(55, u=5)
p = sult.p_x(55, t=math.floor(a))
isclose(0.92118, p, question="Q5.8")

# + [markdown] id="ad7d5d47"
# __SOA Question 5.9__ :  (C) 0.015
#
#
# *hints:*
#
#
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
# __SOA Question 6.1__ : (D) 35.36
#
# **6.1.** You are given the following information about a special fully discrete 2-payment, 2-year term insurance on (80):
#
#  \(i\) Mortality follows the Standard Ultimate Life Table
#
#  \(ii\) *i* = 0.03
#
#  \(iii\) The death benefit is 1000 plus a return of all premiums paid
#  without interest
#
#  \(iv\) Level premiums are calculated using the equivalence principle
#
#  Calculate the net premium for this special insurance.
#
#  \[A modified version of Question 22 on the Fall 2012 exam\]
#
# *hints:*
#
#
# - solve net premium such that EPV annuity = EPV insurance + IA factor for returns of premiums without interest

# + colab={"base_uri": "https://localhost:8080/"} id="68d68c2e" outputId="0ca740e0-370e-40f3-f91c-f1267c58d20b"
P = SULT().set_interest(i=0.03)\
          .net_premium(80, t=2, b=1000, return_premium=True)
isclose(35.36, P, question="Q6.1")

# + [markdown] id="8a9f7924"
# __SOA Question 6.2__ : (E) 3604
#
# **6.2.** For a fully discrete 10-year term life insurance policy on (*x*), you are given:
#
#  \(i\) Death benefits are 100,000 plus the return of all gross premiums paid without interest
#
#  \(ii\) Expenses are 50% of the first year's gross premium, 5% of
#  renewal gross premiums and 200 per policy expenses each year
#
#  \(iii\) Expenses are payble at the beginnig of the year
#
#  \(iv\) $A^1_{x:\overline{10|}} = 0.17094$
#
#  \(v\) $(IA)^1_{x:\overline{10|}} = 0.96728$
#
#  \(vi\) $\ddot{a}^1_{x:\overline{10|}} = 6.8865$
#
#  Calculate the gross premium using the equivalence principle.
#
#  \[Question 25 on the Fall 2012 exam\]
#
# *hints:*
#
#
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
# __SOA Question 6.3__ :  (C) 0.390
#
# S, now age 65, purchased a 20-year deferred whole life
# annuity-due of 1 per year at age
# 45. You are given:
#
# 1. Equal annual premiums, determined using the equivalence
#  principle, were paid at
# the beginning of each year during the deferral period
#
# 2. Mortality at ages 65 and older follows the Standard Ultimate
#  Life Table
#
# 3. *i* = 0.05
#
# 4. *Y* is the present value random variable at age 65 for S's
#  annuity benefits
#
# Calculate the probability that *Y* is less than the actuarial
# accumulated value of S's
# premiums.
#
# *hints:*
#
#
# - solve lifetime $t$ such that PV annuity certain = PV whole life annuity at age 65
# - calculate mortality rate through the year before curtate lifetime   
#

# + colab={"base_uri": "https://localhost:8080/"} id="1d438209" outputId="7e2e7ab7-eb87-4736-eeab-808846b22e23"
life = SULT()
t = life.Y_to_t(life.whole_life_annuity(65))
q = 1 - life.p_x(65, t=math.floor(t) - 1)
isclose(0.39, q, question="Q6.3")

# + [markdown] id="8afc2a87"
# __SOA Question 6.4__ :  (E) 1890
#
# For whole life annuities-due of 15 per month on each of 200
# lives age 62 with
# independent future lifetimes, you are given:
#
# 1. *i* = 0.06
#
# 2. $A^{12}_{62} = 0.4075$ and $^2A^{(12)}_{62} = 0.2105$
#
# 3. $\pi$ is the single premium to be paid by each of the 200 lives
#
# 4. *S* is the present value random variable at time 0 of total
#  payments made to the 200 lives
#
# Using the normal approximation, calculate $\pi$ such at $Pr(200 \pi > S) = 0.90$

# + colab={"base_uri": "https://localhost:8080/"} id="5b9948fb" outputId="28337f0d-0910-46c6-b4e4-bc5a8745bf24"
mthly = Mthly(m=12, life=Annuity().set_interest(i=0.06))
A1, A2 = 0.4075, 0.2105
mean = mthly.annuity_twin(A1) * 15 * 12
var = mthly.annuity_variance(A1=A1, A2=A2, b=15 * 12)
S = Annuity.portfolio_percentile(mean=mean, variance=var, prob=.9, N=200) / 200
isclose(1890, S, question="Q6.4")

# + [markdown] id="fd4150b6"
# __SOA Question 6.5__ :  (D) 33
#
# For a fully discrete whole life insurance of 1000 on (30), you
# are given:
#
# 1. Mortality follows the Standard Ultimate Life Table
#
# 2. *i* = 0.05
#
# 3. The premium is the net premium
#
# Calculate the first year for which the expected present value at issue
# of that year's premium is less than the expected present value at issue of that
# year's benefit.
#

# + colab={"base_uri": "https://localhost:8080/"} id="bda89a9a" outputId="aee7da6d-8a7d-4dae-c28a-6bbcee937c29"
life = SULT()
P = life.net_premium(30, b=1000)
def gain(k): 
    return life.Y_x(30, t=k) * P - life.Z_x(30, t=k) * 1000
k = min([k for k in range(100) if gain(k) < 0]) + 1  # add 1 because k=0 is first policy year
isclose(33, k, question="Q6.5")

# + [markdown] id="bba959b2"
# __SOA Question 6.6__ :  (B) 0.79
#
# For fully discrete whole life insurance policies of 10,000 issued on 600 lives with independent future lifetimes, each age 62, you are given:
#
# 1. Mortality follows the Standard Ultimate Life Table
#
# 2. *i* = 0.05
#
# 3. Expenses of 5% of the first year gross premium are incurred at
#  issue
#
# 4. Expenses of 5 per policy are incurred at the beginning of each
#  policy year
#
# 5. The gross premium is 103% of the net premium.
#
# 6. $_0L$ is the aggregate present value of future loss at issue
#  random variable
#
# Calculate $Pr( _0L < 40,000)$, using the normal approximation.
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
# __SOA Question 6.7__ :  (C) 2880
#
# For a special fully discrete 20-year endowment insurance on
# (40), you are given:
#
# 1. The only death benefit is the return of annual net premiums
#  accumulated with interest at 5% to the end of the year of death
#
# 2. The endowment benefit is 100,000
#
# 3. Mortality follows the Standard Ultimate Life Table
#
# 4. *i* = 0.05
#
#  Calculate the annual net premium.
#

# + colab={"base_uri": "https://localhost:8080/"} id="56437e4c" outputId="1ed51001-e7f6-4570-e42b-de1a87146e6b"
life = SULT()
a = life.temporary_annuity(40, t=20) 
A = life.E_x(40, t=20)
IA = a - life.interest.annuity(t=20) * life.p_x(40, t=20)
G = life.gross_premium(a=a, A=A, IA=IA, benefit=100000)
isclose(2880, G, question="Q6.7")

# + [markdown] id="af651363"
# __SOA Question 6.8__ :  (B) 9.5
#
# For a fully discrete whole life insurance on (60), you are
# given:
#
# 1. Mortality follows the Standard Ultimate Life Table
#
# 2. *i* = 0.05
#
# 3. The expected company expenses, payable at the beginning of the
#  year, are:
#
#   - 50 in the first year
#
#   - 10 in years 2 through 10
#
#   - 5 in years 11 through 20
#
#   - 0 after year 20
#
#  Calculate the level annual amount that is actuarially equivalent to
#  the expected company expenses.
#
#
# *hints:*
#
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
# __SOA Question 6.9__ :  (D) 647
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
# __SOA Question 6.10__ :  (D) 0.91
#
# For a fully discrete 3-year term insurance of 1000 on (*x*),
# you are given:
#
# 1. $p_x$ = 0.975
#
# 2. *i* = 0.06
#
# 3. The actuarial present value of the death benefit is 152.85
#
# 4. The annual net premium is 56.05
#
#  Calculate $p_{x+2}$.
#

# + colab={"base_uri": "https://localhost:8080/"} id="a6ea62e1" outputId="eeae8f35-92a8-48f3-c9e8-c2db9782fd03"
x = 0
life = Recursion(depth=5).set_interest(i=0.06)\
                         .set_p(0.975, x=x)\
                         .set_a(152.85/56.05, x=x, t=3)\
                         .set_A(152.85, x=x, t=3, b=1000)
p = life.p_x(x=x+2)
isclose(0.91, p, question="Q6.10")

# + [markdown] id="1a93e76e"
# __SOA Question 6.11__ :  (C) 0.041
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
# __SOA Question 6.12__ :  (E) 88900
#
# For a fully discrete whole life insurance of 1000 on (x), you are given:
# 1. The following expenses are incurred at the beginning of each year:
#
# | | Year 1 | Years 2+ |
# |---|---|---|
# | Percent of premium | 75% | 10% |
# | Maintenance expenses | 10 | 2 |
#
# 2. An additional expense of 20 is paid when the death benefit is paid
# 3. The gross premium is determined using the equivalence principle
# 4. $i = 0.06$
# 5. $\ddot{a}_x = 12.0$
# 6. $^2A_x = 0.14$
#
# Calculate the variance of the loss at issue random variable.
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
# __SOA Question 6.13__ :  (D) -400
#
# For a fully discrete whole life insurance of 10,000 on (45),
# you are given:
#
# 1. Commissions are 80% of the first year premium and 10% of
# subsequent premiums. There are no other expenses
#
# 2. Mortality follows the Standard Ultimate Life Table
#
# 3. *i* = 0.05
#
# 4. $_0L$ denotes the loss at issue random variable
#
# 5. If $T_{45} = 10.5$, then $_0L = 4953$
#
#  Calculate $E[_0L]$ .
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
# __SOA Question 6.14__ :  (D) 1150
#
# For a special fully discrete whole life insurance of 100,000
# on (40), you are given: 
#
# 1. The annual net premium is *P* for years 1 through 10, 0.5*P* for
# years 11 through 20, and 0 thereafter
#
# 2. Mortality follows the Standard Ultimate Life Table
#
# 3. *i* = 0.05
#
# Calculate *P*.

# + colab={"base_uri": "https://localhost:8080/"} id="d6f0c625" outputId="9eabb789-da5b-4b87-b927-fba149cc4bef"
life = SULT().set_interest(i=0.05)
a = life.temporary_annuity(40, t=10) + 0.5*life.deferred_annuity(40, u=10, t=10)
A = life.whole_life_insurance(40)
P = life.gross_premium(a=a, A=A, benefit=100000)
isclose(1150, P, question="Q6.14")

# + [markdown] id="ba7ed0a0"
# __SOA Question 6.15__ :  (B) 1.002
#
# For a fully discrete whole life insurance of 1000 on (x) with net premiums payable quarterly, you are given:
# 1. $i = 0.05$
# 2. $\ddot{a}_x = 3.4611$
# 3. $P^{(W)}$ and $P^{(UDD)}$ are the annualized net premiums calculated using the 2-term Woolhouse (W) and the uniform distribution of deaths (UDD) assumptions,
# respectively
#
# Calculate $\dfrac{P^{(UDD)}}{P^{(W)}}$.
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
# __SOA Question 6.16__ : (A) 2408.6
#
# For a fully discrete 20-year endowment insurance of 100,000 on (30), you are given:
# 1. d = 0.05
# 2. Expenses, payable at the beginning of each year, are:
#
# | | First Year | First Year | Renewal Years | Renewal Years |
# | --- | --- | --- | --- | --- |
# | | Percent of Premium | Per Policy | Percent of Premium | Per Policy |
# | Taxes | 4% | 0 | 4% | 0  |
# | Sales Commission | 35% | 0 | 2% | 0  |
# | Policy Maintenance | 0% | 250 | 0% | 50  |
#
# 3. The net premium is 2143
#
# Calculate the gross premium using the equivalence principle.
#
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
# __SOA Question 6.17__ :  (A) -30000
#
# An insurance company sells special fully discrete two-year endowment insurance policies to smokers (S) and non-smokers (NS) age x. You are given:
#
# 1. The death benefit is 100,000; the maturity benefit is 30,000
# 2. The level annual premium for non-smoker policies is determined by the
# equivalence principle
# 3. The annual premium for smoker policies is twice the non-smoker annual premium
# 4. $\mu^{NS}_{x+t} = 0.1.\quad t > 0$
# 5. $q^S_{x+k} = 1.5 q_{x+k}^{NS}$, for $k = 0, 1$
# 6. $i = 0.08$
#
# Calculate the expected present value of the loss at issue random variable on a smoker policy.
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
# __SOA Question 6.18__ :  (D) 166400
#
# For a 20-year deferred whole life annuity-due with annual
# payments of 30,000 on (40), you are given:
#
# 1. The single net premium is refunded without interest at the end of
# the year of death if death occurs during the deferral period
#
# 2. Mortality follows the Standard Ultimate Life Table
#
# 3.  *i* = 0.05 
#
#
# Calculate the single net premium for this annuity.
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
# __SOA Question 6.19__ :  (B) 0.033
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
# __SOA Question 6.20__ :  (B) 459
#
# For a special fully discrete 3-year term insurance on (75), you are given:
#
# 1. The death benefit during the first two years is the sum of the net premiums paid
# without interest
#
# 2. The death benefit in the third year is 10,000
#
# | $x$ | $p_x$ |
# |---|---|
# | 75 | 0.90 |
# | 76 | 0.88 |
# | 77 | 0.85 |
#
# 3. $i = 0.04$
#
# Calculate the annual net premium.
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
# __SOA Question 6.21__ :  (C) 100
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
# __SOA Question 6.22__ :  (C) 102
#
# For a whole life insurance of 100,000 on (45) with premiums payable monthly for a
# period of 20 years, you are given:
# 1. The death benefit is paid immediately upon death
# 2. Mortality follows the Standard Ultimate Life Table
# 3. Deaths are uniformly distributed over each year of age
# 4. $i = 0.05$
#
# Calculate the monthly net premium.
#

# + colab={"base_uri": "https://localhost:8080/"} id="e154a4ce" outputId="9dd469c1-ed38-4c9d-d8d3-e3df7ab5f861"
life=SULT(udd=True)
a = UDD(m=12, life=life).temporary_annuity(45, t=20)
A = UDD(m=0, life=life).whole_life_insurance(45)
P = life.gross_premium(A=A, a=a, benefit=100000) / 12
isclose(102, P, question="Q6.22")

# + [markdown] id="1f2bd9fa"
# __SOA Question 6.23__ :  (D) 44.7
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
# __SOA Question 6.24__ :  (E) 0.30
#
# For a fully continuous whole life insurance of 1 on (x), you are given:
#
# 1. L is the present value of the loss at issue random variable if the premium rate is
# determined by the equivalence principle
# 2. L^* is the present value of the loss at issue random variable if the premium rate is 0.06
# 3. $\delta = 0.07$
# 4. $\overline{A}_x = 0.30$
# 5. $Var(L) = 0.18$
#
# Calculate $Var(L^*)$.
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
# __SOA Question 6.25__ :  (C) 12330
#
# For a fully discrete 10-year deferred whole life annuity-due of 1000 per month on (55), you are given:
# 1. The premium, $G$, will be paid annually at the beginning of each year during the deferral period
# 2. Expenses are expected to be 300 per year for all years, payable at the beginning of the year
# 3. Mortality follows the Standard Ultimate Life Table
# 4. $i = 0.05$
# 5. Using the two-term Woolhouse approximation, the expected loss at issue is -800
#
# Calculate $G$.
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
# __SOA Question 6.26__ :  (D) 180
#
# For a special fully discrete whole life insurance policy of
# 1000 on (90), you are given:
#
# 1. The first year premium is 0
#
# 2. *P* is the renewal premium
#
# 3. Mortality follows the Standard Ultimate Life Table
#
# 4. *i* = 0.05
#
# 5. Premiums are calculated using the equivalence principle
#
# Calculate *P*.
#

# + colab={"base_uri": "https://localhost:8080/"} id="e0bc9ac7" outputId="8ac47789-329d-4c0b-b0d3-9248ca8d4fd5"
life = SULT().set_interest(i=0.05)
def fun(P): 
    return P - life.net_premium(90, b=1000, initial_cost=P)
P = life.solve(fun, target=0, grid=[150, 190])
isclose(180, P, question="Q6.26")

# + [markdown] id="984c9535"
# __SOA Question 6.27__ :  (D) 10310
#
# For a special fully continuous whole life insurance on (x), you are given:
#
# 1. Premiums and benefits:
#
# | | First 20 years | After 20 years |
# | --- | --- | --- |
# | Premium Rate | 3P | P |
# | Benefit | 1,000,000 | 500,000 |
#
#
# 2. $\mu_{x+t} = 0.03, \quad t \ge 0$
# 3. $\delta = 0.06$
#
# Calculate $P$ using the equivalence principle.

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
# __SOA Question 6.28__ :  (B) 36
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
# __SOA Question 6.29__ :  (B) 20.5
#
#
# (35) purchases a fully discrete whole life insurance policy of 100,000.
# You are given:
# 1. The annual gross premium, calculated using the equivalence principle, is 1770
# 2. The expenses in policy year 1 are 50% of premium and 200 per policy
# 3. The expenses in policy years 2 and later are 10% of premium and 50 per policy
# 4. All expenses are incurred at the beginning of the policy year
# 5. $i = 0.035$
#
# Calculate $\ddot{a}_{35}$.

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
# __SOA Question 6.30__ :  (A) 900
#
# For a fully discrete whole life insurance of 100 on (x), you are given:
# 1. The first year expense is 10% of the gross annual premium
# 2. Expenses in subsequent years are 5% of the gross annual premium
# 3. The gross premium calculated using the equivalence principle is 2.338
# 4. $i = 0.04$
# 5. $\ddot{a}_x = 16.50$
# 6. $^2A_x = 0.17$
#
# Calculate the variance of the loss at issue random variable.
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
# __SOA Question 6.31__ :  (D) 1330
#
#
# For a fully continuous whole life insurance policy of 100,000 on (35), you are given:
# 1. The density function of the future lifetime of a newborn:
# $$\begin{align*}
# f(t) & = 0.01 e^{-0.01 t}, \quad 0 \le t < 70\\
# & = g(t), \quad t \ge 70
# \end{align*}$$
#
# 2. $\delta = 0.05$
#
# 3. $\overline{A}_{70} = 0.51791$
#
# Calculate the annual net premium rate for this policy.

# + colab={"base_uri": "https://localhost:8080/"} id="2dfd7470" outputId="da072e23-ddf1-4a4b-8c17-ae2c11ae0069"
life = ConstantForce(mu=0.01).set_interest(delta=0.05)
A = (life.term_insurance(35, t=35, discrete=False) 
     + life.E_x(35, t=35)*0.51791)     # A_35
P = life.premium_equivalence(A=A, b=100000, discrete=False)
isclose(1330, P, question="Q6.31")

# + [markdown] id="9876aca3"
# __SOA Question 6.32__ :  (C) 550
#
# For a whole life insurance of 100,000 on (x), you are given:
# 1. Death benefits are payable at the moment of death
# 2. Deaths are uniformly distributed over each year of age
# 3. Premiums are payable monthly
# 4. $i = 0.05$
# 5. $\ddot{a}_x = 9.19$
#
# Calculate the monthly net premium.
#

# + colab={"base_uri": "https://localhost:8080/"} id="9775a2e0" outputId="c93d4c69-b676-4b51-b093-82b48840c969"
x = 0
life = Recursion().set_interest(i=0.05).set_a(9.19, x=x)
benefits = UDD(m=0, life=life).whole_life_insurance(x)
payments = UDD(m=12, life=life).whole_life_annuity(x)
P = life.gross_premium(a=payments, A=benefits, benefit=100000)/12
isclose(550, P, question="Q6.32")

# + [markdown] id="3765e3c2"
# __SOA Question 6.33__ :  (B) 0.13
#
#
# An insurance company sells 15-year pure endowments of 10,000 to 500 lives, each age x, with independent future lifetimes. The single premium for each pure endowment is determined by the equivalence principle.
#
# You are given:
# 1. $i$ = 0.03
# 2. $\mu_x(t) = 0.02 t, \quad t \ge 0$
# 3. $_0L$ is the aggregate loss at issue random variable for these pure endowments.
#
# Using the normal approximation without continuity correction, calculate $Pr(_0L) > 50,000)$.

# + colab={"base_uri": "https://localhost:8080/"} id="5410107c" outputId="a40a6c88-d79e-4ea5-f08e-6b236286457d"
life = Insurance().set_survival(mu=lambda x,t: 0.02*t).set_interest(i=0.03)
x = 0
var = life.E_x(x, t=15, moment=life.VARIANCE, endowment=10000)
p = 1- life.portfolio_cdf(mean=0, variance=var, value=50000, N=500)
isclose(0.13, p, question="Q6.33", rel_tol=0.02)

# + [markdown] id="d47dfed4"
# __SOA Question 6.34__ :  (A) 23300
#
# For a fully discrete whole life insurance policy on (61), you
# are given:
#
# 1. The annual gross premium using the equivalence principle is 500
#
# 2. Initial expenses, incurred at policy issue, are 15% of the
#  premium
#
# 3. Renewal expenses, incurred at the beginning of each year after
# the first, are 3% of the premium
#
# 4. Mortality follows the Standard Ultimate Life Table
#
# 5. *i* = 0.05
#
# Calculate the amount of the death benefit.
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
# __SOA Question 6.35__ :  (D) 530
#
# For a fully discrete whole life insurance policy of 100,000 on (35), you are given:
# 1. First year commissions are 19% of the annual gross premium
# 2. Renewal year commissions are 4% of the annual gross premium
# 3. Mortality follows the Standard Ultimate Life Table
# 4. $i = 0.05$
#
# Calculate the annual gross premium for this policy using the equivalence principle.
#

# + colab={"base_uri": "https://localhost:8080/"} id="2079db39" outputId="26c60fcf-c5e0-4ece-dfdf-6138ebc4886e"
sult = SULT()
A = sult.whole_life_insurance(35, b=100000)
a = sult.whole_life_annuity(35)
P = sult.gross_premium(a=a, A=A, initial_premium=.19, renewal_premium=.04)
isclose(530, P, question="Q6.35")

# + [markdown] id="c0b919c2"
# __SOA Question 6.36__ :  (B) 500
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
# __SOA Question 6.37__ :  (D) 820
#
# For a fully discrete whole life insurance policy of 50,000 on (35), with premiums payable for a maximum of 10 years, you are given:
#
# 1. Expenses of 100 are payable at the end of each year including the year of death
# 2. Mortality follows the Standard Ultimate Life Table
# 3. $i = 0.05$
#
# Calculate the annual gross premium using the equivalence principle.
#

# + colab={"base_uri": "https://localhost:8080/"} id="fa96592b" outputId="21e8c04c-d45f-4a90-9dc1-a718d7a908ec"
sult = SULT()
benefits = sult.whole_life_insurance(35, b=50000 + 100)
expenses = sult.immediate_annuity(35, b=100)
a = sult.temporary_annuity(35, t=10)
P = (benefits + expenses) / a
isclose(820, P, question="Q6.37")

# + [markdown] id="07e0a134"
# __SOA Question 6.38__ :  (B) 11.3
#
# For an n-year endowment insurance of 1000 on (x), you are given:
# 1. Death benefits are payable at the moment of death
# 2. Premiums are payable annually at the beginning of each year
# 3. Deaths are uniformly distributed over each year of age
# 4. $i = 0.05$
# 5. $_nE_x = 0.172$
# 6. $\overline{A}_{x:\overline{n|}} = 0.192$
#
# Calculate the annual net premium for this insurance.
#

# + colab={"base_uri": "https://localhost:8080/"} id="017b6427" outputId="537f2fb4-781e-4db9-fa08-5c5bfddb0ef1"
x, n = 0, 10
life = Recursion().set_interest(i=0.05)\
                  .set_A(0.192, x=x, t=n, endowment=1, discrete=False)\
                  .set_E(0.172, x=x, t=n)
a = life.temporary_annuity(x, t=n, discrete=False)

def fun(a):   # solve for discrete annuity, given continuous
    life = Recursion(verbose=False).set_interest(i=0.05)\
                                   .set_a(a, x=x, t=n)\
                                   .set_E(0.172, x=x, t=n)
    return UDD(m=0, life=life).temporary_annuity(x, t=n)
a = life.solve(fun, target=a, grid=a)  # discrete annuity
P = life.gross_premium(a=a, A=0.192, benefit=1000)
isclose(11.3, P, question="Q6.38")

# + [markdown] id="801638b7"
# __SOA Question 6.39__ :  (A) 29
#
# XYZ Insurance writes 10,000 fully discrete whole life insurance policies of 1000 on lives age 40 and an additional 10,000 fully discrete whole life policies of 1000 on lives age 80.
#
# XYZ used the following assumptions to determine the net premiums for these policies:
#
# 1. Mortality follows the Standard Ultimate Life Table
# 2. i = 0.05
#
# During the first ten years, mortality did follow the Standard Ultimate Life Table.
#
# Calculate the average net premium per policy in force received at the beginning of the eleventh year.
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
# __SOA Question 6.40__ : (C) 116 
#
# For a special fully discrete whole life insurance, you are given:
#
# 1. The death benefit is $1000(1.03)^k$ for death in policy year k, for $k = 1, 2, 3...$
# 2. $q_x = 0.05$
# 3. $i = 0.06$
# 4. $\ddot{a}_{x+1} = 7.00$
# 5. The annual net premium for this insurance at issue age x is 110
#
# Calculate the annual net premium for this insurance at issue age $x + 1$.
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
# __SOA Question 6.41__ :  (B) 1417
#
#
# For a special fully discrete 2-year term insurance on (x), you are given:
# 1. $q_x = 0.01$
# 2. $q_{x + 1} = 0.02$
# 3. $i = 0.05$
# 4. The death benefit in the first year is 100,000
# 5. Both the benefits and premiums increase by 1% in the second year
#
# Calculate the annual net premium in the first year.

# + colab={"base_uri": "https://localhost:8080/"} id="a76e5f76" outputId="a7e2d0ff-e1d9-4685-9f04-32b9557003bc"
x = 0
life = LifeTable().set_interest(i=0.05).set_table(q={x:.01, x+1:.02})
a = 1 + life.E_x(x, t=1) * 1.01
A = life.deferred_insurance(x, u=0, t=1) + 1.01*life.deferred_insurance(x, u=1, t=1)
P = 100000 * A / a
isclose(1417, P, question="Q6.41")

# + [markdown] id="88c94cf0"
# __SOA Question 6.42__ :  (D) 0.113
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
# __SOA Question 6.43__ :  (C) 170
#
# For a fully discrete, 5-payment 10-year term insurance of 200,000 on (30), you are given:
# 1. Mortality follows the Standard Ultimate Life Table
# 2. The following expenses are incurred at the beginning of each respective year:
#
# | | Percent of Premium | Per Policy | Percent of Premium | Per Policy |
# |---|---|---|---|---|
# | | Year 1 | Year 1 | Years 2 - 10 | Years 2 - 10 |
# | Taxes | 5% | 0 | 5% | 0 |
# | Commissions | 30% | 0 | 10% | 0 |
# | Maintenance | 0% | 8 | 0% | 4 |
#
# 3. i = 0.05
# 4. $\ddot{a}_{30:\overline{5|}} = 4.5431$
#
# Calculate the annual gross premium using the equivalence principle.
#
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
# __SOA Question 6.44__ :  (D) 2.18
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
# __SOA Question 6.45__ :  (E) 690
#
# For a fully continuous whole life insurance of 100,000 on
# (35), you are given:
#
# 1. The annual rate of premium is 560
#
# 2. Mortality follows the Standard Ultimate Life Table
#
# 3. Deaths are uniformly distributed over each year of age
#
# 4. *i* = 0.05
#
# Calculate the 75th percentile of the loss at issue random variable for
# this policy.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 522} id="e434ceb8" outputId="36a56753-5525-44d3-8341-84f3645c71e4"
life = SULT(udd=True)
contract = Contract(benefit=100000, premium=560, discrete=False)
L = life.L_from_prob(x=35, prob=0.75, contract=contract)
life.L_plot(x=35, contract=contract, 
            T=life.L_to_t(L=L, contract=contract))
isclose(690, L, question="Q6.45")

# + [markdown] id="96fbe650"
# __SOA Question 6.46__ :  (E) 208
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
# __SOA Question 6.47__ :  (D) 66400
#
# For a 10-year deferred whole life annuity-due with payments of 100,000 per year on (70), you are given:
# 1. Annual gross premiums of $G$ are payable for 10 years
# 2. First year expenses are 75% of premium
# 3. Renewal expenses for years 2 and later are 5% of premium during the premium paying period
# 4. Mortality follows the Standard Ultimate Life Table
# 5. i = 0.05
#
# Calculate $G$ using the equivalence principle.
#

# + colab={"base_uri": "https://localhost:8080/"} id="5f701e65" outputId="87db7fc3-48dc-45b0-db79-6de75a5444c2"
sult = SULT()
a = sult.temporary_annuity(70, t=10)
A = sult.deferred_annuity(70, u=10)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_premium=0.75,
                        renewal_premium=0.05)
isclose(66400, P, question="Q6.47")

# + [markdown] id="7ed3e46c"
# __SOA Question 6.48__ :  (A) 3195
#
# For a special fully discrete 5-year deferred 3-year term insurance of 100,000 on (x) you are given:
# 1. There are two premium payments, each equal to P . The first is paid at the beginning of the first year and the second is paid at the end of the 5-year deferral period
# 2. $p_x = 0.95$
# 3. $q_{x + 5} = 0.02$
# 4. $q_{x + 6} = 0.03$
# 5. $q_{x + 7} = 0.04$
# 6. $i = 0.06$
#
# Calculate P using the equivalence principle.
#

# + colab={"base_uri": "https://localhost:8080/"} id="022f6301" outputId="68a325b9-2d97-476d-bfc2-e0c36019dcb4"
x = 0
life = Recursion(depth=5).set_interest(i=0.06)\
                         .set_p(.95, x=x, t=5)\
                         .set_q(.02, x=x+5)\
                         .set_q(.03, x=x+6)\
                         .set_q(.04, x=x+7)
a = 1 + life.E_x(x, t=5)
A = life.deferred_insurance(x, u=5, t=3)
P = life.gross_premium(A=A, a=a, benefit=100000)
isclose(3195, P, question="Q6.48")

# + [markdown] id="3d130096"
# __SOA Question 6.49__ :  (C) 86
#
# For a special whole life insurance of 100,000 on (40), you are given:
# 1. The death benefit is payable at the moment of death
# 2. Level gross premiums are payable monthly for a maximum of 20 years
# 3. Mortality follows the Standard Ultimate Life Table
# 4. $i = 0.05$
# 5. Deaths are uniformly distributed over each year of age
# 6. Initial expenses are 200
# 7. Renewal expenses are 4% of each premium including the first
# 8. Gross premiums are calculated using the equivalence principle
#
# Calculate the monthly gross premium.
#

# + colab={"base_uri": "https://localhost:8080/"} id="c0fe3957" outputId="6593ce49-458a-4489-e5a9-edfde9e285f6"
sult = SULT(udd=True)
a = UDD(m=12, life=sult).temporary_annuity(40, t=20)
A = sult.whole_life_insurance(40, discrete=False)
P = sult.gross_premium(a=a, A=A, benefit=100000, initial_policy=200,
                       renewal_premium=0.04, initial_premium=0.04) / 12
isclose(86, P, question="Q6.49")

# + [markdown] id="c442a990"
# __SOA Question 6.50__ :  (A) -47000
#
# On July 15, 2017, XYZ Corp buys fully discrete whole life
# insurance policies of 1,000 on each
# of its 10,000 workers, all age 35. It uses the death benefits to partially pay the premiums for the following year.
#
# You are given:
#
# 1. Mortality follows the Standard Ultimate Life Table
#
# 2. *i* = 0.05
#
# 3. The insurance is priced using the equivalence principle
#
# Calculate XYZ Corp's expected net cash flow from these policies during
# July 2018.
#

# + colab={"base_uri": "https://localhost:8080/"} id="f5713876" outputId="026de178-685d-4d57-f4d4-c90fe5b5a27b"
life = SULT()
P = life.premium_equivalence(a=life.whole_life_annuity(35), b=1000) 
a = life.deferred_annuity(35, u=1, t=1)
A = life.term_insurance(35, t=1, b=1000)
cash = (A - a * P) * 10000 / life.interest.v
isclose(-47000, cash, question="Q6.50")

# + [markdown] id="8aa16e1d"
# __SOA Question 6.51__ :  (D) 34700
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
# __SOA Question 6.52__ :  (D) 50.80
#
# For a fully discrete 10-payment whole life insurance of H on (45), you are given:
# 1. Expenses payable at the beginning of each year are as follows:
#
# | Expense Type | First Year | Years 2-10 | Years 11+ |
# |---|---|---|---|
# | Per policy | 100 | 20 | 10 |
# | % of Premium |105% | 5% | 0% |
#
# 2. Mortality follows the Standard Ultimate Life Table
# 3. i = 0.05
# 4. The gross annual premium, calculated using the equivalence principle, is of the form $G = gH + f$, where $g$ is the premium rate per 1 of insurance and $f$ is the per policy fee
#
# Calculate $f$.
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
# __SOA Question 6.53__ :  (D) 720
#
# A warranty pays 2000 at the end of the year of the first failure if a washing machine fails within
# three years of purchase. The warranty is purchased with a single premium, G, paid at the time of
# purchase of the washing machine.
# You are given:
# 1. 10% of the washing machines that are working at the start of each year fail by the end of that year
# 2. *i* = 0.08
# 3. The sales commission is 35% of G
# 4. G is calculated using the equivalence principle
#
# Calculate G.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="a9d23ae6" outputId="831df386-d243-4a38-8135-aff0e29af063"
x = 0
life = LifeTable().set_interest(i=0.08).set_table(q={x:.1, x+1:.1, x+2:.1})
A = life.term_insurance(x, t=3)
P = life.gross_premium(a=1, A=A, benefit=2000, initial_premium=0.35)
isclose(720, P, question="Q6.53")

# + [markdown] id="41e939f0"
# __SOA Question 6.54__ :  (A) 25440
#
# For a fully discrete whole life insurance of 200,000 on (45),
# you are given:
#
# 1. Mortality follows the Standard Ultimate Life Table.
#
# 2. *i* = 0.05
#
# 3. The annual premium is determined using the equivalence principle.
#
# Calculate the standard deviation of
# $_0L$ , the present value random variable for the loss at issue. 
#
# [A modified version of Question 12 on the Fall 2017 exam]
#

# + colab={"base_uri": "https://localhost:8080/"} id="2ea3fc85" outputId="d1d9bcd6-0894-4605-dd25-3f4a44d69c89"
life = SULT()
std = math.sqrt(life.net_policy_variance(45, b=200000))
isclose(25440, std, question="Q6.54")

# + [markdown] id="04a31b19"
# ## 7 Policy Values

# + [markdown] id="b265fc75"
# __SOA Question 7.1__ :  (C) 11150
#
# For a special fully discrete whole life insurance on (40), you
# are given:
#
# 1. The death benefit is 50,000 in the first 20 years and 100,000
#  thereafter
#
# 2. Level net premiums of 875 are payable for 20 years
#
# 3. Mortality follows the Standard Ultimate Life Table
#
# 4. *i* = 0.05
#
#  Calculate $_{10}V$ the net premium policy value at the end of year 10
#  for this insurance.
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
# __SOA Question 7.2__ :  (C) 1152
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
# __SOA Question 7.3__ :  (E) 730
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
# __SOA Question 7.4__ :  (B) -74 
#
# For a special fully discrete whole life insurance on (40), you
# are given:
#
# 1. The death benefit is 1000 during the first 11 years and 5000
#  thereafter
#
# 2. Expenses, payable at the beginning of the year, are 100 in year
#  1 and 10 in years 2 and later
#
# 3. $\pi$ is the level annual premium, determined using the
#  equivalence principle
#
# 4. $G = 1.02 \times \pi$ is the level annual gross premium
#
# 5. Mortality follows the Standard Ultimate Life Table
#
# 6. *i* = 0.05
#
# 7. $_{11}E_{40} = 0.57949$
#
#  Calculate the gross premium policy value at the end of year 1 for this
#  insurance.
#
#  *hints:*
#
# - split benefits into two policies
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
# __SOA Question 7.5__ :  (E) 1900
#
# For a fully discrete whole life insurance of 10,000 on (*x*),
# you are given:
#
# 1. Deaths are uniformly distributed over each year of age
#
# 2. The net premium is 647.46
#
# 3. The net premium policy value at the end of year 4 is 1405.08
#
# 4. $q_{x+4}$ = 0.04561
#
# 5. *i* = 0.03
#
#  Calculate the net premium policy value at the end of 4.5 years.
#
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
# __SOA Question 7.7__ :  (D) 1110
#
# For a whole life insurance of 10,000 on (x), you are given:
# 1. Death benefits are payable at the end of the year of death
# 2. A premium of 30 is payable at the start of each month
# 3. Commissions are 5% of each premium
# 4. Expenses of 100 are payable at the start of each year
# 5. $i = 0.05$
# 6. $1000 A_{x+10} = 400$
# 7. $_{10} V$ is the gross premium policy value at the end of year 10 for this insurance
#
# Calculate $_{10} V$ using the two-term Woolhouse formula for annuities.
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
# __SOA Question 7.8__ :  (C) 29.85
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
# __SOA Question 7.9__ :  (A) 38100
#
# For a semi-continuous 20-year endowment insurance of 100,000 on (45), you are given:
# 1. Net premiums of 253 are payable monthly
# 2. Mortality follows the Standard Ultimate Life Table
# 3. Deaths are uniformly distributed over each year of age
# 4. $i = 0.05$
#
# Calculate $_{10}V$, the net premium policy value at the end of year 10 for this insurance.
#

# + colab={"base_uri": "https://localhost:8080/"} id="95ee4b66" outputId="a39768fc-33af-451c-e8f8-740877c107b4"
sult = SULT(udd=True)
x, n, t = 45, 20, 10
a = UDD(m=12, life=sult).temporary_annuity(x=x+10, t=n-t)
A = UDD(m=0, life=sult).endowment_insurance(x=x+10, t=n-t)
contract = Contract(premium=253*12, endowment=100000, benefit=100000)
V = sult.gross_future_loss(A=A, a=a, contract=contract)
isclose(38100, V, question="Q7.9")

# + [markdown] id="4f341dc3"
# __SOA Question 7.10__ : (C) -970
#
# For a fully discrete whole life insurance of 100,000 on (45),
# you are given:
#
# 1. Mortality follows the Standard Ultimate Life Table
#
# 2. *i* = 0.05
#
# 3. Commission expenses are 60% of the first year's gross premium
#  and 2% of renewal gross premiums
#
# 4. Administrative expenses are 500 in the first year and 50 in
#  each renewal year
#
# 5. All expenses are payable at the start of the year
#
# 6. The gross premium, calculated using the equivalence principle,
#  is 977.60
#
#  Calculate $_5V^e$, the expense reserve at the end of year 5 for this
#  insurance.
#

# + colab={"base_uri": "https://localhost:8080/"} id="83268a47" outputId="7de54cc6-30cb-4944-ec28-ab998b7aa9c0"
life = SULT()
G = 977.6
P = life.net_premium(45, b=100000)
contract = Contract(benefit=0, premium=G-P, renewal_policy=.02*G + 50)
V = life.gross_policy_value(45, t=5, contract=contract)
isclose(-970, V, question="Q7.10")

# + [markdown] id="55157c76"
# __SOA Question 7.11__ :  (B) 1460
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
# __SOA Question 7.12__ :  (E) 4.09
#
# For a special fully discrete 25-year endowment insurance on
# (44), you are given:
#
# 1. The death benefit is ( 26−*k* ) for death in year ,*k* for *k* =
#  1,2,3,...,25
#
# 2. The endowment benefit in year 25 is 1
#
# 3. Net premiums are level
#
# 4. $q_{55}$= 0.15
#
# 5. *i* = 0.04
#
# 6. $_{11}V$ the net premium policy value at the end of year 11, is
#  5.00
#
# 1. $_{24}V$ the net premium policy value at the end of year 24, is
#  0.60
#
#  Calculate $_{12}V$ the net premium policy value at end of year 12.
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
# __SOA Question 7.13__ : (A) 180
#

# + colab={"base_uri": "https://localhost:8080/"} id="0b8778cc" outputId="e86b40bf-745e-4d08-a5ae-bc70e33c4c37"
life = SULT()
V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
isclose(180, V, question="Q7.13")

# + [markdown] id="58f053bd"
# __SOA Question 7.14__ :  (A) 2200
#
# For a fully discrete whole life insurance of 100,000 on (45),
# you are given:
#
# 1. The gross premium policy value at duration 5 is 5500 and at
#  duration 6 is 7100
#
# 2. $q_{50}$ = 0.009
#
# 3. *i* = 0.05
#
# 4. Renewal expenses at the start of each year are 50 plus 4% of the gross premium.
#
# 5. Claim expenses are 200.
#
#  Calculate the gross premium.
#
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
# __SOA Question 7.15__ :  (E) 50.91
#
# For a fully discrete whole life insurance of 100 on (*x*), you
# are given:
#
# 1. $q_{x+ 15} = 0.10$
#
# 2. Deaths are uniformly distributed over each year of age
#
# 3. *i* = 0.05
#
# 4. $_tV$ denotes the net premium policy value at time *t*
#
# 5. $_{16}V$ = 49.78
#
#  Calculate 15.6.
#

# + colab={"base_uri": "https://localhost:8080/"} id="bbed6a97" outputId="5ceafecc-598b-465b-b1b5-a509c56afca5"
x = 0
V = Recursion(udd=True).set_interest(i=0.05)\
                       .set_q(0.1, x=x+15)\
                       .set_reserves(T=3, V={16: 49.78})\
                       .r_V_backward(x, s=15, r=0.6, benefit=100)
isclose(50.91, V, question="Q7.15")

# + [markdown] id="cf793972"
# __SOA Question 7.16__ :  (D) 380
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
# __SOA Question 7.17__ :  (D) 1.018
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
# __SOA Question 7.18__ :  (A) 17.1
#
# For a fully discrete whole life insurance of 1 on (*x*), you
# are given:
#
# 1. The net premium policy value at the end of the first year is 0.012
#
# 2. $q_x$ = 0.009
#
# 3. *i* = 0.04
#
# Calculate $\ddot{a}_x$
#

# + colab={"base_uri": "https://localhost:8080/"} id="789aef65" outputId="08fc06d5-23b0-47b8-cab4-464ce5900496"
x = 10
life = Recursion(verbose=False).set_interest(i=0.04).set_q(0.009, x=x)
def fun(a):
    return life.set_a(a, x=x).net_policy_value(x, t=1)
a = life.solve(fun, target=0.012, grid=[17.1, 19.1])
isclose(17.1, a, question="Q7.18")

# + [markdown] id="bcd7d9ae"
# __SOA Question 7.19__ :  (D) 720
#
#  For a fully discrete whole life insurance of 100,000 on (40)
# you are given:
#
# 1. Expenses incurred at the beginning of the first year are 300
#  plus 50% of the first year premium
#
# 2. Renewal expenses, incurred at the beginning of the year, are
#  10% of each of the renewal premiums
#
# 3. Mortality follows the Standard Ultimate Life Table
#
# 4. *i* = 0.05
#
# 5. Gross premiums are calculated using the equivalence principle
#
# Calculate the gross premium policy value for this insurance immediately
# after the second premium and associated renewal expenses are paid.
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
# __SOA Question 7.20__ : (E) -277.23
#
# For a fully discrete whole life insurance of 1000 on (35), you
# are given:
#
# 1. First year expenses are 30% of the gross premium plus 300
#
# 2. Renewal expenses are 4% of the gross premium plus 30
#
# 3. All expenses are incurred at the beginning of the policy year
#
# 4. Gross premiums are calculated using the equivalence principle
#
# 5. The gross premium policy value at the end of the first policy
#  year is *R*
#
# 6. Using the Full Preliminary Term Method, the modified reserve at
#  the end of the first policy year is *S*
#
# 7. Mortality follows the Standard Ultimate Life Table
#
# 8. *i* = 0.05
#
# Calculate *R*−*S* .
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
# __SOA Question 7.21__ :  (D) 11866
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
# __SOA Question 7.22__ :  (C) 46.24
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
# __SOA Question 7.23__ :  (D) 233
#

# + colab={"base_uri": "https://localhost:8080/"} id="c4c42da2" outputId="e4ea3f12-649d-4e2f-b614-dcb99e226f4d"
life = Recursion().set_interest(i=0.04).set_p(0.995, x=25)
A = life.term_insurance(25, t=1, b=10000)
def fun(beta):  # value of premiums in first 20 years must be equal
    return beta * 11.087 + (A - beta) 
beta = life.solve(fun, target=216 * 11.087, grid=[140, 260])
isclose(233, beta, question="Q7.23")

# + [markdown] id="4b82caf4"
# __SOA Question 7.24__ :  (C) 680
#
# For a fully discrete whole life insurance policy of 1,000,000
# on (50), you are given:
#
# 1. The annual gross premium, calculated using the equivalence
#  principle, is 11,800
#
# 2. Mortality follows the Standard Ultimate Life Table
#
# 3. *i* = 0.05
#
#  Calculate the expense loading, *P* for this policy.
#

# + colab={"base_uri": "https://localhost:8080/"} id="75d8d20b" outputId="4a1e843e-c342-4b5b-b9fa-a8b11d772431"
life = SULT()
P = life.premium_equivalence(A=life.whole_life_insurance(50), b=1000000)
isclose(680, 11800 - P, question="Q7.24")

# + [markdown] id="8410e40c"
# __SOA Question 7.25__ :  (B) 3947.37
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
# __SOA Question 7.26__ :  (D) 28540 
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
# __SOA Question 7.27__ :  (B) 213
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
# __SOA Question 7.28__ :  (D) 24.3
#

# + colab={"base_uri": "https://localhost:8080/"} id="99412e64" outputId="87fc6efa-2bb9-4bc5-f91d-b74731102985"
life = SULT()
PW = life.net_premium(65, b=1000)   # 20_V=0 => P+W is net premium for A_65
P = life.net_premium(45, t=20, b=1000)  # => P is net premium for A_45:20
isclose(24.3, PW - P, question="Q7.28")

# + [markdown] id="04bd97d2"
# __SOA Question 7.29__ :  (E) 2270
#
# For a fully discrete whole life insurance of *B* on  *(x)*,
# you are given:
#
# 1. Expenses, incurred at the beginning of each year, equal 30 in
#  the first year and 5 in subsequent years
#
# 2. The net premium policy value at the end of year 10 is 2290
#
# 3. Gross premiums are calculated using the equivalence principle
#
# 4. *i* = 0.04
#
# 5. $\ddot{a}_x$ = 14.8
#
# 6. $\ddot{a}_{x+10}$ = 11.4
#
#  Calculate $_{10}V^{g}$, the gross premium policy value at the end of year
#  10.
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
# __SOA Question 7.30__ :  (E) 9035
#
# Ten years ago J, then age 25, purchased a fully discrete
# 10-payment whole life policy of 10,000.
#
#  All actuarial calculations for this policy were based on the
#  following:
#
# 1. Mortality follows the Standard Ultimate Life Table
#
# 2. *i* = 0.05
#
# 3. The equivalence principle
#
#  In addition:
#
# 1. $L_{10}$ is the present value of future losses random variable at
#  time 10
#
# 2. At the end of policy year 10, the interest rate used to
#  calculate $L_{10}$ is changed to 0%
#
#  Calculate the increase in $E[L_{10}]$ that results from this change.
#
#

# + colab={"base_uri": "https://localhost:8080/"} id="cdd25b58" outputId="27b419e7-2e45-4c15-b867-a7c22fbb8c08"
b = 10000  # premiums=0 after t=10
L = SULT().set_interest(i=0.05).whole_life_insurance(x=35, b=b)
V = SULT().set_interest(i=0).whole_life_insurance(x=35, b=b)
isclose(9035, V - L, question="Q7.30")

# + [markdown] id="df03679b"
# __SOA Question 7.31__ :  (E) 0.310
#
# For a fully discrete 3-year endowment insurance of 1000 on (x), you are given:
# 1.  Expenses, payable at the beginning of the year, are:
#
# | Year(s) | Percent of Premium | Per Policy |
# |---|---|---|
# | 1 | 20% | 15 |
# | 2 and 3 | 8% | 5 |
#
# 2. The expense reserve at the end of year 2 is –23.64
# 3. The gross annual premium calculated using the equivalence principle is G = 368.
# 4. $G = 1000 P_{x:\overline{3|}} + P^e$ , where $P^e$ is the expense loading
#
# Calculate $P_{x:\overline{3|}}$ .
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
# __SOA Question 7.32__ :  (B) 1.4
#
# For two fully continuous whole life insurance policies on (x), you are given:
#
# | | Death Benefit | Annual Premium Rate | Variance of the PV of Future Loss at t |
# |---|---|---|---|
# | Policy A | 1 | 0.10 | 0.455 |
# | Policy B | 2 | 0.16 | - |
#
# - $\delta= 0.06$
#
# Calculate the variance of the present value of future loss at $t$ for Policy B.
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
