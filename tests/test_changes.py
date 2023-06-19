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

# June 8, 2023
"""1.1102230246251565e-16
66.41315159665598
0.00020973561925718975
65.91315159665601"""

life = SULT().set_interest(i=0)
for discrete in [True, False]: # divide by zero
    for variance in [True, False]:
        print(life.whole_life_annuity(20, discrete=discrete, variance=variance))
print()

"""2.220446049250313e-16
26.30938051545656
0.00011454063198124143
26.148627042747307"""

for discrete in [True, False]: # divide by zero
    for variance in [True, False]:
        print(life.temporary_annuity(60, t=30, discrete=discrete,
                                     variance=variance))
print()

"""0 False inf
0 False 0.0
0 True 111.0
0 True 0.0
1 False 1.0
1 False 1.0
1 True 1.5819767068693262
1 True 0.9999999999999999
9034.654127845053"""

for mu in [0,1]:
    for discrete in [False, True]:
        life = ConstantForce(mu=mu)
        print(mu, discrete, life.whole_life_annuity(20, discrete=discrete))
        print(mu, discrete, life.whole_life_insurance(20, discrete=discrete))

b = 10000  # premiums=0 after t=10
L = SULT().set_interest(i=0.05).whole_life_insurance(x=35, b=b)
V = SULT().set_interest(i=0).whole_life_insurance(x=35, b=b)
print(V-L)
        
