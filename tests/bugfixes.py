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
        
