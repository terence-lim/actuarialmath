"""
FAML
Survival
  Lifetime
  Insurance
  Annuity
  Premiums
  PolicyValues
  Reserves
Fractional
MortalityLaws
  ConstantForce
LifeTable
  SULT
  Select
Recursion
Mthly
  UDD
  Woolhouse
AdjustMortality
Appendix
"""
import matplotlib.pyplot as plt
import numpy as np
from faml import FAML
from survival import Survival
from lifetime import Lifetime
from insurance import Insurance
from annuity import Annuity
from premiums import Premiums
from policyvalues import PolicyValues
from reserves import Reserves
from fractional import Fractional
from recursion import Recursion
from lifetable import LifeTable
from sult import SULT
from selectlife import Select
from constantforce import ConstantForce
from mortalitylaws import Beta, Uniform, Makeham, Gompertz
from adjustmortality import Adjust
from mthly import Mthly
from udd import UDD
from woolhouse import Woolhouse

if __name__ == "__main__":
    # SOA Question 4.8  (C) 191
    from sult import SULT
    v_t = lambda t: 1.04**(-t) if t < 1 else 1.04**(-1) * 1.05**(-t+1)
    life = SULT(interest=dict(v_t=v_t))
    print(life.whole_life_insurance(50, b=1000))

    # SOA Question 4.1:  (A) 0.27212
    from recursion import Recursion
    life = Recursion(interest=dict(i=0.03))
    life.set_A(0.36987, x=40)
    life.set_A(0.62567, x=60)
    life.set_E(0.51276, x=40, t=20)
    life.set_E(0.17878, x=60, t=20)
    Z2 = 0.24954
    A = (2 * life.term_insurance(40, t=20) 
         + life.deferred_insurance(40, u=20))
    print(math.sqrt(life.insurance_variance(A2=Z2, A1=A)))


    
    # SOA Question 4.6:  (B) 29.85
    from sult import SULT
    sult = SULT()
    life = LifeTable(q={70+k: .95**k * sult.q_x(70+k) for k in range(3)}).fill()
    print(life.term_insurance(70, t=3, b=1000))

    # SOA Question 7.28:  (D) 24.3
    life = SULT()
    PW = life.net_premium(65, b=1000)   # 20_V=0 => P+W is net premium for A_65
    P = life.net_premium(45, t=20, b=1000)  # => P is net premium for A_45:20
    print(PW - P)

    # SOA Question 7.24:  (C) 680
    life = SULT()
    P = life.premium_equivalence(A=life.whole_life_insurance(50), b=1000000)
    print(11800 - P)

    # SOA Question 6.3:  (C) 0.390
    life = SULT()
    P = life.net_premium(45, u=20, annuity=True)
    t = life.Y_to_t(life.whole_life_annuity(65))
    print(1 - life.p_x(65, t=math.floor(t) - 1))

    # SOA Question 6.50:  (A) -47000
    life = SULT()
    P = life.premium_equivalence(a=life.whole_life_annuity(35), b=1000) 
    a = life.deferred_annuity(35, u=1, t=1)
    A = life.term_insurance(35, t=1, b=1000)
    cash = (A - a * P) * 10000 / life.interest.v
    print(a, A, P, cash)

    # SOA Question 6.34:  (A) 23300
    life = SULT()
    def fun(benefit):
        A = life.whole_life_insurance(61)
        a = life.whole_life_annuity(61)
        return life.gross_premium(A=A, a=a, benefit=benefit, initial_premium=0.15,
                                  renewal_premium=0.03) - 500
    print(fun(23294))
    print(life.solve(fun, guess=[23300, 23700]))

    # SOA Question 6.29  (B) 20.5
    life = Premiums(interest=dict(i=0.035))
    def fun(a):
        return life.gross_premium(A=life.insurance_twin(a=a),
                                  a=a, benefit=100000, 
                                  initial_policy=200, initial_premium=.5,
                                  renewal_policy=50, renewal_premium=.1) - 1770
    print(life.solve(fun, [20, 22]))

    # SOA Question 6.28  (B) 36
    life = SULT(interest=dict(i=0.05))
    a = life.temporary_annuity(40, t=5)
    A = life.whole_life_insurance(40)
    P = life.gross_premium(a=a, A=A, benefit=1000, 
                           initial_policy=10, renewal_premium=.05,
                           renewal_policy=5, initial_premium=.2)
    print(P)

    # SOA Question 6.26  (D) 180
    life = SULT(interest=dict(i=0.05))
    P = 180
    def fun(P):
        return P - life.net_premium(90, b=1000, initial_cost=P)
    P = life.solve(fun, guess=[150, 190])
    print(P)

    # SOA Question 6.14  (D) 1150
    life = SULT(interest=dict(i=0.05))
    def fun(P):
        A = (life.term_insurance(40, t=20, b=P)
             + life.deferred_annuity(40, u=20, b=30000))
        return life.gross_premium(a=1, A=A) - P
    P = life.solve(fun, [162000, 168800])
    print(P)

    # SOA Question 6.14  (D) 1150
    life = SULT(interest=dict(i=0.05))
    a = (life.temporary_annuity(40, t=10)
         + 0.5 * life.deferred_annuity(40, u=10, t=10))
    A = life.whole_life_insurance(40)
    P = life.gross_premium(a=a, A=A, benefit=100000)
    print(P)

    # SOA Question 6.9:  (D) 647
    life = SULT()
    a = life.temporary_annuity(50, t=10)
    A = life.term_insurance(50, t=20)
    costs = 25 * life.deferred_annuity(50, u=10, t=10)
    P = life.gross_premium(a=a, A=A, benefit=100000,
                           initial_premium=0.42, renewal_premium=0.12,
                           initial_policy=75 + costs, renewal_policy=25)
    print(P, a, A, costs)

    # SOA Question 6.8:  (B) 9.5
    life = SULT()
    initial_cost = (50 + 10 * life.deferred_annuity(60, u=1, t=9)
                    + 5 * life.deferred_annuity(60, u=10, t=10))
    P = life.net_premium(60, initial_cost=initial_cost)
    print(P)



    # SOA Question 6.7:  (C) 2880
    life=SULT()
    a = life.temporary_annuity(40, t=20) 
    A = life.E_x(40, t=20)
    IA = a - life.interest.annuity(t=20) * life.p_x(40, t=20)
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=100000)
    print(P, IA, A, a)


    # SOA Question 6.5:  (D) 33
    life = SULT()
    P = life.net_premium(30, b=1000)
    def fun(k):
        return (life.Y_x(30, t=k) * P
                - life.Z_x(30, t=k) * 1000)
    print(min([k for k in range(20, 40) if fun(k) < 0]))


    # SOA Question 6.2: (E) 3604
    life = Premiums()
    A, IA, a = 0.17094, 0.96728, 6.8865
    print(life.gross_premium(a=a, A=A, IA=IA, benefit=100000,
                             initial_premium=0.5, renewal_premium=.05,
                             renewal_policy=200, initial_policy=200))

    # SOA Question 6.1: (D) 35.36
    life = SULT(interest=dict(i=0.03))
    print(life.net_premium(80, t=2, b=1000, return_premium=True))

    # SOA Question 6.16: (A) 2408.6
    life = Premiums(interest=dict(d=0.05))
    A = life.insurance_from_net(premium=2143, b=100000)
    a = life.annuity_from_net(premium=2143, b=100000)
    p = life.gross_premium(A=A, a=a, benefit=100000, settlement_policy=0,
                           initial_policy=250, initial_premium=.04+.35,
                           renewal_policy=50, renewal_premium=.04+.02) 
    print(A, a, p)

    # SOA Question 6.20:  (B) 459
    life = Premiums(interest=dict(i=0.04), 
                    l=lambda x,s: dict(zip([75, 76, 77, 78],
                                       np.cumprod([1,.9,.88,.85]))).get(x+s, 0))
    a = life.temporary_annuity(75, t=3)
    IA = life.increasing_insurance(75, t=2)
    A = life.deferred_insurance(75, u=2, t=1)
    print(life.solve(lambda P: P*IA + A*10000 - P*a, 100))

    # SOA Question 6.45:  (E) 690
    life = SULT(udd=True)
    t = life.Z_t(35, 0.75, discrete=False)
    policy = life.Policy(benefit=100000, premium=560, discrete=False)
    p = life.L_from_prob(35, prob=0.75, policy=policy)
    print(p, t)



    # SOA Question 6.6:  (B) 0.79
    life = SULT()
    P = life.net_premium(62, b=10000)
    print(P)
    policy = life.Policy(premium=1.03*P, renewal_policy=5,
                         initial_policy=5, initial_premium=0.05, benefit=10000)
    L = life.gross_policy_value(62, policy=policy)
    var = life.gross_policy_variance(62, policy=policy)
    print(life.portfolio_cdf(mean=L, variance=var, value=40000, N=600))

    # SOA Question 6.12:  (E) 88900
    life = PolicyValues(interest=dict(i=0.06))
    a = 12
    A = life.insurance_twin(a)
    policy = life.Policy(benefit=1000, settlement_policy=20, 
                         initial_policy=10, initial_premium=0.75, 
                         renewal_policy=2, renewal_premium=0.1)
    policy.premium = life.gross_premium(A=A, a=a, **policy.terms)
    print(A, policy.premium)
    L = life.gross_variance_loss(A1=A, A2=0.14, policy=policy)
    print(L)

    # SOA Question 6.13:  (D) -400
    life = SULT(interest=dict(i=0.05))
    A = life.whole_life_insurance(45)
    policy = life.Policy(benefit=10000, initial_premium=.8, renewal_premium=.1)
    def fun(P):   # Solve for premium, given Loss(t=0) = 4953
        return life.L_from_t(t=10.5, policy=policy.set(premium=P)) - 4953
    policy.premium = life.solve(fun, 100)
    L = life.gross_policy_value(45, policy=policy)
    print(L)

    # SOA Question 6.19:  (B) 0.033
    life = SULT()
    policy = life.Policy(initial_policy=.2, renewal_policy=.01)
    a = life.whole_life_annuity(50)
    A = life.whole_life_insurance(50)
    policy.premium = life.gross_premium(A=A, a=a, **policy.terms)
    L = life.gross_policy_variance(50, policy=policy)
    print(L)

    # SOA Question 6.24:  (E) 0.30
    x = 0
    A1 = 0.30
    life = PolicyValues(interest=dict(delta=0.07))
    P = life.premium_equivalence(A=A1, discrete=False)
    def fun(A2):  # Solve for A2 from given Var(Loss)
        policy=life.Policy(premium=P, discrete=False)
        return 0.18 - life.gross_variance_loss(A1=A1, A2=A2, policy=policy)
    A2 = life.solve(fun, guess=0.18)
    policy = life.Policy(premium=0.06, discrete=False)
    variance = life.gross_variance_loss(A1=A1, A2=A2, policy=policy)
    print(variance)

    # SOA Question 6.30:  (A) 900
    life = PolicyValues(interest=dict(i=0.04))
    policy = life.Policy(premium=2.338, benefit=100, initial_premium=.1,
                         renewal_premium=0.05)
    var = life.gross_variance_loss(A1=life.insurance_twin(16.50),
                                   A2=0.17, policy=policy)
    print(var)

    # SOA Question 6.42:  (D) 0.113
    x = 0
    life = ConstantForce(interest=dict(delta=0.06), mu=0.06)
    policy = life.Policy(premium=315.8, T=3, endowment=1000, benefit=1000)
    L = [life.L_from_t(t, policy=policy) for t in range(3)]
    print(L)
    Q = [life.q_x(x, u=u, t=1) for u in range(3)]
    Q[-1] = 1 - sum(Q[:-1])  # follows SOA Solution incorrect treat endowment!
    p = sum([q for (q, l) in zip (Q, L) if l > 0])
    print(p)

    # SOA Question 6.54:  (A) 25440
    life = SULT()
    s = math.sqrt(life.net_policy_variance(45, b=200000))
    print(s)


    # SOA Question 7.1:  (C) 11150
    life = SULT()
    P = life.gross_premium(a=life.whole_life_annuity(40),
                           A=life.whole_life_insurance(40),
                           initial_policy=100, renewal_policy=10,
                           benefit=1000)
    P += life.gross_premium(a=life.whole_life_annuity(40),
                            A=life.deferred_insurance(40, u=11),
                            benefit=4000)
    print(P)
    policy = life.Policy(benefit=1000, premium=1.02*P, 
                         renewal_policy=10, initial_policy=100)
    A = life.whole_life_insurance(41)         # WL twin shortcut
    V = life.gross_future_loss(A=A, policy=policy.renewals)  # renewal expenses

    policy = life.Policy(benefit=4000, premium=0)
    A1 = life.deferred_insurance(41, u=10)
    V1 = life.gross_future_loss(A=A1, a=0, policy=policy)  # WL shortcut
    print(A, V, V+V1)

    # SOA Question 7.1:  (C) 11150
    life = SULT()
    x, n, t = 40, 20, 10
    A = (life.whole_life_insurance(x+t, b=50000)
         + life.deferred_insurance(x+t, u=n-t, b=50000))
    a = life.temporary_annuity(x+t, t=n-t, b=875)
    L = life.gross_future_loss(A=A, a=a)
    print(L, A, a)

    # SOA Question 7.4:  (B) -74 -- split benefits into two policies
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

    print(-74, V, 7.4)


    # SOA QUestion 7.6:  (E) -25.4
    life = SULT()
    P = life.net_premium(45, b=2000)
    print(P)
    policy = life.Policy(benefit=2000, initial_premium=.25, renewal_premium=.05,
                         initial_policy=2*1.5 + 30, renewal_policy=2*.5 + 10)
    a = life.whole_life_annuity(45)
    G = life.gross_premium(A=life.insurance_twin(a), a=a, **policy.terms)
    print(G)
    net = life.net_policy_value(45, t=10, b=2000)
    print(net)
    gross = life.gross_policy_value(45, t=10, policy=policy.set(premium=G))
    print(gross)
    V = gross - net
    print(V)

    # SOA Question 7.10: (C) -970
    life = SULT()
    G = 977.6
    P = life.net_premium(45, b=100000)
    policy = life.Policy(benefit=0, premium=G-P, renewal_policy=.02*G + 50)
    V = life.gross_policy_value(45, t=5, policy=policy)
    print(V)
    
     # SOA Question 7.11:  (B) 1460
    life=Recursion(interest=dict(i=0.05)).set_a(13.4205, x=55)
    policy=life.Policy(benefit=10000)
    def fun(P):
        return 4450 - life.L_from_t(t=10, policy=policy.set(premium=P))
    P = life.solve(fun, guess=400)
    V = life.gross_policy_value(45, t=10, policy=policy.set(premium=P))
    print(V)


    # SOA Question 7.16:  (D) 380
    life = Select(interest=dict(v=.95), A={86: [683/1000]},
                  q={80+k: [.01*(k+1)] for k in range(6)}).fill()
    x, t, n = 80, 3, 5
    A = life.whole_life_insurance(x+t)
    a = life.temporary_annuity(x+t, t=n-t)
    V = life.gross_future_loss(A=A, a=a, 
                               policy=life.Policy(benefit=1000, premium=130))
    print(A, a, V)
    # SOA Question 7.19:  (D) 720
    life = SULT()
    policy = life.Policy(benefit=100000, initial_policy=300, initial_premium=.5,
                         renewal_premium=.1)
    P = life.gross_premium(A=life.whole_life_insurance(40), **policy.terms)
    A = life.whole_life_insurance(41)
    a = life.immediate_annuity(41)   # after premium and expenses are paid
    V = life.gross_future_loss(A=A, a=a, policy=policy.set(premium=P).renewals)
    print(V)

    # SOA Question 7.22:  (C) 46.24
    life = PolicyValues(interest=dict(i=0.06))
    policy = life.Policy(benefit=8, premium=1.250)
    def fun(A2):
        return 20.55 - life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    A2 = life.solve(fun, guess=20.55/8**2)
    policy = life.Policy(benefit=12, premium=1.875)
    var = life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    print(var)

    # SOA Question 7.21:  (D) 11866
    life = SULT()
    x, t, u = 55, 9, 10
    P = life.gross_premium(IA=0.14743, a=life.temporary_annuity(x, t=u),
                           A=life.deferred_annuity(x, u=u), benefit=1000)
    policy = life.Policy(initial_policy=life.term_insurance(x+t, t=1, b=10*P),
                         premium=P, benefit=1000)
    a = life.temporary_annuity(x+t, t=u-t)
    A = life.deferred_annuity(x+t, u=u-t)
    V = life.gross_future_loss(A=A, a=a, policy=policy)
    print(a, A, V)
        

    # SOA Question 7.30:  (E) 9035
    policy = SULT.Policy(premium=0, benefit=10000)  # premiums=0 after t=10
    L = SULT().gross_policy_value(35, policy=policy)
    V = SULT(interest=dict(i=0)).gross_policy_value(35, policy=policy) # 10000
    print(L, V, V - L)

    # SOA Question 7.32:  (B) 1.4
    life = PolicyValues(interest=dict(i=0.06))
    policy = life.Policy(benefit=1, premium=0.1)
    def fun(A2):
        return 0.455 - life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    A2 = life.solve(fun, guess=0.455)
    policy = life.Policy(benefit=2, premium=0.16)
    var = life.gross_variance_loss(A1=0, A2=A2, policy=policy)
    print(var)



    # SOA Question 7.31:  (E) 0.310
    x = 0
    life = Reserves().set_reserves(T=3)
    print(life._reserves)
    G = 368.05
    def fun(P):  # solve net premium from expense reserve equation
        return -23.64 - life.t_V(x=x, t=2, premium=G-P, benefit=lambda t: 0, 
                                 per_policy=5 + .08*G)
    P = life.solve(fun, guess=[.29, .31]) / 1000
    print(P)

    # SOA Question 7.5:  (E) 1900
    x = 0
    life = Recursion(interest=dict(i=0.03), udd=True).set_q(0.04561, x=x+4)
    life.set_reserves(T=3, V={4: 1405.08})
    V = life.r_V_backward(x, s=4, r=0.5, benefit=10000, premium=647.46)
    print(V)


    # SOA Question 7.15:  (E) 50.91
    x = 0
    life = Recursion(udd=True, interest=dict(i=0.05)).set_q(0.1, x=x+15)
    life.set_reserves(T=3, V={16: 49.78})
    V = life.r_V_forward(x, s=15, r=0.6, benefit=100)
    print(V)

    # SOA Question 7.3:  (E) 730
    x = 0  # x=0 is (90) and interpret every 3 months as t=1 year
    life = LifeTable(interest=dict(i=0.08/4), 
                     l={0:1000, 1:898, 2:800, 3:706}).fill()
    life.set_reserves(T=8, V={3: 753.72})
    life.set_reserves(V={2: life.t_V_forward(x=0, t=2, premium=60*0.9, 
                                             benefit=lambda t: 1000)})
    print(life._reserves['V'][2])
    V = life.t_V_forward(x=0, t=1, premium=0, benefit=lambda t: 1000)
    print(V)

    # SOA Question 7.2:  (C) 1152
    x = 0
    life = Recursion(interest=dict(i=.1)).set_q(0.15, x=x).set_q(0.165, x=x+1)
    life.set_reserves(T=2, endowment=2000)

    def fun(P):  # solve P s.t. V is equal backwards and forwards
        policy = dict(t=1, premium=P, benefit=lambda t: 2000, reserve_benefit=True)
        return life.t_V_backward(x, **policy) - life.t_V_forward(x, **policy)
    P = life.solve(fun, [1070, 1230])
    print(P)

    # SOA Question 7.26:  (D) 28540 -- backward-forward reserve recursion
    x = 0
    life = Recursion(interest=dict(i=.05)).set_p(0.85, x=x).set_p(0.85, x=x+1)
    life.set_reserves(T=2, endowment=50000)
    benefit = lambda k: k*25000
    def fun(P):  # solve P s.t. V is equal backwards and forwards
        policy = dict(t=1, premium=P, benefit=benefit, reserve_benefit=True)
        return life.t_V_backward(x, **policy) - life.t_V_forward(x, **policy)
    P = life.solve(fun, guess=[27650, 28730])
    print(P)
#    P = life.solve(fun, [1070, 1230])
#    print(P)

    
    # SOA Question 7.14:  (A) 2200
    x = 45
    life = Recursion(interest=dict(i=0.05)).set_q(0.009, x=50)
    life.set_reserves(T=10, V={5: 5500})
    def fun(P):  # solve for net premium, from year 6 reserve
        return 7100 - life.t_V(x=x, t=6, premium=P*0.96 - 50, 
                               benefit=lambda t: 100000 + 200)
    P = life.solve(fun, guess=[2200, 2400])
    print(P)

    # SOA Question 7.12:  (E) 4.09
    benefit = lambda k: 26 - k
    x = 44
    life = Recursion(interest=dict(i=0.04)).set_q(0.15, x=55)
    life.set_reserves(T=25, endowment=1, V={11: 5.})
    def fun(P):  # solve for net premium, from final year recursion
        return 0.6 - life.t_V(x=x, t=24, premium=P, benefit=benefit)
    P = life.solve(fun, guess=0.5)
    print(P)
    V = life.t_V(x, t=12, premium=P, benefit=benefit)
    print(V)

    # SOA Question 7.8:  (C) 29.85
    sult = SULT()
    x = 70
    q = {x: [sult.q_x(x+k)*(.7 + .1*k) for k in range(3)] + [sult.q_x(x+3)]}
    life = Recursion(interest=dict(i=.05)).set_q(sult.q_x(70)*.7, x=x)\
                                          .set_reserves(T=3)
    V = life.t_V(x=70, t=1, premium=35.168, benefit=lambda t: 1000)
    print(V)


    # SOA Question 7.27:  (B) 213
    x = 0
    life = Recursion(interest=dict(i=0.03)).set_q(0.008, x=x)
    life.set_reserves(V={0: 0})
    print(life._reserves)
    print(life.p_x(x))    
    G = 212.97
    def fun(G):  # Solve gross premium from expense reserves equation
        return -38.70 - life.t_V(x=x, t=1, premium=G-187, benefit=lambda t: 0, 
                                 per_policy=10 + 0.25*G)
    G = life.solve(fun, guess=[200, 252])
    print(G)


    # SOA Question 7.25:  (B) 3947.37
    life = Select(interest=dict(i=.04), A={55: [.23, .24, .25],
                                           56: [.25, .26, .27],
                                           57: [.27, .28, .29],
                                           58: [.20, .30, .31]})
    a = life.whole_life_annuity(55, s=3)
    A = life.whole_life_insurance(55,s=1)
    V = life.FPT_policy_value(55, t=3, b=100000)
    print(a, A, V)


    # SOA Question 7.20: (E) -280
    life = SULT()
    S = life.FPT_policy_value(35, t=1, b=1000)
    policy = life.Policy(benefit=1000, initial_premium=.3, initial_policy=300,
                         renewal_premium=.04, renewal_policy=30)
    print(S)
    P = life.gross_premium(A=life.whole_life_insurance(35), **policy.terms)
    print(P)
    R = life.gross_policy_value(35, t=1, policy=policy.set(premium=P))
    print(R)

    # SOA Question 7.13: (A) 180
    life = SULT()
    V = life.FPT_policy_value(40, t=10, n=30, endowment=1000, b=1000)
    print(V)

    # SOA Question 6.17:  (A) -30000
    x = 0
    life = ConstantForce(mu=0.1, interest=dict(i=0.08))
    print(life.q_x(x), life.q_x(x+1))
    A = life.endowment_insurance(x, t=2, b=100000, endowment=30000)
    a = life.temporary_annuity(x, t=2)
    P = life.gross_premium(a=a, A=A)
    print(P, A, a)
    life1 = Recursion(interest=dict(i=0.08))
    life1.set_q(life.q_x(x, t=1) * 1.5, x=x, t=1)
    life1.set_q(life.q_x(x+1, t=1) * 1.5, x=x+1, t=1)
    A =life1.endowment_insurance(x, t=2, b=100000, endowment=30000)
    a = life1.temporary_annuity(x, t=2)
    print(A, a, A - a * P * 2)

    policy = life1.Policy(premium=P*2, benefit=100000, endowment=30000)
    L = life1.gross_policy_value(x, t=0, n=2, policy=policy)
    print(L)
    
    # SOA Question 6.48:  (A) 3195
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

    # SOA Question 4.7:  (B) 0.06
    from faml import FAML
    def fun(i):
        life = Recursion(interest=dict(i=i))
        life.set_p(0.57, x=0, t=25)
        return 0.1*life.E_x(0, t=25) - life.E_x(0, t=25, moment=life.VARIANCE)
    print(Recursion.solve(fun, 0.05))


    # SOA Question 6.23:  (D) 44.7
    x = 0
    life = Recursion().set_a(15.3926, x=x)\
                      .set_a(10.1329, x=x, t=15)\
                      .set_a(14.0145, x=x, t=30)
    def fun(P):
        per_policy = 30 + (30 * life.whole_life_annuity(x))
        per_premium = (.6 + .1 * life.temporary_annuity(x, t=15)
                        + .1 * life.temporary_annuity(x, t=30))
        a = life.temporary_annuity(x, t=30)
        print(per_policy, per_premium, per_premium - a)   
        return (P * a) - (per_policy + per_premium * P)
    print(fun(44.71))

    P = life.solve(fun, [30.3, 49.5])
    print(P)


    # SOA Question 6.10:  (D) 0.91
    x = 0
    life = Recursion(interest=dict(i=0.06)).set_p(0.975, x=x)
    a = 152.85/56.05
    print(a)

    def fun(p):  # solve p_x+2, given a_x:3
        return a - life.set_p(p, x=x+1).temporary_annuity(x, t=3)
    p = life.solve(fun, guess=0.975)
    life.set_p(p, x=x+1)
    print(p)

    def fun(p):  # finally solve p_x+3, given A_x:3
        return 152.85 - life.set_p(p, x=x+2).term_insurance(x, t=3, b=1000)
    print(fun(0.91))
    p = life.solve(fun, guess=0.975)
    print(p)

    # SOA Question 6.17:  (A) -30000
    x = 0
    life = ConstantForce(mu=0.1, interest=dict(i=0.08))
    A = life.endowment_insurance(x, t=2, b=100000, endowment=30000)
    a = life.temporary_annuity(x, t=2)
    P = life.gross_premium(a=a, A=A)
    print(A, a, P)
    life1 = Recursion(interest=dict(i=0.08))
    life1.set_q(life.q_x(x, t=1) * 1.5, x=x, t=1)
    life1.set_q(life.q_x(x+1, t=1) * 1.5, x=x+1, t=1)
    policy = life1.Policy(premium=P*2, benefit=100000, endowment=30000)
    L = life1.gross_policy_value(x, t=0, n=2, policy=policy)
    print(-30000, L, 6.17)

    # SOA Question 7.29:  (E) 2270
    x = 0
    life = Recursion(interest=dict(i=0.04)).set_a(14.8, x=x)\
                                           .set_a(11.4, x=x+10)
    def fun(B):   # Solve for benefit B given net 10_V = 2290
        return 2290 - life.net_policy_value(x, t=10, b=B)
    B = life.solve(fun, guess=2290 * 10)
    print(B)
    policy = life.Policy(initial_policy=30, renewal_policy=5, benefit=B)
    G = life.gross_premium(a=life.whole_life_annuity(x), **policy.premium_terms)
    print(G)
    V = life.gross_policy_value(x, t=10, policy=policy.set(premium=G))
    print(V)


    # SOA Question 7.23:  (D) 233
    life = Recursion(interest=dict(i=0.04)).set_p(0.995, x=25)
    A = life.term_insurance(25, t=1, b=10000)
    print(A)
    def fun(beta):  # value of premiums in first 20 years must be equal
        return 216 * 11.087 - (beta * 11.087 + (A - beta)) 
    beta = life.solve(fun, guess=[140, 260])
    print(beta)

    # SOA Question 7.18:  (A) 17.1  - recursion
    x = 10
    life = Recursion(interest=dict(i=0.04)).set_q(0.009, x=x)
    def fun(a):
        life.set_a(a, x=x)
        return 0.012 - life.net_policy_value(x, t=1)
    a = life.solve(fun, guess=[17.1, 19.1])
    print(a)

    # SOA Question 7.17:  (D) 1.018
    x = 0
    life = Recursion(interest=dict(v=math.sqrt(0.90703)))
    life.set_q(0.02067, x=x+10)
    life.set_A(0.52536, x=x+11)
    life.set_A(0.30783, x=x+11, moment=2)
    A1 = life.whole_life_insurance(x+10)
    print(A1)
    A2 = life.whole_life_insurance(x+10, moment=2)
    print(A2)
    V1 = life.insurance_variance(A2=A2, A1=A1)
    var = V1 / life.insurance_variance(A2=0.30783, A1=0.52536)
    print(var)

    # SOA Question 4.3: (D) 0.878
    life = Recursion(interest=dict(i=0.05)).set_q(0.01, x=60)
    def fun(q):   # solve for q_61
        life.set_q(q, x=61)
        return life.endowment_insurance(60, t=3) - 0.86545
    q = life.solve(fun, 0.01)
    print(q)
    life.set_q(q, x=61)
    life.set_interest(i=0.045)
    A = life.endowment_insurance(60, t=3)
    print(A)

    # SOA Question 6.51:  (D) 34700
    life = Recursion()
    life.set_DA(0.4891, x=62, t=10)
    life.set_A(0.0910, x=62, t=10)
    life.set_a(12.2758, x=62)
    life.set_a(7.4574, x=62, t=10)
    print(life.decreasing_insurance(62, t=10))
    IA = life.increasing_insurance(62, t=10)
    print(IA)
    A = life.deferred_annuity(62, u=10)
    a = life.temporary_annuity(62, t=10)
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=50000)
    print(A, a, P)




    # SOA Question 6.46:  (E) 208
    life = Recursion(interest=dict(i=0.05))
    life.set_IA(0.51213, x=55, t=10)
    life.set_a(12.2758, x=55)
    life.set_a(7.4575, x=55, t=10)
    A = life.deferred_annuity(55, u=10)
    IA = life.increasing_insurance(55, t=10)
    a = life.temporary_annuity(55, t=10)
    P = life.gross_premium(a=a, A=A, IA=IA, benefit=300)
    print(P)


    # SOA Question 6.44:  (D) 2.18
    life = Recursion(interest=dict(i=0.05))
    life.set_IA(0.15, x=50, t=10)
    life.set_a(17, x=50)
    life.set_a(15, x=60)
    life.set_E(0.6, x=50, t=10)
    A = life.deferred_insurance(50, u=10)
    IA = life.increasing_insurance(50, t=10)
    a = life.temporary_annuity(50, t=10)
    P = life.gross_premium(a=a, A=A, IA=IA)
    print(P)


    # SOA Question 6.40: (C) 116 
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

    # SOA Question 5.9: 
    x = 0
    life1 = Recursion().set_a(21.854, x=x)
    life2 = Recursion().set_a(22.167, x=x)
    life1.set_p(0.9, x=x)
    def fun(k):
        life2.set_p((1+k)*0.9, x=x)
        return life1.whole_life_annuity(x+1) - life2.whole_life_annuity(x+1)
    print(life2.solve(fun, 0.01))
    
    # SOA Question 5.2:  (B) 9.64
    life = Recursion(interest=dict(i=0.05))
    x, n = 0, 10
    life.set_A(0.3, x)
    life.set_A(0.4, x+n)
    life.set_E(0.35, x, t=n)
    print(life.immediate_annuity(x, t=n))

    # SOA Question 6.21:  (C) 100
    life = Recursion(interest=dict(d=0.04))
    life.set_A(0.7, x=75, t=15, endowment=1)
    life.set_E(0.11, x=75, t=15)
    print(life.term_insurance(75, t=15))
    print(life.term_insurance(75, t=15, b=1000))
    P = 100.85
    print(life.E_x(75, t=15, endowment=1510))
    print(life.endowment_insurance(75, t=15, b=1000, endowment=15*P))
    print(P * life.temporary_annuity(75, t=15))
    print(life.temporary_annuity(75, t=15))
    print()
    def fun(P):
        P = float(P)
        return (P * life.temporary_annuity(75, t=15)
                - life.endowment_insurance(75, t=15, b=1000, endowment=15*P))
    print(fun(P))
    P = life.solve(fun, (80, 120))
    print(P)

    # SOA Question 4.9:  (D) 0.5
    life = Recursion().set_A(0.39, x=35, t=15, endowment=1)\
                      .set_A(0.25, x=35, t=15)
    E = life.E_x(35, t=15)

    life = Recursion().set_A(0.32, x=35)\
                      .set_E(E, x=35, t=15)
    print(life.E_x(35, t=15))
    def fun(A):
        life.set_A(A, x=50)
        return life.term_insurance(35, t=15)  - 0.25
    print(fun(.4))
    A = life.solve(fun, [0.35, 0.55])
    print(0.5, A, 4.9)



    # SOA Question 6.20:  (B) 459
    life = Recursion(interest=dict(i=0.04))
    life.set_p(.9, x=75).set_p(.88, x=76).set_p(.85, x=77)
    print(life.db)
    print(life.temporary_annuity(75, t=3))
#    print(life.E_x(75, t=1))
    print(life.term_insurance(75, t=1))
    print(life.term_insurance(76, t=1))
    print(life.deferred_insurance(75, t=1, u=1))
    print()
    print(life.term_insurance(75, t=1) +
          2 * life.deferred_insurance(75, t=1, u=1))
    print(life.deferred_insurance(75, t=1, u=2))
    def fun(P):
        a = life.temporary_annuity(75, t=3)
        A = ((P * life.term_insurance(75, t=1))
             + (2 * P * life.deferred_insurance(75, u=1, t=1))
             + (10000 * life.deferred_insurance(75, u=2, t=1)))
        return life.gross_premium(a=a, A=A) - P
    print(fun(459))
    P = life.solve(fun, [449, 489])
    print(P)

    # SOA Question 6.11:  (C) 0.041
    life = Recursion(interest=dict(i=0.04))
    life.set_A(0.39788, 51)
    life.set_q(0.0048, 50)
    A = life.whole_life_insurance(50)
    print(A)
    P = life.gross_premium(A=A, a=life.annuity_twin(A=A))
    print(P)
    life.set_q(0.048, 50)
    A = life.whole_life_insurance(50)
    print(A)
    print(A - life.annuity_twin(A) * P)


    # SOA Question 4.3: (D) 0.878
    life = Recursion(interest=dict(i=0.05))
    life.set_A(0.86545, x=60, t=3, endowment=1)
    life.set_q(0.01, x=60)
    A62 = life.endowment_insurance(62, t=1)
    print(A62)

    A61 = life.endowment_insurance(61, t=2)
    print(A61)

    life.set_A(None, x=60, t=3, endowment=1)
    life.set_A(None, x=61, t=2, endowment=1)
    life.set_A(A62, x=62, t=1, endowment=1)
    def fun(q):
        life.set_q(q, x=61)
        return life.endowment_insurance(61, t=2) - A61
    q = life.solve(fun, 0.01)
    print(q)

    life = Recursion(interest=dict(i=0.045)).set_q(0.01, x=60).set_q(q, x=61)
    for t in range(1, 4):
        A = life.endowment_insurance(63-t, t=t)
        life.set_A(A, x=63-t, t=t, endowment=1)
    print(A)

    # n = 3
    # x = 60
    # A = {x+t: [life.endowment_insurance(x+t, t=n-t)]
    #      for t in range(n+1)}
    # print(A)

    # select = Select(A=A, q={60: [0.01]}).fill()


    # SOA Question 2.5:  (B) 37.1
    life = Recursion().set_e(25, x=60, curtate=True)\
                      .set_q(0.2, x=40, t=20)\
                      .set_q(0.003, x=40)
    e_40_20 = 18
    def fun(e, target, x, t):  # e_x = e_x:t + t_p_x e_x+t
        life.set_e(e, x=x, curtate=True)
        return life.e_x(x, t=t, curtate=True) - target
    e40 = life.solve(fun, guess=40, args=(e_40_20, 40, 20))
    #print(e40)
    life.set_e(e40, x=40, curtate=True)
    #print(life.get_e(x=40, curtate=True))
    e41 = life.e_x(41, curtate=True)
    print(e41)
#    e41 = life.solve(fun, guess=40, args=(?, 40, 1))
#    print(life.e_x(40, t=1, curtate=True))
#    e41 = life.solve(fun, guess=41, args=(, 40, 20))
