from actuarialmath import Recursion

x = 0
life = Recursion(depth=5).set_interest(i=0.06)\
                         .set_p(0.975, x=x)\
                         .set_a(152.85/56.05, x=x, t=3)\
                         .set_A(152.85, x=x, t=3, b=1000)
p = life.p_x(x=x+2)
print(0.91, p, "Q6.10")




life = Recursion().set_A(0.39, x=35, t=15, endowment=1)\
                  .set_A(0.25, x=35, t=15)\
                  .set_A(0.32, x=35)
A = life.whole_life_insurance(x=50)

#def fun(A): return life.set_A(A, x=50).term_insurance(35, t=15)
#A = life.solve(fun, target=0.25, grid=[0.35, 0.55])
print(0.5, A, "Q4.9")

