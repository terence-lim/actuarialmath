# Actuarial Math

## Actuarial Math - Life Contingent Risks

This `actuarialmath` package implements in Python the general
formulas, recursive relationships and shortcut equations for
Fundamentals of Long Term Actuarial Mathematics, to solve the SOA
sample FAM-L questions and more.

- The concepts are developed hierarchically in [object-oriented Python](https://github.com/terence-lim/actuarialmath.git).

- Each module incrementally introduces the [formulas used, with usage examples](https://terence-lim.github.io/notes/actuarialmath.pdf).

- The SOA sample questions (released in August 2022) are solved in an
[executable Google Colab Notebook](https://colab.research.google.com/drive/1qguTCMQSk0m273IHApXA7IpUJwSoKEb-?usp=sharing).

Enjoy!

Terence Lim

MIT License. Copyright 2022, Terence Lim

## Concepts and Class Inheritance

![actuarialmath](FAM-L.png)

## Examples

- SOA sample question 5.7: Given $A_{35} = 0.188$, $A_{65} = 0.498$, $S_{35}(30) = 0.883$, calculate the EPV of a temporary annuity $\ddot{a}^{(2)}_{35:\overline{30|}}$ paid half-yearly using the Woolhouse approximation.

```
life = Recursion(interest=dict(i=0.04))
life.set_A(0.188, x=35).set_A(0.498, x=65).set_p(0.883, x=35, t=30)
mthly = Woolhouse(m=2, life=life, three_term=False)
print(1000 * mthly.temporary_annuity(35, t=30))
```

- SOA sample question 7.20: Calculate the policy value and
  modified reserve, where gross premiums are given by the 
  equivalence principle, of a whole life insurance where 
  mortality follows the Standard Ultimate Life Table.

```
life = SULT()
S = life.FPT_policy_value(35, t=1, b=1000)  # is 0 for FPT at t=0,1
policy = life.Policy(benefit=1000, initial_premium=.3, initial_policy=300,
                     renewal_premium=.04, renewal_policy=30)
P = life.gross_premium(A=life.whole_life_insurance(35), **policy.premium_terms)
R = life.gross_policy_value(35, t=1, policy=policy.set(premium=P))
```

## Resources

- Documentation and formulas: [actuarialmath.pdf](https://terence-lim.github.io/notes/actuarialmath.pdf)

- Executable Colab Notebook: [faml.ipynb](https://colab.research.google.com/drive/1qguTCMQSk0m273IHApXA7IpUJwSoKEb-?usp=sharing)

- Github repo: [https://github.com/terence-lim/actuarialmath.git](https://github.com/terence-lim/actuarialmath.git)

- SOA FAM-L Sample Solutions: [copy retrieved Aug 2022](https://terence-lim.github.io/notes/2022-10-exam-fam-l-sol.pdf)

- SOA FAM-L Sample Questions: [copy retrieved Aug 2022](https://terence-lim.github.io/notes/2022-10-exam-fam-l-quest.pdf)

- Actuarial Mathematics for Life Contingent Risks (Dickson, Hardy and Waters), Institute and Faculty of Actuaries, published by Cambridge University Press

## Contact me

Linkedin: [https://www.linkedin.com/in/terencelim](https://www.linkedin.com/in/terencelim)

Github: [https://terence-lim.github.io](https://terence-lim.github.io)


