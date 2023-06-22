actuarialmath - Life Contingent Risks with Python
=================================================

This Python package implements fundamental methods for modeling life contingent risks, and closely follows the coverage of traditional topics in actuarial exams and standard texts such as the "Fundamentals of Actuarial Math - Long-term" exam syllabus by the Society of Actuaries, and "Actuarial Mathematics for Life Contingent Risks" by Dickson, Hardy and Waters.

Overview
--------

The package comprises three sets of classes, which:

1. Implement general actuarial methods

   - Basic interest theory and probability laws

   - Survival functions, expected future lifetimes and fractional ages

   - Insurance, annuity, premiums, policy values, and reserves calculations


2. Adjust results for

   - Extra mortality risks

   - 1/mthly payment frequency using UDD or Woolhouse approaches

3. Specify and load a particular form of assumptions

   - Life table, select life table, or standard ultimate life table

   - Mortality laws, such as constant force of maturity, beta and uniform distributions, or Makeham's and Gompertz's laws

   - Recursion inputs

     
Quick Start
-----------

1. ``pip install actuarialmath``
   
   - also requires `numpy`, `scipy`, `matplotlib` and `pandas`.
     
2. Start Python (version >= 3.10) or Jupyter-notebook

   - Select a suitable subclass to initialize with your actuarial assumptions, such as `MortalityLaws` (or a special law like `ConstantForce`), `LifeTable`, `SULT`, `SelectLife` or `Recursion`.
      
   - Call appropriate methods to compute intermediate or final results, or to `solve` parameter values implicitly.

   - Adjust answers with `ExtraRisk` or `Mthly` (or its `UDD` or `Woolhouse`) classes.

Examples
--------

::

  # SOA FAM-L sample question 5.7
  from actuarialmath import Recursion, Woolhouse
  # initialize Recursion class with actuarial inputs
  life = Recursion().set_interest(i=0.04)\
                    .set_A(0.188, x=35)\
                    .set_A(0.498, x=65)\
                    .set_p(0.883, x=35, t=30)
  # modfy the standard results with Woolhouse mthly approximation
  mthly = Woolhouse(m=2, life=life, three_term=False)
  # compute the desired temporary annuity value
  print(1000 * mthly.temporary_annuity(35, t=30)) #   solution = 17376.7

::

  # SOA FAM-L sample question 7.20
  from actuarialmath import SULT, Contract
  life = SULT()
  # compute the required FPT policy value
  S = life.FPT_policy_value(35, t=1, b=1000)  # is always 0 in year 1!
  # input the given policy contract terms
  contract = Contract(benefit=1000,
                      initial_premium=.3,
                      initial_policy=300,
                      renewal_premium=.04,
                      renewal_policy=30)
  # compute gross premium using the equivalence principle
  G = life.gross_premium(A=life.whole_life_insurance(35), **contract.premium_terms)
  # compute the required policy value
  R = life.gross_policy_value(35, t=1, contract=contract.set_contract(premium=G))
  print(R-S)   # solution = -277.19

Resources
---------

1. `Colab <https://colab.research.google.com/drive/1TcNr1x5HbT2fF3iFMYGXdN_cvRKiSua4?usp=sharing>`_ or `Jupyter notebook <https://terence-lim.github.io/notes/faml.ipynb>`_, to solve all sample SOA FAM-L exam questions

2. `Online User Guide <https://terence-lim.github.io/actuarialmath-guide/>`_, or `download pdf <https://terence-lim.github.io/notes/actuarialmath-guide.pdf>`_

3. `API reference <https://actuarialmath.readthedocs.io/en/latest/>`_

4. `Github repo <https://github.com/terence-lim/actuarialmath.git>`_ and `issues <https://github.com/terence-lim/actuarialmath/issues>`_

.. toctree::
   :maxdepth: 2
   :caption: Submodules:

   actuarialmath.actuarial
   actuarialmath.interest
   actuarialmath.life
   actuarialmath.survival
   actuarialmath.lifetime
   actuarialmath.fractional
   actuarialmath.insurance
   actuarialmath.annuity
   actuarialmath.premiums
   actuarialmath.policyvalues
   actuarialmath.reserves
   actuarialmath.lifetable
   actuarialmath.sult
   actuarialmath.selectlife
   actuarialmath.recursion
   actuarialmath.mortalitylaws
   actuarialmath.constantforce
   actuarialmath.extrarisk
   actuarialmath.mthly
   actuarialmath.udd
   actuarialmath.woolhouse


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
