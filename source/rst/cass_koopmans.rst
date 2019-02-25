.. _cass_koopmans:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3


***********************************
Cass-Koopmans Optimal Growth Model
***********************************

.. contents:: :depth: 2

**Coauthor: Brandon Kaplowitz**

Overview
=========

This lecture describes a model that Tjalling Koopmans :cite:`Koopmans`
and David Cass :cite:`Cass` used to analyze optimal growth

The model can be viewed as an extension of the model of Robert Solow
described in `an earlier lecture <https://lectures.quantecon.org/py/python_oop.html>`__ 
but adapted to make the savings rate the outcome of an optimal choice

(Solow assumed a constant saving rate determined outside the model)

We describe two versions of the model to illustrate what is in fact a
more general connection between a **planned economy** and an economy
organized as a **competitive equilibrium**

The lecture uses important ideas including

-  Hicks-Arrow prices named after John R. Hicks and Kenneth Arrow

-  A max-min problem for solving a planning problem

-  A **shooting algorithm** for solving difference equations subject
   to initial and terminal conditions

-  A connection between some Lagrange multipliers in the max-min
   problem and the Hicks-Arrow prices

-  A **Big K, little k** trick widely used in
   macroeconomic dynamics

  *  We shall encounter this trick in `this lecture <https://lectures.quantecon.org/py/rational_expectations.html#>`__
     and also in `this lecture <https://lectures.quantecon.org/py/dyn_stack.html#>`__

-  An application of a **guess and verify** method for solving a
   system of difference equations

-  The intimate connection between the cases for the optimalty of two
   competing visions of good ways to organize an economy, namely:

  *  **socialism** in which a central planner commands the
     allocation of resources, and

  *  **capitalism** (also known as **a free markets economy**) in
     which competitive equiibrium **prices** induce individual
     consumers and producers to choose a socially optimal allocation
     as an unintended consequence of their completely selfish
     decisions

-  A **turnpike** property that describes optimal paths for
   long-but-finite horizon economies

-  A non-stochastic version of a theory of the **term structure of
   interest rates**
  
Let's start with some imports

.. code:: ipython

  from numba import njit
  import numpy as np
  import matplotlib.pyplot as plt
  %matplotlib inline

The Growth Model
==================

Time is discrete and takes values :math:`t = 0, 1 , \ldots, T`

(We leave open the possibility that :math:`T = + \infty`, but that
will require special care in interpreting and using **terminal
condition** on :math:`K_t` at :math:`t = T+1` to be described
below)

A single good can either be consumed or invested in physical capital

The consumption good is not durable and depreciates completely if not
consumed immediately

The capital good is durable but depreciates each period at rate
:math:`\delta \in (0,1)`

We let :math:`C_t` be a nondurable consumption good at time t

Let :math:`K_t` be the stock of physical capital at time t

Let :math:`\vec{C}`\ =\ :math:`\{C_0,\dots, C_T\}` and
:math:`\vec{K}`\ =\ :math:`\{K_1,\dots,K_{T+1}\}`

A representative household is endowed with one unit of labor at each
:math:`t`\ and likes the consumption good at each\ :math:`t`

The representative household inelastically supplies a single unit of
labor :math:`N_t`\ at each\ :math:`t`, so that
:math:`N_t =1 \text{ for all } t \in [0,T]`

The representative household has preferences over consumption bundles
ordered by the utility functional:

.. math:: 
    :label: utility-functional
    
    U(\vec{C}) = \sum_{t=0}^{T} \beta^t \frac{C_t^{1-\gamma}}{1-\gamma}

where :math:`\beta \in (0,1)` is a discount factor and :math:`\gamma >0`
governs the curvature of the one-period utility function

Note that

.. math::
    :label: utility-oneperiod
  
    u(C_t) = \beta^t \frac{C_t^{1-\gamma}}{1-\gamma}

satisfies :math:`u'>0,u''<0`

:math:`u' > 0` asserts the consumer prefers more to less

:math:`u''< 0` asserts that marginal utility declines with increases
in :math:`C_t`

We assume that :math:`K_0 > 0` is a given exogenous level of initial
capital

There is an economy-wide production function

.. math:: 
  :label: production-function
  
  F(K_t,N_t) = A K_t^{\alpha}N_t^{1-\alpha}

with :math:`0 < \alpha<1`, :math:`A > 0`

A feasible allocation :math:`\vec C, \vec K` satisfies

.. math:: 
  :label: allocation
  
  C_t + K_{t+1} = F(K_t,N_t) + (1-\delta) K_t, \quad \text{for all } t \in [0, T]

where :math:`\delta \in (0,1)` is a depreciation rate of capital

Planning problem
------------------

A planner chooses an allocation :math:`\{\vec{C},\vec{K}\}` to
maximize :eq:`utility-functional` subject to :eq:`allocation`

Let :math:`\vec{\mu}=\{\mu_0,\dots,\mu_T\}` be a sequence of
nonnegative **Lagrange multipliers**

To find an optimal allocation, we form a Lagrangian

.. math:: 
  
  \mathcal{L}(\vec{C},\vec{K},\vec{\mu}) = 
  \sum_{t=0}^T \beta^t\left\{ u(C_t)+ \mu_t 
  \left(F(K_t,1) + (1-\delta) K_t- C_t - K_{t+1} \right)\right\}  

and then solve the following max-min problem:

.. math:: 
  :label: max-min-prob
  
  \max_{\vec{C},\vec{K}}\min_{\vec{\mu}}\mathcal{L}(\vec{C},\vec{K},\vec{\mu})

Useful properties of linearly homogeneous production
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following technicalities will help us

Notice that

.. math:: 
  
  F(K_t,N_t) = A K_t^\alpha N_t^{1-\alpha} = N_t A\left(\frac{K_t}{N_t}\right)^\alpha

Define the **output per-capita production function**

.. math:: 
  
  f(\frac{K_t}{N_t}) = A\left(\frac{K_t}{N_t}\right)^\alpha

whose argument is **capital per-capita**

Evidently,

.. math:: 
  
  F(K_t,N_t)=N_t f( \frac{K_t}{N_t})

Now for some useful calculations

First

.. math:: 
  :label: useful-calc1
  
  \frac{\partial F}{\partial K} =
  \frac{\partial N_t f\left( \frac{K_t}{N_t}\right)}{\partial K_t}{=}_{\text{chain rule}} 
  N_t f'\left(\frac{K_t}{N_t}\right)\frac{1}{N_t}=
  f'\left.\left(\frac{K_t}{N_t}\right)\right|_{N_t=1} = f'(K_t)

Also

.. math::

  \frac{\partial F}{\partial N} = \frac{\partial N_t f\left( \frac{K_t}{N_t}\right)}{\partial N_t}=
  _{\text{product rule}}f\left( \frac{K_t}{N_t}\right)+
  N_t\frac{\partial f\left(\frac{K_t}{N_t}\right)}{\partial N_t}{=}_
  {\text{chain rule}}f\left(\frac{K_t}{N_t}\right)+ N_t f'\left(\frac{K_t}{N_t}\right)
  \frac{-K_t}{N_t^2}=f\left(\frac{K_t}{N_t}\right)- \left
  \frac{K_t}{N_t}f'\left(\frac{K_t}{N_t}\right)\right|_{N_t=1} =
  f\left(K_t\right)-K_t f'\left(K_t\right)


Back to solving the problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve the Lagrangian extremization problem, we compute first
derivatives of the Lagrangian and set them equal to 0

-  **Note:** Our objective function and constraints satisfy
   conditions that work to assure that required second-order
   conditions are satisfied at an allocation that satisfies the
   first-order conditions that we are about to compute

Here are the **first order necessary conditions** for extremization
(i.e., maximization with respect to :math:`\vec C, \vec K`,
minimization with respect to\ :math:`\vec \mu`):

.. math::
    :label: constraint1
  
    C_t:& \quad u'(C_t)-\mu_t=0 \quad &&\text{for all } t=
    0,1,\dots,T
  
.. math::
    :label: constraint2
    
    K_t:& \quad \beta \mu_t\left[(1-\delta)+f'(K_t)\right] - 
    \mu_{t-1}=0 \quad &&\text{for all } t=1,2,\dots,T+1
  
.. math::
    :label: constraint3
    
    \mu_t:&  \quad F(K_t,1)+ (1-\delta) K_t  - C_t - K_{t+1}=0
    \quad &&\text{for all } t=0,1,\dots,T

.. math::
    :label: constraint4
    
    \quad -\mu_T \leq 0, \ <0 \text{ if } K_{T+1}=0; \ =0 \text{ if } K_{T+1}>0 

Note that in :eq:`constraint1` we plugged in for
:math:`\frac{\partial F}{\partial K}` using our formula :eq:`useful-calc1`
above

Because :math:`N_t = 1` for :math:`t = 1, \ldots, T`, need not
differentiate with respect to those arguments

Note that :eq:`constraint2` comes from the occurrence
of :math:`K_t` in both the period :math:`t` and period :math:`t-1`
feasibility constraints

:eq:`constraint3` comes from differentiating with respect
to :math:`K_{T+1}` in the last period and applying the following
condition called a **Karush-Kuhn-Tucker condition** (KKT):

.. math:: 
    :label: kkt
  
    \mu_T K_{T+1}=0

See `Karush-Kuhn-Tucker conditions <en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions>`__

Combining :eq:`constraint1` and :eq:`constraint2` gives

.. math:: 
  u'\left(C_t\right)\left[(1-\delta)+f'\left(K_t\right)\right]-u'\left(C_{t-1}\right)=0 
  \quad \text{ for all } t=1,2,\dots, T+1

Rewriting gives

.. math:: 
  u'\left(C_{t+1}\right)\left[(1-\delta)+f'\left(K_{t+1}\right)\right]=
  u'\left(C_{t}\right) \quad \text{ for all } t=0,1,\dots, T

Taking the inverse of the utility function on both sides of the above
equation gives

.. math:: 
  C_{t+1} =u'^{-1}\left(\left(\frac{\beta}{u'(C_t)}[f'(K_{t+1}) +(1-\delta)]\right)^{-1}\right)

or using our utility function :eq:`utility-oneperiod`

.. math:: 
  
  \begin{align} C_{t+1} =\left(\beta C_t^{\gamma}[f'(K_{t+1}) + 
  (1-\delta)]\right)^{1/\gamma} \notag\\= C_t\left(\beta [f'(K_{t+1}) +
  (1-\delta)]\right)^{1/\gamma} \end{align}

The above first-order condition for consumption is called an **Euler
equation**

It tells us how consumption in adjacent periods are optimally related
to each other and to capital next period

We now use some of the the equations above to calculate some
variables and functions that we'll soon use to solve the planning
problem with Python

.. code:: python3

    δ = .02  # Depreciation rate on capital
    ᾱ = .33  # Return to capital per capita
    γ̄ = 2    # Coefficient of relative risk aversion
    T = 10   # Maximum time period
    Ā = 1    # Technology 
    β = .95  # Discount rate

    @njit
    def u(c, γ):
        '''
        Utility function
        ASIDE: If you have a utility function that is hard to solve by hand
        you can use automatic or symbolic  differentiation
        See https://github.com/HIPS/autograd 
        '''
        if γ == 1:
            ## If γ = 1 we can show via L'hopital's Rule that the utility becomes log
            return np.log(c) 
        else:
            return c**(1 - γ) / (1 - γ)

    @njit
    def u_prime(c, γ):
        '''Derivative of utility'''
        if γ == 1:
            return 1 / c
        else: 
            return c**(-γ)

    @njit
    def u_prime_inv(c, γ):
        '''Inverse utility'''
        if γ == 1:
            return c
        else: 
            return c**(-1 / γ)

    @njit
    def f(A, k, α):
        '''Production function'''
        return A * k**α
    
    @njit
    def f_prime(A, k, α):
        '''Derivative of production function'''
        return α * A * k**(α - 1)
    
    @njit
    def f_prime_inv(A, k, α):
        return (k / (A * α))**(1 / (α - 1))

    # Define an initial value for all c

    C = np.zeros(T + 1)  # T periods of consumption initialized to 0
    K = np.zeros(T + 2)  # T periods of capital initialized to 0 (T+2 to include t+1 variable as well)


Shooting Method
----------------

We shall use a **shooting method** to compute an optimal allocation
:math:`\vec C, \vec K` and an associated Lagrange multiplier sequence
:math:`\vec \mu`

The first-order necessary conditions for the planning problem,
namely, equations :eq:`constraint1`, :eq:`constraint2`, and
:eq:`constraint3`, form a system of **difference equations** with
two boundary conditions:

-  :math:`K_0` is a given **initial condition** for capital

-  :math:`K_{T+1} =0` is a **terminal condition** for capital that we 
   deduced from the first-order necessary condition for :math:`K_{T+1}`
   the KKT condition :eq:`kkt`

We have no initial condition for the Lagrange multiplier
:math:`\mu_0`

If we did, solving for the allocation would be simple:

-  Given :math:`\mu_0`\ and :math:`k_0`, we could compute $c_0 $ from
   equation (:eq:`constraint1`) and then :math:`k_1` from equation
   (:eq:`constraint3`) and\ :math:`\mu_1` from equation
   (:eq:`constraint2`)

-  We could then iterate on to compute the remaining elements of
   :math:`\vec C, \vec K, \vec \mu`

But we don't have an initial condition for :math:`\mu_0`, so this
won't work

But a simple modification called the **shooting algorithm** will
work

The **shooting algorithm** is an instance of a **guess and verify**
algorithm

It proceeds as follows:

-  Guess a value for the initial Lagrange multiplier :math:`\mu_0`

-  Apply the **simple algorithm** described above

-  Compute the implied value of :math:`k_{T+1}` and check whether it
   equals zero

-  If the implied :math:`K_{T+1} =0`, we have solved the problem

-  If :math:`K_{T+1} > 0`, lower :math:`\mu_0` and try again

-  If :math:`K_{T+1} < 0`, raise :math:`\mu_0` and try again

The following Python code implements the shooting algorithm for the
planning problem

We make a slight modification starting with a guess of
:math:`c_0` but since :math:`c_0` is a function of :math:`\mu_0`
there is no difference to the procedure above

We'll apply it with an initial guess that will turn out not to be
perfect, as we'll soon see

.. code:: python3

    # Initial guesses
    K[0] = 0.3  # Initial k
    C[0] = 0.2  # Guess of c_0

    @njit
    def shooting_method(c,       # Initial consumption
                        k,       # Initial capital
                        T,       # Number of periods
                        γ̄=γ̄, 
                        δ=δ,     # Depreciation rate
                        β=β,     # Discount factor
                        ᾱ=ᾱ, 
                        Ā=Ā):

        for t in range(T):
            k[t+1] = f(A=Ā, k=k[t], α=ᾱ) + (1 - δ) * k[t] - c[t]  # Equation 1 with inequality
            if k[t+1] < 0:   # Ensure nonnegativity
                    k[t+1] = 0 

          # Equation 2: We keep in the general form to show how we would 
          # solve if we didn't want to do any simplification
            
            if β * (f_prime(A=Ā, k=k[t+1], α=ᾱ) + (1 - δ)) == np.inf:
                # This only occurs if k[t+1] is 0, in which case, we won't 
                # produce anything next period, so consumption will have to be 0
                c[t+1] = 0
            else:    
                c[t+1] = u_prime_inv(u_prime(c=c[t], γ=γ̄) / (β * (f_prime(A=Ā, k=k[t+1], α=ᾱ) + (1 - δ))), γ=γ̄)

        # Terminal condition calculation      
        k[T+1] = f(A=Ā, k=k[T], α=ᾱ) + (1 - δ) * k[T] - c[T]

        return c, k

    paths = shooting_method(C, K, T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    colors = ['blue', 'red']
    titles = ['Consumption', 'Capital']
    ylabels = ['$c_t$', '$k_t$']
    ranges = [range(T+1), range(T+2)]

    for path, c, t, y, r, ax in zip(paths, colors, titles, ylabels, ranges, axes):
        ax.plot(r, path, c=c, alpha=0.7)
        ax.set(title=t, ylabel=y, xlabel='t', xlim=(0, T+1.5))
        ax.axhline(0, color='k', lw=1)

    ax.scatter(T+1, 0, s=80)
    ax.axvline(T+1, color='k', ls='--', lw=1)

    plt.tight_layout()
    plt.show()

Evidently our initial guess for :math:`\mu_0` is too high and makes
initial consumption is too low

We know this becuase we miss our :math:`K_{T+1}=0` target on the high
side

Now we automate things with a search-for-a-good :math:`\mu_0`
algorithm that stops when we hit the target :math:`K_{t+1} = 0`

The search procedure is to use a **bisection method**

Here is how we apply the bisection method

We take an initial guess for :math:`C_0` (we can eliminate
:math:`\mu_0` because :math:`C_0` is an exact function of
:math:`\mu_0`)

We know that the lowest :math:`C_0` can ever be is :math:`0` and the
largest it can be is initial output :math:`f(K_0)`

We take a :math:`C_0` guess and shoot forward to :math:`T+1`

If the :math:`K_{T+1}>0`, let it be our new **lower** bound
on\ :math:`C_0`

If :math:`K_{T+1}<0`, let it be our new **upper** bound

Make a new guess for :math:`C_0` exactly halfway between our new
upper and lower bounds

Shoot forward again and iterate the procedure

When :math:`K_{T+1}` gets close enough to 0 (within some error
tolerance bounds), stop and declare victory

.. code:: python3

  @njit
  def bisection_method(c_init_guess, 
                       c_val,
                       k_val,
                       T,
                       γ̄=γ̄,
                       δ=δ,
                       β=β,
                       ᾱ=ᾱ,
                       Ā=Ā,
                       tol=1e-4,
                       max_iter=10000,
                       terminal=0):     # The value we are shooting towards
                       
      i = 1                             # Initial iteration
      c_high = f(k=k_val[0], α=ᾱ, A=Ā)  # Initial high value of c
      c_low = 0                         # Initial low value of c
      c_val[0] = c_init_guess
      
      path_c, path_k = shooting_method(c_val, k_val, T, γ̄, δ, β, ᾱ, Ā)
      
      condition_1 = path_k[T+1] - terminal > tol
      condition_2 = path_k[T+1] - terminal < -tol
      condition_3 = path_k[T] == terminal
      
      while (condition_1 or condition_2 or condition_3) and (i < max_iter):
          print(f'Iteration: {i}')
          if path_k[T+1] - terminal > tol:
              # If assets are too high the initial c we chose is 
              # now a lower bound on possible values of c[0]
              c_low = c_val[0] 
          elif path_k[T+1] - terminal < -tol:
              # If assets fell too quickly, the initial c we 
              # chose is now an upper bound on possible values of c[0]
              c_high = c_val[0]
          elif path_k[T] == terminal:
              # If assets fell too quickly, the initial c we 
              # chose is now an upper bound on possible values of c[0]
              c_high = c_val[0]
          
          c_val[0] = (c_high + c_low) / 2  # Value in middle of high and low value -- bisection part
          
          path_c, path_k = shooting_method(c_val, k_val, T, γ̄, δ, β, ᾱ, Ā)
          i = i + 1
      
      condition_4 = path_k[T+1] - terminal < tol
      condition_5 = path_k[T] != terminal
      condition_6 = path_k[T+1]-terminal > -tol
      
      if condition_4 & condition_5 & condition_6:
         print(f'Converged successfully on iteration {i-1}')
      else:
         print('Failed to converge and hit maximum iteration')
         
      mu = u_prime(c=path_c, γ=γ̄)
      return (path_c.copy(), path_k.copy(), mu)
  

Now we can plot:

.. code:: python3

    paths = bisection_method(0.3, C, K, T)
    
    fix, axes = plt.subplots(3, 1)
    colors = ['blue, 'red', 'green']
    ylabels = ['C_t', 'K_t', '$\mu_t$']
    titles = ['Consumption', 'Capital', 'Lagrange Multiplier']
    
    for path, c, y, t, ax in zip(paths, colors, ylabels, titles, axes):
        ax.plot(

    plt.subplot(131)
    plt.plot(range(T+1),path_opt_C,color='blue',alpha=.7)
    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T+2),path_opt_K,color='red',alpha=.7)
    plt.axvline(11,color='black',ls='--',lw=1)
    plt.axhline(0,color='black', lw=1)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlim(0,)
    plt.xlabel('$t$')
    plt.scatter(11,0,s=80)
    plt.subplot(133)
    plt.plot(range(T+1),path_opt_mu,color='green',alpha=.7)
    plt.title('Lagrange Multiplier')
    plt.ylabel('$\mu_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)
    plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)

    plt.show()

Setting :math:`K_0` equal to steady state
------------------------------------------

If :math:`T \rightarrow +\infty`, the optimal allocation converges to
steady state values of :math:`C_t`\ and\ :math:`K_t`

It is instructive to compute these and then to set :math:`K_0` equal
to its steady state value

In a steady state :math:`K_{t+1} = K_t=\bar{K}`\ for all very
large\ :math:`t` the feasibility constraint :eq:`allocation` is

.. math:: f(\bar{K})-\delta \bar{K} = \bar{C}

Substituting :math:`K_t = \bar K`\ and :math:`C_t=\bar C` for
all\ :math:`t` into (`1.16 <#1.16>`__) gives

.. math:: 1=\beta \frac{u'(\bar{C})}{u'(\bar{C})}[f'(\bar{K})+(1-\delta)]

Defining :math:`\beta = \frac{1}{1+\rho}`, and cancelling gives

.. math:: 1+\rho = 1[f'(\bar{K}) + (1-\delta)]

Simplifying gives

.. math:: f'(\bar{K}) = \rho +\delta

and

.. math:: \bar{K} = f'^{-1}(\rho+\delta)

Using our production function (`1.2 <#1.2>`__) gives

.. math:: \alpha \bar{K}^{\alpha-1} = \rho + \delta

Finally, using :math:`\alpha= .33`,
:math:`\rho = 1/\beta-1 =1/(19/20)-1 = 20/19-19/19 = 1/19`, :math:`\delta = 1/50`,
we get

.. math:: \bar{K} = \left(\frac{\frac{33}{100}}{\frac{1}{50}+\frac{1}{19}}\right)^{\frac{67}{100}} \approx 9.57583  \tag{2.7} \label{2.7}

Let's verify this with Python and then use this steady state
:math:`\bar K` as our initial capital stock :math:`K_0`

.. code:: python3

    ρ = 1 / β - 1
    K_ss = f_prime_inv(k=ρ+δ, A=Ā, α=ᾱ)

    print(f'Steady state for capital is: {K_ss}')


.. code:: python3

    K_init_val = K_ss  # At our steady state
    T_new = 150
    C_new = np.zeros(T_new+1)
    K_new = np.zeros(T_new+2)
    K_new[0] = K_init_val
    path_opt_C_new, path_opt_K_new, path_opt_mu_new = bisection_method(.3, C_new, K_new, T_new)

And now we plot

.. code:: python

    plt.subplot(131)
    plt.plot(range(T_new+1),path_opt_C_new,color='blue',alpha=.7, )

    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T_new+2),path_opt_K_new,alpha=.7)
    plt.axhline(K_ss,linestyle='-.',color='black', lw=1,alpha=.7)
    plt.axhline(0,color='black', lw=1)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.scatter(T_new+1,0,s=80) 
    plt.subplot(133)
    plt.plot(range(T_new+1),path_opt_mu_new,color='green',alpha=.7)
    plt.title('Lagrange Multiplier')
    plt.ylabel('$\mu_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)
    plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)

    plt.show()

Evidently in this economy with a large value of
:math:`T`, :math:`K_t` stays near its initial value at the until the
end of time approaches closely

Evidently, the planner likes the steady state capital stock and wants
to stay near there for a long time

Let's see what happens when we push the initial
:math:`K_0` below :math:`\bar K`

.. code:: python3

  K_init_val = K_ss / 3   # Below our steady state
  T_new = 150
  C_new = np.zeros(T_new + 1)
  K_new = np.zeros(T_new + 2)
  K_new[0] = K_init_val
  path_opt_C_new, path_opt_K_new, path_opt_mu_new = bisection_method(0.3, C_new, K_new, T_new)

The following code plots the optimal allocation

.. code:: python3

    plt.subplot(131)
    plt.plot(range(T_new+1),path_opt_C_new,color='blue',alpha=.7)

    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T_new+2),path_opt_K_new,alpha=.7)
    plt.axhline(K_ss,linestyle='-.',color='black', lw=1,alpha=.7)
    plt.axhline(0,color='black', lw=1)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.scatter(T_new+1,0,s=80)
    plt.subplot(133)
    plt.plot(range(T_new+1),path_opt_mu_new,color='green',alpha=.7)
    plt.title('Lagrange Multiplier')
    plt.ylabel('$\mu_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)
    plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)
    plt.show()

Notice how the planner pushes capital toward the steady state, stays
near there for a while, then pushes :math:`K_t` toward the terminal
value :math:`K_{T+1} =0` as :math:`t` gets close to :math:`T`

The following graphs compare outcomes as we vary :math:`T`

.. code:: python

    K_new[0] = K_init_val
    path_opt_C_new, path_opt_K_new, path_opt_mu_new = bisection_method(.3, C_new, K_new, T_new)
    T_new_2 = 75
    C_new_2 = np.zeros(T_new_2+1)
    K_new_2 = np.zeros(T_new_2+2)
    K_new_2[0] = K_init_val
    path_opt_C_2, path_opt_K_2, path_opt_mu_new_2 = bisection_method(.3, C_new_2, K_new_2, T_new_2);
    T_new_3 = 50
    C_new_3 = np.zeros(T_new_3+1)
    K_new_3 = np.zeros(T_new_3+2)
    K_new_3[0] = K_init_val
    path_opt_C_3, path_opt_K_3, path_opt_mu_new_3 = bisection_method(.3, C_new_3, K_new_3, T_new_3);
    T_new_4 = 25
    C_new_4 = np.zeros(T_new_4+1)
    K_new_4 = np.zeros(T_new_4+2)
    K_new_4[0] = K_init_val
    path_opt_C_4, path_opt_K_4, path_opt_mu_new_4 = bisection_method(.3, C_new_4, K_new_4, T_new_4);

.. code:: python3

    plt.subplot(131)
    plt.plot(range(T_new+1),path_opt_C_new,path_opt_C_2,alpha=.7)
    plt.plot(range(T_new_3+1),path_opt_C_3, alpha=.7)
    plt.plot(range(T_new_4+1),path_opt_C_4,color='darkslateblue', alpha=.7)
    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T_new+2),path_opt_K_new,path_opt_K_2,alpha=.7)
    plt.plot(range(T_new_3+2),path_opt_K_3, alpha=.7)
    plt.plot(range(T_new_4+2),path_opt_K_4,color='darkslateblue', alpha=.7)
    plt.axhline(0,color='black', lw=1)
    plt.axhline(K_ss,linestyle='-.',color='black', lw=1,alpha=.7)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.scatter(T_new+1,0,s=80)
    plt.scatter(T_new_2+1,0,s=80)
    plt.scatter(T_new_3+1,0,s=80)
    plt.scatter(T_new_4+1,0,s=80, color='slateblue')
    plt.subplots_adjust(left=0.2, wspace=0.5, top=0.8)
    plt.subplot(133)
    plt.plot(range(T_new+1),path_opt_mu_new,path_opt_mu_new_2,alpha=.7)
    plt.plot(range(T_new_3+1),path_opt_mu_new_3,alpha=.7)
    plt.plot(range(T_new_4+1),path_opt_mu_new_4,color='darkslateblue',alpha=.7)
    plt.title('Lagrange Multiplier')
    plt.ylabel('$\mu_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)
    plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)
    plt.show()

The following calculation shows that when we set :math:`T` very large
the planner makes the capital stock spend most of its time close to
its steady state value

.. code:: python3

    T_new_large = 250
    C_new_large = np.zeros(T_new_large+1)
    K_new_large = np.zeros(T_new_large+2)
    K_new_large[0] = K_init_val
    path_opt_C_large, path_opt_K_large, path_opt_mu_large = bisection_method(.3, C_new_large, K_new_large, T_new_large)

.. code:: python3

    plt.subplot(131)
    plt.plot(range(T_new+1),path_opt_C_new,path_opt_C_2,alpha=.7)
    plt.plot(range(T_new_3+1),path_opt_C_3, alpha=.7)
    plt.plot(range(T_new_large+1),path_opt_C_large,alpha=.7)
    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T_new+2),path_opt_K_new,path_opt_K_2,alpha=.7)
    plt.plot(range(T_new_3+2),path_opt_K_3, alpha=.7)
    plt.plot(range(T_new_large+2),path_opt_K_large,alpha=.7)
    plt.axhline(0,linestyle='-',color='black', lw=1)
    plt.axhline(K_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.scatter(T_new+1,0,s=80)
    plt.scatter(T_new_2+1,0,s=80)
    plt.scatter(T_new_3+1,0,s=80)
    plt.scatter(T_new_large+1,0,s=80)
    plt.subplot(133)
    plt.plot(range(T_new+1),path_opt_mu_new,path_opt_mu_new_2,alpha=.7)
    plt.plot(range(T_new_3+1),path_opt_mu_new_3,alpha=.7)
    plt.plot(range(T_new_large+1),path_opt_mu_large,alpha=.7)
    plt.title('Lagrange Multiplier')
    plt.ylabel('$\mu_t$')
    plt.xlabel('$t$')
    plt.xlim(0,)
    plt.axhline(0,color='black', lw=1)
    plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)
    plt.show()

The different colors in the above graphs are tied to outcomes with
different horizons :math:`T`

Notice that as the hoizon increases, the planner puts :math:`K_t`
closer to the steady state value :math:`\bar K` for longer

This pattern reflects a **turnpike** property of the steady state.

A rule of thumb for the planner is

-  for whatever :math:`K_0` you start with, push :math:`K_t` toward
   the stady state and stay there for as long as you can

In loose language: head for the turnpick and stay near it for as long as you can

As we drive :math:`T \toward +\infty`, the planner
keeps :math:`K_t` very close to its steady state for all dates after
some transition toward the steady state

The planner makes the saving rate :math:`\frac{f(K_t) - C_t}{f(K_t)}`
vary over time

Let's calculate it

.. code:: python3

  def S(K, δ, T):
      '''Aggregate savings'''
      S = np.zeros(T+1)
      for t in range(0, T+1):
          S[t] = K[t+1] - (1 - δ) * K[t]
      return S
      
  def s(A, K, α, δ, T):
      '''Savings rate'''
      S = S(K, δ, T)
      Y = f(A, K, α)
      Y = Y[0:T+1]
      s = S / Y
      return s

  path_opt_s_new=s(Ā,path_opt_K_new,ᾱ,δ,T_new)
  path_opt_s_2=s(Ā,path_opt_K_2,ᾱ,δ,T_new_2)
  path_opt_s_3=s(Ā,path_opt_K_3,ᾱ,δ,T_new_3)
  path_opt_s_large=s(Ā,path_opt_K_large,ᾱ,δ,T_new_large)

  plt.subplot(131)
  plt.plot(range(T_new+1),path_opt_C_new,path_opt_C_2,alpha=.7)
  plt.plot(range(T_new_3+1),path_opt_C_3, alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_C_large,alpha=.7)
  plt.title('Consumption')
  plt.ylabel('$C_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(132)
  plt.plot(range(T_new+2),path_opt_K_new,path_opt_K_2,alpha=.7)
  plt.plot(range(T_new_3+2),path_opt_K_3, alpha=.7)
  plt.plot(range(T_new_large+2),path_opt_K_large,alpha=.7)
  plt.axhline(0,linestyle='-',color='black', lw=1)
  plt.axhline(K_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
  plt.title('Capital')
  plt.ylabel('$K_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.scatter(T_new+1,0,s=80)
  plt.scatter(T_new_2+1,0,s=80)
  plt.scatter(T_new_3+1,0,s=80)
  plt.scatter(T_new_large+1,0,s=80)
  plt.subplot(133)
  plt.plot(range(T_new+1),path_opt_s_new,path_opt_s_2,alpha=.7)
  plt.plot(range(T_new_3+1),path_opt_s_3, alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_s_large,alpha=.7)
  plt.axhline(0,color='black', lw=1)
  plt.title('Savings rate')
  plt.ylabel('$s_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.subplots_adjust(left=0.0, wspace=0.4, top=0.8,right=1.2)
  plt.show()

:math:`T=+\infty` economy
--------------------------

We now consider an economy in which :math:`T = +\infty`

The appropriate thing to do is to replace terminal condition
(:eq:`condition4`) with

.. math:: 
  
  \lim_{T \rightarrow +\infty} \beta^T u'(C_T) K_T = 0

This condition will be satisfied by a path that converges to an
optimal steady state

We can approximate the optimal path from an arbitrary initial
:math:`K_0` and shooting toward the optimal steady state
:math:`K` at a large but finite :math:`T+1`

In the following code, we do this for a large :math:`T`; we shoot
towards the **steady state** and plot consumption, capital and the
savings rate

We know that in the steady state that the saving rate must be fixed
and that :math:`\bar s= \frac{f(\bar K)-\bar C}{f(\bar K)}`

From (`2.1 <#2.1>`__) the steady state saving rate equals

.. math:: 
  
  \bar s =\frac{ \delta \bar{K}}{f(\bar K)}

The steady-state savings level :math:`\bar S = \bar s f(\bar K)` is
the amount required to offset capital depreciation each period

We first study optimal capital paths that start below the steady
state

.. code:: python3

    T = 130                     # Long time series
    K_init_val = K_ss / 3       # Below our steady state
    S_ss = δ * K_ss
    C_ss = f(Ā, K_ss, ᾱ) - S_ss
    s_ss = S_ss / f(Ā, K_ss, ᾱ)
    
    C = np.zeros(T+1)
    K = np.zeros(T+2)
    K[0] = K_init_val
    path_opt_C, path_opt_K, path_opt_mu = bisection_method(.3, C, K, T, tol=1e-8, terminal=K_ss)
    path_opt_s = s(Ā, path_opt_K, ᾱ, δ, T)

.. code:: python3

    plt.subplot(131)
    plt.plot(range(T+1),path_opt_C_vlarge,alpha=.7,color='blue')
    plt.axhline(C_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,T)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T+2),path_opt_K_vlarge,alpha=.7,color='red')
    plt.axhline(0,linestyle='-',color='black', lw=1)
    plt.axhline(K_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlabel('$t$')
    plt.xlim(0,T+1)

    plt.subplot(133)
    plt.plot(range(T+1),path_opt_s_vlarge,alpha=.7,color='purple')
    plt.axhline(s_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.axhline(0,color='black', lw=1)
    plt.title('Savings rate')
    plt.ylabel('$s_t$')
    plt.xlabel('$t$')
    plt.ylim(-.015,.3)
    plt.xlim(0,T)
    plt.subplots_adjust(left=0.0, wspace=0.4, top=0.8,right=1.2)
    plt.show()

Since :math:`K_0<\bar K`, :math:`f'(K_0)>\rho +\delta`

The planner choose a positive saving rate above the steady state
level offsetting depreciation that enables us to increase our capital
stock

Note, :math:`f''(K)<0`, so as :math:`K` rises, :math:`f'(K)` declines

The planner slowly lowers the savings rate until reaching a steady
state where :math:`f'(K)=\rho +\delta`

Exercise
---------

-  Plot the optimal consumption, capital, and savings paths when the
   initial capital level begins at 1.5 times the steady state level
   as we shoot towards the steady state at :math:`T=130`

-  Why does the savings rate respond like it does?

Solution
----------

.. code:: python

    T = 130
    K_init_val = K_ss * 1.5  # Above our steady state
    S_ss = δ * K_ss
    C_ss = f(Ā, K_ss, ᾱ) - S_ss
    s_ss = S_ss / f(Ā, K_ss, ᾱ)
    C = np.zeros(T+1)
    K = np.zeros(T+2)
    K[0] = K_init_val
    path_opt_C, path_opt_K, path_opt_mu = bisection_method(0.3, C_vlarge, K_vlarge, T, tol=1e-8, terminal=K_ss)
    path_opt_s = s(Ā, path_opt_K, ᾱ, δ, T)

.. code:: python3

    plt.subplot(131)
    plt.plot(range(T+1), path_opt_C_vlarge, alpha=.7, color='blue')
    plt.axhline(C_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.title('Consumption')
    plt.ylabel('$C_t$')
    plt.xlabel('$t$')
    plt.xlim(0,T)
    plt.axhline(0,color='black', lw=1)

    plt.subplot(132)
    plt.plot(range(T+2),path_opt_K_vlarge,alpha=.7,color='red')
    plt.axhline(0,linestyle='-',color='black', lw=1)
    plt.axhline(K_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.title('Capital')
    plt.ylabel('$K_t$')
    plt.xlabel('$t$')
    plt.xlim(0,T+1)

    plt.subplot(133)
    plt.plot(range(T+1),path_opt_s_vlarge,alpha=.7,color='purple')
    plt.axhline(s_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
    plt.axhline(0,color='black', lw=1)
    plt.title('Savings rate')
    plt.ylabel('$s_t$')
    plt.xlabel('$t$')
    plt.ylim(-.015,.3)
    plt.xlim(0,T)
    plt.subplots_adjust(left=0.0, wspace=0.4, top=0.8,right=1.2)
    plt.show()

Competitive Equilibrium
========================

Next we study a decentralized version of an economy with same
technology and preference structure as our planned economy

But now there is no planner

Market prices adjust to reconcile distinct decisions that are made
separately by a representative household and a representative firm

The technology for producing goods and accumulating capital via
physical investment remains as in our planned economy

There is a representative consumer who has the same preferences over
consumption plans as did the consumer in the planned economy

Instead of being told what to consume and save by a planner, the
household chooses for itself subject to a budget constraint

-  At each time :math:`t`, the household receives wages and rentals
   of capital from a firm -- these comprise its **income** at
   time\ :math:`t`

-  The consumer decides how much income to allocate to consumption or
   to savings

-  The household can save either by acquiring additional physical
   capital (it trades one for one with time :math:`t` consumption)
   or by acquiring claims on consumption at dates other
   than :math:`t`

-  A utility maximizing household owns all physical capital and labor
   and rents them to the firm

-  The household consumes, supplies labor, and invests in physical
   capital

-  A profit-maximizing representative firm operates the producion
   technology

-  The firm rents labor and capital each period from the
   representative household and sells its output each period to the
   household

-  The representative household and the representative firm are both
   **price takers:**

   -  they (correctly) believe that prices are not affected by their
      choices

**Note:** We are free to think of there being a large number
:math:`M` of identical representative consumers and :math:`M`
identical representative firms

Firm Problem
-------------

At time :math:`t` the representative firm hires labor
:math:`\tilde n_t` and capital :math:`\tilde k_t`

The firm's profits at time :math:`t` are

.. math:: F(\tilde k_t, \tilde n_t)-w_t \tilde n_t -\eta_t \tilde k_t

where :math:`w_t` is a wage rate at :math:`t` and
and :math:`\eta_t` is the rental rate on capital at :math:`t`

As in the planned economy model

.. math:: F(\tilde k_t, \tilde n_t) = A \tilde k_t^\alpha \tilde n_t^{1-\alpha}

Zero profit conditions
^^^^^^^^^^^^^^^^^^^^^^^^

Zero-profits condition for capital and labor are

.. math:: F_k(\tilde k_t, \tilde n_t) =\eta_t

and

.. math:: F_n(\tilde k_t, \tilde n_t) =w_t

These conditions emerge from a no-arbitrage requirement

To describe this line of reasoning, we begin by applying a theorm of
Euler about linearly homogenous functions

The theorem applies to the Cobb-Douglas production function because
it assumed displays constant returns to scale:

.. math:: \alpha F(\tilde k_t, \tilde n_t) =  F(\alpha  \tilde k_t, \alpha \tilde n_t)

for :math:`\alpha \in (0,1)`

Taking the partial derivative
:math:`\frac{\partial F }{\partial \alpha}` on both sides of the
above equation gives

.. math:: 
  
  F(\tilde k_t,\tilde n_t) =_\text{chain rule} \frac{\partial F}{\partial \tilde k_t} 
  \tilde k_t + \frac{\partial F}{\partial \tilde  n_t} \tilde n_t

Rewrite the firm's profits as

.. math:: 
  
  \frac{\partial F}{\partial \tilde k_t} \tilde k_t + 
  \frac{\partial F}{\partial \tilde  n_t} \tilde n_t-w_t \tilde n_t -\eta_t k_t

or

.. math:: 
  
  \left(\frac{\partial F}{\partial \tilde k_t}-\eta_t\right) \tilde k_t + 
  \left(\frac{\partial F}{\partial \tilde  n_t}-w_t\right) \tilde n_t \label{3.8} \tag{3.8}

Because :math:`F`\ is homogeneous of degree\ :math:`1`, it follows
that :math:`\frac{\partial F}{\partial \tilde k_t}`\ and
:math:`\frac{\partial F}{\partial \tilde n_t}` are homogoneous of
degree :math:`0` and therefore fixed with respect to
:math:`\tilde k_t` and\ :math:`\tilde n_t`

If :math:`\frac{\partial F}{\partial \tilde k_t}> \eta_t`, then the
firm makes positive profits on each additional unit of
:math:`\tilde k_t`, so it will want to make :math:`\tilde k_t`
arbitrarily large

But setting :math:`\tilde k_t = + \infty` is not physically feasible,
so presumably **equilibrium** prices will assume values that present
the firm with no such arbitrage opportunity

A related argument applies if
:math:`\frac{\partial F}{\partial \tilde n_t}> w_t`

If :math:`\frac{\partial \tilde k_t}{\partial \tilde k_t}< \eta_t`,
the firm will set\ :math:`\tilde k_t` to zero

Again, **equilibrium** prices won't incentive the firm to do that.

And so on...

It is convenient to define
:math:`\vec w_t =\{w_0, \dots,w_T\}`\ and\ :math:`\vec \eta_t = \{\eta_0, \dots, \eta_T\}`

Household Problem
------------------

A representative household lives at :math:`t=0,1,\dots, T`

At :math:`t`, the household rents :math:`1`\ unit of labor
and\ :math:`k_t` units of capital to a firm and receives income

.. math:: w_t 1+ \eta_t k_t

At :math:`t` the household allocates its income to the following
purchases

.. math:: \left(c_t + (k_{t+1} -(1-\delta)k_t)\right)

Here :math:`\left(k_{t+1} -(1-\delta)k_t)\right)` is the household's
net investment in physical capital and :math:`\delta \in (0,1)` is
again a depreciation rate of capital

In period :math:`t` is free to purchase more goods to be consumed and
invested in physical capital than its income from supplying capital
and labor to the firm, provided that in some other periods its income
exceeds its purchases

A household's net excess demand for time :math:`t` consumption goods
is the gap

.. math:: e_t \equiv \left(c_t + (k_{t+1} -(1-\delta)k_t)\right)-(w_t 1 + \eta_t k_t) \label{3.11} \tag{3.11}

Let :math:`\vec c = \{c_0,\dots,c_T\}` and let :math:`\vec k = \{k_1,\dots,k_T+1\}`

:math:`k_0` is given to the household

Market structure for intertemporal trades
-------------------------------------------

There is a **single** grand competitive market in which a
representative household can trade date :math:`0` goods for goods at
all other dates\ :math:`t=1, 2, \ldots, T`

What matters are not **bilateral** trades of the good at one date
:math:`t` for the good at another date :math:`\tilde t \neq t`.

Instead, think of there being **multilateral** and **multitemporal**
trades in which bundles of goods at some dates can be traded for
bundles of goods at some other dates.

There exist **complete markets** in such bundles with associated
market prices

Market prices
--------------

Let :math:`q^0_t` be the price of a good at date :math:`t` relative
to a good at date :math:`0`

:math:`\{q^0_t\}_{t=0}^T` is a vector of **Hicks-Arrow prices**,
named after the 1972 joint economics Nobel prize winners who used
such prices in some of their important work

Evidently,

.. math:: q^0_t=\frac{\text{# of time 0 goods}}{\text{# of time t goods}}

Because :math:`q^0_t` is a **relative price**, the units in terms of
which prices are quoted are arbitrary -- we can normalize them
without substantial consequence

If we use the price vector :math:`\{q^0_t\}_{t=0}^T` to evaluate a
stream of excess demands :math:`\{e_t\}_{t=0}^T`\ we compute the
**present value** of :math:`\{e_t\}_{t=0}^T` to be
:math:`\sum_{t=0}^T q^0_t e_t`

That the market is **multitemporal** is reflected in the situation
that the household faces a **single** budget constraint

It states that the present value of the household's net excess
demands must be zero:

.. math:: \sum_{t=0}^T q^0_t e_t  \leq 0

or

.. math:: \sum_{t=0}^T q^0_t  \left(c_t + (k_{t+1} -(1-\delta)k_t)-(w_t 1 + \eta_t k_t) \right) \leq 0

Household problem
------------------

The household faces the constrained optimization problem:

.. math:: \begin {align*}& \max_{\vec c, \vec k}  \sum_{t=0}^T \beta^t u(c_t) \label{3.15} \tag{3.15}\\ \text{subject to} \ \   & \sum_{t=0}^T q_t^0\left(c_t +\left(k_{t+1}-(1-\delta) k_t\right) -w_t -\eta_t k_t\right) \leq 0  \notag \end{align*}

Definitions
------------

-  A **price system** is a sequence
   :math:`\{q_t^0,\eta_t,w_t\}_{t=0}^T= \{\vec q, \vec \eta, \vec w\}`

-  An **allocation** is a sequence
   :math:`\{c_t,k_{t+1}, k_{t+1},n_t=1\}_{t=0}^T = \{\vec c, \vec k, \vec n =1\}`

-  A **competitive equilibrium** is a price system and an allocation
   for which

   -  Given the price system, the allocation solves the household's
      problem

   -  Given the price system, the allocation solves the firm's
      problem

Computing a Competitive Equilibrium
-------------------------------------

We shall compute a competitive equilibrium using a **guess and
verify** approach

-  We shall **guess** equilibrium price sequences
   :math:`\{\vec q, \vec \eta, \vec w\}`

-  We shall then **verify** that at those prices, the household and
   the firm choose the same allocation

Guess for price system
^^^^^^^^^^^^^^^^^^^^^^^

We have computed an allocation :math:`\{\vec C, \vec K, \vec 1\}`
that solves the planning problem

We use that allocation to construct our guess for the equilibium
price system

In particular, we guess that for :math:`t=0,\dots,T`:

.. math:: \lambda q_t^0 = \beta^t u'(K_t) =\beta^t \mu_t

.. math:: w_t = f(K_t) -K_t f'(K_t)

.. math:: \eta_t = f'(K_t)

At these prices, let the capital chosen by the household be

.. math:: k^*_t(\vec q, \vec w, \vec \eta) , \quad t \geq 0

and let the allocation chosen by the firm be

.. math:: \tilde k^*_t(\vec q, \vec  w, \vec \eta), \quad t \geq 0

and so on

If our guess for the equilibrium price system is correct, then it
must occur that

.. math::

   \begin{align} 
  k_t^* & = \tilde k_t^*  \label{3.21} \tag{3.21}\\
  1 &  = \tilde n_t^*  \label{3.22} \tag{3.22}\\
  c_t^* + k_{t+1}^* - (1-\delta) k_t^* & = F(\tilde k_t^*, \tilde n_t^*) 
  \end{align} 

We shall verify that for :math:`t=0,\dots,T` the allocations chosen
by the household and the firm both equal the allocation that solves
the planning problem:

.. math:: k^*_t = \tilde k^*_t=K_t, \tilde n_t=1, c^*_t=C_t

Verification procedure
-----------------------

Our approach is to stare at first-order necessary conditions for the
optimization problems of the household and the firm

At the price system we have guessed, both sets of first-order
conditions are satisfied at the allocation that solves the planning
problem

Household's Lagrangian
------------------------

To solve the household's problem, we formulate the appropriate
Lagrangian and pose the max-min problem:

.. math:: 
  
  \max_{\vec{c},\vec{k}}\min_{\lambda}\mathcal{L}(\vec{c},\vec{k},\lambda)=
  \sum_{t=0}^T \beta^t u(c_t)+ \lambda \left(\sum_{t=0}^T q_t^0\left(c_t -\left(k_{t+1})-(1-\delta) 
  k_t -w_t\right) -\eta_t k_t\right)\right)

First-order conditions are

.. math:: 
  
  \begin{align} &&& c_t:& \quad \beta^t u'(c_t)-\lambda q_t^0=0 \quad 
  && t=0,1,\dots,T \label{3.26} \tag{3.26}\\ &&& k_t:& \quad -\lambda q_t^0 
  \left[(1-\delta)+\eta_t \right]+\lambda q^0_{t-1}=0 \quad && t=1,2,\dots,T+1 
  \label{3.27} \tag{3.27}\\ &&& \lambda:&  \quad \left(\sum_{t=0}^T 
  q_t^0\left(c_t -\left(k_{t+1}-(1-\delta) k_t\right) -w_t -\eta_t k_t\right)\right) = 
  0 \label{3.28} \tag{3.28}\\ &&& k_{T+1}:& \quad -\lambda q_0^{T+1} \leq 0, \ <0 
  \text{ if } K_{T+1}=0; \ =0 \text{ if } K_{T+1}>0 \end{align}

Now we plug in for our guesses of prices and derive all the FONC of
the planner problem (`1.10 <#1.10>`__)-(\ `1.13 <#1.13>`__):

Combining (`3.26 <#3.26>`__) and (`3.16 <#3.16>`__), we get:

.. math:: u'(C_t) = \mu_t \label{3.30} \tag{3.30}

which is (`1.10 <#1.10>`__).

Combining (`3.27 <#3.27>`__), (`3.16 <#3.16>`__), and
(`3.18 <#3.18>`__) we get:

.. math:: -\lambda \beta^t \mu_t\left[(1-\delta) +f'(K_t)\right] +\lambda \beta^{t-1}\mu_{t-1}=0

Rewriting :math:`(\ref{3.31})`\ by dividing by\ :math:`\lambda` on
both sides (which is nonzero due to u'>0) we get:

.. math:: \beta^t \mu_t [(1-\delta+f'(K_t)] = \beta^{t-1} \mu_{t-1} \label{3.32} \tag{3.32}

or

.. math:: \beta \mu_t [(1-\delta+f'(K_t)] = \mu_{t-1} \label{3.33} \tag{3.33}

which is (`1.11 <#1.11>`__).

Combining (`3.28 <#3.28>`__), (`3.16 <#3.16>`__), (`3.17 <#3.17>`__)
and (`3.18 <#3.18>`__) after multiplying both sides of
(`3.28 <#3.28>`__) by :math:`\lambda`, we get:

.. math:: \sum_{t=0}^T \beta^t \mu_{t} \left(C_t+ (K_{t+1} -(1-\delta)K_t)-f(K_t)+K_t f'(K_t)-f'(K_t)K_t\right) \leq 0  \label{3.34} \tag{3.34}

Cancelling,

.. math:: \sum_{t=0}^T  \beta^t \mu_{t} \left(C_t +K_{t+1} -(1-\delta)K_t - F(K_t,1)\right) \leq 0 \label{3.35} \tag{3.35}

Since :math:`\beta^t`\ and\ :math:`\mu_t` are always positive here,
(excepting perhaps the T+1 period) we get:

.. math:: C_t+K_{t+1}-(1-\delta)K_t -F(K_t,1)=0 \quad  \text{ for all }t \text{ in } 0,\dots,T \label{3.36} \tag{3.36}

\ which is (`1.12 <#1.12>`__)

Combining (`3.29 <#3.29>`__) and (`3.16 <#3.16>`__), we get:

.. math:: - \beta^{T+1} \mu_{T+1} \leq 0 \label{3.37} \tag{3.37}

Dviding both sides by :math:`\beta^{T+1}` which will be strictly
positive here, we get:

.. math:: -\mu_{T+1} \leq 0

\ which is the (`1.13 <#1.13>`__) of our social planner problem

**Thus, at our guess of the equibrium price system the allocation
that solves the planning problem also solves the problem that the
representative household faces in a competitive equilibrium**

We now consider the problem faced by a firm in a competitive
equilibrium:

If we plug in (`3.24 <#3.24>`__) into (`3.4 <#3.3>`__) for all t, we
get

.. math:: \frac{\partial F(K_t, 1)}{\partial K_t} = f'(K_t) = \eta_t, \label{3.39} \tag{3.39}

\ which is (`3.18 <#3.18>`__)

If we now plug (`3.24 <#3.24>`__) into (`3.4 <#3.4>`__) for all t, we
get:

.. math:: \frac{\partial F(\tilde K_t, 1)}{\partial \tilde L} = f(K_t)-f'(K_t)K_t=w_t

\ which is exactly (`3.19 <#3.19>`__)

**Thus, at our guess of the equibrium price system the allocation
that solves the planning problem also solves the problem that a firm
faces in a competitive equilibrium**

By (`3.21 <#3.21>`__) and (`3.22 <#3.22>`__) this allocation is
identical to the one that solves the consumer's problem

**Note:** Because budget sets are affected only by relative prices,
:math:`\{q_0^t\}` is determined only up to multiplication by a
positive constant

**Normalization:** We are free to choose a :math:`\{q_0^t\}`\ that
makes :math:`\lambda=1`, thereby making :math:`q_0^t` be measured in
units of the marginal utility of time\ :math:`0` goods

Brandon's take on what all this means
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following are Brandon's thoughts and have not been edited or
certified by Tom

In solving this problem we just showed two of the most famous results
in economics

-  One is the big :math:`K` little :math:`k` result -- the individual cannot affect the
   aggregate, and only their individual choice matters

  -  Individual choices still generate identical aggregate law of
     motions given a set of prices generated by those optimal
     aggregate laws of motions

-  The second is the two welfare theorems

  -  The First Welfare Theorem states that any allocation generated
     by a social planner problem can be achieved (supported) by a
     competative equilibrium with the introduction of prices and
     transfers between individuals
  -  The Second welfare Theorem states that the competative
     equilibrium is **pareto optimal** -- this loosely means one
     cannot be made better off than the individual-optimal
     allocation without making others worse off. Our competitive
     equilibrium allocation being equal to the social planner
     allocation that optimizes total welfare demonstrates this

What we just did is we took a sequence of prices, used it to generate
a sequence of little k's and c's and :math:`\tilde k`'s
and\ :math:`\tilde c`'sthat are the firm and households optimal
response, and then showed that when we look at the aggregates
generated by the little k's and little c's, they are exactly the
sequence of aggregates (:math:`K_t, C_t`) that generate the prices
that we started with in the first place

In mathematical terminology this is known as a **fixed point in a
sequence space** and results from using the big K little k trick. We
oftentimes call an equilibrium that satisfies this property, a
**rational expectations equilibrium** and it is a result you will see
over and over in economics

What this shows is that if someone asks which is better, a central
planner, or a free market, we can say they are exactly the same so
neither is better than the other

With complete markets, from one perspective, we have a perfect
socialist or communist system and from another perspective a perfect
free market system and these give the exact same allocation

The mechanisms driving each couldn't be more different, however, with
the central planner solving an optimization problem for the entire
economy with no prices, and the free market having no single
aggregate optimization problem, but optimal allocations that result
from Adam Smith's individuals acting in selfish interest, guided by
an "invisible hand"

Question
^^^^^^^^^

What is Adam Smith's 'invisible hand' that guides individuals to
produce the same optimal allocation as in the central planner
problem?

Answer
^^^^^^^

Prices.

We will also plot q, w and :math:`\eta` below to show the prices that
generate the aggregate movements we saw earlier in the social planner
problem.

.. code:: python3

  @njit
  def q(β,C,T,γ):
      # Here we choose numeraire to be u'(c_0) -- this is q^(t_0)_t
      q = np.zeros(T+1) 
      q[0] = 1
      for t in range(1, T+2):
          q[t] = β**t * u_prime(C[t], γ)
      return q
  
  @njit
  def w(A, k, α):
      w = f(A, k, α) - k * f_prime(A=A, k=k, α=α)
      return w
  
  @njit
  def η(A, k, α):
      η = f_prime(A=A, k=k, α=α)
      return η

Now we calculate and plot for each :math:`T`

.. code:: python3

  q_path_1 = q(β,path_opt_C_new,T_new,γ̄)
  q_path_2 = q(β,path_opt_C_2,T_new_2,γ̄)
  q_path_3 = q(β,path_opt_C_3,T_new_3,γ̄)
  q_path_large = q(β,path_opt_C_large,T_new_large,γ̄)

  w_path_1 =w(Ā,path_opt_K_new,ᾱ)
  w_path_2= w(Ā,path_opt_K_2,ᾱ)
  w_path_3 = w(Ā,path_opt_K_3,ᾱ)
  w_path_large=w(Ā,path_opt_K_large,ᾱ)

  eta_path_1 = η(Ā,path_opt_K_new,ᾱ)
  eta_path_2 = η(Ā,path_opt_K_2,ᾱ)
  eta_path_3 = η(Ā,path_opt_K_3,ᾱ)
  eta_path_large = η(Ā,path_opt_K_large,ᾱ)

.. code:: python3

  plt.subplot(231)
  plt.plot(range(T_new+1),q_path_1,q_path_2,alpha=.7)
  plt.plot(range(T_new_3+1),q_path_3, alpha=.7)
  plt.plot(range(T_new_large+1),q_path_large,alpha=.7)
  plt.title('Arrow-Hicks Prices')
  plt.ylabel('$q_t^0$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(232)
  plt.plot(range(T_new+1),w_path_1[:T_new+1],alpha=.7)
  plt.plot(range(T_new_2+1),w_path_2[:T_new_2+1],alpha=.7)
  plt.plot(range(T_new_3+1),w_path_3[:T_new_3+1], alpha=.7)
  plt.plot(range(T_new_large+1),w_path_large[:T_new_large+1],alpha=.7)
  plt.title('Labor Rental Rate')
  plt.ylabel('$w_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(233)
  plt.plot(range(T_new+1),eta_path_1[:T_new+1],eta_path_2[:T_new_2+1],alpha=.7)
  plt.plot(range(T_new_3+1),eta_path_3[:T_new_3+1], alpha=.7)
  plt.plot(range(T_new_large+1),eta_path_large[:T_new_large+1],alpha=.7)
  plt.title('Capital Rental Rate')
  plt.ylabel('$\eta_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)
  plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)

  plt.subplot(234)
  plt.plot(range(T_new+1),path_opt_C_new,path_opt_C_2,alpha=.7)
  plt.plot(range(T_new_3+1),path_opt_C_3, alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_C_large,alpha=.7)
  plt.title('Consumption')
  plt.ylabel('$C_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(235)
  plt.plot(range(T_new+2),path_opt_K_new,path_opt_K_2,alpha=.7)
  plt.plot(range(T_new_3+2),path_opt_K_3, alpha=.7)
  plt.plot(range(T_new_large+2),path_opt_K_large,alpha=.7)
  plt.axhline(0,linestyle='-',color='black', lw=1)
  plt.axhline(K_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
  plt.title('Capital')
  plt.ylabel('$K_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.scatter(T_new+1,0,s=80)
  plt.scatter(T_new_2+1,0,s=80)
  plt.scatter(T_new_3+1,0,s=80)
  plt.scatter(T_new_large+1,0,s=80)

  plt.subplot(236)
  plt.plot(range(T_new+1),path_opt_mu_new,path_opt_mu_new_2,alpha=.7)
  plt.plot(range(T_new_3+1),path_opt_mu_new_3,alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_mu_large,alpha=.7)
  plt.title('Lagrange Multiplier')
  plt.ylabel('$\mu_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)
  plt.subplots_adjust(right=2, wspace=0.2, top=2)
  plt.show()

Varying :math:`\gamma`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Now we see how our results change if we keep T constant, but allow
the curvature parameter, :math:`\gamma`\ to vary, starting
with\ :math:`K_0` below the steady state.

We plot the results for :math:`T=150`

.. code:: python3

  C_ss = f(k=K_ss, A=Ā, α=ᾱ) - δ * K_ss
  C_ss_growth = 0
  γ_1 = 1.1
  γ_2 = 4
  γ_3 = 6
  γ_4 = 8
  T_new_large=T_new
  C_new_large=C_new
  K_new_large=K_new
  path_opt_C_large,path_opt_K_large,path_opt_mu_large=bisection_method(.3,C_new_large,K_new_large,T_new_large,γ̄,δ,β,ᾱ,Ā);
  path_opt_C_large_1,path_opt_K_large_1,path_opt_mu_large_1=bisection_method(.3,C_new_large,K_new_large,T_new_large,γ_1,δ,β,ᾱ,Ā);
  path_opt_C_large_2,path_opt_K_large_2,path_opt_mu_large_2=bisection_method(.3,C_new_large,K_new_large,T_new_large,γ_2,δ,β,ᾱ,Ā);
  path_opt_C_large_3,path_opt_K_large_3,path_opt_mu_large_3=bisection_method(.3,C_new_large,K_new_large,T_new_large,γ_3,δ,β,ᾱ,Ā);
  path_opt_C_large_4,path_opt_K_large_4,path_opt_mu_large_4=bisection_method(.3,C_new_large,K_new_large,T_new_large,γ_4,δ,β,ᾱ,Ā);

  q_path_large = q(β,path_opt_C_large,T_new_large,γ̄)
  q_path_large_1 = q(β,path_opt_C_large_1,T_new_large,γ_1)
  q_path_large_2 = q(β,path_opt_C_large_2,T_new_large,γ_2)
  q_path_large_3 = q(β,path_opt_C_large_3,T_new_large,γ_3)
  q_path_large_4 = q(β,path_opt_C_large_4,T_new_large,γ_4)

  w_path_large =w(Ā,path_opt_K_large,ᾱ)
  w_path_large_1= w(Ā,path_opt_K_large_1,ᾱ)
  w_path_large_2 = w(Ā,path_opt_K_large_2,ᾱ)
  w_path_large_3=w(Ā,path_opt_K_large_3,ᾱ)
  w_path_large_4=w(Ā,path_opt_K_large_4,ᾱ)

  eta_path_large = η(Ā,path_opt_K_large,ᾱ)
  eta_path_large_1 = η(Ā,path_opt_K_large_1,ᾱ)
  eta_path_large_2 = η(Ā,path_opt_K_large_2,ᾱ)
  eta_path_large_3 = η(Ā,path_opt_K_large_3,ᾱ)
  eta_path_large_4 = η(Ā,path_opt_K_large_4,ᾱ)

.. code:: python3

  plt.subplot(231)
  plt.plot(range(T_new_large+1),q_path_large,q_path_large_1,alpha=.7)
  plt.plot(range(T_new_large+1),q_path_large_2, alpha=.7)
  plt.plot(range(T_new_large+1),q_path_large_3,alpha=.7)
  plt.plot(range(T_new_large+1),q_path_large_4,alpha=.7)

  plt.title('Arrow-Hicks Prices')
  plt.ylabel('$q_t^0$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(232)
  plt.plot(range(T_new_large+1),w_path_large[:T_new_large+1],alpha=.7)
  plt.plot(range(T_new_large+1),w_path_large_1[:T_new_large+1],alpha=.7)
  plt.plot(range(T_new_large+1),w_path_large_2[:T_new_large+1], alpha=.7)
  plt.plot(range(T_new_large+1),w_path_large_3[:T_new_large+1],alpha=.7)
  plt.plot(range(T_new_large+1),w_path_large_4[:T_new_large+1],alpha=.7)
  plt.title('Labor Rental Rate')
  plt.ylabel('$w_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(233)
  plt.plot(range(T_new_large+1),eta_path_large[:T_new_large+1],eta_path_large_1[:T_new_large+1],alpha=.7)
  plt.plot(range(T_new_large+1),eta_path_large_2[:T_new_large+1], alpha=.7)
  plt.plot(range(T_new_large+1),eta_path_large_3[:T_new_large+1],alpha=.7)
  plt.plot(range(T_new_large+1),eta_path_large_4[:T_new_large+1],alpha=.7)

  plt.title('Capital Rental Rate')
  plt.ylabel('$\eta_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)
  plt.subplots_adjust(left=0.0, wspace=0.5, top=0.8)


  plt.subplot(234)
  plt.plot(range(T_new_large+1),path_opt_C_large,path_opt_C_large_1,alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_C_large_2,alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_C_large_3,alpha=.7)
  plt.plot(range(T_new_large+1),path_opt_C_large_4,alpha=.7)
  plt.axhline(C_ss,linestyle='-.',color='black', lw=1,alpha=.7)
  plt.title('Consumption')
  plt.ylabel('$C_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)


  plt.subplot(235)
  plt.plot(range(T_new_large+2),path_opt_K_large,path_opt_K_large_1,alpha=.7)
  plt.plot(range(T_new_large+2),path_opt_K_large_2,alpha=.7)
  plt.plot(range(T_new_large+2),path_opt_K_large_3,alpha=.7)
  plt.plot(range(T_new_large+2),path_opt_K_large_4,alpha=.7)
  plt.axhline(0,linestyle='-',color='black', lw=1)
  plt.axhline(K_ss,linestyle='-.',color='black',lw=1,  alpha=.7)
  plt.title('Capital')
  plt.ylabel('$K_t$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.scatter(T_new_large+1,0,s=80)

  plt.subplot(236)
  plt.plot(range(T_new_large+1),path_opt_mu_large,path_opt_mu_large_1,alpha=.7) 
  plt.plot(range(T_new_large+1),path_opt_mu_large_2,alpha=.7) 
  plt.plot(range(T_new_large+1),path_opt_mu_large_3,alpha=.7) 
  plt.plot(range(T_new_large+1),path_opt_mu_large_4,alpha=.7) 
  plt.axhline(0,linestyle='-',color='black', lw=1)
  plt.title('Lagrange Multiplier')
  plt.ylabel('$\mu_t$')
  plt.xlabel('$t$')
  plt.axhline(0,color='black', lw=1)
  plt.xlim(0,)
  plt.subplots_adjust(left=0.0, wspace=0.6, top=1.8,hspace=0.25,right=1.2)
  plt.show()


Adjusting :math:`\gamma` means adjusting how much individuals prefer
to smooth consumption

Higher :math:`\gamma` means individuals prefer to smooth more
resulting in slower adjustments to the steady state allocations

Vice-versa for lower :math:`\gamma`


Yield Curves and Hicks-Arrow Prices Again
------------------------------------------

Now, we compute Hicks-Arrow prices again, but also calculate the
implied the yields to maturity.

This will let us plot a **yield curve**

The key formulas are:

The **yield to maturity**

.. math:: 
  
  r_{t_0,t}= -\frac{\log q^{t_0}_t}{t} \label{4.1} \tag{4.1}

A generic Hicks-Arrow price for any base-year :math:`t_0\leq t`

.. math:: 
  
  q^{t_0}_t = \beta^{t-t_0} \frac{u'(c_t)}{u'(c_{t_0})}= \beta^{t-t_0} 
  \frac{c_t^{-\gamma}}{c_{t_0}^{-\gamma}}

We redefine our function for :math:`q`\ to allow arbitrary base
years, and define a new function for\ :math:`r`, then plot both

First we plot when :math:`t_0=0`\ as before, for different values of
:math:`T`, with\ :math:`K_0` below the steady state

.. code:: python3

  @njit
  def q(t_0, β, c, T, γ):
      # Here we choose numeraire to be u'(c_0) -- this is q^(t_0)_t
      q = np.zeros(T+1-t_0) 
      q[0] = 1
      for t in range(t_0+1, T+2):
          q[t-t_0] = β**(t - t_0) * u_prime(c[t], γ) / u_prime(c[t_0], γ) 
      return q
      
  @njit
  def r_m(t_0, β, c, T, γ):
      '''Yield to maturity'''
      r = np.zeros(T+1-t_0)
      for t in range(t_0+1, T+2):
          r[t-t_0]= -np.log(q(t_0,β, c, T, γ)[t-t_0]) / (t - t_0)
      return r

  t_init=0
  q_path_1 = q(t_init,β,path_opt_C_new,T_new,γ̄)
  q_path_2 = q(t_init,β,path_opt_C_2,T_new_2,γ̄)
  q_path_3 = q(t_init,β,path_opt_C_3,T_new_3,γ̄)
  q_path_large = q(t_init,β,path_opt_C_large,T_new_large,γ̄)

  r_path_1=r_m(t_init,β,path_opt_C_new,T_new,γ̄)
  r_path_2 = r_m(t_init,β,path_opt_C_2,T_new_2,γ̄)
  r_path_3 = r_m(t_init,β,path_opt_C_3,T_new_3,γ̄)
  r_path_large = r_m(t_init,β,path_opt_C_large,T_new_large,γ̄)


  plt.subplot(121)
  plt.plot(range(T_new+1-t_init),q_path_1,q_path_2,alpha=.7)
  plt.plot(range(T_new_3+1-t_init),q_path_3, alpha=.7)
  plt.plot(range(T_new_large+1-t_init),q_path_large,alpha=.7)
  plt.title('Hicks-Arrow Prices')
  plt.ylabel('$q_t^0$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(122)
  plt.plot(range(T_new+1-t_init),r_path_1,r_path_2,alpha=.7)
  plt.plot(range(T_new_3+1-t_init),r_path_3, alpha=.7)
  plt.plot(range(T_new_large+1-t_init),r_path_large,alpha=.7)
  plt.title('Yields')
  plt.ylabel('$r_{0,t}$')
  plt.xlabel('$t$')
  plt.ylim(-.01,.3)
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)
  plt.subplots_adjust(left=0.0, wspace=0.3)

Now we plot when :math:`t_0=20`

.. code:: python

  t_init=20
  q_path_1 = q(t_init,β,path_opt_C_new,T_new,γ̄)
  q_path_2 = q(t_init,β,path_opt_C_2,T_new_2,γ̄)
  q_path_3 = q(t_init,β,path_opt_C_3,T_new_3,γ̄)
  q_path_large = q(t_init,β,path_opt_C_large,T_new_large,γ̄)

  r_path_1=r_m(t_init,β,path_opt_C_new,T_new,γ̄)
  r_path_2 = r_m(t_init,β,path_opt_C_2,T_new_2,γ̄)
  r_path_3 = r_m(t_init,β,path_opt_C_3,T_new_3,γ̄)
  r_path_large = r_m(t_init,β,path_opt_C_large,T_new_large,γ̄)

  plt.subplot(121)
  plt.plot(range(T_new+1-t_init),q_path_1,q_path_2,alpha=.7)
  plt.plot(range(T_new_3+1-t_init),q_path_3, alpha=.7)
  plt.plot(range(T_new_large+1-t_init),q_path_large,alpha=.7)
  plt.title('Arrow Prices at $t_0=20$')
  plt.ylabel('$q_t^{20}$')
  plt.xlabel('$t$')
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)

  plt.subplot(122)
  plt.plot(range(T_new+1-t_init),r_path_1,r_path_2,alpha=.7)
  plt.plot(range(T_new_3+1-t_init),r_path_3, alpha=.7)
  plt.plot(range(T_new_large+1-t_init),r_path_large,alpha=.7)
  plt.title('Yields at $t_0=20$')
  plt.ylabel('$r_{20,t}$')
  plt.xlabel('$t$')
  plt.ylim(-.01,.4)
  plt.xlim(0,)
  plt.axhline(0,color='black', lw=1)
  plt.subplots_adjust(left=0.0, wspace=0.3)

We shall have more to say about the term structure of interest rates
in a later lecture on the topic
