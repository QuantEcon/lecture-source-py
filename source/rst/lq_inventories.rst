.. _Inventory_sales_smoothing-v6:

.. include:: /_static/includes/header.raw

.. highlight:: python3


************************************
Production Smoothing via Inventories
************************************


.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture employs the following library:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


Overview
=========

This lecture can be viewed as an application of the :doc:`quantecon lecture<lqcontrol>`.

It formulates a discounted dynamic program for a firm that
chooses a production schedule to balance

-  minimizing costs of production across time, against

-  keeping costs of holding inventories low

In the tradition of a classic book by Holt, Modigliani, Muth, and
Simon :cite:`Holt_Modigliani_Muth_Simon`, we simplify the
firm’s problem by formulating it as a linear quadratic discounted
dynamic programming problem of the type studied in this :doc:`quantecon<lqcontrol>`.

Because its costs of production are increasing and quadratic in
production, the firm wants to smooth production across time provided
that holding inventories is not too costly.

But the firm also prefers to sell out of existing inventories, a
preference that we represent by a cost that is quadratic in the
difference between sales in a period and the firm’s beginning of period
inventories.

We compute examples designed to indicate how the firm optimally chooses
to smooth production and manage inventories while keeping inventories
close to sales.

To introduce components of the model, let

-  :math:`S_t` be sales at time :math:`t`
-  :math:`Q_t` be production at time :math:`t`
-  :math:`I_t` be inventories at the beginning of time :math:`t`
-  :math:`\beta \in (0,1)` be a discount factor
-  :math:`c(Q_t) = c_1 Q_t + c_2 Q_t^2`, be a cost of production
   function, where :math:`c_1>0, c_2>0`, be an inventory cost function
-  :math:`d(I_t, S_t) = d_1 I_t + d_2 (S_t - I_t)^2`, where
   :math:`d_1>0, d_2 >0`, be a cost-of-holding-inventories function,
   consisting of two components:

   -  a cost :math:`d_1 t` of carrying inventories, and
   -  a cost :math:`d_2 (S_t - I_t)^2` of having inventories deviate
      from sales

-  :math:`p_t = a_0 - a_1 S_t + v_t` be an inverse demand function for a
   firm’s product, where :math:`a_0>0, a_1 >0` and :math:`v_t` is a
   demand shock at time :math:`t`
-  :math:`\pi\_t = p_t S_t - c(Q_t) - d(I_t, S_t)` be the firm’s
   profits at time :math:`t`
-  :math:`\sum_{t=0}^\infty \beta^t \pi_t` 
   be the present value of the firm’s profits at
   time :math:`0`
-  :math:`I_{t+1} = I_t + Q_t - S_t` be the law of motion of inventories
-  :math:`z_{t+1} = A_{22} z_t + C_2 \epsilon\_{t+1}` be the law
   of motion for an exogenous state vector :math:`z_t` that contains
   time :math:`t` information useful for predicting the demand shock
   :math:`v_t`
-  :math:`v_t = G z_t` link the demand shock to the information set
   :math:`z_t`
-  the constant :math:`1` be the first component of :math:`z_t`

To map our problem into a linear-quadratic discounted dynamic
programming problem (also known as an optimal linear regulator), we
define the **state** vector at time :math:`t` as

.. math::  x_t = \begin{bmatrix} I_t \cr z_t \end{bmatrix} 

and the **control** vector as

.. math::  u_t =  \begin{bmatrix} Q_t \cr S_t \end{bmatrix}  

The law of motion for the state vector :math:`x_t` is evidently

.. math::

    \begin{aligned}
    \begin{bmatrix} I_{t+1} \cr z_t \end{bmatrix} = \left[\begin{array}{cc}
   1 & 0\\
   0 & A_{22}
   \end{array}\right] \begin{bmatrix} I_t \cr z_t \end{bmatrix} 
                + \begin{bmatrix} 1 & -1 \cr
                0 & 0 \end{bmatrix} \begin{bmatrix} Q_t \cr S_t \end{bmatrix} 
                + \begin{bmatrix} 0 \cr C_2 \end{bmatrix} \epsilon_{t+1} \end{aligned}


or

.. math::  x_{t+1} = A x_t + B u_t + C \epsilon_{t+1} 

(At this point, we ask that you please forgive us for using :math:`Q_t`
to be the firm’s production at time :math:`t`, while below we use
:math:`Q` as the matrix in the quadratic form :math:`u_t' Q u_t` that
appears in the firm’s one-period profit function)

We can express the firm’s profit as a function of states and controls as

.. math::  \pi_t =  - (x_t' R x_t + u_t' Q u_t + 2 u_t' H x_t ) 

To form the matrices :math:`R, Q, H`, we note that the firm’s profits at
time :math:`t` function can be expressed

.. math::


   \begin{equation}
   \begin{split}
   \pi_{t} =&p_{t}S_{t}-c\left(Q_{t}\right)-d\left(I_{t},S_{t}\right)  \\
       =&\left(a_{0}-a_{1}S_{t}+v_{t}\right)S_{t}-c_{1}Q_{t}-c_{2}Q_{t}^{2}-d_{1}I_{t}-d_{2}\left(S_{t}-I_{t}\right)^{2}  \\
       =&a_{0}S_{t}-a_{1}S_{t}^{2}+Gz_{t}S_{t}-c_{1}Q_{t}-c_{2}Q_{t}^{2}-d_{1}I_{t}-d_{2}S_{t}^{2}-d_{2}I_{t}^{2}+2d_{2}S_{t}I_{t}  \\
       =&-\left(\underset{x_{t}^{\prime}Rx_{t}}{\underbrace{d_{1}I_{t}+d_{2}I_{t}^{2}}}\underset{u_{t}^{\prime}Qu_{t}}{\underbrace{+a_{1}S_{t}^{2}+d_{2}S_{t}^{2}+c_{2}Q_{t}^{2}}}\underset{2u_{t}^{\prime}Hx_{t}}{\underbrace{-a_{0}S_{t}-Gz_{t}S_{t}+c_{1}Q_{t}-2d_{2}S_{t}I_{t}}}\right) \\
       =&-\left(\left[\begin{array}{cc}
   I_{t} & z_{t}^{\prime}\end{array}\right]\underset{\equiv R}{\underbrace{\left[\begin{array}{cc}
   d_{2} & \frac{d_{1}}{2}S_{c}\\
   \frac{d_{1}}{2}S_{c}^{\prime} & 0
   \end{array}\right]}}\left[\begin{array}{c}
   I_{t}\\
   z_{t}
   \end{array}\right]+\left[\begin{array}{cc}
   Q_{t} & S_{t}\end{array}\right]\underset{\equiv Q}{\underbrace{\left[\begin{array}{cc}
   c_{2} & 0\\
   0 & a_{1}+d_{2}
   \end{array}\right]}}\left[\begin{array}{c}
   Q_{t}\\
   S_{t}
   \end{array}\right]+2\left[\begin{array}{cc}
   Q_{t} & S_{t}\end{array}\right]\underset{\equiv N}{\underbrace{\left[\begin{array}{cc}
   0 & \frac{c_{1}}{2}S_{c}\\
   -d_{2} & -\frac{a_{0}}{2}S_{c}-\frac{G}{2}
   \end{array}\right]}}\left[\begin{array}{c}
   I_{t}\\
   z_{t}
   \end{array}\right]\right)
   \end{split}
   \end{equation}

where :math:`S_{c}=\left[1,0\right]`.

**Remark on notation:** The notation for cross product term in the
QuantEcon library is :math:`N` instead of :math:`H`.

The firms’ optimum decision rule takes the form

.. math::  u_t = - F x_t 

and the evolution of the state under the optimal decision rule is

.. math::  x_{t+1} = (A - BF ) x_t + C \epsilon_{t+1} 

Here is code for computing an optimal decision rule and for analyzing
its consequences.

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code-block:: python3

    class smoothing_example:
        """
        Class for constructing, solving, and plotting results for
        inventories and sales smoothing problem.
        """
    
        def __init__(self,
                     β=0.96,           # Discount factor
                     c1=1,             # Cost-of-production
                     c2=1,
                     d1=1,             # Cost-of-holding inventories
                     d2=1,
                     a0=10,            # Inverse demand function
                     a1=1,
                     A22=[[1,   0],    # z process
                          [1, 0.9]],
                     C2=[[0], [1]],
                     G=[0, 1]):
    
            self.β = β
            self.c1, self.c2 = c1, c2
            self.d1, self.d2 = d1, d2
            self.a0, self.a1 = a0, a1
            self.A22 = np.atleast_2d(A22)
            self.C2 = np.atleast_2d(C2)
            self.G = np.atleast_2d(G)
    
            # Dimensions
            k, j = self.C2.shape        # Dimensions for randomness part
            n = k + 1                   # Number of states
            m = 2                       # Number of controls
            
            Sc = np.zeros(k)
            Sc[0] = 1
    
            # Construct matrices of transition law
            A = np.zeros((n, n))
            A[0, 0] = 1
            A[1:, 1:] = A22
    
            B = np.zeros((n, m))
            B[0, :] = 1, -1
    
            C = np.zeros((n, j))
            C[1:, :] = C2
    
            self.A, self.B, self.C = A, B, C
    
            # Construct matrices of one period profit function
            R = np.zeros((n, n))
            R[0, 0] = d2
            R[1:, 0] = d1 / 2 * Sc
            R[0, 1:] = d1 / 2 * Sc
    
            Q = np.zeros((m, m))
            Q[0, 0] = c2
            Q[1, 1] = a1 + d2
    
            N = np.zeros((m, n))
            N[1, 0] = - d2
            N[0, 1:] = c1 / 2 * Sc
            N[1, 1:] = - a0 / 2 * Sc - self.G / 2
    
            self.R, self.Q, self.N = R, Q, N
    
            # Construct LQ instance
            self.LQ = qe.LQ(Q, R, A, B, C, N, beta=β)
            self.LQ.stationary_values()
    
        def simulate(self, x0, T=100):
    
            c1, c2 = self.c1, self.c2
            d1, d2 = self.d1, self.d2
            a0, a1 = self.a0, self.a1
            G = self.G
    
            x_path, u_path, w_path = self.LQ.compute_sequence(x0, ts_length=T)
    
            I_path = x_path[0, :-1]
            z_path = x_path[1:, :-1]
            𝜈_path = (G @ z_path)[0, :]
    
            Q_path = u_path[0, :]
            S_path = u_path[1, :]
    
            revenue = (a0 - a1 * S_path + 𝜈_path) * S_path
            cost_production = c1 * Q_path + c2 * Q_path ** 2
            cost_inventories = d1 * I_path + d2 * (S_path - I_path) ** 2
    
            Q_no_inventory = (a0 + 𝜈_path - c1) / (2 * (a1 + c2))
            Q_hardwired = (a0 + 𝜈_path - c1) / (2 * (a1 + c2 + d2))
    
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
            ax[0, 0].plot(range(T), I_path, label="inventories")
            ax[0, 0].plot(range(T), S_path, label="sales")
            ax[0, 0].plot(range(T), Q_path, label="production")
            ax[0, 0].legend(loc=1)
            ax[0, 0].set_title("inventories, sales, and production")
    
            ax[0, 1].plot(range(T), (Q_path - S_path), color='b')
            ax[0, 1].set_ylabel("change in inventories", color='b')
            span = max(abs(Q_path - S_path))
            ax[0, 1].set_ylim(0-span*1.1, 0+span*1.1)
            ax[0, 1].set_title("demand shock and change in inventories")
    
            ax1_ = ax[0, 1].twinx()
            ax1_.plot(range(T), 𝜈_path, color='r')
            ax1_.set_ylabel("demand shock", color='r')
            span = max(abs(𝜈_path))
            ax1_.set_ylim(0-span*1.1, 0+span*1.1)
    
            ax1_.plot([0, T], [0, 0], '--', color='k')
    
            ax[1, 0].plot(range(T), revenue, label="revenue")
            ax[1, 0].plot(range(T), cost_production, label="cost_production")
            ax[1, 0].plot(range(T), cost_inventories, label="cost_inventories")
            ax[1, 0].legend(loc=1)
            ax[1, 0].set_title("profits decomposition")
    
            ax[1, 1].plot(range(T), Q_path, label="production")
            ax[1, 1].plot(range(T), Q_hardwired, label='production when  $I_t$ \
                forced to be zero')
            ax[1, 1].plot(range(T), Q_no_inventory, label='production when \
                inventories not useful')
            ax[1, 1].legend(loc=1)
            ax[1, 1].set_title('three production concepts')
    
            plt.show()

Notice that the above code sets parameters at the following default
values

-  discount factor β=0.96,

-  inverse demand function: :math:`a0=10, a1=1`

-  cost of production :math:`c1=1, c2=1`

-  costs of holding inventories :math:`d1=1, d2=1`

In the examples below, we alter some or all of these parameter values.

Example 1
=========

In this example, the demand shock follows AR(1) process:

.. math::


   \nu_t = \alpha + \rho \nu_{t-1} + \epsilon_t,

which implies

.. math::


   z_{t+1}=\left[\begin{array}{c}
   1\\
   v_{t+1}
   \end{array}\right]=\left[\begin{array}{cc}
   1 & 0\\
   \alpha & \rho
   \end{array}\right]\underset{z_{t}}{\underbrace{\left[\begin{array}{c}
   1\\
   v_{t}
   \end{array}\right]}}+\left[\begin{array}{c}
   0\\
   1
   \end{array}\right]\epsilon_{t+1}.

We set :math:`\alpha=1` and :math:`\rho=0.9`, their default values.

We’ll calculate and display outcomes, then discuss them below the
pertinent figures.

.. code-block:: python3

    ex1 = smoothing_example()
    
    x0 = [0, 1, 0]
    ex1.simulate(x0)






The figures above illustrate various features of an optimal production
plan.

Starting from zero inventories, the firm builds up a stock of
inventories and uses them to smooth costly production in the face of
demand shocks.

Optimal decisions evidently respond to demand shocks.

Inventories are always less than sales, so some sales come from current
production, a consequence of the cost, :math:`d_1 I_t` of holding
inventories.

The lower right panel shows differences between optimal production and
two alternative production concepts that come from altering the firm’s
cost structure – i.e., its technology.

These two concepts correspond to these distinct altered firm problems.

-  a setting in which inventories are not needed

-  a setting in which they are needed but we arbitrarily prevent the
   firm from holding inventories by forcing it to set :math:`I_t=0`
   always

We use these two alternative production concepts in order to shed light on the baseline model.   

Inventories Not Useful
=======================

Let’s turn first to the setting in which inventories aren’t needed.

In this problem, the firm forms an output plan that maximizes the expected
value of

.. math::  \sum_{t=0}^\infty \beta^t \{ p_t Q_t - C(Q_t) \} 

It turns out that the optimal plan for :math:`Q_t` for this problem also
solves a sequence of static problems
:math:`\max_{Q_t}\{p_t Q_t - c(Q_t)\}`.



When inventories aren’t required or used,  sales always equal
production.

This simplifies the problem and the optimal no-inventory production
maximizes the expected value of

.. math::


   \sum_{t=0}^{\infty}\beta^{t}\left\{ p_{t}Q_{t}-C\left(Q_{t}\right)\right\}.

The optimum decision rule is

.. math::


   Q_{t}^{ni}=\frac{a_{0}+\nu_{t}-c_{1}}{c_{2}+a_{1}}.

Inventories Useful but are Hardwired to be Zero Always
==========================================================

Next, we turn to a distinct problem in which inventories are useful –
meaning that there are costs of :math:`d_2 (I_t - S_t)^2` associated
with having sales not equal to inventories – but we arbitrarily impose on the firm
the costly restriction that it never hold inventories.

Here the firm’s maximization problem is

.. math::


   \max_{\{I_t, Q_t, S_t\}}\sum_{t=0}^{\infty}\beta^{t}\left\{ p_{t}S_{t}-C\left(Q_{t}\right)-d\left(I_{t},S_{t}\right)\right\}

subject to the restrictions that :math:`I_{t}=0` for all :math:`t` and
that :math:`I_{t+1}=I_{t}+Q_{t}-S_{t}`.

The restriction that :math:`I_t = 0` implies that :math:`Q_{t}=S_{t}`
and that the maximization problem reduces to

.. math::


   \max_{Q_t}\sum_{t=0}^{\infty}\beta^{t}\left\{ p_{t}Q_{t}-C\left(Q_{t}\right)-d\left(0,Q_{t}\right)\right\}

Here the optimal production plan is

.. math::


   Q_{t}^{h}=\frac{a_{0}+\nu_{t}-c_{1}}{c_{2}+a_{1}+d_{2}}.

We introduce this :math:`I_t` **is hardwired to zero** specification in
order to shed light on the role that inventories play by comparing outcomes
with those under our two other versions of the problem.

The bottom right panel displays an production path for the original
problem that we are interested in (the blue line) as well with an
optimal production path for the model in which inventories are not
useful (the green path) and also for the model in which, although
inventories are useful, they are hardwired to zero and the firm pays
cost :math:`d(0, Q_t)` for not setting sales :math:`S_t = Q_t` equal to
zero (the orange line).

Notice that it is typically optimal for the firm to produce more when
inventories aren’t useful. Here there is no requirement to sell out of
inventories and no costs from having sales deviate from inventories.

But “typical” does not mean “always”.

Thus, if we look closely, we notice that for small :math:`t`, the green
“production when inventories aren’t useful” line in the lower right
panel is below optimal production in the original model.

High optimal production in the original model early on occurs because the
firm wants to accumulate inventories quickly in order to acquire high 
inventories for use in later periods.

But how the green line compares to the blue line early on depends on the
evolution of the demand shock, as we will see in a
deterministically seasonal demand shock example to be analyzed below.

In that example,  the original firm optimally accumulates inventories slowly
because the next positive demand shock is in the distant future.

To make the green-blue model production comparison easier to see, let’s
confine the graphs to the first 10 periods:

.. code-block:: python3

    ex1.simulate(x0, T=10)




Example 2
=========

Next, we shut down randomness in demand and assume that the demand shock
:math:`\nu_t` follows a deterministic path:

.. math::


   \nu_t = \alpha + \rho \nu_{t-1}

Again, we’ll compute and display outcomes in some figures

.. code-block:: python3

    ex2 = smoothing_example(C2=[[0], [0]])
    
    x0 = [0, 1, 0]
    ex2.simulate(x0)






Example 3
=========

Now we’ll put randomness back into the demand shock process and also
assume that there are zero costs of holding inventories.

In particular, we’ll look at a situation in which :math:`d_1=0` but
:math:`d_2>0`.

Now it becomes optimal to set sales approximately equal to
inventories and to use inventories to smooth production quite well, as
the following figures confirm

.. code-block:: python3

    ex3 = smoothing_example(d1=0)
    
    x0 = [0, 1, 0]
    ex3.simulate(x0)






Example 4
=========

To bring out some features of the optimal policy that are related to
some technical issues in linear control theory, we’ll now temporarily
assume that it is costless to hold inventories.

When we completely shut down the cost of holding inventories by setting
:math:`d_1=0` and :math:`d_2=0`, something absurd happens (because the
Bellman equation is opportunistic and very smart).

(Technically, we have set parameters that end up violating conditions
needed to assure **stability** of the optimally controlled state.)

The firm finds it optimal to set
:math:`Q_t \equiv Q^* = \frac{-c_1}{2c_2}`, an output level that sets
the costs of production to zero (when :math:`c_1 >0`, as it is with our
default settings, then it is optimal to set production negative,
whatever that means!).

Recall the law of motion for inventories

.. math:: I_{t+1} = I_t + Q_t - S_t 

So when :math:`d_1=d_2= 0` so that the firm finds it optimal to set
:math:`Q_t = \frac{-c_1}{2c_2}` for all :math:`t`, then

.. math::  I_{t+1} - I_t = \frac{-c_1}{2c_2} - S_t < 0 

for almost all values of :math:`S_t` under our default parameters that
keep demand positive almost all of the time.

The dynamic program instructs the firm to set production costs to zero
and to **run a Ponzi scheme** by running inventories down forever.

(We can interpret this as the firm somehow **going short in** or
**borrowing** inventories)

The following figures confirm that inventories head south without limit

.. code-block:: python3

    ex4 = smoothing_example(d1=0, d2=0)
    
    x0 = [0, 1, 0]
    ex4.simulate(x0)




Let’s shorten the time span displayed in order to highlight what is
going on.

We’ll set the horizon :math:`T =30` with the following code

.. code-block:: python3

    # shorter period
    ex4.simulate(x0, T=30)





Example 5
==========

Now we’ll assume that the demand shock that follows a linear time trend

.. math::  v_t = b + a t  , a> 0, b> 0 

To represent this, we set
:math:`C_2 = \begin{bmatrix} 0 \cr 0 \end{bmatrix}` and

.. math::


   A_{22}=\left[\begin{array}{cc}
   1 & 0\\
   1 & 1
   \end{array}\right],x_{0}=\left[\begin{array}{c}
   1\\
   0
   \end{array}\right],
   G=\left[\begin{array}{cc}
   b & a\end{array}\right]

.. code-block:: python3

    # Set parameters
    a = 0.5
    b = 3.

.. code-block:: python3

    ex5 = smoothing_example(A22=[[1, 0], [1, 1]], C2=[[0], [0]], G=[b, a])
    
    x0 = [0, 1, 0] # set the initial inventory as 0
    ex5.simulate(x0, T=10)





Example 6
==========

Now we’ll assume a deterministically seasonal demand shock.

To represent this we’ll set

.. math::

    A_{22} = \begin{bmatrix}  1 & 0 & 0 & 0 & 0  \cr 0 & 0 & 0 & 0  & 1 \cr
        0 & 1 & 0 & 0 & 0 \cr
        0 & 0 & 1 & 0 & 0 \cr
        0 & 0 & 0 & 1 & 0 \end{bmatrix}, 
      C_2 = \begin{bmatrix} 0 \cr 0 \cr 0 \cr 0 \cr 0 \end{bmatrix},  G' = \begin{bmatrix} b \cr a \cr 0 \cr 0 \cr 0 
      \end{bmatrix}

where :math:`a > 0, b>0` and

.. math::  x_0 = \begin{bmatrix} 1 \cr 0 \cr 1 \cr 0 \cr 0 \end{bmatrix} 

.. code-block:: python3

    ex5 = smoothing_example(A22=[[1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 0]],
                            C2=[[0], [0], [0], [0], [0]],
                            G=[b, a, 0, 0, 0])
    
    x00 = [0, 1, 0, 1, 0, 0] # Set the initial inventory as 0
    ex5.simulate(x00, T=20)





Now we’ll generate some more examples that differ simply from the
initial **season** of the year in which we begin the demand shock

.. code-block:: python3

    x01 = [0, 1, 1, 0, 0, 0]
    ex5.simulate(x01, T=20)






.. code-block:: python3

    x02 = [0, 1, 0, 0, 1, 0]
    ex5.simulate(x02, T=20)






.. code-block:: python3

    x03 = [0, 1, 0, 0, 0, 1]
    ex5.simulate(x03, T=20)





Exercises
==============

Please try to analyze some inventory sales smoothing problems using the
``smoothing_example`` class.

Exercise 1
~~~~~~~~~~

Assume that the demand shock follows AR(2) process below:

.. math::


   \nu_{t}=\alpha+\rho_{1}\nu_{t-1}+\rho_{2}\nu_{t-2}+\epsilon_{t}.

You need to construct :math:`A22`, :math:`C`, and :math:`G` matrices
properly and then to input them as the keyword arguments of
``smoothing_example`` class. Simulate paths starting from the initial
condition :math:`x_0 = \left[0, 1, 0, 0\right]^\prime`.

After this, try to construct a very similar ``smoothing_example`` with
the same demand shock process but exclude the randomness
:math:`\epsilon_t`. Compute the stationary states :math:`\bar{x}` by
simulating for a long period. Then try to add shocks with different
magnitude to :math:`\bar{\nu}_t` and simulate paths. You should see how
firms respond differently by staring at the production plans.

Exercise 2
~~~~~~~~~~

Change parameters of :math:`C(Q_t)` and :math:`d(I_t, S_t)`.

1. Make production more costly, by setting :math:`c_2=5`.
2. Increase the cost of having inventories deviate from sales, by
   setting :math:`d_2=5`.

Solution 1
~~~~~~~~~~

.. code-block:: python3

    # set parameters
    α = 1
    ρ1 = 1.2
    ρ2 = -.3

.. code-block:: python3

    # construct matrices
    A22 =[[1,  0,  0],
              [1, ρ1, ρ2],
              [0,  1, 0]]
    C2 = [[0], [1], [0]]
    G = [0, 1, 0]

.. code-block:: python3

    ex1 = smoothing_example(A22=A22, C2=C2, G=G)
    
    x0 = [0, 1, 0, 0] # initial condition
    ex1.simulate(x0)

.. code-block:: python3

    # now silence the noise
    ex1_no_noise = smoothing_example(A22=A22, C2=[[0], [0], [0]], G=G)
    
    # initial condition
    x0 = [0, 1, 0, 0]
    
    # compute stationary states
    x_bar = ex1_no_noise.LQ.compute_sequence(x0, ts_length=250)[0][:, -1]
    x_bar

In the following, we add small and large shocks to :math:`\bar{\nu}_t`
and compare how firm responds differently in quantity. As the shock is
not very persistent under the parameterization we are using, we focus on
a short period response.

.. code-block:: python3

    T = 40

.. code-block:: python3

    # small shock
    x_bar1 = x_bar.copy()
    x_bar1[2] += 2
    ex1_no_noise.simulate(x_bar1, T=T)

.. code-block:: python3

    # large shock
    x_bar1 = x_bar.copy()
    x_bar1[2] += 10
    ex1_no_noise.simulate(x_bar1, T=T)

Solution 2
~~~~~~~~~~

.. code-block:: python3

    x0 = [0, 1, 0]

.. code-block:: python3

    smoothing_example(c2=5).simulate(x0)

.. code-block:: python3

    smoothing_example(d2=5).simulate(x0)



