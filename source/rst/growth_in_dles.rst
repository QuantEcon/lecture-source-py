.. _growth_in_dles:

.. include:: /_static/includes/header.raw


.. index::
    single: python

**********************************
Growth in Dynamic Linear Economies
**********************************

.. contents:: :depth: 2

**Co-author:** Sebastian Graves



This is another member of a suite of lectures that use the quantecon DLE class to instantiate models within the
:cite:`HS2013` class of models described in detail in :doc:`Recursive Models of Dynamic Linear Economies<hs_recursive_models>`.

In addition to what's included in  Anaconda, this lecture uses the quantecon library.

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

This lecture describes several complete market economies having a
common linear-quadratic-Gaussian structure.

Three examples of such economies show how the DLE class can be used to
compute equilibria of such economies in Python and to illustrate how
different versions of these economies can or cannot generate sustained
growth.


We require the following imports

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from quantecon import LQ, DLE
    

Common Structure
================

Our example economies have the following features

-  Information flows are governed by an exogenous stochastic process
   :math:`z_t` that follows

   .. math:: z_{t+1} = A_{22}z_t + C_2w_{t+1}

   where :math:`w_{t+1}` is a martingale difference sequence.

-  Preference shocks :math:`b_t` and technology shocks :math:`d_t` are
   linear functions of :math:`z_t`

   .. math:: b_t = U_bz_t

   .. math:: d_t = U_dz_t

-  Consumption and physical investment goods are produced using the
   following technology

   .. math::  \Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t

   .. math:: k_t = \Delta_k k_{t-1} + \Theta_k i_t

   .. math:: g_t \cdot g_t = l_t^2

   where :math:`c_t` is a vector of consumption goods, :math:`g_t` is a
   vector of intermediate goods, :math:`i_t` is a vector of investment
   goods, :math:`k_t` is a vector of physical capital goods, and
   :math:`l_t` is the amount of labor supplied by the representative
   household.

-  Preferences of a representative household are described by

   .. math::  -\frac{1}{2}\mathbb{E}\sum_{t=0}^\infty \beta^t [(s_t-b_t)\cdot(s_t - b_t) + l_t^2], 0 < \beta < 1

   .. math::  s_t = \Lambda h_{t-1} + \Pi c_t

   .. math::  h_t = \Delta_h h_{t-1} + \Theta_h c_t


where :math:`s_t` is a vector of consumption services, and
:math:`h_t` is a vector of household capital stocks.


Thus, an instance of this class of economies is described by the
matrices

.. math::  \{ A_{22}, C_2, U_b, U_d, \Phi_c, \Phi_g, \Phi_i, \Gamma, \Delta_k, \Theta_k,\Lambda, \Pi, \Delta_h, \Theta_h \}

and the scalar :math:`\beta`.

A Planning Problem
==================

The first welfare theorem asserts that a competitive equilibrium
allocation solves the following planning problem.

Choose :math:`\{c_t, s_t, i_t, h_t, k_t, g_t\}_{t=0}^\infty` to maximize

   .. math:: - \frac{1}{2}\mathbb{E}\sum_{t=0}^\infty \beta^t [(s_t-b_t)\cdot(s_t - b_t) + g_t \cdot g_t]

subject to the linear constraints

.. math::  \Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t

.. math:: k_t = \Delta_k k_{t-1} + \Theta_k i_t

.. math::  h_t = \Delta_h h_{t-1} + \Theta_h c_t

.. math::  s_t = \Lambda h_{t-1} + \Pi c_t

and

.. math:: z_{t+1} = A_{22}z_t + C_2w_{t+1}

.. math:: b_t = U_bz_t

.. math:: d_t = U_dz_t

The DLE class in Python maps this planning problem into a linear-quadratic dynamic programming problem and then solves it by using
QuantEcon's LQ class.

(See Section 5.5 of Hansen & Sargent (2013) :cite:`HS2013` for a full
description of how to map these economies into an LQ setting, and how to
use the solution to the LQ problem to construct the output matrices in
order to simulate the economies)

The state for the LQ problem is

.. math::

   x_t =
   \left[ {\begin{array}{c}
   h_{t-1} \\ k_{t-1} \\ z_t
   \end{array} }
   \right]

and the control variable is :math:`u_t = i_t`.

Once the LQ problem has been solved, the law of motion for the state is

.. math:: x_{t+1} = (A-BF)x_t + Cw_{t+1}

where the optimal control law is :math:`u_t = -Fx_t`.

Letting :math:`A^o = A-BF` we write this law of motion as

.. math:: x_{t+1} = A^ox_t + Cw_{t+1}

Example Economies
=================

Each of the example economies shown here will share a number of
components. In particular, for each we will consider preferences of the
form

.. math:: - \frac{1}{2}\mathbb{E}\sum_{t=0}^\infty \beta^t [(s_t-b_t)^2 + l_t^2], 0 < \beta < 1

.. math::  s_t = \lambda h_{t-1} + \pi c_t

.. math::  h_t = \delta_h h_{t-1} + \theta_h c_t

.. math:: b_t = U_bz_t

Technology of the form

.. math::  c_t + i_t = \gamma_1 k_{t-1} + d_{1t}

.. math:: k_t = \delta_k k_{t-1} + i_t

.. math::  g_t = \phi_1 i_t \, , \phi_1 > 0

.. math::

    \left[ {\begin{array}{c}
      d_{1t} \\ 0
      \end{array} }
      \right] = U_dz_t

And information of the form

.. math::

    z_{t+1} =
   \left[ {\begin{array}{ccc}
      1 & 0 & 0 \\ 0 & 0.8 & 0 \\ 0 & 0 & 0.5
      \end{array} }
      \right]
      z_t +
       \left[ {\begin{array}{cc}
      0 & 0 \\ 1 & 0 \\ 0 & 1
      \end{array} }
      \right]
      w_{t+1}

.. math::

    U_b =
      \left[ {\begin{array}{ccc}
      30 & 0 & 0
      \end{array} }
      \right]

.. math::

   U_d =
      \left[ {\begin{array}{ccc}
      5 & 1 & 0 \\ 0 & 0 & 0
      \end{array} }
      \right]

We shall vary
:math:`\{\lambda, \pi, \delta_h, \theta_h, \gamma_1, \delta_k, \phi_1\}`
and the initial state :math:`x_0` across the three economies.

Example 1: Hall (1978)
----------------------

First, we set parameters such that consumption follows a random walk. In
particular, we set

.. math::  \lambda = 0, \pi = 1, \gamma_1 = 0.1, \phi_1 = 0.00001, \delta_k = 0.95, \beta = \frac{1}{1.05}

(In this economy :math:`\delta_h` and :math:`\theta_h` are arbitrary as
household capital does not enter the equation for consumption services
We set them to values that will become useful in Example 3)

It is worth noting that this choice of parameter values ensures that
:math:`\beta(\gamma_1 + \delta_k) = 1`.

For simulations of this economy, we choose an initial condition of

.. math::

   x_0 =
      \left[ {\begin{array}{ccccc}
      5 & 150 & 1 & 0 & 0
      \end{array} }
      \right]'

.. code-block:: python3

    # Parameter Matrices
    γ_1 = 0.1
    ϕ_1 = 1e-5

    ϕ_c, ϕ_g, ϕ_i, γ, δ_k, θ_k = (np.array([[1], [0]]),
                                  np.array([[0], [1]]),
                                  np.array([[1], [-ϕ_1]]),
                                  np.array([[γ_1], [0]]),
                                  np.array([[.95]]),
                                  np.array([[1]]))

    β, l_λ, π_h, δ_h, θ_h = (np.array([[1 / 1.05]]),
                             np.array([[0]]),
                             np.array([[1]]),
                             np.array([[.9]]),
                             np.array([[1]]) - np.array([[.9]]))

    a22, c2, ub, ud = (np.array([[1,   0,   0],
                                 [0, 0.8,   0],
                                 [0,   0, 0.5]]),
                       np.array([[0, 0],
                                 [1, 0],
                                 [0, 1]]),
                       np.array([[30, 0, 0]]),
                       np.array([[5, 1, 0],
                                 [0, 0, 0]]))

    # Initial condition
    x0 = np.array([[5], [150], [1], [0], [0]])

    info1 = (a22, c2, ub, ud)
    tech1 = (ϕ_c, ϕ_g, ϕ_i, γ, δ_k, θ_k)
    pref1 = (β, l_λ, π_h, δ_h, θ_h)

These parameter values are used to define an economy of the DLE class.

.. code-block:: python3

    econ1 = DLE(info1, tech1, pref1)

We can then simulate the economy for a chosen length of time, from our
initial state vector :math:`x_0`

.. code-block:: python3

    econ1.compute_sequence(x0, ts_length=300)

The economy stores the simulated values for each variable. Below we plot
consumption and investment

.. code-block:: python3

    # This is the right panel of Fig 5.7.1 from p.105 of HS2013
    plt.plot(econ1.c[0], label='Cons.')
    plt.plot(econ1.i[0], label='Inv.')
    plt.legend()
    plt.show()


Inspection of the plot shows that the sample paths of consumption and
investment drift in ways that suggest that each has or nearly has a
**random walk** or **unit root** component.

This is confirmed by checking the eigenvalues of :math:`A^o`

.. code-block:: python3

    econ1.endo, econ1.exo


The endogenous eigenvalue that appears to be unity reflects the random
walk character of consumption in Hall's model.

-  Actually, the largest endogenous eigenvalue is very slightly below 1.

-  This outcome comes from the small adjustment cost :math:`\phi_1`.

.. code-block:: python3

    econ1.endo[1]




The fact that the largest endogenous eigenvalue is strictly less than
unity in modulus means that it is possible to compute the non-stochastic
steady state of consumption, investment and capital.

.. code-block:: python3

    econ1.compute_steadystate()
    np.set_printoptions(precision=3, suppress=True)
    print(econ1.css, econ1.iss, econ1.kss)



However, the near-unity endogenous eigenvalue means that these steady
state values are of little relevance.

Example 2: Altered Growth Condition
-----------------------------------

We generate our next economy by making two alterations to the parameters
of Example 1.

-  First, we raise :math:`\phi_1` from 0.00001 to 1.

   -  This will lower the endogenous eigenvalue that is close to 1,
      causing the economy to head more quickly to the vicinity of its
      non-stochastic steady-state.

-  Second, we raise :math:`\gamma_1` from 0.1 to 0.15.

   -  This has the effect of raising the optimal steady-state value of
      capital.

We also start the economy off from an initial condition with a lower
capital stock

.. math::

   x_0 =
      \left[ {\begin{array}{ccccc}
      5 & 20 & 1 & 0 & 0
      \end{array} }
      \right]'

Therefore, we need to define the following new parameters

.. code-block:: python3

    γ2 = 0.15
    γ22 = np.array([[γ2], [0]])

    ϕ_12 = 1
    ϕ_i2 = np.array([[1], [-ϕ_12]])

    tech2 = (ϕ_c, ϕ_g, ϕ_i2, γ22, δ_k, θ_k)

    x02 = np.array([[5], [20], [1], [0], [0]])

Creating the DLE class and then simulating gives the following plot for
consumption and investment

.. code-block:: python3

    econ2 = DLE(info1, tech2, pref1)

    econ2.compute_sequence(x02, ts_length=300)

    plt.plot(econ2.c[0], label='Cons.')
    plt.plot(econ2.i[0], label='Inv.')
    plt.legend()
    plt.show()



Simulating our new economy shows that consumption grows quickly in the
early stages of the sample.

However, it then settles down around the new non-stochastic steady-state
level of consumption of 17.5, which we find as follows

.. code-block:: python3

    econ2.compute_steadystate()
    print(econ2.css, econ2.iss, econ2.kss)



The economy converges faster to this level than in Example 1 because the
largest endogenous eigenvalue of :math:`A^o` is now significantly lower
than 1.

.. code-block:: python3

    econ2.endo, econ2.exo




Example 3: A Jones-Manuelli (1990) Economy
------------------------------------------

For our third economy, we choose parameter values with the aim of
generating *sustained* growth in consumption, investment and capital.

To do this, we set parameters so that Jones and Manuelli's "growth
condition" is just satisfied.

In our notation, just satisfying the growth condition is actually
equivalent to setting :math:`\beta(\gamma_1 + \delta_k) = 1`, the
condition that was necessary for consumption to be a random walk in
Hall's model.

Thus, we lower :math:`\gamma_1` back to 0.1.

In our model, this is a necessary but not sufficient condition for
growth.

To generate growth we set preference parameters to reflect habit
persistence.

In particular, we set :math:`\lambda = -1`, :math:`\delta_h = 0.9` and
:math:`\theta_h = 1 - \delta_h = 0.1`.

This makes preferences assume the form

.. math:: - \frac{1}{2}\mathbb{E}\sum_{t=0}^\infty \beta^t [(c_t-b_t - (1-\delta_h)\sum_{j=0}^\infty \delta_h^jc_{t-j-1})^2 + l_t^2]

These preferences reflect habit persistence

-  the effective "bliss point"
   :math:`b_t + (1-\delta_h)\sum_{j=0}^\infty \delta_h^jc_{t-j-1}` now
   shifts in response to a moving average of past consumption

Since :math:`\delta_h` and :math:`\theta_h` were defined earlier, the
only change we need to make from the parameters of Example 1 is to
define the new value of :math:`\lambda`.

.. code-block:: python3

    l_λ2 = np.array([[-1]])
    pref2 = (β, l_λ2, π_h, δ_h, θ_h)

.. code-block:: python3

    econ3 = DLE(info1, tech1, pref2)

We simulate this economy from the original state vector

.. code-block:: python3

    econ3.compute_sequence(x0, ts_length=300)

    # This is the right panel of Fig 5.10.1 from p.110 of HS2013
    plt.plot(econ3.c[0], label='Cons.')
    plt.plot(econ3.i[0], label='Inv.')
    plt.legend()
    plt.show()



Thus, adding habit persistence to the Hall model of Example 1 is enough
to generate sustained growth in our economy.

The eigenvalues of
:math:`A^o` in this new economy are

.. code-block:: python3

    econ3.endo, econ3.exo




We now have two unit endogenous eigenvalues. One stems from satisfying
the growth condition (as in Example 1).

The other unit eigenvalue results from setting :math:`\lambda = -1`.

To show the importance of both of these for generating growth, we
consider the following experiments.

Example 3.1: Varying Sensitivity
-------------------------------------------

Next we raise :math:`\lambda` to -0.7

.. code-block:: python3

    l_λ3 = np.array([[-0.7]])
    pref3 = (β, l_λ3, π_h, δ_h, θ_h)

    econ4 = DLE(info1, tech1, pref3)

    econ4.compute_sequence(x0, ts_length=300)

    plt.plot(econ4.c[0], label='Cons.')
    plt.plot(econ4.i[0], label='Inv.')
    plt.legend()
    plt.show()



We no longer achieve sustained growth if :math:`\lambda` is raised from -1 to -0.7.

This is related to the fact that one of the endogenous
eigenvalues is now less than 1.

.. code-block:: python3

    econ4.endo, econ4.exo



Example 3.2: More Impatience
-----------------------------------------

Next let's lower :math:`\beta` to 0.94

.. code-block:: python3

    β_2 = np.array([[0.94]])
    pref4 = (β_2, l_λ, π_h, δ_h, θ_h)

    econ5 = DLE(info1, tech1, pref4)

    econ5.compute_sequence(x0, ts_length=300)

    plt.plot(econ5.c[0], label='Cons.')
    plt.plot(econ5.i[0], label='Inv.')
    plt.legend()
    plt.show()

Growth also fails if we lower :math:`\beta`, since we now have
:math:`\beta(\gamma_1 + \delta_k) < 1`.

Consumption and investment explode downwards, as a lower value of
:math:`\beta` causes the representative consumer to front-load
consumption.

This explosive path shows up in the second endogenous eigenvalue now
being larger than one.

.. code-block:: python3

    econ5.endo, econ5.exo
