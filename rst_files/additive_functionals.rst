.. _additive_functionals:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3


***********************
Additive Functionals
***********************


.. index::
    single: Models; Additive functionals

.. contents:: :depth: 2


**Co-authors: Chase Coleman and Balint Szoke**

Overview
=============

Some time series are nonstationary

For example, output, prices, and dividends are typically nonstationary, due to irregular but persistent growth

Which kinds of models are useful for studying such time series?

Hansen and Scheinkman :cite:`Hans_Scheink_2009` analyze two classes of time series models that accommodate growth

They are:

#.  **additive functionals** that display random "arithmetic growth"

#.  **multiplicative functionals** that display random "geometric growth"

These two classes of processes are closely connected

For example, if a process :math:`\{y_t\}` is an additive functional and :math:`\phi_t = \exp(y_t)`, then :math:`\{\phi_t\}` is a multiplicative functional

Hansen and Sargent :cite:`Hans_Sarg_book_2016` (chs. 5 and 6) describe discrete time versions of additive and multiplicative functionals

In this lecture we discuss the former (i.e., additive functionals)

In the :doc:`next lecture <multiplicative_functionals>` we discuss multiplicative functionals

We also consider fruitful decompositions of additive and multiplicative processes, a more in depth discussion of which can be found in Hansen and Sargent :cite:`Hans_Sarg_book_2016` 


A Particular Additive Functional
====================================

This lecture focuses on a particular type of additive functional: a scalar process :math:`\{y_t\}_{t=0}^\infty` whose increments are driven by a Gaussian vector autoregression

It is simple to construct, simulate, and analyze

This additive functional consists of two components, the first of which is a **first-order vector autoregression** (VAR) 

.. math::
    :label: old1_additive_functionals

    x_{t+1} = A x_t + B z_{t+1} 


Here 

* :math:`x_t` is an :math:`n \times 1` vector, 
  
* :math:`A` is an :math:`n \times n` stable matrix (all eigenvalues lie within the open unit circle),
  
* :math:`z_{t+1} \sim {\cal N}(0,I)` is an :math:`m \times 1` i.i.d. shock,
  
* :math:`B` is an :math:`n \times m` matrix, and

* :math:`x_0 \sim {\cal N}(\mu_0, \Sigma_0)` is a random initial condition for :math:`x`

The second component is an equation that expresses increments
of :math:`\{y_t\}_{t=0}^\infty` as linear functions of 

* a scalar constant :math:`\nu`, 
  
* the vector :math:`x_t`, and 
  
* the same Gaussian vector :math:`z_{t+1}` that appears in the VAR :eq:`old1_additive_functionals`

In particular,

.. math::
    :label: old2_additive_functionals

    y_{t+1} - y_{t} = \nu + D x_{t} + F z_{t+1}  


Here :math:`y_0 \sim {\cal N}(\mu_{y0}, \Sigma_{y0})` is a random
initial condition

The nonstationary random process :math:`\{y_t\}_{t=0}^\infty` displays
systematic but random *arithmetic growth*


A linear state space representation
------------------------------------

One way to represent the overall dynamics is to use a :doc:`linear state space system <linear_models>`

To do this, we set up state and observation vectors 

.. math::

    \hat{x}_t = \begin{bmatrix} 1 \\  x_t \\ y_t  \end{bmatrix} 
    \quad \text{and} \quad
    \hat{y}_t = \begin{bmatrix} x_t \\ y_t  \end{bmatrix}


Now we construct the state space system

.. math::

     \begin{bmatrix} 
         1 \\ 
         x_{t+1} \\ 
         y_{t+1}  
     \end{bmatrix} = 
     \begin{bmatrix} 
        1 & 0 & 0  \\ 
        0  & A & 0 \\ 
        \nu & D' &  1 \\ 
    \end{bmatrix} 
    \begin{bmatrix} 
        1 \\ 
        x_t \\ 
        y_t 
    \end{bmatrix} + 
    \begin{bmatrix} 
        0 \\  B \\ F'  
    \end{bmatrix} 
    z_{t+1} 


.. math::

    \begin{bmatrix} 
        x_t \\ 
        y_t  
    \end{bmatrix} 
    = \begin{bmatrix} 
        0  & I & 0  \\ 
        0 & 0  & 1 
    \end{bmatrix} 
    \begin{bmatrix} 
        1 \\  x_t \\ y_t  
    \end{bmatrix}


This can be written as

.. math::

    \begin{aligned}
      \hat{x}_{t+1} &= \hat{A} \hat{x}_t + \hat{B} z_{t+1} \\
      \hat{y}_{t} &= \hat{D} \hat{x}_t
    \end{aligned}


which is a standard linear state space system

To study it, we could map it into an instance of `LinearStateSpace <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py>`_ from `QuantEcon.py <http://quantecon.org/python_index.html>`_ 

We will in fact use a different set of code for simulation, for reasons described below




Dynamics
============

Let's run some simulations to build intuition

.. _addfunc_eg1:

In doing so we'll assume that :math:`z_{t+1}` is scalar and that :math:`\tilde x_t` follows a 4th-order scalar autoregession

.. math::
    :label: ftaf

    \tilde x_{t+1} = \phi_1 \tilde x_{t} + \phi_2 \tilde x_{t-1} +
    \phi_3 \tilde x_{t-2} +
    \phi_4 \tilde x_{t-3} + \sigma z_{t+1} 


Let the increment in :math:`\{y_t\}` obey

.. math::

    y_{t+1} - y_t =  \nu + \tilde x_t + \sigma z_{t+1}  


with an initial condition for :math:`y_0`

While :eq:`ftaf` is not a first order system like :eq:`old1_additive_functionals`, we know that it can be mapped  into a first order system

* for an example of such a mapping, see :ref:`this example <lss_sode>` 


In fact this whole model can be mapped into the additive functional system definition in :eq:`old1_additive_functionals` -- :eq:`old2_additive_functionals`  by appropriate selection of the matrices :math:`A, B, D, F`

You can try writing these matrices down now as an exercise --- the correct expressions will appear in the code below

Simulation
---------------

When simulating we embed our variables into a bigger system 

This system also constructs the components of the decompositions of :math:`y_t` and of :math:`\exp(y_t)` proposed by Hansen and Scheinkman :cite:`Hans_Scheink_2009`


All of these objects are computed using the code below


.. literalinclude:: /_static/code/additive_functionals/amflss.py

For now, we just plot :math:`y_t` and :math:`x_t`, postponing until later a description of exactly how we compute them

.. _addfunc_egcode:



.. code-block:: python3

    ϕ_1, ϕ_2, ϕ_3, ϕ_4 = 0.5, -0.2, 0, 0.5
    σ = 0.01
    ν = 0.01 # Growth rate
    
    # A matrix should be n x n
    A = np.array([[ϕ_1, ϕ_2, ϕ_3, ϕ_4],
                  [ 1,   0,    0,   0],
                  [ 0,   1,    0,   0],
                  [ 0,   0,    1,   0]])
    
    # B matrix should be n x k
    B = np.array([[σ, 0, 0, 0]]).T
    
    D = np.array([1, 0, 0, 0]) @ A
    F = np.array([1, 0, 0, 0]) @ B

    amf = AMF_LSS_VAR(A, B, D, F, ν=ν)

    T = 150
    x, y = amf.lss.simulate(T)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 9))
    
    ax[0].plot(np.arange(T), y[amf.nx, :], color='k')
    ax[0].set_title('A particular path of $y_t$')
    ax[1].plot(np.arange(T), y[0, :], color='g')
    ax[1].axhline(0, color='k', linestyle='-.')
    ax[1].set_title('Associated path of $x_t$')
    plt.show()




Notice the irregular but persistent growth in :math:`y_t`


Decomposition
---------------

Hansen and Sargent :cite:`Hans_Sarg_book_2016` describe how to construct a decomposition of
an additive functional into four parts:

-  a constant inherited from initial values :math:`x_0` and :math:`y_0`

-  a linear trend

-  a martingale

-  an (asymptotically) stationary component

To attain this decomposition for the particular class of additive
functionals defined by :eq:`old1_additive_functionals` and :eq:`old2_additive_functionals`, we first construct the matrices

.. math::

    \begin{aligned}
      H & := F + B'(I - A')^{-1} D 
      \\
      g & := D' (I - A)^{-1}
    \end{aligned}


Then the Hansen-Scheinkman :cite:`Hans_Scheink_2009` decomposition is

.. math::

    \begin{aligned}
      y_t 
      &= \underbrace{t \nu}_{\text{trend component}} + 
         \overbrace{\sum_{j=1}^t H z_j}^{\text{Martingale component}} - 
         \underbrace{g x_t}_{\text{stationary component}} + 
         \overbrace{g x_0 + y_0}^{\text{initial conditions}}
    \end{aligned}


At this stage you should pause and verify that :math:`y_{t+1} - y_t` satisfies :eq:`old2_additive_functionals`

It is convenient for us to introduce the following notation:

-  :math:`\tau_t = \nu t` , a linear, deterministic trend

-  :math:`m_t = \sum_{j=1}^t H z_j`, a martingale with time :math:`t+1` increment :math:`H z_{t+1}`

-  :math:`s_t = g x_t`, an (asymptotically) stationary component

We want to characterize and simulate components :math:`\tau_t, m_t, s_t` of the decomposition.

A convenient way to do this is to construct an appropriate instance of a :doc:`linear state space system <linear_models>` by using `LinearStateSpace <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py>`_ from `QuantEcon.py <http://quantecon.org/python_index.html>`_

This will allow us to use the routines in `LinearStateSpace <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py>`_ to study dynamics

To start, observe that, under the dynamics in :eq:`old1_additive_functionals` and :eq:`old2_additive_functionals` and with the
definitions just given,

.. math::

    \begin{bmatrix} 
        1 \\ 
        t+1 \\ 
        x_{t+1} \\ 
        y_{t+1} \\ 
        m_{t+1} 
    \end{bmatrix} = 
    \begin{bmatrix} 
        1 & 0 & 0 & 0 & 0 \\ 
        1 & 1 & 0 & 0 & 0 \\ 
        0 & 0 & A & 0 & 0 \\ 
        \nu & 0 & D' & 1 & 0 \\ 
        0 & 0 & 0 & 0 & 1 
    \end{bmatrix} 
    \begin{bmatrix} 
        1 \\ 
        t \\ 
        x_t \\ 
        y_t \\ 
        m_t 
    \end{bmatrix} +
    \begin{bmatrix} 
        0 \\ 
        0 \\ 
        B \\ 
        F' \\ 
        H' 
    \end{bmatrix} 
    z_{t+1} 


and

.. math::

    \begin{bmatrix} 
        x_t \\ 
        y_t \\ 
        \tau_t \\ 
        m_t \\ 
        s_t 
    \end{bmatrix} = 
    \begin{bmatrix} 
        0 & 0 & I & 0 & 0 \\ 
        0 & 0 & 0 & 1 & 0 \\ 
        0 & \nu & 0 & 0 & 0 \\ 
        0 & 0 & 0 & 0 & 1 \\ 
        0 & 0 & -g & 0 & 0 
    \end{bmatrix} 
    \begin{bmatrix} 
        1 \\ 
        t \\ 
        x_t \\ 
        y_t \\ 
        m_t 
    \end{bmatrix}


With

.. math::

    \tilde{x} := \begin{bmatrix} 1 \\ t \\ x_t \\ y_t \\ m_t \end{bmatrix} 
    \quad \text{and} \quad
    \tilde{y} := \begin{bmatrix} x_t \\ y_t \\ \tau_t \\ m_t \\ s_t \end{bmatrix}


we can write this as the linear state space system

.. math::

    \begin{aligned}
      \tilde{x}_{t+1} &= \tilde{A} \tilde{x}_t + \tilde{B} z_{t+1} \\
      \tilde{y}_{t} &= \tilde{D} \tilde{x}_t
    \end{aligned}


By picking out components of :math:`\tilde y_t`, we can track all variables of
interest

Code
=====================

The class `AMF_LSS_VAR <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/additive_functionals/amflss.py>`__ mentioned above does all that we want to study our additive functional

In fact `AMF_LSS_VAR <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/additive_functionals/amflss.py>`__ does more, as we shall explain below

(A hint that it does more is the name of the class -- here AMF stands for
"additive and multiplicative functional" -- the code will do things for
multiplicative functionals too)
   
Let's use this code (embedded above) to explore the :ref:`example process described above <addfunc_eg1>`

If you run :ref:`the code that first simulated that example <addfunc_egcode>` again and then the method call
you will generate (modulo randomness) the plot



.. code-block:: python3

    amf.plot_additive(T)
    plt.show()



When we plot multiple realizations of a component in the 2nd, 3rd, and 4th panels, we also plot population 95% probability coverage sets computed using the LinearStateSpace class

We have chosen to simulate many paths, all starting from the *same* nonrandom initial conditions :math:`x_0, y_0` (you can tell this from the shape of the 95% probability coverage shaded areas)

Notice tell-tale signs of these probability coverage shaded areas

*  the purple one for the martingale component :math:`m_t` grows with
   :math:`\sqrt{t}`

*  the green one for the stationary component :math:`s_t` converges to a
   constant band


An associated multiplicative functional
------------------------------------------

Where :math:`\{y_t\}` is our additive functional, let :math:`M_t = \exp(y_t)`

As mentioned above, the process :math:`\{M_t\}` is called a **multiplicative functional**

Corresponding to the additive decomposition described above we have the multiplicative decomposition of the :math:`M_t`

.. math::

    \frac{M_t}{M_0} 
    = \exp (t \nu) \exp \Bigl(\sum_{j=1}^t H \cdot Z_j \Bigr) \exp \biggl( D'(I-A)^{-1} x_0 - D'(I-A)^{-1} x_t \biggr)  


or

.. math::

    \frac{M_t}{M_0} =  \exp\left( \tilde \nu t \right) \Biggl( \frac{\widetilde M_t}{\widetilde M_0}\Biggr) \left( \frac{\tilde e (X_0)} {\tilde e(x_t)} \right)


where

.. math::

    \tilde \nu =  \nu + \frac{H \cdot H}{2} ,
    \quad
    \widetilde M_t = \exp \biggl( \sum_{j=1}^t \biggl(H \cdot z_j -\frac{ H \cdot H }{2} \biggr) \biggr),  \quad \widetilde M_0 =1 


and

.. math::

    \tilde e(x) = \exp[g(x)] = \exp \bigl[ D' (I - A)^{-1} x \bigr]


An instance of class `AMF_LSS_VAR <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/additive_functionals/amflss.py>`__ includes this associated multiplicative functional as an attribute

Let's plot this multiplicative functional for our example



If you run :ref:`the code that first simulated that example <addfunc_egcode>` again and then the method call



.. code-block:: python3

    amf.plot_multiplicative(T)
    plt.show()



As before, when we plotted multiple realizations of a component in the 2nd, 3rd, and 4th panels, we also plotted population 95% confidence bands computed using the LinearStateSpace class

Comparing this figure and the last also helps show how geometric growth differs from
arithmetic growth 


A peculiar large sample property
---------------------------------

Hansen and Sargent :cite:`Hans_Sarg_book_2016` (ch. 6) note that the martingale component
:math:`\widetilde M_t` of the multiplicative decomposition has a peculiar property

*  While :math:`E_0 \widetilde M_t = 1` for all :math:`t \geq 0`,
   nevertheless :math:`\ldots`

*  As :math:`t \rightarrow +\infty`, :math:`\widetilde M_t` converges to
   zero almost surely

The following simulation of many paths of :math:`\widetilde M_t` illustrates this property



.. code-block:: python3

    np.random.seed(10021987)
    amf.plot_martingales(12000)
    plt.show()





