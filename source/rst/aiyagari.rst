.. _aiyagari:

.. include:: /_static/includes/header.raw

.. highlight:: python3


*******************************************************
The Aiyagari Model
*******************************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
============

In this lecture, we describe the structure of a class of models that build on work by Truman Bewley :cite:`Bewley1977`.

.. only:: html

    We begin by discussing an example of a Bewley model due to :download:`Rao Aiyagari <_static/lecture_specific/aiyagari/aiyagari_obit.pdf>`.

.. only:: latex

    We begin by discussing an example of a Bewley model due to `Rao Aiyagari <https://lectures.quantecon.org/_downloads/aiyagari_obit.pdf>`__.

The model features

* Heterogeneous agents

* A single exogenous vehicle for borrowing and lending

* Limits on amounts individual agents may borrow


The Aiyagari model has been used to investigate many topics, including

* precautionary savings and the effect of liquidity constraints :cite:`Aiyagari1994`

* risk sharing and asset pricing :cite:`Heaton1996`

* the shape of the wealth distribution :cite:`benhabib2015`

* etc., etc., etc.




References
-------------

The primary reference for this lecture is :cite:`Aiyagari1994`.

A textbook treatment is available in chapter 18 of :cite:`Ljungqvist2012`.

A continuous time version of the model by SeHyoun Ahn and Benjamin Moll can be found `here <http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/aiyagari_continuous_time.ipynb>`__.


The Economy
==============





Households
---------------


Infinitely lived households / consumers face idiosyncratic income shocks.

A unit interval of  *ex-ante* identical households face a common borrowing constraint.

The savings problem faced by a typical  household is

.. math::

    \max \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)


subject to

.. math::

    a_{t+1} + c_t \leq w z_t + (1 + r) a_t
    \quad
    c_t \geq 0,
    \quad \text{and} \quad
    a_t \geq -B


where

* :math:`c_t` is current consumption

* :math:`a_t` is assets

* :math:`z_t` is an exogenous component of labor income capturing stochastic unemployment risk, etc.

* :math:`w` is a wage rate

* :math:`r` is a net interest rate

* :math:`B` is the maximum amount that the agent is allowed to borrow

The exogenous process :math:`\{z_t\}` follows a finite state Markov chain with given stochastic matrix :math:`P`.

The wage and interest rate are fixed over time.

In this simple version of the model, households supply labor  inelastically because they do not value leisure.



Firms
=======


Firms produce output by hiring capital and labor.

Firms act competitively and face constant returns to scale.

Since returns to scale are constant the number of firms does not matter.

Hence we can consider a single (but nonetheless competitive) representative firm.

The firm's output is

.. math::

    Y_t = A K_t^{\alpha} N^{1 - \alpha}


where

* :math:`A` and :math:`\alpha` are parameters with :math:`A > 0` and :math:`\alpha \in (0, 1)`

* :math:`K_t` is aggregate capital

* :math:`N` is total labor supply (which is constant in this simple version of the model)


The firm's problem is

.. math::

    max_{K, N} \left\{ A K_t^{\alpha} N^{1 - \alpha} - (r + \delta) K - w N \right\}


The parameter :math:`\delta` is the depreciation rate.


From the first-order condition with respect to capital, the firm's inverse demand for capital is

.. math::
    :label: aiy_rgk

    r = A \alpha  \left( \frac{N}{K} \right)^{1 - \alpha} - \delta


Using this expression and the firm's first-order condition for labor, we can pin down
the equilibrium wage rate as a function of :math:`r` as

.. math::
    :label: aiy_wgr

    w(r) = A  (1 - \alpha)  (A \alpha / (r + \delta))^{\alpha / (1 - \alpha)}


Equilibrium
-----------------

We construct  a *stationary rational expectations equilibrium* (SREE).

In such an equilibrium

* prices induce behavior that generates aggregate quantities consistent with the prices

* aggregate quantities and prices are constant over time


In more detail, an SREE lists a set of prices, savings and production policies such that

* households want to choose the specified savings policies taking the prices as given

* firms maximize profits taking the same prices as given

* the resulting aggregate quantities are consistent with the prices; in particular, the demand for capital equals the supply

* aggregate quantities (defined as cross-sectional averages) are constant


In practice, once parameter values are set, we can check for an SREE by the following steps

#. pick a proposed quantity :math:`K` for aggregate capital

#. determine corresponding prices, with interest rate :math:`r` determined by :eq:`aiy_rgk` and a wage rate :math:`w(r)` as given in :eq:`aiy_wgr`

#. determine the common optimal savings policy of the households given these prices

#. compute aggregate capital as the mean of steady state capital given this savings policy

If this final quantity agrees with :math:`K` then we have a SREE



Code
====

Let's look at how we might compute such an equilibrium in practice.

To solve the household's dynamic programming problem we'll use the `DiscreteDP <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/ddp.py>`_ class from `QuantEcon.py <http://quantecon.org/python_index.html>`_.

Our first task is the least exciting one: write code that maps parameters for a household problem into the ``R`` and ``Q`` matrices needed to generate an instance of ``DiscreteDP``.

Below is a piece of boilerplate code that does just this.

In reading the code, the following information will be helpful

* ``R`` needs to be a matrix where ``R[s, a]`` is the reward at state ``s`` under action ``a``

* ``Q`` needs to be a three-dimensional array where ``Q[s, a, s']`` is the probability of transitioning to state ``s'`` when the current state is ``s`` and the current action is ``a``


(For a detailed discussion of ``DiscreteDP`` see :doc:`this lecture <discrete_dp>`)

Here we take the state to be :math:`s_t := (a_t, z_t)`, where :math:`a_t` is assets and :math:`z_t` is the shock.

The action is the choice of next period asset level :math:`a_{t+1}`.



We use Numba to speed up the loops so we can update the matrices efficiently
when the parameters change.



The class also includes a default set of parameters that we'll adopt unless otherwise specified

.. literalinclude:: /_static/lecture_specific/aiyagari/aiyagari_household.py

As a first example of what we can do, let's compute and plot an optimal accumulation policy at fixed prices

.. literalinclude:: /_static/lecture_specific/aiyagari/aiyagari_compute_policy.py

The plot shows asset accumulation policies at different values of the exogenous state.



Now we want to calculate the equilibrium.

Let's do this visually as a first pass.

The following code draws aggregate supply and demand curves.

The intersection gives equilibrium interest rates and capital


.. literalinclude:: /_static/lecture_specific/aiyagari/aiyagari_compute_equilibrium.py
