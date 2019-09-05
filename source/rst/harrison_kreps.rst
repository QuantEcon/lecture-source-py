.. _harrison_kreps:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*************************************
Asset Pricing III: Incomplete Markets
*************************************

.. index::
    single: Models; Harrison Kreps

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
=========

This lecture describes a version of a model of Harrison and Kreps :cite:`HarrKreps1978`.

The model determines the price of a dividend-yielding asset that is traded by two types of self-interested investors.

The model features

* heterogeneous beliefs

* incomplete markets

* short sales constraints, and possibly :math:`\ldots`

* (leverage) limits on an investor's ability to borrow in order to finance purchases of a risky asset


Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import scipy.linalg as la


References
----------

Prior to reading the following, you might like to review our lectures on

* :doc:`Markov chains <finite_markov>`

* :doc:`Asset pricing with finite state space <markov_asset>`


Bubbles
-------

Economists differ in how they define a *bubble*.

The Harrison-Kreps model illustrates the following notion of a bubble that attracts many economists:

    *A component of an asset price can be interpreted as a bubble when all investors agree that the current price of the asset exceeds what they believe the asset's underlying dividend stream justifies*.


Structure of the Model
======================


The model simplifies by ignoring alterations in the distribution of wealth
among investors having different beliefs about the fundamentals that determine
asset payouts.

There is a fixed number :math:`A` of shares of an asset.

Each share entitles its owner to a stream of dividends :math:`\{d_t\}` governed by a Markov chain defined on a state space :math:`S \in \{0, 1\}`.

The dividend obeys

.. math::

    d_t =
    \begin{cases}
        0 & \text{ if } s_t = 0 \\
        1 & \text{ if } s_t = 1
    \end{cases}


The owner of a share at the beginning of time :math:`t` is entitled to the dividend paid at time :math:`t`.

The owner of the share at the beginning of time :math:`t` is also entitled to sell the share to another investor during time :math:`t`.

Two types :math:`h=a, b` of investors differ only in their beliefs about a Markov transition matrix :math:`P` with typical element

.. math::

    P(i,j) = \mathbb P\{s_{t+1} = j \mid s_t = i\}


Investors of type :math:`a` believe the transition matrix

.. math::

    P_a =
        \begin{bmatrix}
            \frac{1}{2} & \frac{1}{2} \\
            \frac{2}{3} & \frac{1}{3}
        \end{bmatrix}


Investors of  type :math:`b` think the transition matrix is

.. math::

    P_b =
        \begin{bmatrix}
            \frac{2}{3} & \frac{1}{3} \\
            \frac{1}{4} & \frac{3}{4}
        \end{bmatrix}


The stationary (i.e., invariant) distributions of these two matrices can be calculated as follows:



.. code-block:: python3

    qa = np.array([[1/2, 1/2], [2/3, 1/3]])
    qb = np.array([[2/3, 1/3], [1/4, 3/4]])
    mca = qe.MarkovChain(qa)
    mcb = qe.MarkovChain(qb)
    mca.stationary_distributions

.. code-block:: python3

    mcb.stationary_distributions



The stationary distribution of :math:`P_a` is approximately :math:`\pi_A = \begin{bmatrix} .57 & .43 \end{bmatrix}`.

The stationary distribution of :math:`P_b` is approximately :math:`\pi_B = \begin{bmatrix} .43 & .57 \end{bmatrix}`.


Ownership Rights
----------------

An owner of the asset at the end of time :math:`t` is entitled to the dividend at time :math:`t+1` and also has the right to sell the asset at time :math:`t+1`.

Both types of investors are risk-neutral and both have the same fixed discount factor :math:`\beta \in (0,1)`.

In our numerical example, we’ll set :math:`\beta = .75`, just as Harrison and Kreps did.

We’ll eventually study the consequences of two different assumptions about the number of shares :math:`A` relative to the resources that our two types of investors can invest in the stock.

#. Both types of investors have enough resources (either wealth or the capacity to borrow) so that they can purchase the entire available stock of the asset [#f1]_.

#. No single type of investor has sufficient resources to purchase the entire stock.

Case 1 is the case studied in Harrison and Kreps.

In case 2, both types of investors always hold at least some of the asset.


Short Sales Prohibited
-----------------------

No short sales are allowed.

This matters because it limits pessimists from expressing their opinions.

* They can express their views by selling their shares.

* They cannot express their pessimism more loudly by artificially "manufacturing shares" -- that is, they cannot borrow shares from more optimistic investors and sell them immediately.


Optimism and Pessimism
----------------------

The above specifications of the perceived transition matrices :math:`P_a` and :math:`P_b`, taken directly from Harrison and Kreps, build in stochastically alternating temporary optimism and pessimism.

Remember that state :math:`1` is the high dividend state.

* In state :math:`0`, a type :math:`a` agent is more optimistic about next period's dividend than a type :math:`b` agent.

* In state :math:`1`, a type :math:`b` agent is more optimistic about next period's dividend.

However, the stationary distributions :math:`\pi_A = \begin{bmatrix} .57 & .43 \end{bmatrix}` and :math:`\pi_B = \begin{bmatrix} .43 & .57 \end{bmatrix}` tell us that a type :math:`B` person is more optimistic about the dividend process in the long run than is a type A person.

Transition matrices for the temporarily optimistic and pessimistic investors are constructed as follows.

Temporarily optimistic investors (i.e., the investor with the most optimistic
beliefs in each state) believe the transition matrix

.. math::

    P_o =
        \begin{bmatrix}
            \frac{1}{2} & \frac{1}{2} \\
            \frac{1}{4} & \frac{3}{4}
        \end{bmatrix}


Temporarily pessimistic believe the transition matrix

.. math::

    P_p =
        \begin{bmatrix}
            \frac{1}{2} & \frac{1}{2} \\
            \frac{1}{4} & \frac{3}{4}
        \end{bmatrix}

We'll return to these matrices and their significance in the exercise.

Information
--------------

Investors know a price function mapping the state :math:`s_t` at :math:`t` into the equilibrium price :math:`p(s_t)` that prevails in that state.

This price function is endogenous and to be determined below.

When investors choose whether to purchase or sell the asset at :math:`t`, they also know :math:`s_t`.



Solving the Model
====================

Now let's turn to solving the model.

This amounts to determining equilibrium prices under the different possible specifications of beliefs and constraints listed above.

In particular, we compare equilibrium price functions under the following alternative
assumptions about beliefs:

#. There is only one type of agent, either :math:`a` or :math:`b`.

#. There are two types of agents differentiated only by their beliefs. Each type of agent has sufficient resources to purchase all of the asset (Harrison and Kreps's setting).

#. There are two types of agents with different beliefs, but because of limited wealth and/or limited leverage, both types of investors hold the asset each period.



Summary Table
---------------

The following table gives a summary of the findings obtained in the remainder of the lecture
(you will be asked to recreate the table in an exercise).

It records implications of Harrison and Kreps's specifications of :math:`P_a, P_b, \beta`.

.. raw:: html

   <div class="content-table"></div>

+-----------------------+------+------+
|      :math:`s_t`      |  0   |  1   |
+=======================+======+======+
| :math:`p_a`           | 1.33 | 1.22 |
+-----------------------+------+------+
| :math:`p_b`           | 1.45 | 1.91 |
+-----------------------+------+------+
| :math:`p_o`           | 1.85 | 2.08 |
+-----------------------+------+------+
| :math:`p_p`           |    1 |    1 |
+-----------------------+------+------+
| :math:`\hat{p}_a`     | 1.85 | 1.69 |
+-----------------------+------+------+
| :math:`\hat{p}_b`     | 1.69 | 2.08 |
+-----------------------+------+------+

Here

* :math:`p_a` is the equilibrium price function  under homogeneous beliefs :math:`P_a`

* :math:`p_b` is the equilibrium price function under homogeneous beliefs :math:`P_b`

* :math:`p_o` is the equilibrium price function under heterogeneous beliefs with optimistic marginal investors

* :math:`p_p` is the equilibrium price function under heterogeneous beliefs with pessimistic marginal investors

* :math:`\hat{p}_a` is the amount type :math:`a` investors are willing to pay for the asset

* :math:`\hat{p}_b` is the amount type :math:`b` investors are willing to pay for the asset

We'll explain these values and how they are calculated one row at a time.


Single Belief Prices
--------------------

We’ll start by pricing the asset under homogeneous beliefs.

(This is the case treated in :doc:`the lecture <markov_asset>` on asset pricing with finite Markov states)

Suppose that there is only one type of investor, either of type :math:`a` or :math:`b`, and that this investor always "prices the asset".

Let :math:`p_h = \begin{bmatrix} p_h(0) \cr p_h(1) \end{bmatrix}` be the equilibrium price vector when all investors are of type :math:`h`.

The price today equals the expected discounted value of tomorrow's dividend and tomorrow's price of the asset:

.. math::

    p_h(s) = \beta \left( P_h(s,0) (0 + p_h(0)) + P_h(s,1) ( 1 + p_h(1)) \right), \quad s = 0, 1


These equations imply that the equilibrium price vector is

.. math::
    :label: HarrKrep1

    \begin{bmatrix} p_h(0) \cr p_h(1) \end{bmatrix}
    = \beta [I - \beta P_h]^{-1} P_h \begin{bmatrix} 0 \cr 1 \end{bmatrix}


The first two rows of the table report :math:`p_a(s)` and :math:`p_b(s)`.

Here's a function that can be used to compute these values

.. code-block:: python3

    """

    Authors: Chase Coleman, Tom Sargent

    """

    def price_single_beliefs(transition, dividend_payoff, β=.75):
        """
        Function to Solve Single Beliefs
        """
        # First compute inverse piece
        imbq_inv = la.inv(np.eye(transition.shape[0]) - β * transition)

        # Next compute prices
        prices = β * imbq_inv @ transition @ dividend_payoff

        return prices


Single Belief Prices as Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These equilibrium prices under homogeneous beliefs are important benchmarks for the subsequent analysis.

* :math:`p_h(s)` tells what investor :math:`h` thinks is the "fundamental value" of the asset.

* Here "fundamental value" means the expected discounted present value of future dividends.

We will compare these fundamental values of the asset with equilibrium values when traders have different beliefs.


Pricing under Heterogeneous Beliefs
-----------------------------------

There are several cases to consider.

The first is when both types of agents have sufficient wealth to purchase all of the asset themselves.

In this case, the marginal investor who prices the asset is the more optimistic type so that the equilibrium price :math:`\bar p` satisfies Harrison and Kreps's key equation:

.. math::
    :label: hakr2

    \bar p(s) =
    \beta
    \max
    \left\{
            P_a(s,0) \bar p(0) + P_a(s,1) ( 1 +  \bar p(1))
            ,\;
            P_b(s,0) \bar p(0) + P_b(s,1) ( 1 +  \bar p(1))
    \right\}


for :math:`s=0,1`.

The marginal investor who prices the asset in state :math:`s` is of type :math:`a` if

.. math::

    P_a(s,0)  \bar p(0) + P_a(s,1) ( 1 +  \bar p(1)) >
    P_b(s,0)  \bar p(0) + P_b(s,1) ( 1 +  \bar p(1))


The marginal investor is of type  :math:`b` if

.. math::

    P_a(s,1)  \bar p(0) + P_a(s,1) ( 1 +  \bar  p(1)) <
    P_b(s,1)  \bar p(0) + P_b(s,1) ( 1 +  \bar  p(1))


**Thus the marginal investor is the (temporarily) optimistic type**.

Equation :eq:`hakr2` is a functional equation that, like a Bellman equation, can be solved by

* starting with a guess for the price vector :math:`\bar p` and

* iterating to convergence on the operator that maps a guess :math:`\bar p^j` into an updated guess
  :math:`\bar p^{j+1}` defined by the right side of :eq:`hakr2`, namely

.. math::
    :label: HarrKrep3

    \bar  p^{j+1}(s)
     = \beta \max
     \left\{
            P_a(s,0) \bar p^j(0) + P_a(s,1) ( 1 + \bar p^j(1))
            ,\;
            P_b(s,0) \bar p^j(0) + P_b(s,1) ( 1 + \bar p^j(1))
    \right\}


for :math:`s=0,1`.

The third row of the table reports equilibrium prices that solve the functional equation when :math:`\beta = .75`.

Here the type that is optimistic about :math:`s_{t+1}` prices the asset in state :math:`s_t`.

It is instructive to compare these prices with the equilibrium prices for the homogeneous belief economies that solve under beliefs :math:`P_a` and :math:`P_b`.

Equilibrium prices :math:`\bar p` in the heterogeneous beliefs economy exceed what any prospective investor regards as the fundamental value of the asset in each possible state.

Nevertheless, the economy recurrently visits a state that makes each investor want to
purchase the asset for more than he believes its future dividends are
worth.

The reason is that he expects to have the option to sell the asset later to another investor who will value the asset more highly than he will.

* Investors of type :math:`a` are willing to pay the following price for the asset

.. math::

    \hat p_a(s) =
    \begin{cases}
    \bar p(0)  & \text{ if } s_t = 0 \\
    \beta(P_a(1,0) \bar p(0) + P_a(1,1) ( 1 +  \bar p(1))) & \text{ if } s_t = 1
    \end{cases}


* Investors of type :math:`b` are willing to pay the following price for the asset

.. math::

    \hat p_b(s) =
    \begin{cases}
        \beta(P_b(0,0) \bar p(0) + P_b (0,1) ( 1 +  \bar p(1)))  & \text{ if } s_t = 0 \\
        \bar p(1)  & \text{ if } s_t =1
    \end{cases}


Evidently, :math:`\hat p_a(1) < \bar p(1)` and :math:`\hat p_b(0) < \bar p(0)`.

Investors of type :math:`a` want to sell the asset in state :math:`1` while investors of type :math:`b` want to sell it in state :math:`0`.

*  The asset changes hands whenever the state changes from :math:`0` to :math:`1` or from :math:`1` to :math:`0`.

*  The valuations :math:`\hat p_a(s)` and :math:`\hat p_b(s)` are displayed in the fourth and fifth rows of the table.

*  Even the pessimistic investors who don't buy the asset think that it is worth more than they think future dividends are worth.

Here's code to solve for :math:`\bar p`, :math:`\hat p_a` and :math:`\hat p_b` using the iterative method described above

.. code-block:: python3

    def price_optimistic_beliefs(transitions, dividend_payoff, β=.75,
                                max_iter=50000, tol=1e-16):
        """
        Function to Solve Optimistic Beliefs
        """
        # We will guess an initial price vector of [0, 0]
        p_new = np.array([[0], [0]])
        p_old = np.array([[10.], [10.]])

        # We know this is a contraction mapping, so we can iterate to conv
        for i in range(max_iter):
            p_old = p_new
            p_new = β * np.max([q @ p_old
                                + q @ dividend_payoff for q in transitions],
                                1)

            # If we succeed in converging, break out of for loop
            if np.max(np.sqrt((p_new - p_old)**2)) < 1e-12:
                break

        ptwiddle = β * np.min([q @ p_old
                              + q @ dividend_payoff for q in transitions],
                              1)

        phat_a = np.array([p_new[0], ptwiddle[1]])
        phat_b = np.array([ptwiddle[0], p_new[1]])

        return p_new, phat_a, phat_b


Insufficient Funds
------------------

Outcomes differ when the more optimistic type of investor has insufficient wealth --- or insufficient ability to borrow enough --- to hold the entire stock of the asset.

In this case, the asset price must adjust to attract pessimistic investors.

Instead of equation :eq:`hakr2`, the equilibrium price satisfies

.. math::
    :label: HarrKrep4

    \check p(s)
    = \beta \min
    \left\{
        P_a(s,1)  \check  p(0) + P_a(s,1) ( 1 +   \check  p(1)) ,\;
        P_b(s,1)  \check p(0) + P_b(s,1) ( 1 + \check p(1))
    \right\}


and the marginal investor who prices the asset is always the one that values it *less* highly than does the other type.

Now the marginal investor is always the (temporarily) pessimistic type.

Notice from the sixth row of that the pessimistic price :math:`\underline p` is lower than the homogeneous belief prices :math:`p_a` and :math:`p_b` in both states.

When pessimistic investors price the asset according to :eq:`HarrKrep4`, optimistic investors think that the asset is underpriced.

If they could, optimistic investors would willingly borrow at the one-period gross interest rate :math:`\beta^{-1}` to purchase more of the asset.

Implicit constraints on leverage prohibit them from doing so.

When optimistic investors price the asset as in equation :eq:`hakr2`, pessimistic investors think that the asset is overpriced and would like to sell the asset short.

Constraints on short sales prevent that.

Here's code to solve for :math:`\check p` using iteration

.. code-block:: python3

    def price_pessimistic_beliefs(transitions, dividend_payoff, β=.75,
                                max_iter=50000, tol=1e-16):
        """
        Function to Solve Pessimistic Beliefs
        """
        # We will guess an initial price vector of [0, 0]
        p_new = np.array([[0], [0]])
        p_old = np.array([[10.], [10.]])

        # We know this is a contraction mapping, so we can iterate to conv
        for i in range(max_iter):
            p_old = p_new
            p_new = β * np.min([q @ p_old
                                + q @ dividend_payoff for q in transitions],
                               1)

            # If we succeed in converging, break out of for loop
            if np.max(np.sqrt((p_new - p_old)**2)) < 1e-12:
                break

        return p_new



Further Interpretation
-------------------------

:cite:`Scheinkman2014` interprets the Harrison-Kreps model as a model of a bubble --- a situation in which an asset price exceeds what every investor thinks is merited by the asset's underlying dividend stream.

Scheinkman stresses these features of the Harrison-Kreps model:

* Compared to the homogeneous beliefs setting leading to the pricing formula, high volume occurs when the Harrison-Kreps pricing formula prevails.

Type :math:`a` investors sell the entire stock of the asset to type :math:`b` investors every time the state switches from :math:`s_t =0` to :math:`s_t =1`.

Type :math:`b` investors sell the asset to type :math:`a` investors every time the state switches from :math:`s_t = 1` to :math:`s_t =0`.

Scheinkman takes this as a strength of the model because he observes high volume during *famous bubbles*.

* If the *supply* of the asset is increased sufficiently either physically (more "houses" are built) or artificially (ways are invented to short sell "houses"), bubbles end when the supply has grown enough to outstrip optimistic investors’ resources for purchasing the asset.

* If optimistic investors finance purchases by borrowing, tightening leverage constraints can extinguish a bubble.

Scheinkman extracts insights about the effects of financial regulations on bubbles.

He emphasizes how limiting short sales and limiting leverage have opposite effects.

Exercises
==========

Exercise 1
-----------

Recreate the summary table using the functions we have built above.

+-----------------------+------+------+
|      :math:`s_t`      |  0   |  1   |
+=======================+======+======+
| :math:`p_a`           | 1.33 | 1.22 |
+-----------------------+------+------+
| :math:`p_b`           | 1.45 | 1.91 |
+-----------------------+------+------+
| :math:`p_o`           | 1.85 | 2.08 |
+-----------------------+------+------+
| :math:`p_p`           |    1 |    1 |
+-----------------------+------+------+
| :math:`\hat{p}_a`     | 1.85 | 1.69 |
+-----------------------+------+------+
| :math:`\hat{p}_b`     | 1.69 | 2.08 |
+-----------------------+------+------+


You will first need to define the transition matrices and dividend payoff vector.


Solutions
==========

Exercise 1
-----------

First, we will obtain equilibrium price vectors with homogeneous beliefs, including when all
investors are optimistic or pessimistic.



.. code-block:: python3

    qa = np.array([[1/2, 1/2], [2/3, 1/3]])    # Type a transition matrix
    qb = np.array([[2/3, 1/3], [1/4, 3/4]])    # Type b transition matrix
    # Optimistic investor transition matrix
    qopt = np.array([[1/2, 1/2], [1/4, 3/4]])
    # Pessimistic investor transition matrix
    qpess = np.array([[2/3, 1/3], [2/3, 1/3]])

    dividendreturn = np.array([[0], [1]])

    transitions = [qa, qb, qopt, qpess]
    labels = ['p_a', 'p_b', 'p_optimistic', 'p_pessimistic']

    for transition, label in zip(transitions, labels):
        print(label)
        print("=" * 20)
        s0, s1 = np.round(price_single_beliefs(transition, dividendreturn), 2)
        print(f"State 0: {s0}")
        print(f"State 1: {s1}")
        print("-" * 20)



We will use the `price_optimistic_beliefs` function to find the price under
heterogeneous beliefs.



.. code-block:: python3

    opt_beliefs = price_optimistic_beliefs([qa, qb], dividendreturn)
    labels = ['p_optimistic', 'p_hat_a', 'p_hat_b']

    for p, label in zip(opt_beliefs, labels):
        print(label)
        print("=" * 20)
        s0, s1 = np.round(p, 2)
        print(f"State 0: {s0}")
        print(f"State 1: {s1}")
        print("-" * 20)




Notice that the equilibrium price with heterogeneous beliefs is equal to the price under single beliefs
with optimistic investors - this is due to the marginal investor being the temporarily optimistic type.

.. rubric:: Footnotes

.. [#f1] By assuming that both types of agents always have "deep enough pockets" to purchase all of the asset, the model takes wealth dynamics off the table. The Harrison-Kreps model generates high trading volume when the state changes either from 0 to 1 or from 1 to 0.
