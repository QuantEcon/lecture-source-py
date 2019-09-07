.. _coase:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

***********************************
:index:`Coase's Theory of the Firm`
***********************************

.. contents:: :depth: 2

Overview
========

In 1937, Ronald Coase wrote a brilliant essay on the nature of the firm :cite:`coase1937nature`.

Coase was writing at a time when the Soviet Union was rising to become a significant industrial power.

At the same time, many free-market economies were afflicted by a severe and painful depression.

This contrast led to an intensive debate on the relative
merits of decentralized, price-based allocation versus top-down planning.

In the midst of this debate, Coase made an important observation:
even in free-market economies, a great deal of top-down planning does in fact take place.

This is because *firms* form an integral part of free-market economies and, within firms, allocation is by planning.

In other words, free-market economies blend both planning (within firms) and decentralized production coordinated by prices.

The question Coase asked is this: if prices and free markets are so efficient, then why do firms even exist?

Couldn't the associated within-firm planning be done more efficiently by the market?

We'll use the following imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from scipy.optimize import fminbound
    from interpolation import interp


Why Firms Exist
---------------

On top of asking a deep and fascinating question, Coase also supplied an illuminating answer: firms exist because of transaction costs.

Here's one example of a transaction cost:

Suppose agent A is considering setting up a small business and needs a web developer to construct and help run an online store.

She can use the labor of agent B, a web developer, by writing up a freelance contract for these tasks and agreeing on a suitable price.

But contracts like this can be time-consuming and difficult to verify

* How will agent A be able to specify exactly what she wants, to the finest detail, when she herself isn't sure how the business will evolve?

* And what if she isn't familiar with web technology?  How can she specify all the relevant details?

* And, if things go badly, will failure to comply with the contract be verifiable in court?

In this situation, perhaps it will be easier to *employ* agent B under a simple labor contract.

The cost of this contract is far smaller because such contracts are simpler and more standard.

The basic agreement in a labor contract is: B will do what A asks him to do for the term of the contract, in return for a given salary.

Making this agreement is much easier than trying to map every task out in advance in a contract that will hold up in a court of law.

So agent A decides to hire agent B and a firm of nontrivial size appears, due to transaction costs.


A Trade-Off
-----------

Actually, we haven't yet come to the heart of Coase's investigation.

The issue of why firms exist is a binary question: should firms have positive size or zero size?

A better and more general question is: **what determines the size of firms**?

The answer Coase came up with was that "a firm will tend to expand until the costs of organizing an extra
transaction within the firm become equal to the costs of carrying out the same
transaction by means of an exchange on the open market..." (:cite:`coase1937nature`, p. 395).

But what are these internal and external costs?

In short, Coase envisaged a trade-off between

* transaction costs, which add to the expense of operating *between* firms, and

* diminishing returns to management, which adds to the expense of operating *within* firms


We discussed an example of transaction costs above (contracts).

The other cost, diminishing returns to management, is a catch-all for the idea
that big operations are increasingly costly to manage.

For example, you could think of management as a pyramid, so hiring more workers to implement more tasks
requires expansion of the pyramid, and hence labor costs grow at a rate more than
proportional to the range of tasks.

Diminishing returns to management makes in-house production expensive, favoring small firms.


Summary
-------

Here's a summary of our discussion:

* Firms grow because transaction costs encourage them to take some operations in house.

* But as they get large, in-house operations become costly due to diminishing returns to management.

* The size of firms is determined by balancing these effects, thereby equalizing the marginal costs of each form of operation.



A Quantitative Interpretation
-----------------------------

Coases ideas were expressed verbally, without any mathematics.

In fact, his essay is a wonderful example of how far you can get with clear thinking and plain English.

However, plain English is not good for quantitative analysis, so let's bring some mathematical and computation tools to bear.

In doing so we'll add a bit more structure than Coase did, but this price will be worth paying.


Our exposition is based on :cite:`kikuchi2018span`.


The Model
=========


The model we study involves production of a single unit of a final good.

Production requires a linearly ordered chain, requiring sequential completion
of a large number of processing stages.

The stages are indexed by :math:`t \in [0,1]`, with :math:`t=0` indicating that no tasks
have been undertaken and :math:`t=1` indicating that the good is complete.


Subcontracting
--------------

The subcontracting scheme by which tasks are allocated across firms is illustrated in the figure below

.. figure:: /_static/lecture_specific/coase/subcontracting.png

In this example,

* Firm 1 receives a contract to sell one unit of the completed good to a final buyer.

* Firm 1 then forms a contract with firm 2 to purchase the partially completed good at stage :math:`t_1`, with
  the intention of implementing the remaining :math:`1 - t_1` tasks in-house (i.e., processing from stage :math:`t_1` to stage :math:`1`).

* Firm 2 repeats this procedure, forming a contract with firm 3 to purchase the good at stage :math:`t_2`.

* Firm 3 decides to complete the chain, selecting :math:`t_3 = 0`.

At this point, production unfolds in the opposite direction (i.e., from upstream to downstream).

* Firm 3 completes processing stages from :math:`t_3 = 0` up to :math:`t_2` and transfers the good to firm 2.

* Firm 2 then processes from :math:`t_2` up to :math:`t_1` and transfers the good to firm 1,

* Firm 1 processes from :math:`t_1` to :math:`1` and delivers the completed good to the final buyer.

The length of the interval of stages (range of tasks) carried out by firm :math:`i` is denoted by :math:`\ell_i`.

.. figure:: /_static/lecture_specific/coase/allocation.png

Each firm chooses only its *upstream* boundary, treating its downstream boundary as given.

The benefit of this formulation is that it implies a recursive structure for the decision problem for each firm.

In choosing how many processing stages to subcontract, each successive firm faces essentially the same
decision problem as the firm above it in the chain, with the only difference being that the decision space is a subinterval of the decision space for the firm above.

We will exploit this recursive structure in our study of equilibrium.


Costs
-----

Recall that we are considering a trade-off between two types of costs.

Let's discuss these costs and how we represent them mathematically.

**Diminishing returns to management** means rising costs per task when a firm expands the range of productive activities coordinated by its managers.

We represent these ideas by taking the cost of carrying out :math:`\ell` tasks in-house to be :math:`c(\ell)`, where :math:`c` is increasing and strictly convex.

Thus, the average cost per task rises with the range of tasks performed in-house.

We also assume that :math:`c` is continuously differentiable, with :math:`c(0)=0` and :math:`c'(0) > 0`.

**Transaction costs** are represented as a wedge between the buyer's and seller's prices.

It matters little for us whether the transaction cost is borne by the buyer or the seller.

Here we assume that the cost is borne only by the buyer.

In particular, when two firms agree to a trade at face value :math:`v`, the buyer's total outlay is :math:`\delta v`, where :math:`\delta > 1`.

The seller receives only :math:`v`, and the difference is paid to agents outside the model.



Equilibrium
===========

We assume that all firms are *ex-ante* identical and act as price takers.

As price takers, they face a price function :math:`p`, which is a map from :math:`[0, 1]` to :math:`\mathbb R_+`, with :math:`p(t)` interpreted
as the price of the good at processing stage :math:`t`.

There is a countable infinity of firms indexed by :math:`i` and no barriers to entry.

The cost of supplying the initial input (the good processed up to stage zero) is set to zero for simplicity.

Free entry and the infinite fringe of competitors rule out positive profits for incumbents,
since any incumbent could be replaced by a member of the competitive fringe
filling the same role in the production chain.

Profits are never negative in equilibrium because firms can freely exit.

Informal Definition of Equilibrium
----------------------------------

An equilibrium in this setting is an allocation of firms and a price function such that

#. all active firms in the chain make zero profits, including suppliers of raw materials

#. no firm in the production chain has an incentive to deviate, and

#. no inactive firms can enter and extract positive profits



Formal Definition of Equilibrium
--------------------------------

Let's make this definition more formal.

(You might like to skip this section on first reading)

An **allocation** of firms is a nonnegative sequence :math:`\{\ell_i\}_{i \in \mathbb N}` such that :math:`\ell_i=0` for all sufficiently large :math:`i`.

Recalling the figures above,

* :math:`\ell_i` represents the range of tasks implemented by the :math:`i`-th firm

As a labeling convention, we assume that firms enter in order, with firm 1 being the furthest downstream.

An allocation :math:`\{\ell_i\}` is called **feasible** if :math:`\sum_{\, i \geq 1} \ell_i = 1`.

In a feasible allocation, the entire production process is completed by finitely many firms.

Given a feasible allocation, :math:`\{\ell_i\}`, let :math:`\{t_i\}` represent the corresponding transaction stages, defined by

.. math::
    :label: ab

    t_0 = s \quad \text{and} \quad t_i = t_{i-1} - \ell_i

In particular, :math:`t_{i-1}` is the downstream boundary of firm :math:`i` and :math:`t_i` is its upstream boundary.

As transaction costs are incurred only by the buyer, its profits are

.. math::
    :label: prof

    \pi_i = p(t_{i-1}) - c(\ell_i) - \delta p(t_i)

Given a price function :math:`p` and a feasible allocation :math:`\{\ell_i\}`, let

* :math:`\{t_i\}` be the corresponding firm boundaries.

* :math:`\{\pi_i\}` be corresponding profits, as defined in :eq:`prof`.

This price-allocation pair is called an **equilibrium** for the production chain if

#. :math:`\; p(0) = 0`,

#. :math:`\; \pi_i = 0` for all :math:`i`, and

#. :math:`p(s) - c(s - t) -  \delta p(t) \leq 0` for any pair :math:`s, t` with :math:`0 \leq s \leq t \leq 1`.


The rationale behind these conditions was given in our informal definition of equilibrium above.


Existence, Uniqueness and Computation of Equilibria
===================================================

We have defined an equilibrium but does one exist?  Is it unique?  And, if so, how can we compute it?

A Fixed Point Method
--------------------

To address these questions, we introduce the operator :math:`T` mapping a nonnegative function :math:`p` on :math:`[0, 1]` to :math:`Tp` via

.. math::
    :label: coase_deft

    Tp(s) = \min_{t \leq s}  \,
            \{ c(s-t) + \delta p(t) \}
           \quad \text{for all} \quad
           s \in [0,1].

Here and below, the restriction :math:`0 \leq t` in the minimum is understood.

The operator :math:`T` is similar to a Bellman operator.

Under this analogy, :math:`p` corresponds to a value function and :math:`\delta` to a discount factor.

But :math:`\delta > 1`, so :math:`T` is not a contraction in any obvious metric, and in fact, :math:`T^n p` diverges for many choices of :math:`p`.

Nevertheless, there exists a domain on which :math:`T` is well-behaved:  the set
of convex increasing continuous functions :math:`p \colon [0,1] \to \mathbb R` such that :math:`c'(0) s \leq p(s) \leq c(s)` for all :math:`0 \leq s \leq 1`.

We denote this set of functions by :math:`\mathcal P`.

In :cite:`kikuchi2018span` it is shown that the following statements are true:

#. :math:`T` maps :math:`\mathcal P` into itself.
#. :math:`T` has a unique fixed point in :math:`\mathcal P`, denoted below by :math:`p^*`.
#. For all :math:`p \in \mathcal P` we have :math:`T^k p \to p^*` uniformly as :math:`k \to \infty`.


Now consider the choice function

.. math::
    :label: coase_argmins

    t^*(s) := \text{ the solution to } \min_{ t \leq s} \{c(s - t) + \delta p^*(t) \}

By definition, :math:`t^*(s)` is the cost-minimizing upstream boundary for a firm that
is contracted to deliver the good at stage :math:`s` and faces the price
function :math:`p^*`.

Since :math:`p^*` lies in :math:`\mathcal P` and since :math:`c` is strictly convex, it follows that the right-hand side of :eq:`coase_argmins` is continuous and strictly convex in :math:`t`.

Hence the minimizer :math:`t^*(s)` exists and is uniquely defined.

We can use :math:`t^*` to construct an equilibrium allocation as follows:

Recall that firm 1 sells the completed good at stage :math:`s=1`, its optimal upstream boundary is :math:`t^*(1)`.

Hence firm 2's optimal upstream boundary is :math:`t^*(t^*(1))`.

Continuing in this way produces the sequence :math:`\{t^*_i\}` defined by

.. math::
    :label: coase_ralloc

    t_0^* = 1 \quad \text{and} \quad t_i^* = t^*(t_{i-1})

The sequence ends when a firm chooses to complete all remaining tasks.

We label this firm (and hence the number of firms in the chain) as

.. math::
    :label: coase_eqfn

    n^* := \inf \{i \in \mathbb N \,:\, t^*_i = 0\}

The task allocation corresponding to :eq:`coase_ralloc` is given by :math:`\ell_i^* := t_{i-1}^* - t_i^*` for all :math:`i`.

In :cite:`kikuchi2018span` it is shown that

#. The value :math:`n^*` in :eq:`coase_eqfn` is well-defined and finite,

#. the allocation :math:`\{\ell^*_i\}` is feasible, and

#. the price function :math:`p^*` and this allocation together forms an equilibrium for the production chain.

While the proofs are too long to repeat here,
much of the insight can be obtained by observing that, as a fixed point of :math:`T`, the equilibrium price function must satisfy

.. math::
    :label: coase_epe

    p^*(s) =
    \min_{t \leq s} \, \{ c(s - t) + \delta p^*(t)  \}
    \quad \text{for all} \quad s \in [0, 1]

From this equation, it is clear that so profits are zero for all incumbent firms.


Marginal Conditions
-------------------

We can develop some additional insights on the behavior of firms by examining marginal conditions associated with the equilibrium.

As a first step, let :math:`\ell^*(s) := s - t^*(s)`.

This is the cost-minimizing range of in-house tasks for a firm with downstream boundary :math:`s`.

In :cite:`kikuchi2018span` it is shown that :math:`t^*` and :math:`\ell^*` are increasing and continuous, while :math:`p^*` is continuously differentiable at all :math:`s \in (0, 1)` with

.. math::
    :label: coase_env

    (p^*)'(s) = c'(\ell^*(s))

Equation :eq:`coase_env` follows from :math:`p^*(s) = \min_{t \leq s}  \, \{ c(s-t) + \delta p^*(t) \}` and the envelope theorem for derivatives.

A related equation is the first order condition for :math:`p^*(s) = \min_{t \leq s} \, \{ c(s-t) + \delta p^*(t) \}`, the minimization problem for a firm with upstream boundary :math:`s`, which is

.. math::
    :label: coase_foc

    \delta (p^*)'(t^*(s)) = c'(s - t^*(s))

This condition matches the marginal condition expressed verbally by
Coase that we stated above:

   "A firm will tend to expand until the costs of organizing an extra
   transaction within the firm become equal to the costs of carrying
   out the same transaction by means of an exchange on the
   open market..."

Combining :eq:`coase_env` and :eq:`coase_foc` and evaluating at :math:`s=t_i`, we see that active firms that are adjacent satisfy

.. math::
    :label: coase_euler

    \delta \, c'(\ell^*_{i+1}) = c'(\ell^*_i)

In other words, the marginal in-house cost per task at a given firm is equal to that of its upstream partner multiplied by gross transaction cost.

This expression can be thought of as a **Coase--Euler equation,** which determines
inter-firm efficiency by indicating how two costly forms of coordination (markets and management) are jointly minimized in equilibrium.



Implementation
==============

For most specifications of primitives, there is no closed-form solution for the equilibrium as far as we are aware.

However, we know that we can compute the equilibrium corresponding to a given transaction cost parameter :math:`\delta`  and a cost function :math:`c`
by applying the results stated above.

In particular, we can

#. fix initial condition :math:`p \in \mathcal P`,
#. iterate with :math:`T` until :math:`T^n p` has converged to :math:`p^*`, and
#. recover firm choices via the choice function :eq:`coase_deft`

As we step between iterates, we will use linear interpolation of functions, as
we did in our :doc:`lecture on optimal growth <optgrowth>` and several other
places.

To begin, here's a class to store primitives and a grid

.. code-block:: python3

   class ProductionChain:

       def __init__(self,
               n=1000,
               delta=1.05,
               c=lambda t: np.exp(10 * t) - 1):

           self.n, self.delta, self.c = n, delta, c
           self.grid = np.linspace(0, 1, n)

Now let's implement and iterate with :math:`T` until convergence.

Recalling that our initial condition must lie in :math:`\mathcal P`, we set
:math:`p_0 = c`

.. code-block:: python3

   def compute_prices(pc, tol=1e-5, max_iter=5000):
       """
       Compute prices by iterating with T

           * pc is an instance of ProductionChain
           * The initial condition is p = c

       """
       delta, c, n, grid = pc.delta, pc.c, pc.n, pc.grid
       p = c(grid)  # Initial condition is c(s), as an array
       new_p = np.empty_like(p)
       error = tol + 1
       i = 0

       while error > tol and i < max_iter:
           for i, s in enumerate(grid):
               Tp = lambda t: delta * interp(grid, p, t) + c(s - t)
               new_p[i] = Tp(fminbound(Tp, 0, s))
           error = np.max(np.abs(p - new_p))
           p = new_p
           i = i + 1

       if i < max_iter:
           print(f"Iteration converged in {i} steps")
       else:
           print(f"Warning: iteration hit upper bound {max_iter}")

       p_func = lambda x: interp(grid, p, x)
       return p_func

The next function computes optimal choice of upstream boundary and range of
task implemented for a firm face price function `p_function` and with downstream boundary :math:`s`.

.. code-block:: python3

   def optimal_choices(pc, p_function, s):
       """
       Takes p_func as the true function, minimizes on [0,s]

       Returns optimal upstream boundary t_star and optimal size of
       firm ell_star

       In fact, the algorithm minimizes on [-1,s] and then takes the
       max of the minimizer and zero. This results in better results
       close to zero

       """
       delta, c = pc.delta, pc.c
       f = lambda t: delta * p_function(t) + c(s - t)
       t_star = max(fminbound(f, -1, s), 0)
       ell_star = s - t_star
       return t_star, ell_star

The allocation of firms can be computed by recursively stepping through firms' choices of
their respective upstream boundary, treating the previous firm's upstream boundary as their own downstream boundary.

In doing so, we start with firm 1, who has downstream boundary :math:`s=1`.

.. code-block:: python3

   def compute_stages(pc, p_function):
       s = 1.0
       transaction_stages = [s]
       while s > 0:
           s, ell = optimal_choices(pc, p_function, s)
           transaction_stages.append(s)
       return np.array(transaction_stages)

Let's try this at the default parameters.

The next figure shows the equilibrium price function, as well as the
boundaries of firms as vertical lines

.. code-block:: python3

   pc = ProductionChain()
   p_star = compute_prices(pc)

   transaction_stages = compute_stages(pc, p_star)

   fig, ax = plt.subplots()

   ax.plot(pc.grid, p_star(pc.grid))
   ax.set_xlim(0.0, 1.0)
   ax.set_ylim(0.0)
   for s in transaction_stages:
       ax.axvline(x=s, c="0.5")
   plt.show()


Here's the function :math:`\ell^*`, which shows how large a firm with
downstream boundary :math:`s` chooses to be

.. code-block:: python3

   ell_star = np.empty(pc.n)
   for i, s in enumerate(pc.grid):
       t, e = optimal_choices(pc, p_star, s)
       ell_star[i] = e

   fig, ax = plt.subplots()
   ax.plot(pc.grid, ell_star, label="$\ell^*$")
   ax.legend(fontsize=14)
   plt.show()

Note that downstream firms choose to be larger, a point we return to below.


Exercises
=========

.. _lucas_asset_exer1:

Exercise 1
----------

The number of firms is endogenously determined by the primitives.

What do you think will happen in terms of the number of firms as :math:`\delta` increases?  Why?

Check your intuition by computing the number of firms at `delta in (1.01, 1.05, 1.1)`.


Exercise 2
----------

The **value added** of firm :math:`i` is :math:`v_i := p^*(t_{i-1}) - p^*(t_i)`.

One of the interesting predictions of the model is that value added is increasing with downstreamness, as are several other measures of firm size.

Can you give any intution?

Try to verify this phenomenon (value added increasing with downstreamness) using the code above.


Solutions
=========


Exercise 1
----------



.. code-block:: python3

   for delta in (1.01, 1.05, 1.1):

      pc = ProductionChain(delta=delta)
      p_star = compute_prices(pc)
      transaction_stages = compute_stages(pc, p_star)
      num_firms = len(transaction_stages)
      print(f"When delta={delta} there are {num_firms} firms")



Exercise 2
----------

Firm size increases with downstreamness because :math:`p^*`, the equilibrium price function, is increasing and strictly convex.

This means that, for a given producer, the marginal cost of the input purchased from the producer just upstream from itself in the chain increases as we go further downstream.

Hence downstream firms choose to do more in house than upstream firms --- and are therefore larger.

The equilibrium price function is strictly convex due to both transaction
costs and diminishing returns to management.

One way to put this is that firms are prevented from completely mitigating the costs
associated with diminishing returns to management --- which induce convexity --- by transaction costs.  This is because transaction costs force firms to have nontrivial size.

Here's one way to compute and graph value added across firms


.. code-block:: python3

   pc = ProductionChain()
   p_star = compute_prices(pc)
   stages = compute_stages(pc, p_star)

   va = []

   for i in range(len(stages) - 1):
       va.append(p_star(stages[i]) - p_star(stages[i+1]))

   fig, ax = plt.subplots()
   ax.plot(va, label="value added by firm")
   ax.set_xticks((5, 25))
   ax.set_xticklabels(("downstream firms", "upstream firms"))
   plt.show()
