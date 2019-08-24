.. _ree:

.. include:: /_static/includes/header.raw

******************************************
:index:`Rational Expectations Equilibrium`
******************************************

.. contents:: :depth: 2

.. epigraph::

    "If you're so smart, why aren't you rich?"

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========

This lecture introduces the concept of *rational expectations equilibrium*.

To illustrate it, we describe a linear quadratic version of a famous and important model
due to Lucas and Prescott :cite:`LucasPrescott1971`.

This 1971 paper is one of a small number of research articles that kicked off the *rational expectations revolution*.

We follow Lucas and Prescott by employing a setting that is readily "Bellmanized" (i.e., capable of being formulated in terms of dynamic programming problems).

Because we use linear quadratic setups for demand and costs, we can adapt the LQ programming techniques described in :doc:`this lecture <lqcontrol>`.

We will learn about how a representative agent's problem differs from a planner's, and how a planning problem can be used to compute rational expectations quantities.

We will also learn about how a rational expectations equilibrium can be characterized as a `fixed point <https://en.wikipedia.org/wiki/Fixed_point_%28mathematics%29>`_ of a mapping from a *perceived law of motion* to an *actual law of motion*.

Equality between a perceived and an actual law of motion for endogenous market-wide objects captures in a nutshell what the rational expectations equilibrium concept is all about.

Finally, we will learn about the important "Big :math:`K`, little :math:`k`" trick, a modeling device widely used in macroeconomics.

Except that for us

* Instead of "Big :math:`K`" it will be "Big :math:`Y`".

* Instead of "little :math:`k`" it will be "little :math:`y`".


Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

We'll also use the LQ class from QuantEcon.py.

.. code-block:: ipython

    from quantecon import LQ


The Big Y, Little y Trick
-------------------------

This widely used method applies in contexts in which a "representative firm" or agent is a "price taker" operating within a competitive equilibrium.

We want to impose that

* The representative firm or individual takes *aggregate* :math:`Y` as given when it chooses individual :math:`y`, but :math:`\ldots`.

* At the end of the day, :math:`Y = y`, so that the representative firm is indeed representative.

The Big :math:`Y`, little :math:`y` trick accomplishes these two goals by

* Taking :math:`Y` as beyond control when posing the choice  problem of who chooses :math:`y`;  but :math:`\ldots`.

* Imposing :math:`Y = y` *after* having solved the individual's optimization  problem.

Please watch for how this strategy is applied as the lecture unfolds.

We begin by applying the  Big :math:`Y`, little :math:`y` trick in a very simple static context.


A Simple Static Example of the Big Y, Little y Trick
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Consider a static model in which a collection of :math:`n` firms produce a homogeneous good that is sold in a competitive market.

Each of these :math:`n` firms sell output :math:`y`.

The price :math:`p` of the good lies on an inverse demand curve

.. math::
    :label: ree_comp3d_static

    p = a_0 - a_1 Y


where

* :math:`a_i > 0` for :math:`i = 0, 1`

* :math:`Y = n y` is the market-wide level of output

Each firm has a total cost function

.. math::

    c(y) = c_1 y + 0.5 c_2 y^2,
    \qquad c_i > 0 \text{ for } i = 1,2


The profits of a representative firm are :math:`p y - c(y)`.

Using :eq:`ree_comp3d_static`, we can express the problem of the representative firm as

.. math::
    :label: max_problem_static

    \max_{y} \Bigl[ (a_0 - a_1 Y) y - c_1 y - 0.5 c_2 y^2 \Bigr]


In posing problem :eq:`max_problem_static`, we want the firm to be a *price taker*.

We do that by regarding :math:`p` and therefore :math:`Y` as exogenous to the firm.

The essence of the Big :math:`Y`, little :math:`y` trick is *not* to set :math:`Y = n y` *before* taking the first-order condition with respect
to :math:`y` in problem :eq:`max_problem_static`.

This assures that the firm is a price taker.

The first-order condition for problem :eq:`max_problem_static` is

.. math::
    :label: BigYsimpleFONC

    a_0 - a_1 Y - c_1 - c_2 y = 0


At this point, *but not before*, we substitute :math:`Y = ny` into :eq:`BigYsimpleFONC`
to obtain the following linear equation

.. math::
    :label: staticY

    a_0 - c_1 - (a_1 + n^{-1} c_2) Y = 0


to be solved for the competitive equilibrium market-wide output :math:`Y`.

After solving for :math:`Y`, we can compute the competitive equilibrium price :math:`p` from the inverse demand curve :eq:`ree_comp3d_static`.


Further Reading
---------------

References for this lecture include

* :cite:`LucasPrescott1971`

* :cite:`Sargent1987`, chapter XIV

* :cite:`Ljungqvist2012`, chapter 7


Defining Rational Expectations Equilibrium
==========================================

.. index::
    single: Rational Expectations Equilibrium; Definition

Our first illustration of a rational expectations equilibrium involves a market with :math:`n` firms, each of which seeks to maximize the discounted present value of profits in the face of adjustment costs.

The adjustment costs induce the firms to make gradual adjustments, which in turn requires consideration of future prices.

Individual firms understand that, via the inverse demand curve, the price is determined by the amounts supplied by other firms.

Hence each firm wants to  forecast future total industry supplies.

In our context, a forecast is generated by a belief about the law of motion for the aggregate state.

Rational expectations equilibrium prevails when this belief coincides with the actual
law of motion generated by production choices induced by this belief.

We formulate a rational expectations equilibrium in terms of a fixed point of an operator that maps beliefs into optimal beliefs.

.. _ree_ce:

Competitive Equilibrium with Adjustment Costs
---------------------------------------------

.. index::
    single: Rational Expectations Equilibrium; Competitive Equilbrium (w. Adjustment Costs)

To illustrate, consider a collection of :math:`n` firms producing a homogeneous good that is sold in a competitive market.

Each of these :math:`n` firms sell output :math:`y_t`.

The price :math:`p_t` of the good lies on the inverse demand curve

.. math::
    :label: ree_comp3d

    p_t = a_0 - a_1 Y_t


where

* :math:`a_i > 0` for :math:`i = 0, 1`

* :math:`Y_t = n y_t` is the market-wide level of output

.. _ree_fp:

The Firm's Problem
^^^^^^^^^^^^^^^^^^

Each firm is a price taker.

While it faces no uncertainty, it does face adjustment costs

..  We can get closer to Lucas and Prescott's formulation by adding
    uncertainty to the problem --- for example, by assuming that the inverse demand function
    is :math:`p_t = a_0 - a_1 Y_t + u_t`, where :math:`u_{t+1} = \rho u_t +
    \sigma_u \epsilon_{t+1}` where :math:`|\rho| < 1` and :math:`\epsilon_{t+1} \sim N(0,1)`
    is IID.    We ask you to study the consequences of this change in exercise XXXXX.

In particular, it chooses a production plan to maximize

.. math::
    :label: ree_obj

    \sum_{t=0}^\infty \beta^t r_t


where

.. math::
    :label: ree_comp2

    r_t := p_t y_t - \frac{ \gamma (y_{t+1} - y_t )^2 }{2},
    \qquad  y_0 \text{ given}


Regarding the parameters,

* :math:`\beta \in (0,1)` is a discount factor

* :math:`\gamma > 0` measures the cost of adjusting the rate of output

Regarding timing, the firm observes :math:`p_t` and :math:`y_t` when it chooses :math:`y_{t+1}` at time :math:`t`.

To state the firm's optimization problem completely requires that we specify dynamics for all state variables.

This includes ones that the firm cares about but does not control like :math:`p_t`.

We turn to this problem now.


Prices and Aggregate Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^


In view of :eq:`ree_comp3d`, the firm's incentive to forecast the market price translates into an incentive to forecast aggregate output :math:`Y_t`.

Aggregate output depends on the choices of other firms.

We assume that :math:`n` is such a large number  that the output of any single firm has a negligible effect on aggregate output.

That justifies firms in regarding their forecasts of aggregate output as being unaffected by their own output decisions.



The Firm's Beliefs
^^^^^^^^^^^^^^^^^^

We suppose the firm believes that market-wide output :math:`Y_t` follows the law of motion

.. math::
    :label: ree_hlom

    Y_{t+1} =  H(Y_t)


where :math:`Y_0` is a known initial condition.

The *belief function* :math:`H` is an equilibrium object, and hence remains to be determined.



Optimal Behavior Given Beliefs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For now, let's fix a particular belief :math:`H` in :eq:`ree_hlom` and investigate the firm's response to it.

Let :math:`v` be the optimal value function for the firm's problem given :math:`H`.

The value function satisfies the Bellman equation

.. math::
    :label: comp4

    v(y,Y) = \max_{y'} \left\{ a_0 y - a_1 y Y - \frac{ \gamma (y' - y)^2}{2}   + \beta v(y', H(Y))\right\}


Let's denote the firm's optimal policy function by :math:`h`, so that

.. math::
    :label: comp9

    y_{t+1} = h(y_t, Y_t)


where

.. math::
    :label: ree_opbe

    h(y, Y) := \argmax_{y'}
    \left\{ a_0 y - a_1 y Y - \frac{ \gamma (y' - y)^2}{2}   + \beta v(y', H(Y))\right\}


Evidently :math:`v` and :math:`h` both depend on :math:`H`.

A First-Order Characterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In what follows it will be helpful to have a second characterization of :math:`h`, based on first-order conditions.

The first-order necessary condition for choosing :math:`y'` is


.. math::
    :label: comp5

    -\gamma (y' - y) + \beta v_y(y', H(Y) ) = 0


An important useful envelope result of Benveniste-Scheinkman  :cite:`BenvenisteScheinkman1979` implies that to
differentiate :math:`v` with respect to :math:`y` we can naively differentiate
the right side of :eq:`comp4`, giving

.. math::

    v_y(y,Y) = a_0 - a_1 Y + \gamma (y' - y)


Substituting this equation into :eq:`comp5` gives the *Euler equation*

.. math::
    :label: ree_comp7

    -\gamma (y_{t+1} - y_t) + \beta [a_0 - a_1 Y_{t+1} + \gamma (y_{t+2} - y_{t+1} )] =0


The firm optimally sets  an output path that satisfies :eq:`ree_comp7`, taking :eq:`ree_hlom` as given, and  subject to

* the initial conditions for :math:`(y_0, Y_0)`.

* the terminal condition :math:`\lim_{t \rightarrow \infty } \beta^t y_t v_y(y_{t}, Y_t) = 0`.

This last condition is called the *transversality condition*, and acts as a first-order necessary condition "at infinity".

The firm's decision rule solves the difference equation :eq:`ree_comp7` subject to the given initial condition :math:`y_0` and the transversality condition.

Note that solving the Bellman equation :eq:`comp4` for :math:`v` and then :math:`h` in :eq:`ree_opbe` yields
a decision rule that automatically imposes both the Euler equation :eq:`ree_comp7` and the transversality condition.



The Actual Law of Motion for Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we've seen, a given belief translates into a particular decision rule :math:`h`.

Recalling that :math:`Y_t = ny_t`, the *actual law of motion* for market-wide output is then

.. math::
    :label: ree_comp9a

    Y_{t+1} = n h(Y_t/n, Y_t)


Thus, when firms believe that the law of motion for market-wide output is :eq:`ree_hlom`, their optimizing behavior makes the actual law of motion be :eq:`ree_comp9a`.

.. _ree_def:

Definition of Rational Expectations Equilibrium
-----------------------------------------------

A *rational expectations equilibrium* or *recursive competitive equilibrium*  of the model with adjustment costs is a decision rule :math:`h` and an aggregate law of motion :math:`H` such that


#.  Given belief :math:`H`, the map :math:`h` is the firm's optimal policy function.

#.  The law of motion :math:`H` satisfies :math:`H(Y)= nh(Y/n,Y)` for all
    :math:`Y`.


Thus, a rational expectations equilibrium equates the perceived and actual laws of motion :eq:`ree_hlom` and :eq:`ree_comp9a`.


Fixed Point Characterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we've seen, the firm's optimum problem induces a mapping :math:`\Phi` from a perceived law of motion :math:`H` for market-wide output to an actual law of motion :math:`\Phi(H)`.

The mapping :math:`\Phi` is the composition of two operations, taking a perceived law of motion into a decision rule via :eq:`comp4`--:eq:`ree_opbe`, and a decision rule into an actual law via :eq:`ree_comp9a`.

The :math:`H` component of a rational expectations equilibrium is a fixed point of :math:`\Phi`.



Computation of an Equilibrium
=============================

.. index::
    single: Rational Expectations Equilibrium; Computation

Now let's consider the problem of computing the rational expectations equilibrium.


Failure of Contractivity
------------------------

Readers accustomed to dynamic programming arguments might try to address this problem by choosing some guess :math:`H_0` for the aggregate law of motion and then iterating with :math:`\Phi`.

Unfortunately, the mapping :math:`\Phi` is not a contraction.

In particular, there is no guarantee that direct iterations on :math:`\Phi` converge [#fn_im]_.

Fortunately, there is another method that works here.

The method exploits a general connection between equilibrium and Pareto optimality expressed in
the fundamental theorems of welfare economics (see, e.g, :cite:`MCWG1995`).

Lucas and Prescott :cite:`LucasPrescott1971` used this method to construct a rational expectations equilibrium.

The details follow.


.. _ree_pp:

A Planning Problem Approach
---------------------------

.. index::
    single: Rational Expectations Equilibrium; Planning Problem Approach

Our plan of attack is to match the Euler equations of the market problem with those for a  single-agent choice problem.

As we'll see, this planning problem can be solved by LQ control (:doc:`linear regulator <lqcontrol>`).

The optimal quantities from the planning problem are rational expectations equilibrium quantities.

The rational expectations equilibrium price can be obtained as a shadow price in the planning problem.

For convenience, in this section, we set :math:`n=1`.

We first compute a sum of  consumer and producer surplus at time :math:`t`

.. math::
    :label: comp10

    s(Y_t, Y_{t+1})
    := \int_0^{Y_t} (a_0 - a_1 x) \, dx - \frac{ \gamma (Y_{t+1} - Y_t)^2}{2}


The first term is the area under the demand curve, while the second measures the social costs of changing output.

The *planning problem* is to choose a production plan :math:`\{Y_t\}` to maximize

.. math::

    \sum_{t=0}^\infty \beta^t s(Y_t, Y_{t+1})


subject to an initial condition for :math:`Y_0`.


Solution of the Planning Problem
--------------------------------

Evaluating the integral in :eq:`comp10` yields the quadratic form :math:`a_0
Y_t - a_1 Y_t^2 / 2`.

As a result, the Bellman equation for the planning problem is

.. math::
    :label: comp12

    V(Y) = \max_{Y'}
    \left\{a_0  Y - {a_1 \over 2} Y^2 - \frac{ \gamma (Y' - Y)^2}{2} + \beta V(Y') \right\}


The associated first-order condition is

.. math::
    :label: comp14

    -\gamma (Y' - Y) + \beta V'(Y') = 0


Applying the same Benveniste-Scheinkman formula gives

.. math::

    V'(Y) = a_0 - a_1 Y + \gamma (Y' - Y)


Substituting this into equation :eq:`comp14` and rearranging leads to the Euler
equation

.. math::
    :label: comp16

    \beta a_0 + \gamma Y_t - [\beta a_1 + \gamma (1+ \beta)]Y_{t+1} + \gamma \beta Y_{t+2} =0


The Key Insight
---------------

Return to equation :eq:`ree_comp7` and set :math:`y_t = Y_t` for all :math:`t`.

(Recall that for this section we've set :math:`n=1` to simplify the
calculations)

A small amount of algebra will convince you that when :math:`y_t=Y_t`, equations :eq:`comp16` and :eq:`ree_comp7` are identical.

Thus, the Euler equation for the planning problem matches the second-order difference equation
that we derived by

#. finding the Euler equation of the representative firm and

#. substituting into it the expression :math:`Y_t = n y_t` that "makes the representative firm be representative".

If it is appropriate to apply the same terminal conditions for these two difference equations, which it is, then we have verified that a solution of the planning problem is also a rational expectations equilibrium quantity sequence.

.. Setting :math:`y_t = Y_t` in equation :eq:`ree_comp7` amounts to dropping equation :eq:`ree_hlom` and instead solving for the coefficients :math:`\kappa_0, \kappa_1` that make :math:`y_t = Y_t` true and that jointly solve equations :eq:`ree_hlom` and :eq:`ree_comp7`.

It follows that for this example we can compute equilibrium quantities by forming the optimal linear regulator problem corresponding to the Bellman equation :eq:`comp12`.

The optimal policy function for the planning problem is the aggregate law of motion
:math:`H` that the representative firm faces within a rational expectations equilibrium.


Structure of the Law of Motion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As you are asked to show in the exercises, the fact that the planner's
problem is an LQ problem implies an optimal policy --- and hence aggregate law
of motion --- taking the form

.. math::
    :label: ree_hlom2

    Y_{t+1}
    = \kappa_0 + \kappa_1 Y_t


for some parameter pair :math:`\kappa_0, \kappa_1`.


Now that we know the aggregate law of motion is linear, we can see from the
firm's Bellman equation :eq:`comp4` that the firm's problem can also be framed as
an LQ problem.

As you're asked to show in the exercises, the LQ formulation of the firm's
problem implies a law of motion that looks as follows

.. math::
    :label: ree_ex5

    y_{t+1} = h_0 + h_1 y_t + h_2 Y_t


Hence a rational expectations equilibrium will be defined by the parameters
:math:`(\kappa_0, \kappa_1, h_0, h_1, h_2)` in :eq:`ree_hlom2`--:eq:`ree_ex5`.


Exercises
=========

.. _ree_ex1:

Exercise 1
----------

Consider the firm problem :ref:`described above <ree_fp>`.

Let the firm's belief function :math:`H` be as given in :eq:`ree_hlom2`.


Formulate the firm's problem as a discounted optimal linear regulator problem, being careful to describe all of the objects needed.


Use the class ``LQ`` from the `QuantEcon.py <http://quantecon.org/python_index.html>`_ package to solve the firm's problem for the following parameter values:

.. math::

    a_0= 100, a_1= 0.05, \beta = 0.95, \gamma=10, \kappa_0 = 95.5, \kappa_1 = 0.95


Express the solution of the firm's problem in the form :eq:`ree_ex5` and give the values for each :math:`h_j`.

If there were :math:`n` identical competitive firms all behaving according to :eq:`ree_ex5`, what would :eq:`ree_ex5`  imply for the *actual* law of motion :eq:`ree_hlom` for market supply.



.. _ree_ex2:

Exercise 2
----------

Consider the following :math:`\kappa_0, \kappa_1` pairs as candidates for the
aggregate law of motion component of a rational expectations equilibrium (see
:eq:`ree_hlom2`).

Extending the program that you wrote for exercise 1, determine which if any
satisfy :ref:`the definition <ree_def>` of a rational expectations equilibrium

*  (94.0886298678, 0.923409232937)

*  (93.2119845412, 0.984323478873)

*  (95.0818452486, 0.952459076301)


Describe an iterative algorithm that uses the program that you wrote for exercise 1 to compute a rational expectations equilibrium.

(You are not being asked actually to use the algorithm you are suggesting)




.. _ree_ex3:

Exercise 3
----------


Recall the planner's problem :ref:`described above <ree_pp>`

#. Formulate the planner's problem as an LQ problem.

#. Solve it using the same parameter values in exercise 1

    * :math:`a_0= 100, a_1= 0.05, \beta = 0.95, \gamma=10`

#.  Represent the solution in the form :math:`Y_{t+1} = \kappa_0 + \kappa_1 Y_t`.

#.  Compare your answer with the results from exercise 2.




.. _ree_ex4:

Exercise 4
----------

A monopolist faces the industry demand curve :eq:`ree_comp3d`  and chooses :math:`\{Y_t\}` to maximize :math:`\sum_{t=0}^{\infty} \beta^t r_t` where

.. math::

    r_t = p_t Y_t - \frac{\gamma (Y_{t+1} - Y_t)^2 }{2}


Formulate this problem as an LQ problem.

Compute the optimal policy using the same parameters as the previous exercise.

In particular, solve for the parameters in

.. math::

    Y_{t+1} = m_0 + m_1 Y_t


Compare your results with the previous exercise -- comment.



Solutions
=========

Exercise 1
----------

To map a problem into a `discounted optimal linear control
problem <https://lectures.quantecon.org/py/lqcontrol.html>`__, we need to define

-  state vector :math:`x_t` and control vector :math:`u_t`

-  matrices :math:`A, B, Q, R` that define preferences and the law of
   motion for the state

For the state and control vectors, we choose

.. math::


       x_t = \begin{bmatrix} y_t \\ Y_t \\ 1 \end{bmatrix},
       \qquad
       u_t = y_{t+1} - y_{t}

For :math:`B, Q, R` we set

.. math::


       A =
       \begin{bmatrix}
           1 & 0 & 0 \\
           0 & \kappa_1 & \kappa_0 \\
           0 & 0 & 1
       \end{bmatrix},
       \quad
       B = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} ,
       \quad
       R =
       \begin{bmatrix}
           0 & a_1/2 & -a_0/2 \\
           a_1/2 & 0 & 0 \\
           -a_0/2 & 0 & 0
       \end{bmatrix},
       \quad
       Q = \gamma / 2

By multiplying out you can confirm that

-  :math:`x_t' R x_t + u_t' Q u_t = - r_t`

-  :math:`x_{t+1} = A x_t + B u_t`

We'll use the module ``lqcontrol.py`` to solve the firm's problem at the
stated parameter values.

This will return an LQ policy :math:`F` with the interpretation
:math:`u_t = - F x_t`, or

.. math::


       y_{t+1} - y_t = - F_0 y_t - F_1 Y_t - F_2

Matching parameters with :math:`y_{t+1} = h_0 + h_1 y_t + h_2 Y_t` leads
to

.. math::


       h_0 = -F_2, \quad h_1 = 1 - F_0, \quad h_2 = -F_1

Here's our solution

.. code-block:: python3


    # Model parameters

    a0 = 100
    a1 = 0.05
    β = 0.95
    γ = 10.0

    # Beliefs

    κ0 = 95.5
    κ1 = 0.95

    # Formulate the LQ problem

    A = np.array([[1, 0, 0], [0, κ1, κ0], [0, 0, 1]])
    B = np.array([1, 0, 0])
    B.shape = 3, 1
    R = np.array([[0, a1/2, -a0/2], [a1/2, 0, 0], [-a0/2, 0, 0]])
    Q = 0.5 * γ

    # Solve for the optimal policy

    lq = LQ(Q, R, A, B, beta=β)
    P, F, d = lq.stationary_values()
    F = F.flatten()
    out1 = f"F = [{F[0]:.3f}, {F[1]:.3f}, {F[2]:.3f}]"
    h0, h1, h2 = -F[2], 1 - F[0], -F[1]
    out2 = f"(h0, h1, h2) = ({h0:.3f}, {h1:.3f}, {h2:.3f})"

    print(out1)
    print(out2)


The implication is that

.. math::


       y_{t+1} = 96.949 + y_t - 0.046 \, Y_t

For the case :math:`n > 1`, recall that :math:`Y_t = n y_t`, which,
combined with the previous equation, yields

.. math::


       Y_{t+1}
       = n \left( 96.949 + y_t - 0.046 \, Y_t \right)
       = n 96.949 + (1 - n 0.046) Y_t

Exercise 2
----------

To determine whether a :math:`\kappa_0, \kappa_1` pair forms the
aggregate law of motion component of a rational expectations
equilibrium, we can proceed as follows:

-  Determine the corresponding firm law of motion
   :math:`y_{t+1} = h_0 + h_1 y_t + h_2 Y_t`.

-  Test whether the associated aggregate law
   ::math:`Y_{t+1} = n h(Y_t/n, Y_t)` evaluates to
   :math:`Y_{t+1} = \kappa_0 + \kappa_1 Y_t`.

In the second step, we can use :math:`Y_t = n y_t = y_t`, so that
:math:`Y_{t+1} = n h(Y_t/n, Y_t)` becomes

.. math::


       Y_{t+1} = h(Y_t, Y_t) = h_0 + (h_1 + h_2) Y_t

Hence to test the second step we can test :math:`\kappa_0 = h_0` and
:math:`\kappa_1 = h_1 + h_2`.

The following code implements this test

.. code-block:: python3


    candidates = ((94.0886298678, 0.923409232937),
                  (93.2119845412, 0.984323478873),
                  (95.0818452486, 0.952459076301))

    for κ0, κ1 in candidates:

        # Form the associated law of motion
        A = np.array([[1, 0, 0], [0, κ1, κ0], [0, 0, 1]])

        # Solve the LQ problem for the firm
        lq = LQ(Q, R, A, B, beta=β)
        P, F, d = lq.stationary_values()
        F = F.flatten()
        h0, h1, h2 = -F[2], 1 - F[0], -F[1]

        # Test the equilibrium condition
        if np.allclose((κ0, κ1), (h0, h1 + h2)):
            print(f'Equilibrium pair = {κ0}, {κ1}')
            print('f(h0, h1, h2) = {h0}, {h1}, {h2}')
            break


The output tells us that the answer is pair (iii), which implies
:math:`(h_0, h_1, h_2) = (95.0819, 1.0000, -.0475)`.

(Notice we use ``np.allclose`` to test equality of floating-point
numbers, since exact equality is too strict).

Regarding the iterative algorithm, one could loop from a given
:math:`(\kappa_0, \kappa_1)` pair to the associated firm law and then to
a new :math:`(\kappa_0, \kappa_1)` pair.

This amounts to implementing the operator :math:`\Phi` described in the
lecture.

(There is in general no guarantee that this iterative process will
converge to a rational expectations equilibrium)

Exercise 3
----------

We are asked to write the planner problem as an LQ problem.

For the state and control vectors, we choose

.. math::


       x_t = \begin{bmatrix} Y_t \\ 1 \end{bmatrix},
       \quad
       u_t = Y_{t+1} - Y_{t}

For the LQ matrices, we set

.. math::


       A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},
       \quad
       B = \begin{bmatrix} 1 \\ 0 \end{bmatrix},
       \quad
       R = \begin{bmatrix} a_1/2 & -a_0/2 \\ -a_0/2 & 0 \end{bmatrix},
       \quad
       Q = \gamma / 2

By multiplying out you can confirm that

-  :math:`x_t' R x_t + u_t' Q u_t = - s(Y_t, Y_{t+1})`

-  :math:`x_{t+1} = A x_t + B u_t`

By obtaining the optimal policy and using :math:`u_t = - F x_t` or

.. math::


       Y_{t+1} - Y_t = -F_0 Y_t - F_1

we can obtain the implied aggregate law of motion via
:math:`\kappa_0 = -F_1` and :math:`\kappa_1 = 1-F_0`.

The Python code to solve this problem is below:

.. code-block:: python3


    # Formulate the planner's LQ problem

    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1], [0]])
    R = np.array([[a1 / 2, -a0 / 2], [-a0 / 2, 0]])
    Q = γ / 2

    # Solve for the optimal policy

    lq = LQ(Q, R, A, B, beta=β)
    P, F, d = lq.stationary_values()

    # Print the results

    F = F.flatten()
    κ0, κ1 = -F[1], 1 - F[0]
    print(κ0, κ1)


The output yields the same :math:`(\kappa_0, \kappa_1)` pair obtained as
an equilibrium from the previous exercise.

Exercise 4
----------

The monopolist's LQ problem is almost identical to the planner's problem
from the previous exercise, except that

.. math::


       R = \begin{bmatrix}
           a_1 & -a_0/2 \\
           -a_0/2 & 0
       \end{bmatrix}

The problem can be solved as follows

.. code-block:: python3


    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1], [0]])
    R = np.array([[a1, -a0 / 2], [-a0 / 2, 0]])
    Q = γ / 2

    lq = LQ(Q, R, A, B, beta=β)
    P, F, d = lq.stationary_values()

    F = F.flatten()
    m0, m1 = -F[1], 1 - F[0]
    print(m0, m1)


We see that the law of motion for the monopolist is approximately
:math:`Y_{t+1} = 73.4729 + 0.9265 Y_t`.

In the rational expectations case, the law of motion was approximately
:math:`Y_{t+1} = 95.0818 + 0.9525 Y_t`.

One way to compare these two laws of motion is by their fixed points,
which give long-run equilibrium output in each case.

For laws of the form :math:`Y_{t+1} = c_0 + c_1 Y_t`, the fixed point is
:math:`c_0 / (1 - c_1)`.

If you crunch the numbers, you will see that the monopolist adopts a
lower long-run quantity than obtained by the competitive market,
implying a higher market price.

This is analogous to the elementary static-case results




.. rubric:: Footnotes


.. [#fn_im]
    A literature that studies whether models populated  with agents
    who learn can converge  to rational expectations equilibria features
    iterations on a modification of the mapping :math:`\Phi` that can be
    approximated as :math:`\gamma \Phi + (1-\gamma)I`. Here :math:`I` is the
    identity operator and :math:`\gamma \in (0,1)` is a *relaxation parameter*.
    See :cite:`MarcetSargent1989` and :cite:`EvansHonkapohja2001` for statements
    and applications of this approach to establish conditions under which
    collections of adaptive agents who use least squares learning to converge to a
    rational expectations equilibrium.
