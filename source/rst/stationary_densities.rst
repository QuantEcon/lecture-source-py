.. _statd:

.. include:: /_static/includes/header.raw

.. highlight:: python3

***************************************
:index:`Continuous State Markov Chains`
***************************************

.. index::
    single: Markov Chains; Continuous State

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========

In a :doc:`previous lecture <finite_markov>`, we learned about finite Markov chains, a relatively elementary class of stochastic dynamic models.

The present lecture extends this analysis to continuous (i.e., uncountable) state Markov chains.

Most stochastic dynamic models studied by economists either fit directly into this class or can be represented as continuous state Markov chains after minor modifications.

In this lecture, our focus will be on continuous Markov models that

* evolve in discrete-time
* are often nonlinear

The fact that we accommodate nonlinear models here is significant, because
linear stochastic models have their own highly developed toolset, as we'll
see :doc:`later on <arma>`.

The question that interests us most is: Given a particular stochastic dynamic
model, how will the state of the system evolve over time?

In particular,

* What happens to the distribution of the state variables?

* Is there anything we can say about the "average behavior" of these variables?

* Is there a notion of "steady state" or "long-run equilibrium" that's applicable to the model?

    * If so, how can we compute it?


Answering these questions will lead us to revisit many of the topics that occupied us in the finite state case,
such as simulation, distribution dynamics, stability, ergodicity, etc.

.. note::
    For some people, the term "Markov chain" always refers to a process with a
    finite or discrete state space.  We follow the mainstream
    mathematical literature (e.g., :cite:`MeynTweedie2009`) in using the term to refer to any discrete **time**
    Markov process.

Let's start with some standard imports:

.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    from scipy.stats import lognorm, beta
    from quantecon import LAE
    from scipy.stats import norm, gaussian_kde


.. _statd_density_case:

The Density Case
================

You are probably aware that some distributions can be represented by densities
and some cannot.

(For example, distributions on the real numbers :math:`\mathbb R` that put positive probability
on individual points have no density representation)

We are going to start our analysis by looking at Markov chains where the one-step transition probabilities have density representations.

The benefit is that the density case offers a very direct parallel to the finite case in terms of notation and intuition.

Once we've built some intuition we'll cover the general case.



Definitions and Basic Properties
--------------------------------

In our :doc:`lecture on finite Markov chains <finite_markov>`, we studied discrete-time Markov chains that evolve on a finite state space :math:`S`.

In this setting, the dynamics of the model are described by a stochastic matrix --- a nonnegative square matrix :math:`P = P[i, j]` such that each row :math:`P[i, \cdot]` sums to one.

The interpretation of :math:`P` is that :math:`P[i, j]` represents the
probability of transitioning from state :math:`i` to state :math:`j` in one
unit of time.

In symbols,

.. math::

    \mathbb P \{ X_{t+1} = j \,|\, X_t = i \} = P[i, j]


Equivalently,

* :math:`P` can be thought of as a family of distributions :math:`P[i, \cdot]`, one for each :math:`i \in S`

* :math:`P[i, \cdot]` is the distribution of :math:`X_{t+1}` given :math:`X_t = i`

(As you probably recall, when using NumPy arrays, :math:`P[i, \cdot]` is expressed as ``P[i,:]``)

In this section, we'll allow :math:`S` to be a subset of :math:`\mathbb R`, such as

* :math:`\mathbb R` itself
* the positive reals :math:`(0, \infty)`
* a bounded interval :math:`(a, b)`

The family of discrete distributions :math:`P[i, \cdot]` will be replaced by a family of densities :math:`p(x, \cdot)`, one for each :math:`x \in S`.

Analogous to the finite state case, :math:`p(x, \cdot)` is to be understood as the distribution (density) of :math:`X_{t+1}` given :math:`X_t = x`.

More formally, a *stochastic kernel on* :math:`S` is a function :math:`p \colon S \times S \to \mathbb R` with the property that

#. :math:`p(x, y) \geq 0` for all :math:`x, y \in S`

#. :math:`\int p(x, y) dy = 1` for all :math:`x \in S`

(Integrals are over the whole space unless otherwise specified)

For example, let :math:`S = \mathbb R` and consider the particular stochastic
kernel :math:`p_w` defined by

.. math::
    :label: statd_rwsk

    p_w(x, y) := \frac{1}{\sqrt{2 \pi}} \exp \left\{ - \frac{(y - x)^2}{2} \right\}


What kind of model does :math:`p_w` represent?

The answer is, the (normally distributed) random walk

.. math::
    :label: statd_rw

    X_{t+1} = X_t + \xi_{t+1}
    \quad \text{where} \quad
    \{ \xi_t \} \stackrel {\textrm{ IID }} {\sim} N(0, 1)


To see this, let's find the stochastic kernel :math:`p` corresponding to :eq:`statd_rw`.

Recall that :math:`p(x, \cdot)` represents the distribution of :math:`X_{t+1}` given :math:`X_t = x`.

Letting :math:`X_t = x` in :eq:`statd_rw` and considering the distribution of :math:`X_{t+1}`, we see that :math:`p(x, \cdot) = N(x, 1)`.

In other words, :math:`p` is exactly :math:`p_w`, as defined in :eq:`statd_rwsk`.



Connection to Stochastic Difference Equations
---------------------------------------------

In the previous section, we made the connection between stochastic difference
equation :eq:`statd_rw` and stochastic kernel :eq:`statd_rwsk`.

In economics and time-series analysis we meet stochastic difference equations of all different shapes and sizes.

It will be useful for us if we have some systematic methods for converting stochastic difference equations into stochastic kernels.

To this end, consider the generic (scalar) stochastic difference equation given by

.. math::
    :label: statd_srs

    X_{t+1} = \mu(X_t) + \sigma(X_t) \, \xi_{t+1}


Here we assume that

* :math:`\{ \xi_t \} \stackrel {\textrm{ IID }} {\sim} \phi`, where :math:`\phi` is a given density on :math:`\mathbb R`

* :math:`\mu` and :math:`\sigma` are given functions on :math:`S`, with :math:`\sigma(x) > 0` for all :math:`x`

**Example 1:** The random walk :eq:`statd_rw` is a special case of :eq:`statd_srs`, with :math:`\mu(x) = x` and :math:`\sigma(x) = 1`.

**Example 2:** Consider the `ARCH model <https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity>`_

.. math::

    X_{t+1} = \alpha X_t + \sigma_t \,  \xi_{t+1},
    \qquad \sigma^2_t = \beta + \gamma X_t^2,
    \qquad \beta, \gamma > 0


Alternatively, we can write the model as

.. math::
    :label: statd_arch

    X_{t+1} = \alpha X_t + (\beta + \gamma X_t^2)^{1/2} \xi_{t+1}


This is a special case of :eq:`statd_srs` with :math:`\mu(x) = \alpha x` and :math:`\sigma(x) = (\beta + \gamma x^2)^{1/2}`.

.. _solow_swan:

**Example 3:** With stochastic production and a constant savings rate, the one-sector neoclassical growth model leads to a law of motion for capital per worker such as

.. math::
    :label: statd_ss

    k_{t+1} = s  A_{t+1} f(k_t) + (1 - \delta) k_t


Here

* :math:`s` is the rate of savings

* :math:`A_{t+1}` is a production shock

    * The :math:`t+1` subscript indicates that :math:`A_{t+1}` is not visible at time :math:`t`

* :math:`\delta` is a depreciation rate

* :math:`f \colon \mathbb R_+ \to \mathbb R_+` is a production function satisfying :math:`f(k) > 0` whenever :math:`k > 0`

(The fixed savings rate can be rationalized as the optimal policy for a particular set of technologies and preferences (see :cite:`Ljungqvist2012`, section
3.1.2), although we omit the details here).

Equation :eq:`statd_ss` is a special case of :eq:`statd_srs` with :math:`\mu(x) = (1 - \delta)x` and :math:`\sigma(x) = s f(x)`.

Now let's obtain the stochastic kernel corresponding to the generic model :eq:`statd_srs`.

To find it, note first that if :math:`U` is a random variable with
density :math:`f_U`, and :math:`V = a + b U` for some constants :math:`a,b`
with :math:`b > 0`, then the density of :math:`V` is given by

.. math::
    :label: statd_dv

    f_V(v)
    = \frac{1}{b}
    f_U \left( \frac{v - a}{b} \right)


(The proof is :ref:`below <statd_appendix>`.  For a multidimensional version
see `EDTC <http://johnstachurski.net/edtc.html>`_, theorem 8.1.3).

Taking :eq:`statd_dv` as given for the moment, we can
obtain the stochastic kernel :math:`p` for :eq:`statd_srs` by recalling that
:math:`p(x, \cdot)` is the conditional density of :math:`X_{t+1}` given
:math:`X_t = x`.

In the present case, this is equivalent to stating that :math:`p(x, \cdot)` is the density of :math:`Y := \mu(x) + \sigma(x) \, \xi_{t+1}` when :math:`\xi_{t+1} \sim \phi`.

Hence, by :eq:`statd_dv`,

.. math::
    :label: statd_srssk

    p(x, y)
    = \frac{1}{\sigma(x)}
    \phi \left( \frac{y - \mu(x)}{\sigma(x)} \right)


For example, the growth model in :eq:`statd_ss` has stochastic kernel

.. math::
    :label: statd_sssk

    p(x, y)
    = \frac{1}{sf(x)}
    \phi \left( \frac{y - (1 - \delta) x}{s f(x)} \right)


where :math:`\phi` is the density of :math:`A_{t+1}`.

(Regarding the state space :math:`S` for this model, a natural choice is :math:`(0, \infty)` --- in which case
:math:`\sigma(x) = s f(x)` is strictly positive for all :math:`s` as required)



Distribution Dynamics
---------------------

In :ref:`this section <mc_md>` of our lecture on **finite** Markov chains, we
asked the following question: If

#. :math:`\{X_t\}` is a Markov chain with stochastic matrix :math:`P`
#. the distribution of :math:`X_t` is known to be :math:`\psi_t`

then what is the distribution of :math:`X_{t+1}`?

Letting :math:`\psi_{t+1}` denote the distribution of :math:`X_{t+1}`, the
answer :ref:`we gave <mc_fdd>` was that

.. math::

    \psi_{t+1}[j] = \sum_{i \in S} P[i,j] \psi_t[i]


This intuitive equality states that the probability of being at :math:`j`
tomorrow is the probability of visiting :math:`i` today and then going on to
:math:`j`, summed over all possible :math:`i`.

In the density case, we just replace the sum with an integral and probability
mass functions with densities, yielding

.. math::
    :label: statd_fdd

    \psi_{t+1}(y) = \int p(x,y) \psi_t(x) \, dx,
    \qquad \forall y \in S


It is convenient to think of this updating process in terms of an operator.

(An operator is just a function, but the term is usually reserved for a function that sends functions into functions)

Let :math:`\mathscr D` be the set of all densities on :math:`S`, and let
:math:`P` be the operator from :math:`\mathscr D` to itself that takes density
:math:`\psi` and sends it into new density :math:`\psi P`, where the latter is
defined by

.. math::
    :label: def_dmo

    (\psi P)(y) = \int p(x,y) \psi(x) dx


This operator is usually called the *Markov operator* corresponding to :math:`p`

.. note::

    Unlike most operators, we write :math:`P` to the right of its argument,
    instead of to the left (i.e., :math:`\psi P` instead of :math:`P \psi`).
    This is a common convention, with the intention being to maintain the
    parallel with the finite case --- see :ref:`here <mc_fddv>`

With this notation, we can write :eq:`statd_fdd` more succinctly as :math:`\psi_{t+1}(y) = (\psi_t P)(y)` for all :math:`y`, or, dropping the :math:`y` and letting ":math:`=`" indicate equality of functions,

.. math::
    :label: statd_p

    \psi_{t+1} = \psi_t P


Equation :eq:`statd_p` tells us that if we specify a distribution for :math:`\psi_0`, then the entire sequence
of future distributions can be obtained by iterating with :math:`P`.

It's interesting to note that :eq:`statd_p` is a deterministic difference equation.

Thus, by converting a stochastic difference equation such as
:eq:`statd_srs` into a stochastic kernel :math:`p` and hence an operator
:math:`P`, we convert a stochastic difference equation into a deterministic
one (albeit in a much higher dimensional space).

.. note::

    Some people might be aware that discrete Markov chains are in fact
    a special case of the continuous Markov chains we have just described.  The reason is
    that probability mass functions are densities with respect to
    the `counting measure <https://en.wikipedia.org/wiki/Counting_measure>`_.

Computation
-----------

To learn about the dynamics of a given process, it's useful to compute and study the sequences of densities generated by the model.

One way to do this is to try to implement the iteration described by :eq:`def_dmo` and :eq:`statd_p` using numerical integration.

However, to produce :math:`\psi P` from :math:`\psi` via :eq:`def_dmo`, you
would need to integrate at every :math:`y`, and there is a continuum of such
:math:`y`.

Another possibility is to discretize the model, but this introduces errors of unknown size.

A nicer alternative in the present setting is to combine simulation with an elegant estimator called the *look-ahead* estimator.

Let's go over the ideas with reference to the growth model :ref:`discussed above <solow_swan>`, the dynamics of which we repeat here for convenience:

.. math::
    :label: statd_ss2

    k_{t+1} = s  A_{t+1} f(k_t) + (1 - \delta) k_t


Our aim is to compute the sequence :math:`\{ \psi_t \}` associated with this model and fixed initial condition :math:`\psi_0`.

To approximate :math:`\psi_t` by simulation, recall that, by definition, :math:`\psi_t` is the density of :math:`k_t` given :math:`k_0 \sim \psi_0`.

If we wish to generate observations of this random variable,  all we need to do is

#. draw :math:`k_0` from the specified initial condition :math:`\psi_0`

#. draw the shocks :math:`A_1, \ldots, A_t` from their specified density :math:`\phi`

#. compute :math:`k_t` iteratively via :eq:`statd_ss2`

If we repeat this :math:`n` times, we get :math:`n` independent observations :math:`k_t^1, \ldots, k_t^n`.

With these draws in hand, the next step is to generate some kind of representation of their distribution :math:`\psi_t`.

A naive approach would be to use a histogram, or perhaps a `smoothed histogram <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ using  SciPy's ``gaussian_kde`` function.

However, in the present setting, there is a much better way to do this, based on the look-ahead estimator.

With this estimator, to construct an estimate of :math:`\psi_t`, we
actually generate :math:`n` observations of :math:`k_{t-1}`, rather than :math:`k_t`.

Now we take these :math:`n` observations :math:`k_{t-1}^1, \ldots,
k_{t-1}^n` and form the estimate

.. math::
    :label: statd_lae1

    \psi_t^n(y) = \frac{1}{n} \sum_{i=1}^n p(k_{t-1}^i, y)


where :math:`p` is the growth model stochastic kernel in :eq:`statd_sssk`.

What is the justification for this slightly surprising estimator?

The idea is that, by the strong :ref:`law of large numbers <lln_ksl>`,

.. math::

    \frac{1}{n} \sum_{i=1}^n p(k_{t-1}^i, y)
    \to
    \mathbb E p(k_{t-1}^i, y)
    = \int p(x, y) \psi_{t-1}(x) \, dx
    = \psi_t(y)


with probability one as :math:`n \to \infty`.

Here the first equality is by the definition of :math:`\psi_{t-1}`, and the
second is by :eq:`statd_fdd`.

We have just shown that our estimator :math:`\psi_t^n(y)` in :eq:`statd_lae1`
converges almost surely to :math:`\psi_t(y)`, which is just what we want to compute.

.. only:: html

    In fact, much stronger convergence results are true (see, for example, :download:`this paper </_static/lecture_specific/stationary_densities/ECTA6180.pdf>`).

.. only:: latex

    In fact, much stronger convergence results are true (see, for example, `this paper <https://lectures.quantecon.org/_downloads/ECTA6180.pdf>`__).





Implementation
--------------



A class called ``LAE`` for estimating densities by this technique can be found in `lae.py <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lae.py>`__.





Given our use of the ``__call__`` method, an instance  of ``LAE`` acts as a callable object, which is essentially a function that can store its own data  (see :ref:`this discussion <call_method>`).




This function returns the right-hand side of :eq:`statd_lae1` using



* the data and stochastic kernel that it stores as its instance data

* the value :math:`y` as its argument



The function is vectorized, in the sense that if ``psi`` is such an instance and ``y`` is an array, then the call ``psi(y)`` acts elementwise.

(This is the reason that we reshaped ``X`` and ``y`` inside the class --- to make vectorization work)



Because the implementation is fully vectorized, it is about as efficient as it
would be in C or Fortran.



Example
-------

The following code is an example of usage for the stochastic growth model :ref:`described above <solow_swan>`



.. code-block:: python3

    # == Define parameters == #
    s = 0.2
    δ = 0.1
    a_σ = 0.4                    # A = exp(B) where B ~ N(0, a_σ)
    α = 0.4                      # We set f(k) = k**α
    ψ_0 = beta(5, 5, scale=0.5)  # Initial distribution
    ϕ = lognorm(a_σ)


    def p(x, y):
        """
        Stochastic kernel for the growth model with Cobb-Douglas production.
        Both x and y must be strictly positive.
        """
        d = s * x**α
        return ϕ.pdf((y - (1 - δ) * x) / d) / d

    n = 10000    # Number of observations at each date t
    T = 30       # Compute density of k_t at 1,...,T+1

    # == Generate matrix s.t. t-th column is n observations of k_t == #
    k = np.empty((n, T))
    A = ϕ.rvs((n, T))
    k[:, 0] = ψ_0.rvs(n)  # Draw first column from initial distribution
    for t in range(T-1):
        k[:, t+1] = s * A[:, t] * k[:, t]**α + (1 - δ) * k[:, t]

    # == Generate T instances of LAE using this data, one for each date t == #
    laes = [LAE(p, k[:, t]) for t in range(T)]

    # == Plot == #
    fig, ax = plt.subplots()
    ygrid = np.linspace(0.01, 4.0, 200)
    greys = [str(g) for g in np.linspace(0.0, 0.8, T)]
    greys.reverse()
    for ψ, g in zip(laes, greys):
        ax.plot(ygrid, ψ(ygrid), color=g, lw=2, alpha=0.6)
    ax.set_xlabel('capital')
    ax.set_title(f'Density of $k_1$ (lighter) to $k_T$ (darker) for $T={T}$')
    plt.show()



The figure shows part of the density sequence :math:`\{\psi_t\}`, with each
density computed via the look-ahead estimator.

Notice that the sequence of densities shown in the figure seems to be
converging --- more on this in just a moment.

Another quick comment is that each of these distributions could be interpreted
as a cross-sectional distribution (recall :ref:`this discussion <mc_eg1-1>`).



Beyond Densities
================

Up until now, we have focused exclusively on continuous state Markov chains
where all conditional distributions :math:`p(x, \cdot)` are densities.

As discussed above, not all distributions can be represented as densities.

If the conditional distribution of :math:`X_{t+1}` given :math:`X_t = x`
**cannot** be represented as a density for some :math:`x \in S`, then we need a slightly
different theory.

The ultimate option is to switch from densities to `probability measures
<https://en.wikipedia.org/wiki/Probability_measure>`_, but not all readers will
be familiar with measure theory.

We can, however, construct a fairly general theory using distribution functions.


Example and Definitions
-----------------------

To illustrate the issues, recall that Hopenhayn and Rogerson :cite:`HopenhaynRogerson1993` study a model of firm dynamics where individual firm productivity follows the exogenous process

.. math::

    X_{t+1} = a + \rho X_t + \xi_{t+1},
    \quad \text{where} \quad
    \{ \xi_t \} \stackrel {\textrm{ IID }} {\sim} N(0, \sigma^2)


As is, this fits into the density case we treated above.

However, the authors wanted this process to take values in :math:`[0, 1]`, so they added boundaries at the endpoints 0 and 1.

One way to write this is

.. math::

    X_{t+1} = h(a + \rho X_t + \xi_{t+1})
    \quad \text{where} \quad
    h(x) := x \, \mathbf 1\{0 \leq x \leq 1\} + \mathbf 1 \{ x > 1\}


If you think about it, you will see that for any given :math:`x \in [0, 1]`,
the conditional distribution of :math:`X_{t+1}` given :math:`X_t = x`
puts positive probability mass on 0 and 1.

Hence it cannot be represented as a density.

What we can do instead is use cumulative distribution functions (cdfs).

To this end, set

.. math::

    G(x, y) := \mathbb P \{ h(a + \rho x + \xi_{t+1}) \leq y \}
    \qquad (0 \leq x, y \leq 1)


This family of cdfs :math:`G(x, \cdot)` plays a role analogous to the stochastic kernel in the density case.

The distribution dynamics in :eq:`statd_fdd` are then replaced by

.. math::
    :label: statd_fddc

    F_{t+1}(y) = \int G(x,y) F_t(dx)


Here :math:`F_t` and :math:`F_{t+1}` are cdfs representing the distribution of the current state and next period state.

The intuition behind :eq:`statd_fddc` is essentially the same as for :eq:`statd_fdd`.



Computation
-----------

If you wish to compute these cdfs, you cannot use the look-ahead estimator as before.

Indeed, you should not use any density estimator, since the objects you are
estimating/computing are not densities.

One good option is simulation as before, combined with the `empirical distribution function <https://en.wikipedia.org/wiki/Empirical_distribution_function>`__.





Stability
=========

In our :doc:`lecture <finite_markov>` on finite Markov chains, we also studied stationarity, stability and ergodicity.

Here we will cover the same topics for the continuous case.

We will, however, treat only the density case (as in :ref:`this section <statd_density_case>`), where the stochastic kernel is a family of densities.

The general case is relatively similar --- references are given below.



Theoretical Results
-------------------

Analogous to :ref:`the finite case <mc_stat_dd>`, given a stochastic kernel :math:`p` and corresponding Markov operator as
defined in :eq:`def_dmo`, a density :math:`\psi^*` on :math:`S` is called
*stationary* for :math:`P` if it is a fixed point of the operator :math:`P`.

In other words,

.. math::
    :label: statd_dsd

    \psi^*(y) = \int p(x,y) \psi^*(x) \, dx,
    \qquad \forall y \in S


As with the finite case, if :math:`\psi^*` is stationary for :math:`P`, and
the distribution of :math:`X_0` is :math:`\psi^*`, then, in view of
:eq:`statd_p`, :math:`X_t` will have this same distribution for all :math:`t`.

Hence :math:`\psi^*` is the stochastic equivalent of a steady state.

In the finite case, we learned that at least one stationary distribution exists, although there may be many.

When the state space is infinite, the situation is more complicated.

Even existence can fail very easily.

For example, the random walk model has no stationary density (see, e.g., `EDTC <http://johnstachurski.net/edtc.html>`_, p. 210).

However, there are well-known conditions under which a stationary density :math:`\psi^*` exists.

With additional conditions, we can also get a unique stationary density (:math:`\psi \in \mathscr D \text{ and } \psi = \psi P \implies \psi = \psi^*`),  and also global convergence in the sense that

.. math::
    :label: statd_dca

    \forall \, \psi \in \mathscr D, \quad \psi P^t \to \psi^*
        \quad \text{as} \quad t \to \infty


This combination of existence, uniqueness and global convergence in the sense
of :eq:`statd_dca` is often referred to as *global stability*.

Under very similar conditions, we get *ergodicity*, which means that

.. math::
    :label: statd_lln

    \frac{1}{n} \sum_{t = 1}^n h(X_t)  \to \int h(x) \psi^*(x) dx
        \quad \text{as } n \to \infty


for any (`measurable <https://en.wikipedia.org/wiki/Measurable_function>`_) function :math:`h \colon S \to \mathbb R`  such that the right-hand side is finite.

Note that the convergence in :eq:`statd_lln` does not depend on the distribution (or value) of :math:`X_0`.

This is actually very important for simulation --- it means we can learn about :math:`\psi^*` (i.e., approximate the right-hand side of :eq:`statd_lln` via the left-hand side) without requiring any special knowledge about what to do with :math:`X_0`.

So what are these conditions we require to get global stability and ergodicity?

In essence, it must be the case that

#. Probability mass does not drift off to the "edges" of the state space.

#. Sufficient "mixing" obtains.

For one such set of conditions see theorem 8.2.14 of `EDTC <http://johnstachurski.net/edtc.html>`_.

In addition

* :cite:`StokeyLucas1989`  contains a classic (but slightly outdated) treatment of these topics.

* From the mathematical literature, :cite:`LasotaMackey1994`  and :cite:`MeynTweedie2009` give outstanding in-depth treatments.

* Section 8.1.2 of `EDTC <http://johnstachurski.net/edtc.html>`_ provides detailed intuition, and section 8.3 gives additional references.

* `EDTC <http://johnstachurski.net/edtc.html>`_, section 11.3.4
  provides a specific treatment for the growth model we considered in this
  lecture.





An Example of Stability
-----------------------

As stated above, the :ref:`growth model treated here <solow_swan>` is stable under mild conditions
on the primitives.

* See `EDTC <http://johnstachurski.net/edtc.html>`_, section 11.3.4 for more details.

We can see this stability in action --- in particular, the convergence in :eq:`statd_dca` --- by simulating the path of densities from various initial conditions.

Here is such a figure.

.. _statd_egs:

.. figure:: /_static/lecture_specific/stationary_densities/solution_statd_ex2.png

All sequences are converging towards the same limit, regardless of their initial condition.

The details regarding initial conditions and so on are given in :ref:`this exercise <statd_ex2>`, where you are asked to replicate the figure.



Computing Stationary Densities
------------------------------


In the preceding figure, each sequence of densities is converging towards the unique stationary density :math:`\psi^*`.

Even from this figure, we can get a fair idea what :math:`\psi^*` looks like, and where its mass is located.

However, there is a much more direct way to estimate the stationary density,
and it involves only a slight modification of the look-ahead estimator.

Let's say that we have a model of the form :eq:`statd_srs` that is stable and
ergodic.

Let :math:`p` be the corresponding stochastic kernel, as given in :eq:`statd_srssk`.

To approximate the stationary density :math:`\psi^*`, we can simply generate a
long time-series :math:`X_0, X_1, \ldots, X_n` and estimate :math:`\psi^*` via

.. math::
    :label: statd_lae2

    \psi_n^*(y) = \frac{1}{n} \sum_{t=1}^n p(X_t, y)


This is essentially the same as the look-ahead estimator :eq:`statd_lae1`,
except that now the observations we generate are a single time-series, rather
than a cross-section.

The justification for :eq:`statd_lae2` is that, with probability one as :math:`n \to \infty`,

.. math::

    \frac{1}{n} \sum_{t=1}^n p(X_t, y)
    \to
    \int p(x, y) \psi^*(x) \, dx
    = \psi^*(y)


where the convergence is by :eq:`statd_lln` and the equality on the right is by
:eq:`statd_dsd`.

The right-hand side is exactly what we want to compute.

On top of this asymptotic result, it turns out that the rate of convergence
for the look-ahead estimator is very good.

The first exercise helps illustrate this point.




Exercises
=========


.. _statd_ex1:

Exercise 1
----------

Consider the simple threshold autoregressive model

.. math::
    :label: statd_tar

    X_{t+1} = \theta |X_t| + (1- \theta^2)^{1/2} \xi_{t+1}
    \qquad \text{where} \quad
    \{ \xi_t \} \stackrel {\textrm{ IID }} {\sim} N(0, 1)


This is one of those rare nonlinear stochastic models where an analytical
expression for the stationary density is available.

In particular, provided that :math:`|\theta| < 1`, there is a unique
stationary density :math:`\psi^*` given by

.. math::
    :label: statd_tar_ts

    \psi^*(y) = 2 \, \phi(y) \, \Phi
    \left[
        \frac{\theta y}{(1 - \theta^2)^{1/2}}
    \right]


Here :math:`\phi` is the standard normal density and :math:`\Phi` is the standard normal cdf.

As an exercise, compute the look-ahead estimate of :math:`\psi^*`, as defined
in :eq:`statd_lae2`, and compare it with :math:`\psi^*`  in :eq:`statd_tar_ts` to see whether they
are indeed close for large :math:`n`.

In doing so, set :math:`\theta = 0.8` and :math:`n = 500`.

The next figure shows the result of such a computation

.. figure:: /_static/lecture_specific/stationary_densities/solution_statd_ex1.png

The additional density (black line) is a `nonparametric kernel density estimate <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_, added to the solution for illustration.

(You can try to replicate it before looking at the solution if you want to)

As you can see, the look-ahead estimator is a much tighter fit than the kernel
density estimator.

If you repeat the simulation you will see that this is consistently the case.


.. _statd_ex2:

Exercise 2
----------

Replicate the figure on global convergence :ref:`shown above <statd_egs>`.

The densities come from the stochastic growth model treated :ref:`at the start of the lecture <solow_swan>`.

Begin with the code found in `stochasticgrowth.py <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/stationary_densities/stochasticgrowth.py>`__.

Use the same parameters.



For the four initial distributions, use the shifted beta distributions

.. code-block:: python3
    :class: no-execute

    ψ_0 = beta(5, 5, scale=0.5, loc=i*2)



.. _statd_ex3:

Exercise 3
----------

A common way to compare distributions visually is with `boxplots <https://en.wikipedia.org/wiki/Box_plot>`_.

To illustrate, let's generate three artificial data sets and compare them with a boxplot.

The three data sets we will use are:

.. math::

    \{ X_1, \ldots, X_n \} \sim LN(0, 1), \;\;
    \{ Y_1, \ldots, Y_n \} \sim N(2, 1), \;\;
    \text{ and } \;
    \{ Z_1, \ldots, Z_n \} \sim N(4, 1), \;


Here is the code and figure:


.. code-block:: python3

    n = 500
    x = np.random.randn(n)        # N(0, 1)
    x = np.exp(x)                 # Map x to lognormal
    y = np.random.randn(n) + 2.0  # N(2, 1)
    z = np.random.randn(n) + 4.0  # N(4, 1)

    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.boxplot([x, y, z])
    ax.set_xticks((1, 2, 3))
    ax.set_ylim(-2, 14)
    ax.set_xticklabels(('$X$', '$Y$', '$Z$'), fontsize=16)
    plt.show()


Each data set is represented by a box, where the top and bottom of the box are the third and first quartiles of the data, and the red line in the center is the median.

The boxes give some indication as to

* the location of probability mass for each sample

* whether the distribution is right-skewed (as is the lognormal distribution), etc

Now let's put these ideas to use in a simulation.

Consider the threshold autoregressive model in :eq:`statd_tar`.

We know that the distribution of :math:`X_t` will converge to :eq:`statd_tar_ts` whenever :math:`|\theta| < 1`.

Let's observe this convergence from different initial conditions using
boxplots.

In particular, the exercise is to generate `J` boxplot figures, one for each initial condition :math:`X_0` in





.. code-block:: python3
    :class: no-execute

    initial_conditions = np.linspace(8, 0, J)



For each :math:`X_0` in this set,

#. Generate :math:`k` time-series of length :math:`n`, each starting at :math:`X_0` and obeying :eq:`statd_tar`.

#. Create a boxplot representing :math:`n` distributions, where the :math:`t`-th distribution shows the :math:`k` observations of :math:`X_t`.


Use :math:`\theta = 0.9, n = 20, k = 5000, J = 8`


.. TODO: Exercise 4, to be written: From LAE to GLAE --- GARCH as in MOR



Solutions
=========



Exercise 1
----------

Look-ahead estimation of a TAR stationary density, where the TAR model
is

.. math::     X_{t+1} = \theta |X_t| + (1 - \theta^2)^{1/2} \xi_{t+1}

and :math:`\xi_t \sim N(0,1)`.

Try running at ``n = 10, 100, 1000, 10000`` to get an idea of the speed of convergence

.. code-block:: python3

    ϕ = norm()
    n = 500
    θ = 0.8
    # == Frequently used constants == #
    d = np.sqrt(1 - θ**2)
    δ = θ / d

    def ψ_star(y):
        "True stationary density of the TAR Model"
        return 2 * norm.pdf(y) * norm.cdf(δ * y)

    def p(x, y):
        "Stochastic kernel for the TAR model."
        return ϕ.pdf((y - θ * np.abs(x)) / d) / d

    Z = ϕ.rvs(n)
    X = np.empty(n)
    for t in range(n-1):
        X[t+1] = θ * np.abs(X[t]) + d * Z[t]
    ψ_est = LAE(p, X)
    k_est = gaussian_kde(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    ys = np.linspace(-3, 3, 200)
    ax.plot(ys, ψ_star(ys), 'b-', lw=2, alpha=0.6, label='true')
    ax.plot(ys, ψ_est(ys), 'g-', lw=2, alpha=0.6, label='look-ahead estimate')
    ax.plot(ys, k_est(ys), 'k-', lw=2, alpha=0.6, label='kernel based estimate')
    ax.legend(loc='upper left')
    plt.show()

Exercise 2
----------

Here's one program that does the job

.. code-block:: python3

    # == Define parameters == #
    s = 0.2
    δ = 0.1
    a_σ = 0.4                                # A = exp(B) where B ~ N(0, a_σ)
    α = 0.4                                  # f(k) = k**α

    ϕ = lognorm(a_σ)

    def p(x, y):
        "Stochastic kernel, vectorized in x.  Both x and y must be positive."
        d = s * x**α
        return ϕ.pdf((y - (1 - δ) * x) / d) / d

    n = 1000                            # Number of observations at each date t
    T = 40                              # Compute density of k_t at 1,...,T

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    xmax = 6.5

    for i in range(4):
        ax = axes[i]
        ax.set_xlim(0, xmax)
        ψ_0 = beta(5, 5, scale=0.5, loc=i*2)  # Initial distribution

        # == Generate matrix s.t. t-th column is n observations of k_t == #
        k = np.empty((n, T))
        A = ϕ.rvs((n, T))
        k[:, 0] = ψ_0.rvs(n)
        for t in range(T-1):
            k[:, t+1] = s * A[:,t] * k[:, t]**α + (1 - δ) * k[:, t]

        # == Generate T instances of lae using this data, one for each t == #
        laes = [LAE(p, k[:, t]) for t in range(T)]

        ygrid = np.linspace(0.01, xmax, 150)
        greys = [str(g) for g in np.linspace(0.0, 0.8, T)]
        greys.reverse()
        for ψ, g in zip(laes, greys):
            ax.plot(ygrid, ψ(ygrid), color=g, lw=2, alpha=0.6)
        ax.set_xlabel('capital')
    plt.show()

Exercise 3
----------

Here's a possible solution.

Note the way we use vectorized code to simulate the :math:`k` time
series for one boxplot all at once

.. code-block:: python3

    n = 20
    k = 5000
    J = 6

    θ = 0.9
    d = np.sqrt(1 - θ**2)
    δ = θ / d

    fig, axes = plt.subplots(J, 1, figsize=(10, 4*J))
    initial_conditions = np.linspace(8, 0, J)
    X = np.empty((k, n))

    for j in range(J):

        axes[j].set_ylim(-4, 8)
        axes[j].set_title(f'time series from t = {initial_conditions[j]}')

        Z = np.random.randn(k, n)
        X[:, 0] = initial_conditions[j]
        for t in range(1, n):
            X[:, t] = θ * np.abs(X[:, t-1]) + d * Z[:, t]
        axes[j].boxplot(X)

    plt.show()





Appendix
========

.. _statd_appendix:

Here's the proof of :eq:`statd_dv`.

Let :math:`F_U` and :math:`F_V` be the cumulative distributions of :math:`U` and :math:`V` respectively.

By the definition of :math:`V`, we have :math:`F_V(v) = \mathbb P \{ a + b U \leq v \} = \mathbb P \{ U \leq (v - a) / b \}`.

In other words, :math:`F_V(v) = F_U ( (v - a)/b )`.

Differentiating with respect to :math:`v` yields :eq:`statd_dv`.
