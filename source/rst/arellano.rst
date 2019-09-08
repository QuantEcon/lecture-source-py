.. _arellano:

.. include:: /_static/includes/header.raw

.. highlight:: python3

***************************************
Default Risk and Income Fluctuations
***************************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
============

This lecture computes versions of  Arellano's  :cite:`arellano2008default` model of sovereign default.

The model describes interactions among default risk, output,  and  an equilibrium interest rate  that includes a premium for endogenous default risk.

The decision maker is a government of a small open economy that borrows from risk-neutral foreign creditors.

The foreign lenders must be compensated for default risk.

The government borrows and lends abroad in order to  smooth the consumption of its citizens.

The government repays its debt only if it wants to, but declining to pay has adverse consequences.

The interest rate on government debt adjusts in response to the state-dependent default probability chosen by government.

The model yields outcomes that help interpret sovereign default experiences, including

* countercyclical interest rates on sovereign debt

* countercyclical trade balances

* high volatility of consumption relative to output

Notably, long recessions caused by bad draws in the income process increase the government's incentive to default.

This can lead to

* spikes in interest rates

* temporary losses of access to international credit markets

* large drops in output, consumption, and welfare

* large capital outflows during recessions

Such dynamics are consistent with experiences of many countries.


Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline
    from numba import jit
    import random

Structure
===============

In this section we describe the main features of the model.

Output, Consumption and Debt
-----------------------------


A small open economy is endowed with an exogenous stochastically fluctuating potential output stream :math:`\{y_t\}`.

Potential output is realized only in periods in which the government honors its sovereign debt.

The output good can be traded or consumed.

The sequence :math:`\{y_t\}` is described by a Markov process with stochastic density kernel :math:`p(y, y')`.

Households within the country are identical and rank stochastic consumption streams according to

.. math::
    :label: utility

    \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)


Here

* :math:`0 < \beta < 1` is a time discount factor

* :math:`u` is an increasing and strictly concave utility function

Consumption sequences enjoyed by households are affected by the government's decision to borrow or lend internationally.

The government is benevolent in the sense that its aim is to maximize :eq:`utility`.

The government is the only domestic actor with access to foreign credit.

Because household are averse to consumption fluctuations, the government will try to smooth consumption by borrowing from (and lending to) foreign creditors.





Asset Markets
---------------

The only credit instrument available to the government is a one-period bond traded in international credit markets.

The bond market has the following features

* The bond matures in one period and is not state contingent.

* A purchase of a bond with face value :math:`B'` is a claim to :math:`B'` units of the
  consumption good next period.

* To purchase :math:`B'`  next period costs :math:`q B'` now, or, what is equivalent.

* For selling :math:`-B'` units of next period goods the seller earns :math:`- q B'` of today's goods.


    * If :math:`B' < 0`, then :math:`-q B'` units of the good are received in the current period, for a promise to repay :math:`-B'` units next period.

    * There is an equilibrium  price function :math:`q(B', y)` that makes :math:`q` depend on both :math:`B'` and :math:`y`.


Earnings on the government portfolio are distributed (or, if negative, taxed) lump sum to households.


When the government is not excluded from financial markets, the one-period national budget constraint is

.. math::
    :label: resource

    c = y + B - q(B', y) B'


Here and below, a prime denotes a next period value or a claim maturing next period.

To rule out Ponzi schemes, we also require that :math:`B \geq -Z` in every period.

* :math:`Z` is chosen to be sufficiently large that the constraint never binds in equilibrium.


Financial Markets
-------------------

Foreign creditors

* are risk neutral

* know the domestic output stochastic process :math:`\{y_t\}` and observe :math:`y_t, y_{t-1}, \ldots,` at time :math:`t`

* can borrow or lend without limit in an international credit market at a constant international interest rate :math:`r`

* receive full payment if the government chooses to pay

* receive zero if the government defaults on its one-period debt due

When a government is expected to default next period with  probability :math:`\delta`,  the expected value of a promise to pay one unit of consumption next period is :math:`1 - \delta`.

Therefore, the discounted expected value of a promise to pay :math:`B` next period is

.. math::
    :label: epc

    q = \frac{1 - \delta}{1 + r}


Next we turn to how the government in effect chooses the default probability :math:`\delta`.


Government's Decisions
-----------------------


At each point in time :math:`t`, the government chooses between

#. defaulting

#. meeting its current obligations and purchasing or selling an optimal quantity of  one-period sovereign debt

Defaulting means declining to repay all of  its current obligations.

If the government defaults in the current period, then consumption equals current output.

But a sovereign default has two consequences:

#. Output immediately falls from :math:`y` to :math:`h(y)`, where :math:`0 \leq h(y) \leq y`.

     * It returns to :math:`y` only after the country regains access to  international credit markets.

#. The country loses access to foreign credit markets.

Reentering International Credit Market
---------------------------------------

While in a state of default, the economy regains access to
foreign credit in each subsequent period with probability
:math:`\theta`.



Equilibrium
===============


Informally, an equilibrium is a sequence of interest rates on its sovereign debt, a stochastic sequence of government default decisions  and an implied flow of household consumption such that

#. Consumption and assets satisfy the national budget constraint.

#. The government maximizes household utility taking into account

   * the resource constraint

   * the effect of its choices on the price of bonds

   * consequences of defaulting now for future net output and future borrowing and lending opportunities

#. The interest rate on the government's debt includes a risk-premium sufficient to make foreign creditors expect on average to earn the constant risk-free international interest rate.


To express these ideas more precisely, consider first the choices of the
government, which

#. enters a period with initial assets :math:`B`, or  what is the same thing, initial debt to be repaid now of :math:`-B`

#. observes current output :math:`y`, and

#. chooses either

    #. to default, or

    #. to pay  :math:`-B` and set next period's debt due to :math:`-B'`

In a  recursive formulation,

* state variables for the government comprise the pair :math:`(B, y)`

* :math:`v(B, y)` is the optimum value of the government's problem when at the beginning of a period  it faces the choice of whether to honor or default

* :math:`v_c(B, y)` is the value of choosing to pay obligations falling due

* :math:`v_d(y)` is the value of choosing to default

:math:`v_d(y)` does not depend on :math:`B` because, when access to credit is eventually regained, net foreign assets equal :math:`0`.


Expressed recursively, the value of defaulting is

.. math::

    v_d(y) = u(h(y)) +
                \beta \int \left\{
                \theta v(0, y') + (1 - \theta) v_d(y')
                \right\}
                p(y, y') dy'


The value of paying is

.. math::

    v_c(B, y) = \max_{B' \geq -Z}
           \left\{
                u(y - q(B', y) B' + B) +
                \beta \int v(B', y') p(y, y') dy'
          \right\}


The three value functions are linked by

.. math::

    v(B, y) = \max\{ v_c(B, y), v_d(y) \}


The government chooses to default when

.. math::

    v_c(B, y) < v_d(y)


and hence given :math:`B'` the probability of default next period  is

.. math::
    :label: delta

    \delta(B', y) := \int \mathbb 1\{v_c(B', y') < v_d(y') \} p(y, y') dy'


Given zero profits for foreign creditors in equilibrium, we can combine :eq:`epc` and :eq:`delta` to pin down
the bond price function:

.. math::
    :label: bondprice

    q(B', y) = \frac{1 - \delta(B', y)}{1 + r}


Definition of Equilibrium
---------------------------

An *equilibrium* is

* a  pricing function :math:`q(B',y)`,

* a triple of value functions :math:`(v_c(B, y), v_d(y), v(B,y))`,

* a decision rule telling the government when to default and when to pay as a function of the state  :math:`(B, y)`, and

* an asset accumulation rule that, conditional on choosing not to  default,  maps :math:`(B,y)` into :math:`B'`

such that

* The three Bellman equations for :math:`(v_c(B, y), v_d(y), v(B,y))` are satisfied

* Given the price function :math:`q(B',y)`, the default decision rule and the asset accumulation decision rule attain the optimal value function  :math:`v(B,y)`, and

* The price function :math:`q(B',y)` satisfies equation :eq:`bondprice`




Computation
===============

Let's now compute an equilibrium of Arellano's model.

The equilibrium objects are the value function :math:`v(B, y)`, the associated default decision rule, and the pricing function :math:`q(B', y)`.

We'll use our code to replicate Arellano's results.

After that we'll perform some additional simulations.

The majority of the code below was written by `Chase Coleman <https://github.com/cc7768>`_.

It uses a slightly modified version of the algorithm recommended by Arellano.

* The appendix to :cite:`arellano2008default` recommends
  value function iteration until convergence, updating the price, and then repeating.

* Instead, we update the bond price at every value function iteration step.

The second approach is faster and the two different procedures deliver very similar results.


Here is a more detailed description of our algorithm:

1. Guess a value function :math:`v(B, y)` and price function :math:`q(B', y)`.

2. At each pair :math:`(B, y)`,

   * update the value of defaulting :math:`v_d(y)`.

   * update the value of continuing :math:`v_c(B, y)`.

3.  Update the value function :math:`v(B, y)`, the default rule,  the implied ex ante default probability,  and the price function.

4. Check for convergence. If converged, stop -- if not, go to step 2.

We use simple discretization on a grid of asset holdings and income levels.

.. note that the next line will need to be changed when we add a Julia version.

The output process is discretized using `Tauchen's quadrature method <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/approximation.py>`_.


:ref:`Numba <numba_link>` has been used in two places to speed up the code.


.. code-block:: python3

    """

    Authors: Chase Coleman, John Stachurski

    """

    class Arellano_Economy:
        """
        Arellano 2008 deals with a small open economy whose government
        invests in foreign assets in order to smooth the consumption of
        domestic households. Domestic households receive a stochastic
        path of income.

        Parameters
        ----------
        β : float
            Time discounting parameter
        γ : float
            Risk-aversion parameter
        r : float
            int lending rate
        ρ : float
            Persistence in the income process
        η : float
            Standard deviation of the income process
        θ : float
            Probability of re-entering financial markets in each period
        ny : int
            Number of points in y grid
        nB : int
            Number of points in B grid
        tol : float
            Error tolerance in iteration
        maxit : int
            Maximum number of iterations
        """

        def __init__(self,
                    β=.953,         # time discount rate
                    γ=2.,           # risk aversion
                    r=0.017,        # international interest rate
                    ρ=.945,         # persistence in output
                    η=0.025,        # st dev of output shock
                    θ=0.282,        # prob of regaining access
                    ny=21,          # number of points in y grid
                    nB=251,         # number of points in B grid
                    tol=1e-8,       # error tolerance in iteration
                    maxit=10000):

            # Save parameters
            self.β, self.γ, self.r = β, γ, r
            self.ρ, self.η, self.θ = ρ, η, θ
            self.ny, self.nB = ny, nB

            # Create grids and discretize Markov process
            self.Bgrid = np.linspace(-.45, .45, nB)
            self.mc = qe.markov.tauchen(ρ, η, 0, 3, ny)
            self.ygrid = np.exp(self.mc.state_values)
            self.Py = self.mc.P

            # Output when in default
            ymean = np.mean(self.ygrid)
            self.def_y = np.minimum(0.969 * ymean, self.ygrid)

            # Allocate memory
            self.Vd = np.zeros(ny)
            self.Vc = np.zeros((ny, nB))
            self.V = np.zeros((ny, nB))
            self.Q = np.ones((ny, nB)) * .95  # Initial guess for prices
            self.default_prob = np.empty((ny, nB))

            # Compute the value functions, prices, and default prob
            self.solve(tol=tol, maxit=maxit)
            # Compute the optimal savings policy conditional on no default
            self.compute_savings_policy()

        def solve(self, tol=1e-8, maxit=10000):
            # Iteration Stuff
            it = 0
            dist = 10.

            # Alloc memory to store next iterate of value function
            V_upd = np.zeros((self.ny, self.nB))

            # Main loop
            while dist > tol and maxit > it:

                # Compute expectations for this iteration
                Vs = self.V, self.Vd, self.Vc
                EV, EVd, EVc = (self.Py @ v for v in Vs)

                # Run inner loop to update value functions Vc and Vd.
                # Note that Vc and Vd are updated in place.  Other objects
                # are not modified.
                _inner_loop(self.ygrid, self.def_y,
                            self.Bgrid, self.Vd, self.Vc,
                            EVc, EVd, EV, self.Q,
                            self.β, self.θ, self.γ)

                # Update prices
                Vd_compat = np.repeat(self.Vd, self.nB).reshape(self.ny, self.nB)
                default_states = Vd_compat > self.Vc
                self.default_prob[:, :] = self.Py @ default_states
                self.Q[:, :] = (1 - self.default_prob)/(1 + self.r)

                # Update main value function and distance
                V_upd[:, :] = np.maximum(self.Vc, Vd_compat)
                dist = np.max(np.abs(V_upd - self.V))
                self.V[:, :] = V_upd[:, :]

                it += 1
                if it % 25 == 0:
                    print(f"Running iteration {it} with dist of {dist}")

            return None

        def compute_savings_policy(self):
            """
            Compute optimal savings B' conditional on not defaulting.
            The policy is recorded as an index value in Bgrid.
            """

            # Allocate memory
            self.next_B_index = np.empty((self.ny, self.nB))
            EV = self.Py @ self.V

            _compute_savings_policy(self.ygrid, self.Bgrid, self.Q, EV,
                                    self.γ, self.β, self.next_B_index)

        def simulate(self, T, y_init=None, B_init=None):
            """
            Simulate time series for output, consumption, B'.
            """
            # Find index i such that Bgrid[i] is near 0
            zero_B_index = np.searchsorted(self.Bgrid, 0)

            if y_init is None:
                # Set to index near the mean of the ygrid
                y_init = np.searchsorted(self.ygrid, self.ygrid.mean())
            if B_init is None:
                B_init = zero_B_index
            # Start off not in default
            in_default = False

            y_sim_indices = self.mc.simulate_indices(T, init=y_init)
            B_sim_indices = np.empty(T, dtype=np.int64)
            B_sim_indices[0] = B_init
            q_sim = np.empty(T)
            in_default_series = np.zeros(T, dtype=np.int64)

            for t in range(T-1):
                yi, Bi = y_sim_indices[t], B_sim_indices[t]
                if not in_default:
                    if self.Vc[yi, Bi] < self.Vd[yi]:
                        in_default = True
                        Bi_next = zero_B_index
                    else:
                        new_index = self.next_B_index[yi, Bi]
                        Bi_next = new_index
                else:
                    in_default_series[t] = 1
                    Bi_next = zero_B_index
                    if random.uniform(0, 1) < self.θ:
                        in_default = False
                B_sim_indices[t+1] = Bi_next
                q_sim[t] = self.Q[yi, int(Bi_next)]

            q_sim[-1] = q_sim[-2]  # Extrapolate for the last price
            return_vecs = (self.ygrid[y_sim_indices],
                        self.Bgrid[B_sim_indices],
                        q_sim,
                        in_default_series)

            return return_vecs


    @jit(nopython=True)
    def u(c, γ):
        return c**(1-γ)/(1-γ)


    @jit(nopython=True)
    def _inner_loop(ygrid, def_y, Bgrid, Vd, Vc, EVc,
                    EVd, EV, qq, β, θ, γ):
        """
        This is a numba version of the inner loop of the solve in the
        Arellano class. It updates Vd and Vc in place.
        """
        ny, nB = len(ygrid), len(Bgrid)
        zero_ind = nB // 2  # Integer division
        for iy in range(ny):
            y = ygrid[iy]   # Pull out current y

            # Compute Vd
            Vd[iy] = u(def_y[iy], γ) + \
                    β * (θ * EVc[iy, zero_ind] + (1 - θ) * EVd[iy])

            # Compute Vc
            for ib in range(nB):
                B = Bgrid[ib]  # Pull out current B

                current_max = -1e14
                for ib_next in range(nB):
                    c = max(y - qq[iy, ib_next] * Bgrid[ib_next] + B, 1e-14)
                    m = u(c, γ) + β * EV[iy, ib_next]
                    if m > current_max:
                        current_max = m
                Vc[iy, ib] = current_max

        return None


    @jit(nopython=True)
    def _compute_savings_policy(ygrid, Bgrid, Q, EV, γ, β, next_B_index):
        # Compute best index in Bgrid given iy, ib
        ny, nB = len(ygrid), len(Bgrid)
        for iy in range(ny):
            y = ygrid[iy]
            for ib in range(nB):
                B = Bgrid[ib]
                current_max = -1e10
                for ib_next in range(nB):
                    c = max(y - Q[iy, ib_next] * Bgrid[ib_next] + B, 1e-14)
                    m = u(c, γ) + β * EV[iy, ib_next]
                    if m > current_max:
                        current_max = m
                        current_max_index = ib_next
                next_B_index[iy, ib] = current_max_index
        return None



Results
===============

Let's start by trying to replicate the results obtained in :cite:`arellano2008default`.

In what follows, all results are computed using Arellano's parameter values.

The values can be seen in the ``__init__`` method of the ``Arellano_Economy`` shown above.

* For example, ``r=0.017`` matches the average quarterly rate on a 5 year US treasury
  over the period 1983--2001.

Details on how to compute the figures are reported as solutions to the exercises.


The first figure shows the bond price schedule and replicates Figure 3 of Arellano, where :math:`y_L` and :math:`Y_H` are particular below average and above average values of output :math:`y`.

.. figure:: /_static/lecture_specific/arellano/arellano_bond_prices.png

* :math:`y_L` is 5% below the mean of the :math:`y` grid values

* :math:`y_H` is 5% above  the mean of the :math:`y` grid values

The grid used to compute this figure was relatively coarse (``ny, nB = 21, 251``) in order to match Arrelano's findings.

Here's the same relationships computed on a finer grid (``ny, nB = 51, 551``)

.. figure:: /_static/lecture_specific/arellano/arellano_bond_prices_2.png

In either case, the figure shows that

* Higher levels of debt (larger :math:`-B'`) induce larger discounts on the face value, which correspond to  higher interest rates.

* Lower income also causes more discounting, as foreign creditors anticipate greater likelihood of default.


The next figure plots value functions and replicates the right hand panel of Figure 4 of :cite:`arellano2008default`.

.. figure:: /_static/lecture_specific/arellano/arellano_value_funcs.png


We can use the results of the computation to study the default probability
:math:`\delta(B', y)` defined in :eq:`delta`.

The next plot shows these default probabilities over :math:`(B', y)` as a heat
map.

.. figure:: /_static/lecture_specific/arellano/arellano_default_probs.png

As anticipated, the probability that the government chooses to default in the
following period increases with indebtedness and falls with income.

Next let's run a time series simulation of :math:`\{y_t\}`, :math:`\{B_t\}` and :math:`q(B_{t+1}, y_t)`.

The grey vertical bars correspond to periods when the economy is excluded from financial markets because of a past default.

.. figure:: /_static/lecture_specific/arellano/arellano_time_series.png

One notable feature of the simulated data is the nonlinear response of interest rates.

Periods of relative stability are followed by sharp spikes in the discount rate on government debt.


Exercises
==============

.. _arellano_ex1:

Exercise 1
------------

To the extent that you can, replicate the figures shown above

* Use the parameter values listed as defaults in the ``__init__`` method of the ``Arellano_Economy``.

* The time series will of course vary depending on the shock draws.


Solutions
==========



Compute the value function, policy and equilibrium prices

.. code-block:: python3

    ae = Arellano_Economy(β=.953,        # Time discount factor
                          γ=2.,          # Risk aversion
                          r=0.017,       # International interest rate
                          ρ=.945,        # Persistence in output
                          η=0.025,       # st dev of output shock
                          θ=0.282,       # Prob of regaining access
                          ny=21,         # Number of points in y grid
                          nB=251,        # Number of points in B grid
                          tol=1e-8,      # Error tolerance in iteration
                          maxit=10000)



Compute the bond price schedule as seen in figure 3 of Arellano (2008)

.. code-block:: python3


    # Create "Y High" and "Y Low" values as 5% devs from mean
    high, low = np.mean(ae.ygrid) * 1.05, np.mean(ae.ygrid) * .95
    iy_high, iy_low = (np.searchsorted(ae.ygrid, x) for x in (high, low))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_title("Bond price schedule $q(y, B')$")

    # Extract a suitable plot grid
    x = []
    q_low = []
    q_high = []
    for i in range(ae.nB):
        b = ae.Bgrid[i]
        if -0.35 <= b <= 0:  # To match fig 3 of Arellano
            x.append(b)
            q_low.append(ae.Q[iy_low, i])
            q_high.append(ae.Q[iy_high, i])
    ax.plot(x, q_high, label="$y_H$", lw=2, alpha=0.7)
    ax.plot(x, q_low, label="$y_L$", lw=2, alpha=0.7)
    ax.set_xlabel("$B'$")
    ax.legend(loc='upper left', frameon=False)
    plt.show()


Draw a plot of the value functions

.. code-block:: python3


    # Create "Y High" and "Y Low" values as 5% devs from mean
    high, low = np.mean(ae.ygrid) * 1.05, np.mean(ae.ygrid) * .95
    iy_high, iy_low = (np.searchsorted(ae.ygrid, x) for x in (high, low))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_title("Value Functions")
    ax.plot(ae.Bgrid, ae.V[iy_high], label="$y_H$", lw=2, alpha=0.7)
    ax.plot(ae.Bgrid, ae.V[iy_low], label="$y_L$", lw=2, alpha=0.7)
    ax.legend(loc='upper left')
    ax.set(xlabel="$B$", ylabel="$V(y, B)$")
    ax.set_xlim(ae.Bgrid.min(), ae.Bgrid.max())
    plt.show()



Draw a heat map for default probability

.. code-block:: python3


    xx, yy = ae.Bgrid, ae.ygrid
    zz = ae.default_prob

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6.5))
    hm = ax.pcolormesh(xx, yy, zz)
    cax = fig.add_axes([.92, .1, .02, .8])
    fig.colorbar(hm, cax=cax)
    ax.axis([xx.min(), 0.05, yy.min(), yy.max()])
    ax.set(xlabel="$B'$", ylabel="$y$", title="Probability of Default")
    plt.show()


Plot a time series of major variables simulated from the model

.. code-block:: python3

    T = 250
    y_vec, B_vec, q_vec, default_vec = ae.simulate(T)

    # Pick up default start and end dates
    start_end_pairs = []
    i = 0
    while i < len(default_vec):
        if default_vec[i] == 0:
            i += 1
        else:
            # If we get to here we're in default
            start_default = i
            while i < len(default_vec) and default_vec[i] == 1:
                i += 1
            end_default = i - 1
            start_end_pairs.append((start_default, end_default))

    plot_series = y_vec, B_vec, q_vec
    titles = 'output', 'foreign assets', 'bond price'

    fig, axes = plt.subplots(len(plot_series), 1, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.3)

    for ax, series, title in zip(axes, plot_series, titles):
        # Determine suitable y limits
        s_max, s_min = max(series), min(series)
        s_range = s_max - s_min
        y_max = s_max + s_range * 0.1
        y_min = s_min - s_range * 0.1
        ax.set_ylim(y_min, y_max)
        for pair in start_end_pairs:
            ax.fill_between(pair, (y_min, y_min), (y_max, y_max),
                            color='k', alpha=0.3)
        ax.grid()
        ax.plot(range(T), series, lw=2, alpha=0.7)
        ax.set(title=title, xlabel="time")

    plt.show()
