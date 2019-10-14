.. _lake_model:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*******************************************
A Lake Model of Employment and Unemployment
*******************************************

.. index::
    single: Lake Model

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========

This lecture describes what has come to be called a *lake model*.

The lake model is a basic tool for modeling unemployment.

It allows us to analyze

* flows between unemployment and employment.

* how these flows influence steady state employment and unemployment rates.

It is a good model for interpreting monthly labor department reports on gross and net jobs created and jobs destroyed.

The "lakes" in the model are the pools of employed and unemployed.

The "flows" between the lakes are caused by

* firing and hiring

* entry and exit from the labor force

For the first part of this lecture, the parameters governing transitions into
and out of unemployment and employment are exogenous.

Later, we'll determine some of these transition rates endogenously using the :doc:`McCall search model <mccall_model>`.

We'll also use some nifty concepts like ergodicity, which provides a fundamental link between *cross-sectional* and *long run time series* distributions.

These concepts will help us build an equilibrium model of ex-ante homogeneous workers whose different luck generates variations in their ex post experiences.

Let's start with some imports:

.. code-block:: ipython

    import numpy as np
    from quantecon import MarkovChain
    import matplotlib.pyplot as plt
    %matplotlib inline
    from scipy.stats import norm
    from scipy.optimize import brentq
    from quantecon.distributions import BetaBinomial
    from numba import jit

Prerequisites
-------------

Before working through what follows, we recommend you read the :doc:`lecture
on finite Markov chains <finite_markov>`.

You will also need some basic :doc:`linear algebra <linear_algebra>` and probability.




The Model
=========

The economy is inhabited by a very large number of ex-ante identical workers.

The workers live forever, spending their lives moving between unemployment and employment.

Their rates of  transition between employment and unemployment are  governed by the following parameters:

* :math:`\lambda`, the job finding rate for currently unemployed workers

* :math:`\alpha`, the dismissal rate for currently employed workers

* :math:`b`, the entry rate into the labor force

* :math:`d`, the exit rate from the labor force

The growth rate of the labor force evidently equals :math:`g=b-d`.


Aggregate Variables
-------------------

We want to derive the dynamics of the following aggregates

* :math:`E_t`, the total number of employed workers at date :math:`t`

* :math:`U_t`, the total number of unemployed workers at :math:`t`

* :math:`N_t`, the number of workers in the labor force at :math:`t`

We also want to know the values of the following objects

* The employment rate :math:`e_t := E_t/N_t`.

* The unemployment rate :math:`u_t := U_t/N_t`.


(Here and below, capital letters represent stocks and lowercase letters represent flows)



Laws of Motion for Stock Variables
----------------------------------

We begin by constructing laws of motion for the aggregate variables :math:`E_t,U_t, N_t`.

Of the mass of workers :math:`E_t` who are employed at date :math:`t`,

* :math:`(1-d)E_t` will remain in the labor force

* of these, :math:`(1-\alpha)(1-d)E_t` will remain employed

Of the mass of workers :math:`U_t` workers who are currently unemployed,

* :math:`(1-d)U_t` will remain in the labor force

* of these, :math:`(1-d) \lambda U_t` will become employed

Therefore,  the number of workers who will be employed at date :math:`t+1` will be

.. math::

    E_{t+1} = (1-d)(1-\alpha)E_t + (1-d)\lambda U_t


A similar analysis implies

.. math::

    U_{t+1} = (1-d)\alpha E_t + (1-d)(1-\lambda)U_t + b (E_t+U_t)


The value :math:`b(E_t+U_t)` is the mass of new workers entering the labor force unemployed.

The total stock of workers :math:`N_t=E_t+U_t` evolves as

.. math::

    N_{t+1} = (1+b-d)N_t = (1+g)N_t


Letting :math:`X_t := \left(\begin{matrix}U_t\\E_t\end{matrix}\right)`, the law of motion for :math:`X`  is

.. math::

    X_{t+1} = A X_t
    \quad \text{where} \quad
    A :=
    \begin{pmatrix}
        (1-d)(1-\lambda) + b & (1-d)\alpha + b  \\
        (1-d)\lambda & (1-d)(1-\alpha)
    \end{pmatrix}


This law tells us how total employment and unemployment evolve over time.


Laws of Motion for Rates
------------------------

Now let's derive the law of motion for rates.

To get these we can divide both sides of :math:`X_{t+1} = A X_t` by  :math:`N_{t+1}` to get

.. math::

    \begin{pmatrix}
        U_{t+1}/N_{t+1} \\
        E_{t+1}/N_{t+1}
    \end{pmatrix} =
    \frac1{1+g} A
    \begin{pmatrix}
        U_{t}/N_{t}
        \\
        E_{t}/N_{t}
    \end{pmatrix}


Letting

.. math::

    x_t :=
    \left(\begin{matrix}
        u_t\\ e_t
    \end{matrix}\right) =
    \left(\begin{matrix}
        U_t/N_t\\ E_t/N_t
    \end{matrix}\right)


we can also write this as

.. math::

    x_{t+1} = \hat A x_t
    \quad \text{where} \quad
    \hat A := \frac{1}{1 + g} A


You can check that :math:`e_t + u_t = 1` implies that :math:`e_{t+1}+u_{t+1} = 1`.

This follows from the fact that the columns of :math:`\hat A` sum to 1.


Implementation
==============


Let's code up these equations.



To do this we're going to use a class that we'll call `LakeModel`.

This class will

#. store the primitives :math:`\alpha, \lambda, b, d`

#. compute and store the implied objects :math:`g, A, \hat A`

#. provide methods to simulate dynamics of the stocks and rates

#. provide a method to compute the state state of the rate


To write a nice implementation, there's an issue we have to address.

Derived data such as :math:`A` depend on the primitives like :math:`\alpha`
and :math:`\lambda`.

If a user alters these primitives, we would ideally like derived data to
update automatically.

(For example, if a user changes the value of :math:`b` for a given instance of the class, we would like :math:`g = b - d` to update automatically)

To achieve this outcome, we're going to use descriptors and decorators such as `@property`.

If you need to refresh your understanding of how these work, consult :doc:`this lecture <python_advanced_features>`.



Here's the code:


.. code-block:: python3

    class LakeModel:
        """
        Solves the lake model and computes dynamics of unemployment stocks and
        rates.

        Parameters:
        ------------
        λ : scalar
            The job finding rate for currently unemployed workers
        α : scalar
            The dismissal rate for currently employed workers
        b : scalar
            Entry rate into the labor force
        d : scalar
            Exit rate from the labor force

        """
        def __init__(self, λ=0.283, α=0.013, b=0.0124, d=0.00822):
            self._λ, self._α, self._b, self._d = λ, α, b, d
            self.compute_derived_values()

        def compute_derived_values(self):
            # Unpack names to simplify expression
            λ, α, b, d = self._λ, self._α, self._b, self._d

            self._g = b - d
            self._A = np.array([[(1-d) * (1-λ) + b,      (1 - d) * α + b],
                                [        (1-d) * λ,   (1 - d) * (1 - α)]])

            self._A_hat = self._A / (1 + self._g)

        @property
        def g(self):
            return self._g

        @property
        def A(self):
            return self._A

        @property
        def A_hat(self):
            return self._A_hat

        @property
        def λ(self):
            return self._λ

        @λ.setter
        def λ(self, new_value):
            self._α = new_value
            self.compute_derived_values()

        @property
        def α(self):
            return self._α

        @α.setter
        def α(self, new_value):
            self._α = new_value
            self.compute_derived_values()

        @property
        def b(self):
            return self._b

        @b.setter
        def b(self, new_value):
            self._b = new_value
            self.compute_derived_values()

        @property
        def d(self):
            return self._d

        @d.setter
        def d(self, new_value):
            self._d = new_value
            self.compute_derived_values()


        def rate_steady_state(self, tol=1e-6):
            """
            Finds the steady state of the system :math:`x_{t+1} = \hat A x_{t}`

            Returns
            --------
            xbar : steady state vector of employment and unemployment rates
            """
            x = 0.5 * np.ones(2)
            error = tol + 1
            while error > tol:
                new_x = self.A_hat @ x
                error = np.max(np.abs(new_x - x))
                x = new_x
            return x

        def simulate_stock_path(self, X0, T):
            """
            Simulates the sequence of Employment and Unemployment stocks

            Parameters
            ------------
            X0 : array
                Contains initial values (E0, U0)
            T : int
                Number of periods to simulate

            Returns
            ---------
            X : iterator
                Contains sequence of employment and unemployment stocks
            """

            X = np.atleast_1d(X0)  # Recast as array just in case
            for t in range(T):
                yield X
                X = self.A @ X

        def simulate_rate_path(self, x0, T):
            """
            Simulates the sequence of employment and unemployment rates

            Parameters
            ------------
            x0 : array
                Contains initial values (e0,u0)
            T : int
                Number of periods to simulate

            Returns
            ---------
            x : iterator
                Contains sequence of employment and unemployment rates

            """
            x = np.atleast_1d(x0)  # Recast as array just in case
            for t in range(T):
                yield x
                x = self.A_hat @ x


As desired, if we create an instance and update a primitive like
:math:`\alpha`, derived objects like :math:`A` will also change

.. code-block:: python3

    lm = LakeModel()
    lm.α

.. code-block:: python3

    lm.A

.. code-block:: python3

    lm.α = 2
    lm.A




Aggregate Dynamics
------------------


Let's run a simulation under the default parameters (see above) starting from :math:`X_0 = (12, 138)`



.. code-block:: python3

  lm = LakeModel()
  N_0 = 150      # Population
  e_0 = 0.92     # Initial employment rate
  u_0 = 1 - e_0  # Initial unemployment rate
  T = 50         # Simulation length

  U_0 = u_0 * N_0
  E_0 = e_0 * N_0

  fig, axes = plt.subplots(3, 1, figsize=(10, 8))
  X_0 = (U_0, E_0)
  X_path = np.vstack(tuple(lm.simulate_stock_path(X_0, T)))

  axes[0].plot(X_path[:, 0], lw=2)
  axes[0].set_title('Unemployment')

  axes[1].plot(X_path[:, 1], lw=2)
  axes[1].set_title('Employment')

  axes[2].plot(X_path.sum(1), lw=2)
  axes[2].set_title('Labor force')

  for ax in axes:
      ax.grid()

  plt.tight_layout()
  plt.show()




The aggregates :math:`E_t` and :math:`U_t` don't converge because  their sum :math:`E_t + U_t` grows at rate :math:`g`.


On the other hand, the vector of employment and unemployment rates :math:`x_t` can be in a steady state :math:`\bar x` if
there exists an :math:`\bar x`  such that

* :math:`\bar x = \hat A \bar x`

* the components satisfy :math:`\bar e + \bar u = 1`

This equation tells us that a steady state level :math:`\bar x` is an  eigenvector of :math:`\hat A` associated with a unit eigenvalue.

We also have :math:`x_t \to \bar x` as :math:`t \to \infty` provided that the remaining eigenvalue of :math:`\hat A` has modulus less that 1.

This is the case for our default parameters:



.. code-block:: python3

    lm = LakeModel()
    e, f = np.linalg.eigvals(lm.A_hat)
    abs(e), abs(f)



Let's look at the convergence of the unemployment and employment rate to steady state levels (dashed red line)


.. code-block:: python3

    lm = LakeModel()
    e_0 = 0.92     # Initial employment rate
    u_0 = 1 - e_0  # Initial unemployment rate
    T = 50         # Simulation length

    xbar = lm.rate_steady_state()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    x_0 = (u_0, e_0)
    x_path = np.vstack(tuple(lm.simulate_rate_path(x_0, T)))

    titles = ['Unemployment rate', 'Employment rate']

    for i, title in enumerate(titles):
        axes[i].plot(x_path[:, i], lw=2, alpha=0.5)
        axes[i].hlines(xbar[i], 0, T, 'r', '--')
        axes[i].set_title(title)
        axes[i].grid()

    plt.tight_layout()
    plt.show()


Dynamics of an Individual Worker
================================


An individual worker's employment dynamics are governed by a :doc:`finite state Markov process <finite_markov>`.

The worker can be in one of two states:

* :math:`s_t=0` means unemployed

* :math:`s_t=1` means employed

Let's start off under the assumption that :math:`b = d = 0`.

The associated transition matrix is then

.. math::

    P = \left(
            \begin{matrix}
                1 - \lambda & \lambda \\
                \alpha & 1 - \alpha
            \end{matrix}
        \right)


Let :math:`\psi_t` denote the :ref:`marginal distribution <mc_md>` over employment/unemployment states for the worker at time :math:`t`.

As usual, we regard it as a row vector.

We know :ref:`from an earlier discussion <mc_md>` that :math:`\psi_t` follows the law of motion

.. math::

    \psi_{t+1} = \psi_t P


We also know from the :doc:`lecture on finite Markov chains <finite_markov>`
that if :math:`\alpha \in (0, 1)` and :math:`\lambda \in (0, 1)`, then
:math:`P` has a unique stationary distribution, denoted here by :math:`\psi^*`.

The unique stationary distribution satisfies

.. math::

    \psi^*[0] = \frac{\alpha}{\alpha + \lambda}


Not surprisingly, probability mass on the unemployment state increases with
the dismissal rate and falls with the job finding rate.





Ergodicity
------------------------

Let's look at a typical lifetime of employment-unemployment spells.

We want to compute the average amounts of time an infinitely lived worker would spend employed and unemployed.


Let

.. math::

    \bar s_{u,T} := \frac1{T} \sum_{t=1}^T \mathbb 1\{s_t = 0\}


and

.. math::

    \bar s_{e,T} := \frac1{T} \sum_{t=1}^T \mathbb 1\{s_t = 1\}


(As usual, :math:`\mathbb 1\{Q\} = 1` if statement :math:`Q` is true and 0 otherwise)

These are the fraction of time a worker spends unemployed and employed, respectively, up until period :math:`T`.

If :math:`\alpha \in (0, 1)` and :math:`\lambda \in (0, 1)`, then :math:`P` is :ref:`ergodic <ergodicity>`, and hence we have

.. math::

    \lim_{T \to \infty} \bar s_{u, T} = \psi^*[0]
    \quad \text{and} \quad
    \lim_{T \to \infty} \bar s_{e, T} = \psi^*[1]


with probability one.


Inspection tells us that :math:`P` is exactly the transpose of :math:`\hat A` under the assumption :math:`b=d=0`.

Thus, the percentages of time that an  infinitely lived worker  spends employed and unemployed equal the fractions of workers employed and unemployed in the steady state distribution.


Convergence Rate
------------------

How long does it take for time series sample averages to converge to cross-sectional averages?

We can use `QuantEcon.py's <http://quantecon.org/quantecon-py>`__
`MarkovChain` class to investigate this.

Let's plot the path of the sample averages over 5,000 periods


.. code-block:: python3

    lm = LakeModel(d=0, b=0)
    T = 5000  # Simulation length

    α, λ = lm.α, lm.λ

    P = [[1 - λ,        λ],
        [    α,    1 - α]]

    mc = MarkovChain(P)

    xbar = lm.rate_steady_state()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    s_path = mc.simulate(T, init=1)
    s_bar_e = s_path.cumsum() / range(1, T+1)
    s_bar_u = 1 - s_bar_e

    to_plot = [s_bar_u, s_bar_e]
    titles = ['Percent of time unemployed', 'Percent of time employed']

    for i, plot in enumerate(to_plot):
        axes[i].plot(plot, lw=2, alpha=0.5)
        axes[i].hlines(xbar[i], 0, T, 'r', '--')
        axes[i].set_title(titles[i])
        axes[i].grid()

    plt.tight_layout()
    plt.show()


The stationary probabilities are given by the dashed red line.

In this case it takes much of the sample for these two objects to converge.

This is largely due to the high persistence in the Markov chain.




Endogenous Job Finding Rate
============================


We now make the hiring rate endogenous.

The transition rate from unemployment to employment will be determined by the McCall search model :cite:`McCall1970`.

All details relevant to the following discussion can be found in :doc:`our treatment <mccall_model>` of that model.


Reservation Wage
----------------

The most important thing to remember about the model is that optimal decisions
are characterized by a reservation wage :math:`\bar w`

*  If the wage offer :math:`w` in hand is greater than or equal to :math:`\bar w`, then the worker accepts.

*  Otherwise, the worker rejects.

As we saw in :doc:`our discussion of the model <mccall_model>`, the reservation wage depends on the wage offer distribution and the parameters

* :math:`\alpha`, the separation rate

* :math:`\beta`, the discount factor

* :math:`\gamma`, the offer arrival rate

* :math:`c`, unemployment compensation




Linking the McCall Search Model to the Lake Model
-------------------------------------------------

Suppose that  all workers inside a lake model behave according to the McCall search model.

The exogenous probability of leaving employment remains :math:`\alpha`.

But their optimal decision rules determine the probability :math:`\lambda` of leaving unemployment.

This is now

.. math::
    :label: lake_lamda

    \lambda
    = \gamma \mathbb P \{ w_t \geq \bar w\}
    = \gamma \sum_{w' \geq \bar w} p(w')


Fiscal Policy
-------------

We can use the McCall search version of the Lake Model  to find an optimal level of unemployment insurance.

We assume that  the government sets unemployment compensation :math:`c`.

The government imposes a lump-sum tax :math:`\tau` sufficient to finance total unemployment payments.

To attain a balanced budget at a steady state, taxes, the steady state unemployment rate :math:`u`, and the unemployment compensation rate must satisfy

.. math::

    \tau = u c


The lump-sum tax applies to everyone, including unemployed workers.

Thus, the post-tax income of an employed worker with wage :math:`w` is :math:`w - \tau`.

The post-tax income of an unemployed worker is :math:`c - \tau`.

For each specification :math:`(c, \tau)` of government policy, we can solve for the worker's optimal reservation wage.

This determines :math:`\lambda` via :eq:`lake_lamda` evaluated at post tax wages, which in turn determines a steady state unemployment rate :math:`u(c, \tau)`.

For a given level of unemployment benefit :math:`c`, we can solve for a tax that balances the budget in the steady state

.. math::

    \tau = u(c, \tau) c


To evaluate alternative government tax-unemployment compensation pairs, we require a welfare criterion.

We use a steady state welfare criterion

.. math::

    W := e \,  {\mathbb E} [V \, | \,  \text{employed}] + u \,  U


where the notation :math:`V` and :math:`U` is as defined in the :doc:`McCall search model lecture <mccall_model>`.

The wage offer distribution will be a discretized version of the lognormal distribution :math:`LN(\log(20),1)`, as shown in the next figure


.. figure:: /_static/lecture_specific/lake_model/lake_distribution_wages.png

We take a period to be a month.

We set :math:`b` and :math:`d` to match monthly `birth <http://www.cdc.gov/nchs/fastats/births.htm>`_ and `death rates <http://www.cdc.gov/nchs/fastats/deaths.htm>`_, respectively, in the U.S. population

* :math:`b = 0.0124`

* :math:`d = 0.00822`

Following :cite:`davis2006flow`, we set :math:`\alpha`, the hazard rate of leaving employment, to

* :math:`\alpha = 0.013`



Fiscal Policy Code
------------------


We will make use of techniques from the :doc:`McCall model lecture <mccall_model>`

The first piece of code implements value function iteration

.. code-block:: python3
    :class: collapse

    # A default utility function

    @jit
    def u(c, σ):
        if c > 0:
            return (c**(1 - σ) - 1) / (1 - σ)
        else:
            return -10e6


    class McCallModel:
        """
        Stores the parameters and functions associated with a given model.
        """

        def __init__(self, 
                    α=0.2,       # Job separation rate
                    β=0.98,      # Discount rate
                    γ=0.7,       # Job offer rate
                    c=6.0,       # Unemployment compensation
                    σ=2.0,       # Utility parameter
                    w_vec=None,  # Possible wage values
                    p_vec=None): # Probabilities over w_vec

            self.α, self.β, self.γ, self.c = α, β, γ, c
            self.σ = σ

            # Add a default wage vector and probabilities over the vector using
            # the beta-binomial distribution
            if w_vec is None:
                n = 60  # Number of possible outcomes for wage
                # Wages between 10 and 20
                self.w_vec = np.linspace(10, 20, n)
                a, b = 600, 400  # Shape parameters
                dist = BetaBinomial(n-1, a, b)
                self.p_vec = dist.pdf()  
            else:
                self.w_vec = w_vec
                self.p_vec = p_vec

    @jit
    def _update_bellman(α, β, γ, c, σ, w_vec, p_vec, V, V_new, U):
        """
        A jitted function to update the Bellman equations.  Note that V_new is
        modified in place (i.e, modified by this function).  The new value of U
        is returned.

        """
        for w_idx, w in enumerate(w_vec):
            # w_idx indexes the vector of possible wages
            V_new[w_idx] = u(w, σ) + β * ((1 - α) * V[w_idx] + α * U)

        U_new = u(c, σ) + β * (1 - γ) * U + \
                        β * γ * np.sum(np.maximum(U, V) * p_vec)

        return U_new


    def solve_mccall_model(mcm, tol=1e-5, max_iter=2000):
        """
        Iterates to convergence on the Bellman equations 
        
        Parameters
        ----------
        mcm : an instance of McCallModel
        tol : float
            error tolerance
        max_iter : int
            the maximum number of iterations
        """

        V = np.ones(len(mcm.w_vec))  # Initial guess of V
        V_new = np.empty_like(V)     # To store updates to V
        U = 1                        # Initial guess of U
        i = 0
        error = tol + 1

        while error > tol and i < max_iter:
            U_new = _update_bellman(mcm.α, mcm.β, mcm.γ, 
                    mcm.c, mcm.σ, mcm.w_vec, mcm.p_vec, V, V_new, U)
            error_1 = np.max(np.abs(V_new - V))
            error_2 = np.abs(U_new - U)
            error = max(error_1, error_2)
            V[:] = V_new
            U = U_new
            i += 1

        return V, U


The second piece of code is used to complete the reservation wage:


.. code-block:: python3
    :class: collapse

    def compute_reservation_wage(mcm, return_values=False):
        """
        Computes the reservation wage of an instance of the McCall model
        by finding the smallest w such that V(w) > U.

        If V(w) > U for all w, then the reservation wage w_bar is set to
        the lowest wage in mcm.w_vec.

        If v(w) < U for all w, then w_bar is set to np.inf.
        
        Parameters
        ----------
        mcm : an instance of McCallModel
        return_values : bool (optional, default=False)
            Return the value functions as well 

        Returns
        -------
        w_bar : scalar
            The reservation wage
            
        """

        V, U = solve_mccall_model(mcm)
        w_idx = np.searchsorted(V - U, 0)  

        if w_idx == len(V):
            w_bar = np.inf
        else:
            w_bar = mcm.w_vec[w_idx]

        if return_values == False:
            return w_bar
        else:
            return w_bar, V, U


Now let's compute and plot welfare, employment, unemployment, and tax revenue as a
function of the unemployment compensation rate


.. code-block:: python3

    # Some global variables that will stay constant
    α = 0.013
    α_q = (1-(1-α)**3)   # Quarterly (α is monthly)
    b = 0.0124
    d = 0.00822
    β = 0.98
    γ = 1.0
    σ = 2.0

    # The default wage distribution --- a discretized lognormal
    log_wage_mean, wage_grid_size, max_wage = 20, 200, 170
    logw_dist = norm(np.log(log_wage_mean), 1)
    w_vec = np.linspace(1e-8, max_wage, wage_grid_size + 1)
    cdf = logw_dist.cdf(np.log(w_vec))
    pdf = cdf[1:] - cdf[:-1]
    p_vec = pdf / pdf.sum()
    w_vec = (w_vec[1:] + w_vec[:-1]) / 2


    def compute_optimal_quantities(c, τ):
        """
        Compute the reservation wage, job finding rate and value functions
        of the workers given c and τ.

        """

        mcm = McCallModel(α=α_q,
                        β=β,
                        γ=γ,
                        c=c-τ,          # Post tax compensation
                        σ=σ,
                        w_vec=w_vec-τ,  # Post tax wages
                        p_vec=p_vec)

        w_bar, V, U = compute_reservation_wage(mcm, return_values=True)
        λ = γ * np.sum(p_vec[w_vec - τ > w_bar])
        return w_bar, λ, V, U

    def compute_steady_state_quantities(c, τ):
        """
        Compute the steady state unemployment rate given c and τ using optimal
        quantities from the McCall model and computing corresponding steady
        state quantities

        """
        w_bar, λ, V, U = compute_optimal_quantities(c, τ)

        # Compute steady state employment and unemployment rates
        lm = LakeModel(α=α_q, λ=λ, b=b, d=d)
        x = lm.rate_steady_state()
        u, e = x

        # Compute steady state welfare
        w = np.sum(V * p_vec * (w_vec - τ > w_bar)) / np.sum(p_vec * (w_vec -
                τ > w_bar))
        welfare = e * w + u * U

        return e, u, welfare


    def find_balanced_budget_tax(c):
        """
        Find the tax level that will induce a balanced budget.

        """
        def steady_state_budget(t):
            e, u, w = compute_steady_state_quantities(c, t)
            return t - u * c

        τ = brentq(steady_state_budget, 0.0, 0.9 * c)
        return τ


    # Levels of unemployment insurance we wish to study
    c_vec = np.linspace(5, 140, 60)

    tax_vec = []
    unempl_vec = []
    empl_vec = []
    welfare_vec = []

    for c in c_vec:
        t = find_balanced_budget_tax(c)
        e_rate, u_rate, welfare = compute_steady_state_quantities(c, t)
        tax_vec.append(t)
        unempl_vec.append(u_rate)
        empl_vec.append(e_rate)
        welfare_vec.append(welfare)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plots = [unempl_vec, empl_vec, tax_vec, welfare_vec]
    titles = ['Unemployment', 'Employment', 'Tax', 'Welfare']

    for ax, plot, title in zip(axes.flatten(), plots, titles):
        ax.plot(c_vec, plot, lw=2, alpha=0.7)
        ax.set_title(title)
        ax.grid()

    plt.tight_layout()
    plt.show()


Welfare first increases and then decreases as unemployment benefits rise.

The level that maximizes steady state welfare is approximately 62.


Exercises
=========

Exercise 1
----------

Consider an economy with an initial stock  of workers :math:`N_0 = 100` at the
steady state level of employment in the baseline parameterization

* :math:`\alpha = 0.013`

* :math:`\lambda = 0.283`

* :math:`b = 0.0124`

* :math:`d = 0.00822`

(The values for :math:`\alpha` and :math:`\lambda` follow :cite:`davis2006flow`)

Suppose that in response to new legislation the hiring rate reduces to :math:`\lambda = 0.2`.

Plot the transition dynamics of the unemployment and employment stocks for 50 periods.

Plot the transition dynamics for the rates.

How long does the economy take to converge to its new steady state?

What is the new steady state level of employment?




Exercise 2
----------

Consider an economy with an initial stock  of workers :math:`N_0 = 100` at the
steady state level of employment in the baseline parameterization.

Suppose that for 20 periods the birth rate was temporarily high (:math:`b = 0.0025`) and then returned to its original level.

Plot the transition dynamics of the unemployment and employment stocks for 50 periods.

Plot the transition dynamics for the rates.

How long does the economy take to return to its original steady state?




Solutions
=========




Lake Model Solutions
====================

Exercise 1
-----------

We begin by constructing the class containing the default parameters and assigning the
steady state values to `x0`

.. code-block:: python3

    lm = LakeModel()
    x0 = lm.rate_steady_state()
    print(f"Initial Steady State: {x0}")

Initialize the simulation values

.. code-block:: python3

    N0 = 100
    T = 50

New legislation changes :math:`\lambda` to :math:`0.2`

.. code-block:: python3

    lm.lmda = 0.2

    xbar = lm.rate_steady_state()  # new steady state
    X_path = np.vstack(tuple(lm.simulate_stock_path(x0 * N0, T)))
    x_path = np.vstack(tuple(lm.simulate_rate_path(x0, T)))
    print(f"New Steady State: {xbar}")

Now plot stocks

.. code-block:: python3

  fig, axes = plt.subplots(3, 1, figsize=[10, 9])

  axes[0].plot(X_path[:, 0])
  axes[0].set_title('Unemployment')

  axes[1].plot(X_path[:, 1])
  axes[1].set_title('Employment')

  axes[2].plot(X_path.sum(1))
  axes[2].set_title('Labor force')

  for ax in axes:
      ax.grid()

  plt.tight_layout()
  plt.show()


And how the rates evolve

.. code-block:: python3

  fig, axes = plt.subplots(2, 1, figsize=(10, 8))

  titles = ['Unemployment rate', 'Employment rate']

  for i, title in enumerate(titles):
      axes[i].plot(x_path[:, i])
      axes[i].hlines(xbar[i], 0, T, 'r', '--')
      axes[i].set_title(title)
      axes[i].grid()

  plt.tight_layout()
  plt.show()


We see that it takes 20 periods for the economy to converge to its new
steady state levels.

Exercise 2
----------

This next exercise has the economy experiencing a boom in entrances to
the labor market and then later returning to the original levels.

For 20 periods the economy has a new entry rate into the labor market.

Let's start off at the baseline parameterization and record the steady
state

.. code-block:: python3

    lm = LakeModel()
    x0 = lm.rate_steady_state()

Here are the other parameters:

.. code-block:: python3

    b_hat = 0.003
    T_hat = 20

Let's increase :math:`b` to the new value and simulate for 20 periods

.. code-block:: python3

    lm.b = b_hat
    # Simulate stocks
    X_path1 = np.vstack(tuple(lm.simulate_stock_path(x0 * N0, T_hat)))
    # Simulate rates
    x_path1 = np.vstack(tuple(lm.simulate_rate_path(x0, T_hat)))

Now we reset :math:`b` to the original value and then, using the state
after 20 periods for the new initial conditions, we simulate for the
additional 30 periods

.. code-block:: python3

    lm.b = 0.0124
    # Simulate stocks
    X_path2 = np.vstack(tuple(lm.simulate_stock_path(X_path1[-1, :2], T-T_hat+1)))
    # Simulate rates
    x_path2 = np.vstack(tuple(lm.simulate_rate_path(x_path1[-1, :2], T-T_hat+1)))

Finally, we combine these two paths and plot

.. code-block:: python3

  # note [1:] to avoid doubling period 20
  x_path = np.vstack([x_path1, x_path2[1:]])
  X_path = np.vstack([X_path1, X_path2[1:]])

  fig, axes = plt.subplots(3, 1, figsize=[10, 9])

  axes[0].plot(X_path[:, 0])
  axes[0].set_title('Unemployment')

  axes[1].plot(X_path[:, 1])
  axes[1].set_title('Employment')

  axes[2].plot(X_path.sum(1))
  axes[2].set_title('Labor force')

  for ax in axes:
      ax.grid()

  plt.tight_layout()
  plt.show()


And the rates

.. code-block:: python3

  fig, axes = plt.subplots(2, 1, figsize=[10, 6])

  titles = ['Unemployment rate', 'Employment rate']

  for i, title in enumerate(titles):
      axes[i].plot(x_path[:, i])
      axes[i].hlines(x0[i], 0, T, 'r', '--')
      axes[i].set_title(title)
      axes[i].grid()

  plt.tight_layout()
  plt.show()
