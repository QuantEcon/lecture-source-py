.. _multiplicative_functionals:

.. include:: /_static/includes/header.raw

.. highlight:: python3


**************************
Multiplicative Functionals
**************************

.. index::
    single: Models; Multiplicative functionals

.. contents:: :depth: 2


**Co-authors: Chase Coleman and Balint Szoke**

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon

Overview
========

This lecture is a sequel to the :doc:`lecture on additive functionals <additive_functionals>`.

That lecture

#. defined a special class of **additive functionals** driven by a first-order vector VAR

#. by taking the exponential of that additive functional, created an associated **multiplicative functional**

This lecture uses this special class to create and analyze two examples

* A  **log-likelihood process**, an object at the foundation of both frequentist and Bayesian approaches to statistical inference.

* A version of Robert E. Lucas's :cite:`lucas2003macroeconomic` and Thomas Tallarini's :cite:`Tall2000` approaches to measuring the benefits of moderating aggregate fluctuations.


Let's start with some imports

.. code-block:: ipython

    import numpy as np
    import scipy as sp
    import scipy.linalg as la
    import quantecon as qe
    import matplotlib.pyplot as plt
    %matplotlib inline
    from scipy.stats import norm, lognorm


A Log-Likelihood Process
========================

Consider a vector of additive functionals :math:`\{y_t\}_{t=0}^\infty`
described by

.. math::

    \begin{aligned}
        x_{t+1} & = A x_t + B z_{t+1}
        \\
        y_{t+1} - y_t & = D x_{t} + F z_{t+1},
    \end{aligned}


where :math:`A` is a stable matrix, :math:`\{z_{t+1}\}_{t=0}^\infty` is
an IID sequence of :math:`{\cal N}(0,I)` random vectors, :math:`F` is
nonsingular, and :math:`x_0` and :math:`y_0` are vectors of known
numbers.

Evidently,

.. math::

    x_{t+1} = \left(A - B F^{-1}D \right)x_t +
    B F^{-1} \left(y_{t+1} - y_t \right),


so that :math:`x_{t+1}` can be constructed from observations on
:math:`\{y_{s}\}_{s=0}^{t+1}` and :math:`x_0`.

The distribution of :math:`y_{t+1} - y_t` conditional on :math:`x_t` is normal with mean :math:`Dx_t` and nonsingular covariance matrix :math:`FF'`.

Let :math:`\theta` denote the vector of free parameters of the model.

These parameters pin down the elements of :math:`A, B, D, F`.

The **log-likelihood function** of :math:`\{y_s\}_{s=1}^t` is

.. math::

    \begin{aligned}
        \log L_{t}(\theta)  =
        & - {\frac 1 2} \sum_{j=1}^{t} (y_{j} - y_{j-1} -
             D x_{j-1})'(FF')^{-1}(y_{j} - y_{j-1} - D x_{j-1})
        \\
        & - {\frac t 2} \log \det (FF') - {\frac {k t} 2} \log( 2 \pi)
    \end{aligned}


Let's consider the case of a scalar process in which :math:`A, B, D, F` are scalars and :math:`z_{t+1}` is a scalar stochastic process.

We let :math:`\theta_o` denote the "true" values of :math:`\theta`, meaning the values that generate the data.

For the purposes of this exercise,  set :math:`\theta_o = (A, B, D, F) = (0.8, 1, 0.5, 0.2)`.

Set :math:`x_0 = y_0 = 0`.


Simulating Sample Paths
-----------------------

Let's write a program to simulate sample paths of :math:`\{ x_t, y_{t} \}_{t=0}^{\infty}`.

We'll do this by formulating the additive functional as a linear state space model and putting the `LinearStateSpace <https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lss.py>`_ class to work.

 .. literalinclude:: /_static/lecture_specific/multiplicative_functionals/amflss_scalar.py

The heavy lifting is done inside the `AMF_LSS_VAR` class.

The following code adds some simple functions that make it straightforward to generate sample paths from an instance of `AMF_LSS_VAR`



.. code-block:: python3

    def simulate_xy(amf, T):
        "Simulate individual paths."
        foo, bar = amf.lss.simulate(T)
        x = bar[0, :]
        y = bar[1, :]

        return x, y

    def simulate_paths(amf, T=150, I=5000):
        "Simulate multiple independent paths."

        # Allocate space
        storeX = np.empty((I, T))
        storeY = np.empty((I, T))

        for i in range(I):
            # Do specific simulation
            x, y = simulate_xy(amf, T)

            # Fill in our storage matrices
            storeX[i, :] = x
            storeY[i, :] = y

        return storeX, storeY

    def population_means(amf, T=150):
        # Allocate Space
        xmean = np.empty(T)
        ymean = np.empty(T)

        # Pull out moment generator
        moment_generator = amf.lss.moment_sequence()

        for tt in range (T):
            tmoms = next(moment_generator)
            ymeans = tmoms[1]
            xmean[tt] = ymeans[0]
            ymean[tt] = ymeans[1]

        return xmean, ymean



Now that we have these functions in our tool kit, let's apply them to run some
simulations.

In particular, let's use our program to generate :math:`I = 5000` sample paths of length :math:`T = 150`, labeled :math:`\{ x_{t}^i, y_{t}^i \}_{t=0}^\infty` for :math:`i = 1, ..., I`.

Then we compute averages of :math:`\frac{1}{I} \sum_i x_t^i` and :math:`\frac{1}{I} \sum_i y_t^i` across the sample paths and compare them with the population means of :math:`x_t` and :math:`y_t`.

Here goes



.. code-block:: python3

    A, B, D, F = [0.8, 1.0, 0.5, 0.2]
    amf = AMF_LSS_VAR(A, B, D, F=F)

    T = 150
    I = 5000

    # Simulate and compute sample means
    Xit, Yit = simulate_paths(amf, T, I)
    Xmean_t = np.mean(Xit, 0)
    Ymean_t = np.mean(Yit, 0)

    # Compute population means
    Xmean_pop, Ymean_pop = population_means(amf, T)

    # Plot sample means vs population means
    fig, ax = plt.subplots(2, figsize=(14, 8))

    ax[0].plot(Xmean_t, label=r'$\frac{1}{I}\sum_i x_t^i$', color="b")
    ax[0].plot(Xmean_pop, label='$\mathbb{E} x_t$', color="k")
    ax[0].set_title('$x_t$')
    ax[0].set_xlim((0, T))
    ax[0].legend(loc=0)

    ax[1].plot(Ymean_t, label=r'$\frac{1}{I}\sum_i y_t^i$', color="b")
    ax[1].plot(Ymean_pop, label='$\mathbb{E} y_t$', color="k")
    ax[1].set_title('$y_t$')
    ax[1].set_xlim((0, T))
    ax[1].legend(loc=0)

    plt.show()





Simulating Log-likelihoods
--------------------------

Our next aim is to write a program to simulate :math:`\{\log L_t \mid \theta_o\}_{t=1}^T`.

We want as inputs to this program the *same* sample paths :math:`\{x_t^i, y_t^i\}_{t=0}^T` that we  have already computed.

We now want to simulate :math:`I = 5000` paths of :math:`\{\log L_t^i  \mid \theta_o\}_{t=1}^T`.

-  For each path, we compute :math:`\log L_T^i / T`.

-  We also compute :math:`\frac{1}{I} \sum_{i=1}^I \log L_T^i / T`.

Then we to compare these objects.

Below we plot the histogram of :math:`\log L_T^i / T` for realizations :math:`i = 1, \ldots, 5000`



.. code-block:: python3

    def simulate_likelihood(amf, Xit, Yit):
        # Get size
        I, T = Xit.shape

        # Allocate space
        LLit = np.empty((I, T-1))

        for i in range(I):
            LLit[i, :] = amf.loglikelihood_path(Xit[i, :], Yit[i, :])

        return LLit

    # Get likelihood from each path x^{i}, Y^{i}
    LLit = simulate_likelihood(amf, Xit, Yit)

    LLT = 1/T * LLit[:, -1]
    LLmean_t = np.mean(LLT)

    fig, ax = plt.subplots()

    ax.hist(LLT)
    ax.vlines(LLmean_t, ymin=0, ymax=I//3, color="k", linestyle="--", alpha=0.6)
    plt.title(r"Distribution of $\frac{1}{T} \log L_{T}  \mid \theta_0$")

    plt.show()




Notice that the log-likelihood is almost always nonnegative, implying that :math:`L_t` is typically bigger than 1.

Recall that the likelihood function is a pdf (probability density function) and **not** a probability measure, so it can take values larger than 1.

In the current case, the conditional variance of :math:`\Delta y_{t+1}`, which equals  :math:`FF^T=0.04`, is so small that the maximum value of the pdf is 2 (see the figure below).

This implies that approximately :math:`75\%` of the time (a bit more than one sigma deviation),  we should expect the **increment** of the log-likelihood to be nonnegative.


Let's see this in a simulation



.. code-block:: python3

    normdist = sp.stats.norm(0, F)
    mult = 1.175
    print(f'The pdf at +/- {mult} sigma takes the value:  {normdist.pdf(mult * F)}')
    print(f'Probability of dL being larger than 1 is approx: {normdist.cdf(mult * F) - normdist.cdf(-mult * F)}')

    # Compare this to the sample analogue:
    L_increment = LLit[:, 1:] - LLit[:, :-1]
    r, c = L_increment.shape
    frac_nonegative = np.sum(L_increment >= 0) / (c * r)
    print(f'Fraction of dlogL being nonnegative in the sample is: {frac_nonegative}')




Let's also plot the conditional pdf of :math:`\Delta y_{t+1}`



.. code-block:: python3

    xgrid = np.linspace(-1, 1, 100)
    plt.plot(xgrid, normdist.pdf(xgrid))
    plt.title('Conditional pdf $f(\Delta y_{t+1} \mid x_t)$')
    print(f'The pdf at +/- one sigma takes the value: {normdist.pdf(F)}')
    plt.show()





An Alternative Parameter Vector
-------------------------------

Now consider alternative parameter vector :math:`\theta_1 = [A, B, D, F] = [0.9, 1.0, 0.55, 0.25]`.

We want to compute :math:`\{\log L_t \mid \theta_1\}_{t=1}^T`.

The :math:`x_t, y_t` inputs to this program should be exactly the **same** sample paths :math:`\{x_t^i, y_t^i\}_{t=0}^T` that we computed above.

This is because we want to generate data under the :math:`\theta_o` probability model but evaluate the likelihood under the :math:`\theta_1` model.

So our task is to use our program to simulate :math:`I = 5000` paths of :math:`\{\log L_t^i  \mid \theta_1\}_{t=1}^T`

-  For each path, compute :math:`\frac{1}{T} \log L_T^i`.

-  Then compute :math:`\frac{1}{I}\sum_{i=1}^I \frac{1}{T} \log L_T^i`.

We want to compare these objects with each other and with the analogous objects that we computed above.

Then we want to interpret outcomes.

A function that we constructed can  handle these tasks.

The only innovation is that we must create an alternative model to feed in.

We will creatively call the new model ``amf2``.

We make three graphs

* the first sets the stage by repeating an earlier graph

* the second contains two histograms of values of  log-likelihoods of the two models  over the period :math:`T`

* the third compares likelihoods under the true and alternative models


Here's the code



.. code-block:: python3

    # Create the second (wrong) alternative model
    A2, B2, D2, F2 = [0.9, 1.0, 0.55, 0.25]   #  parameters for θ_1 closer to Θ_0
    amf2 = AMF_LSS_VAR(A2, B2, D2, F=F2)

    # Get likelihood from each path x^{i}, y^{i}
    LLit2 = simulate_likelihood(amf2, Xit, Yit)

    LLT2 = 1/(T-1) * LLit2[:, -1]
    LLmean_t2 = np.mean(LLT2)

    fig, ax = plt.subplots()

    ax.hist(LLT2)
    ax.vlines(LLmean_t2, ymin=0, ymax=1400, color="k", linestyle="--", alpha=0.6)

    plt.title(r"Distribution of $\frac{1}{T} \log L_{T}  \mid \theta_1$")
    plt.show()



Let's see a histogram of the log-likelihoods under the true and the alternative model (same sample paths)



.. code-block:: python3

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.hist(LLT, bins=50, alpha=0.5, label='True', density=True)
    plt.hist(LLT2, bins=50, alpha=0.5, label='Alternative', density=True)
    plt.vlines(np.mean(LLT), 0, 10, color='k', linestyle="--", linewidth= 4)
    plt.vlines(np.mean(LLT2), 0, 10, color='k', linestyle="--", linewidth= 4)
    plt.legend()

    plt.show()




Now we'll plot the histogram of the difference in log-likelihood ratio



.. code-block:: python3

    LLT_diff = LLT - LLT2

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(LLT_diff, bins=50)
    plt.title(r"$\frac{1}{T}\left[\log (L_T^i  \mid \theta_0) - \log (L_T^i \mid \theta_1)\right]$")

    plt.show()




Interpretation
--------------

These histograms of  log-likelihood ratios illustrate  important features of **likelihood ratio tests** as tools for discriminating between statistical models.

*  The log-likelihood is higher on average under the true model -- obviously a very useful property.

*  Nevertheless, for a positive fraction of realizations, the log-likelihood is higher for the incorrect than for the true model

  * in these instances, a likelihood ratio test mistakenly selects the wrong model

* These mechanics underlie the statistical theory of **mistake probabilities** associated with model selection tests based on  likelihood ratio.

(In a subsequent lecture, we'll use some of the code prepared in this lecture to illustrate mistake probabilities)







Benefits from Reduced Aggregate Fluctuations
============================================

Now let's turn to a new example of multiplicative functionals.

This example illustrates  ideas in the literature on

*  **long-run risk** in the consumption based asset pricing literature (e.g., :cite:`bansal2004risks`, :cite:`hansen2008consumption`, :cite:`hansen2007beliefs`)

*  **benefits of eliminating aggregate fluctuations** in representative agent macro models (e.g., :cite:`Tall2000`, :cite:`lucas2003macroeconomic`)

Let :math:`c_t` be consumption at date :math:`t \geq 0`.

Suppose that :math:`\{\log c_t \}_{t=0}^\infty` is an additive functional described by

.. math::

    \log c_{t+1} - \log c_t = \nu + D \cdot x_t + F \cdot z_{t+1}


where

.. math::

    x_{t+1} = A x_t + B z_{t+1}


Here :math:`\{z_{t+1}\}_{t=0}^\infty` is an IID sequence of :math:`{\cal N}(0,I)` random vectors.

A representative household ranks consumption processes :math:`\{c_t\}_{t=0}^\infty` with a utility functional :math:`\{V_t\}_{t=0}^\infty` that satisfies

.. math::
    :label: old1mf

    \log V_t - \log c_t = U \cdot x_t + {\sf u}


where

.. math::

    U = \exp(-\delta) \left[ I - \exp(-\delta) A' \right]^{-1} D


and

.. math::

    {\sf u}
      = {\frac {\exp( -\delta)}{ 1 - \exp(-\delta)}} {\nu} + \frac{(1 - \gamma)}{2} {\frac {\exp(-\delta)}{1 - \exp(-\delta)}}
    \biggl| D' \left[ I - \exp(-\delta) A \right]^{-1}B + F \biggl|^2,


Here :math:`\gamma \geq 1` is a risk-aversion coefficient and :math:`\delta > 0` is a rate of time preference.





Consumption as a Multiplicative Process
---------------------------------------

We begin by showing that consumption is a **multiplicative functional** with representation

.. math::
    :label: old2mf

    \frac{c_t}{c_0}
    = \exp(\tilde{\nu}t )
    \left( \frac{\tilde{M}_t}{\tilde{M}_0} \right)
    \left( \frac{\tilde{e}(x_0)}{\tilde{e}(x_t)} \right)


where :math:`\left( \frac{\tilde{M}_t}{\tilde{M}_0} \right)` is a likelihood ratio process and :math:`\tilde M_0 = 1`.

At this point, as an exercise, we ask the reader please to verify the following formulas for :math:`\tilde{\nu}` and :math:`\tilde{e}(x_t)` as functions of :math:`A, B, D, F`:

.. math::

    \tilde \nu =  \nu + \frac{H \cdot H}{2}


and

.. math::

    \tilde e(x) = \exp[g(x)] = \exp \bigl[ D' (I - A)^{-1} x \bigr]


Simulating a Likelihood Ratio Process Again
-------------------------------------------

Next, we want a program to simulate the likelihood ratio process :math:`\{ \tilde{M}_t \}_{t=0}^\infty`.

In particular, we want to simulate 5000 sample paths of length :math:`T=1000` for the case in which :math:`x` is a scalar and :math:`[A, B, D, F] = [0.8, 0.001, 1.0, 0.01]` and :math:`\nu = 0.005`.

After accomplishing this, we want to display a histogram of :math:`\tilde{M}_T^i` for
:math:`T=1000`.

Here is code that accomplishes these tasks



.. code-block:: python3

    def simulate_martingale_components(amf, T=1000, I=5000):
        # Get the multiplicative decomposition
        ν, H, g = amf.multiplicative_decomp()

        # Allocate space
        add_mart_comp = np.empty((I, T))

        # Simulate and pull out additive martingale component
        for i in range(I):
            foo, bar = amf.lss.simulate(T)

            # Martingale component is the third component
            add_mart_comp[i, :] = bar[2, :]

        mul_mart_comp = np.exp(add_mart_comp - (np.arange(T) * H**2) / 2)

        return add_mart_comp, mul_mart_comp


    # Build model
    amf_2 = AMF_LSS_VAR(0.8, 0.001, 1.0, 0.01,.005)

    amc, mmc = simulate_martingale_components(amf_2, 1000, 5000)

    amcT = amc[:, -1]
    mmcT = mmc[:, -1]

    print("The (min, mean, max) of additive Martingale component in period T is")
    print(f"\t ({np.min(amcT)}, {np.mean(amcT)}, {np.max(amcT)})")

    print("The (min, mean, max) of multiplicative Martingale component in period T is")
    print(f"\t ({np.min(mmcT)}, {np.mean(mmcT)}, {np.max(mmcT)})")





Comments
^^^^^^^^


-  The preceding min, mean, and max of the cross-section of the date
   :math:`T` realizations of the multiplicative martingale component of
   :math:`c_t` indicate that the sample mean is close to its population
   mean of 1.

    * This outcome prevails for all values of the horizon :math:`T`.

-  The cross-section distribution of the multiplicative martingale
   component of :math:`c` at date :math:`T` approximates a log-normal
   distribution well.

-  The histogram of the additive martingale component of
   :math:`\log c_t` at date :math:`T` approximates a normal distribution
   well.

Here's a histogram of the additive martingale component



.. code-block:: python3

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(amcT, bins=25, normed=True)
    plt.title("Histogram of Additive Martingale Component")

    plt.show()




Here's a histogram of the multiplicative martingale component



.. code-block:: python3

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(mmcT, bins=25, normed=True)
    plt.title("Histogram of Multiplicative Martingale Component")
    plt.show()







Representing the Likelihood Ratio Process
-----------------------------------------

The likelihood ratio process :math:`\{\widetilde M_t\}_{t=0}^\infty` can be represented as

.. math::

    \widetilde M_t = \exp \biggl( \sum_{j=1}^t \biggl(H \cdot z_j -\frac{ H \cdot H }{2} \biggr) \biggr),  \quad \widetilde M_0 =1 ,


where :math:`H =  [F + B'(I-A')^{-1} D]`.

It follows that :math:`\log {\widetilde M}_t \sim {\mathcal N} ( -\frac{t H \cdot H}{2}, t H \cdot H )` and that consequently :math:`{\widetilde M}_t` is log-normal.

Let's plot the probability density functions for :math:`\log {\widetilde M}_t` for
:math:`t=100, 500, 1000, 10000, 100000`.


Then let's use the plots to  investigate how these densities evolve through time.

We will plot the densities of :math:`\log {\widetilde M}_t` for different values of :math:`t`.



Note: ``scipy.stats.lognorm`` expects you to pass the standard deviation
first :math:`(tH \cdot H)` and then the exponent of the mean as a
keyword argument ``scale`` (``scale=``\ :math:`\exp(-tH \cdot H/2)`).

* See the documentation `here
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm>`__.

This is peculiar, so make sure you are careful in working with the log-normal distribution.



Here is some code that tackles these tasks



.. code-block:: python3

    def Mtilde_t_density(amf, t, xmin=1e-8, xmax=5.0, npts=5000):

        # Pull out the multiplicative decomposition
        νtilde, H, g = amf.multiplicative_decomp()
        H2 = H * H

        # The distribution
        mdist = lognorm(np.sqrt(t * H2), scale=np.exp(-t * H2 / 2))
        x = np.linspace(xmin, xmax, npts)
        pdf = mdist.pdf(x)

        return x, pdf


    def logMtilde_t_density(amf, t, xmin=-15.0, xmax=15.0, npts=5000):

        # Pull out the multiplicative decomposition
        νtilde, H, g = amf.multiplicative_decomp()
        H2 = H * H

        # The distribution
        lmdist = norm(-t * H2 / 2, np.sqrt(t * H2))
        x = np.linspace(xmin, xmax, npts)
        pdf = lmdist.pdf(x)

        return x, pdf


    times_to_plot = [10, 100, 500, 1000, 2500, 5000]
    dens_to_plot = map(lambda t: Mtilde_t_density(amf_2, t, xmin=1e-8, xmax=6.0), times_to_plot)
    ldens_to_plot = map(lambda t: logMtilde_t_density(amf_2, t, xmin=-10.0, xmax=10.0), times_to_plot)

    fig, ax = plt.subplots(3, 2, figsize=(8, 14))
    ax = ax.flatten()

    fig.suptitle(r"Densities of $\tilde{M}_t$", fontsize=18, y=1.02)
    for (it, dens_t) in enumerate(dens_to_plot):
        x, pdf = dens_t
        ax[it].set_title(f"Density for time {times_to_plot[it]}")
        ax[it].fill_between(x, np.zeros_like(pdf), pdf)

    plt.tight_layout()
    plt.show()






These probability density functions illustrate a **peculiar property** of log-likelihood ratio processes:

* With respect to the true model probabilities, they have mathematical expectations equal to :math:`1` for all :math:`t \geq 0`.

* They almost surely converge to zero.




Welfare Benefits of Reduced Random Aggregate Fluctuations
---------------------------------------------------------

Suppose in the tradition of a strand of macroeconomics (for example Tallarini :cite:`Tall2000`, :cite:`lucas2003macroeconomic`) we want to estimate the welfare benefits from removing random fluctuations around trend growth.

We shall  compute how much initial consumption :math:`c_0` a representative consumer who ranks consumption streams according to :eq:`old1mf` would be willing to sacrifice to enjoy the consumption stream

.. math::

    \frac{c_t}{c_0} = \exp (\tilde{\nu} t)


rather than the stream described by equation :eq:`old2mf`.

We want to compute the implied percentage reduction in :math:`c_0` that the representative consumer would accept.

To accomplish this, we write a function that computes the coefficients :math:`U`
and :math:`u` for the original values of :math:`A, B, D, F, \nu`, but
also for the case that  :math:`A, B, D, F = [0, 0, 0, 0]` and
:math:`\nu = \tilde{\nu}`.

Here's our code



.. code-block:: python3

    def Uu(amf, δ, γ):
        A, B, D, F, ν = amf.A, amf.B, amf.D, amf.F, amf.ν
        ν_tilde, H, g = amf.multiplicative_decomp()

        resolv = 1 / (1 - np.exp(-δ) * A)
        vect = F + D * resolv * B

        U_risky = np.exp(-δ) * resolv * D
        u_risky = (np.exp(-δ) / (1 - np.exp(-δ))) * (ν + (.5) * (1 - γ) * (vect**2))

        U_det = 0
        u_det = (np.exp(-δ) / (1 - np.exp(-δ))) * ν_tilde

        return U_risky, u_risky, U_det, u_det

    # Set remaining parameters
    δ = 0.02
    γ = 2.0

    # Get coefficients
    U_r, u_r, U_d, u_d = Uu(amf_2, δ, γ)



The values of the two processes are

.. math::

    \begin{aligned}
        \log V^r_0 &= \log c^r_0 + U^r x_0 + u^r
         \\
        \log V^d_0 &= \log c^d_0 + U^d x_0 + u^d
    \end{aligned}


We look for the ratio :math:`\frac{c^r_0-c^d_0}{c^r_0}` that makes
:math:`\log V^r_0 - \log V^d_0 = 0`

.. math::

    \begin{aligned}
        \underbrace{ \log V^r_0 - \log V^d_0}_{=0} + \log c^d_0 - \log c^r_0
          &= (U^r-U^d) x_0 + u^r - u^d
        \\
     \frac{c^d_0}{ c^r_0}
         &= \exp\left((U^r-U^d) x_0 + u^r - u^d\right)
    \end{aligned}


Hence, the implied percentage reduction in :math:`c_0` that the
representative consumer would accept is given by

.. math::

    \frac{c^r_0-c^d_0}{c^r_0} = 1 - \exp\left((U^r-U^d) x_0 + u^r - u^d\right)


Let's compute this



.. code-block:: python3

    x0 = 0.0  # initial conditions
    logVC_r = U_r * x0 + u_r
    logVC_d = U_d * x0 + u_d

    perc_reduct = 100 * (1 - np.exp(logVC_r - logVC_d))
    perc_reduct



We find that the consumer would be willing to take a percentage reduction of initial consumption equal to around 1.081.
