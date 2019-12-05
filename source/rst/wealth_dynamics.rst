.. include:: /_static/includes/header.raw

.. highlight:: python3


****************************
Wealth Distribution Dynamics
****************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install --upgrade quantecon


Overview
========

This notebook gives an introduction to wealth distribution dynamics, with a
focus on 

- modeling and computing the wealth distribution via simulation

- measures of inequality such as the Lorenz curve and Gini coefficient

- how inequality is affected by the properties of wage income and returns on assets.

Clearly, the evolution of wealth for any given household depends on their
savings behavior, and modeling such behavior will form an important part of
this lecture series.

However, in this lecture, we will be content with ad hoc but plausible savings rules.

We do this in order to easily explore the implications of different specifications for income and investment returns.

All of the techniques discussed here can be plugged into models that use more sophisticated methods to obtain savings rules.

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    import quantecon as qe
    from numba import njit, jitclass, float64, prange


Lorenz Curves and the Gini Coefficient
======================================

Before we investigate wealth dynamics, we briefly review some measures of
inequality.

Lorenz Curves
-------------

One popular graphical measure of inequality is the `Lorenz curve <https://en.wikipedia.org/wiki/Lorenz_curve>`__.

The package `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`__, already imported above, contains a function to compute Lorenz curves.

To illustrate, suppose that 

.. code:: ipython3

    n = 10_000                      # size of sample
    w = np.exp(np.random.randn(n))  # lognormal draws

is data representing the wealth of 10,000 households.

We can compute and plot the Lorenz curve as follows:

.. code:: ipython3

    f_vals, l_vals = qe.lorenz_curve(w)

    fig, ax = plt.subplots()
    ax.plot(f_vals, l_vals, label='Lorenz curve, lognormal sample')
    ax.plot(f_vals, f_vals, label='Lorenz curve, equality')
    ax.legend()
    plt.show()

The construction of this curve is as follows: if point :math:`x,y` lies on the curve, it means that, collectively, the bottom :math:`(100x)\%` of the population holds :math:`(100y)\%` of the wealth.

The "equality" line is the 45 degree line (which might not be exactly 45
degrees in the figure, depending on the aspect ratio).

A sample that produces this line exhibits perfect equality.

The lognormal sample deviates significantly from this benchmark.

For example, the bottom 80% of the population holds only 40% of total wealth.

Here is another example, which shows how the Lorenz curve shifts as the
underlying distribution changes.

We generate 10,000 observations using the Pareto distribution with a range of
parameters, and then compute the Lorenz curve corresponding to each set of
observations.

.. code:: ipython3

    a_vals = (1, 2, 5, 10_000_000)  # Pareto tail index 
    n = 10_000                      # size of each sample
    fig, ax = plt.subplots()
    for a in a_vals:
        y = np.random.uniform(size=n)**(-1/a)      # Pareto with tail index a
        f_vals, l_vals = qe.lorenz_curve(y)
        ax.plot(f_vals, l_vals, label=f'$a = {a}$')
    ax.plot(f_vals, f_vals, label='equality')
    ax.legend()
    plt.show()

You can see that, as the tail parameter of the Pareto distribution increases, inequality decreases.

This is to be expected, because a higher tail index implies less weight in the tail of the Pareto distribution.

The Gini Coefficient
--------------------

The definition and interpretation of the Gini coefficient can be found on the corresponding `Wikipedia page <https://en.wikipedia.org/wiki/Gini_coefficient>`__.

A value of 0 indicates perfect equality (corresponding the case where the
Lorenz curve matches the 45 degree line) and a value of 1 indicates complete
inequality (all wealth held by the richest household).

The `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`__ library contains a function to calculate the Gini coefficient.

We can test it on the Weibull distribution with parameter :math:`a`, where the Gini coefficient is known to be

.. math::  G = 1 - 2^{-1/a} 

Let's see if the Gini coefficient computed from a simulated sample matches
this at each fixed value of :math:`a`.

.. code:: ipython3

    a_vals = range(1, 20)
    ginis = []
    ginis_theoretical = []
    n = 100
    
    fig, ax = plt.subplots()
    for a in a_vals:
        y = np.random.weibull(a, size=n)
        ginis.append(qe.gini_coefficient(y))
        ginis_theoretical.append(1 - 2**(-1/a))
    ax.plot(a_vals, ginis, label='estimated gini coefficient')
    ax.plot(a_vals, ginis_theoretical, label='theoretical gini coefficient')
    ax.legend()
    ax.set_xlabel("Weibull parameter $a$")
    ax.set_ylabel("Gini coefficient")
    plt.show()

The simulation shows that the fit is good.



A Model of Wealth Dynamics
==========================

Having discussed inequality measures, let us now turn to wealth dynamics.

The model we will study is

.. math::  w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1} 

where

-  :math:`w_t` is wealth at time :math:`t` for a given household,
-  :math:`r_t` is the rate of return of financial assets,
-  :math:`y_t` is current non-financial (e.g., labor) income and
-  :math:`s(w_t)` is current wealth net of consumption

Letting :math:`\{z_t\}` be a correlated state process of the form

.. math::  z_{t+1} = a z_t + b + \sigma_z \epsilon_{t+1} 

we’ll assume that

.. math::  R_t := 1 + r_t = c_r \exp(z_t) + \exp(\mu_r + \sigma_r \xi_t) 

and

.. math::  y_t = c_y \exp(z_t) + \exp(\mu_y + \sigma_y \zeta_t) 

Here :math:`\{ (\epsilon_t, \xi_t, \zeta_t) \}` is IID and standard
normal in :math:`\mathbb R^3`.

The value of :math:`c_r` should be close to zero, since rates of return
on assets do not exhibit large trends.

When we simulate a population of households, we will take

-  :math:`\{z_t\}` to be an **aggregate shock** that is common to all
   households
-  :math:`\{\xi_t\}` and :math:`\{\zeta_t\}` to be **idiosyncratic
   shocks**

Idiosyncratic shocks are specific to individual households and
independent across them.

Regarding the savings function :math:`s`, our default model will be

.. math::  s(w) = s_0 w \cdot \mathbb 1\{w \geq \hat w\} 

where :math:`s_0` is a positive constant.

Thus, for :math:`w < \hat w`, the household saves nothing. For
:math:`w \geq \bar w`, the household saves a fraction :math:`s_0` of
their wealth.

We are using something akin to a fixed savings rate model, while
acknowledging that low wealth households tend to save very little.

Implementation
==============

Here's some type information to help Numba.

.. code:: ipython3

    wealth_dynamics_data = [
        ('w_hat',  float64),    # savings parameter
        ('s_0',    float64),    # savings parameter
        ('c_y',    float64),    # labor income parameter
        ('μ_y',    float64),    # labor income paraemter
        ('σ_y',    float64),    # labor income parameter
        ('c_r',    float64),    # rate of return parameter
        ('μ_r',    float64),    # rate of return parameter
        ('σ_r',    float64),    # rate of return parameter
        ('a',      float64),    # aggregate shock parameter
        ('b',      float64),    # aggregate shock parameter
        ('σ_z',    float64),    # aggregate shock parameter
        ('z_mean', float64),    # mean of z process
        ('y_mean', float64),    # mean of y process
        ('R_mean', float64)     # mean of R process
    ]

Here's a class that stores instance data and implements methods that update
the aggregate state and household wealth.

.. code:: ipython3

    @jitclass(wealth_dynamics_data)
    class WealthDynamics:
        
        def __init__(self,
                     w_hat=10.0,
                     s_0=0.75, 
                     c_y=1.0, 
                     μ_y=1.5,
                     σ_y=0.2, 
                     c_r=0.05,
                     μ_r=0.0, 
                     σ_r=0.6, 
                     a=0.9,  
                     b=0.0, 
                     σ_z=0.1):
        
            self.w_hat, self.s_0 = w_hat, s_0
            self.c_y, self.μ_y, self.σ_y = c_y, μ_y, σ_y
            self.c_r, self.μ_r, self.σ_r = c_r, μ_r, σ_r
            self.a, self.b, self.σ_z = a, b, σ_z
            
            # Record stationary moments
            self.z_mean = b / (1 - a)
            z_var = σ_z**2 / (1 - a**2)
            exp_z_mean = np.exp(self.z_mean + z_var / 2)
            self.R_mean = c_r * exp_z_mean + np.exp(μ_r + σ_r**2 / 2)
            self.y_mean = c_y * exp_z_mean + np.exp(μ_y + σ_y**2 / 2)
            
            # Test a stability condition that ensures wealth does not diverge
            # to infinity.
            α = self.R_mean * self.s_0
            if α >= 1:
                raise ValueError("Stability condition failed.")
                
        def parameters(self):
            """
            Collect and return parameters.
            """
            parameters = (self.w_hat, self.s_0, 
                          self.c_y, self.μ_y, self.σ_y,
                          self.c_r, self.μ_r, self.σ_r,
                          self.a, self.b, self.σ_z)
            return parameters
            
        def update_z(self, z):
            """
            Update exogenous state one period, given current state z.
            """   
            # Simplify names
            params = self.parameters()
            w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z = params
            
            # Update z
            zp = a * z + b + σ_z * np.random.randn()
            return zp
            
        def update_wealth(self, w, zp):
            """
            Update one period, given current wealth w and next period 
            exogenous state zp.
            """   
            # Simplify names
            params = self.parameters()
            w_hat, s_0, c_y, μ_y, σ_y, c_r, μ_r, σ_r, a, b, σ_z = params

            # Update wealth
            y = c_y * np.exp(zp) + np.exp(μ_y + σ_y * np.random.randn())   
            wp = y
            if w >= w_hat:
                R = c_r * np.exp(zp) + np.exp(μ_r + σ_r * np.random.randn())
                wp += R * s_0 * w 
            return wp
            
        
Here’s a function to simulate the aggregate state:

.. code:: ipython3

    @njit
    def z_time_series(wdy, ts_length=10_000):
        """
        Generate a single time series of length ts_length for the 
        aggregate shock process {z_t}.

        * wdy is an instance of WealthDynamics

        """
        z = np.empty(ts_length)
        z[0] = wdy.z_mean
        for t in range(ts_length-1):
            z[t+1] = wdy.update_z(z[t])
        return z

Here’s a function to simulate wealth of a single household, taken the
time path of the aggregate state as an argument.

.. code:: ipython3

    @njit
    def wealth_time_series(wdy, w_0, z):
        """
        Generate a single time series for wealth given aggregate shock 
        sequence z.

        * wdy: an instance of WealthDynamics
        * w_0: a scalar representing initial wealth.
        * z: array_like
        
        The returned time series w satisfies len(w) = len(z).
        """
        n = len(z)
        w = np.empty(n)
        w[0] = w_0
        for t in range(n-1):
            w[t+1] = wdy.update_wealth(w[t], z[t+1])
        return w


Finally, here's function to simulate a cross section of households forward in
time.

.. code:: ipython3

    @njit(parallel=True)
    def update_cross_section(wdy, w_distribution, shift_length=500):
        """
        Shifts a cross-section of household forward in time

        * wdy: an instance of WealthDynamics
        * w_distribution: array_like, represents current cross-section
        
        Takes a current distribution of wealth values as w_distribution
        and updates each w_t in w_distribution to w_{t+j}, where 
        j = shift_length.  
        
        Returns the new distribution.
        """
        new_distribution = np.empty_like(w_distribution)
    
        z = z_time_series(wdy, ts_length=shift_length)
    
        # Update each household
        for i in prange(len(new_distribution)):
            w = w_distribution[i]
            for t in range(shift_length-1):
                w = wdy.update_wealth(w, z[t+1])
            new_distribution[i] = w
        return new_distribution
    


Applications 
============

Let's try simulating the model at different parameter values and investigate
the implications for the wealth distribution.


Time Series
-----------

Let’s look at the wealth dynamics of an individual household. Large
spikes are indicative of heavy tails.

.. code:: ipython3

    wdy = WealthDynamics(σ_r=0.7)
    
    z = z_time_series(wdy, ts_length=100_000)
    w = wealth_time_series(wdy, wdy.y_mean, z)
    
    fig, ax = plt.subplots()
    ax.plot(w)
    plt.show()




Inequality Measures
-------------------


Let’s look at how inequality varies with returns on financial assets.


The next function generates a cross section and then computes the Lorenz
curve and Gini coefficient.

.. code:: ipython3

    def generate_lorenz_and_gini(wdy, num_households=250_000, T=500):
        """
        Generate the Lorenz curve data and gini coefficient corresponding to a 
        WealthDynamics mode by simulating num_households forward to time T.
        """
        ψ_0 = np.ones(num_households) * wdy.y_mean
        z_0 = wdy.z_mean
    
        ψ_star = update_cross_section(wdy, ψ_0, shift_length=T)
        return gini(ψ_star), lorenz(ψ_star)

Here’s the Lorenz curves as a function of :math:`\mu_r`, a parameter
that shifts up mean returns.

.. code:: ipython3

    fig, ax = plt.subplots()
    μ_r_vals = np.linspace(0.0, 0.05, 3)
    gini_vals = []
    
    for μ_r in μ_r_vals:
        wdy = WealthDynamics(μ_r=μ_r)
        gv, (f_vals, l_vals) = generate_lorenz_and_gini(wdy)
        ax.plot(f_vals, l_vals, label=f'$\psi^*$ at $\mu_r = {μ_r:0.2}$')
        gini_vals.append(gv)
        
    ax.plot(f_vals, f_vals, label='equality')
    ax.legend(loc="upper left")
    plt.show()

Here’s the Gini coefficient. Look how inequality increases with mean
returns on financial income.

.. code:: ipython3

    fig, ax = plt.subplots()    
    ax.plot(μ_r_vals, gini_vals, label='gini coefficient')
    ax.set_xlabel("$\mu_r$")
    ax.legend()
    plt.show()


Exercises
=========

Exercise 1
----------

It was claimed in the discussion above that, when the tail index of the Pareto
distribution increases, inequality fails, since a higher tail index implies
less weight in the right hand tail.

In fact, it is possible to prove that the Gini coefficient of the Pareto
distribution with tail index :math:`a` is :math:`1/(2a - 1)`.

To the extent that you can, confirm this by simulation.

In particular, generate a plot of the Gini coefficient against the tail index
using both the theoretical value and the value computed from a sample via
``qe.gini_coefficient``.

For the values of the tail index, use ``a_vals = np.linspace(1, 10, 25)``.

Use sample of size 1,000 for each :math:`a` and the sampling method for generating
Pareto draws employed in the discussion of Lorenz curves for the Pareto distribution.


Solutions
=========

Exercise 1
----------

.. code:: ipython3

    a_vals = np.linspace(1, 10, 25)  # Pareto tail index 
    ginis = np.empty_like(a_vals)

    n = 1000                         # size of each sample
    fig, ax = plt.subplots()
    for i, a in enumerate(a_vals):
        y = np.random.uniform(size=n)**(-1/a)      
        ginis[i] = qe.gini_coefficient(y)
    ax.plot(a_vals, ginis, label='sampled')
    ax.plot(a_vals, 1/(2*a_vals - 1), label='theoretical')
    ax.legend()
    plt.show()
