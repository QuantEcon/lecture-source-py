.. _odu:

.. include:: /_static/includes/lecture_howto_py.raw

.. highlight:: python3

******************************************
Job Search III: Search with Learning
******************************************

.. contents:: :depth: 2

Overview
============

In this lecture we consider an extension of the :doc:`previously studied <mccall_model>` job search model of McCall :cite:`McCall1970`

In the McCall model, an unemployed worker decides when to accept a permanent position at a specified wage, given

* his or her discount rate

* the level of unemployment compensation

* the distribution from which wage offers are drawn



In the version considered below, the wage distribution is unknown and must be learned

* The following is based on the presentation in :cite:`Ljungqvist2012`, section 6.6


Model features
----------------

* Infinite horizon dynamic programming with two states and one binary control

* Bayesian updating to learn the unknown distribution




Model
========

.. index::
    single: Models; McCall

Let's first review the basic McCall model :cite:`McCall1970` and then add the variation we want to consider



The Basic McCall Model
------------------------

.. index::
    single: McCall Model

Recall that, :doc:`in the baseline model <mccall_model>`, an unemployed worker is presented in each period with a
permanent job offer at wage :math:`W_t`

At time :math:`t`, our worker either

#. accepts the offer and works permanently at constant wage :math:`W_t`

#. rejects the offer, receives unemployment compensation :math:`c` and reconsiders next period

The wage sequence :math:`\{W_t\}` is iid and generated from known density :math:`h`

The worker aims to maximize the expected discounted sum of earnings :math:`\mathbb{E} \sum_{t=0}^{\infty} \beta^t y_t`
The function :math:`V` satisfies the recursion

.. math::
    :label: odu_odu_pv

    V(w)
    = \max \left\{
    \frac{w}{1 - \beta}, \, c + \beta \int V(w')h(w') dw'
    \right\}


The optimal policy has the form :math:`\mathbf{1}\{w \geq \bar w\}`, where
:math:`\bar w` is a constant depending called the *reservation wage*


Offer Distribution Unknown
----------------------------

Now let's extend the model by considering the variation presented in :cite:`Ljungqvist2012`, section 6.6

The model is as above, apart from the fact that

* the density :math:`h` is unknown

* the worker learns about :math:`h` by starting with a prior and updating based on wage offers that he/she observes

The worker knows there are two possible distributions :math:`F` and :math:`G` --- with densities :math:`f` and :math:`g`

At the start of time, "nature" selects :math:`h` to be either :math:`f` or
:math:`g` --- the wage distribution from which the entire sequence :math:`\{W_t\}` will be drawn

This choice is not observed by the worker, who puts prior probability :math:`\pi_0` on :math:`f` being chosen

Update rule: worker's time :math:`t` estimate of the distribution is :math:`\pi_t f + (1 - \pi_t) g`, where :math:`\pi_t` updates via

.. math::
    :label: odu_pi_rec

    \pi_{t+1}
    = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}


This last expression follows from Bayes' rule, which tells us that

.. math::

    \mathbb{P}\{h = f \,|\, W = w\}
    = \frac{\mathbb{P}\{W = w \,|\, h = f\}\mathbb{P}\{h = f\}}
    {\mathbb{P}\{W = w\}}
    \quad \text{and} \quad
    \mathbb{P}\{W = w\} = \sum_{\psi \in \{f, g\}} \mathbb{P}\{W = w \,|\, h = \psi\} \mathbb{P}\{h = \psi\}


The fact that :eq:`odu_pi_rec` is recursive allows us to progress to a recursive solution method


Letting

.. math::

    h_{\pi}(w) := \pi f(w) + (1 - \pi) g(w)
    \quad \text{and} \quad
    q(w, \pi) := \frac{\pi f(w)}{\pi f(w) + (1 - \pi) g(w)}


we can express the value function for the unemployed worker recursively as
follows

.. math::
    :label: odu_mvf

    V(w, \pi)
    = \max \left\{
    \frac{w}{1 - \beta}, \, c + \beta \int V(w', \pi') \, h_{\pi}(w') \, dw'
    \right\}
    \quad \text{where} \quad
    \pi' = q(w', \pi)


Notice that the current guess :math:`\pi` is a state variable, since it affects the worker's perception of probabilities for future rewards

Parameterization
------------------

Following  section 6.6 of :cite:`Ljungqvist2012`, our baseline parameterization will be

* :math:`f` is :math:`\operatorname{Beta}(1, 1)` scaled (i.e., draws are multiplied by) some factor :math:`w_m` 
  
* :math:`g` is :math:`\operatorname{Beta}(3, 1.2)` scaled (i.e., draws are multiplied by) the same factor :math:`w_m` 

* :math:`\beta = 0.95` and :math:`c = 0.6`

With :math:`w_m = 2`, the densities :math:`f` and :math:`g` have the following shape



.. code-block:: python3

  from scipy.stats import beta
  import matplotlib.pyplot as plt
  import numpy as np

  w_m = 2  # Scale factor

  x = np.linspace(0, w_m, 200)
  plt.figure(figsize=(10, 8))
  plt.plot(x, beta.pdf(x, 1, 1, scale=w_m), label='$f$', lw=2)
  plt.plot(x, beta.pdf(x, 3, 1.2, scale=w_m), label='$g$', lw=2)
  plt.xlim(0, w_m)
  plt.legend()
  plt.show()
  



.. _looking_forward:

Looking Forward
------------------

What kind of optimal policy might result from :eq:`odu_mvf` and the parameterization specified above?

Intuitively, if we accept at :math:`w_a` and :math:`w_a \leq w_b`, then --- all other things being given --- we should also accept at :math:`w_b`

This suggests a policy of accepting whenever :math:`w` exceeds some threshold value :math:`\bar w`

But :math:`\bar w` should depend on :math:`\pi` --- in fact it should be decreasing in :math:`\pi` because

* :math:`f` is a less attractive offer distribution than :math:`g`
* larger :math:`\pi` means more weight on :math:`f` and less on :math:`g`

Thus larger :math:`\pi` depresses the worker's assessment of her future prospects, and relatively low current offers become more attractive

**Summary:**  We conjecture that the optimal policy is of the form
:math:`\mathbb 1\{w \geq \bar w(\pi) \}` for some decreasing function
:math:`\bar w`



Take 1: Solution by VFI
==================================

Let's set about solving the model and see how our results match with our intuition

We begin by solving via value function iteration (VFI), which is natural but ultimately turns out to be second best

The code is as follows

.. _odu_vfi_code:

.. literalinclude:: /_static/code/odu/odu.py

The class ``SearchProblem`` is used to store parameters and methods needed to compute optimal actions

The Bellman operator is implemented as the method ``.bellman_operator()``, while ``.get_greedy()``
computes an approximate optimal policy from a guess ``v`` of the value function

We will omit a detailed discussion of the code because there is a more efficient solution method

These ideas are implemented in the ``.res_wage_operator()`` method

Before explaining it let's look at solutions computed from value function iteration

Here's the value function:



.. code-block:: python3

  from mpl_toolkits.mplot3d.axes3d import Axes3D
  from matplotlib import cm
  from quantecon import compute_fixed_point

  sp = SearchProblem(w_grid_size=100, π_grid_size=100)
  v_init = np.zeros(len(sp.grid_points)) + sp.c / (1 - sp.β)
  v = compute_fixed_point(sp.bellman_operator, v_init)
  policy = sp.get_greedy(v)

  # Make functions from these arrays by interpolation
  vf = LinearNDInterpolator(sp.grid_points, v)
  pf = LinearNDInterpolator(sp.grid_points, policy)

  π_plot_grid_size, w_plot_grid_size = 100, 100
  π_plot_grid = np.linspace(0.001, 0.99, π_plot_grid_size)
  w_plot_grid = np.linspace(0, sp.w_max, w_plot_grid_size)
  
  Z = np.empty((w_plot_grid_size, π_plot_grid_size))
  for i in range(w_plot_grid_size):
      for j in range(π_plot_grid_size):
          Z[i, j] = vf(w_plot_grid[i], π_plot_grid[j])
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.contourf(π_plot_grid, w_plot_grid, Z, 12, alpha=0.6, cmap=cm.jet)
  cs = ax.contour(π_plot_grid, w_plot_grid, Z, 12, colors="black")
  ax.clabel(cs, inline=1, fontsize=10)
  ax.set_xlabel('$\pi$', fontsize=14)
  ax.set_ylabel('$w$', fontsize=14, rotation=0, labelpad=15)
  
  plt.show()
  



The optimal policy:



.. code-block:: python3

  Z = np.empty((w_plot_grid_size, π_plot_grid_size))
  for i in range(w_plot_grid_size):
      for j in range(π_plot_grid_size):
          Z[i, j] = pf(w_plot_grid[i], π_plot_grid[j])

  fig, ax = plt.subplots(figsize=(6, 6))
  ax.contourf(π_plot_grid, w_plot_grid, Z, 1, alpha=0.6, cmap=cm.jet)
  ax.contour(π_plot_grid, w_plot_grid, Z, 1, colors="black")
  ax.set_xlabel('$\pi$', fontsize=14)
  ax.set_ylabel('$w$', fontsize=14, rotation=0, labelpad=15)
  ax.text(0.4, 1.0, 'reject')
  ax.text(0.7, 1.8, 'accept')
  
  plt.show()
  


The code takes several minutes to run

The results fit well with our intuition from section :ref:`looking forward <looking_forward>`

* The black line in the figure above corresponds to the function :math:`\bar w(\pi)` introduced there

* It is decreasing as expected


Take 2: A More Efficient Method
==================================

Our implementation of VFI can be optimized to some degree

But instead of pursuing that, let's consider another method to solve for the optimal policy

We will use iteration with an operator that has the same contraction rate as the Bellman operator, but

* one dimensional rather than two dimensional
* no maximization step

As a consequence, the algorithm is orders of magnitude faster than VFI

This section illustrates the point that when it comes to programming, a bit of 
mathematical analysis goes a long way


Another Functional Equation
-----------------------------

To begin, note that when :math:`w = \bar w(\pi)`, the worker is indifferent
between accepting and rejecting

Hence the two choices on the right-hand side of :eq:`odu_mvf` have equal value:

.. math::
    :label: odu_mvf2

    \frac{\bar w(\pi)}{1 - \beta}
    = c + \beta \int V(w', \pi') \, h_{\pi}(w') \, dw'


Together, :eq:`odu_mvf` and :eq:`odu_mvf2` give

.. math::
    :label: odu_mvf3

    V(w, \pi) =
    \max
    \left\{
        \frac{w}{1 - \beta} ,\, \frac{\bar w(\pi)}{1 - \beta}
    \right\}


Combining :eq:`odu_mvf2` and :eq:`odu_mvf3`, we obtain

.. math::

    \frac{\bar w(\pi)}{1 - \beta}
    = c + \beta \int \max \left\{
        \frac{w'}{1 - \beta} ,\, \frac{\bar w(\pi')}{1 - \beta}
    \right\}
    \, h_{\pi}(w') \, dw'


Multiplying by :math:`1 - \beta`, substituting in :math:`\pi' = q(w', \pi)` and using :math:`\circ` for composition of functions yields

.. math::
    :label: odu_mvf4

    \bar w(\pi)
    = (1 - \beta) c +
    \beta \int \max \left\{ w', \bar w \circ q(w', \pi) \right\} \, h_{\pi}(w') \, dw'


Equation :eq:`odu_mvf4` can be understood as a functional equation, where :math:`\bar w` is the unknown function

* Let's call it the *reservation wage functional equation* (RWFE)
* The solution :math:`\bar w` to the RWFE is the object that we wish to compute


Solving the RWFE
--------------------------------

To solve the RWFE, we will first show that its solution is the
fixed point of a `contraction mapping <https://en.wikipedia.org/wiki/Contraction_mapping>`_

To this end, let

* :math:`b[0,1]` be the bounded real-valued functions on :math:`[0,1]`
* :math:`\| \psi \| := \sup_{x \in [0,1]} | \psi(x) |`

Consider the operator :math:`Q` mapping :math:`\psi \in b[0,1]` into :math:`Q\psi \in b[0,1]` via


.. math::
    :label: odu_dq

    (Q \psi)(\pi)
    = (1 - \beta) c +
    \beta \int \max \left\{ w', \psi \circ q(w', \pi) \right\} \, h_{\pi}(w') \, dw'


Comparing :eq:`odu_mvf4` and :eq:`odu_dq`, we see that the set of fixed points of :math:`Q` exactly coincides with the set of solutions to the RWFE

* If :math:`Q \bar w = \bar w` then :math:`\bar w` solves :eq:`odu_mvf4` and vice versa

Moreover, for any :math:`\psi, \phi \in b[0,1]`, basic algebra and the
triangle inequality for integrals tells us that

.. math::
    :label: odu_nt

    |(Q \psi)(\pi) - (Q \phi)(\pi)|
    \leq \beta \int
    \left|
    \max \left\{w', \psi \circ q(w', \pi) \right\}
    - \max \left\{w', \phi \circ q(w', \pi) \right\}
    \right|
    \, h_{\pi}(w') \, dw'


Working case by case, it is easy to check that for real numbers :math:`a, b, c` we always have

.. math::
    :label: odu_nt2

    | \max\{a, b\} - \max\{a, c\}| \leq | b - c|


Combining :eq:`odu_nt` and :eq:`odu_nt2` yields

.. math::
    :label: odu_nt3

    |(Q \psi)(\pi) - (Q \phi)(\pi)|
    \leq \beta \int
    \left| \psi \circ q(w', \pi) -  \phi \circ q(w', \pi) \right|
    \, h_{\pi}(w') \, dw'
    \leq \beta \| \psi - \phi \|


Taking the supremum over :math:`\pi` now gives us

.. math::
    :label: odu_rwc

    \|Q \psi - Q \phi\|
    \leq \beta \| \psi - \phi \|


In other words, :math:`Q` is a contraction of modulus :math:`\beta` on the
complete metric space :math:`(b[0,1], \| \cdot \|)`

Hence

* A unique solution :math:`\bar w` to the RWFE exists in :math:`b[0,1]`
* :math:`Q^k \psi \to \bar w` uniformly as :math:`k \to \infty`, for any :math:`\psi \in b[0,1]`

Implementation
^^^^^^^^^^^^^^^^^

These ideas are implemented in the ``.res_wage_operator()`` method from ``odu.py`` as shown above

The method corresponds to action of the operator :math:`Q`

The following exercise asks you to exploit these facts to compute an approximation to :math:`\bar w`


Exercises
=============

.. _odu_ex1:

Exercise 1
------------

Use the default parameters and the ``.res_wage_operator()`` method to compute an optimal policy

Your result should coincide closely with the figure for the optimal policy :ref:`shown above<odu_pol_vfi>`

Try experimenting with different parameters, and confirm that the change in
the optimal policy coincides with your intuition



Solutions
==========



Exercise 1
----------

This code solves the "Offer Distribution Unknown" model by iterating on
a guess of the reservation wage function

You should find that the run time is much shorter than that of the value 
function approach in ``odu_vfi.py``

.. code-block:: python3

    sp = SearchProblem(π_grid_size=50)
    
    ϕ_init = np.ones(len(sp.π_grid)) 
    w_bar = compute_fixed_point(sp.res_wage_operator, ϕ_init)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(sp.π_grid, w_bar, linewidth=2, color='black')
    ax.set_ylim(0, 2)
    ax.grid(axis='x', linewidth=0.25, linestyle='--', color='0.25')
    ax.grid(axis='y', linewidth=0.25, linestyle='--', color='0.25')
    ax.fill_between(sp.π_grid, 0, w_bar, color='blue', alpha=0.15)
    ax.fill_between(sp.π_grid, w_bar, 2, color='green', alpha=0.15)
    ax.text(0.42, 1.2, 'reject')
    ax.text(0.7, 1.8, 'accept')
    plt.show()
   
Appendix
=========

The next piece of code is just a fun simulation to see what the effect of a change in the
underlying distribution on the unemployment rate is

At a point in the simulation, the distribution becomes significantly worse

It takes a while for agents to learn this, and in the meantime they are too optimistic, 
and turn down too many jobs

As a result, the unemployment rate spikes

The code takes a few minutes to run

.. code-block:: python3

    # Set up model and compute the function w_bar
    sp = SearchProblem(π_grid_size=50, F_a=1, F_b=1)
    π_grid, f, g, F, G = sp.π_grid, sp.f, sp.g, sp.F, sp.G
    ϕ_init = np.ones(len(sp.π_grid)) 
    w_bar_vals = compute_fixed_point(sp.res_wage_operator, ϕ_init)
    w_bar = lambda x: np.interp(x, π_grid, w_bar_vals)
    
    
    class Agent:
        """
        Holds the employment state and beliefs of an individual agent.
        """
    
        def __init__(self, π=1e-3):
            self.π = π
            self.employed = 1
    
        def update(self, H):
            "Update self by drawing wage offer from distribution H."
            if self.employed == 0:
                w = H.rvs()
                if w >= w_bar(self.π):
                    self.employed = 1
                else:
                    self.π = 1.0 / (1 + ((1 - self.π) * g(w)) / (self.π * f(w)))
    
    
    num_agents = 5000
    separation_rate = 0.025  # Fraction of jobs that end in each period 
    separation_num = int(num_agents * separation_rate)
    agent_indices = list(range(num_agents))
    agents = [Agent() for i in range(num_agents)]
    sim_length = 600
    H = G  # Start with distribution G
    change_date = 200  # Change to F after this many periods
    
    unempl_rate = []
    for i in range(sim_length):
        if i % 20 == 0:
            print(f"date = {i}")
        if i == change_date:
            H = F
        # Randomly select separation_num agents and set employment status to 0
        np.random.shuffle(agent_indices)
        separation_list = agent_indices[:separation_num]
        for agent_index in separation_list:
            agents[agent_index].employed = 0
        # Update agents
        for agent in agents:
            agent.update(H)
        employed = [agent.employed for agent in agents]
        unempl_rate.append(1 - np.mean(employed))
    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(unempl_rate, lw=2, alpha=0.8, label='unemployment rate')
    ax.axvline(change_date, color="red")
    ax.legend()
    plt.show()


