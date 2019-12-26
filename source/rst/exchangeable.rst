.. _odu_v3:

.. include:: /_static/includes/header.raw

.. highlight:: python3


Exchangeability and Bayesian Updating
======================================

Overview
---------

This lecture studies an example  of learning 
via Bayes' Law.

We touch on foundations of Bayesian statistical inference invented by Bruno DeFinetti :cite:`definetti`.

The relevance of DeFinetti's work for economists is presented forcefully 
in chapter 11 of :cite:`kreps` by David Kreps.

The example  that we study in this lecture  is a key component of :doc:`this lecture <odu>` that augments the 
:doc:`classic  <mccall_model>`  job search model of McCall
:cite:`McCall1970` by presenting an unemployed worker with a statistical inference problem. 

Here we create  graphs that illustrate the role that  a  likelihood ratio  
plays in  Bayes' Law.

We'll use such graphs to provide insights into the mechanics driving outcomes in :doc:`this lecture<odu>` about learning in an augmented McCall job
search model.

Among other things, this lecture discusses  connections between the statistical concepts of sequences of random variables
that are

- independently and identically distributed

- exchangeable

Under standing the distinction between these concepts is essential for appreciating how Bayesian updating 
works in our example.


Below, we'll often use 

- :math:`W` to denote a random variable

- :math:`w` to denote a particular realization of a random variable :math:`W` 

Let’s start with some imports:

.. code-block:: ipython
    :class: hide-output

    from numba import njit, vectorize
    from math import gamma
    import scipy.optimize as op
    from scipy.integrate import quad
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline


Independently and Identically Distributed 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by looking at the notion of an  **independently and identically  distributed sequence** of random variables.

An independently and identically distributed sequence is often abbreviated as IID.

Two notions are involved, **independently** and **identically** distributed.

A sequence :math:`W_0, W_1, \ldots` is **independently distributed** if the joint probability density
of the sequence is the **product** of the densities of the  components of the sequence. 

The sequence :math:`W_0, W_1, \ldots` is **independently and identically distributed** if in addition the marginal
density of :math:`W_t` is the same for all :math:`t =0, 1, \ldots`.  

For example,  let :math:`p(W_0, W_1, \ldots)` be the **joint density** of the sequence and
let :math:`p(W_t)` be the **marginal density** for a particular :math:`W_t` for all :math:`t =0, 1, \ldots`.

Then the joint density of the sequence :math:`W_0, W_1, \ldots` is IID if 

.. math:: p(W_0, W_1, \ldots) =  p(W_0) p(W_1) \cdots 

so that the joint density is the product of a sequence of identical marginal densities.


IID Means Past Observations Don't Tell Us Anything About Future Observations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a sequence is random variables is IID, past information provides no information about future realizations.

In this sense, there is **nothing to learn**  about the future from the past.  



To understand these statements, let the joint distribution of a sequence of random variables :math:`\{W_t\}_{t=0}^T`
that is not necessarily IID, be

.. math::  p(W_T, W_{T-1}, \ldots, W_1, W_0)


Using the laws of probability, we can always factor such a joint density into a product of conditional densities:

.. math::

   \begin{align}
     p(W_T, W_{T-1}, \ldots, W_1, W_0)    = & p(W_T | W_{t-1}, \ldots, W_0) p(W_{T-1} | W_{T-2}, \ldots, W_0) \cdots  \cr
     & p(W_1 | W_0) p(W_0) 
   \end{align}



In general,   

.. math::  p(W_t | W_{t-1}, \ldots, W_0)   \neq   p(W_t)             

which states that the **conditional density** on the left side does not equal the **marginal density** on the right side.

In the special IID case, 

.. math::  p(W_t | W_{t-1}, \ldots, W_0)   =  p(W_t)

and partial history :math:`W_{t-1}, \ldots, W_0` contains no information about the probability of :math:`W_t`.

So in the IID case, there is **nothing to learn** about the densities of future random variables from past data.   
 
In the general case, there is something go learn from past data.

We turn next to an instance of this general case in which there is something to learn from past data. 

Please keep your eye out for **what** there is to learn from past data. 



A Setting in Which Past Observations Are Informative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now turn to a setting in which there **is** something to learn.  

Let :math:`\{W_t\}_{t=0}^\infty` be a sequence of nonnegative
scalar random variables with a joint probability distribution
constructed as follows.

There are two distinct cumulative distribution functions :math:`F` and :math:`G`
— with densities :math:`f` and :math:`g` for a nonnegative scalar random
variable :math:`W`.

Before the start of time, say at time :math:`t= -1`, “nature” once and for
all selects **either** :math:`f` **or** :math:`g` — and thereafter at each time
:math:`t \geq 0` draws a random :math:`W` from the selected
distribution.

So  the data are permanently generated as independently and identically distributed (IID) draws from **either** :math:`F` **or**
:math:`G`.

We could say that *objectively* the probability that the data are generated as draws from :math:`F` is either :math:`0`
or :math:`1`.  

We now drop into this setting a decision maker who knows :math:`F` and :math:`G` and that nature picked one 
of them once and for all and then drew an IID sequence of draws from that distribution.

But our decision maker does not know which of the two distributions nature selected.  

The decision maker summarizes his ignorance about this by picking a **subjective probability** 
:math:`\tilde \pi` and reasons as if  nature had selected :math:`F` with probability
:math:`\tilde \pi \in (0,1)` and
:math:`G` with probability :math:`1 - \tilde \pi`.

Thus, we  assume that the decision maker

 - **knows** both :math:`F` and :math:`G`

 - **doesnt't know** which of these two distributions that nature has drawn 

 - summarizing his ignorance by acting  as if or **thinking** that nature chose distribution :math:`F` with probability :math:`\tilde \pi \in (0,1)` and distribution
   :math:`G` with probability :math:`1 - \tilde \pi` 


 - at date :math:`t \geq 0` has observed  the partial history :math:`w_t, w_{t-1}, \ldots, w_0` of draws from the appropriate joint
   density of the partial history

But what do we mean by the *appropriate joint distribution*?

We'll discuss that next and in the process describe the concept of **exchangeability**.

Relationship Between IID and Exchangeable 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Conditional on nature selecting :math:`F`, the joint density of the
sequence :math:`W_0, W_1, \ldots` is

.. math::  f(W_0) f(W_1) \cdots 

Conditional on nature selecting :math:`G`, the joint density of the
sequence :math:`W_0, W_1, \ldots` is

.. math::  g(W_0) g(W_1) \cdots 

Notice that **conditional on nature having selected** :math:`F`, the
sequence :math:`W_0, W_1, \ldots` is independently and
identically distributed.

Furthermore,  **conditional on nature having
selected** :math:`G`, the sequence :math:`W_0, W_1, \ldots` is also
independently and identically distributed.

But what about the unconditional distribution?

The unconditional distribution of :math:`W_0, W_1, \ldots` is
evidently

.. math::
    :label: eq_definetti

    h(W_0, W_1, \ldots ) \equiv \tilde \pi [f(W_0) f(W_1) \cdots ] + ( 1- \tilde \pi) [g(W_0) g(W_1) \cdots ] 

Under the unconditional distribution :math:`h(W_0, W_1, \ldots )`, the
sequence :math:`W_0, W_1, \ldots` is **not** independently and
identically distributed.

To verify this claim, it is sufficient to notice, for example, that

.. math::

    h(w_0, w_1) = \tilde \pi f(w_0)f (w_1) + (1 - \tilde \pi) g(w_0)g(w_1) \neq
                  (\tilde \pi f(w_0) + (1-\tilde \pi) g(w_0))(
                   \tilde \pi f(w_1) + (1-\tilde \pi) g(w_1))  

Thus, the conditional distribution

.. math::

    h(w_1 | w_0) \equiv \frac{h(w_0, w_1)}{(\tilde \pi f(w_0) + (1-\tilde \pi) g(w_0))}
     \neq ( \tilde \pi f(w_1) + (1-\tilde \pi) g(w_1)) 

This means that the realization :math:`w_0` contains information about :math:`w_1`.

So there is something to learn.  

But what and how?

Exchangeability 
^^^^^^^^^^^^^^^     

While the sequence :math:`W_0, W_1, \ldots` is not IID, it can be verified that it is
**exchangeable**, which means that

.. math::  h(w_0, w_1) = h(w_1, w_0) 

and so on.


More generally, a sequence of random variables is said to be **exchangeable** if  the  joint probability distribution
for the sequence does not change when the positions in the sequence in which finitely many of the random variables
appear are altered.

Equation :eq:`eq_definetti` represents our instance of an exchangeable joint density over a sequence of random 
variables  as a **mixture**  of  two IID joint densities over a sequence of random variables.

For a Bayesian statistician, the mixing parameter :math:`\tilde \pi \in (0,1)` has a special interpretation
as a **prior probability** that nature selected probability distribution :math:`F`.

DeFinetti :cite:`definetti` established a related representation of an exchangeable process created by mixing
sequences of IID Bernoulli random variables with parameters :math:`\theta` and mixing probability :math:`\pi(\theta)`
for a density :math:`\pi(\theta)` that a Bayesian statistician would interpret as a prior over the unknown
Bernoulli paramter :math:`\theta`.


Bayes' Law
^^^^^^^^^^^^

We noted above that in our example model there is something to learn about about the future from past data drawn 
from our particular instance of a process that is exchangeable but not IID.

But how can we learn?

And about what?

The answer to the *about what* question is about :math:`\tilde pi`.

The answer to the *how* question is to use  Bayes' Law.

Another way to say *use Bayes' Law* is to say *compute an appropriate conditional distribution*.

Let's dive into Bayes' Law in this context.



Let :math:`q` represent the distribution that nature actually draws from
:math:`w` from and let

.. math::   \pi = \mathbb{P}\{q = f \} 

where we regard :math:`\pi` as the decision maker's **subjective probability**  (also called a **personal probability**.

Suppose that at :math:`t \geq 0`, the decision maker has  observed a history
:math:`w^t \equiv [w_t, w_{t-1}, \ldots, w_0]`.

We let

.. math::  \pi_t  = \mathbb{P}\{q = f  | w^t \} 

where we adopt the convention

.. math:: \pi_{-1}  = \tilde \pi 

The distribution of :math:`w_{t+1}` conditional on :math:`w^t` is then

.. math::  \pi_t f + (1 - \pi_t) g . 

Bayes’ rule for updating :math:`\pi_{t+1}` is

.. math::
   :label: eq_Bayes102

   \pi_{t+1}
   = \frac{\pi_t f(w_{t+1})}{\pi_t f(w_{t+1}) + (1 - \pi_t) g(w_{t+1})}

The last expression follows from Bayes’ rule, which
tells us that

.. math::


   \mathbb{P}\{q = f \,|\, W = w\}
   = \frac{\mathbb{P}\{W = w \,|\, q = f\}\mathbb{P}\{q = f\}}
   {\mathbb{P}\{W = w\}}
   \quad \text{and} \quad
   \mathbb{P}\{W = w\} = \sum_{\omega \in \{f, g\}} \mathbb{P}\{W = w \,|\, q = \omega\} \mathbb{P}\{q = \omega\}

More Details about Bayesian Updating 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's stare at and rearrange Bayes' Law as represented in equation :eq:`eq_Bayes102` with the aim of understanding
how the **posterior** :math:`\pi_{t+1}` is influenced by the **prior** :math:`\pi_t` and the **likelihood ratio**


.. math::

   l(w) = \frac{f(w)}{g(w)}


It is convenient for us to rewrite the updating rule :eq:`eq_Bayes102` as

.. math::


   \pi_{t+1}   =\frac{\pi_{t}f\left(w_{t+1}\right)}{\pi_{t}f\left(w_{t+1}\right)+\left(1-\pi_{t}\right)g\left(w_{t+1}\right)}
       =\frac{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}}{\pi_{t}\frac{f\left(w_{t+1}\right)}{g\left(w_{t+1}\right)}+\left(1-\pi_{t}\right)}
       =\frac{\pi_{t}l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}


This implies that

.. math::
   :label: eq_Bayes103

   \frac{\pi_{t+1}}{\pi_{t}}=\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\begin{cases}
   >1 & \text{if }l\left(w_{t+1}\right)>1\\
   \leq1 & \text{if }l\left(w_{t+1}\right)\leq1
   \end{cases}

Notice how the likelihood ratio and the prior interact to determine whether an observation :math:`w_{t+1}` leads the decision maker 
to increase or decrease the subjective probability he/she attaches to distribution :math:`F`. 

When the likelihood ratio :math:`l(w_{t+1})` exceeds one, the observation :math:`w_{t+1}` nudges the probability 
:math:`\pi` put on distribution :math:`F` upward,
and when the likelihood ratio :math:`l(w_{t+1})` is less that  one, the observation :math:`w_{t+1}` nudges :math:`\pi` downward.



Representation :eq:`eq_Bayes103` is the foundation of the graphs that we'll use to display the dynamics of 
:math:`\{\pi_t\}_{t=0}^\infty` that are  induced by
Bayes' Law.  

We’ll plot :math:`l\left(w\right)` as a way to enlighten us about how
learning – i.e., Bayesian updating of the probability :math:`\pi` that
nature has chosen distribution :math:`f` – works.


To create the Python infrastructure to do our work for us,  we construct a wrapper function that displays informative graphs
given parameters of :math:`f` and :math:`g`.

.. code-block:: python3

    @vectorize
    def p(x, a, b):
        "The general beta distribution function."
        r = gamma(a + b) / (gamma(a) * gamma(b))
        return r * x ** (a-1) * (1 - x) ** (b-1)

    def learning_example(F_a=1, F_b=1, G_a=3, G_b=1.2):
        """
        A wrapper function that displays the updating rule of belief π,
        given the parameters which specify F and G distributions.
        """

        f = njit(lambda x: p(x, F_a, F_b))
        g = njit(lambda x: p(x, G_a, G_b))

        # l(w) = f(w) / g(w)
        l = lambda w: f(w) / g(w)
        # objective function for solving l(w) = 1
        obj = lambda w: l(w) - 1

        x_grid = np.linspace(0, 1, 100)
        π_grid = np.linspace(1e-3, 1-1e-3, 100)

        w_max = 1
        w_grid = np.linspace(1e-12, w_max-1e-12, 100)

        # the mode of beta distribution
        # use this to divide w into two intervals for root finding
        G_mode = (G_a - 1) / (G_a + G_b - 2)
        roots = np.empty(2)
        roots[0] = op.root_scalar(obj, bracket=[1e-10, G_mode]).root
        roots[1] = op.root_scalar(obj, bracket=[G_mode, 1-1e-10]).root

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.plot(l(w_grid), w_grid, label='$l$', lw=2)
        ax1.vlines(1., 0., 1., linestyle="--")
        ax1.hlines(roots, 0., 2., linestyle="--")
        ax1.set_xlim([0., 2.])
        ax1.legend(loc=4)
        ax1.set(xlabel='$l(w)=f(w)/g(w)$', ylabel='$w$')

        ax2.plot(f(x_grid), x_grid, label='$f$', lw=2)
        ax2.plot(g(x_grid), x_grid, label='$g$', lw=2)
        ax2.vlines(1., 0., 1., linestyle="--")
        ax2.hlines(roots, 0., 2., linestyle="--")
        ax2.legend(loc=4)
        ax2.set(xlabel='$f(w), g(w)$', ylabel='$w$')

        area1 = quad(f, 0, roots[0])[0]
        area2 = quad(g, roots[0], roots[1])[0]
        area3 = quad(f, roots[1], 1)[0]

        ax2.text((f(0) + f(roots[0])) / 4, roots[0] / 2, f"{area1: .3g}")
        ax2.fill_between([0, 1], 0, roots[0], color='blue', alpha=0.15)
        ax2.text(np.mean(g(roots)) / 2, np.mean(roots), f"{area2: .3g}")
        w_roots = np.linspace(roots[0], roots[1], 20)
        ax2.fill_betweenx(w_roots, 0, g(w_roots), color='orange', alpha=0.15)
        ax2.text((f(roots[1]) + f(1)) / 4, (roots[1] + 1) / 2, f"{area3: .3g}")
        ax2.fill_between([0, 1], roots[1], 1, color='blue', alpha=0.15)

        W = np.arange(0.01, 0.99, 0.08)
        Π = np.arange(0.01, 0.99, 0.08)

        ΔW = np.zeros((len(W), len(Π)))
        ΔΠ = np.empty((len(W), len(Π)))
        for i, w in enumerate(W):
            for j, π in enumerate(Π):
                lw = l(w)
                ΔΠ[i, j] = π * (lw / (π * lw + 1 - π) - 1)

        q = ax3.quiver(Π, W, ΔΠ, ΔW, scale=2, color='r', alpha=0.8)

        ax3.fill_between(π_grid, 0, roots[0], color='blue', alpha=0.15)
        ax3.fill_between(π_grid, roots[0], roots[1], color='green', alpha=0.15)
        ax3.fill_between(π_grid, roots[1], w_max, color='blue', alpha=0.15)
        ax3.hlines(roots, 0., 1., linestyle="--")
        ax3.set(xlabel='$\pi$', ylabel='$w$')
        ax3.grid()

        plt.show()


Now we'll create a group of graphs designed to illustrate the dynamics induced by Bayes' Law.

We'll begin with the default values of various objects, then change them in a subsequent example.

.. code-block:: python3

    learning_example()

Please look at the three graphs above created for an instance in which :math:`f` is a uniform distribution on :math:`[0,1]`
(i.e., a Beta distribution with parameters :math:`F_a=1, F_b=1`, while  :math:`g` is a Beta distribution with the default parameter values :math:`G_a=3, G_b=1.2`.  



The graph in the left  plots the likehood ratio :math:`l(w)` on the coordinate axis against :math:`w` on the coordinate axis.

The middle graph plots both :math:`f(w)` and :math:`g(w)`  against :math:`w`, with the horizontal dotted lines showing values 
of :math:`w` at which the likelihood ratio equals :math:`1`.

The graph on the right side plots arrows to the right that show when Bayes' Law  makes :math:`\pi` increase and arrows 
to the left that show when Bayes' Law make :math:`\pi` decrease.  

Notice how the length of the arrows, which show the magnitude of the force from Bayes' Law impelling :math:`\pi` to change,
depend on both the prior probability :math:`\pi` on the ordinate axis and the evidence in the form of the current draw of 
:math:`w` on the coordinate axis.  


The fractions in the colored areas of the middle graphs are probabilities under :math:`F` and :math:`G`, respectively,
that  realizations of :math:`w` fall
into the interval that updates the belief :math:`\pi` in a correct direction (i.e., toward :math:`0` when :math:`G` is the true 
distribution, and towards :math:`1` when :math:`F` is the true distribution).

For example, 
in the above  example, under true distribution :math:`F`,  :math:`\pi` will  be updated toward :math:`0` if :math:`w` falls into the interval
:math:`[0.524, 0.999]`, which occurs with probability :math:`1 - .524 = .476` under :math:`F`.  But this
would occur with probability
:math:`0.816` if :math:`G` were the true distribution.  The fraction :math:`0.816`
in the orange region is the integral of :math:`g(w)` over this interval.


Next we use our code to create graphs for another instance of our model.

We keep :math:`F` the same as in the preceding instance, namely a uniform distribution, but now assume that :math:`G`
is a Beta distribution with parameters :math:`G_a=2, G_b=1.6`. 

.. code-block:: python3

    learning_example(G_a=2, G_b=1.6)

Notice how the likelihood ratio, the middle graph, and the arrows compare with the previous instance of our example.    



Appendix 
-----------

Sample Paths of :math:`\pi_t`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we'll have some fun by plotting multiple realizations of sample paths of :math:`\pi_t` under two possible 
assumptions about nature's choice of distribution:

-  that nature permanently draws from :math:`F`

-  that nature permanently draws from :math:`G`


Outcomes depend on a peculiar property of likelihood ratio processes that are discussed in
:doc:`this lecture<additive_functionals>` 

To do this, we create some Python code.

.. code-block:: python3

    def function_factory(F_a=1, F_b=1, G_a=3, G_b=1.2):

        # define f and g
        f = njit(lambda x: p(x, F_a, F_b))
        g = njit(lambda x: p(x, G_a, G_b))

        @njit
        def update(a, b, π):
            "Update π by drawing from beta distribution with parameters a and b"

            # Draw
            w = np.random.beta(a, b)

            # Update belief
            π = 1 / (1 + ((1 - π) * g(w)) / (π * f(w)))

            return π

        @njit
        def simulate_path(a, b, T=50):
            "Simulates a path of beliefs π with length T"

            π = np.empty(T+1)

            # initial condition
            π[0] = 0.5

            for t in range(1, T+1):
                π[t] = update(a, b, π[t-1])

            return π

        def simulate(a=1, b=1, T=50, N=200, display=True):
            "Simulates N paths of beliefs π with length T"

            π_paths = np.empty((N, T+1))
            if display:
                fig = plt.figure()

            for i in range(N):
                π_paths[i] = simulate_path(a=a, b=b, T=T)
                if display:
                    plt.plot(range(T+1), π_paths[i], color='b', lw=0.8, alpha=0.5)

            if display:
                plt.show()

            return π_paths

        return simulate

.. code-block:: python3

    simulate = function_factory()


We begin by generating :math:`N` simulated :math:`\{\pi_t\}` paths with :math:`T`
periods when the sequence is truly IID draws from :math:`F`. We set the initial prior :math:`\pi_{-1} = .5`.

.. code-block:: python3

    T = 50


.. code-block:: python3

    # when nature selects F
    π_paths_F = simulate(a=1, b=1, T=T, N=1000)


In the above graph we observe that  for most paths :math:`\pi_t \rightarrow 1`. So Bayes' Law evidently eventually
discovers the truth for most of our paths.

Next, we generate paths with :math:`T`
periods when the sequence is truly IID draws from :math:`G`. Again, we set the initial prior :math:`\pi_{-1} = .5`.


.. code-block:: python3

    # when nature selects G
    π_paths_G = simulate(a=3, b=1.2, T=T, N=1000)

In  the above graph we observe that now  most paths :math:`\pi_t \rightarrow 0`.    


Rates of convergence 
^^^^^^^^^^^^^^^^^^^^^

We study rates of  convergence of :math:`\pi_t` to :math:`1` when nature generates the data as IID draws from :math:`F`
and of :math:`\pi_t` to :math:`0` when nature generates the data as IID draws from :math:`G`.


We do this by averaging across simulated paths of :math:`\{\pi_t\}_{t=0}^T`.

Using   :math:`N` simulated :math:`\pi_t` paths, we compute
:math:`1 - \sum_{i=1}^{N}\pi_{i,t}` at each :math:`t` when the data are generated as draws from  :math:`F`
and compute :math:`\sum_{i=1}^{N}\pi_{i,t}` when the data are generated as draws from :math:`G`.

.. code-block:: python3

    plt.plot(range(T+1), 1 - np.mean(π_paths_F, 0), label='F generates')
    plt.plot(range(T+1), np.mean(π_paths_G, 0), label='G generates')
    plt.legend()
    plt.title("convergence");

From the above graph, rates of convergence appear not to depend on whether :math:`F` or :math:`G` generates the data.


Another Graph of Population Dynamics of :math:`\pi_t`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

More insights about the dynamics of :math:`\{\pi_t\}` can be gleaned by computing the following
conditional expectations of :math:`\frac{\pi_{t+1}}{\pi_{t}}` as functions of :math:`\pi_t` via integration with respect
to the pertinent probability distribution:

.. math::


   \begin{aligned}
   E\left[\frac{\pi_{t+1}}{\pi_{t}}\biggm|q=\omega, \pi_{t}\right] &=E\left[\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\biggm|q=\omega, \pi_{t}\right], \\
       &=\int_{0}^{1}\frac{l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}\omega\left(w_{t+1}\right)dw_{t+1}
   \end{aligned}

where :math:`\omega=f,g`.

The following code approximates the integral above:

.. code-block:: python3

    def expected_ratio(F_a=1, F_b=1, G_a=3, G_b=1.2):

        # define f and g
        f = njit(lambda x: p(x, F_a, F_b))
        g = njit(lambda x: p(x, G_a, G_b))

        l = lambda w: f(w) / g(w)
        integrand_f = lambda w, π: f(w) * l(w) / (π * l(w) + 1 - π)
        integrand_g = lambda w, π: g(w) * l(w) / (π * l(w) + 1 - π)

        π_grid = np.linspace(0.02, 0.98, 100)

        expected_rario = np.empty(len(π_grid))
        for q, inte in zip(["f", "g"], [integrand_f, integrand_g]):
            for i, π in enumerate(π_grid):
                expected_rario[i]= quad(inte, 0, 1, args=(π,))[0]
            plt.plot(π_grid, expected_rario, label=f"{q} generates")

        plt.hlines(1, 0, 1, linestyle="--")
        plt.xlabel("$π_t$")
        plt.ylabel("$E[\pi_{t+1}/\pi_t]$")
        plt.legend()

        plt.show()

First, consider the case where :math:`F_a=F_b=1` and
:math:`G_a=3, G_b=1.2`.

.. code-block:: python3

    expected_ratio()

The above graphs shows that when :math:`F` generates the data, :math:`\pi_t` on average always heads north, while 
when :math:`G` generates the data, :math:`\pi_t` heads south. 



Next, we'll look at a degenerate case in whcih  :math:`f` and :math:`g` are identical beta
distributions, and :math:`F_a=G_a=3, F_b=G_b=1.2`. 

In a sense, here  there
is nothing to learn.

.. code-block:: python3

    expected_ratio(F_a=3, F_b=1.2)

The above graph says that :math:`\pi_t` is inert and would remain at its initial value.

Finally, let's look at a case in which  :math:`f` and :math:`g` are neither very
different nor identical, in particular one in which  :math:`F_a=2, F_b=1` and
:math:`G_a=3, G_b=1.2`.

.. code-block:: python3

    expected_ratio(F_a=2, F_b=1, G_a=3, G_b=1.2)
