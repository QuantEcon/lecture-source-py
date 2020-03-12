.. _likelihood_ratio_process:

.. include:: /_static/includes/header.raw

.. highlight:: python3

****************************
Likelihood Ratio Processes
****************************

.. contents:: :depth: 2


.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    from numba import vectorize, njit
    from math import gamma
    %matplotlib inline


Overview
=========
This lecture describes likelihood ratio processes and some of their uses by frequentist and Bayesian statisticians.

We'll use the simple statistical setting also used in :doc:`this lecture <exchangeable>`.

Among the things that we'll learn about are

   * A peculiar property of likelihood ratio processes

   * How for a Bayesian  a likelihood ratio process and a prior probability determine a posterior probability

   * How a likelihood ratio process is the key ingredient in frequentist hypothesis testing

   * How a **receiver operator characteristic curve** summarizes information about a false alarm probability and power in frequentist hypothesis testing 



Likelihood Ratio Process
========================  


A nonnegative random variable :math:`W` has one of two probability density functions, either
:math:`f` or :math:`g`.

Before the beginning of time, nature once and all decides whether she will draw a sequence of i.i.d. draws from either
:math:`f` or :math:`g`.

We will sometimes let :math:`q` be the density that nature chose once and for all, so
that :math:`q` is either :math:`f` or :math:`g`, permanently.

Nature knows which density it permanently draws from, but we the observers don't know.

We do know both :math:`f` and :math:`g` but we don’t know which density nature
chose.

But we want to know.

To do that, we use observations.

We observe a sequence :math:`\{w_t\}_{t=1}^T` of :math:`T` i.i.d. draws 
from either :math:`f` or :math:`g`.

We want to use these observations to infer whether nature chose :math:`f` or
:math:`g` before the beginning of time. 

A **likelihood ratio process** is a useful tool for this task. 

To begin, we define the key component of a likelihood ratio process, namely, the time :math:`t` likelihood ratio  as the random variable

.. math::


   l\left(w_{t}\right)=\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)},\quad t\geq1.


We assume that :math:`f` and :math:`g` both put positive probabilities on the
same intervals of possible realizations of :math:`w_t`.

That means that under the :math:`g` density,  :math:`l\left(w_{t}\right)=\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}`
is evidently a nonnegative  random variable with mean :math:`1`.     


A **likelihood ratio process** for sequence
:math:`\left\{ l\left(w_{t}\right)\right\} _{t=1}^{\infty}` is defined as

.. math::


   L\left(w^{t}\right)=\prod_{i=1}^{t}l\left(w_{i}\right).

where :math:`w^{t}=\left\{ w_{1},\dots,w_{t}\right\}` is a history of
observations up to and including time :math:`t`.

Sometimes for shorthand we'll write :math:`L_t =  L\left(w^{t}\right)`.

Notice that the likelihood process satisfies the *recursion* or
*multiplicative decomposition*

.. math::


   L\left(w^t\right) = l\left(w_t\right) L\left(w^{t-1}\right) .

The likelihood ratio and its logarithm are key tools for making
inferences using a classic frequentist approach due to Neyman and
Pearson :cite:`Neyman_Pearson`.

To help us appreciate how things work, the following Python code evaluates :math:`f` and :math:`g` as different
beta distributions, then computes and simulates an associated likelihood
ratio process by generating a sequence :math:`w^t` from **some**
probability distribution, for example, a sequence of  i.i.d. draws from :math:`g`.

.. code-block:: python3

    F_a, F_b = 1, 1
    G_a, G_b = 3, 1.2

    @vectorize
    def p(x, a, b):
        r = gamma(a + b) / (gamma(a) * gamma(b))
        return r * x** (a-1) * (1 - x) ** (b-1)

    f = njit(lambda x: p(x, F_a, F_b))
    g = njit(lambda x: p(x, G_a, G_b))

.. code-block:: python3

    @njit
    def simulate(a, b, T=50, N=500):

        l_arr = np.empty((N, T))

        for i in range(N):

            for j in range(T):
                w = np.random.beta(a, b)
                l_arr[i, j] = f(w) / g(w)

        return l_arr

Nature Permanently Draws from Density g
=============================================

We first simulate the likelihood ratio process when nature permanently
draws from :math:`g`.

.. code-block:: python3

    l_arr_g = simulate(G_a, G_b)
    l_seq_g = np.cumprod(l_arr_g, axis=1)

.. code-block:: python3

    N, T = l_arr_g.shape

    for i in range(N):

        plt.plot(range(T), l_seq_g[i, :], color='b', lw=0.8, alpha=0.5)

    plt.ylim([0, 3])
    plt.title("$L(w^{t})$ paths");

Evidently, as sample length :math:`T` grows, most probability mass
shifts toward zero

To see it this more clearly clearly, we plot over time the fraction of
paths :math:`L\left(w^{t}\right)` that fall in the interval
:math:`\left[0, 0.01\right]`.

.. code-block:: python3

    plt.plot(range(T), np.sum(l_seq_g <= 0.01, axis=0) / N)

Despite the evident convergence of virtually all probability mass to a
very small interval near :math:`0`, a peculiar fact about a likelihood
ratio process is that the unconditional mean of
:math:`L\left(w^t\right)` under probability density :math:`g` is
identically :math:`1` for all :math:`t`.

To verify this assertion, first notice that as mentioned earlier the unconditional mean
:math:`E_{0}\left[l\left(w_{t}\right)\bigm|q=g\right]` is :math:`1` for
all :math:`t`:

.. math::


   \begin{aligned}
   E_{0}\left[l\left(w_{t}\right)\bigm|q=g\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
       &=\int f\left(w_{t}\right)dw_{t} \\
       &=1,
   \end{aligned}

which immediately implies

.. math::


   \begin{aligned}
   E_{0}\left[L\left(w^{1}\right)\bigm|q=g\right]  &=E_{0}\left[l\left(w_{1}\right)\bigm|q=g\right]\\
       &=1.\\
   \end{aligned}

Because :math:`L(w^t)` is a multiplicative process and
:math:`\{w_t\}_{t=1}^t` is an i.i.d. sequence, we have

.. math::


   \begin{aligned}
   E_{0}\left[L\left(w^{t}\right)\bigm|q=g\right]  &=E_{0}\left[L\left(w^{t-1}\right)l\left(w_{t}\right)\bigm|q=g\right] \\
       &=E_{0}\left[L\left(w^{t-1}\right)E\left[l\left(w_{t}\right)\bigm|q=g,w^{t-1}\right]\bigm|q=g\right] \\
       &=E_{0}\left[L\left(w^{t-1}\right)E\left[l\left(w_{t}\right)\bigm|q=g\right]\bigm|q=g\right] \\
       &=E_{0}\left[L\left(w^{t-1}\right)\bigm|q=g\right] \\
   \end{aligned}

for any :math:`t \geq 1`.

Mathematical induction implies
:math:`E_{0}\left[L\left(w^{t}\right)\bigm|q=g\right]=1` for all
:math:`t \geq 1`.

Peculiar Property of Likelihood Ratio Process
-----------------------------------------------

How can this possibly be true if probability mass of the likelihood
ratio process is piling up near :math:`1` as
:math:`t \rightarrow + \infty`?

The answer has to be that as :math:`t \rightarrow + \infty`, the
distribution of :math:`L_t` becomes more and more thin-tailed as
sufficient small mass shifts to very large values of :math:`L_t` to make
the mean of :math:`L_t` continue to be zero despite the mass piling up
near :math:`0`.

To illustrate this peculiar property, we simulate many paths and
calculate the unconditional mean of :math:`L\left(w^t\right)` by
averaging across these many paths at each :math:`t`.

.. code-block:: python3

    l_arr_g = simulate(G_a, G_b, N=50000)
    l_seq_g = np.cumprod(l_arr_g, axis=1)

The following Python code approximates the unconditional means
:math:`E_{0}\left[L\left(w^{t}\right)\right]` by averaging across sample
paths.

Please notice that while  sample averages  hover around their population means of :math:`1`, there is quite a bit
of variability, a consequence of the *fat tail* of the distribution of  :math:`L\left(w^{t}\right)`.

.. code-block:: python3

    N, T = l_arr_g.shape
    plt.plot(range(T), np.mean(l_arr_g, axis=0))
    plt.hlines(1, 0, T, linestyle='--')

Nature Permanently Draws from Density f
==========================================

Now suppose that before time :math:`0` nature permanently decided to draw repeatedly from density :math:`f`.

A useful property is that while the mean of the likelihood ratio :math:`l\left(w_{t}\right)` under density 
:math:`g` is :math:`1`, its mean under the density :math:`f` exceeds one.  

To see this, we compute 

.. math::


   \begin{aligned}
   E_{0}\left[l\left(w_{t}\right)\bigm|q=f\right]  &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}f\left(w_{t}\right)dw_{t} \\
       &=\int\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}\frac{f\left(w_{t}\right)}{g\left(w_{t}\right)}g\left(w_{t}\right)dw_{t} \\
       &=\int l\left(w_{t}\right)^{2}g\left(w_{t}\right)dw_{t} \\
       &=E_{0}\left[l\left(w_{t}\right)^{2}\mid q=g\right] \\
       &=E_{0}\left[l\left(w_{t}\right)\mid q=g\right]^{2}+Var\left(l\left(w_{t}\right)\mid q=g\right) \\
       &>E_{0}\left[l\left(w_{t}\right)\mid q=g\right]^{2} \\
       &=1 \\
   \end{aligned}


This in turn implies that the unconditional mean of the likelihood ratio process :math:`L(w^t)` 
diverges toward :math:`+ \infty`.

Simulations below confirm this conclusion.

Please note the scale of the :math:`y` axis.

.. code-block:: python3

    l_arr_f = simulate(F_a, F_b, N=50000)
    l_seq_f = np.cumprod(l_arr_f, axis=1)

.. code-block:: python3

    N, T = l_arr_f.shape
    plt.plot(range(T), np.mean(l_seq_f, axis=0))

We also plot the probability that :math:`L\left(w^t\right)` falls into
the interval :math:`[10000, \infty)` as a function of time and watch how
fast probability mass diverges  to :math:`+\infty`.

.. code-block:: python3

    plt.plot(range(T), np.sum(l_seq_f > 10000, axis=0) / N)

Likelihood Ratio Process and Bayes’ Law
==========================================


Let :math:`\pi_t` be a Bayesian posterior defined as

.. math::  \pi_t = {\rm Prob}(q=f|w^t)

The likelihood ratio process is a principal actor in the formula that governs the evolution
of the posterior probability :math:`\pi_t`, an instance of **Bayes' Law**.

Bayes’ law implies that :math:`\{\pi_t\}` obeys the recursion

.. math::
   :label: eq_recur1

   \pi_t=\frac{\pi_{t-1} l_t(w_t)}{\pi_{t-1} l_t(w_t)+1-\pi_{t-1}}

with :math:`\pi_{0}` be a Bayesian prior probability that :math:`q = f`,
i.e., a belief about :math:`q` based on having seen no data.

Below we define a Python function that updates belief :math:`\pi` using
likelihood ratio :math:`l` according to  recursion :eq:`eq_recur1`

.. code-block:: python3

    @njit
    def update(π, l):
        "Update π using likelihood l"

        # Update belief
        π = π * l / (π * l + 1 - π)

        return π

Formula :eq:`eq_recur1` can be generalized in a useful way.

We do this by iterating on recursion :eq:`eq_recur1` in order to derive an expression for  the time :math:`t` posterior 
:math:`\pi_{t+1}` as a function of the time :math:`0` prior :math:`\pi_0` and the likelihood ratio process
:math:`L(w^{t+1})` at time :math:`t`.

To begin, notice that the updating rule

.. math::


   \pi_{t+1}=\frac{\pi_{t}l\left(w_{t+1}\right)}{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}

implies

.. math::


   \begin{aligned}
   \frac{1}{\pi_{t+1}} &=\frac{\pi_{t}l\left(w_{t+1}\right)+\left(1-\pi_{t}\right)}{\pi_{t}l\left(w_{t+1}\right)} \\
       &=1-\frac{1}{l\left(w_{t+1}\right)}+\frac{1}{l\left(w_{t+1}\right)}\frac{1}{\pi_{t}}.
   \end{aligned}

.. math::


   \Rightarrow\frac{1}{\pi_{t+1}}-1=\frac{1}{l\left(w_{t+1}\right)}\left(\frac{1}{\pi_{t}}-1\right).

Therefore

.. math::


   \begin{aligned}
   \frac{1}{\pi_{t+1}}-1   =\frac{1}{\prod_{i=1}^{t+1}l\left(w_{i}\right)}\left(\frac{1}{\pi_{0}}-1\right)
       =\frac{1}{L\left(w^{t+1}\right)}\left(\frac{1}{\pi_{0}}-1\right).
   \end{aligned}

Since :math:`\pi_{0}\in\left(0,1\right)` and
:math:`L\left(w^{t+1}\right)>0`, we can verify that
:math:`\pi_{t+1}\in\left(0,1\right)`.

After rearranging the preceding equation, we can express :math:`\pi_{t+1}` as a
function of  :math:`L\left(w^{t+1}\right)`, the  likelihood ratio process at :math:`t+1`,
and the initial prior :math:`\pi_{0}`

.. math::
   :label: eq_Bayeslaw103

   \pi_{t+1}=\frac{\pi_{0}L\left(w^{t+1}\right)}{\pi_{0}L\left(w^{t+1}\right)+1-\pi_{0}} .

Formula :eq:`eq_Bayeslaw103` generalizes generalizes formula :eq:`eq_recur1`.

Formula :eq:`eq_Bayeslaw103`  can be regarded as a one step  revision of prior probability :math:`\pi_0` after seeing
the batch of data :math:`\left\{ w_{i}\right\} _{i=1}^{t+1}`.

Formula :eq:`eq_Bayeslaw103` shows the key role that the likelihood ratio process  :math:`L\left(w^{t+1}\right)` plays in determining
the posterior probability :math:`\pi_{t+1}`.

Formula :eq:`eq_Bayeslaw103` is the foundation for the insight that, because of the way the likelihood ratio process behaves
as :math:`t \rightarrow + \infty`, the likelihood ratio process dominates the initial prior :math:`\pi_0` in determining the 
limiting behavior of :math:`\pi_t`.

To illustrate this insight, below we will plot  graphs showing **one** simulated
path of the  likelihood ratio process :math:`L_t` along with two paths of
:math:`\pi_t` that are associated with the *same* realization of the likelihood ratio process but *different* initial prior probabilities
probabilities :math:`\pi_{0}`.

First, we specify the two values of :math:`\pi_0`.

.. code-block:: python3

    π1, π2 = 0.2, 0.8

Next we generate paths of the likelihood ratio process :math:`L_t` and the posteior :math:`\pi_t` for a history drawn as i.i.d.
draws from density :math:`f`.

.. code-block:: python3

    T = l_arr_f.shape[1]
    π_seq_f = np.empty((2, T+1))
    π_seq_f[:, 0] = π1, π2

    for t in range(T):
        for i in range(2):
            π_seq_f[i, t+1] = update(π_seq_f[i, t], l_arr_f[0, t])

.. code-block:: python3

    fig, ax1 = plt.subplots()

    for i in range(2):
        ax1.plot(range(T+1), π_seq_f[i, :], label=f"$\pi_0$={π_seq_f[i, 0]}")

    ax1.set_ylabel("$\pi_t$")
    ax1.set_xlabel("t")
    ax1.legend()
    ax1.set_title("when f governs data")

    ax2 = ax1.twinx()
    ax2.plot(range(1, T+1), np.log(l_seq_f[0, :]), '--', color='b')
    ax2.set_ylabel("$log(L(w^{t}))$")

    plt.show()


The dotted line in the graph above records the logarithm of the  likelihood ratio process :math:`\log L(w^t)`.


Please note that there are two different scales on the :math:`y` axis.

Now let's study what happens when the history consists of i.i.d. draws from density :math:`g`


.. code-block:: python3

    T = l_arr_g.shape[1]
    π_seq_g = np.empty((2, T+1))
    π_seq_g[:, 0] = π1, π2

    for t in range(T):
        for i in range(2):
            π_seq_g[i, t+1] = update(π_seq_g[i, t], l_arr_g[0, t])

.. code-block:: python3

    fig, ax1 = plt.subplots()

    for i in range(2):
        ax1.plot(range(T+1), π_seq_g[i, :], label=f"$\pi_0$={π_seq_g[i, 0]}")

    ax1.set_ylabel("$\pi_t$")
    ax1.set_xlabel("t")
    ax1.legend()
    ax1.set_title("when g governs data")

    ax2 = ax1.twinx()
    ax2.plot(range(1, T+1), np.log(l_seq_g[0, :]), '--', color='b')
    ax2.set_ylabel("$log(L(w^{t}))$")

    plt.show()


Below we offer Python code that verifies this in a setting in which
nature chose permanently to draw from density :math:`f`.

.. code-block:: python3

    π_seq = np.empty((2, T+1))
    π_seq[:, 0] = π1, π2

    for i in range(2):
        πL = π_seq[i, 0] * l_seq_f[0, :]
        π_seq[i, 1:] = πL / (πL + 1 - π_seq[i, 0])

.. code-block:: python3

    np.abs(π_seq - π_seq_f).max() < 1e-10

Likelihood Ratio Test
======================

Having seen how the likelihood ratio process is a key ingredient of the formula :eq:`eq_Bayeslaw103` for 
a Bayesian's posteior probabilty that nature has drawn history :math:`w^t` as repeated draws from density 
:math:`g`, we now turn to how a frequentist statistician would employ the hypothesis testing theory
of Neyman and Pearson :cite:`Neyman_Pearson` to test the hypothesis that  history :math:`w^t` is generated by repeated
i.i.d. draws from density :math:`g`. 

Denote :math:`q` as the data generating process, so that
:math:`q=f \text{ or } g`.

Upon observing a sample :math:`\{W_i\}_{i=1}^t`, we want to figure out
which one is the data generating process by executing a frequentist
hypothesis test.

We specify

-  Null hypothesis :math:`H_0`: :math:`q=f`,
-  Alternative hypothesis :math:`H_1`: :math:`q=g`.

Neyman and Pearson proved that the best way to test this hospital is to use a **likelihood ratio test**:

-  reject :math:`H_0` if :math:`L(W^t) < c`,
-  accept :math:`H_0` otherwise.

where the discrimination threshold :math:`c` is arbitrarily given.

This test is *best* in the sense that it is the uniformly most powerful test.  

To understand what this means, we have to look at probabilities of two important events.

These two probabilities allow us to characterize a test associated with particular
threshold :math:`c`.

The two probabities are:

1. Probability of detection (= *statistical power* = 1 minus probability
   of Type II error):

.. math::


   1-\beta \equiv \Pr\left\{ L\left(w^{t}\right)<c\mid q=g\right\}

2. Probability of false alarm (= *significance level* = probability of
   Type I error):

.. math::


   \alpha \equiv  \Pr\left\{ L\left(w^{t}\right)<c\mid q=f\right\}

The `Neyman-Pearson
Lemma <https://en.wikipedia.org/wiki/Neyman–Pearson_lemma>`__
states that among all possible tests, the likelihood ratio test
maximizes the probability of detection for a given probability of false
alarm.

To have made a confident inference, we want a small probability of
false alarm and a large probability of detection.

With sample size :math:`t` fixed, we can change our two probabilities by
moving :math:`c`.

A troublesome "that's life" fact is that these two probabilities  move in the same direction as we vary the critical value
:math:`c`.

Without specifying the penalties on making Type I and Type II errors, there is little that we can say 
about how we *should*  trade off probabilities of the two types of mistakes.

What we do know that increasing sample size :math:`t` improves
statistical inference.

Below we plot some informative figures that display showing this.

We also present a classical frequentist method for choosing a sample
size :math:`t` that provides confident inferences.

Let’s start with a case where we fix the threshold :math:`c` at
:math:`1`.

.. code-block:: python3

    c = 1

Below we plot empirical distributions of logarithms of the cumulative
likelihood ratios simulated above, which are generated by either
:math:`f` or :math:`g`.

Taking logarithms has no effect on calculating the probabilities because
the log  is a monotonic transformation.

As :math:`t` increases, the probabilities of making Type I and Type II
errors both decrease, which is good.

This is because most of the probability mass of log\ :math:`(L(w^t))`
moves toward :math:`-\infty` when :math:`g` is the data generating
process, ; while log\ :math:`(L(w^t))` goes to
:math:`\infty` when data are generated by :math:`f`.

This diverse behavior is what makes it possible to distinguish
:math:`q=f` from :math:`q=g`.

.. code-block:: python3

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('distribution of $log(L(w^t))$ under f or g', fontsize=15)

    for i, t in enumerate([1, 7, 14, 21]):
        nr = i // 2
        nc = i % 2

        axs[nr, nc].axvline(np.log(c), color="k", ls="--")

        hist_f, x_f = np.histogram(np.log(l_seq_f[:, t]), 200, density=True)
        hist_g, x_g = np.histogram(np.log(l_seq_g[:, t]), 200, density=True)

        axs[nr, nc].plot(x_f[1:], hist_f, label="dist under f")
        axs[nr, nc].plot(x_g[1:], hist_g, label="dist under g")

        for i, (x, hist, label) in enumerate(zip([x_f, x_g], [hist_f, hist_g], ["Type I error", "Type II error"])):
            ind = x[1:] <= np.log(c) if i == 0 else x[1:] > np.log(c)
            axs[nr, nc].fill_between(x[1:][ind], hist[ind], alpha=0.5, label=label)

        axs[nr, nc].legend()
        axs[nr, nc].set_title(f"t={t}")

    plt.show()

The graph below shows more clearly that, when we hold the threshold
:math:`c` fixed, the probability of detection monotonically increases in
:math:`t` and that the probability of a false alarm moves in the opposite
direction.

.. code-block:: python3

    PD = np.empty(T)
    PFA = np.empty(T)

    for t in range(T):
        PD[t] = np.sum(l_seq_g[:, t] < c) / N
        PFA[t] = np.sum(l_seq_f[:, t] < c) / N

    plt.plot(range(T), PD, label="Probability of detection")
    plt.plot(range(T), PFA, label="Probability of false alarm")
    plt.xlabel("t")
    plt.title("$c=1$")
    plt.legend()
    plt.show()

Notice that the threshold :math:`c` uniquely pins down  probabilities
of both types of error.

If we now free up :math:`c` and move it, we will sweep out the probability
of detection as a function of the probability of false alarm.

This produces what is called a `receiver operating characteristic
curve <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`__.

Below, we plot receiver operating characteristic curves for different
sample sizes :math:`t`.

.. code-block:: python3

    PFA = np.arange(0, 100, 1)

    for t in range(1, 15, 4):
        percentile = np.percentile(l_seq_f[:, t], PFA)
        PD = [np.sum(l_seq_g[:, t] < p) / N for p in percentile]

        plt.plot(PFA / 100, PD, label=f"t={t}")

    plt.scatter(0, 1, label="perfect detection")
    plt.plot([0, 1], [0, 1], color='k', ls='--', label="random detection")

    plt.arrow(0.5, 0.5, -0.15, 0.15, head_width=0.03)
    plt.text(0.35, 0.7, "better")
    plt.xlabel("Probability of false alarm")
    plt.ylabel("Probability of detection")
    plt.legend()
    plt.title("Receiver Operating Characteristic Curve")
    plt.show()

Notice that as :math:`t` increases, we are assured a larger probability
of detection and a smaller probability of false alarm associated with
a given discrimination threshold :math:`c`.

As :math:`t \rightarrow + \infty`, we approach the the perfect detection
curve that is indicated by a right angle hinging on the green dot.

It is up to the test user to decide how to trade off probabilities of
making two types of errors.

But we know how to choose the smallest sample size given any targets for
the probabilities.

Typically, frequentists aim for a high probability of detection that
respects an upper bound on the probability of false alarm.

Below we show a case in which we fix the probability of false alarm at
:math:`0.05`.

The required sample size for making a decision is then determined by a
target probability of detection, for example, :math:`0.9`.

.. code-block:: python3

    PFA = 0.05
    PD = np.empty(T)

    for t in range(T):

        c = np.percentile(l_seq_f[:, t], PFA * 100)
        PD[t] = np.sum(l_seq_g[:, t] < c) / N

    plt.plot(range(T), PD)
    plt.axhline(0.9, color="k", ls="--")

    plt.xlabel("t")
    plt.ylabel("Probability of detection")
    plt.title(f"Probability of false alarm={PFA}")
    plt.show()
