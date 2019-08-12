.. _hs_recursive_models:

.. include:: /_static/includes/header.raw

.. index::
    single: python


********************************************
Recursive Models of Dynamic Linear Economies
********************************************

.. contents:: :depth: 2

.. epigraph::

    "Mathematics is the art of giving the same name to different things" -- Henri Poincare

.. epigraph::

    "Complete market economies are all alike" --     Robert E. Lucas, Jr., (1989)


.. epigraph::

    "Every partial equilibrium model can be reinterpreted as a general equilibrium model." --   Anonymous


A Suite of Models
=================

This lecture presents  a class of  linear-quadratic-Gaussian models of general economic equilibrium
designed by Lars Peter Hansen and Thomas J. Sargent :cite:`HS2013`.

The class of models is implemented in a Python class DLE that is part of  quantecon.

Subsequent lectures use the DLE class to implement various instances that have appeared in the economics literature



  1. :doc:`Growth in Dynamic Linear Economies <growth_in_dles>`

  2. :doc:`Lucas Asset Pricing using DLE <lucas_asset_pricing_dles>`

  3. :doc:`IRFs in Hall Model <irfs_in_hall_model>`

  4. :doc:`Permanent Income Using the DLE class <permanent_income_dles>`

  5. :doc:`Rosen schooling model <rosen_schooling_model>`

  6. :doc:`Cattle cycles <cattle_cycles>`

  7. :doc:`Shock Non Invertibility <hs_invertibility_example>`



Overview of the Models
----------------------

In saying that "complete markets are all alike", Robert E. Lucas, Jr. was noting  that all of them have

- a commodity space.

- a space dual to the commodity space in which prices reside.

- endowments of resources.

- peoples' preferences over goods.

- physical technologies for transforming resources into goods.

- random processes that govern shocks to technologies and preferences and associated information flows.

- a single budget constraint per person.

- the existence of a representative consumer even when there are many people in the model.

- a concept of competitive equilibrium.

- theorems connecting competitive equilibrium allocations to allocations that would be chosen by a benevolent social planner.


The models have **no frictions** such as :math:`\ldots`

-  Enforcement difficulties

-  Information asymmetries

-  Other forms of transactions costs

-  Externalities

The models extensively use the  powerful ideas of


-  Indexing commodities and their prices by time (John R. Hicks).

-  Indexing commodities and their prices by chance (Kenneth Arrow).


Much of  the imperialism of complete markets models comes from applying these two tricks.

The Hicks trick of indexing commodities by time is the idea that **dynamics are a special case of statics**.

The Arrow trick of indexing commodities by chance is the idea that **analysis of trade under uncertainty is a special
case of the analysis of trade under certainty**.

The :cite:`HS2013` class of models specify the commodity space, preferences, technologies, stochastic shocks and information flows in ways
that allow the models to be analyzed completely using only the tools of linear time series models and linear-quadratic optimal control described
in the two lectures  :doc:`Linear State Space Models<linear_models>` and :doc:`Linear Quadratic Control<lqcontrol>`.

There are costs and benefits associated with the simplifications and specializations needed to make a particular model fit within the
:cite:`HS2013` class

- the costs are that  linear-quadratic structures are sometimes too confining.

- benefits include computational speed, simplicity, and ability to analyze many model features analytically or nearly analytically.


A variety of superficially different models are all instances of the :cite:`HS2013` class of models


-  Lucas asset pricing model

-  Lucas-Prescott model of investment under uncertainty

-  Asset pricing models with habit persistence

-  Rosen-Topel equilibrium model of housing

-  Rosen schooling models

-  Rosen-Murphy-Scheinkman model of cattle cycles

-  Hansen-Sargent-Tallarini model of robustness and asset pricing

-  Many more :math:`\ldots`

.. _section-1:

The diversity of these models conceals an essential unity that illustrates the quotation by Robert E. Lucas, Jr., with which
we began this lecture.


Forecasting?
------------

A consequence of a single budget constraint per person  plus the Hicks-Arrow tricks is that households and firms need not forecast.

But there exist equivalent structures called **recursive competitive equilibria** in which  they do appear to need to forecast.

In these structures, to forecast, households and firms use:

- equilibrium pricing functions, and

- knowledge of the  Markov structure of the economy’s state vector.

Theory and Econometrics
-----------------------

For an application of the  :cite:`HS2013` class of models, the outcome of theorizing is a stochastic process, i.e., a probability
distribution over sequences of prices and quantities, indexed by  parameters describing preferences, technologies, and information flows.

Another name for that object is a likelihood function, a key object of both frequentist and Bayesian statistics.

There are two important uses of an **equilibrium stochastic process** or **likelihood function**.

The first is to solve the **direct problem**.

The **direct problem** takes as inputs values of the parameters that define preferences, technologies, and information flows and as
an output characterizes or simulates random paths of quantities and prices.

The second use of an equilibrium stochastic process or likelihood function is to solve the **inverse problem**.

The **inverse problem** takes as an input a time series sample of observations on a subset of prices and quantities determined by the model
and from them makes inferences about the parameters that define the model's preferences, technologies, and information flows.


More Details
------------

A :cite:`HS2013` economy consists of **lists of matrices** that describe peoples’ household technologies, their preferences over
consumption services,  their production technologies, and their information sets.

There are complete markets in history-contingent commodities.

Competitive equilibrium allocations and prices

- satisfy equations that are easy to write down and solve

- have representations that are convenient econometrically

Different example economies manifest themselves simply as different settings for various matrices.

:cite:`HS2013` use these tools:

-  A theory of recursive dynamic competitive economies

-  Linear optimal control theory

-  Recursive methods for estimating and interpreting vector
   autoregressions


The models are flexible enough to express alternative senses of a representative household

-  A single ‘stand-in’ household of the type used to good effect by Edward C. Prescott.

-  Heterogeneous households satisfying conditions for Gorman aggregation
   into a representative household.

-  Heterogeneous household technologies that  violate conditions for Gorman
   aggregation but are still susceptible to aggregation into a single
   representative household via ‘non-Gorman' or 'mongrel' aggregation’.

These three alternative types of aggregation have different consequences in terms of how  prices and allocations can be computed.

In particular, can prices and an aggregate allocation be computed before the
equilibrium allocation to individual heterogeneous households is computed?

-  Answers are “Yes” for Gorman aggregation, “No” for non-Gorman
   aggregation.


In summary, the insights and practical benefits from economics to be introduced in this lecture
are

-  Deeper understandings that come from recognizing common underlying
   structures.

-  Speed and ease of computation that comes from unleashing a common suite of Python programs.

We'll use the following **mathematical tools**


-  Stochastic Difference Equations (Linear).

-  Duality: LQ Dynamic Programming and Linear Filtering  are the same things
   mathematically.

-  The Spectral Factorization Identity (for understanding vector
   autoregressions and non-Gorman aggregation).

So here is our roadmap.

We'll describe sets of matrices that pin down

-  Information

-  Technologies

-  Preferences

Then we'll describe

-  Equilibrium concept and computation

-  Econometric representation and estimation

.. _section-3:

Stochastic Model of Information Flows and Outcomes
--------------------------------------------------

We'll use  stochastic linear difference equations to describe information flows **and** equilibrium outcomes.


The sequence :math:`\{w_t : t=1,2, \ldots\}` is said to be a martingale
difference sequence adapted to :math:`\{J_t : t=0, 1, \ldots \}` if
:math:`E(w_{t+1} \vert J_t) = 0` for :math:`t=0, 1, \ldots\,`.

The sequence :math:`\{w_t : t=1,2,\ldots\}` is said to be conditionally
homoskedastic if :math:`E(w_{t+1}w_{t+1}^\prime \mid J_t) = I` for
:math:`t=0,1, \ldots\,`.

We assume that the :math:`\{w_t : t=1,2,\ldots\}` process is
conditionally homoskedastic.

.. _linear-stochastic-difference-equations-1:



Let :math:`\{x_t : t=1,2,\ldots\}` be a sequence of
:math:`n`-dimensional random vectors, i.e. an :math:`n`-dimensional
stochastic process.


The process :math:`\{x_t : t=1,2,\ldots\}` is constructed recursively
using an initial random vector
:math:`x_0\sim {\mathcal N}(\hat x_0, \Sigma_0)` and a time-invariant
law of motion:

.. math::

    x_{t+1} = Ax_t + Cw_{t+1}

for :math:`t=0,1,\ldots`  where :math:`A` is an :math:`n` by :math:`n` matrix and :math:`C` is an
:math:`n` by :math:`N` matrix.


Evidently, the distribution of :math:`x_{t+1}` conditional on :math:`x_t` is
:math:`{\mathcal N}(Ax_t, CC')`.

Information Sets
----------------

Let :math:`J_0` be generated by :math:`x_0` and :math:`J_t` be generated
by :math:`x_0, w_1, \ldots ,
w_t`, which means that :math:`J_t` consists of the set of all measurable
functions of :math:`\{x_0, w_1,\ldots,
w_t\}`.

Prediction Theory
-----------------

The optimal forecast of :math:`x_{t+1}` given current information is

.. math::

    E(x_{t+1} \mid J_t) = Ax_t

and the one-step-ahead forecast error is

.. math::

    x_{t+1} - E(x_{t+1} \mid J_t) = Cw_{t+1}

The covariance matrix of :math:`x_{t+1}` conditioned on :math:`J_t` is



.. math::

    E (x_{t+1} - E ( x_{t+1} \mid J_t) ) (x_{t+1} - E ( x_{t+1} \mid J_t))^\prime  = CC^\prime

A nonrecursive expression for :math:`x_t` as a function of
:math:`x_0, w_1, w_2, \ldots,  w_t` is

.. math::

   \begin{aligned}
    x_t &= Ax_{t-1} + Cw_t \\
   &= A^2 x_{t-2} + ACw_{t-1} + Cw_t \\
   &= \Bigl[\sum_{\tau=0}^{t-1} A^\tau Cw_{t-\tau} \Bigr] + A^t x_0 \end{aligned}

.. _prediction-theory-1:


Shift forward in time:

.. math::

    x_{t+j} = \sum^{j-1}_{s=0} A^s C w_{t+j-s} + A^j x_t

Projecting on the information set :math:`\{ x_0, w_t, w_{t-1},
\ldots, w_1\}` gives

.. math:: E_t x_{t+j} = A^j x_t

where :math:`E_t (\cdot) \equiv  E [ (\cdot) \mid x_0, w_t, w_{t-1}, \ldots, w_1]
= E (\cdot) \mid J_t`, and :math:`x_t` is in :math:`J_t`.

.. _prediction-theory-2:




It is useful to obtain the covariance matrix of the :math:`j`-step-ahead
prediction error :math:`x_{t+j} - E_t x_{t+j} = \sum^{j-1}_{s=0} A^s C w_{t-s+j}`.

Evidently,

.. math::

   E_t (x_{t+j} - E_t x_{t+j})  (x_{t+j} - E_t x_{t+j})^\prime =
   \sum^{j-1}_{k=0} A^k C C^\prime A^{k^\prime} \equiv v_j

:math:`v_j` can be calculated recursively via

.. math::

   \begin{aligned}
    v_1 &= CC^\prime \\
    v_j &= CC^\prime + A v_{j-1} A^\prime, \quad j \geq 2 \end{aligned}


Orthogonal Decomposition
------------------------

To decompose these covariances into parts attributable to the individual
components of :math:`w_t`, we let :math:`i_\tau` be an
:math:`N`-dimensional column vector of zeroes except in position
:math:`\tau`, where there is a one. Define a matrix
:math:`\upsilon_{j,\tau}`

.. math::

   \upsilon_{j,\tau} = \sum_{k=0}^{j-1} A^k C i_\tau i_\tau^\prime C^\prime
   A^{^\prime k} .

Note that :math:`\sum_{\tau=1}^N i_\tau i_\tau^\prime = I`, so that we
have

.. math:: \sum_{\tau=1}^N \upsilon_{j, \tau} = \upsilon_j

Evidently, the matrices
:math:`\{ \upsilon_{j, \tau} , \tau = 1, \ldots, N \}` give an
orthogonal decomposition of the covariance matrix of
:math:`j`-step-ahead prediction errors into the parts attributable to
each of the components :math:`\tau =
1, \ldots, N`.

Taste and Technology Shocks
---------------------------

:math:`E(w_t \mid J_{t-1}) = 0` and :math:`E(w_t
w_t^\prime \mid J_{t-1}) = I` for :math:`t=1,2, \ldots`

.. math::

    b_t = U_b z_t \hbox{ and } d_t = U_dz_t,

:math:`U_b` and :math:`U_d` are matrices that select entries of
:math:`z_t`. The law of motion for :math:`\{z_t : t=0, 1, \ldots\}` is

.. math::

    z_{t+1} = A_{22} z_t + C_2 w_{t+1} \ \hbox { for } t = 0, 1, \ldots

where :math:`z_0` is a given initial condition. The eigenvalues of the
matrix :math:`A_{22}` have absolute values that are less than or equal
to one.


Thus, in summary, our model of **information and shocks** is

.. math::

   \begin{aligned}
    z_{t+1} &=A_{22} z_t + C_2 w_{t+1}
   \\  b_t &= U_b z_t \\ d_t &= U_d z_t .\end{aligned}


We can now briefly summarize other components of our economies, in particular


-  Production technologies

-  Household technologies

-  Household preferences




Production Technology
----------------------

Where :math:`c_t` is a vector of consumption rates, :math:`k_t` is a vector of physical capital goods, :math:`g_t` is
a vector intermediate productions goods, :math:`d_t` is a vector of technology shocks, the production technology is

.. math::

   \begin{aligned}
    \Phi_c c_t +  \Phi_g g_t + \Phi_i i_t &=\Gamma k_{t-1} + d_t \\
   k_t &=\Delta_k k_{t-1} + \Theta_k i_t \\ g_t \cdot g_t &=\ell_t^2 \end{aligned}

Here :math:`\Phi_c, \Phi_g, \Phi_i, \Gamma, \Delta_k, \Theta_k` are all matrices conformable to the vectors they multiply and
:math:`\ell_t` is a disutility generating resource supplied by the household.


For technical reasons that facilitate computations, we make the following.

**Assumption:** :math:`[\Phi_c\ \Phi_g]` is nonsingular.



Household Technology
---------------------

Households confront a technology that allows them to devote consumption goods to construct a vector :math:`h_t` of household capital goods
and a vector :math:`s_t` of utility generating  house services

.. math::

   \begin{aligned}
    s_t &=  \Lambda h_{t-1} + \Pi c_t \\
    h_t &= \Delta_h h_{t-1} + \Theta_h c_t \end{aligned}

where :math:`\Lambda, \Pi, \Delta_h, \Theta_h` are matrices that pin down the household technology.

We make the following

**Assumption:** The absolute values of the eigenvalues of :math:`\Delta_h`
are less than or equal to one.

Below, we'll outline further assumptions that we shall occasionally impose.

Preferences
------------

Where :math:`b_t` is a stochastic process of preference shocks that will play the role of demand shifters, the representative household orders
stochastic processes of consumption services :math:`s_t` according to


.. math::

   \Bigl( {1 \over 2}\Bigr)  E \sum_{t=0}^\infty \beta^t [ (s_t -
   b_t) \cdot ( s_t - b_t) + \ell_t^2 ] \bigl| J_0 , \ 0 < \beta < 1


We now proceed to give examples of production and household technologies that appear in various models that appear in the literature.

First, we give examples of production Technologies


.. math:: \Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t

.. math:: \mid g_t \mid \leq \ell_t

so we'll be looking for specifications of the matrices :math:`\Phi_c, \Phi_g, \Phi_i, \Gamma, \Delta_k, \Theta_k` that define them.

Endowment Economy
------------------

There is a single consumption good that cannot be stored over time.

In time period :math:`t`, there is an endowment :math:`d_t` of this single
good.

There is neither a capital stock, nor an intermediate good, nor a
rate of investment.

So :math:`c_t = d_t`.

To implement this specification, we can choose :math:`A_{22}, C_2`, and
:math:`U_d` to make :math:`d_t` follow any of a variety of stochastic
processes.

To satisfy our earlier rank assumption, we set:

.. math::

    c_t + i_t = d_{1t}

.. math::

    g_t = \phi_1 i_t

where :math:`\phi_1` is a small positive number.

To implement this
version, we set :math:`\Delta_k = \Theta_k = 0` and

.. math::

   \Phi_c =  \begin{bmatrix}  1 \\ 0 \\ \end{bmatrix},
   \Phi_i = \begin{bmatrix} 1 \\ \phi_1 \\ \end{bmatrix} , \ \ \Phi_g =
   \begin{bmatrix} 0 \\ -1 \\ \end{bmatrix},  \ \ \Gamma = \begin{bmatrix}
    0 \\ 0 \end{bmatrix},  \ \ d_t  = \begin{bmatrix} d_{1t}
   \\ 0 \end{bmatrix}

We can use this specification to create a linear-quadratic version of
Lucas’s (1978) asset pricing model.

Single-Period Adjustment Costs
-------------------------------

There is a single consumption good, a single intermediate good, and a
single investment good.

The technology is described by

.. math::

   \begin{aligned}
   c_t &=\gamma k_{t-1} + d_{1t} ,\ \ \gamma > 0 \\
   \phi_1 i_t &= g_t + d_{2t}, \ \ \phi_1 > 0 \\
   \ell^2_t &=  g^2_t \\
   k_t &= \delta_k k_{t-1} + i_t ,\ 0< \delta_k < 1 \end{aligned}

Set

.. math::

   \Phi_c = \begin{bmatrix}1 \\ 0 \end{bmatrix} ,\ \Phi_g = \begin{bmatrix}0 \\
   -1 \end{bmatrix}, \ \Phi_i = \begin{bmatrix} 0 \\ \phi_1 \end{bmatrix}

.. math::

   \Gamma = \begin{bmatrix} \gamma \\ 0 \end{bmatrix}, \ \Delta_k = \delta_k,
   \ \Theta_k = 1

We set :math:`A_{22}, C_2` and :math:`U_d` to make
:math:`(d_{1t}, d_{2t})^\prime = d_t` follow a desired stochastic
process.



Now we describe some examples of preferences, which as we have seen are ordered by

.. math::

   -\left({1 \over 2}\right) E \sum^\infty_{t=0} \beta^t \left[ (s_t - b_t) \cdot (s_t -
   b_t) + (\ell_t)^2 \right] \mid J_0 \quad ,\ 0 < \beta < 1

where household services are produced via the household technology


.. math:: h_t = \Delta_h h_{t-1} + \Theta_h c_t

.. math:: s_t = \Lambda h_{t-1} + \Pi c_t

and we make

**Assumption:** The absolute values of the eigenvalues of :math:`\Delta_h`
are less than or equal to one.

Later we shall introduce **canonical**  household technologies that  satisfy an ‘invertibility’
requirement relating sequences :math:`\{s_t\}` of services and
:math:`\{c_t\}` of consumption flows.


And  we’ll describe how to obtain a canonical representation
of a household technology from one that is not canonical.

Here are some examples of household preferences.

**Time Separable preferences**

.. math::

   -{1\over 2} E \sum^\infty_{t=0} \beta^t \left[ (c_t - b_t)^2 + \ell_t^2
   \right] \mid J_0 \quad ,\ 0 < \beta < 1

**Consumer Durables**


.. math:: h_t = \delta_h h_{t-1} + c_t \quad ,\ 0 < \delta_h < 1

Services at :math:`t` are related to the stock of durables at the
beginning of the period:

.. math:: s_t = \lambda h_{t-1} \ , \ \lambda > 0

Preferences are ordered by

.. math::

   -{1 \over 2} E \sum^\infty_{t=0} \beta^t \left[(\lambda h_{t-1} -
   b_t)^2 + \ell_t^2\right] \mid J_0

Set :math:`\Delta_h = \delta_h,
\Theta_h =1, \Lambda = \lambda, \Pi = 0`.

**Habit Persistence**


.. math::

   -\Bigl({1\over 2}\Bigr)\, E \sum^\infty_{t=0} \beta^t \Bigl[\bigl(c_t - \lambda
    (1-\delta_h) \sum^\infty_{j=0}\, \delta^j_h\, c_{t-j-1}-b_t\bigr)^2+\ell^2_t\Bigl]  \bigl| J_0

.. math::

    0<\beta < 1\ ,\ 0 < \delta_h < 1\ ,\ \lambda > 0

Here the effective bliss point :math:`b_t + \lambda (1 - \delta_h)
\sum^\infty_{j=0} \delta^j_h\, c_{t-j-1}` shifts in response to a moving
average of past consumption.


**Initial Conditions**

Preferences of this form require an initial condition for the geometric
sum :math:`\sum^\infty_{j=0} \delta_h^j c_{t - j-1}` that we specify as
an initial condition for the ‘stock of household durables,’
:math:`h_{-1}`.

.. _habit-persistence-1:


Set

.. math:: h_t = \delta_h h_{t-1} + (1-\delta_h) c_t \quad ,\ 0 < \delta_h < 1

.. math::

   h_t = (1 - \delta_h) \sum^t_{j=0} \delta_h^j\, c_{t-j} + \delta^{t+1}_h\,
   h_{-1}

.. math:: s_t = - \lambda h_{t-1} + c_t, \ \lambda > 0

To implement, set
:math:`\Lambda = -\lambda,\ \Pi = 1,\ \Delta_h = \delta_h,\ \Theta_h=1-\delta_h`.

**Seasonal Habit Persistence**


.. math::

   -\Bigl({1\over 2}\Bigr) \, E \sum^\infty_{t=0} \beta^t  \Bigl[\bigl(c_t - \lambda
    (1-\delta_h) \sum^\infty_{j=0}\, \delta^{j}_h\, c_{t-4j-4}-b_t\bigr)^2+\ell^2_t\Bigr]

.. math::

    0<\beta < 1\ ,\ 0 < \delta_h < 1\ ,\ \lambda > 0

Here the effective bliss point :math:`b_t + \lambda (1 - \delta_h) \sum^\infty_{j=0} \delta^j_h\, c_{t-4j-4}` shifts in response to a
moving average of past consumptions of the same quarter.




To implement, set

.. math::

    \tilde h_t = \delta_h \tilde h_{t-4} + (1-\delta_h) c_t \quad ,\ 0 < \delta_h < 1

This implies that

.. math::

    h_t = \begin{bmatrix}\tilde h_t \\
          \tilde h_{t-1}\\
          \tilde h_{t-2}\\
          \tilde  h_{t-3}\end{bmatrix}  =
          \begin{bmatrix} 0 & 0 & 0 & \delta_h \\
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \end{bmatrix}
                    \begin{bmatrix} \tilde h_{t-1} \\ \tilde h_{t-2} \\ \tilde h_{t-3} \\ \tilde h_{t-4} \end{bmatrix}
                    + \begin{bmatrix}(1 - \delta_h) \\ 0 \\ 0 \\ 0 \end{bmatrix} c_t

with consumption services

.. math::

    s_t = - \begin{bmatrix}0 & 0 & 0 & -\lambda\end{bmatrix}  h_{t-1} + c_t \quad , \ \lambda > 0

**Adjustment Costs**.


Recall


.. math::

   -\Bigl({1 \over 2}\Bigr) E \sum^\infty_{t=0} \beta^t [(c_t - b_{1t})^2 +
   \lambda^2 (c_t - c_{t-1})^2 + \ell^2_t ] \mid J_0

.. math:: 0 < \beta < 1 \quad, \ \lambda > 0

To capture adjustment costs, set

.. math:: h_t  = c_t

.. math::

   s_t = \begin{bmatrix} 0 \\ - \lambda \end{bmatrix} h_{t-1} +
   \begin{bmatrix} 1 \\ \lambda \end{bmatrix} c_t

so that

.. math:: s_{1t} = c_t

.. math:: s_{2t} = \lambda (c_t - c_{t-1} )

We set the first component :math:`b_{1t}` of :math:`b_t` to capture the
stochastic bliss process and set the second component identically equal
to zero.

Thus, we set :math:`\Delta_h = 0, \Theta_h = 1`

.. math::

   \Lambda = \begin{bmatrix} 0 \\ -\lambda \end{bmatrix}\ ,\ \Pi =
   \begin{bmatrix} 1 \\ \lambda \end{bmatrix}

**Multiple Consumption Goods**


.. math::

   \Lambda = \begin{bmatrix} 0\\0\end{bmatrix} \ \hbox { and } \ \Pi =
   \begin{bmatrix}\pi_1 & 0 \\ \pi_2 & \pi_3 \end{bmatrix}

.. math:: -{1 \over 2} \beta^t (\Pi c_t - b_t)^\prime (\Pi c_t - b_t)

.. math:: mu_t= - \beta^t [\Pi^\prime \Pi\, c_t - \Pi^\prime\, b_t]

.. math::

   c_t = - (\Pi^\prime \Pi)^{-1} \beta^{-t} mu_t + (\Pi^\prime \Pi)^{-1}
   \Pi^\prime b_t

This is called the **Frisch demand function** for consumption.


We can think of the vector :math:`mu_t` as playing the role of prices,
up to a common factor, for all dates and states.

The scale factor is
determined by the choice of numeraire.


Notions of **substitutes and complements** can be defined in terms of these
Frisch demand functions.

Two goods can be said to be **substitutes** if the
cross-price effect is positive and to be **complements** if this effect is
negative.

Hence this classification is determined by the off-diagonal
element of :math:`-(\Pi^\prime \Pi)^{-1}`, which is equal to
:math:`\pi_2 \pi_3 /\det
(\Pi^\prime \Pi)`.

If :math:`\pi_2` and :math:`\pi_3` have the same
sign, the goods are substitutes.

If they have opposite signs, the goods
are complements.


To summarize, our economic structure consists of the matrices that define the following components:

**Information and shocks**

.. math::

   \begin{aligned}
   z_{t+1} &=A_{22} z_t + C_2 w_{t+1}
   \\  b_t &= U_b z_t \\ d_t &= U_d z_t \end{aligned}

**Production Technology**

.. math::

   \begin{aligned}
   \Phi_c c_t &+ \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t \\
   k_t &=\Delta_k k_{t-1} + \Theta_k i_t \\ g_t \cdot g_t &=\ell_t^2 \end{aligned}

**Household Technology**

.. math::

   \begin{aligned}
   s_t &=
   \Lambda h_{t-1} + \Pi c_t \\ h_t &= \Delta_h h_{t-1} + \Theta_h c_t \end{aligned}

**Preferences**

.. math::

   \Bigl( {1 \over 2}\Bigr)  E \sum_{t=0}^\infty \beta^t [ (s_t -
   b_t) \cdot ( s_t - b_t) + \ell_t^2 ] \bigl| J_0 , \ 0 < \beta < 1


*Next steps:** we move on to discuss two closely connected concepts

-  A Planning Problem or Optimal Resource Allocation Problem

-  Competitive Equilibrium


Optimal Resource Allocation
----------------------------

Imagine a planner who chooses sequences :math:`\{c_t, i_t, g_t\}_{t=0}^\infty` to maximize


.. math::

   -(1/2)E \sum_{t=0}^\infty \beta^t [ (s_t - b_t) \cdot (s_t - b_t) + g_t \cdot g_t ] \bigl| J_0

subject to the constraints

.. math::

   \begin{aligned}
    \Phi_c c_t &+ \Phi_g \, g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t,
   \\
   k_t &= \Delta_k k_{t-1} + \Theta_k i_t , \\
   h_t &= \Delta_h h_{t-1} + \Theta_h c_t , \\
   s_t &=\Lambda h_{t-1} + \Pi c_t , \\
     z _{t+1} &= A_{22} z_t + C_2 w_{t+1} , \ b_t = U_b z_t,  \  \hbox{ and } \
   d_t = U_d z_t \end{aligned}

and initial conditions for :math:`h_{-1}, k_{-1}`, and :math:`z_0`.


Throughout, we shall impose the following **square summability** conditions



.. math::

   E \sum^\infty_{t=0} \beta^t h_t \cdot h_t \mid J_0 < \infty\ \hbox { and }\
   E \sum^\infty_{t=0} \beta^t k_t \cdot k_t \mid J_0 < \infty

Define:

.. math::

   L_0^2 = [ \{ y_t \}  : y_t \ \hbox{is a random
   variable in } \  J_t \ \hbox{ and } \  E \sum_{t=0}^\infty \beta^t
   y_t^2 \mid J_0 < + \infty]

Thus, we require that each component of :math:`h_t` and each component of
:math:`k_t` belong to :math:`L_0^2`.


We shall compare and utilize two approaches to solving the planning problem

-  Lagrangian formulation

-  Dynamic programming

Lagrangian Formulation
------------------------

Form the Lagrangian

.. math::

   \begin{aligned}
    {\mathcal L} &= - E \sum_{t=0}^\infty \beta^t \biggl[
   \Bigl( {1 \over 2} \Bigr) [ (s_t - b_t) \cdot (s_t - b_t) + g_t
   \cdot g_t] \\ &+   {\cal M}_t^{d \prime} \cdot ( \Phi _cc_t  +
   \Phi_gg_t + \Phi_ii_t - \Gamma k_{t-1} - d_t ) \\ &+{\cal M}_t^{k
   \prime} \cdot (k_t - \Delta_k k_{t-1} - \Theta_k i_t ) \\ &+ {\cal
   M}_t^{h \prime} \cdot (h_t - \Delta_h h_{t-1} - \Theta_h c_t) \\ &+
   {\cal M}_t^{s \prime} \cdot (s_t - \Lambda h_{t-1} - \Pi c_t )
   \biggr] \Bigl| J_0 \end{aligned}

The planner maximizes :math:`{\mathcal L}` with respect to the quantities :math:`\{c_t, i_t, g_t\}_{t=0}^\infty`
and minimizes with respect to the Lagrange multipliers :math:`{\cal M}_t^d, {\cal M}_t^k, {\cal M}_t^h, {\cal M}_t^s`.

First-order necessary conditions for maximization with respect to
:math:`c_t, g_t,
h_t, i_t, k_t`, and :math:`s_t`, respectively, are:

.. math::

   \begin{aligned}
    - \Phi_c^\prime  {\cal M}_t^d &+\Theta_h^\prime {\cal
   M}_t^h + \Pi^\prime {\cal M}_t^s = 0 , \\
   &- g_t - \Phi_g^\prime  {\cal M}_t^d = 0 , \\
   - {\cal M}_t^h &+ \beta E ( \Delta_h^\prime {\cal M}^h_{t+1} +
   \Lambda^\prime {\cal M}_{t+1}^s ) \mid J_t = 0 , \\
   &- \Phi_i^\prime {\cal M}_t^d + \Theta_k^\prime {\cal M}_t^k = 0 , \\
   - {\cal M}_t^k &+ \beta E ( \Delta_k^\prime {\cal M}^k_{t+1} + \Gamma^\prime
   {\cal M}_{t+1}^d) \mid J_t = 0 , \\
   &- s_t + b_t - {\cal M}_t^s = 0 \end{aligned}

for :math:`t=0,1, \ldots`.

In addition, we have the complementary
slackness conditions (these recover the original transition equations)
and also transversality conditions

.. math::

   \begin{aligned}
    \lim_{t \to \infty}& \beta^t  E [ {\cal M}_t^{k \prime} k_t ]
   \mid J_0 = 0  \\
    \lim_{t \to \infty}& \beta^t   E [ {\cal M}_t^{h \prime} h_t ]
   \mid J_0 = 0\end{aligned}



The system formed by the FONCs and the transition equations can be  handed over to Python.

Python will solve the planning problem for fixed parameter values.


Here are the **Python  Ready Equations**

.. math::

   \begin{aligned}
    - \Phi_c^\prime  {\cal M}_t^d &+\Theta_h^\prime {\cal
   M}_t^h + \Pi^\prime {\cal M}_t^s = 0 , \\
   &- g_t - \Phi_g^\prime  {\cal M}_t^d = 0 , \\
   - {\cal M}_t^h &+ \beta E ( \Delta_h^\prime {\cal M}^h_{t+1} +
   \Lambda^\prime {\cal M}_{t+1}^s ) \mid J_t = 0 , \\
   &- \Phi_i^\prime {\cal M}_t^d + \Theta_k^\prime {\cal M}_t^k = 0 , \\
   - {\cal M}_t^k &+ \beta E ( \Delta_k^\prime {\cal M}^k_{t+1} + \Gamma^\prime
   {\cal M}_{t+1}^d) \mid J_t = 0 , \\
   &- s_t + b_t - {\cal M}_t^s = 0 \\
   \Phi_c c_t &+ \Phi_g \, g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t,
   \\
   k_t &= \Delta_k k_{t-1} + \Theta_k i_t , \\
   h_t &= \Delta_h h_{t-1} + \Theta_h c_t , \\
   s_t &=\Lambda h_{t-1} + \Pi c_t , \\
     z _{t+1} &= A_{22} z_t + C_2 w_{t+1} , \ b_t = U_b z_t,  \  \hbox{ and } \
   d_t = U_d z_t  \end{aligned}



The Lagrange multipliers or **shadow prices** satisfy

.. math:: {\cal M}_t^s = b_t - s_t

.. math::

   {\cal M}_t^h = E \biggl[ \sum_{\tau =1}^\infty \beta^\tau
   (\Delta_h^\prime)^{\tau - 1} \Lambda^\prime {\cal M}_{t+ \tau}^s
   \mid J_t\biggr]

.. math::

   {\cal M}_t^d = \begin{bmatrix}\Phi_c^\prime\\ \Phi_g^\prime\\\end{bmatrix}^{-1}\
   \begin{bmatrix} \Theta_h^\prime {\cal M}_t^h  + \Pi^\prime {\cal M}_t^s \\
   -g_t \end{bmatrix}

.. math::

   {\cal M}_t^k = E \biggl[\sum _{\tau=1}^\infty \beta^\tau (\Delta_k^\prime)^{\tau-1}
   \Gamma^\prime {\cal M}_{t+ \tau}^d \mid J_t\biggr]

.. math::

   {\cal M}_t^i =
   \Theta _k^\prime {\cal M}_t^k

Although it is possible to use matrix operator methods to solve the above **Python ready equations**, that is not
the approach we'll use.

Instead, we'll use dynamic programming to get recursive representations for both quantities and shadow prices.


Dynamic Programming
--------------------

Dynamic Programming  always starts with the word **let**.

Thus, let :math:`V(x_0)` be the optimal value function for the  planning problem as a function of the initial state vector :math:`x_0`.

(Thus, in essence, dynamic programming amounts to an application of a **guess and verify** method in which we begin with a guess about the answer
to the problem we want to solve. That's why we start with **let** :math:`V(x_0)` be the (value of the) answer to the problem, then establish and
verify a bunch of conditions :math:`V(x_0)` has to satisfy if indeed it is the answer)

The optimal value function :math:`V(x)` satisfies the **Bellman equation**


.. math::

   V(x_0) = \max_{c_0, i_0, g_0} [ - .5 [ (s_0 - b_0) \cdot (s_0 - b_0) + g_0 \cdot g_0 ] + \beta  E V
   (x_1) ]

subject to the linear constraints

.. math::

   \begin{aligned}
    \Phi _cc_0 &+ \Phi_g g_0 + \Phi_ii_0 = \Gamma k_{-1} + d_0 ,\\
   k_0 &= \Delta_k k_{-1} + \Theta_k i_0 , \\
   h_0 &= \Delta_h h_{-1} + \Theta_h c_0 , \\
   s_0 &= \Lambda h_{-1} + \Pi c_0 , \\
    z_1 &= A_{22} z_0 + C_2 w_1,\ b_0 = U_b z_0 \ \hbox{ and }\  d_0 =
   U_d z_0 \end{aligned}

Because this is a  linear-quadratic dynamic programming problem, it turns out that the value function has the form

.. math:: V(x) = x' P x + \rho

.. _dynamic-programming-1:

Thus, we want to solve  an instance of the following linear-quadratic dynamic programming problem:

Choose a contingency plan for :math:`\{x_{t+1}, u_t \}_{t=0}^\infty` to
maximize

.. math::

   - E \sum_{t=0}^\infty  \beta^t [ x_t^\prime  R x_t + u_t^\prime  Q
   u_t + 2 u_t^\prime  W' x_t ], \ 0 < \beta < 1

subject to

.. math:: x_{t+1} = A x_t + B u_t + C w_{t+1}, \ t \geq 0

where :math:`x_0` is given; :math:`x_t` is an :math:`n \times 1` vector
of state variables, and :math:`u_t` is a :math:`k \times 1` vector of
control variables.

We assume :math:`w_{t+1}` is a martingale difference
sequence with :math:`E w_t w_t^\prime = I`, and that :math:`C` is a
matrix conformable to :math:`x` and :math:`w`.

The optimal value function :math:`V(x)` satisfies the Bellman equation

.. math::

   V (x_t) = \max_{u_t} \Bigl\{-( x_t^\prime R x_t + u_t^\prime Q u_t + 2
   u_t^\prime W x_t) + \beta E_t V (x_{t+1}) \Bigr\}

where maximization is subject to

.. math:: x_{t+1} = A x_t + B u_t + C w_{t+1}, \ t \geq 0

.. math:: V(x_t) = - x_t^\prime P x_t - \rho

:math:`P` satisfies

.. math::

   P =  R + \beta A^\prime P A - (\beta A^\prime P
   B + W)   (Q + \beta B^\prime P B)^{-1} (\beta B^\prime P
   A + W')

This equation in :math:`P` is called the **algebraic matrix Riccati
equation**.

The optimal decision rule is :math:`u_t = - F x_t`, where

.. math::

   F = (Q + \beta B^\prime P B)^{-1} (\beta B^\prime P A +
   W')

The optimum decision rule for :math:`u_t` is independent of the
parameters :math:`C`, and so of the noise statistics.



Iterating on the Bellman operator leads to

.. math::

   V_{j+1} (x_t) = \max_{u_t} \Bigl\{-( x_t^\prime R x_t + u_t^\prime Q
   u_t + 2 u_t^\prime W x_t) + \beta E_t V_j (x_{t+1}) \Bigr\}

.. math:: V_j (x_t) =- x_t^\prime P_{j} x_t - \rho_{j}

where :math:`P_{j}` and :math:`\rho_{j}` satisfy the equations

.. math::

   \begin{aligned}
    P_{j+1} &= R + \beta A^\prime P_{j} A - (\beta
   A^\prime P_{j} B + W)  (Q + \beta B^\prime P_{j} B)^{-1} (\beta B^\prime P_{j}
   A + W')\\  \rho_{j+1} &=\beta \rho_{j} + \beta \ {\rm trace} \ P_{j} C C^\prime\end{aligned}


We can now state the planning problem as a dynamic programming problem


.. math::

   \max_{ \{u_t, x_{t+1}\} }\ - E \sum_{t=0}^\infty \beta^t [x_t^\prime
   Rx_t + u_t^\prime Q u_t + 2u_t^\prime W 'x_t ] , \quad 0 < \beta < 1

where  maximization is subject to

.. math:: x_{t+1} = Ax_t + B u_t + Cw_{t+1} , \ t \geq 0

.. math::

   x_t = \begin{bmatrix} h_{t-1} \\ k_{t-1} \\ z_t \end{bmatrix} , \qquad
    u_t = i_t

where


.. math::

   \begin{aligned}
    A &=\begin{bmatrix} \Delta_h & \Theta_h U_c [ \Phi_c \ \
   \Phi_g]^{-1} \Gamma & \Theta_h U_c [ \Phi_c \ \ \Phi_g]^{-1}  U_d \\ 0
   & \Delta_k & 0 \\ 0 & 0 & A_{22} \\  \end{bmatrix} \\
   B &= \begin{bmatrix} - \Theta_h U_c [ \Phi_c \ \ \Phi_g]^{-1} \Phi_i
   \\ \Theta_k \\ 0 \end{bmatrix}  \ ,\ C = \begin{bmatrix} 0 \\ 0 \\
   C_2 \end{bmatrix} \end{aligned}

.. math::

   \begin{bmatrix} x_t \\ u_t  \end{bmatrix}^\prime S \begin{bmatrix} x_t
   \\ u_t \end{bmatrix} = \begin{bmatrix} x_t \\ u_t \end{bmatrix}^\prime\ \
   \begin{bmatrix} R & W \\ W' & Q  \end{bmatrix}\ \ \begin{bmatrix} x_t
   \\ u_t \end{bmatrix}

.. math::
 
   S = (G^\prime G + H^\prime H) / 2

.. math::

   H = [\Lambda \ \vdots \ \Pi U_c [ \Phi_c \ \ \Phi_g]^{-1} \Gamma  \
   \vdots \ \Pi U_c [ \Phi_c \ \ \Phi_g]^{-1} U_d - U_b  \ \vdots \
    - \Pi U_c [\Phi_c \ \ \Phi_g]^{-1} \Phi_i]

.. math::

   G =
   U_g [ \Phi_c \ \ \Phi_g]^{-1} [0 \ \vdots \ \Gamma \ \vdots \ U_d \
   \vdots \ - \Phi_i] .


**Lagrange multipliers as gradient of value function**

A useful fact is that Lagrange multipliers equal gradients of the  planner’s value function


.. math::

   \begin{aligned}
   {\mathcal M}_t^k &= M_k x_t\ \hbox{ and }\ {\cal M}_t^h = M_h
   x_t \ \hbox{ where } \\
   M_k &= 2 \beta [ 0 \ I \ 0 ] P A^o  \\
   M_h &= 2 \beta [ I \ 0 \ 0 ] P A^o \end{aligned}

.. math::

   {\mathcal M}_t^s = M_s x_t \ \hbox{ where }\ M_s = (S_b -
   S_s)\ \hbox{ and } \ S_b = [ 0 \ 0 \ U_b ]

.. math::

   {\mathcal M}_t^d = M_d x_t\ \hbox{ where }\ M_d = \begin{bmatrix}
   \Phi_c^\prime \\ \Phi_g^\prime \\ \end{bmatrix} ^{-1}
   \begin{bmatrix}\Theta_h^\prime M_h + \Pi^\prime M_s \\ -S_g \end{bmatrix}

.. math::

   {\mathcal M}_t^c = M_c x_t\ \hbox{ where }\ M_c = \Theta_h^\prime
   M_h + \Pi^\prime M_s

.. math:: {\mathcal M}_t^i = M_i x_t\ \hbox{ where } \ M_i = \Theta_k^\prime M_k


We will use this fact and these equations to compute competitive equilibrium prices.

Other mathematical infrastructure
----------------------------------

Let's start with describing the **commodity space** and **pricing functional** for our competitive equilibrium.

For the  **commodity space**, we use

.. math::

   L_0^2 = [ \{ y_t \}  : y_t \ \hbox{is a random
   variable in } \  J_t \ \hbox{ and } \  E \sum_{t=0}^\infty \beta^t
   y_t^2 \mid J_0 < + \infty]

For **pricing functionals**, we express  values as inner products

.. math:: \pi (c) = E \sum_{t=0}^\infty \beta^t p_t^0 \cdot c_t \mid J_0

where :math:`p_t^0` belongs to :math:`L_0^2`.


With these objects in our toolkit, we move on to state the
problem of a **Representative Household in a competitive equilibrium**.


Representative Household
-------------------------


The representative household owns endowment process and initial stocks of :math:`h` and :math:`k` and
chooses stochastic processes for :math:`\{c_t,\, s_t,\, h_t,\,
\ell_t\}^\infty_{t=0}`, each element of which is in :math:`L^2_0`, to
maximize

.. math::

   -\ {1 \over 2}\ E_0 \sum^\infty_{t=0} \beta^t\, \Bigl[(s_t-b_t) \cdot (s_t -
   b_t) + \ell_t^2\Bigr]

subject to

.. math::

   E\sum^\infty_{t=0} \beta^t\, p^0_t \cdot c_t \mid J_0 = E \sum^\infty_{t=0}
   \beta^t\, (w^0_t \ell_t + \alpha^0_t \cdot d_t) \mid J_0 +
   v_0 \cdot k_{-1}

.. math:: s_t = \Lambda h_{t-1} + \Pi c_t

.. math::

   h_t = \Delta_h h_{t-1} + \Theta_h c_t, \quad h_{-1}, k_{-1}\
   \hbox{ given}


We now describe the problems faced by two types of firms called type I and type II.

Type I Firm
-------------

A type I firm rents capital and labor and endowments and produces
:math:`c_t, i_t`.

It chooses stochastic processes for
:math:`\{c_t, i_t, k_t, \ell_t,
g_t, d_t\}`, each element of which is in :math:`L^2_0`, to maximize

.. math::

   E_0\, \sum^\infty_{t=0} \beta^t\, (p^0_t \cdot c_t + q^0_t \cdot i_t - r^0_t
   \cdot k_{t-1} - w^0_t \ell_t - \alpha^0_t \cdot d_t)

subject to

.. math:: \Phi_c\, c_t + \Phi_g\, g_t + \Phi_i\, i_t = \Gamma k_{t-1} + d_t

.. math:: -\, \ell_t^2 + g_t \cdot g_t = 0

Type II Firm
-------------

A firm of type II  acquires capital via investment and then rents
stocks of capital to the :math:`c,i`-producing type I firm.

A type II firm is a price taker facing the vector :math:`v_0` and the stochastic
processes :math:`\{r^0_t, q^0_t\}`.

The firm chooses :math:`k_{-1}` and
stochastic processes for :math:`\{k_t, i_t\}^\infty_{t=0}` to maximize

.. math::

   E \sum^\infty_{t=0} \beta^t (r_t^0 \cdot k_{t-1} - q^0_t \cdot i_t) \mid
   J_0 - v_0 \cdot k_{-1}

subject to

.. math:: k_t = \Delta_k k_{t-1} + \Theta_k i_t


Competitive Equilibrium:  Definition
-------------------------------------

We can now state the following.

**Definition:** A competitive equilibrium is a price system
:math:`[v_0, \{p^0_t, w^0_t, \alpha^0_t, q^0_t, r^0_t\}^\infty_{t=0}]`
and an allocation :math:`\{c_t, i_t, k_t, h_t, g_t, d_t\}^\infty_{t=0}`
that satisfy the following conditions:

   * Each component of the price system and the allocation resides in the space :math:`L^2_0`.


   * Given the price system and given :math:`h_{-1},\, k_{-1}`, the allocation solves the representative household’s problem and
     the problems of the two types of firms.

Versions of the two classical welfare theorems prevail under our assumptions.

We exploit that fact in our algorithm for computing a competitive equilibrium.

**Step 1:** Solve the planning problem by using dynamic programming.

 The allocation (i.e., **quantities**) that solve the planning problem **are** the
 competitive equilibrium quantities.

**Step 2:** use the following formulas to compute the **equilibrium price system**


.. math::

   p^0_t = \bigl[\Pi^\prime {\cal M}^s_t + \Theta^\prime_h {\cal M}^h_t\bigr]/
   \mu^w_0 = {\cal M}^c_t / \mu^w_0

.. math:: w^0_t = \mid S_g x_t \mid / \mu^w_0

.. math:: r^0_t = \Gamma^\prime {\cal M}^d_t / \mu^w_0

.. math::

   q^0_t = \Theta^\prime_k {\cal M}^k_t / \mu^w_0 = {\cal M}^i_t /
   \mu^w_0

.. math:: \alpha^0_t =   {\cal M}^d_t / \mu^w_0

.. math::

   v_0 = \Gamma^\prime {\cal M}^d_0 / \mu^w_0 + \Delta^\prime_k
   {\cal M}^k_0 / \mu^w_0

**Verification:** With this price system, values can be assigned to the Lagrange
multipliers for each of our three classes of agents that cause all
first-order necessary conditions to be satisfied at these prices and at
the quantities associated with the optimum of the planning problem.


Asset pricing
-------------

An important use of an equilibrium pricing system is to do asset pricing.

Thus, imagine that we are presented a dividend stream: :math:`\{y_t\} \in L^2_0`
and want to compute the value of a perpetual claim to this stream.

To value this asset we simply take **price times quantity** and add to get
an asset value: :math:`a_0 =  E\, \sum_{t=0}^\infty\, \beta^t\ p_t^0 \cdot y_t \mid J_0`.

To compute :math:`ao` we proceed as follows.

We let

.. math:: y_t = U_a\, x_t

.. math:: a_0 = E \sum^\infty_{t=0}\, \beta^t\, x^\prime_t\, Z_a x_t \mid J_0

.. math:: Z_a = U^\prime_a M_c / \mu^w_0

We have the following convenient formulas:

.. math:: a_0 = x^\prime_0\, \mu_a\, x_0 + \sigma_a

.. math::

   \mu_a = \sum^\infty_{\tau=0}\, \beta^\tau\, (A^{o \prime})^\tau\ Z_a\,
   A^{o \tau}

.. math::

   \sigma_a = {\beta \over 1 - \beta}\ {\rm trace } \left( Z_a \sum^\infty_{\tau = 0}
   \,\beta^\tau\, (A^o)^\tau\, C C^\prime (A^{o \prime})^\tau \right)


Re-Opening Markets
-------------------

We have assumed that all trading occurs once-and-for-all at time :math:`t=0`.

If we were to **re-open markets** at some time :math:`t >0` at time :math:`t` wealth levels implicitly defined by
time :math:`0` trades, we would obtain the same equilibrium allocation (i.e., quantities) and the following time :math:`t`
price system


.. math::

   \begin{aligned}
    L^2_t &= [\{y_s\}^\infty_{s=t} : \ y_s \ \hbox{ is a random variable
   in }\ J_s\ \hbox{ for }\ s \geq t \\
   &\hbox {and } E\, \sum^\infty_{s=t}\, \beta^{s-t}\ y^2_s \mid J_t < + \infty] .\end{aligned}

.. math:: p^t_s = M_c x_s / [\bar e_j M_c x_t ], \qquad s \geq t

.. math:: w^t_s = \mid S_g x_s \vert / [\bar e_j M_c x_t], \ \ s \geq t

.. math:: r^t_s = \Gamma^\prime M_d x_s / [\bar e_j M_c x_t],\ \ s \geq t

.. math:: q^t_s = M_i x_s / [\bar e_j \, M_c x_t], \qquad s \geq t

.. math:: \alpha^t_s =   M_d x_s / [\bar e_j \, M_c x_t] , \ \ s \geq t

.. math:: v_t = [\Gamma^\prime M_d + \Delta^\prime_k M_k] x_t / \, [\bar e_j \, M_c x_t]

Econometrics
==============

Up to now, we have described how to solve the **direct problem** that maps model parameters into an (equilibrium) stochastic process
of prices and quantities.

Recall the **inverse problem** of inferring model parameters from a single realization of a time series of some of the prices and quantities.

Another name for the inverse problem is **econometrics**.

An advantage of the :cite:`HS2013` structure is that it comes with a self-contained theory of econometrics.

It is really just a tale of two state-space representations.


Here they are:



**Original State-Space Representation:**

.. math::

   \begin{aligned}
    x_{t+1} &= A^o x_t + Cw_{t+1} \\
   y_t & =  Gx_t + v_t\end{aligned}

where :math:`v_t` is a martingale difference sequence of measurement
errors that satisfies :math:`Ev_t
v_t' = R, E w_{t+1} v_s' = 0` for all :math:`t+1 \geq s` and

.. math:: x_0 \sim {\mathcal N}(\hat x_0,\Sigma_0)

**Innovations Representation:**

.. math::

   \begin{aligned}
   \hat x_{t+1} &=A^o \hat x_t + K_t a_t \\
   y_t &= G \hat x_t + a_t,\end{aligned}

where :math:`a_t = y_t - E[y_t | y^{t-1}], E a_t a_t^\prime \equiv \Omega_t =  G \Sigma_t G^\prime + R`.

.. _statistical-representations-1:



Compare numbers of shocks in the two representations:

  *  :math:`n_w + n_y` versus  :math:`n_y`

Compare spaces spanned


   * :math:`H(y^t) \subset H(w^t,v^t)`

   * :math:`H(y^t) = H(a^t)`



**Kalman Filter:**.

Kalman gain:

.. math:: K_t = A^o \Sigma_t G^\prime (G \Sigma_t G^\prime + R)^{-1}

Riccati Difference Equation:

.. math::

   \begin{aligned}
    \Sigma_{t+1} &= A^o \Sigma_t A^{o \prime} + CC^\prime \\
   &- A^o \Sigma_t G^\prime (G \Sigma_t G^\prime + R)^{-1} G \Sigma_t A^{o \prime}\end{aligned}

**Innovations Representation as Whitener**


Whitening Filter:

.. math::

   \begin{aligned}
    a_t &=y_t - G \hat x_t \\
   \hat x_{t+1} &=  A^o \hat x_t + K_t  a_t \end{aligned}

can be used recursively to construct a record of innovations
:math:`\{ a_t \}^T_{t=0}` from an :math:`(\hat x_0, \Sigma_0)` and a
record of observations :math:`\{ y_t \}^T_{t=0}`.

**Limiting Time-Invariant Innovations Representation**


.. math::

   \begin{aligned}
    \Sigma &= A^o \Sigma A^{o \prime} + CC^\prime \\
   &- A^o \Sigma G^\prime (G \Sigma G^\prime + R)^{-1} G \Sigma A^{o \prime} \\
    K &= A^o \Sigma_t G^\prime (G \Sigma G^\prime + R)^{-1}\end{aligned}

.. math::

   \begin{aligned}
    \hat x_{t+1} &= A^o \hat x_t + K a_t \\
   y_t &= G \hat x_t + a_t\end{aligned}

where :math:`E a_t a_t^\prime \equiv \Omega =  G \Sigma G^\prime + R`.

Factorization of Likelihood Function
--------------------------------------

Sample of observations :math:`\{y_s\}_{s=0}^T` on a
:math:`(n_y \times 1)` vector.

.. math::

   \begin{aligned}
    f(y_T, y_{T-1}, \ldots, y_0 )&=
         f_T(y_T \vert y_{T-1}, \ldots, y_0) f_{T-1}(y_{T-1} \vert
         y_{T-2}, \ldots, y_0) \cdots   f_1(y_1 \vert y_0)  f_0(y_0 )  \\
    &= g_T(a_T) g_{T-1} (a_{T-1}) \ldots g_1(a_1) f_0(y_0).\end{aligned}

Gaussian Log-Likelihood:

.. math::

   -.5 \sum_{t=0}^T \biggl\{ n_y \ln (2 \pi ) + \ln \vert \Omega _t \vert
         + a_t' \Omega_t^{-1} a_t \biggr\}

Covariance Generating Functions
--------------------------------

Autocovariance: :math:`C_x(\tau) = E x_t x_{t-\tau}'`.

Generating Function:
:math:`S_x(z) = \sum_{\tau = -\infty}^\infty C_x(\tau) z^\tau, z \in C`.

Spectral Factorization Identity
-------------------------------

Original state-space representation has too many shocks and implies:

.. math::

   S_y(z) = G (zI - A^o)^{-1} C C^\prime (z^{-1} I - (A^o)^\prime)^{-1}
   G^\prime + R

Innovations representation has as many shocks as dimension of
:math:`y_t` and implies

.. math::

   S_y(z) = [G(zI-A^o)^{-1}K +I] [G \Sigma G^\prime + R]
   [K^\prime (z^{-1} I -A^{o\prime})^{-1} G^\prime + I]



Equating these two leads to:

.. math::

   \begin{aligned}
    & G (zI -  A^o)^{-1} C C^\prime (z^{-1} I - A^{o\prime})^{-1} G^\prime + R = \\
   & [G(zI-A^o)^{-1}K +I] [G \Sigma G^\prime + R] [K'(z^{-1} I -A^{o\prime})^{-1}
   G^\prime + I] .\end{aligned}

**Key Insight:** The zeros of the polynomial
:math:`\det [G(zI-A^o)^{-1}K +I]` all lie inside the unit circle, which
means that :math:`a_t` lies in the space spanned by square summable
linear combinations of :math:`y^t`.

.. math:: H(a^t) = H(y^t)

**Key Property:** Invertibility


Wold and Vector Autoregressive Representations
--------------------------------------------------------------

Let's start with some lag operator arithmetic.

The lag operator :math:`L` and the inverse lag operator  :math:`L^{-1}` each map an infinite sequence into an infinite sequence according to the
transformation rules


.. math:: L x_t \equiv x_{t-1}

.. math:: L^{-1} x_t \equiv x_{t+1}


A **Wold moving average representation**  for :math:`\{y_t\}` is

.. math:: y_t = [ G(I-A^oL)^{-1}KL + I] a_t

Applying the inverse of the operator on the right side and using

.. math:: [G(I-A^oL)^{-1}KL+I]^{-1} = I - G[I - (A^o-KG)L]^{-1}KL

gives the **vector autoregressive representation**

.. math:: y_t = \sum_{j=1}^\infty G (A^o -KG)^{j-1} K y_{t-j} + a_t

.. _wold-and-vector-autoregressive-representations-1:




Dynamic Demand Curves and Canonical Household Technologies
===========================================================

Canonical Household Technologies
---------------------------------


.. math::

   \begin{aligned}
    h_t &=\Delta_h h_{t-1} + \Theta_h  c_t \\
                s_t &= \Lambda h_{t-1} + \Pi c_t  \\
                b_t  &=U_b z_t\end{aligned}

**Definition:** A household service technology
:math:`(\Delta_h, \Theta_h, \Pi,\Lambda, U_b)` is said to be **canonical**
if

  - :math:`\color{blue}{\Pi}` is nonsingular, and

  - the absolute values of the eigenvalues of :math:`\color{blue}{(\Delta_h - \Theta_h \Pi^{-1}\Lambda)}` are strictly less than :math:`1/\sqrt\beta`.


**Key invertibility property:** A canonical household service
technology maps a service process :math:`\{s_t\}` in :math:`L_0^2`
into a corresponding consumption process :math:`\{c_t\}` for which the
implied household capital stock process :math:`\{h_t\}` is also in
:math:`L^2_0`.

An inverse household technology:

.. math::

   \begin{aligned}
    c_t &= - \Pi^{-1} \Lambda h_{t-1} + \Pi^{-1} s_t\\
   h_t &= (\Delta_h - \Theta_h\Pi^{-1} \Lambda) h_{t-1} + \Theta_h \Pi^{-1}
   s_t \end{aligned}

The restriction  on the eigenvalues of the matrix
:math:`(\Delta_h - \Theta_h \Pi^{-1}
\Lambda)` keeps the household capital stock :math:`\{h_t\}` in
:math:`L_0^2`.

.. _canonical-household-technologies-3:




Dynamic Demand Functions
--------------------------

.. math::

   \rho^0_t \equiv \Pi^{-1 \prime} \Bigl[p^0_t - \Theta _h^\prime E_t
   \sum^\infty_{\tau=1} \beta^\tau (\Delta_h^\prime - \Lambda^\prime \Pi^{-1 \prime}
   \Theta_h^\prime )^{\tau-1}\Lambda^\prime \Pi^{-1 \prime} p^0_{t+\tau} \Bigr]

.. math::

   \begin{aligned}
   s_{i,t}&= \Lambda h_{i,t-1} \\
   h_{i,t}&= \Delta _h h_{i,t-1}\end{aligned}

where :math:`h_{i,-1} = h_{-1}`.

.. _dynamic-demand-functions-1:


.. math:: W_0 = E_0\sum^\infty_{t=0}\beta ^t(w^0_t\ell _t + \alpha ^0_t\cdot d_t) + v_0\cdot k_{-1}

.. math::

   \mu^w_0 = {E_0 \sum^\infty_{t=0} \beta^t \rho^0_t\cdot
    (b_t -s_{i,t}) - W_0 \over E_0 \sum^\infty_{t=0}
   \beta^t \rho^0_t \cdot \rho^0_t}

.. math::

   \begin{aligned}
    c_t &= -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t
    - \Pi^{-1} \mu_0^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h ' \\
   & \qquad [I - (\Delta_h ' - \Lambda ' \Pi^{\prime \, -1} \Theta_h ')\beta L^{-1}]
      ^{-1} \Lambda ' \Pi^{\prime -1} \beta L^{-1} \}  p_t^0  \\
       h_t &= \Delta_h h_{t-1} + \Theta_h c_t \end{aligned}



This system expresses consumption demands at date :math:`t` as
functions of: (i) time-\ :math:`t` conditional expectations of future
scaled Arrow-Debreu prices :math:`\{p_{t+s}^0\}_{s=0}^\infty`; (ii) the
stochastic process for the household’s endowment :math:`\{d_t\}` and
preference shock :math:`\{b_t\}`, as mediated through the multiplier
:math:`\mu_0^w` and wealth :math:`W_0`; and (iii) past values of
consumption, as mediated through the state variable :math:`h_{t-1}`.


Gorman Aggregation and Engel Curves
===================================

We shall explore how the dynamic demand schedule for consumption goods
opens up the possibility of satisfying Gorman’s (1953) conditions for
aggregation in a heterogeneous consumer model.

The first equation of our demand system is an Engel curve for consumption that is linear in the
marginal utility :math:`\mu_0^2` of individual wealth with a coefficient
on :math:`\mu_0^w` that depends only on prices.

The multiplier :math:`\mu_0^w` depends on wealth in an affine relationship, so that
consumption is linear in wealth.

In a model with multiple consumers who have the same household technologies
(:math:`\Delta_h, \Theta_h, \Lambda, \Pi`) but possibly different
preference shock processes and initial values of household capital
stocks, the coefficient on the marginal utility of wealth is the same
for all consumers.

Gorman showed that when Engel curves satisfy this
property, there exists a unique community or aggregate preference
ordering over aggregate consumption that is independent of the
distribution of wealth.

Re-Opened Markets
-------------------------------

.. math::

   \rho^t_{t} \equiv \Pi^{-1 \prime} \Bigl[p^t_{t} - \Theta _h^\prime E_t
   \sum^\infty_{\tau=1} \beta^\tau (\Delta_h^\prime - \Lambda^\prime \Pi^{-1 \prime}
   \Theta_h^\prime )^{\tau-1}\Lambda^\prime \Pi^{-1 \prime} p^t_{t+\tau} \Bigr]

.. math::

   \begin{aligned}
    s_{i,t}&= \Lambda h_{i,t-1} \\
   h_{i,t}&= \Delta _h h_{i,t-1},\end{aligned}

where now :math:`h_{i,t-1} = h_{t-1}`. Define time :math:`t` wealth
:math:`W_t`

.. math:: W_t = E_t\sum^\infty_{j=0}\beta ^j(w^t_{t+j}\ell_{t+j} + \alpha ^t_{t+j}\cdot d_{t+j}) + v_t\cdot k_{t-1}

.. math::

   \mu^w_t = {E_t \sum^\infty_{j=0} \beta^j \rho^t_{t+j}\cdot
    (b_{t+j} -s_{i,t+j}) - W_t \over E_t \sum^\infty_{t=0}
   \beta^j \rho^t_{t+j} \cdot \rho^t_{t+j}}

.. math::

   \begin{aligned}
    c_t &= -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t
    - \Pi^{-1} \mu_t^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h ' \\
    & \qquad [I - (\Delta_h ' - \Lambda ' \Pi^{\prime \, -1} \Theta_h ')\beta L^{-1}]
      ^{-1} \Lambda ' \Pi^{\prime -1} \beta L^{-1} \}  p_t^t  \\
       h_t &= \Delta_h h_{t-1} + \Theta_h c_t \end{aligned}

Dynamic Demand
-------------------------------

Define a time :math:`t` continuation of a sequence
:math:`\{z_t\}_{t=0}^\infty` as the sequence
:math:`\{z_\tau\}_{\tau=t}^\infty`. The demand system indicates that the
time :math:`t` vector of demands for :math:`c_t` is influenced by:

Through the multiplier :math:`\mu^w_t`, the time :math:`t` continuation
of the preference shock process :math:`\{b_t\}` and the time :math:`t`
continuation of :math:`\{s_{i,t}\}`.

The time :math:`t-1` level of household durables :math:`h_{t-1}`.

Everything that affects the household’s time :math:`t` wealth, including
its stock of physical capital :math:`k_{t-1}` and its value :math:`v_t`,
the time :math:`t` continuation of the factor prices
:math:`\{w_t, \alpha_t\}`, the household’s continuation endowment
process, and the household’s continuation plan for :math:`\{\ell_t\}`.

The time :math:`t` continuation of the vector of prices
:math:`\{p_t^t\}`.

Attaining a Canonical Household Technology
--------------------------------------------

Apply the following version of a factorization identity:

.. math::

   \begin{aligned}
    [\Pi &+ \beta^{1/2} L^{-1} \Lambda (I - \beta^{1/2} L^{-1}
   \Delta_h)^{-1} \Theta_h]^\prime [\Pi + \beta^{1/2} L
   \Lambda (I - \beta^{1/2} L \Delta_h)^{-1} \Theta_h]\\
   &= [\hat\Pi + \beta^{1/2} L^{-1} \hat\Lambda
   (I - \beta^{1/2} L^{-1} \Delta_h)^{-1} \Theta_h]^\prime
   [\hat\Pi + \beta^{1/2} L \hat\Lambda
   (I - \beta^{1/2} L \Delta_h)^{-1} \Theta_h]\end{aligned}

The factorization identity guarantees that the
:math:`[\hat \Lambda, \hat \Pi]` representation satisfies both
requirements for a canonical representation.



Partial Equilibrium
====================

Now we'll provide quick overviews of  examples of  economies that fit within our framework

We provide details for a number  of these examples in subsequent lectures 



  1. :doc:`Growth in Dynamic Linear Economies <growth_in_dles>`

  2. :doc:`Lucas Asset Pricing using DLE <lucas_asset_pricing_dles>`

  3. :doc:`IRFs in Hall Model <irfs_in_hall_model>`

  4. :doc:`Permanent Income Using the DLE class <permanent_income_dles>`

  5. :doc:`Rosen schooling model <rosen_schooling_model>`

  6. :doc:`Cattle cycles <cattle_cycles>`

  7. :doc:`Shock Non Invertibility <hs_invertibility_example>`


We'll start with an example of a **partial equilibrium** in which we posit demand and supply curves

Suppose that we want to capture the dynamic  demand curve:

.. math::

   \begin{aligned}
     c_t &=  -\Pi^{-1} \Lambda h_{t-1} + \Pi ^{-1} b_t - \Pi^{-1}
       \mu_0^w E_t \{ \Pi^{\prime\, -1} - \Pi^{\prime\, -1}\Theta_h' \\
     & \qquad[I - (\Delta_h' - \Lambda' \Pi^{\prime\, -1} \Theta_h')\beta
        L^{-1}]^{-1} \Lambda' \Pi^{\prime -1} \beta L^{-1} \}  p_t  \\
     h_t &= \Delta_h h_{t-1} + \Theta_h c_t \end{aligned}


From material described earlier in this lecture, we know how to reverse engineer preferences that generate this demand system

  * note how the demand equations are cast in terms of the matrices in our standard preference representation

Now let's turn to supply.

A representative firm takes as given and beyond its control the
stochastic process :math:`\{p_t\}_{t=0}^\infty`.

The firm sells its
output :math:`c_t` in a competitive market each period.

Only spot markets convene at each date :math:`t\geq 0`.

The firm also faces an exogenous process of cost disturbances :math:`d_t`.

The firm chooses stochastic processes :math:`\{c_t, g_t, i_t, k_t\}_{t=0}^\infty` to maximize

.. math:: E_0 \sum_{t=0}^\infty \beta^t \{ p_t \cdot c_t - g_t \cdot g_t/2 \}

subject to given :math:`k_{-1}` and

.. math::

   \begin{aligned}
   \Phi_c c_t  +  \Phi_i i_t + \Phi_g g_t &=\Gamma k_{t-1} + d_t  \\
      k_t& =   \Delta_k k_{t-1} + \Theta_k i_t . \\
    %  x_{t+1}& = A^o x_t + C w_{t+1}  \\
    %  d_t& = S_d x_t  \\
    %  p_t& = M_c x_t
                     \end{aligned}

Equilibrium Investment Under Uncertainty
=========================================

A representative firm maximizes

.. math:: E \sum_{t=0}^\infty \beta^t \{ p_t c_t - g_t^2/2 \}

subject to the technology

.. math::

   \begin{aligned}
    c_t &= \gamma k_{t-1} \\
                k_t &= \delta_k k_{t-1} + i_t \\
                g_t &= f_1 i_t + f_2 d_t \end{aligned}

where :math:`d_t` is a cost shifter, :math:`\gamma> 0`, and
:math:`f_1 >0` is a cost parameter and :math:`f_2 =1`. Demand is
governed by

.. math:: p_t = \alpha_0 - \alpha_1 c_t + u_t

where :math:`u_t` is a demand shifter with mean zero and
:math:`\alpha_0, \alpha_1` are positive parameters.

Assume that :math:`u_t, d_t` are uncorrelated first-order autoregressive processes.

A Rosen-Topel Housing Model
============================

.. math::

   \begin{aligned}
    R_t &= b_t + \alpha h_t \\
                p_t &= E_t \sum_{\tau =0}^\infty (\beta \delta_h)^\tau
                          R_{t+\tau} \end{aligned}

where :math:`h_t` is the stock of housing at time :math:`t`
:math:`R_t` is the rental rate for housing, :math:`p_t` is the price of
new houses, and :math:`b_t` is a demand shifter; :math:`\alpha < 0` is a
demand parameter, and :math:`\delta_h` is a depreciation factor for
houses.


We cast this demand specification within our class of models by letting
the stock of houses :math:`h_t` evolve according to

.. math:: h_t = \delta_h h_{t-1} + c_t, \quad \delta_h \in (0,1)

where :math:`c_t` is the rate of production of new houses.

Houses produce services :math:`s_t` according to
:math:`s_t  = \bar \lambda h_t` or
:math:`s_t  = \lambda h_{t-1} + \pi c_t,` where
:math:`\lambda= \bar \lambda \delta_h, \pi = \bar \lambda`.

We can take :math:`\bar \lambda \rho_t^0  = R_t` as the rental rate on housing at
time :math:`t`, measured in units of time :math:`t` consumption (housing).


Demand for housing services is

.. math:: s_t = b_t - \mu_0 \rho_t^0

where the price of new houses :math:`p_t` is related to
:math:`\rho_t^0` by
:math:`\rho_t^0 = \pi^{-1} [  p_t - \beta \delta_h E_t p_{t+1}]`.

Cattle Cycles
=============

Rosen, Murphy, and Scheinkman (1994). Let :math:`p_t` be the price of
freshly slaughtered beef, :math:`m_t` the feeding cost of preparing an
animal for slaughter, :math:`\tilde h_t` the one-period holding cost for
a mature animal, :math:`\gamma_1 \tilde h_t` the one-period holding cost
for a yearling, and :math:`\gamma_0 \tilde
h_t` the one-period holding cost for a calf.

The cost processes
:math:`\{\tilde h_t, m_t\}_{t=0}^\infty` are exogenous, while the
stochastic process :math:`\{p_t\}_{t=0}^\infty` is determined by a
rational expectations equilibrium. Let :math:`\tilde x_t` be the
breeding stock, and :math:`\tilde y_t` be the total stock of animals.

The law of motion for cattle stocks is

.. math:: \tilde x_t = (1-\delta) \tilde x_{t-1} + g \tilde x_{t-3} - c_t

where :math:`c_t` is a rate of slaughtering. The total head-count of
cattle

.. math:: \tilde y_t = \tilde x_t + g \tilde x_{t-1} + g \tilde x_{t-2}

is the sum of adults, calves, and yearlings, respectively.

.. _cattle-cycles-1:



A representative farmer chooses :math:`\{c_t, \tilde x_t\}` to maximize

.. math::

   \begin{aligned}
    E_0 \sum_{t=0}^\infty \beta^t \{ p_t c_t &-
        \tilde h_t \tilde x_t
           -(\gamma_0 \tilde h_t) (g \tilde x_{t-1}) - (\gamma_1 \tilde h_t)
            (g \tilde x_{t-2}) - m_t c_t \\
            &-   \Psi(\tilde x_t, \tilde x_{t-1},
            \tilde x_{t-2}, c_t) \}\end{aligned}

where

.. math::

   \Psi = {\psi_1 \over 2} \tilde x_t^2 + {\psi_2 \over 2} \tilde x_{t-1}^2
         + {\psi_3 \over 2} \tilde x_{t-2}^2 + {\psi_4 \over 2} c_t^2

Demand is governed by

.. math:: c_t = \alpha_0 - \alpha_1 p_t + \tilde d_t

where :math:`\alpha_0 > 0`, :math:`\alpha_1 > 0`, and
:math:`\{\tilde d_t\}_{t=0}^\infty` is a stochastic process with mean
zero representing a demand shifter.

For more details see :doc:`Cattle cycles <cattle_cycles>`

Models of Occupational Choice and Pay
======================================

We'll describe the following pair of schooling models that view education as a time-to-build process:


-  Rosen schooling model for engineers

-  Two-occupation model

Market for Engineers
---------------------

Ryoo and Rosen’s (2004) :cite:`ryoo2004engineering` model consists of the following equations:

first, a demand curve for engineers

.. math:: w_t = - \alpha_d N_t + \epsilon_{1t}\ ,\ \alpha_d > 0

second, a time-to-build structure of the education process

.. math:: N_{t+k} = \delta_N N_{t+k-1} + n_t\ ,\ 0<\delta_N<1

third, a definition of the discounted present value of each new
engineering student

.. math::

   v_t = \beta^k E_t \sum^\infty_{j=0} (\beta  \delta_N)^j
      w_{t+k+j};

and fourth, a supply curve of new students driven by :math:`v_t`

.. math:: n_t = \alpha_s v_t + \epsilon_{2t}\ ,\ \alpha_s > 0

Here :math:`\{\epsilon_{1t}, \epsilon_{2t}\}` are stochastic processes
of labor demand and supply shocks.

.. _market-for-engineers-1:



**Definition:** A partial equilibrium is a stochastic process
:math:`\{w_t, N_t, v_t, n_t\}^\infty_{t=0}` satisfying these four
equations, and initial conditions
:math:`N_{-1}, n_{-s}, s=1, \ldots, -k`.


We sweep the time-to-build structure and the demand for engineers into
the household technology and putting the supply of new engineers into
the technology for producing goods.

.. math::

   \begin{aligned}
    s_t &= [\lambda_1 \ 0 \ \ldots \ 0]\ \begin{bmatrix}
   h_{1t-1}\\ h_{2t-1}\\ \vdots \\ h_{k+1,t-1}\end{bmatrix} + 0 \cdot c_t \\
   \begin{bmatrix} h_{1t}\\ h_{2t}\\ \vdots\\ h_{k,t} \\
      h_{k+1,t}\end{bmatrix} &=
   \begin{bmatrix} \delta_N & 1 & 0 & \cdots & 0\\ 0 & 0 & 1 & \cdots & 0\\
   \vdots & \vdots & \vdots & \ddots & \vdots\\ 0 & \cdots & \cdots & 0 & 1\\
   0 & 0 & 0 & \cdots & 0 \end{bmatrix} \begin{bmatrix}h_{1t-1}\\ h_{2t-1}\\ \vdots\\ h_{k,t-1} \\
         h_{k+1,t-1}\end{bmatrix} + \begin{bmatrix}0\\ 0\\ \vdots\\  0\\ 1\\\end{bmatrix}  c_t \\
   %b_t &= \epsilon_{1t}
    \end{aligned}

This specification sets Rosen’s :math:`N_t = h_{1t-1}, n_t = c_t,
h_{\tau+1,t-1} = n_{t-\tau}, \tau=1, \ldots, k`, and uses the
home-produced service to capture the demand for labor. Here
:math:`\lambda_1` embodies Rosen’s demand parameter :math:`\alpha_d`.


- The supply of new workers becomes our consumption.

- The dynamic demand curve becomes Rosen’s dynamic supply curve for new workers.

**Remark:** This has an Imai-Keane flavor.

For more details and Python code see :doc:`Rosen schooling model <rosen_schooling_model>`.




Skilled and Unskilled Workers
-------------------------------

First, a demand curve for labor

.. math::

   \begin{bmatrix} w_{ut} \\ w_{st} \end{bmatrix}
        = \alpha_d \begin{bmatrix} N_{ut} \\ N_{st} \end{bmatrix}
           + \epsilon_{1t}

where :math:`\alpha_d` is a :math:`(2 \times 2)` matrix of demand
parameters and :math:`\epsilon_{1t}` is a vector of demand shifters
second, time-to-train specifications for skilled and unskilled labor,
respectively:

.. math::

   \begin{aligned}
    N_{st+k} &=\delta_N N_{st+k-1} + n_{st} \\
                N_{ut} &=\delta_N N_{ut-1} + n_{ut} ; \end{aligned}

where :math:`N_{st}, N_{ut}` are stocks of the two types of labor, and
:math:`n_{st}, n_{ut}` are entry rates into the two occupations.

.. _skilled-and-unskilled-workers-1:


third, definitions of discounted present values of new entrants to the
skilled and unskilled occupations, respectively:

.. math::

   \begin{aligned}
    v_{st} &= E_t \beta^k \sum_{j=0}^\infty (\beta \delta_N)^j
            w_{st+k+j} \\
                v_{ut} &=E_t \sum_{j=0}^\infty (\beta \delta_N)^j
       w_{ut+j}\end{aligned}

where :math:`w_{ut}, w_{st}` are wage rates for the two occupations;
and fourth, supply curves for new entrants:

.. math::

   \begin{bmatrix}n_{st} \\ n_{ut}\end{bmatrix}
         = \alpha_s \begin{bmatrix} v_{ut} \\ v_{st} \end{bmatrix} +
           \epsilon_{2t}

**Short Cut**

As an alternative, Siow simply used the  **equalizing differences**
condition

.. math:: v_{ut} = v_{st}


Permanent Income Models
=======================

We'll describe a class of permanent income models that feature

-  Many consumption goods and services

-  A single capital good with :math:`R \beta =1`

 - The physical production technology 

.. math::

   \begin{aligned}
    \phi_c \cdot c_t+i_t&=\gamma k_{t-1}+e_t \\
               k_t&= k_{t-1} + i_t \end{aligned}

.. math:: \phi_ii_t-g_t=0

**Implication One:**

Equality of Present Values of Moving Average Coefficients of :math:`c` and :math:`e`

.. math:: k_{t-1} = \beta \sum_{j=0}^\infty \beta^j (\phi_c \cdot c_{t+j} - e_{t+j}) \quad  \Rightarrow

.. math::

   k_{t-1} = \beta \sum_{j=0}^\infty  \beta^j E (\phi_c
       \cdot c_{t+j} - e_{t+j})|J_t \quad \Rightarrow

.. math::

   \sum_{j=0}^\infty \beta^j (\phi_c)^\prime \chi_j =
    \sum_{j=0}^\infty \beta^j \epsilon_j

where :math:`\chi_j w_t` is the response of :math:`c_{t+j}` to
:math:`w_t` and :math:`\epsilon_j w_t` is the response of endowment
:math:`e_{t+j}` to :math:`w_t`:

**Implication Two:**

Martingales

.. math::

   \begin{aligned}
    {\mathcal M}_t^k  &= E ({\mathcal M}_{t+1}^k | J_t) \\
   {\mathcal M}_t^e  &= E ({\mathcal M}_{t+1}^e | J_t) \end{aligned}

and

.. math:: {\mathcal M}_t^c  =  (\Phi_c)^\prime {\mathcal M}_t^d = \phi_c {\cal M}_t^e


For more details see :doc:`Permanent Income Using the DLE class <permanent_income_dles>`

**Testing Permanent Income Models:**

We have two types of  implications of permanent income models:

-  Equality of present values of moving average coefficients.

-  Martingale :math:`{\mathcal M}_t^k`.

These have been tested in work by Hansen, Sargent, and Roberts (1991) :cite:`sargent1991observable`
and by Attanasio and Pavoni (2011) :cite:`attanasio2011risk`.



Gorman Heterogeneous Households
================================

We now assume that there is a finite number of households, each with its own household  technology and
preferences over consumption services.

Household :math:`j` orders preferences over consumption processes according to

.. math::

   -\ \left({1 \over 2}\right)\ E \sum_{t=0}^\infty\, \beta^t\, \bigl[(s_{jt} -
   b_{jt}) \cdot (s_{jt} - b_{jt}) + \ell_{jt}^{2}\bigr] \mid J_0

.. math:: s_{jt} = \Lambda\, h_{j,t-1} + \Pi\, c_{jt}

.. math:: h_{jt} = \Delta_h\, h_{j,t-1} + \Theta_h\, c_{jt}

and :math:`h_{j,-1}` is given

.. math:: b_{jt} = U_{bj} z_t

.. math::

   E\, \sum_{t=0}^\infty\, \beta^t\, p_t^0\, \cdot c_{jt} \mid J_0 = E\,
   \sum_{t=0}^\infty\, \beta^t\, (w_t^0\, \ell_{jt} + \alpha_t^0\, \cdot
        d_{jt}) \mid
   J_0 + v_0\, \cdot k_{j,-1},

where :math:`k_{j,-1}` is given. The :math:`j^{\rm th}` consumer owns
an endowment process :math:`d_{jt}`, governed by the stochastic process
:math:`d_{jt} = U_{dj}\, z_t`.

.. _gorman-heterogeneous-households-1:

We refer to this as a setting with  Gorman heterogeneous households.


This specification confines heterogeneity among consumers to:

- differences in the preference processes :math:`\{b_{jt}\}`, represented by different selections of :math:`U_{bj}`

- differences in the endowment processes :math:`\{d_{jt}\}`, represented by different selections of :math:`U_{dj}`

- differences in :math:`h_{j,-1}` and

- differences in :math:`k_{j,-1}`

The matrices :math:`\Lambda,\,\Pi,\,\Delta_h,\,\Theta_h` do not depend on :math:`j`.

This makes everybody’s demand system have the form described earlier,
with different :math:`\mu_{j0}^w`\ ’s (reflecting different wealth
levels) and different :math:`b_{jt}` preference shock processes and
initial conditions for household capital stocks.


**Punchline:** there exists a representative consumer.

We can use the representative consumer to compute a competitive equilibrium **aggregate** allocation and price system.

With the equilibrium aggregate allocation and price system in hand, we can then compute allocations to each household.

**Computing  Allocations to Individuals:**


Set

.. math:: \ell_{jt} = (\mu_{0j}^w/\mu_{0a}^w) \ell_{at}

Then solve the following equation for :math:`\mu_{0j}^{w}`:

.. math::

   \mu_{0j}^{w}  E_0 \sum_{t=0}^\infty \beta^t \{\rho_t^0 \cdot \rho_t^0
       + (w_t^0/ \mu_{0a}^{w}) \ell_{at} \}
       = E_0 \sum_{t=0}^\infty \beta^t \{ \rho_t^0 \cdot (b_{jt} - s_{jt}^i)
       -  \alpha_t^0 \cdot d_{jt} \}
          - v_0 k_{j,-1}

.. math:: s_{jt} - b_{jt} = \mu_{0j}^w\rho^0_t

.. math::

   \begin{aligned}
    c_{jt} &= - \Pi^{-1} \Lambda h_{j,t-1} + \Pi^{-1}s_{jt} \\
   h_{jt} &= (\Delta_h - \Theta_h \Pi^{-1}\Lambda) h_{j,t-1} + \Pi^{-1}
       \Theta_h  s_{jt} \end{aligned}

Here :math:`h_{j,-1}` given.


Non-Gorman Heterogeneous Households
====================================

We now describe a less tractable type of heterogeneity across households that we dub **Non-Gorman heterogeneity**.

Here is the specification:

Preferences and Household Technologies:

.. math::

   - {1\over 2} E\, \sum^\infty_{t=0}\, \beta^t\, [ (s_{it} - b_{it}) \cdot
   (s_{it} - b_{it}) + \ell^2_{it}]\mid J_0

.. math::

   \begin{aligned}
    s_{it} &= \Lambda_i h_{i t-1} + \Pi_i\, c_{it} \\
   h_{it} &=\Delta_{h_i}\, h_{i t-1} + \Theta_{h_i} c_{it}\ ,\ i=1,2 .\end{aligned}

.. math:: b_{it} = U_{bi} z_t

.. math:: z_{t+1} = A_{22} z_t + C_2 w_{t+1}

**Production Technology**

.. math::

   \Phi_c (c_{1t} + c_{2t}) + \Phi_g g_t + \Phi_i i_t = \Gamma
   k_{t-1} + d_{1t} + d_{2t}

.. math:: k_t = \Delta_k k_{t-1} + \Theta_k i_t

.. math::

   g_t \cdot g_t = \ell^2_t,\qquad
      \ \ell_t = \ell_{1t} + \ell_{2t}

.. math:: d_{it} = U_{d_i} z_t, \quad \ i=1,2

**Pareto Problem:**


.. math::

   \begin{aligned}
    & - {1\over 2}\, \lambda E_0 \sum^\infty_{t=0}\, \beta^t [ (s_{1t}
   - b_{1t})\cdot (s_{1t} - b_{1t}) + \ell^2_{1t}]\\
    &
   - {1\over 2}\, (1-\lambda) E_0 \sum^\infty_{t=0}\, \beta^t [ (s_{2t} -
   b_{2t}) \cdot (s_{2t} - b_{2t}) + \ell^2_{2t}] \end{aligned}

**Mongrel Aggregation: Static**

There is what we call a kind of **mongrel aggregation** in this setting.

We first describe the idea within a simple static setting in which there is a single consumer static inverse demand with
implied preferences:

.. math:: c_t = \Pi^{-1} b_t - \mu_0 \Pi^{-1} \Pi^{-1 \prime} p_t

An inverse demand curve is

.. math:: p_t = \mu_0^{-1} \Pi' b_t - \mu_0^{-1} \Pi' \Pi c_t

Integrating the marginal utility vector shows that preferences can be
taken to be

.. math:: ( - 2 \mu_0)^{-1} (\Pi c_t - b_t) \cdot (\Pi c_t - b_t )

.. _mongrel-aggregation-static-1:



**Key Insight:** Factor the inverse of a ‘covariance matrix’.

Now assume that there are two consumers, :math:`i=1,2`, with demand curves

.. math:: c_{it} = \Pi_i^{-1} b_{it} - \mu_{0i} \Pi_i^{-1} \Pi_i^{-1 \prime} p_t

.. math::

   c_{1t} + c_{2t} = (\Pi_1^{-1} b_{1t} + \Pi_2^{-1} b_{2t})
       - (\mu_{01} \Pi_1^{-1} \Pi_1^{-1 \prime} + \mu_{02} \Pi_2
          \Pi_2^{-1 \prime}) p_t

Setting :math:`c_{1t} + c_{2t} = c_t` and solving for :math:`p_t` gives

.. math::

   \begin{aligned}
    p_t &= (\mu_{01} \Pi_1^{-1} \Pi_1^{-1 \prime} + \mu_{02}
       \Pi_2^{-1} \Pi_2^{-1 \prime})^{-1}
         (\Pi_1^{-1} b_{1t} + \Pi_2^{-1} b_{2t}) \\
     &- (\mu_{01} \Pi_1^{-1} \Pi_1^{-1 \prime} +
        \mu_{02} \Pi_2^{-1} \Pi_2^{-1 \prime}
         )^{-1} c_t\end{aligned}

**Punchline:** choose :math:`\Pi` associated with the aggregate ordering to
satisfy

.. math::

   \mu_0^{-1} \Pi' \Pi = (\mu_{01} \Pi_1^{-1} \Pi_2^{-1 \prime}
         + \mu_{02} \Pi_2^{-1} \Pi_2^{-1 \prime})^{-1}

**Dynamic Analogue:**


We now describe how to extend mongrel aggregation to a dynamic setting.

The key comparison is

-  Static: factor a covariance matrix-like object

-  Dynamic: factor a spectral-density matrix-like object

Programming Problem for Dynamic Mongrel Aggregation:

Our strategy for deducing the mongrel preference ordering over
:math:`c_t = c_{1t} + c_{2t}` is to solve the programming problem:
choose :math:`\{c_{1t},c_{2t}\}` to maximize the criterion

.. math::

   \sum^\infty_{t=0} \beta^t [\lambda (s_{1t} - b_{1t}) \cdot (s_{1t} - b_{1t})
    + (1-\lambda) (s_{2t} - b_{2t}) \cdot (s_{2t} - b_{2t})]

subject to

.. math::

   \begin{aligned}
    h_{jt} &= \Delta_{hj}\, h_{jt-1} + \Theta_{hj}\, c_{jt}, j=1,2\\
   s_{jt} &=\Delta_j h_{jt-1} + \Pi_j c_{jt}\ , j=1,2\\
   c_{1t} +   c_{2t} &=c_t\end{aligned}

subject to :math:`(h_{1, -1},\, h_{2, -1})` given and
:math:`\{b_{1t}\},\, \{b_{2t}\},\, \{c_t\}` being known and fixed
sequences.

Substituting the :math:`\{c_{1t},\, c_{2t}\}` sequences that
solve this problem as functions of :math:`\{b_{1t},\, b_{2t},\, c_t\}`
into the objective determines a mongrel preference ordering over
:math:`\{c_t\} = \{c_{1t} + c_{2t}\}`.


In solving this problem, it is convenient to proceed by using Fourier
transforms.  For details, please see :cite:`HS2013` where they deploy a

**Secret Weapon:** Another application of the spectral factorization
identity.


**Concluding remark:** The :cite:`HS2013` class of models described in this lecture are all complete markets models.  We have exploited
the fact that complete market models **are all alike** to allow us to define a class that **gives the same name to different things** in the
spirit of Henri Poincare.

Could we create such a class for **incomplete markets** models?

That would be nice, but before trying it would be wise to contemplate
the remainder of a statement by Robert E. Lucas, Jr., with which we began this lecture.

.. epigraph::

    "Complete market economies are all alike but each incomplete market  economy is incomplete in its own individual way."   Robert E. Lucas,   Jr., (1989)
