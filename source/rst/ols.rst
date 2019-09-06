.. include:: /_static/includes/header.raw

.. highlight:: python3

***************************
Linear Regression in Python
***************************

.. contents:: :depth: 2

In addition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :class: hide-output

  !pip install linearmodels

Overview
========


Linear regression is a standard tool for analyzing the relationship between two or more variables.

In this lecture, we'll use the Python package ``statsmodels`` to estimate, interpret, and visualize linear regression models.

Along the way, we'll discuss a variety of topics, including

-  simple and multivariate linear regression
-  visualization
-  endogeneity and omitted variable bias
-  two-stage least squares

As an example, we will replicate results from Acemoglu, Johnson and Robinson's seminal paper :cite:`Acemoglu2001`.

* You can download a copy `here <https://economics.mit.edu/files/4123>`__.

In the paper, the authors emphasize the importance of institutions in economic development.

The main contribution is the use of settler mortality rates as a source of *exogenous* variation in institutional differences.

Such variation is needed to determine whether it is institutions that give rise to greater economic growth, rather than the other way around.

Let's start with some standard imports:

.. code-block:: ipython

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.iolib.summary2 import summary_col
    from linearmodels.iv import IV2SLS


Prerequisites
-------------

This lecture assumes you are familiar with basic econometrics.

For an introductory text covering these topics, see, for example,
:cite:`Wooldridge2015`.



Comments
--------

This lecture is coauthored with `Natasha Watkins <https://github.com/natashawatkins>`__.



Simple Linear Regression
========================

:cite:`Acemoglu2001` wish to determine whether or not differences in institutions can help to explain observed economic outcomes.

How do we measure *institutional differences* and *economic outcomes*?

In this paper,

-  economic outcomes are proxied by log GDP per capita in 1995, adjusted for exchange rates.

-  institutional differences are proxied by an index of protection against expropriation on average over 1985-95, constructed by the `Political Risk Services Group <https://www.prsgroup.com/>`__.

These variables and other data used in the paper are available for download on Daron Acemoglu's `webpage <https://economics.mit.edu/faculty/acemoglu/data/ajr2001>`__.

We will use pandas' ``.read_stata()`` function to read in data contained in the ``.dta`` files to dataframes


.. code-block:: python3

    df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable1.dta')
    df1.head()



Let's use a scatterplot to see whether any obvious relationship exists
between GDP per capita and the protection against
expropriation index

.. code-block:: python3

    plt.style.use('seaborn')

    df1.plot(x='avexpr', y='logpgp95', kind='scatter')
    plt.show()

The plot shows a fairly strong positive relationship between
protection against expropriation and log GDP per capita.

Specifically, if higher protection against expropriation is a measure of
institutional quality, then better institutions appear to be positively
correlated with better economic outcomes (higher GDP per capita).

Given the plot, choosing a linear model to describe this relationship
seems like a reasonable assumption.

We can write our model as

.. math::


   {logpgp95}_i = \beta_0 + \beta_1 {avexpr}_i + u_i

where:

-  :math:`\beta_0` is the intercept of the linear trend line on the
   y-axis

-  :math:`\beta_1` is the slope of the linear trend line, representing
   the *marginal effect* of protection against risk on log GDP per
   capita

-  :math:`u_i` is a random error term (deviations of observations from
   the linear trend due to factors not included in the model)

Visually, this linear model involves choosing a straight line that best
fits the data, as in the following plot (Figure 2 in :cite:`Acemoglu2001`)

.. code-block:: python3

    # Dropping NA's is required to use numpy's polyfit
    df1_subset = df1.dropna(subset=['logpgp95', 'avexpr'])

    # Use only 'base sample' for plotting purposes
    df1_subset = df1_subset[df1_subset['baseco'] == 1]

    X = df1_subset['avexpr']
    y = df1_subset['logpgp95']
    labels = df1_subset['shortnam']

    # Replace markers with country labels
    fig, ax = plt.subplots()
    ax.scatter(X, y, marker='')

    for i, label in enumerate(labels):
        ax.annotate(label, (X.iloc[i], y.iloc[i]))

    # Fit a linear trend line
    ax.plot(np.unique(X),
             np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
             color='black')

    ax.set_xlim([3.3,10.5])
    ax.set_ylim([4,10.5])
    ax.set_xlabel('Average Expropriation Risk 1985-95')
    ax.set_ylabel('Log GDP per capita, PPP, 1995')
    ax.set_title('Figure 2: OLS relationship between expropriation \
        risk and income')
    plt.show()


The most common technique to estimate the parameters (:math:`\beta`'s)
of the linear model is Ordinary Least Squares (OLS).

As the name implies, an OLS model is solved by finding the parameters
that minimize *the sum of squared residuals*, i.e.

.. math::


   \underset{\hat{\beta}}{\min} \sum^N_{i=1}{\hat{u}^2_i}

where :math:`\hat{u}_i` is the difference between the observation and
the predicted value of the dependent variable.

To estimate the constant term :math:`\beta_0`, we need to add a column
of 1's to our dataset (consider the equation if :math:`\beta_0` was
replaced with :math:`\beta_0 x_i` and :math:`x_i = 1`)

.. code-block:: python3

    df1['const'] = 1

Now we can construct our model in ``statsmodels`` using the OLS function.

We will use ``pandas`` dataframes with ``statsmodels``, however standard arrays can also be used as arguments



.. code-block:: python3

    reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], \
        missing='drop')
    type(reg1)



So far we have simply constructed our model.

We need to use ``.fit()`` to obtain parameter estimates
:math:`\hat{\beta}_0` and :math:`\hat{\beta}_1`

.. code-block:: python3

    results = reg1.fit()
    type(results)



We now have the fitted regression model stored in ``results``.

To view the OLS regression results, we can call the ``.summary()``
method.

Note that an observation was mistakenly dropped from the results in the
original paper (see the note located in `maketable2.do` from Acemoglu's webpage), and thus the
coefficients differ slightly.

.. code-block:: python3

    print(results.summary())



From our results, we see that

-  The intercept :math:`\hat{\beta}_0 = 4.63`.
-  The slope :math:`\hat{\beta}_1 = 0.53`.
-  The positive :math:`\hat{\beta}_1` parameter estimate implies that.
   institutional quality has a positive effect on economic outcomes, as
   we saw in the figure.
-  The p-value of 0.000 for :math:`\hat{\beta}_1` implies that the
   effect of institutions on GDP is statistically significant (using p <
   0.05 as a rejection rule).
-  The R-squared value of 0.611 indicates that around 61% of variation
   in log GDP per capita is explained by protection against
   expropriation.

Using our parameter estimates, we can now write our estimated
relationship as

.. math::

   \widehat{logpgp95}_i = 4.63 + 0.53 \ {avexpr}_i

This equation describes the line that best fits our data, as shown in
Figure 2.

We can use this equation to predict the level of log GDP per capita for
a value of the index of expropriation protection.

For example, for a country with an index value of 7.07 (the average for
the dataset), we find that their predicted level of log GDP per capita
in 1995 is 8.38.

.. code-block:: python3

    mean_expr = np.mean(df1_subset['avexpr'])
    mean_expr


.. code-block:: python3

    predicted_logpdp95 = 4.63 + 0.53 * 7.07
    predicted_logpdp95


An easier (and more accurate) way to obtain this result is to use
``.predict()`` and set :math:`constant = 1` and
:math:`{avexpr}_i = mean\_expr`

.. code-block:: python3

    results.predict(exog=[1, mean_expr])


We can obtain an array of predicted :math:`{logpgp95}_i` for every value
of :math:`{avexpr}_i` in our dataset by calling ``.predict()`` on our
results.

Plotting the predicted values against :math:`{avexpr}_i` shows that the
predicted values lie along the linear line that we fitted above.

The observed values of :math:`{logpgp95}_i` are also plotted for
comparison purposes

.. code-block:: python3

    # Drop missing observations from whole sample

    df1_plot = df1.dropna(subset=['logpgp95', 'avexpr'])

    # Plot predicted values

    fix, ax = plt.subplots()
    ax.scatter(df1_plot['avexpr'], results.predict(), alpha=0.5, 
            label='predicted')

    # Plot observed values

    ax.scatter(df1_plot['avexpr'], df1_plot['logpgp95'], alpha=0.5,
            label='observed')

    ax.legend()
    ax.set_title('OLS predicted values')
    ax.set_xlabel('avexpr')
    ax.set_ylabel('logpgp95')
    plt.show()



Extending the Linear Regression Model
=====================================

So far we have only accounted for institutions affecting economic
performance - almost certainly there are numerous other factors
affecting GDP that are not included in our model.

Leaving out variables that affect :math:`logpgp95_i` will result in **omitted variable bias**, yielding biased and inconsistent parameter estimates.

We can extend our bivariate regression model to a **multivariate regression model** by adding in other factors that may affect :math:`logpgp95_i`.

:cite:`Acemoglu2001` consider other factors such as:

-  the effect of climate on economic outcomes; latitude is used to proxy
   this

-  differences that affect both economic performance and institutions,
   eg. cultural, historical, etc.; controlled for with the use of
   continent dummies

Let's estimate some of the extended models considered in the paper
(Table 2) using data from ``maketable2.dta``

.. code-block:: python3

    df2 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable2.dta')

    # Add constant term to dataset
    df2['const'] = 1

    # Create lists of variables to be used in each regression
    X1 = ['const', 'avexpr']
    X2 = ['const', 'avexpr', 'lat_abst']
    X3 = ['const', 'avexpr', 'lat_abst', 'asia', 'africa', 'other']

    # Estimate an OLS regression for each set of variables
    reg1 = sm.OLS(df2['logpgp95'], df2[X1], missing='drop').fit()
    reg2 = sm.OLS(df2['logpgp95'], df2[X2], missing='drop').fit()
    reg3 = sm.OLS(df2['logpgp95'], df2[X3], missing='drop').fit()

Now that we have fitted our model, we will use ``summary_col`` to
display the results in a single table (model numbers correspond to those
in the paper)

.. code-block:: python3

    info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
               'No. observations' : lambda x: f"{int(x.nobs):d}"}

    results_table = summary_col(results=[reg1,reg2,reg3],
                                float_format='%0.2f',
                                stars = True,
                                model_names=['Model 1',
                                             'Model 3',
                                             'Model 4'],
                                info_dict=info_dict,
                                regressor_order=['const',
                                                 'avexpr',
                                                 'lat_abst',
                                                 'asia',
                                                 'africa'])

    results_table.add_title('Table 2 - OLS Regressions')

    print(results_table)

Endogeneity
===========

As :cite:`Acemoglu2001` discuss, the OLS models likely suffer from
**endogeneity** issues, resulting in biased and inconsistent model
estimates.

Namely, there is likely a two-way relationship between institutions and
economic outcomes:

-  richer countries may be able to afford or prefer better institutions

-  variables that affect income may also be correlated with
   institutional differences

-  the construction of the index may be biased; analysts may be biased
   towards seeing countries with higher income having better
   institutions

To deal with endogeneity, we can use **two-stage least squares (2SLS)
regression**, which is an extension of OLS regression.

This method requires replacing the endogenous variable
:math:`{avexpr}_i` with a variable that is:

1. correlated with :math:`{avexpr}_i`
2. not correlated with the error term (ie. it should not directly affect
   the dependent variable, otherwise it would be correlated with
   :math:`u_i` due to omitted variable bias)

The new set of regressors is called an **instrument**, which aims to
remove endogeneity in our proxy of institutional differences.

The main contribution of :cite:`Acemoglu2001` is the use of settler mortality
rates to instrument for institutional differences.

They hypothesize that higher mortality rates of colonizers led to the
establishment of institutions that were more extractive in nature (less
protection against expropriation), and these institutions still persist
today.

Using a scatterplot (Figure 3 in :cite:`Acemoglu2001`), we can see protection
against expropriation is negatively correlated with settler mortality
rates, coinciding with the authors' hypothesis and satisfying the first
condition of a valid instrument.

.. code-block:: python3

    # Dropping NA's is required to use numpy's polyfit
    df1_subset2 = df1.dropna(subset=['logem4', 'avexpr'])

    X = df1_subset2['logem4']
    y = df1_subset2['avexpr']
    labels = df1_subset2['shortnam']

    # Replace markers with country labels
    fig, ax = plt.subplots()
    ax.scatter(X, y, marker='')

    for i, label in enumerate(labels):
        ax.annotate(label, (X.iloc[i], y.iloc[i]))

    # Fit a linear trend line
    ax.plot(np.unique(X),
             np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
             color='black')

    ax.set_xlim([1.8,8.4])
    ax.set_ylim([3.3,10.4])
    ax.set_xlabel('Log of Settler Mortality')
    ax.set_ylabel('Average Expropriation Risk 1985-95')
    ax.set_title('Figure 3: First-stage relationship between settler mortality \
        and expropriation risk')
    plt.show()



The second condition may not be satisfied if settler mortality rates in the 17th to 19th centuries have a direct effect on current GDP (in addition to their indirect effect through institutions).

For example, settler mortality rates may be related to the current disease environment in a country, which could affect current economic performance.

:cite:`Acemoglu2001` argue this is unlikely because:

-  The majority of settler deaths were due to malaria and yellow fever
   and had a limited effect on local people.

-  The disease burden on local people in Africa or India, for example,
   did not appear to be higher than average, supported by relatively
   high population densities in these areas before colonization.

As we appear to have a valid instrument, we can use 2SLS regression to
obtain consistent and unbiased parameter estimates.

**First stage**

The first stage involves regressing the endogenous variable
(:math:`{avexpr}_i`) on the instrument.

The instrument is the set of all exogenous variables in our model (and
not just the variable we have replaced).

Using model 1 as an example, our instrument is simply a constant and
settler mortality rates :math:`{logem4}_i`.

Therefore, we will estimate the first-stage regression as

.. math::


   {avexpr}_i = \delta_0 + \delta_1 {logem4}_i + v_i

The data we need to estimate this equation is located in
``maketable4.dta`` (only complete data, indicated by ``baseco = 1``, is
used for estimation)

.. code-block:: python3

    # Import and select the data
    df4 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable4.dta')
    df4 = df4[df4['baseco'] == 1]

    # Add a constant variable
    df4['const'] = 1

    # Fit the first stage regression and print summary
    results_fs = sm.OLS(df4['avexpr'],
                        df4[['const', 'logem4']],
                        missing='drop').fit()
    print(results_fs.summary())


**Second stage**

We need to retrieve the predicted values of :math:`{avexpr}_i` using
``.predict()``.

We then replace the endogenous variable :math:`{avexpr}_i` with the
predicted values :math:`\widehat{avexpr}_i` in the original linear model.

Our second stage regression is thus

.. math::


   {logpgp95}_i = \beta_0 + \beta_1 \widehat{avexpr}_i + u_i

.. code-block:: python3

    df4['predicted_avexpr'] = results_fs.predict()

    results_ss = sm.OLS(df4['logpgp95'],
                        df4[['const', 'predicted_avexpr']]).fit()
    print(results_ss.summary())




The second-stage regression results give us an unbiased and consistent
estimate of the effect of institutions on economic outcomes.

The result suggests a stronger positive relationship than what the OLS
results indicated.

Note that while our parameter estimates are correct, our standard errors
are not and for this reason, computing 2SLS 'manually' (in stages with
OLS) is not recommended.

We can correctly estimate a 2SLS regression in one step using the
`linearmodels <https://github.com/bashtage/linearmodels>`__ package, an extension of ``statsmodels``


Note that when using ``IV2SLS``, the exogenous and instrument variables
are split up in the function arguments (whereas before the instrument
included exogenous variables)

.. code-block:: python3

    iv = IV2SLS(dependent=df4['logpgp95'],
                exog=df4['const'],
                endog=df4['avexpr'],
                instruments=df4['logem4']).fit(cov_type='unadjusted')

    print(iv.summary)



Given that we now have consistent and unbiased estimates, we can infer
from the model we have estimated that institutional differences
(stemming from institutions set up during colonization) can help
to explain differences in income levels across countries today.

:cite:`Acemoglu2001` use a marginal effect of 0.94 to calculate that the
difference in the index between Chile and Nigeria (ie. institutional
quality) implies up to a 7-fold difference in income, emphasizing the
significance of institutions in economic development.



Summary
=======

We have demonstrated basic OLS and 2SLS regression in ``statsmodels`` and ``linearmodels``.

If you are familiar with R, you may want to use the `formula interface <http://www.statsmodels.org/dev/example_formulas.html>`__ to ``statsmodels``, or consider using `r2py <https://rpy2.bitbucket.io/>`__ to call R from within Python.



Exercises
=========

Exercise 1
----------

In the lecture, we think the original model suffers from endogeneity
bias due to the likely effect income has on institutional development.

Although endogeneity is often best identified by thinking about the data
and model, we can formally test for endogeneity using the **Hausman
test**.

We want to test for correlation between the endogenous variable,
:math:`avexpr_i`, and the errors, :math:`u_i`

.. math::

  \begin{aligned}
   H_0 : Cov(avexpr_i, u_i) = 0  \quad (no\ endogeneity) \\
   H_1 : Cov(avexpr_i, u_i) \neq 0 \quad (endogeneity)
   \end{aligned}

This test is running in two stages.

First, we regress :math:`avexpr_i` on the instrument, :math:`logem4_i`

.. math::


   avexpr_i = \pi_0 + \pi_1 logem4_i + \upsilon_i

Second, we retrieve the residuals :math:`\hat{\upsilon}_i` and include
them in the original equation

.. math::


   logpgp95_i = \beta_0 + \beta_1 avexpr_i + \alpha \hat{\upsilon}_i + u_i

If :math:`\alpha` is statistically significant (with a p-value < 0.05),
then we reject the null hypothesis and conclude that :math:`avexpr_i` is
endogenous.

Using the above information, estimate a Hausman test and interpret your
results.

Exercise 2
----------

The OLS parameter :math:`\beta` can also be estimated using matrix
algebra and ``numpy`` (you may need to review the
`numpy <https://lectures.quantecon.org/py/numpy.html>`__ lecture to
complete this exercise).

The linear equation we want to estimate is (written in matrix form)

.. math::


   y = X\beta + u

To solve for the unknown parameter :math:`\beta`, we want to minimize
the sum of squared residuals

.. math::


   \underset{\hat{\beta}}{\min} \hat{u}'\hat{u}

Rearranging the first equation and substituting into the second
equation, we can write

.. math::


   \underset{\hat{\beta}}{\min} \ (Y - X\hat{\beta})' (Y - X\hat{\beta})

Solving this optimization problem gives the solution for the
:math:`\hat{\beta}` coefficients

.. math::


   \hat{\beta} = (X'X)^{-1}X'y

Using the above information, compute :math:`\hat{\beta}` from model 1
using ``numpy`` - your results should be the same as those in the
``statsmodels`` output from earlier in the lecture.







Solutions
=========

Exercise 1
----------

.. code-block:: python3

    # Load in data
    df4 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable4.dta')

    # Add a constant term
    df4['const'] = 1

    # Estimate the first stage regression
    reg1 = sm.OLS(endog=df4['avexpr'],
                  exog=df4[['const', 'logem4']],
                  missing='drop').fit()

    # Retrieve the residuals
    df4['resid'] = reg1.resid

    # Estimate the second stage residuals
    reg2 = sm.OLS(endog=df4['logpgp95'],
                  exog=df4[['const', 'avexpr', 'resid']],
                  missing='drop').fit()

    print(reg2.summary())


The output shows that the coefficient on the residuals is statistically
significant, indicating :math:`avexpr_i` is endogenous.

Exercise 2
----------

.. code-block:: python3

    # Load in data
    df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable1.dta')
    df1 = df1.dropna(subset=['logpgp95', 'avexpr'])

    # Add a constant term
    df1['const'] = 1

    # Define the X and y variables
    y = np.asarray(df1['logpgp95'])
    X = np.asarray(df1[['const', 'avexpr']])

    # Compute β_hat
    β_hat = np.linalg.solve(X.T @ X, X.T @ y)

    # Print out the results from the 2 x 1 vector β_hat
    print(f'β_0 = {β_hat[0]:.2}')
    print(f'β_1 = {β_hat[1]:.2}')

It is also possible to use ``np.linalg.inv(X.T @ X) @ X.T @ y`` to solve
for :math:`\beta`, however ``.solve()`` is preferred as it involves fewer
computations.
