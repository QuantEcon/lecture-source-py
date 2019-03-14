.. _complex_and_trig:

.. include:: /_static/includes/lecture_howto_py.raw

.. index::
    single: python

**********************************
Complex numbers and trignometry
**********************************

.. contents:: :depth: 2

Overview
=========

The Euclidean, polar, and trigonometric forms of a complex number
:math:`z` are:

.. math::

   \begin{equation}
   z = x + iy = re^{i\theta} = r(\cos{\theta} + i \sin{\theta})
   \end{equation}

The second equality above is known as called **Euler's formula**

-  Euler contributed many other formulas too!

The complex conjugate :math:`\bar z` of :math:`z` is defined as

.. math:: \bar z = r e^{-i \theta} = r (\cos{\theta} - i \sin{\theta} )

:math:`x`\ is the **real** part of :math:`z` and :math:`y` is the
**imaginary** part of :math:`z`

:math:`\| z \|` = :math:`\bar z z = r` is called the **modulus**
of\ :math:`z`

:math:`r` is the Euclidean distance of vector :math:`(x,y)` from the
origin:

.. math::

   \begin{equation}
   r = |z| = \sqrt{x^2 + y^2}
   \end{equation}

:math:`\theta` is the angle of :math:`(x,y)` with respect to the real
axis

Evidently, the tangent of :math:`\theta`
is :math:`\left(\frac{y}{x}\right)`

Therefore,

.. math::

   \begin{equation}
   \theta = \tan^{-1} \Big( \frac{y}{x} \Big)
   \end{equation}

Three elementary trigonometric functions are

.. math::

   \begin{equation}
   \cos{\theta} = \frac{x}{r} = \frac{e^{i\theta} + e^{-i\theta}}{2} , \quad
   \sin{\theta} = \frac{y}{r} = \frac{e^{i\theta} - e^{-i\theta}}{2i} , \quad
   \tan{\theta} = \frac{x}{y}
   \end{equation}

We'll need the following imports

.. code:: python3

   import numpy as np
   import matplotlib.pyplot as plt
   

An example
-----------

Consider the complex number :math:`z = 1 + \sqrt{3} i`

For :math:`z = 1 + \sqrt{3} i`, :math:`x = 1`,\ :math:`y = \sqrt{3}`

It follows that :math:`r = 2` and
:math:`\theta = \tan^{-1}(\sqrt{3}) = \frac{\pi}{3} = 60^o`

Let's use Python to plot the trigonometric form of the complex number
:math:`z = 1 + \sqrt{3} i`

.. code:: python3
    
    # Abbreviate useful values and functions
    π = np.pi
    zeros = np.zeros
    ones = np.ones
    
    # Set parameters
    r = 2
    θ = π/3
    x = r * np.cos(θ)
    x_range = np.linspace(0, x, 1000)
    θ_range = np.linspace(0, θ, 1000)
    
    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    ax.plot((0, θ), (0, r), marker='o', color='b')          # plot r
    ax.plot(zeros(x_range.shape), x_range, color='b')       # plot x
    ax.plot(θ_range, x / np.cos(θ_range), color='b')        # plot y
    ax.plot(θ_range, ones(θ_range.shape) * 0.1, color='r')  # plot θ
    
    ax.margins(0) # Let the plot starts at origin
    
    ax.set_title("Trigonometry of complex numbers", va='bottom', fontsize='x-large')
    
    ax.set_rmax(2)
    ax.set_rticks((0.5, 1, 1.5, 2))  # less radial ticks
    ax.set_rlabel_position(-88.5)    # get radial labels away from plotted line
    
    ax.text(θ, r+0.01 , r'$z = x + iy = 1 + \sqrt{3}\, i$')   # label z
    ax.text(θ+0.2, 1 , '$r = 2$')                             # label r
    ax.text(0-0.2, 0.5, '$x = 1$')                            # label x
    ax.text(0.5, 1.2, r'$y = \sqrt{3}$')                      # label y
    ax.text(0.25, 0.15, r'$\theta = 60^o$')                   # label θ
    
    ax.grid(True)
    plt.show()


De Moivre's theorem
====================

de Moivre's theorem states that:

.. math::

   \begin{equation}
   (r(\cos{\theta} + i \sin{\theta}))^n = 
   r^n e^{in\theta} = 
   r^n(\cos{n\theta} + i \sin{n\theta})
   \end{equation}

To prove de Moivre's theorem, note that

.. math:: (r(\cos{\theta} + i \sin{\theta}))^n = \big( re^{i\theta} \big)^n

and compute

Applications of de Moivre's theorem
=====================================

Example 1
----------

We can use de Moivre's theorem to show that
:math:`r = \sqrt{x^2 + y^2}`

We have

.. math::

   \begin{aligned}
   1 &= e^{i\theta} e^{-i\theta} \\
   &= (\cos{\theta} + i \sin{\theta})(\cos{(\text{-}\theta)} + i \sin{(\text{-}\theta)}) \\
   &= (\cos{\theta} + i \sin{\theta})(\cos{\theta} - i \sin{\theta}) \\
   &= \cos^2{\theta} + \sin^2{\theta} \\
   &= \frac{x^2}{r^2} + \frac{y^2}{r^2}
   \end{aligned}

and thus

.. math::

   \begin{equation}
   x^2 + y^2 = r^2
   \end{equation}

We recogize this as a theorem of **Pythagoras**

Example 2
----------

Let :math:`z = re^{i\theta}`\ and\ :math:`\bar{z} = re^{-i\theta}` so
that :math:`\bar{z}`\ is the **complex conjugate** of\ :math:`z`

:math:`(z, \bar z)` form a **complex conjugate pair** of complex numbers

Let :math:`a = pe^{i\omega}` and :math:`\bar{a} = pe^{-i\omega}` be
another complex conjugate pair

We want to calculate :math:`x_n = az^n + \bar{a}\bar{z}^n`

To do so, we can apply de Moivre's formula

Thus,

.. math::

   \begin{aligned}
   x_n &= az^n + \bar{a}\bar{z}^n \\
   &= p e^{i\omega} (re^{i\theta})^n + p e^{-i\omega} (re^{-i\theta})^n \\
   &= pr^n e^{i (\omega + n\theta)} + pr^n e^{-i (\omega + n\theta)} \\
   &= pr^n [\cos{(\omega + n\theta)} + i \sin{(\omega + n\theta)} + 
            \cos{(\omega + n\theta)} - i \sin{(\omega + n\theta)}] \\
   &= 2 pr^n \cos{(\omega + n\theta)}
   \end{aligned}

Example 3
----------

Consider a **second-order linear difference equation**

.. math:: x_{n+2} = c_1 x_{n+1} + c_2 x_n 

whose **characteristic polynomial**

.. math:: z^2 - c_1 z - c_2 = 0

or

.. math:: (z^2 - c_1 z - c_2 ) = (z - z_1)(z- z_2) = 0

has roots :math:`z_1, z_1`

A **solution** is a sequence :math:`\{x_n\}_{n=0}^\infty` that satisfies
the difference equation

Under the following circumstances we can apply our example 2 formula to
solve the difference equation

-  the roots :math:`z_1, z_2` of the characteristic polynomial of the
   difference equation form a complex conjugate pair

-  the values :math:`x_0, x_1` are given initial conditions

To solve the difference equation, recall from example 2 that

.. math:: x_n = 2 pr^n \cos{(\omega + n\theta)}

where :math:`\omega, p` are coefficients to be determined from
information encoded in the initial conditions :math:`x_1, x_0`

Since
:math:`x_0 = 2 p \cos{\omega}` and :math:`x_1 = 2 pr \cos{(\omega + \theta)}`
the ratio of :math:`x_1` to :math:`x_0` is

.. math::

   \begin{equation}
   \frac{x_1}{x_0} = \frac{r \cos{(\omega + \theta)}}{\cos{\omega}}
   \end{equation}

We can solve this equation for :math:`\omega`\ then solve for\ :math:`p`
using :math:`x_0 = 2 pr^0 \cos{(\omega + n\theta)}`

With the ``sympy`` package in Python, we are able to solve and plot the
dynamics of :math:`x_n` given different values of :math:`n`

In this example, we set the initial values: - :math:`r = 0.9` -
:math:`\theta = \frac{1}{4}\pi` - :math:`x_0 = 4` -
:math:`x_1 = r \cdot 2\sqrt{2} = 1.8 \sqrt{2}`

We first numerically solve for :math:`\omega` and :math:`p` using
``nsolve`` in the ``sympy`` package based on the above initial
condition:

.. code:: python3

    from sympy import *
    
    # Set parameters
    r = 0.9
    θ = π/4
    x0 = 4
    x1 = 2 * r * sqrt(2)
    
    # Define symbols to be calculated
    ω, p = symbols('ω p', real=True)
    
    # Solve for ω
    ## Note: we choose the solution near 0
    eq1 = Eq(x1/x0 - r * cos(ω+θ) / cos(ω))
    ω = nsolve(eq1, ω, 0)
    ω = np.float(ω)
    print(f'ω = {ω:1.3f}')
    
    # Solve for p
    eq2 = Eq(x0 - 2 * p * cos(ω))
    p = nsolve(eq2, p, 0)
    p = np.float(p)
    print(f'p = {p:1.3f}')


Using the code above, we compute that
:math:`\omega = 0` and :math:`p = 2`

Then we plug in the values we solve for :math:`\omega` and :math:`p`
and plot the dynamic

.. code:: python3

    # Define range of n
    max_n = 30
    n = np.arange(0, max_n+1, 0.01)
    
    # Define x_n
    x = lambda n: 2 * p * r**n * np.cos(ω + n * θ)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(n, x(n))
    ax.set(xlim=(0, max_n), ylim=(-5, 5), xlabel='$n$', ylabel='$x_n$')
    
    ax.spines['bottom'].set_position('center') # Set x-axis in the middle of the plot
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ticklab = ax.xaxis.get_ticklabels()[0] # Set x-label position
    trans = ticklab.get_transform()
    ax.xaxis.set_label_coords(31, 0, transform=trans) 
    
    ticklab = ax.yaxis.get_ticklabels()[0] # Set y-label position
    trans = ticklab.get_transform()
    ax.yaxis.set_label_coords(0, 5, transform=trans)
    
    ax.grid()
    plt.show()


Trigonometric identities
------------------------

We can obtain a complete suite of trigonometric identities by
appropriately manipulating polar forms of complex numbers

We'll get many of them by deducing implications of the equality

.. math:: e^{i(\omega + \theta)} = e^{i\omega} e^{i\theta}

For example, we'll calculate identities for

:math:`\cos{(\omega + \theta)}` and :math:`\sin{(\omega + \theta)}`

Using the sine and cosine formulas presented at the beginning of this
notebook, we have:

.. math::

   \begin{aligned}
   \cos{(\omega + \theta)} = \frac{e^{i(\omega + \theta)} + e^{-i(\omega + \theta)}}{2} \\
   \sin{(\omega + \theta)} = \frac{e^{i(\omega + \theta)} - e^{-i(\omega + \theta)}}{2i}
   \end{aligned}

We can also obtain the trigonometric identities as follows:

.. math::

   \begin{aligned}
   \cos{(\omega + \theta)} + i \sin{(\omega + \theta)} 
   &= e^{i(\omega + \theta)} \\
   &= e^{i\omega} e^{i\theta} \\
   &= (\cos{\omega} + i \sin{\omega})(\cos{\theta} + i \sin{\theta}) \\
   &= (\cos{\omega}\cos{\theta} - \sin{\omega}\sin{\theta}) + 
   i (\cos{\omega}\sin{\theta} + \sin{\omega}\cos{\theta})
   \end{aligned}

Since both real and imaginary parts of the above formula should be
equal, we get:

.. math::

   \begin{aligned}
   \cos{(\omega + \theta)} = \cos{\omega}\cos{\theta} - \sin{\omega}\sin{\theta} \\
   \sin{(\omega + \theta)} = \cos{\omega}\sin{\theta} + \sin{\omega}\cos{\theta}
   \end{aligned}

The equations above are also known as the **angle sum identities**. We
can verify the equations using the ``simplify`` function in the
``sympy`` package:

.. code:: python3

    # Define symbols
    ω, θ = symbols('ω θ', real=True)
    
    # Verify
    print("cos(ω)cos(θ) - sin(ω)sin(θ) =", simplify(cos(ω)*cos(θ) - sin(ω) * sin(θ)))
    print("cos(ω)sin(θ) + sin(ω)cos(θ) =", simplify(cos(ω)*sin(θ) + sin(ω) * cos(θ)))


Trigonometric Integrals
-----------------------

We can also compute the trigonometric integrals using polar forms of
complex numbers

For example, we want to solve the following integral:

.. math:: \int_{-\pi}^{\pi} \cos(\omega) \sin(\omega) \, d\omega

Using Euler's formula, we have:

.. math::

   \begin{aligned}
   \int \cos(\omega) \sin(\omega) \, d\omega 
   &= 
   \int
   \frac{(e^{i\omega} + e^{-i\omega})}{2}
   \frac{(e^{i\omega} - e^{-i\omega})}{2i}
   \, d\omega  \\
   &=
   \frac{1}{4i}
   \int
   e^{2i\omega} - e^{-2i\omega}
   \, d\omega  \\
   &=
   \frac{1}{4i}
   \bigg( \frac{-i}{2} e^{2i\omega} - \frac{i}{2} e^{-2i\omega} + C_1 \bigg) \\
   &=
   -\frac{1}{8}
   \bigg[ \bigg(e^{i\omega}\bigg)^2 + \bigg(e^{-i\omega}\bigg)^2 - 2 \bigg] + C_2 \\
   &=
   -\frac{1}{8}  (e^{i\omega} - e^{-i\omega})^2  + C_2 \\
   &=
   \frac{1}{2} \bigg( \frac{e^{i\omega} - e^{-i\omega}}{2i} \bigg)^2 + C_2 \\
   &= \frac{1}{2} \sin^2(\omega) + C_2
   \end{aligned}

and thus:

.. math::

   \begin{aligned}
   \int_{-\pi}^{\pi} \cos(\omega) \sin(\omega) \, d\omega = 
   \frac{1}{2}\sin^2(\pi) - \frac{1}{2}\sin^2(-\pi) = 0
   \end{aligned}

We can verify the analytical as well as numerical results using
``integrate`` in the ``sympy`` package:

.. code:: python3

    # Set initial priting
    init_printing()
    
    ω = Symbol('ω')
    print('The analytical solution for integral of cos(ω)sin(ω) is:')
    integrate(cos(ω) * sin(ω), ω)


.. code:: python3

    print('The numerical solution for the integral of cos(ω)sin(ω) from -π to π is:')
    integrate(cos(ω) * sin(ω), (ω, -π, π))

