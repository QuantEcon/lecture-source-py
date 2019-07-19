.. _oop_intro:

.. include:: /_static/includes/header.raw

**************************************************
OOP I: Introduction to Object Oriented Programming
**************************************************

.. contents:: :depth: 2

Overview
============


`OOP <https://en.wikipedia.org/wiki/Object-oriented_programming>`_ is one of the major paradigms in programming.


The traditional programming paradigm (think Fortran, C, MATLAB, etc.) is called *procedural*.

It works as follows

* The program has a state corresponding to the values of its variables.

* Functions are called to act on these data.

* Data are passed back and forth via function calls.


In contrast, in the OOP paradigm

* data and functions are "bundled together" into "objects"

(Functions in this context are referred to as **methods**)







Python and OOP
--------------

Python is a pragmatic language that blends object-oriented and procedural styles, rather than taking a purist approach.

However, at a foundational level, Python *is* object-oriented.

In particular, in Python, *everything is an object*.

In this lecture, we explain what that statement means and why it matters.





Objects
=======

.. index::
    single: Python; Objects


In Python, an *object* is a collection of data and instructions held in computer memory that consists of

#. a type

#. a unique identity

#. data (i.e., content)

#. methods

These concepts are defined and discussed sequentially below.






.. _type:

Type
----

.. index::
    single: Python; Type

Python provides for different types of objects, to accommodate different categories of data.

For example

.. code-block:: python3

    s = 'This is a string'
    type(s)

.. code-block:: python3

    x = 42   # Now let's create an integer
    type(x)

The type of an object matters for many expressions.

For example, the addition operator between two strings means concatenation

.. code-block:: python3

    '300' + 'cc'

On the other hand, between two numbers it means ordinary addition

.. code-block:: python3

    300 + 400

Consider the following expression

.. code-block:: python3
    :class: skip-test

    '300' + 400


Here we are mixing types, and it's unclear to Python whether the user wants to

* convert ``'300'`` to an integer and then add it to ``400``, or

* convert ``400`` to string and then concatenate it with ``'300'``

Some languages might try to guess but Python is *strongly typed*

* Type is important, and implicit type conversion is rare.

* Python will respond instead by raising a ``TypeError``.


To avoid the error, you need to clarify by changing the relevant type.

For example,

.. code-block:: python3

    int('300') + 400   # To add as numbers, change the string to an integer



.. _identity:

Identity
--------

.. index::
    single: Python; Identity

In Python, each object has a unique identifier, which helps Python (and us) keep track of the object.

The identity of an object can be obtained via the ``id()`` function

.. code-block:: python3

    y = 2.5
    z = 2.5
    id(y)

.. code-block:: python3

    id(z)

In this example, ``y`` and ``z`` happen to have the same value (i.e., ``2.5``), but they are not the same object.

The identity of an object is in fact just the address of the object in memory.





Object Content: Data and Attributes
-----------------------------------

.. index::
    single: Python; Content

If we set ``x = 42`` then we create an object of type ``int`` that contains
the data ``42``.

In fact, it contains more, as the following example shows


.. code-block:: python3

    x = 42
    x

.. code-block:: python3

    x.imag

.. code-block:: python3

    x.__class__

When Python creates this integer object, it stores with it various auxiliary information, such as the imaginary part, and the type.

Any name following a dot is called an *attribute* of the object to the left of the dot.

* e.g.,``imag`` and ``__class__`` are attributes of ``x``.


We see from this example that objects have attributes that contain auxiliary information.


They also have attributes that act like functions, called *methods*.

These attributes are important, so let's discuss them in-depth.


.. _methods:

Methods
-------

.. index::
    single: Python; Methods

Methods are *functions that are bundled with objects*.



Formally, methods are attributes of objects that are callable (i.e., can be called as functions)

.. code-block:: python3

    x = ['foo', 'bar']
    callable(x.append)

.. code-block:: python3

    callable(x.__doc__)



Methods typically act on the data contained in the object they belong to, or combine that data with other data

.. code-block:: python3

    x = ['a', 'b']
    x.append('c')
    s = 'This is a string'
    s.upper()

.. code-block:: python3

    s.lower()

.. code-block:: python3

    s.replace('This', 'That')

A great deal of Python functionality is organized around method calls.

For example, consider the following piece of code

.. code-block:: python3

    x = ['a', 'b']
    x[0] = 'aa'  # Item assignment using square bracket notation
    x

It doesn't look like there are any methods used here, but in fact the square bracket assignment notation is just a convenient interface to a method call.

What actually happens is that Python calls the ``__setitem__`` method, as follows

.. code-block:: python3

    x = ['a', 'b']
    x.__setitem__(0, 'aa')  # Equivalent to x[0] = 'aa'
    x

(If you wanted to you could modify the ``__setitem__`` method, so that square bracket assignment does something totally different)




Summary
==========

In Python, *everything in memory is treated as an object*.

This includes not just lists, strings, etc., but also less obvious things, such as

* functions (once they have been read into memory)

* modules  (ditto)

* files opened for reading or writing

* integers, etc.

Consider, for example, functions.

When Python reads a function definition, it creates a **function object** and stores it in memory.


The following code illustrates

.. code-block:: python3

    def f(x): return x**2
    f


.. code-block:: python3

    type(f)

.. code-block:: python3

    id(f)

.. code-block:: python3

    f.__name__

We can see that ``f`` has type, identity, attributes and so on---just like any other object.

It also has methods.

One example is the ``__call__`` method, which just evaluates the function

.. code-block:: python3

    f.__call__(3)

Another is the ``__dir__`` method, which returns a list of attributes.


Modules loaded into memory are also treated as objects

.. code-block:: python3

    import math

    id(math)


This uniform treatment of data in Python (everything is an object) helps keep the language simple and consistent.
