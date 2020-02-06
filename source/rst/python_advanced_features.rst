.. _python_advanced_features:

.. include:: /_static/includes/header.raw

**********************
More Language Features
**********************

.. contents:: :depth: 2

Overview
========

With this last lecture, our advice is to **skip it on first pass**, unless you have a burning desire to read it.

It's here

#. as a reference, so we can link back to it when required, and

#. for those who have worked through a number of applications, and now want to learn more about the Python language

A variety of topics are treated in the lecture, including generators, exceptions and descriptors.






Iterables and Iterators
=======================

.. index::
    single: Python; Iteration

We've :ref:`already said something <iterating_version_1>` about iterating in Python.

Now let's look more closely at how it all works, focusing in Python's implementation of the ``for`` loop.


Iterators
---------

.. index::
    single: Python; Iterators

Iterators are a uniform interface to stepping through elements in a collection.

Here we'll talk about using iterators---later we'll learn how to build our own.

Formally, an *iterator* is an object with a ``__next__`` method.

For example, file objects are iterators .

To see this, let's have another look at the :ref:`US cities data <us_cities_data>`,
which is written to the present working directory in the following cell

.. code-block:: ipython

    %%file us_cities.txt
    new york: 8244910
    los angeles: 3819702
    chicago: 2707120
    houston: 2145146
    philadelphia: 1536471
    phoenix: 1469471
    san antonio: 1359758
    san diego: 1326179
    dallas: 1223229 


.. code-block:: python3

    f = open('us_cities.txt')
    f.__next__()

    
.. code-block:: python3

    f.__next__()
    


We see that file objects do indeed have a ``__next__`` method, and that calling this method returns the next line in the file.

The next method can also be accessed via the builtin function ``next()``,
which directly calls this method

.. code-block:: python3

    next(f)

The objects returned by ``enumerate()`` are also iterators 

.. code-block:: python3
    
    e = enumerate(['foo', 'bar'])
    next(e)
    
.. code-block:: python3
    
    next(e)

as are the reader objects from the ``csv`` module .

Let's create a small csv file that contains data from the NIKKEI index

.. code-block:: ipython

    %%file test_table.csv
    Date,Open,High,Low,Close,Volume,Adj Close
    2009-05-21,9280.35,9286.35,9189.92,9264.15,133200,9264.15
    2009-05-20,9372.72,9399.40,9311.61,9344.64,143200,9344.64
    2009-05-19,9172.56,9326.75,9166.97,9290.29,167000,9290.29
    2009-05-18,9167.05,9167.82,8997.74,9038.69,147800,9038.69
    2009-05-15,9150.21,9272.08,9140.90,9265.02,172000,9265.02
    2009-05-14,9212.30,9223.77,9052.41,9093.73,169400,9093.73
    2009-05-13,9305.79,9379.47,9278.89,9340.49,176000,9340.49
    2009-05-12,9358.25,9389.61,9298.61,9298.61,188400,9298.61
    2009-05-11,9460.72,9503.91,9342.75,9451.98,230800,9451.98
    2009-05-08,9351.40,9464.43,9349.57,9432.83,220200,9432.83

.. code-block:: python3

    from csv import reader

    f = open('test_table.csv', 'r')  
    nikkei_data = reader(f) 
    next(nikkei_data)
    
.. code-block:: python3


    next(nikkei_data)


Iterators in For Loops
----------------------

.. index::
    single: Python; Iterators

All iterators can be placed to the right of the ``in`` keyword in ``for`` loop statements.

In fact this is how the ``for`` loop works:  If we write

.. code-block:: python3
    :class: no-execute

    for x in iterator:
        <code block>

then the interpreter

* calls ``iterator.___next___()`` and binds ``x`` to the result
* executes the code block
* repeats until a ``StopIteration`` error occurs

So now you know how this magical looking syntax works

.. code-block:: python3
    :class: no-execute

    f = open('somefile.txt', 'r')
    for line in f:
        # do something


The interpreter just keeps 

#. calling ``f.__next__()`` and binding ``line`` to the result
#. executing the body of the loop

This continues until a ``StopIteration`` error occurs.




Iterables
---------

.. index::
    single: Python; Iterables

You already know that we can put a Python list to the right of ``in`` in a ``for`` loop 

.. code-block:: python3

    for i in ['spam', 'eggs']:
        print(i)

So does that mean that a list is an iterator?

The answer is no

.. code-block:: python3

    x = ['foo', 'bar']
    type(x)
    
    
.. code-block:: python3
    :class: skip-test

    next(x)
    

So why can we iterate over a list in a ``for`` loop?

The reason is that a list is *iterable* (as opposed to an iterator).

Formally, an object is iterable if it can be converted to an iterator using the built-in function ``iter()``.

Lists are one such object 

.. code-block:: python3

    x = ['foo', 'bar']
    type(x)
    
    
.. code-block:: python3
    
    y = iter(x)
    type(y)
    
    
.. code-block:: python3

    next(y)  
    
.. code-block:: python3

    next(y)
    
.. code-block:: python3
    :class: skip-test

    next(y)    


Many other objects are iterable, such as dictionaries and tuples.

Of course, not all objects are iterable 

.. code-block:: python3
    :class: skip-test

    iter(42)


To conclude our discussion of ``for`` loops

* ``for`` loops work on either iterators or iterables.
* In the second case, the iterable is converted into an iterator before the loop starts.


Iterators and built-ins
-----------------------

.. index::
    single: Python; Iterators

Some built-in functions that act on sequences also work with iterables

* ``max()``, ``min()``, ``sum()``, ``all()``, ``any()``


For example 

.. code-block:: python3

    x = [10, -10]
    max(x)
    
.. code-block:: python3
    
    y = iter(x)
    type(y)    
    
.. code-block:: python3
    
    max(y)


One thing to remember about iterators is that they are depleted by use

.. code-block:: python3

    x = [10, -10]
    y = iter(x)
    max(y)


.. code-block:: python3
    :class: skip-test
    
    max(y)
    

.. _name_res:


Names and Name Resolution
=========================


Variable Names in Python
------------------------

.. index::
    single: Python; Variable Names

Consider the Python statement 

.. code-block:: python3

    x = 42

We now know that when this statement is executed, Python creates an object of
type ``int`` in your computer's memory, containing

* the value ``42``
* some associated attributes

But what is ``x`` itself?

In Python, ``x`` is called a *name*, and the statement ``x = 42`` *binds* the name ``x`` to the integer object we have just discussed.

Under the hood, this process of binding names to objects is implemented as a dictionary---more about this in a moment.

There is no problem binding two or more names to the one object, regardless of what that object is 

.. code-block:: python3

    def f(string):      # Create a function called f
        print(string)   # that prints any string it's passed

    g = f
    id(g) == id(f)
    
.. code-block:: python3

    g('test')

In the first step, a function object is created, and the name ``f`` is bound to it.

After binding the name ``g`` to the same object, we can use it anywhere we would use ``f``.

What happens when the number of names bound to an object goes to zero?

Here's an example of this situation, where the name ``x`` is first bound to one object and then rebound to another 

.. code-block:: python3

    x = 'foo'
    id(x)
    
.. code-block:: python3

    x = 'bar'  # No names bound to the first object

What happens here is that the first object is garbage collected.

In other words, the memory slot that stores that object is deallocated, and returned to the operating system.



Namespaces
----------

.. index::
    single: Python; Namespaces

Recall from the preceding discussion that the statement

.. code-block:: python3

    x = 42

binds the name ``x`` to the integer object on the right-hand side.

We also mentioned that this process of binding ``x`` to the correct object is implemented as a dictionary.

This dictionary is called a *namespace*.

**Definition:** A namespace is a symbol table that maps names to objects in memory.

Python uses multiple namespaces, creating them on the fly as necessary .

For example, every time we import a module, Python creates a namespace for that module.

To see this in action, suppose we write a script ``math2.py`` with a single line

.. code-block:: python3

    %%file math2.py
    pi = 'foobar'

Now we start the Python interpreter and import it 

.. code-block:: python3

    import math2

Next let's import the ``math`` module from the standard library 

.. code-block:: python3

    import math

Both of these modules have an attribute called ``pi``

.. code-block:: python3

    math.pi
    
.. code-block:: python3

    math2.pi

These two different bindings of ``pi`` exist in different namespaces, each one implemented as a dictionary.

We can look at the dictionary directly, using ``module_name.__dict__`` 

.. code-block:: python3

    import math

    math.__dict__.items()
    
.. code-block:: python3

    import math2

    math2.__dict__.items()


As you know, we access elements of the namespace using the dotted attribute notation 

.. code-block:: python3

    math.pi

In fact this is entirely equivalent to ``math.__dict__['pi']`` 

.. code-block:: python3

    math.__dict__['pi'] == math.pi


Viewing Namespaces
------------------

As we saw above, the ``math`` namespace can be printed by typing ``math.__dict__``.

Another way to see its contents is to type ``vars(math)``

.. code-block:: python3

    vars(math).items()

If you just want to see the names, you can type

.. code-block:: python3

    dir(math)[0:10]

Notice the special names ``__doc__`` and ``__name__``.

These are initialized in the namespace when any module is imported

* ``__doc__`` is the doc string of the module
* ``__name__`` is the name of the module

.. code-block:: python3

    print(math.__doc__)
    
.. code-block:: python3

    math.__name__

Interactive Sessions
--------------------

.. index::
    single: Python; Interpreter

In Python, **all** code executed by the interpreter runs in some module.

What about commands typed at the prompt?

These are also regarded as being executed within a module --- in this case, a module called ``__main__``.

To check this, we can look at the current module name via the value of ``__name__`` given at the prompt

.. code-block:: python3

    print(__name__)

When we run a script using IPython's ``run`` command, the contents of the file are executed as part of ``__main__`` too.

To see this, let's create a file ``mod.py`` that prints its own ``__name__`` attribute

.. code-block:: ipython

    %%file mod.py
    print(__name__)

Now let's look at two different ways of running it in IPython 

.. code-block:: python3

    import mod  # Standard import
    
.. code-block:: ipython
    
    %run mod.py  # Run interactively
  
In the second case, the code is executed as part of ``__main__``, so ``__name__`` is equal to ``__main__``.

To see the contents of the namespace of ``__main__`` we use ``vars()`` rather than ``vars(__main__)`` .

If you do this in IPython, you will see a whole lot of variables that IPython
needs, and has initialized when you started up your session.

If you prefer to see only the variables you have initialized, use ``whos``

.. code-block:: ipython

    x = 2
    y = 3

    import numpy as np

    %whos

The Global Namespace
--------------------

.. index::
    single: Python; Namespace (Global)

Python documentation often makes reference to the "global namespace".

The global namespace is *the namespace of the module currently being executed*.

For example, suppose that we start the interpreter and begin making assignments .

We are now working in the module ``__main__``, and hence the namespace for ``__main__`` is the global namespace.

Next, we import a module called ``amodule`` 

.. code-block:: python3
    :class: no-execute

    import amodule

At this point, the interpreter creates a namespace for the module ``amodule`` and starts executing commands in the module.

While this occurs, the namespace ``amodule.__dict__`` is the global namespace.

Once execution of the module finishes, the interpreter returns to the module from where the import statement was made.

In this case it's ``__main__``, so the namespace of ``__main__`` again becomes the global namespace.


Local Namespaces
----------------

.. index::
    single: Python; Namespace (Local)

Important fact: When we call a function, the interpreter creates a *local namespace* for that function, and registers the variables in that namespace.

The reason for this will be explained in just a moment.

Variables in the local namespace are called *local variables*.

After the function returns, the namespace is deallocated and lost.

While the function is executing, we can view the contents of the local namespace with ``locals()``.
 
For example, consider

.. code-block:: python3

    def f(x):
        a = 2
        print(locals())
        return a * x


Now let's call the function 

.. code-block:: python3

    f(1)

You can see the local namespace of ``f`` before it is destroyed.





The ``__builtins__`` Namespace
------------------------------

.. index::
    single: Python; Namespace (__builtins__)

We have been using various built-in functions, such as ``max(), dir(), str(), list(), len(), range(), type()``, etc.

How does access to these names work?

* These definitions are stored in a module called ``__builtin__``.
* They have there own namespace called ``__builtins__``.

.. code-block:: python3

    dir()[0:10]
    
.. code-block:: python3
    
    dir(__builtins__)[0:10]

We can access elements of the namespace as follows 

.. code-block:: python3

    __builtins__.max

But ``__builtins__`` is special, because we can always access them directly as well 

.. code-block:: python3

    max
    
.. code-block:: python3

    __builtins__.max == max


The next section explains how this works ...


Name Resolution
---------------

.. index::
    single: Python; Namespace (Resolution)

Namespaces are great because they help us organize variable names.

(Type ``import this`` at the prompt and look at the last item that's printed)

However, we do need to understand how the Python interpreter works with multiple namespaces .

At any point of execution, there are in fact at least two namespaces that can be accessed directly.

("Accessed directly" means without using a dot, as in  ``pi`` rather than ``math.pi``)

These namespaces are 

* The global namespace (of the module being executed)
* The builtin namespace

If the interpreter is executing a function, then the directly accessible namespaces are 

* The local namespace of the function
* The global namespace (of the module being executed)
* The builtin namespace

Sometimes functions are defined within other functions, like so

.. code-block:: python3

    def f():
        a = 2
        def g():
            b = 4
            print(a * b)
        g()


Here ``f`` is the *enclosing function* for ``g``, and each function gets its
own namespaces.

Now we can give the rule for how namespace resolution works:

The order in which the interpreter searches for names is

#. the local namespace (if it exists)
#. the hierarchy of enclosing namespaces (if they exist)
#. the global namespace
#. the builtin namespace

If the name is not in any of these namespaces, the interpreter raises a ``NameError``.

This is called the **LEGB rule** (local, enclosing, global, builtin).

Here's an example that helps to illustrate .

Consider a script ``test.py`` that looks as follows

.. code-block:: python3

    %%file test.py
    def g(x):
        a = 1
        x = x + a
        return x
    
    a = 0
    y = g(10)
    print("a = ", a, "y = ", y)


What happens when we run this script?  

.. code-block:: ipython

    %run test.py

.. code-block:: python3
    :class: skip-test

    x
    


First,

* The global namespace ``{}`` is created.
* The function object is created, and ``g`` is bound to it within the global namespace.
* The name ``a`` is bound to ``0``, again in the global namespace.

Next ``g`` is called via ``y = g(10)``, leading to the following sequence of actions

* The local namespace for the function is created.
* Local names ``x`` and ``a`` are bound, so that the local namespace becomes ``{'x': 10, 'a': 1}``.
* Statement ``x = x + a`` uses the local ``a`` and local ``x`` to compute ``x + a``, and binds local name ``x`` to the result.
* This value is returned, and ``y`` is bound to it in the global namespace.
* Local ``x`` and ``a`` are discarded (and the local namespace is deallocated).

Note that the global ``a`` was not affected by the local ``a``.

.. _mutable_vs_immutable:

:index:`Mutable` Versus :index:`Immutable` Parameters
-----------------------------------------------------

This is a good time to say a little more about mutable vs immutable objects.

Consider the code segment

.. code-block:: python3

    def f(x):
        x = x + 1
        return x

    x = 1
    print(f(x), x)


We now understand what will happen here: The code prints ``2`` as the value of ``f(x)`` and ``1`` as the value of ``x``.

First ``f`` and ``x`` are registered in the global namespace.

The call ``f(x)`` creates a local namespace and adds ``x`` to it, bound to ``1``.

Next, this local ``x`` is rebound to the new integer object ``2``, and this value is returned.

None of this affects the global ``x``.

However, it's a different story when we use a **mutable** data type such as a list

.. code-block:: python3

    def f(x):
        x[0] = x[0] + 1
        return x

    x = [1]
    print(f(x), x)


This prints ``[2]`` as the value of ``f(x)`` and *same* for ``x``.

Here's what happens

* ``f`` is registered as a function in the global namespace

* ``x`` bound to ``[1]`` in the global namespace

* The call ``f(x)``

    * Creates a local namespace

    * Adds ``x`` to local namespace, bound to ``[1]``

    * The list ``[1]`` is modified to ``[2]``

    * Returns the list ``[2]``

    * The local namespace is deallocated, and local ``x`` is lost

* Global ``x`` has been modified






Handling Errors
===============

.. index::
    single: Python; Handling Errors

Sometimes it's possible to anticipate errors as we're writing code.

For example, the unbiased sample variance of sample :math:`y_1, \ldots, y_n`
is defined as

.. math::

    s^2 := \frac{1}{n-1} \sum_{i=1}^n (y_i - \bar y)^2
    \qquad \bar y = \text{ sample mean}


This can be calculated in NumPy using ``np.var``.

But if you were writing a function to handle such a calculation, you might
anticipate a divide-by-zero error when the sample size is one.

One possible action is to do nothing --- the program will just crash, and spit out an error message.

But sometimes it's worth writing your code in a way that anticipates and deals with runtime errors that you think might arise.

Why?

* Because the debugging information provided by the interpreter is often less useful than the information
  on possible errors you have in your head when writing code.

* Because errors causing execution to stop are frustrating if you're in the middle of a large computation.

* Because it's reduces confidence in your code on the part of your users (if you are writing for others).



Assertions
----------

.. index::
    single: Python; Assertions

A relatively easy way to handle checks is with the ``assert`` keyword.

For example, pretend for a moment that the ``np.var`` function doesn't
exist and we need to write our own

.. code-block:: python3

    def var(y):
        n = len(y)
        assert n > 1, 'Sample size must be greater than one.'
        return np.sum((y - y.mean())**2) / float(n-1)

If we run this with an array of length one, the program will terminate and
print our error message

.. code-block:: python3
    :class: skip-test

    var([1])
    


The advantage is that we can

* fail early, as soon as we know there will be a problem

* supply specific information on why a program is failing

Handling Errors During Runtime
------------------------------

.. index::
    single: Python; Runtime Errors

The approach used above is a bit limited, because it always leads to
termination.

Sometimes we can handle errors more gracefully, by treating special cases.

Let's look at how this is done.

Exceptions
^^^^^^^^^^

.. index::
    single: Python; Exceptions

Here's an example of a common error type

.. code-block:: python3
    :class: skip-test
    
    def f:


Since illegal syntax cannot be executed, a syntax error terminates execution of the program.

Here's a different kind of error, unrelated to syntax

.. code-block:: python3
    :class: skip-test

    1 / 0


Here's another

.. code-block:: python3
    :class: skip-test

    x1 = y1


And another

.. code-block:: python3
    :class: skip-test

    'foo' + 6

And another

.. code-block:: python3
    :class: skip-test

    X = []
    x = X[0]



On each occasion, the interpreter informs us of the error type

* ``NameError``, ``TypeError``, ``IndexError``, ``ZeroDivisionError``, etc.

In Python, these errors are called *exceptions*.

Catching Exceptions
^^^^^^^^^^^^^^^^^^^

We can catch and deal with exceptions using ``try`` -- ``except`` blocks.

Here's a simple example

.. code-block:: python3

    def f(x):
        try:
            return 1.0 / x
        except ZeroDivisionError:
            print('Error: division by zero.  Returned None')
        return None


When we call ``f`` we get the following output

.. code-block:: python3

    f(2)
    
.. code-block:: python3

    f(0)
    
.. code-block:: python3

    f(0.0)


The error is caught and execution of the program is not terminated.

Note that other error types are not caught.

If we are worried the user might pass in a string, we can catch that error too

.. code-block:: python3

    def f(x):
        try:
            return 1.0 / x
        except ZeroDivisionError:
            print('Error: Division by zero.  Returned None')
        except TypeError:
            print('Error: Unsupported operation.  Returned None')
        return None

Here's what happens

.. code-block:: python3

    f(2)
    
.. code-block:: python3
    
    f(0)
    
.. code-block:: python3

    f('foo')


If we feel lazy we can catch these errors together

.. code-block:: python3

    def f(x):
        try:
            return 1.0 / x
        except (TypeError, ZeroDivisionError):
            print('Error: Unsupported operation.  Returned None')
        return None


Here's what happens


.. code-block:: python3

    f(2)
    
.. code-block:: python3

    f(0)
    
.. code-block:: python3

    f('foo')


If we feel extra lazy we can catch all error types as follows

.. code-block:: python3

    def f(x):
        try:
            return 1.0 / x
        except:
            print('Error.  Returned None')
        return None

In general it's better to be specific.



Decorators and Descriptors
==========================

.. index::
    single: Python; Decorators

.. index::
    single: Python; Descriptors

Let's look at some special syntax elements that are routinely used by Python developers.

You might not need the following concepts immediately, but you will see them
in other people's code.

Hence you need to understand them at some stage of your Python education.


Decorators
----------

.. index::
    single: Python; Decorators

Decorators are a bit of syntactic sugar that, while easily avoided, have turned out to be popular.

It's very easy to say what decorators do.

On the other hand it takes a bit of effort to explain *why* you might use them.

An Example
^^^^^^^^^^

Suppose we are working on a program that looks something like this

.. code-block:: python3

    import numpy as np

    def f(x):
        return np.log(np.log(x))

    def g(x):
        return np.sqrt(42 * x)

    # Program continues with various calculations using f and g

Now suppose there's a problem: occasionally negative numbers get fed to ``f`` and ``g`` in the calculations that follow.

If you try it, you'll see that when these functions are called with negative numbers they return a NumPy object called ``nan`` .

This stands for "not a number" (and indicates that you are trying to evaluate
a mathematical function at a point where it is not defined).

Perhaps this isn't what we want, because it causes other problems that are hard to pick up later on.

Suppose that instead we want the program to terminate whenever this happens, with a sensible error message.

This change is easy enough to implement

.. code-block:: python3

    import numpy as np

    def f(x):
        assert x >= 0, "Argument must be nonnegative"
        return np.log(np.log(x))

    def g(x):
        assert x >= 0, "Argument must be nonnegative"
        return np.sqrt(42 * x)

    # Program continues with various calculations using f and g


Notice however that there is some repetition here, in the form of two identical lines of code.

Repetition makes our code longer and harder to maintain, and hence is
something we try hard to avoid.

Here it's not a big deal, but imagine now that instead of just ``f`` and ``g``, we have 20 such functions that we need to modify in exactly the same way.

This means we need to repeat the test logic (i.e., the ``assert`` line testing nonnegativity) 20 times.

The situation is still worse if the test logic is longer and more complicated.

In this kind of scenario the following approach would be neater

.. code-block:: python3

    import numpy as np

    def check_nonneg(func):
        def safe_function(x):
            assert x >= 0, "Argument must be nonnegative"
            return func(x)
        return safe_function

    def f(x):
        return np.log(np.log(x))

    def g(x):
        return np.sqrt(42 * x)

    f = check_nonneg(f)
    g = check_nonneg(g)
    # Program continues with various calculations using f and g

This looks complicated so let's work through it slowly.

To unravel the logic, consider what happens when we say ``f = check_nonneg(f)``.

This calls the function ``check_nonneg`` with parameter ``func`` set equal to ``f``.

Now ``check_nonneg`` creates a new function called ``safe_function`` that
verifies ``x`` as nonnegative and then calls ``func`` on it (which is the same as ``f``).

Finally, the global name ``f`` is then set equal to ``safe_function``.

Now the behavior of ``f`` is as we desire, and the same is true of ``g``.

At the same time, the test logic is written only once.


Enter Decorators
^^^^^^^^^^^^^^^^

.. index::
    single: Python; Decorators

The last version of our code is still not ideal.

For example, if someone is reading our code and wants to know how
``f`` works, they will be looking for the function definition, which is

.. code-block:: python3

    def f(x):
        return np.log(np.log(x))

They may well miss the line ``f = check_nonneg(f)``.

For this and other reasons, decorators were introduced to Python.

With decorators, we can replace the lines

.. code-block:: python3

    def f(x):
        return np.log(np.log(x))

    def g(x):
        return np.sqrt(42 * x)

    f = check_nonneg(f)
    g = check_nonneg(g)

with

.. code-block:: python3

    @check_nonneg
    def f(x):
        return np.log(np.log(x))

    @check_nonneg
    def g(x):
        return np.sqrt(42 * x)

These two pieces of code do exactly the same thing.

If they do the same thing, do we really need decorator syntax?

Well, notice that the decorators sit right on top of the function definitions.

Hence anyone looking at the definition of the function will see them and be
aware that the function is modified.

In the opinion of many people, this makes the decorator syntax a significant improvement to the language.



.. _descriptors:

Descriptors
-----------

.. index::
    single: Python; Descriptors

Descriptors solve a common problem regarding management of variables.

To understand the issue, consider a ``Car`` class, that simulates a car.

Suppose that this class defines the variables ``miles`` and ``kms``, which give the distance traveled in miles
and kilometers respectively.

A highly simplified version of the class might look as follows

.. code-block:: python3

    class Car:

        def __init__(self, miles=1000):
            self.miles = miles
            self.kms = miles * 1.61

        # Some other functionality, details omitted

One potential problem we might have here is that a user alters one of these
variables but not the other

.. code-block:: python3

    car = Car()
    car.miles
    
.. code-block:: python3

    car.kms
    
.. code-block:: python3

    car.miles = 6000
    car.kms


In the last two lines we see that ``miles`` and ``kms`` are out of sync.

What we really want is some mechanism whereby each time a user sets one of these variables, *the other is automatically updated*.

A Solution
^^^^^^^^^^

In Python, this issue is solved using *descriptors*.

A descriptor is just a Python object that implements certain methods.

These methods are triggered when the object is accessed through dotted attribute notation.

The best way to understand this is to see it in action.

Consider this alternative version of the ``Car`` class

.. code-block:: python3

    class Car:

        def __init__(self, miles=1000):
            self._miles = miles
            self._kms = miles * 1.61

        def set_miles(self, value):
            self._miles = value
            self._kms = value * 1.61

        def set_kms(self, value):
            self._kms = value
            self._miles = value / 1.61

        def get_miles(self):
            return self._miles

        def get_kms(self):
            return self._kms

        miles = property(get_miles, set_miles)
        kms = property(get_kms, set_kms)


First let's check that we get the desired behavior

.. code-block:: python3

    car = Car()
    car.miles
    
.. code-block:: python3

    car.miles = 6000
    car.kms

Yep, that's what we want --- ``car.kms`` is automatically updated.

How it Works
^^^^^^^^^^^^

The names ``_miles`` and ``_kms`` are arbitrary names we are using to store the values of the variables.

The objects ``miles`` and ``kms`` are *properties*, a common kind of descriptor.

The methods ``get_miles``, ``set_miles``, ``get_kms`` and ``set_kms`` define
what happens when you get (i.e. access) or set (bind) these variables

* So-called "getter" and "setter" methods.

The builtin Python function ``property`` takes getter and setter methods and creates a property.

For example, after ``car`` is created as an instance of ``Car``, the object ``car.miles`` is a property.

Being a property, when we set its value via ``car.miles = 6000`` its setter
method is triggered --- in this case ``set_miles``.



Decorators and Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
    single: Python; Decorators

.. index::
    single: Python; Properties

These days its very common to see the ``property`` function used via a decorator.

Here's another version of our ``Car`` class that works as before but now uses
decorators to set up the properties

.. code-block:: python3

    class Car:

        def __init__(self, miles=1000):
            self._miles = miles
            self._kms = miles * 1.61

        @property
        def miles(self):
            return self._miles

        @property
        def kms(self):
            return self._kms

        @miles.setter
        def miles(self, value):
            self._miles = value
            self._kms = value * 1.61

        @kms.setter
        def kms(self, value):
            self._kms = value
            self._miles = value / 1.61


We won't go through all the details here.

For further information you can refer to the `descriptor documentation <https://docs.python.org/3/howto/descriptor.html>`_.


.. _paf_generators:

Generators
==========

.. index::
    single: Python; Generators

A generator is a kind of iterator (i.e., it works with a ``next`` function).

We will study two ways to build generators: generator expressions and generator functions.

Generator Expressions
---------------------

The easiest way to build generators is using *generator expressions*.

Just like a list comprehension, but with round brackets.

Here is the list comprehension:


.. code-block:: python3

    singular = ('dog', 'cat', 'bird')
    type(singular)

.. code-block:: python3

    plural = [string + 's' for string in singular]
    plural
    
.. code-block:: python3
    
    type(plural)

And here is the generator expression

.. code-block:: python3

    singular = ('dog', 'cat', 'bird')
    plural = (string + 's' for string in singular)
    type(plural)
    
.. code-block:: python3
    
    next(plural)
    
.. code-block:: python3
    
    next(plural)
    
.. code-block:: python3
    
    next(plural)

Since ``sum()`` can be called on iterators, we can do this

.. code-block:: python3

    sum((x * x for x in range(10)))

The function ``sum()`` calls ``next()`` to get the items, adds successive terms.

In fact, we can omit the outer brackets in this case

.. code-block:: python3

    sum(x * x for x in range(10))


Generator Functions
-------------------

.. index::
    single: Python; Generator Functions

The most flexible way to create generator objects is to use generator functions.

Let's look at some examples.

Example 1
^^^^^^^^^

Here's a very simple example of a generator function

.. code-block:: python3

    def f():
        yield 'start'
        yield 'middle'
        yield 'end'


It looks like a function, but uses a keyword ``yield`` that we haven't met before.

Let's see how it works after running this code

.. code-block:: python3

    type(f)
    
.. code-block:: python3

    gen = f()
    gen

.. code-block:: python3

    next(gen)
    
.. code-block:: python3

    next(gen)
    
.. code-block:: python3

    next(gen)
    
.. code-block:: python3
    :class: skip-test

    next(gen)



The generator function ``f()`` is used to create generator objects (in this case ``gen``).

Generators are iterators, because they support a ``next`` method.

The first call to ``next(gen)``

* Executes code in the body of ``f()`` until it meets a ``yield`` statement.
* Returns that value to the caller of ``next(gen)``.

The second call to ``next(gen)`` starts executing *from the next line*

.. code-block:: python3

    def f():
        yield 'start'
        yield 'middle'  # This line!
        yield 'end'

and continues until the next ``yield`` statement.

At that point it returns the value following ``yield`` to the caller of ``next(gen)``, and so on.

When the code block ends, the generator throws a ``StopIteration`` error.


Example 2
^^^^^^^^^

Our next example receives an argument ``x`` from the caller

.. code-block:: python3

   def g(x):
       while x < 100:
           yield x
           x = x * x

Let's see how it works

.. code-block:: python3
    
    g
    
.. code-block:: python3
    
    gen = g(2)
    type(gen)
    
.. code-block:: python3
    
    next(gen)
    
.. code-block:: python3
    
    next(gen)
    
.. code-block:: python3
    
    next(gen)
    
.. code-block:: python3
    :class: skip-test
    
    next(gen)


The call ``gen = g(2)`` binds ``gen`` to a generator.

Inside the generator, the name ``x`` is bound to ``2``.

When we call ``next(gen)``

* The body of ``g()`` executes until the line ``yield x``, and the value of ``x`` is returned.

Note that value of ``x`` is retained inside the generator.

When we call ``next(gen)`` again, execution continues *from where it left off*

.. code-block:: python3

    def g(x):
        while x < 100:
            yield x
            x = x * x  # execution continues from here

When ``x < 100`` fails, the generator throws a ``StopIteration`` error.

Incidentally, the loop inside the generator can be infinite

.. code-block:: python3

    def g(x):
        while 1:
            yield x
            x = x * x


Advantages of Iterators
-----------------------

What's the advantage of using an iterator here?

Suppose we want to sample a binomial(n,0.5).

One way to do it is as follows

.. code-block:: python3

    import random
    n = 10000000
    draws = [random.uniform(0, 1) < 0.5 for i in range(n)]
    sum(draws)


But we are creating two huge lists here,  ``range(n)`` and ``draws``.

This uses lots of memory and is very slow.

If we make ``n`` even bigger then this happens

.. code-block:: python3
    :class: skip-test

    n = 100000000
    draws = [random.uniform(0, 1) < 0.5 for i in range(n)]


We can avoid these problems using iterators.

Here is the generator function

.. code-block:: python3

    def f(n):
        i = 1
        while i <= n:
            yield random.uniform(0, 1) < 0.5
            i += 1

Now let's do the sum

.. code-block:: python3

    n = 10000000
    draws = f(n)
    draws
    
.. code-block:: python3

    sum(draws)


In summary, iterables

* avoid the need to create big lists/tuples, and
* provide a uniform interface to iteration that can be used transparently in ``for`` loops



.. _recursive_functions:

Recursive Function Calls
========================

.. index::
    single: Python; Recursion

This is not something that you will use every day, but it is still useful --- you should learn it at some stage.

Basically, a recursive function is a function that calls itself.

For example, consider the problem of computing :math:`x_t` for some t when

.. math::
    :label: xseqdoub

    x_{t+1} = 2 x_t, \quad x_0 = 1


Obviously the answer is :math:`2^t`.

We can compute this easily enough with a loop

.. code-block:: python3

    def x_loop(t):
        x = 1
        for i in range(t):
            x = 2 * x
        return x

We can also use a recursive solution, as follows

.. code-block:: python3

    def x(t):
        if t == 0:
            return 1
        else:
            return 2 * x(t-1)


What happens here is that each successive call uses it's own *frame* in the *stack*

* a frame is where the local variables of a given function call are held
* stack is memory used to process function calls
  * a First In Last Out (FILO) queue

This example is somewhat contrived, since the first (iterative) solution would usually be preferred to the recursive solution.

We'll meet less contrived applications of recursion later on.


Exercises
=========

Exercise 1
----------

The Fibonacci numbers are defined by

.. math::
    :label: fib

    x_{t+1} = x_t + x_{t-1}, \quad x_0 = 0, \; x_1 = 1


The first few numbers in the sequence are :math:`0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55`.

Write a function to recursively compute the :math:`t`-th Fibonacci number for any :math:`t`.

Exercise 2
----------

Complete the following code, and test it using `this csv file <https://raw.githubusercontent.com/QuantEcon/lecture-source-py/master/source/_static/lecture_specific/python_advanced_features/test_table.csv>`__, which we assume that you've put in your current working directory


.. code-block:: python3
    :class: no-execute

    def column_iterator(target_file, column_number):
        """A generator function for CSV files.
        When called with a file name target_file (string) and column number
        column_number (integer), the generator function returns a generator
        that steps through the elements of column column_number in file
        target_file.
        """
        # put your code here

    dates = column_iterator('test_table.csv', 1)

    for date in dates:
        print(date)


Exercise 3
----------

Suppose we have a text file ``numbers.txt`` containing the following lines

.. code-block:: none

    prices
    3
    8

    7
    21


Using ``try`` -- ``except``, write a program to read in the contents of the file and sum the numbers, ignoring lines without numbers.



Solutions
=========




Exercise 1
----------

Here's the standard solution

.. code-block:: python3

    def x(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        else:
            return x(t-1) + x(t-2)


Let's test it

.. code-block:: python3

    print([x(i) for i in range(10)])


Exercise 2
----------

One solution is as follows

.. code-block:: python3

    def column_iterator(target_file, column_number):
        """A generator function for CSV files.
        When called with a file name target_file (string) and column number 
        column_number (integer), the generator function returns a generator 
        which steps through the elements of column column_number in file
        target_file.
        """
        f = open(target_file, 'r')
        for line in f:
            yield line.split(',')[column_number - 1]
        f.close()
    
    dates = column_iterator('test_table.csv', 1) 
    
    i = 1
    for date in dates:
        print(date)
        if i == 10:
            break
        i += 1


Exercise 3
----------

Let's save the data first

.. code-block:: python3

    %%file numbers.txt
    prices
    3
    8
    
    7
    21


.. code-block:: python3

    f = open('numbers.txt')
    
    total = 0.0 
    for line in f:
        try:
            total += float(line)
        except ValueError:
            pass
    
    f.close()
    
    print(total)
