.. _troubleshooting:

.. include:: /_static/includes/header.raw

.. highlight:: python3

***************
Troubleshooting
***************

.. contents:: :depth: 2

This page is for readers experiencing errors when running the code from the lectures.

Fixing Your Local Environment
==============================

The basic assumption of the lectures is that code in a lecture should execute whenever

#. it is executed in a Jupyter notebook and

#. the notebook is running on a machine with the latest version of Anaconda Python.

You have installed Anaconda, haven't you, following the instructions in :doc:`this lecture <getting_started>`? 

Assuming that you have, the most common source of problems for our readers is that their Anaconda distribution is not up to date.

`Here's a useful article <https://www.anaconda.com/keeping-anaconda-date/>`__
on how to update Anaconda.

Another option is to simply remove Anaconda and reinstall.

You also need to keep the external code libraries, such as `QuantEcon.py
<https://quantecon.org/quantecon-py>`__ up to date.

For this task you can either

* use `pip install --upgrade quantecon` on the command line, or

* execute `!pip install --upgrade quantecon` within a Jupyter notebook.

If your local environment is still not working you can do two things.

First, you can use a remote machine instead, by clicking on the `Launch Notebook` icon available for each lecture

.. image:: _static/lecture_specific/troubleshooting/launch.png

Second, you can report an issue, so we can try to fix your local set up.

We like getting feedback on the lectures so please don't hesitate to get in
touch.

Reporting an Issue
===================

One way to give feedback is to raise an issue through our `issue tracker 
<https://github.com/QuantEcon/lecture-source-py/issues>`__.

Please be as specific as possible.  Tell us where the problem is and as much
detail about your local set up as you can provide.

Another feedback option is to use our `discourse forum <https://discourse.quantecon.org/>`__.

Finally, you can provide direct feedback to contact@quantecon.org

