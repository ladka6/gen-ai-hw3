"""A convolutional VAE for MNIST (Homework 3, Task 1).

The package is organised so that every task of the assignment can reuse the
same building blocks:

* :mod:`vae.data`     -- loading and splitting MNIST.
* :mod:`vae.model`    -- the (de)convolutional Gaussian VAE.
* :mod:`vae.elbo`     -- the ELBO objective and its individual terms.
* :mod:`vae.training` -- the training loop with early stopping and checkpoints.
* :mod:`vae.visualize`-- the figures requested in the report (curves, grids).
"""
