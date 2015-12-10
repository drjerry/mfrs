# Matrix Factorization for Recommender Systems

A set of tools for investigating algorithms described in
[Matrix Factorization Techniques for Recommender Systems](http://dl.acm.org/citation.cfm?id=1608614).

This minimal implementation aims to provide the following features.

  1. A sparse matrix representation that is optimized for core operations on
     training data.

  2. A parser for loading training/test data in text format.

  3. A minimal Linear Algebra API that leverages underlying BLAS functions.

  4. A representation for latent factor models, as well as methods for
     serializing and deserializing.

  5. A command line interface for training latent factor modules, and a
     command line interface for emitting predictions on test data.
