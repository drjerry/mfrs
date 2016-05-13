# Matrix Factorization for Recommender Systems

A set of tools for investigating algorithms described in
[Matrix Factorization Techniques for Recommender Systems](http://dl.acm.org/citation.cfm?id=1608614).

This minimal implementation supports the following features:

  1. Fitting a model is handled via a command line interface that reads
     training data from a file, performs Stochastic Gradient Descent (for
      a specified number of epochs) and writes the model to a file.

  1. Raw data is read in text format, where each record is a triple `<i,j, r>`
     that represents the entry `r[i,j]` of a "ratings" matrix.  

  3. A fitted model is represented as a
     [protobuf](https://developers.google.com/protocol-buffers/)
     object.

  4. A separate CL interface is provided for evaluating a fitted model on test
     data and emitting the predicted ratings.


## The model

Suppose the user space has dimension M and the item space dimension N, so
that the ratings matrix has dimension M-by-N and is sparse. We use bracket
notation (a la numpy) instead of subscripts. For a L-dimensional latent
factor model, the predicted ratings are given by:

    h[i,j] = a[i] + b[j] + p[i,:] * q[j,:]

where `a` and `b` are column vectors of bias terms of dimensions M and N
respectively, `p` is an M-by-L matrix of latent factors, `q` an N-by-L matrix
of latent factors, and `p[i,:] * q[j,:]` is the inner product between the
i-th row of `p` and j-th row of `q`.


## Stochastic gradient descent

Model fitting minimizes the regularized loss function

    L = sum{ (r[i,j] - h[i,j])^2 + lambda * l2(p[i,:]) + mu * l2(q[j:]) }

where the sum ranges over all pairs (i,j) in the support of the training data,
and where `lambda` and `mu` are regularization parameters for the factors.

For each training example `r[i,j]`, define the error term by

    delta = h[i,j] - r[i,j]

where `h[i,j]` on the right hand side is evaluated from the current state of
parameters. The SGD update for this single example is

    a[i]   -= rho * delta
    b[j]   -= rho * delta
    p[i,:] -= rho * (delta * q[j,:] + lambda * p[i,:])
    q[j,:] -= rho * (delta * p[i,:] + mu * q[j,:])

where `rho` is the learning rate, an algorithmic hyperparameter.
