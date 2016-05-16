# Matrix Factorization for Recommender Systems

A set of tools for investigating algorithms described in
[Matrix Factorization Techniques for Recommender Systems](http://dl.acm.org/citation.cfm?id=1608614).

This minimal implementation supports the following features:

  1. Fitting a model is handled via a command line interface that reads
     training data from a file, performs Stochastic Gradient Descent (for
      a specified number of epochs) and writes the model to a file.

  1. Raw data is read in text format, where each record is a triple `<i,j,r>`
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

    h[i,j] = Pbias[i] + Qbias[j] + Pwts[i,:] * Qwts[j,:]

The names correspond to fields in the [model definition](model.pb.go).
In particular, `Pwts` is an M-by-L matrix, `Qwts` an N-by-L matrix, and
`Pwts[i,:] * Qwts[j,:]` is the Euclidean product of their i-th and j-th rows.
The `Pbias` and `Qbias` terms are vectors of dimension M and N respectively.


## Stochastic gradient descent

The loss function for a single training example `<i,j>` is given by

    E = ((h[i,j] - r[i,j])^2 - lambda * l2(Pwts[i,:])^2 + mu * l2(Qwts[j,:])^2) / 2.

Where `l2()` denotes L2-norm of a vector. Note that regularization is applied
only to the `Pwts` and `Qwts` terms, not the bias terms. From this loss function,
the "delta" to update the terms `Pbias[i]`, `Qbias[j]`, `Pwts[i,:]`, `Qwts[j,:]`
is computed and applied at each step.

The package github.com/drjerry/mfrs/sgd is a command-line interface for
applying SGD to a file of training data. It loads all training data into
memory and performs SGD over the entire set for a specified number of "epochs."
The arguments it takes are:

  - nrow, ncol, ldim: specify the dimensions M, N, and L in advance
  - lambda, mu: the regularization parameters in the loss function
  - learning "rate": rescales the delta (for each term) in the SGD update
  - epochs: number of times to repeat SGD through the entire data set


## Building

The package includes its own minimal set of wrappers around
[CBLAS](http://www.netlib.org/blas/#_cblas) methods, and this library needs
to be present on the target architecture. Installing the package requires
compiler and linker flags to be passed via [CGO](https://golang.org/cmd/cgo/)
environment variables. If your GOPATH is set up and CBLAS is installed in a
standard location, the following should just work:

    $ CGO_LDFLAGS=-lcblas go install github.com/drjerry/mfrs/sgd
    $ CGO_LDFLAGS=-lcblas go install github.com/drjerry/mfrs/eval

If CBLAS is installed in a non-standard location, the "-L" and "-I" flags
may need to be passed as well.
