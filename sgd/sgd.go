package main

import "github.com/drjerry/mfrs/linalg"

/* Stochastic-Gradient-Descent.

Arguments:
  - args  config data containing hyper-parameters
  - r     representation of ratings matrix
  - p, q  matrices of parameters; updated on exit

Returns: Mean-squared-error, ie, mean of (r_{ij} - \hat{r}_{ij})^2 over all
nonzero training examples r_{ij}.
*/
func sgd(args *Args, r linalg.SparseMatrix, p, q linalg.Matrix) float32 {
	var u, v linalg.Vector
	du := linalg.NewVector(args.Ldim)
	dv := linalg.NewVector(args.Ldim)

	iter := r.Iterator()
	var mse, n float32
	for iter.HasNext() {
		i, j, val := iter.Next()
		// e_{ij} <- r_{ij} - p_i \dot q_j
		p.RowView(i, &u)
		q.RowView(j, &v)
		e := val - linalg.Vdot(u, v)
		// update mean online
		n += 1.0
		mse += (e*e - mse) / n
		// dv <- e_{ij} * p_i - \lambda * q_j
		dv.Copy(u)
		linalg.Vscal(e, dv)
		linalg.Vaxpy(-args.Lambda, v, dv)
		// du <- e_{ij} * q_j - \lambda * p_i
		du.Copy(v)
		linalg.Vscal(e, du)
		linalg.Vaxpy(-args.Lambda, u, du)
		// apply gradient
		linalg.Vaxpy(args.Rate, du, u)
		linalg.Vaxpy(args.Rate, dv, v)
	}
	return mse
}
