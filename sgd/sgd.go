package main

import (
	"github.com/drjerry/mfrs"
	"github.com/drjerry/mfrs/linalg"
)

/* Stochastic-Gradient-Descent.

Arguments:
  - args  config data containing hyper-parameters
  - r     representation of ratings matrix
  - p, q  matrices of parameters; updated on exit

Returns: Mean-squared-error, ie, mean of (r_{ij} - \hat{r}_{ij})^2 over all
nonzero training examples r_{ij}.
*/
func sgd(args *Args, r mfrs.Ratings, p, q linalg.Matrix) float32 {
	var u, v linalg.Vector
	du := linalg.NewVector(args.Ldim)
	dv := linalg.NewVector(args.Ldim)

	var mse float32
	for n := 0; n < len(r); n++ {
		// e_{ij} <- r_{ij} - p_i \dot q_j
		p.RowView(r[n].Row, &u)
		q.RowView(r[n].Col, &v)
		e := r[n].Val - linalg.Vdot(u, v)
		// update MSE online
		mse += (e*e - mse) / float32(n+1)
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
