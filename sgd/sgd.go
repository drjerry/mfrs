package main

import (
	"github.com/drjerry/mfrs"
	"github.com/drjerry/mfrs/linalg"
)

/* Stochastic-Gradient-Descent.

Arguments:
  - args   config data containing hyper-parameters
  - r      representation of ratings matrix
  - model  serialized data for model

Returns:
  - Mean-Squared-Error
  - updated members Pvals, Qvals of model.
*/
func sgd(args *Args, r mfrs.Ratings, model *mfrs.Model) float32 {
	var u, v linalg.Vector
	p := linalg.MatrixView(int(model.Nrow), int(model.Ldim), model.Pvals)
	q := linalg.MatrixView(int(model.Ncol), int(model.Ldim), model.Qvals)
	du := linalg.NewVector(int(model.Ldim))
	dv := linalg.NewVector(int(model.Ldim))

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
